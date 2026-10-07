# Projet en standby — Entraînement distribué multi-nœuds (DDP)

> **Statut : EN STANDBY.** À construire pour v3+ si on veut accélérer fortement les
> entraînements (passer de ~70h à ~5-10h par run). NE PAS appliquer à v2 en cours.

---

## Idée

Le cluster Polytechnique a ~25 nœuds (machines voitures : `bentley`, `bugatti`, …, `volvo`),
chacun avec un GPU (RTX A4000 16GB ou A5000 ~20GB). Actuellement on n'en utilise **qu'un seul**
à la fois pour entraîner. L'idée : utiliser **PyTorch DistributedDataParallel (DDP)** pour
splitter un entraînement sur N nœuds → accélération ~Nx.

Gain théorique : 70h (single-node) / 25 nœuds ≈ **2.8h**. Très tentant pour itérer vite
(v3, baseline GPT+BPE, ablations).

---

## Pourquoi ce n'est PAS un quick win — les pièges

### 1. Le speedup réel dépend du réseau

DDP fait un **all-reduce des gradients à chaque step** : pour notre modèle 16M params,
ça fait 16M × 4 bytes = **64 MB de gradients échangés entre tous les nœuds, chaque step**.

- Si le cluster a un interconnect rapide (InfiniBand / NVLink) → proche du speedup linéaire
- Si c'est de l'**Ethernet classique** (probable sur un cluster d'étudiants) → l'all-reduce
  devient le goulot d'étranglement. Speedup réel possiblement **5-10x seulement**, voire pire.

**Le réseau du cluster est inconnu → impossible de promettre "2h" sans mesurer.**

### 2. "Certains nœuds dispo, d'autres non" — le vrai problème

| Moment | Problème ? |
|---|---|
| **Au lancement** | Non. On scanne, on trouve N nœuds libres, on lance DDP avec `world_size=N`. On adapte N à ce qui est disponible. |
| **Pendant le run** | **OUI, critique.** Le DDP standard a un `world_size` **fixe**. Si UN nœud reboote, est pris par un autre utilisateur, ou crashe → l'all-reduce attend ce rank mort → **tout le job se bloque/crashe** (pas juste ce nœud). Un job DDP est aussi stable que son nœud le moins stable. |

**Observation utile** : les nœuds semblent rebooter ~06h30 chaque jour. Si un run dure ~3h
et qu'on le lance à 10h, on évite le reboot **programmé**. ✅
Mais ça ne couvre PAS : un nœud pris par un autre user en cours de route, ni un reboot/crash
non programmé.

**Solution propre** : `torchrun` en **mode élastique** (`--nnodes=12:25`, `--max-restarts=N`).
Le DDP élastique tolère les nœuds qui partent/arrivent en re-formant le groupe de process.
Mais : setup nettement plus complexe (rendez-vous backend c10d/etcd), et chaque perte de
nœud déclenche un reload de checkpoint + redémarrage du collective (disruptif).

### 3. Le gros batch change la dynamique d'entraînement

25 nœuds × batch 4 × grad_accum 8 = **batch effectif 800** (vs 32 en single-node).

Un batch ~25x plus gros nécessite de **re-régler le learning rate** :
- Règle de scaling linéaire : `lr × ~25` + warmup
- Sinon la convergence se dégrade (gros batch = gradient moins bruité = besoin d'un LR plus haut,
  mais trop haut = divergence)

Ce n'est PAS un swap transparent — c'est un ré-entraînement à retuner.

---

## Plan d'implémentation (pour v3+)

À traiter comme un projet à part entière, pas un patch rapide.

1. **Écrire `train_cwt_ddp.py`** basé sur `torchrun` en mode élastique :
   - `torch.distributed` init (backend `nccl` si GPU-to-GPU, `gloo` sinon)
   - `DistributedSampler` sur le `KernDataset`
   - Wrapper `DistributedDataParallel` autour du modèle
   - Checkpoint écrit par le rank 0 uniquement
   - `--nnodes=MIN:MAX` pour l'élasticité

2. **Test 4 nœuds d'abord — mesurer le speedup réel.**
   - Lancer sur 4 nœuds libres, comparer it/s vs single-node
   - Si on obtient ~3.5-4x → le réseau suit, on peut monter à 25
   - Si on plafonne à ~2x → le réseau est le goulot, inutile d'aller plus loin
   - **Ce test décide si le projet vaut le coup.**

3. **Retuner le LR** pour le batch effectif visé (scaling linéaire + warmup).

4. **Script de scan + lancement coordonné** :
   - Scanner les ~25 nœuds (`nvidia-smi` via SSH), lister les libres
   - Générer le `torchrun` multi-nœuds avec la liste des hôtes libres
   - Définir `MASTER_ADDR` / `MASTER_PORT`

5. **Stratégie de robustesse** :
   - Lancer juste après la fenêtre de reboot 06h30 (nœuds les plus "frais")
   - Runs courts (< durée jusqu'au prochain reboot)
   - Checkpoint fréquent (rank 0) sur le NFS partagé
   - Mode élastique pour survivre à la perte d'un nœud

---

## Critère de décision

**Construire le DDP si** : on prévoit beaucoup d'itérations (v3, baseline GPT+BPE, ablations
RoPE/Stage3, sweeps d'hyperparamètres) — l'accélération paye vite.

**Abandonner si** : le test 4-nœuds montre un speedup < 2.5x (réseau trop lent), OU on ne prévoit
que 1-2 entraînements de plus (le coût de mise en place dépasse le gain).

**Ne JAMAIS basculer un entraînement en cours en DDP** — ça implique de jeter le run, réécrire
le code, débugger le multi-nœud, retuner le LR. À réserver à un nouveau run from scratch.

---

## Alternative simple — jobs indépendants en parallèle

Si le DDP s'avère trop fragile/lent, le repli évident : **un job indépendant par nœud**.
Pas de coordination, pas d'all-reduce, un reboot ne tue qu'un seul job. Idéal pour faire
tourner en parallèle : v3 pre-training (nœud A), GPT+BPE baseline (nœud B), ablation sans
RoPE (nœud C), etc. Moins rapide pour UN modèle donné, mais robuste et exploite le cluster
sans aucune complexité réseau.
