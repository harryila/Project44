# Training Strategy — CWT v2

> Document vivant : à mettre à jour au fur et à mesure de l'entraînement.
> Source de vérité pour la configuration, le corpus et le suivi des runs v2.

---

## Objectifs de la v2 vs v1

| Aspect | v1 (figé) | v2 |
|---|---|---|
| Architecture | 12 heads, 2-stage, sum embedding, pos embed appris | 14 heads, **3-stage**, **concat+proj**, **RoPE** |
| Paramètres | ~10.2 M | **~16.2 M** (d=320, layers=10) |
| Corpus pre-train | PDMX seul (213k, bug spine + sans ornements) | PDMX corrigé + ASAP + Chopin + KernScores (~219k) |
| Corpus fine-tune | Filtre `!!!COM:` (Chopin first editions uniquement) | Tout sauf PDMX (ASAP + Chopin + KernScores, ~6k) |
| Ornements | Perdus dans PDMX v1 | Présents (mxl_to_kern v2 + Humdrum encodages) |
| Spine RH/LH | Bug : ~7% inversés | Corrigé via détection BassClef |
| Checkpoints | Tous gardés | **3 derniers + multiples de 15** |

---

## Configuration modèle (`ModelConfig`)

```python
d_model      = 320      # head_dim = 40
n_layers     = 10
n_attn_heads = 8
d_ff         = 1600
max_seq_len  = 2048
dropout      = 0.1
# Paramètres totaux : 16,157,120 (~16.2 M)
```

Décomposition de la croissance v1 → v2 (cf. `archi_model.md` §4).

---

## Corpus d'entraînement

### Phase 1 — Pre-training (tout)

| Source | Nb fichiers | Qualité | Emplacement local | Emplacement serveur |
|---|---|---|---|---|
| **PDMX** (reconverti v2) | 213 369 | Variable (auto-converti) | `pdmx_scrapper/pdmx_data/kern/` | `~/taff/kern_all/pdmx/` (symlink → `~/taff/pdmx_scrapper/pdmx_data/kern/`) |
| **ASAP** (converti 2026-05-19) | 200 | Haute (MXL annoté ; 35 non-piano filtrés) | `kern_scrapping/asap_kern/` | `~/taff/kern_all/asap/` |
| **Chopin first editions** | 512 | Maximale (Humdrum manuel) | `kern_scrapping/humdrum-chopin-first-editions/kern/` | `~/taff/kern_all/chopin_first/` |
| **KernScores composers** | 5 219 | Haute (Humdrum) | `kern_scrapping/kernscores_composers/` | `~/taff/kern_all/kernscores/` |
| **TOTAL** | **219 300** (compté sur serveur) | | | |

### Phase 2 — Fine-tuning (qualité, sans PDMX)

| Source | Nb fichiers |
|---|---|
| ASAP | 235 |
| Chopin first editions | 512 |
| KernScores | 5 219 |
| **TOTAL** | **~5 966** |

Filtre `is_not_pdmx()` : exclut tout fichier dont le nom commence par `Qm` (format hash PDMX).

---

## Hyperparamètres d'entraînement

### Phase 1 — Pre-training (`train_cwt.py`)
- `lr=3e-4`, AdamW, weight_decay=0.01
- `batch=4`, `grad_accum=8` → batch effectif 32
- `epochs=50`
- AMP (mixed precision) activé sur GPU
- `ckpt-every=30` min (intermédiaires `_mid`)

### Phase 2 — Fine-tuning (`finetune_humdrum.py`)
- `lr=1e-4` (3x plus bas qu'en pre-training)
- Mêmes batch/grad_accum
- `epochs=50`
- Démarre depuis le dernier checkpoint pre-training trouvé dans `checkpoints/`
- Nouveaux fichiers nommés `ckpt_ft_ep*.pt` → ne touche pas aux pre-training (`ckpt_ep*.pt`)

---

## Gestion des checkpoints

Politique automatique (fonction `cleanup_checkpoints` dans les 2 scripts) :

- **3 derniers checkpoints** d'epoch (par numéro d'epoch)
- **Multiples de 15** : ep15, ep30, ep45...
- **Intermédiaires `_mid`** : seul le plus récent est gardé
- **Checkpoints pre-training préservés naturellement** quand on passe à fine-tuning (glob `ckpt_ft_ep*.pt` ne matche pas `ckpt_ep*.pt`)

### Checkpoints conservés à la fin
| Phase | Checkpoints gardés |
|---|---|
| Pre-training (50 epochs) | `ckpt_ep015`, `ckpt_ep030`, `ckpt_ep045`, `ckpt_ep048`, `ckpt_ep049`, `ckpt_ep050` (+1 `_mid`) |
| Fine-tuning (50 epochs) | `ckpt_ft_ep015`, `ckpt_ft_ep030`, `ckpt_ft_ep045`, `ckpt_ft_ep048`, `ckpt_ft_ep049`, `ckpt_ft_ep050` (+1 `_mid`) |
| **+ Pre-train final** | `ckpt_ep050` (point de départ du fine-tuning, jamais touché par finetune cleanup) |

Total final : ~13 checkpoints conservés (taille par ckpt avec modèle 16M ≈ 60 MB → ~800 MB total, OK avec le quota).

---

## Setup serveur

Structure cible sur le serveur :
```
~/taff/v2/                          ← code v2 + checkpoints
  cwt_model.py
  train_cwt.py
  finetune_humdrum.py
  postprocess.py
  auto_train.sh                     ← wrapper auto-restart
  README.md                         ← point d'entrée
  docs/                             ← toute la documentation
    archi_model.md
    training_strategy.md            ← copie de ce fichier
    tokenization_rules_reviewed.md
  tokenizers/
    multihead_tokenizer.py
  checkpoints/                      ← géré automatiquement
  train_v2.log                      ← logs phase 1
  finetune_v2.log                   ← logs phase 2

~/taff/kern_all/                    ← corpus unifié (sous-dossiers)
  pdmx/         → symlinks vers ~/taff/pdmx_scrapper/pdmx_data/kern/ (213k)
  asap/         → fichiers convertis (200)
  chopin_first/ → fichiers Humdrum (512)
  kernscores/   → fichiers Humdrum (5219)
```

### Pourquoi des sous-dossiers
- `train_cwt.py` utilise `rglob("*.krn")` → lit tout récursivement
- `finetune_humdrum.py` aussi `rglob` + filtre `is_not_pdmx()` → PDMX automatiquement exclu
- Lisibilité + possibilité d'ajouter/retirer une source en touchant un seul dossier

### Quota d'inodes
- État actuel : ~50-60k inodes utilisés
- Après upload : +6k (ASAP+Chopin+KernScores) + ~13 checkpoints + logs ≈ 70k total
- PDMX reste dans `pdmx_scrapper/pdmx_data/kern/` (213k inodes) → pas re-copié, juste symlink
- Total après setup : ~283k / 300k quota — **tendu mais ça passe**

---

## Setup — historique (déjà effectué)

Étapes 0 à 5 ci-dessous : déjà faites le 2026-05-19/20. Conservées pour référence et reproductibilité.

<details>
<summary>Étapes 0–5 (cliquer pour déplier)</summary>

### 0. Convertir ASAP en local — ✅ fait (200 OK)
```bash
python pdmx_scrapper/mxl_to_kern.py \
  --input kern_scrapping/asap_mxl/ --output kern_scrapping/asap_kern/ \
  --jobs 4 --piano-only
```

### 1. Préparer l'arbo serveur — ✅ fait
```bash
ssh poly-albatros 'mkdir -p ~/taff/v2/checkpoints ~/taff/v2/tokenizers \
                            ~/taff/kern_all/asap ~/taff/kern_all/chopin_first \
                            ~/taff/kern_all/kernscores'
ssh poly-albatros 'ln -sfn ~/taff/pdmx_scrapper/pdmx_data/kern ~/taff/kern_all/pdmx'
```

### 2. Upload code v2 — ✅ fait
```bash
scp .../music_cwt/v2/{cwt_model.py,train_cwt.py,finetune_humdrum.py,postprocess.py,auto_train.sh,README.md} poly-albatros:~/taff/v2/
scp .../music_cwt/v2/docs/*.md poly-albatros:~/taff/v2/docs/
scp .../music_cwt/v2/tokenizers/multihead_tokenizer.py poly-albatros:~/taff/v2/tokenizers/
```

### 3. Upload corpora qualité — ✅ fait
```bash
rsync -avzP .../kern_scrapping/asap_kern/                         poly-albatros:~/taff/kern_all/asap/
rsync -avzP .../kern_scrapping/humdrum-chopin-first-editions/kern/ poly-albatros:~/taff/kern_all/chopin_first/
rsync -avzP .../kern_scrapping/kernscores_composers/              poly-albatros:~/taff/kern_all/kernscores/
```

### 4. Smoke-test serveur — ✅ fait
```bash
ssh poly-albatros 'cd ~/taff/v2 && python3 cwt_model.py'
# Confirme : Paramètres : 16,157,120  (~16.2 M)
```

### 5. Cron @reboot + lancement initial — ✅ fait (sur portugal)
```bash
ssh portugal
(crontab -l 2>/dev/null; echo "@reboot /bin/bash $HOME/taff/v2/auto_train.sh") | crontab -
tmux new -s pretrain_v2
bash ~/taff/v2/auto_train.sh
# Ctrl+B d pour détacher
```

</details>

---

## Phase 1 en cours — surveillance & commandes utiles

```bash
# Suivi live
ssh poly-albatros 'tail -f ~/taff/v2/train_v2.log'

# Meta-log (start/restart/crash du wrapper)
ssh poly-albatros 'tail -20 ~/taff/v2/auto_train.log'

# Vérifier que ça tourne
ssh portugal 'ps aux | grep -E "auto_train|train_cwt" | grep -v grep'

# Liste des checkpoints
ssh poly-albatros 'ls -la ~/taff/v2/checkpoints/'

# Quota disque/inodes
ssh poly-albatros 'quota -s'

# État GPU portugal
ssh portugal 'nvidia-smi'
```

À vérifier régulièrement :
- Loss continue à descendre (val < ~0.10 attendu après ep10)
- Vitesse stable ~10 it/s
- Pas de checkpoint corrompu (warning `[resume] Checkpoint corrompu ignoré`)
- Cleanup automatique fait son travail (max 3 derniers + multiples de 15)

---

## Phase 2 (à lancer APRÈS la fin de la Phase 1)

Quand `Entraînement terminé.` apparaît dans `train_v2.log` ET que le sentinel existe :
```bash
ssh poly-albatros 'cat ~/taff/v2/training_done.flag 2>/dev/null && echo "Phase 1 done"'
```

Lance le fine-tuning sur la machine d'entraînement (portugal ou autre libre) :
```bash
ssh portugal   # ou autre nœud libre
tmux new -s finetune_v2
cd ~/taff/v2
python3 -u finetune_humdrum.py \
  --kern-dir ../kern_all \
  --checkpoint-dir checkpoints \
  --epochs 50 \
  2>&1 | tee finetune_v2.log
# Ctrl+B d
```

Le script :
- Charge automatiquement le dernier `ckpt_ep050.pt` (NFS partagé)
- Applique `lr=1e-4` (vs 3e-4 en pre-training)
- Filtre les fichiers PDMX via `is_not_pdmx()` → ~6k fichiers qualité
- Écrit ses checkpoints sous `ckpt_ft_ep*.pt` → ne touche pas aux pre-training

> Note : un nouveau wrapper auto-restart serait à écrire pour le fine-tuning si on craint des reboots. Pour la phase 2 (50 epochs × ~5 min sur 6k fichiers = ~4h), c'est probablement overkill — un simple tmux suffit.

---

## Journal de bord (à compléter pendant l'entraînement)

### Phase 1 — Pre-training

| Date | Epoch | Train loss | Val loss | Notes |
|---|---|---|---|---|
| 2026-05-19 | ep001 step 600 | 0.66 | — | Premier run sur `angleterre`, 10 it/s, init OK |
| 2026-05-19 | ep002 step 49800 | 0.081 | — | Reboot machine — perte tmux. Loss déjà top |
| 2026-05-20 | ep002 step 11469 (resume) | 0.076 | — | Repris depuis `_mid` ckpt sur portugal (angleterre saturée) |
| 2026-05-20 | ep005 | 0.0648 | 0.0647 | portugal — val ≈ train, pas d'overfitting |
| 2026-05-20 | ep005→ep009 | 0.057 | — | Migration sur `rover` (portugal rebootée). OOM répétés (GPU partagé avec un autre user) → wrapper relance jusqu'à succès |
| 2026-05-21 | ep009 | 0.0569 | — | 10.7 it/s sur rover, stable |

**Setup d'entraînement (au 2026-05-21)** :
- Machine : **rover** (RTX A5000 ~20GB, GPU partagé avec un autre utilisateur)
- Auto-resume : `auto_train.sh` (wrapper bash, flock-protégé) — encaisse les OOM/crashs python
- Cron `@reboot` : ne fire PAS sur les compute nodes Polytechnique → inefficace
- `watchdog.sh` écrit (cron `*/15` sur albatros) mais **pas déployé** — surveillance manuelle pour l'instant
- Logs : `~/taff/v2/auto_train.log` (meta) + `~/taff/v2/train_v2.log` (training)
- Sentinel de fin : `~/taff/v2/training_done.flag`

### Test qualitatif à ep010 (2026-05-21)

Première génération de contrôle avec le checkpoint `ckpt_ep010_step078629.pt` (prompt = 7 premières mesures de la Polonaise sol# mineur de Chopin) :

| Observation | Verdict |
|---|---|
| Ornements (H4) | 1 trille / ~467 notes = 0.21% — cohérent avec le corpus (~0.1%). ✅ fonctionnent |
| Articulations (H5) | 5 accents générés. ✅ fonctionnent |
| Dynamiques (H12) | 7 nuances dans la sortie MXL. ✅ |
| Accords | 70% du contenu musical (majoritairement 2 notes). ✅ |
| Fioritures | 15 générées. ✅ |
| Qualité musicale globale | Brute, **beaucoup de répétitions**, peu de direction long terme — **attendu à ep010 (20% du training)** |
| `SEP` à 36% | Normal : `SEP` est un délimiteur de ligne kern (1 par ligne), pas un bug |

**Conclusion** : le modèle n'est pas perdu — il génère trilles, dynamiques, accords, articulations aux bons taux. La répétitivité et le manque de structure long terme sont normaux à ce stade. À refaire à ep30 et ep050.

Fichiers de test : `music_cwt/v2/generated/v2_ep010_polonaise.mxl` (généré) vs `ORIGINAL_PolonaiseGsharpMin.mxl` (référence).

### Phase 2 — Fine-tuning

| Date | Epoch FT | Train loss | Val loss | Notes |
|---|---|---|---|---|
| (en attente) | | | | |

### Décisions / observations en cours

- **Test OOS Liszt `Jeux d'eaux a la Villa d'Este` — ep050 pre-finetune (2026-06-03)** :
  prompt MusicXML nettoye en 2 portees (`liszt-les-jeux-deaux-a-la-ville-deste_no_top_staff.musicxml`,
  premiere portee supprimee, pas de squeeze/merge multi-staff dans `generate.py`).
  Observation qualitative : `ckpt_ep050_step356402.pt` repete en boucle le passage du debut,
  puis change completement de sujet. Conclusion : mauvais candidat pour le choix final OOS Liszt ;
  a garder seulement comme baseline pre-finetune.

---

## Critères pour passer à v3

Voir aussi `archi_model.md` § 7 (dans ce même dossier) et `../../projet_a_pt_etre_reprendre/accords_atomiques.md`.

- **Loss pondérée inverse-fréquence** : à ajouter en v3 si après fine-tuning, les ornements (H4), phrases (H8), ou dynamiques (H12) n'apparaissent toujours pas dans les générations.
- **Accords atomiques** : à ajouter en v3 si v2 montre une faiblesse harmonique (sons mal résolus, accords incohérents) ou une saturation du contexte (le modèle perd le fil après 1000-1500 tokens malgré max_seq_len=2048).
- **Fine-tuning par style** (ST_chopin, ST_liszt) : à ajouter en v3 si v2 mélange les styles de manière incongrue.
- **GPT+BPE baseline** : à ajouter en v3 pour quantifier l'apport du multi-head vs un transformer classique.
