# music_cwt v2 — Compound Word Transformer

Un transformer autoregressif **16.2 M paramètres** qui génère de la musique pour piano au format **kern** depuis une représentation à 14 heads compound. Entraîné sur ~219 300 fichiers kern (PDMX + ASAP + Chopin first editions + KernScores), puis fine-tuné sur ~6 000 fichiers de qualité (tout sauf PDMX).

> **Statut au 2026-05-20 :** code v2 complet, **Phase 1 (pre-training) en cours** sur `portugal` (machine GPU Polytechnique).
> Voir [docs/training_strategy.md](docs/training_strategy.md) pour le journal de bord et l'état actuel.

---

## Améliorations v1 → v2

| Aspect | v1 (figé) | v2 (en cours) |
|---|---|---|
| Paramètres | 10.2 M | **16.2 M** (d=320, layers=10) |
| Positional encoding | Embedding appris | **RoPE** (Rotary, encodé dans l'attention) |
| Compound embedding | Somme des 12 head embeds | **Concat + projection linéaire** |
| Output stages | 2 (Stage 1 = H1+H11 ; Stage 2 = reste) | **3** (Stage 3 = H4/H5 conditionnés sur H2+H3) |
| Heads | 12 (H1-H12) | **14** (+H13 Pédale, +H14 Tempo) |
| Corpus pre-train | PDMX seul, 213k (avec bugs) | PDMX corrigé + ASAP + Chopin + KernScores, **219 300** |
| Corpus fine-tune | Filtre strict `!!!COM:` (~512 fichiers Chopin) | Tout sauf PDMX (~5 931 fichiers) |
| Ornements PDMX | Perdus à la conversion | **Extraits** (T, t, M, m, W, S, :, q, Q) par `mxl_to_kern.py` v2 |
| Spine RH/LH PDMX | Bug ~7 % inversés | **Corrigé** via détection BassClef |
| Checkpoints | Tous gardés | **3 derniers + multiples de 15** (cleanup auto) |
| Auto-recover | tmux manuel | **Wrapper bash `auto_train.sh` + cron `@reboot`** |

---

## Architecture (résumé — détail dans [docs/archi_model.md](docs/archi_model.md))

```
Input :  14 tokens H1-H14 par position (compound)
   ↓
Compound Embedding :  concat(14 × 320) → Linear(4480→320)
   ↓
Transformer × 10 layers :
   LayerNorm → CausalSelfAttention (+ RoPE sur Q/K) → résiduel
   LayerNorm → FFN(320 → 1600 → 320) → résiduel
   ↓
LayerNorm
   ↓
ThreeStageOutputLayer :
   Stage 1 :  Linear(320 → vocab_h1) + Linear(320 → vocab_h11)
   Stage 2 :  cond = h + embed(h1_pred) + embed(h11_pred) ; Linear(cond → vocab_X) for X ∈ {h2,h3,h6-h10,h12-h14}
   Stage 3 :  cond' = cond + embed(h2_pred) + embed(h3_pred) ; Linear(cond' → vocab_X) for X ∈ {h4,h5}
```

**Pourquoi 3 stages ?** Les ornements (H4) et articulations (H5) dépendent fortement de la durée (H2) et de la hauteur (H3) : un trille a plus de sens sur une noire pointée que sur une triple-croche ; un staccato sur une croche aigüe que sur une ronde grave. Conditionner H4/H5 sur (H2, H3) capture cette dépendance fine.

---

## Configuration

```python
d_model      = 320      # head_dim = 40
n_layers     = 10
n_attn_heads = 8
d_ff         = 1600
max_seq_len  = 2048
dropout      = 0.1
# Paramètres totaux : 16 157 120 (~16.2 M)
```

**Hyperparams d'entraînement** : `lr=3e-4` (pre-train) / `1e-4` (fine-tune), `batch=4`, `grad_accum=8` (effectif 32), AdamW + AMP. ~10 it/s sur RTX A4000 → **~85 min/epoch**.

---

## Structure des fichiers

```
music_cwt/v2/
  tokenizers/
    multihead_tokenizer.py             # tokeniseur 14-heads : kern → CompoundToken
  docs/                                  # toute la documentation v2
    archi_model.md                       # spec architecturale détaillée
    training_strategy.md                 # plan + journal de bord live
    tokenization_rules_reviewed.md       # spec autoritaire des 14 heads
  cwt_model.py                          # ModelConfig + CompoundWordTransformer (RoPE, Stage 3, concat+proj)
  train_cwt.py                          # boucle pre-training + cleanup_checkpoints + rglob/symlinks
  finetune_humdrum.py                   # fine-tuning avec filtre is_not_pdmx
  auto_train.sh                         # wrapper bash auto-restart (flock-protégé, cron @reboot)
  postprocess.py                        # CompoundToken → texte **kern valide
  generate.py / gen_and_convert.py      # inférence + conversion MXL (Windows)
  kern_to_mxl.py / summarize_logs.py / compute_metrics.py
  README.md                             # ce fichier (point d'entrée)
```

Côté serveur (NFS Polytechnique) :
```
~/taff/v2/                              ← mêmes fichiers
  checkpoints/                          ← ckpt_ep*.pt (auto-cleanup)
  train_v2.log                          ← logs Phase 1
  auto_train.log                        ← meta-logs du wrapper
  training_done.flag                    ← sentinel créé en fin Phase 1
~/taff/kern_all/                        ← corpus unifié pour Phase 1
  pdmx/   asap/   chopin_first/   kernscores/
```

---

## Lancement d'entraînement

### Phase 1 — Pre-training (en cours)

Setup (déjà fait) :
```bash
ssh portugal
(crontab -l 2>/dev/null; echo "@reboot /bin/bash $HOME/taff/v2/auto_train.sh") | crontab -
tmux new -s pretrain_v2
bash ~/taff/v2/auto_train.sh
# Ctrl+B d pour détacher
```

### Phase 2 — Fine-tuning (à lancer après Phase 1)

Vérifier que le sentinel existe :
```bash
ssh poly-albatros 'ls ~/taff/v2/training_done.flag 2>/dev/null && echo "Phase 1 done"'
```

Puis :
```bash
ssh portugal
tmux new -s finetune_v2
cd ~/taff/v2
python3 -u finetune_humdrum.py \
  --kern-dir ../kern_all \
  --checkpoint-dir checkpoints \
  --epochs 50 \
  2>&1 | tee finetune_v2.log
# Ctrl+B d
```

Le script charge automatiquement le dernier `ckpt_ep050.pt` et démarre la phase 2 avec `lr=1e-4`.

### Surveillance

```bash
ssh poly-albatros 'tail -20 ~/taff/v2/train_v2.log'       # training en live
ssh poly-albatros 'tail -10 ~/taff/v2/auto_train.log'     # meta (start/restart/crash)
ssh poly-albatros 'ls -la ~/taff/v2/checkpoints/'         # checkpoints actuels
ssh portugal       'nvidia-smi'                           # état GPU
ssh portugal       'ps aux | grep auto_train | grep -v grep'  # process tournant
```

---

## Prompts MusicXML — squeeze multi-portées

Le modèle est entraîné sur du **kern à 2 spines** (RH / LH). Quand un prompt MXL a **3+ portées** (typique chez Liszt : mélodie + arpèges + basse ; cf. *Jeux d'eaux*, *Sonate*, *Vallée d'Obermann*…), `_mxl_to_kern()` dans [generate.py](generate.py) **fusionne automatiquement** en 2 portées :

1. **Classification clef d'abord** : portées en clef de basse → LH, le reste → RH
2. **Fallback heuristique par pitch** (utile pour Liszt qui écrit souvent les 3 portées en clé de sol) : la portée avec la moyenne MIDI la plus grave → LH, toutes les autres mergées dans RH
3. **Merge intra-portée** : notes simultanées au même offset depuis plusieurs portées source → empilées en accord kern (` `4c 4e` `)
4. **Rest vs Note au même offset** : la note l'emporte (le silence d'une voix est implicite quand une autre voix joue)

### Limitations connues
- **Changements de clef mid-piece** dans une portée source : ambigus après merge (le `*clef` peut s'appliquer au mauvais endroit dans la portée fusionnée)
- **Durées différentes au même offset** (ex: une noire dans la voix mélodique + une 32e dans l'arpège) : le merge crée un accord avec la durée de la 1ère voix lue → kern syntaxiquement valide mais imprécis musicalement
- **Doublures cross-staff** : si une même note apparaît dans 2 portées au même offset, elle est dupliquée dans l'accord (négligeable en pratique)

Ces limites n'affectent que les pièces très complexes (3+ portées avec voix indépendantes). Pour 95% du répertoire piano, le squeeze produit un prompt fidèle.

---

## Génération (Windows local)

> **Note :** les exemples ci-dessous utiliseront les checkpoints v2 dès qu'ils seront téléchargés. Pour l'instant (Phase 1 en cours), tu peux toujours utiliser v1 dans `music_cwt/v1/`.

**Python à utiliser** : `<your Python 3.12 with torch + music21>` (a torch + music21).

Toujours utiliser `gen_and_convert.py` sur Windows — il fait génération + conversion MXL dans un seul process, évite les problèmes de sandbox PowerShell.

```bash
# Avec prompt kern
python gen_and_convert.py \
  --checkpoint checkpoints/ckpt_ft_ep050_step<N>.pt \
  --output ../generated/out.krn \
  --prompt-krn ../kern_data/Chopin_nocturne72-1.krn \
  --prompt-measures 6 \
  --temperature 0.8 --top-k 30

# Avec prompt MusicXML (OOS test)
python gen_and_convert.py \
  --checkpoint checkpoints/ckpt_ft_ep050_step<N>.pt \
  --output ../generated/oos_test.krn \
  --prompt-krn ../prompt/009_1et2-1a-W-002.musicxml \
  --prompt-measures 6 \
  --temperature 0.8 --top-k 30

# Cold (sans prompt)
python gen_and_convert.py \
  --checkpoint checkpoints/ckpt_ft_ep050_step<N>.pt \
  --output ../generated/cold.krn \
  --key K_0_maj --time M_3/4 --tempo MM_72
```

Paramètres clés :
- `--temperature` : 0.8 conservateur, 1.0 défaut (plus créatif), plus bas = moins répétitif
- `--top-k` : 30 plus focus que 50 défaut
- `--prompt-measures` : nombre de mesures du prompt (défaut 6)
- `--max-tokens` : tokens à générer après prompt (défaut 1200 ≈ 30-40 mesures)

Le `.mxl` est produit à côté du `.krn` et s'ouvre directement dans MuseScore.

---

## Dépendances

```
torch >= 2.0
music21 >= 9.0
```

Python : `<your Python 3.12 with torch + music21>`

---

## Commandes SSH

```bash
# Depuis WSL — toujours utiliser /mnt/c/... pour les paths Windows
ssh poly-albatros        # gateway (utilise le NFS partagé)
ssh portugal             # nœud GPU actuel (RTX A4000 libre)

# Upload code v2 vers serveur
scp <local copy of taff>/music_cwt/v2/<file> poly-albatros:~/taff/v2/<file>

# Télécharger un checkpoint
scp poly-albatros:~/taff/v2/checkpoints/<ckpt>.pt \
    <local copy of taff>/music_cwt/checkpoints/
```

---

## Projets reportés à v3+

Si après la v2 certains problèmes persistent (voir critères dans [docs/archi_model.md](docs/archi_model.md) §7) :

- **Loss pondérée par fréquence inverse** — si heads rares (H4, H8, H12) mal apprises
- **Accords atomiques** — si faiblesse harmonique ou saturation contexte (cf. [projet_a_pt_etre_reprendre/accords_atomiques.md](../projet_a_pt_etre_reprendre/accords_atomiques.md))
- **Fine-tuning par style** (ST_chopin, ST_liszt)
- **GPT+BPE baseline** pour mesurer l'apport du multi-head
- **Flash Attention** pour passer à seq_len > 2048
- **Renommage PDMX** : script prêt (`pdmx_scrapper/rename_pdmx_kern.py`), à appliquer après Phase 1

---

## Documents associés

- **Spec architecture** : [docs/archi_model.md](docs/archi_model.md)
- **Spec tokenisation** (14 heads, règles structurelles) : [docs/tokenization_rules_reviewed.md](docs/tokenization_rules_reviewed.md)
- **Plan + journal de bord training** : [docs/training_strategy.md](docs/training_strategy.md)
- **Datasets** : [../../datasets.md](../../datasets.md)
- **TODO global du projet** : [../../todo.md](../../todo.md)
- **Référence inter-sessions Claude** : [../../CLAUDE.md](../../CLAUDE.md)
