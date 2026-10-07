# music_cwt v3 — Compound Word Transformer

Un transformer autoregressif **≈ 16.6 M paramètres** (16 598 080) qui génère de la musique pour piano au format **kern** depuis une représentation à **18 heads** compound. Hérite des choix v2 (RoPE, Stage 3, concat+projection, ~219 300 fichiers d'entraînement) et ajoute, par rapport à v2 :

- **Accords atomiques** : compression d'un accord harmonique reconnu en 1 compound token (`Chord_Atomic` + **H15** qualité + **H16** voicing), avec fallback séquentiel garanti et garde lossless.
- **Multi-voix intra-portée** : head **H17** (`V1`/`V2`/`V3`), axe orthogonal à la portée (H11), pour la polyphonie sur une même portée (Chopin : ~28 % des fichiers).
- **Token `Sustain`** : remplace le délimiteur `SEP` de v2 ; ancre toujours émise sur Staff_1/V1 (événement si actif, sinon `.` tenu), qui délimite la ligne — **−23 % de tokens** en texture dense.
- **H18 Caractère / Expression** : `Dolce`, `Cantabile`, `Espr`, `Leggiero`, `Marcato_P`, `Sostenuto`, `Tranquillo`, `Agitato`, `Sotto_Voce`, `Pesante` (event-based, comme H12).
- **Calando/Poco_Calando dans H14** (geste unifié ralentir+diminuer) et **Una_Corda/Tre_Corde dans H13** (pédale gauche).

> **Statut au 2026-06 :** v3 **codée + round-trip validé (idempotent, 200/200 sur kern_data + asap_kern), PAS ENCORE entraînée** (pas de `checkpoints/`).
> - Accords atomiques **figés** — reconstruction 100 % kern (sans MIDI), **garde lossless** (0 partition altérée, vérifié kern→token→kern sur 24.6 k accords). Bilan : compression réelle ~2 % → valeur à trancher par A/B (cf. `todo.md`). **On ne touche plus à l'atomisation.**
> - H15–H18, Sustain, multi-voix, Calando, Una/Tre corde : **implémentés** (tokeniseur + modèle + postprocess).
> - **Restant avant entraînement** : dé-squeeze des 2 convertisseurs mxl→kern pour le multi-staff PHYSIQUE 3+ portées (le tokeniseur/modèle/postprocess gèrent déjà N portées ; seuls les convertisseurs squeezent encore), tri qualité du corpus, A/B accords atomiques. Cf. `todo.md`.

---

## Améliorations v2 → v3

| Aspect | v2 | v3 |
|---|---|---|
| Heads | 14 (H1–H14) | **18** (+H15 Chord Quality, +H16 Chord Voicing, +H17 Voix, +H18 Caractère) |
| H1 vocab | `…`, `SEP` | +`Chord_Atomic`, +`Sustain` (le `Sustain` remplace `SEP` comme délimiteur de ligne) |
| Accord harmonique reconnu | N tokens (`Chord_Start` … `Chord_End`) | **1 token** (`Chord_Atomic`) avec fallback séquentiel garanti |
| Polyphonie sur une portée | squeezée en accords | **head H17 (`V1`/`V2`/`V3`)**, axe orthogonal à la portée |
| Caractère / expression | non capté | **head H18** (`Dolce`, `Cantabile`, `Espr`, `Leggiero`…) |
| Longueur de séquence | 1 `SEP` par ligne (pur bruit) | ancre `Sustain` sur Staff_1/V1 → **−23 % de tokens** en texture dense |
| Paramètres | 16.2 M | **≈ 16.6 M** (16 598 080) |
| Garantie zéro perte d'info | n/a | Fallback automatique sur path séquentiel si l'accord ne matche pas le schéma + garde lossless |
| Inductive bias harmonique | n/a | Le modèle apprend « Do majeur » comme une unité, pas 3 hauteurs indépendantes |

---

## Améliorations v1 → v2 (rappel, intégrées en v3)

| Aspect | v1 (figé) | v2 / v3 |
|---|---|---|
| Positional encoding | Embedding appris | **RoPE** (Rotary, encodé dans l'attention) |
| Compound embedding | Somme des head embeds | **Concat + projection linéaire** |
| Output stages | 2 | **3** (Stage 3 = H4/H5 conditionnés sur H2+H3) |
| Heads | 12 | 14 puis **18** en v3 |
| Corpus pre-train | PDMX seul, 213k (avec bugs) | PDMX corrigé + ASAP + Chopin + KernScores, **219 300** |

---

## Architecture (résumé — détail dans [docs/archi_model.md](docs/archi_model.md))

```
Input :  18 tokens H1-H18 par position (compound)
   ↓
Compound Embedding :  concat(18 × 320) → Linear(5760 → 320)
   ↓
Transformer × 10 layers :
   LayerNorm → CausalSelfAttention (+ RoPE sur Q/K) → résiduel
   LayerNorm → FFN(320 → 1600 → 320) → résiduel
   ↓
LayerNorm
   ↓
ThreeStageOutputLayer :
   Stage 1 :  Linear(320 → vocab_X) for X ∈ {h1, h11, h17}   ← identité structurelle
   Stage 2 :  cond = h + embed(h1_pred) + embed(h11_pred) + embed(h17_pred)
              Linear(cond → vocab_X) for X ∈ {h2, h3, h6-h10, h12-h16, h18}
   Stage 3 :  cond' = cond + embed(h2_pred) + embed(h3_pred)
              Linear(cond' → vocab_X) for X ∈ {h4, h5}
```

**Pourquoi h17 (voix) en Stage 1 ?** C'est l'identité structurelle du token, au même titre que h11 (portée) : nécessaire pour que l'ancre Staff_1/V1 soit prédite même sur un token `Sustain`.

**Contraintes structurelles (loss masquée — telle qu'implémentée dans `cwt_model.py`) :**

Seuls **H1, H11, H17** (Stage 1) sont toujours entraînés. Pour les autres heads, la loss est masquée selon H1 :

| Condition sur H1 | Heads exclus de la loss (non entraînés à cette position) |
|---|---|
| Barline / SEP / **Sustain** (v3) | tous les non-Stage-1 (H2–H10, H12–H16, H18) — sur `Sustain`, H11=`Staff_1`, H17=`V1` restent prédits |
| Rest | H4, H5, H8 |
| Fioritura_Q/q | H4, H8 |
| **Chord_Atomic** (v3) | **H6, H7, H8** (pas de décoration per-note ; H4/H5 communs autorisés depuis la relaxation 2026-06) |
| ≠ **Chord_Atomic** (v3) | **H15, H16** (n'existent que sur un accord atomique) |

> ⚠️ Note d'implémentation : le tokeniseur *stampe* H12/H13/H14/H18 (dynamique/pédale/tempo/caractère du moment) sur tous les tokens, y compris l'ancre `Sustain` ; mais la loss ci-dessus ne les entraîne pas sur `Sustain`/`Barline`/`SEP`. Écart spec (`docs/tokenization_rules_reviewed.md`) ↔ code à trancher avant l'entraînement.

**Pourquoi Chord_Atomic ?** Un accord de N notes occupe N positions du contexte en v2. Le path atomique compresse les accords reconnus (qualité + voicing, aucune décoration per-note) en 1 token unique. Gains visés : (a) compression de séquence ; (b) inductive bias harmonique ; (c) zéro perte via fallback séquentiel + **garde lossless** (un accord n'est atomisé que si sa reconstruction reproduit exactement les notes écrites).
**⚠️ Bilan empirique (2026-06)** : compression réelle **~2 %** seulement (21.4 % des accords 3+ notes = 1.25 % des évènements ; plafond dû aux décorations 32 % / clusters non-tertiens 22 % / hampes mixtes 20 %). Donc la valeur effective ne peut venir que du **biais inductif** → à mesurer par l'**A/B atomisée vs non-atomisée**. Reconstruction 100 % kern (sans MIDI). Étude initiale : [../projet_a_pt_etre_reprendre/accords_atomiques.md](../projet_a_pt_etre_reprendre/accords_atomiques.md).

---

## Configuration

```python
d_model      = 320      # head_dim = 40
n_layers     = 10
n_attn_heads = 8
d_ff         = 1600
max_seq_len  = 2048
dropout      = 0.1
# Paramètres totaux : 16 598 080 (~16.6 M)
```

**Hyperparams d'entraînement** : `lr=3e-4` (pre-train) / `1e-4` (fine-tune), `batch=4`, `grad_accum=8` (effectif 32), AdamW + AMP. ~10 it/s sur RTX A4000 → **~85 min/epoch**.

---

## Structure des fichiers

```
music_cwt/v3/
  tokenizers/
    multihead_tokenizer.py             # tokeniseur 18-heads : kern → CompoundToken
                                       # + détection qualité/voicing + _try_atomize_chord
                                       # + émission voix (H17) + ancre Sustain + carac (H18)
  docs/                                # toute la documentation v3
    archi_model.md                     # spec architecturale détaillée
    tokenization_rules_reviewed.md     # spec autoritaire des heads (incl. Chord_Atomic, Sustain, H17, H18)
    atomisation_accords.md             # mécanisme accords atomiques (détection + reconstruction + garde lossless)
    plan_multivoix_sustain.md          # plan multi-voix intra-portée + token Sustain
    training_strategy.md               # plan + journal de bord
  cwt_model.py                         # ModelConfig + CompoundWordTransformer
                                       # (RoPE, Stage 3, concat+proj, masques H15/H16)
  train_cwt.py                         # boucle pre-training + cleanup_checkpoints
  finetune_humdrum.py                  # fine-tuning avec filtre is_not_pdmx
  auto_train.sh                        # wrapper bash auto-restart (cron @reboot)
  postprocess.py                       # CompoundToken → texte **kern valide
                                       # + expansion Chord_Atomic → 4c 4e- 4g
  generate.py / gen_and_convert.py     # inférence + conversion MXL (Windows)
  kern_to_mxl.py / summarize_logs.py / compute_metrics.py
  roundtrip_test/                      # sorties du round-trip postprocess.py
  README.md                            # ce fichier (point d'entrée)
```

---

## Filtrage corpus (préalable à l'entraînement v3)

Constat sur v2 : génération produit régulièrement (a) mesures sur-condensées et (b) longues pauses. Le scan complet du corpus (`music_cwt/data_analysis/kern_distribution.py` sur les 219 300 fichiers de `~/taff/kern_all`) confirme que la queue de distribution contient des fichiers catastrophiques.

**Quantiles observés (corpus complet) et seuils retenus :**

| Métrique                | p50  | p75  | p90  | p95  | p98  | p99   | max   | **seuil** |
|-------------------------|-----:|-----:|-----:|-----:|-----:|------:|------:|----------:|
| `density_skew`          | 1.43 | 1.65 | 1.95 | 2.23 | —    | **3.93** | 267   | p99 = 4 |
| `max_notes_per_measure` | 7    | 9    | 11   | 13   | —    | **24**   | 533   | p99 = 24 |
| `max_rest_streak`       | 0    | 1    | 2    | 3    | **7** | 15      | 1 841 | p98 = 7  ← meilleur elbow |
| `pct_rests`             | 0    | 0.016 | 0.057 | 0.099 | **0.17** | 0.25 | 1.0   | p98 = 0.17 ← cible pauses inopinées |

→ `max/p99` × 20-120 = la queue cache des outliers extrêmes (un fichier avec 1841 lignes consécutives sans note !).

**Choix :** p99 pour skew/npm (elbow net), p98 pour rest_streak/pct_rests (elbow plus marqué + cible directement les pauses inopinées du modèle).

```bash
# 1. Distribution (1 ligne par fichier)
python3 ~/taff/data_analysis/kern_distribution.py --input ~/taff/kern_all --jobs 8

# 2. Filtrage avec seuils retenus
python3 ~/taff/data_analysis/diagnose_density.py \
    --input ~/taff/kern_all --jobs 8 \
    --max-npm 24 --max-skew 4 \
    --max-rest 7 --max-pct-rests 0.17 \
    --min-measures 4
```

Sortie : `bad_files.csv` (~3-5 % du corpus avec ces seuils). Liste à charger dans le DataLoader v3 comme blacklist nom-de-fichier.

Scripts dans [../data_analysis/](../data_analysis/) :
- `kern_distribution.py` : 1 ligne par fichier, toutes les métriques + quantiles
- `diagnose_density.py` : mode filtre (CSV des seuls bad files avec raison)
- `plot_distributions.py` : graphes (PNG dans `results/plots/`)
- `results/REPORT.md` : rapport visuel complet

---

## Lancement d'entraînement v3

Identique à v2 (mêmes scripts), mais le checkpoint v2 final ne peut pas être chargé directement : les heads H15–H18 et les nouveaux types H1 `Chord_Atomic`/`Sustain` n'existent pas dans v2. Le pre-training v3 doit donc repartir de zéro OU charger v2 avec strict=False (les nouvelles head embeddings et linears sont initialisés normalement).

```bash
ssh portugal
tmux new -s pretrain_v3
cd ~/taff/v3
bash auto_train.sh
# Ctrl+B d pour détacher
```

### Surveillance

```bash
ssh poly-albatros 'tail -20 ~/taff/v3/train_v3.log'
ssh poly-albatros 'ls -la ~/taff/v3/checkpoints/'
```

---

## Round-trip de validation (Chord_Atomic)

```bash
python postprocess.py <fichier.krn> roundtrip_test/<sortie>.reconstructed.krn
```

Pour analyser la couverture atomique sur un corpus :

```bash
python tokenizers/multihead_tokenizer.py <fichier.krn>
# Affiche le détail des 30 premiers tokens (avec H15/H16) et les vocab sizes
```

Statistiques empiriques (corpus local Chopin + KernScores, 2026-06) :
- **21.4 % des accords 3+ notes atomisés** = **1.25 % de tous les évènements** → compression de séquence **~2 %** (le "−11 à −13 %" estimé initialement était trop optimiste).
- Plafond irréductible : décorés (ties/ornements/slurs) 32 %, clusters non-tertiens 22 %, hampes/beams mixtes 20 %.
- Reconstruction **100 % kern** (sans MIDI) + **garde lossless** : vérifié kern→token→kern sur 24.6 k accords, **0 partition altérée**.
- Valeur réelle (compression marginale) → à trancher par l'A/B atomisée vs non-atomisée (cf. `todo.md`).

---

## Génération (Windows local)

> **Note :** un fichier d'entraînement v3 sera nécessaire avant utilisation. Pour les tests immédiats, utiliser les checkpoints v2 avec une couche de compatibilité (à écrire) ou v1.

**Python à utiliser** : `<your Python 3.12 with torch + music21>`.

Toujours utiliser `gen_and_convert.py` sur Windows — il fait génération + conversion MXL dans un seul process, évite les problèmes de sandbox PowerShell.

```bash
python gen_and_convert.py \
  --checkpoint checkpoints/<ckpt>.pt \
  --output ../generated/out.krn \
  --prompt-krn ../kern_data/<prompt>.krn \
  --prompt-measures 6 \
  --temperature 0.8 --top-k 30
```

---

## Dépendances

```
torch >= 2.0
music21 >= 9.0
```

---

## Documents associés

- **Spec architecture** : [docs/archi_model.md](docs/archi_model.md)
- **Spec tokenisation** (18 heads, règles structurelles) : [docs/tokenization_rules_reviewed.md](docs/tokenization_rules_reviewed.md)
- **Plan + journal de bord training** : [docs/training_strategy.md](docs/training_strategy.md)
- **Atomisation des accords — comment ça marche** (détection + reconstruction + garde lossless + bilan) : [docs/atomisation_accords.md](docs/atomisation_accords.md)
- **Plan multi-voix + token Sustain** : [docs/plan_multivoix_sustain.md](docs/plan_multivoix_sustain.md)
- **Étude empirique initiale Chord_Atomic** : [../projet_a_pt_etre_reprendre/accords_atomiques.md](../projet_a_pt_etre_reprendre/accords_atomiques.md)
- **Datasets** : [../../datasets.md](../../datasets.md)
- **TODO global du projet** : [../../todo.md](../../todo.md)
