# Architecture — Compound Word Transformer pour piano kern

## Dataset (métriques calculées sur 437 190 fichiers)

| Métrique | Valeur |
|---|---|
| Total tokens | 86 M |
| p50 | 126 |
| p75 | 181 |
| p90 | 321 |
| p95 | 499 |
| p99 | 1 522 |
| Max | 75 632 |

**max_seq_len = 2048** → couvre >99.5 % des fichiers (p99 = 1 522). Les rares fichiers plus longs (0.1 %) sont tronqués par crop aléatoire.

`max_seq_len` n'a pas à être une puissance de 2 — c'est une convention pratique pour l'alignement mémoire, pas une contrainte matérielle.

Estimation : 2-3 min de musique ≈ 60 mesures × ~25 tokens/mesure = 1 000-1 500 tokens → tient dans un seul contexte de 2048.

---

## Deux modèles prévus

| Modèle | Tokenisation | Objectif |
|---|---|---|
| **CWT** (ce fichier) | Multi-head (14 heads — H1–H14) | Modèle principal, expressif |
| **GPT + BPE** | Flat BPE sur kern brut | Baseline de comparaison (reporté à v3+, cf. §7) |

---

## 1. Tokenisation en entrée — Compound Embedding (v2 : concat + projection)

Chaque token = 14 valeurs catégorielles (H1–H14, avec ajout H13 Pédale et H14 Tempo en v2).

**Représentation v2 : concaténation + projection linéaire** (à la place de la somme v1).

```
x_t = Linear([embed_1(tok_1) || embed_2(tok_2) || ... || embed_14(tok_14)])
```

- Concat : (14 × d_model) = 14 × 320 = 4480 dimensions par token
- Projection linéaire vers d_model = 320 (~1.43 M params pour cette couche)
- Plus expressif que la somme : le modèle apprend à pondérer chaque head au lieu de tout mixer additivement

**Plus de positional embedding appris en entrée** — la position est encodée par RoPE dans l'attention (§ 2).

Vocab total : ~398 valeurs réparties sur 14 heads (voir `tokenization_rules_reviewed.md`).

---

## 2. Backbone — GPT-like (décodeur causal) avec RoPE

Transformer décodeur standard avec masque causal.
Chaque bloc : LayerNorm → CausalSelfAttention(avec RoPE) → résidu → LayerNorm → FFN → résidu.

**RoPE (Rotary Positional Embedding)** appliqué aux Q et K dans chaque couche d'attention :
- Encode la position via une rotation 2D par paire de dimensions
- Pas de paramètres appris (vs ~524 K params économisés par rapport au pos_embed v1)
- Meilleure généralisation aux séquences plus longues que celles vues à l'entraînement
- Encode la position relative implicitement
- Cache cos/sin partagé entre toutes les couches

---

## 3. Génération des heads en sortie — 3 étapes conditionnées (v2)

### Motivation
Quand H1 = `Barline`, H2–H14 doivent tous être `NULL`.
Quand H1 = `Rest`, H4/H5/H8 doivent être `NULL`.
Quand H11 = `LH`, H12 (dynamique) doit être `NULL`.

Les heads "secondaires" voient la prédiction des heads "structuraux" avant de décider.
**v2 ajoute un Stage 3** pour les heads dépendant fortement de la durée+hauteur (ornement, articulation).

### Ordre de génération

**Stage 1** (depuis `h_t` directement) :
- H1 — Type (gatekeeper structurel : Barline, Rest, Chord...)
- H11 — Spine (RH/LH, conditionne la dynamique)

**Stage 2** (depuis `h_t + embed(H1_pred) + embed(H11_pred)`) :
- H2 Duration · H3 Pitch
- H6 Tie · H7 Slur · H8 Phrase
- H9 Voicing · H10 Beam · H12 Dynamic · H13 Pédale · H14 Tempo

**Stage 3** (depuis `cond_stage2 + embed(H2_pred) + embed(H3_pred)`) :
- H4 Ornement · H5 Articulation

Motivation Stage 3 : un trille (`T`) a plus de sens sur une noire pointée que sur une triple-croche ; un staccato (`'`) a plus de sens sur une croche aigüe que sur une ronde grave. Conditionner H4/H5 sur (H2, H3) capture cette dépendance fine au lieu de la laisser au seul backbone d'attention.

En **training** : teacher forcing sur H1/H11 (Stage 1 → Stage 2) et sur H2/H3 (Stage 2 → Stage 3).
En **inférence** : argmax des stages précédents utilisé comme conditionnement.

### Contraintes structurelles (appliquées dans la loss)

| Condition | Heads forcées NULL dans la loss |
|---|---|
| H1 = Barline ou SEP | H2–H14 |
| H1 = Rest | H4, H5, H8 |
| H1 = Fioritura_Q/q | H4, H8 |
| H11 = LH | H12, H13, H14 |

---

## 4. Config modèle (~16.2 M paramètres, v2)

| Hyperparamètre | Valeur v2 | (v1 pour rappel) |
|---|---|---|
| d_model | **320** | 256 |
| n_layers | **10** | 9 |
| n_attn_heads | 8 (head_dim = 40) | 8 (head_dim = 32) |
| d_ff | **1 600** | 1 536 |
| max_seq_len | 2 048 | 2 048 |
| dropout | 0.1 | 0.1 |
| **Paramètres totaux** | **16 157 120 (~16.2 M)** | ~10.2 M |

Croissance v1 → v2 (~6 M params en plus) :
- Backbone agrandi (d_model 256→320, layers 9→10, d_ff 1536→1600) : la majorité
- `+1.43 M` : projection linéaire `concat(14 × 320) → 320` du CompoundEmbedding (v2 spécifique)
- `+ ~50 K` : embeddings de conditionnement Stage 3 (H2, H3) + linear heads Stage 3 (H4, H5)
- `−655 K` : suppression du positional embedding appris (max_seq_len × d_model = 2048 × 320), remplacé par RoPE sans paramètres
- `+ ~60 K` : embeddings + heads pour H13 (Pédale) et H14 (Tempo)

Mesuré sur RTX A4000 (16 GB) : ~10.2 it/s avec batch=4, grad_accum=8 (batch effectif=32), seq_len=2048. → ~85 min/epoch sur 219k fichiers → ~70h pour 50 epochs (3 jours).

---

## 5. Entraînement — stratégie

**Phase 1 (pre-training)** : sur l'ensemble du corpus = PDMX reconverti + ASAP + Chopin first editions + KernScores (~219k fichiers). Voir `training_strategy.md` pour le détail des corpus.
**Phase 2 (fine-tuning)** : sur la partie qualité = tout sauf PDMX (ASAP + Chopin + KernScores, ~6k fichiers). Filtre par `is_not_pdmx()` qui exclut les noms commençant par `Qm`.
**Phase 3+ (futur)** : fine-tuning par style (`ST_chopin`, `ST_liszt`) — voir §7.

**Dataset** : lazy loading (tokenisation à la volée), padding dynamique par batch, masque de padding dans la loss. Pas de build_dataset.py — les fichiers sont lus et tokenisés au moment du training. Listing récursif avec `os.walk(followlinks=True)` (suit les symlinks de dossier, utile pour la structure `kern_all/pdmx -> pdmx_scrapper/pdmx_data/kern`).

**Batching** : séquences < max_seq_len paddées à max(lengths) dans le batch. Séquences > max_seq_len : crop aléatoire à max_seq_len. Positions paddées masquées dans la loss.

**Loss** : somme des cross-entropy sur les 14 heads (pondération uniforme en v2).
Trois masques combinés : contraintes structurelles + padding + NULL obligatoires.

> Pondération inverse-fréquence reportée à v3 (cf. §7) si les heads rares (ornements H4, phrases H8, dynamique H12) ne s'apprennent pas correctement en uniforme.

**Checkpoints** : toutes les 30 min (`_mid`) + à chaque fin d'epoch → stockage NFS Polytechnique (`~/taff/v2/checkpoints/`). Cleanup automatique : 3 derniers + multiples de 15 conservés (cf. `cleanup_checkpoints` dans `train_cwt.py` et `finetune_humdrum.py`).

**Auto-resume après crash/reboot** : wrapper bash `auto_train.sh` (boucle while + flock) + cron `@reboot` sur la machine d'entraînement. Reprise depuis le dernier checkpoint valide via `load_latest_checkpoint`.

---

## 6. Génération (inférence)

```
[Meta_melody: K, M, MM, CLEF_G] → [melody tokens] → SEP → [Meta_score: K, M, MM, CLEF_G, CLEF_F]
→ génération autoregressive → post-processing → fichier .krn valide
```

Sampling : température + top-k.  
Contrainte d'inférence : si N tokens RH consécutifs sans LH → forcer LH.

---

## 7. Améliorations — v2 (implémentées) et v3+ (reportées)

### v2 — implémentées (entraînement en cours)

| Amélioration | Statut | Note |
|---|---|---|
| **RoPE** à la place du positional embedding appris | ✅ Implémenté | Mieux pour longues séquences, réduit la dérive de thème |
| **Stage 3** : H4/H5 conditionnés sur H2+H3 | ✅ Implémenté | Dépendances fin-grain ornement/articulation → durée/hauteur |
| **Concaténation + proj.** à la place de la somme en entrée | ✅ Implémenté | +1.43 M params pour la projection |
| **Modèle agrandi** (d=320, layers=10) | ✅ Implémenté | 10.2 M → 16.2 M params |
| **Corpus enrichi** (PDMX + ASAP + Chopin + KernScores) | ✅ Implémenté | ~219k fichiers vs 213k PDMX seul en v1 |
| **Filtre fine-tuning** : tout sauf PDMX | ✅ Implémenté | `is_not_pdmx()` au lieu du filtre `!!!COM:` strict |
| **Cleanup checkpoints automatique** | ✅ Implémenté | 3 derniers + multiples de 15 |
| **Auto-resume après crash/reboot** | ✅ Implémenté | `auto_train.sh` + cron `@reboot` |

### v3+ — reportées (à reprendre selon résultats v2)

| Amélioration | Critère de reprise | Note |
|---|---|---|
| **Loss pondérée par fréquence inverse** | Si les ornements (H4) ou phrases (H8) n'apparaissent pas dans les générations v2 | Reporté à v3 — heads rares (H4, H8, H12) sous-représentées |
| **Accords atomiques** (H_chord_quality + H_chord_voicing) | Si v2 montre faiblesse harmonique ou saturation de contexte | Plan détaillé : `music_cwt/projet_a_pt_etre_reprendre/accords_atomiques.md` |
| **Fine-tuning par style** (ST_chopin, ST_liszt) | Après v2 validée musicalement | Conditionne la génération sur le compositeur |
| **GPT+BPE baseline** | Quand v2 stable | Comparaison directe pour mesurer l'apport du multi-head |
| **Beam search / nucleus sampling** | Si température+top-k saturent | Alternative de sampling |
| **Flash Attention** | Si passage à cloud (seq_len > 2048) | Drop-in, ~3x plus rapide |
