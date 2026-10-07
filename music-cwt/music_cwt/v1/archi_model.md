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
| **CWT** (ce fichier) | Multi-head (12 heads) | Modèle principal, expressif |
| **GPT + BPE** | Flat BPE sur kern brut | Baseline de comparaison |

---

## 1. Tokenisation en entrée — Compound Embedding

Chaque token = 12 valeurs catégorielles (H1–H12).

**Représentation** : somme des 12 embeddings individuels + positional embedding appris.

```
x_t = pos_embed(t) + Σ_h embed_h(token_h_t)
```

Vocab total : ~378 valeurs réparties sur 12 heads (voir `tokenization_rules_reviewed.md`).

**Amélioration v2** : tester la concaténation + projection linéaire à la place de la somme.

---

## 2. Backbone — GPT-like (décodeur causal)

Transformer décodeur standard avec masque causal.  
Chaque bloc : LayerNorm → CausalSelfAttention → résidu → LayerNorm → FFN → résidu.

---

## 3. Génération des heads en sortie — 2 étapes conditionnées

### Motivation
Quand H1 = `Barline`, H2–H12 doivent tous être `NULL`.  
Quand H1 = `Rest`, H4/H5/H8 doivent être `NULL`.  
Quand H11 = `LH`, H12 (dynamique) doit être `NULL`.  

Il faut donc que les heads "secondaires" voient la prédiction des heads "structuraux" avant de décider.

### Ordre de génération

**Stage 1** (depuis `h_t` directement) :
- H1 — Type (gatekeeper structurel : Barline, Rest, Chord...)
- H11 — Spine (RH/LH, conditionne la dynamique)

**Stage 2** (depuis `h_t + embed(H1_pred) + embed(H11_pred)`) :
- H2 Duration · H3 Pitch · H4 Ornament · H5 Articulation
- H6 Tie · H7 Slur · H8 Phrase
- H9 Voicing · H10 Beam · H12 Dynamic

En **training** : teacher forcing sur H1/H11 (on passe les vrais labels comme conditionnement).  
En **inférence** : argmax de stage 1 utilisé comme conditionnement de stage 2.

### Contraintes structurelles (appliquées dans la loss)

| Condition | Heads forcées NULL dans la loss |
|---|---|
| H1 = Barline ou SEP | H2–H12 |
| H1 = Rest | H4, H5, H8 |
| H1 = Fioritura_Q | H4, H8 |
| H11 = LH | H12 |

**Amélioration v2** : ajouter un stage 3 (H4/H5 conditionnés sur H2+H3) pour capturer les dépendances ornementation/articulation → durée/hauteur.

---

## 4. Config modèle (~10 M paramètres, premier test)

| Hyperparamètre | Valeur |
|---|---|
| d_model | 256 |
| n_layers | 9 |
| n_attn_heads | 8 (head_dim = 32) |
| d_ff | 1 536 |
| max_seq_len | 2 048 |
| dropout | 0.1 |
| **Paramètres totaux** | **~10.2 M** |

Estimation training sur GPU Polytechnique (14h nuit) : ~1-2 epochs sur kern_data avec batch_size=4, grad_accum=8 (batch effectif = 32), seq_len=2048.

---

## 5. Entraînement — stratégie

**Phase 1** : `kern_data/` complet — pièces complètes Humdrum, 2 spines (RH+LH), qualité maximale.  
**Phase 2** (fine-tuning) : harmonisation melody→score via le mécanisme SEP déjà en place.  
**Phase 3** (si résultats OK) : fine-tuning par style (ST_chopin, ST_liszt).

**Dataset** : lazy loading (tokenisation à la volée), padding dynamique par batch, masque de padding dans la loss. Pas de build_dataset.py — les fichiers sont lus et tokenisés au moment du training.

**Batching** : séquences < max_seq_len paddées à max(lengths) dans le batch. Séquences > max_seq_len : crop aléatoire à max_seq_len. Positions paddées masquées dans la loss.

**Loss** : somme des cross-entropy sur les 12 heads (pondération uniforme v1).  
Trois masques combinés : contraintes structurelles + padding + NULL obligatoires.

**Checkpoints** : toutes les 30 min → stockage NFS Polytechnique (`~/taff/checkpoints/`).

---

## 6. Génération (inférence)

```
[Meta_melody: K, M, MM, CLEF_G] → [melody tokens] → SEP → [Meta_score: K, M, MM, CLEF_G, CLEF_F]
→ génération autoregressive → post-processing → fichier .krn valide
```

Sampling : température + top-k.  
Contrainte d'inférence : si N tokens RH consécutifs sans LH → forcer LH.

---

## 7. Améliorations à tester (v2 et au-delà)

| Amélioration | Priorité | Note |
|---|---|---|
| Concaténation + proj. à la place de la somme en entrée | Moyenne | +expressivité, +params |
| Stage 3 : H4/H5 conditionnés sur H2+H3 | Moyenne | Dépendances fin-grain |
| RoPE à la place du positional embedding appris | Haute | Mieux pour longues séquences (>2048) |
| Loss pondérée par fréquence inverse | Basse | Heads rares (H4, H8) sous-représentées |
| Fine-tuning par style (ST_chopin, ST_liszt) | Haute | Après phase 1+2 validées |
| GPT+BPE baseline | Haute | Comparaison directe |
| Beam search / nucleus sampling | Basse | Après que température+top-k marche |
| Flash Attention | Haute | Si passage à cloud (seq_len > 2048) |
