# Projet en standby — Accords atomiques en tokenisation

> **Statut : EN STANDBY.** À tester en parallèle de la v3 du modèle pour comparer empiriquement.
> Le gain de compression mesuré (~11-13% de chord events compressés au total) est modeste,
> mais l'inductive bias harmonique reste un argument à valider expérimentalement.

---

## Synthèse rapide

Pour la v2/v3 du tokeniseur CWT, on envisage de **compresser certains accords en un seul compound token** au lieu de N tokens via `Chord_Start/Cont/End`.

**Décisions de design validées :**
- H1 reçoit un nouveau type `Chord_Atomic` (UNE seule entrée)
- Deux nouveaux heads s'activent seulement quand H1=Chord_Atomic :
  - `H_chord_quality` : qualité harmonique (8 valeurs, voir étude empirique)
  - `H_chord_voicing` : layout complet (inversion + spacing + doublures) en une seule catégorie (10 valeurs)
- H3 sert à coder la **fondamentale** absolue de l'accord
- **Garantie zéro perte d'info** : si un accord ne rentre pas dans le schéma atomique → fallback automatique sur le path séquentiel (Chord_Start/Cont/End) du v1
- Pas de hard-coding des règles inter-accord — le modèle apprend les transitions par les données

---

## Étude empirique (déjà réalisée, 2026-05)

Script : `pdmx_scrapper/chord_shape_stats.py` (initialement à la racine, à déplacer ici).
Données analysées :
- 512 fichiers Chopin first editions
- 5219 fichiers KernScores composers

### Distribution par taille d'accord

| Taille | Chopin | KernScores |
|---|---|---|
| **2 notes (dyades)** | **66.7%** | **71.5%** |
| 3 notes (triades) | 27.2% | 23.4% |
| 4 notes | 5.7% | 4.8% |
| 5+ notes | <0.3% | <0.2% |

**Découverte clé :** la majorité des "chord events" en kern sont des **dyades** (deux voix simultanées dans la même main). Notre schéma ne les atomise pas (triades minimum requises pour matcher une qualité).

### Taux d'atomisation réel

| Population | Chopin | KernScores |
|---|---|---|
| Sur tous les chord events | 11.6% | 13.4% |
| Sur les accords 3+ notes uniquement | **34.8%** | **47.1%** |

### Qualités effectivement utilisées (sur 3+ matched)

Distribution très concentrée :

| Qualité | Chopin | KernScores |
|---|---|---|
| Maj | 41% | 53% |
| min | 30% | 22% |
| dim | 16% | 14% |
| dom7 | 5% | 5% |
| sus2 | 2% | 0.8% |
| dim7 | 1.5% | 2.3% |
| sus4 | 1.4% | 0.6% |
| half_dim7 | 1% | 0.7% |
| aug | 0.7% | 0.4% |
| min7, Maj7, aug7 | <1% | <1% |

→ 4 qualités (Maj, min, dim, dom7) couvrent **92-94%** des cas.

### Voicings effectivement utilisés (sur 3+ matched, couverture cumulative)

| Voicing | Chopin % | Cumul |
|---|---|---|
| close_inv2_nodb | 28.5% | 28.5% |
| close_inv1_nodb | 22.0% | 50.6% |
| close_root_nodb | 17.8% | **68.3%** |
| open_root_nodb | 7.3% | 75.7% |
| open_inv1_dbX | 6.8% | 82.5% |
| open_inv2_dbX | 5.9% | 88.5% |
| open_root_dbR | 4.9% | **93.3%** |
| close_inv3_nodb | 1.9% | 95.2% |
| open_inv1_nodb | 1.5% | 96.7% |
| open_inv2_nodb | 1.0% | 97.7% |

→ **10 voicings couvrent 97.7%**. Pas besoin de "wide", "dbMulti", "other" — fallback séquentiel.

### Décoration per-note (bloque l'atomisation)

25-30% des accords 3+ notes ont une décoration per-note (ornement, articulation, tie, slur) → fallback séquentiel forcé. Déjà compté dans les taux d'atomisation ci-dessus.

---

## Listes finales recommandées

### H_chord_quality (8 valeurs + NULL = 9)
1. `Maj`
2. `min`
3. `dim`
4. `aug`
5. `dom7`
6. `dim7`
7. `half_dim7`
8. `sus4` (ou `sus2`)

→ skip `Maj7`, `min7`, `aug7`, `minMaj7`, `dom9` (<0.5% au total)

### H_chord_voicing (10 valeurs + NULL = 11)
1. `close_root_nodb`
2. `close_inv1_nodb`
3. `close_inv2_nodb`
4. `close_inv3_nodb`
5. `open_root_nodb`
6. `open_root_dbR`
7. `open_inv1_nodb`
8. `open_inv1_dbX`
9. `open_inv2_nodb`
10. `open_inv2_dbX`

---

## Plan d'implémentation détaillé (à reprendre tel quel)

### Architecture des nouveaux heads

**H1 — Type (étendu)**
Ajout d'une seule valeur : `Chord_Atomic`. H1 passe de 8 à 9 valeurs.

**H_chord_quality (nouveau, 9 valeurs)**
Voir liste ci-dessus. NULL forcé quand H1 ≠ Chord_Atomic.

**H_chord_voicing (nouveau, 11 valeurs)**
Voir liste ci-dessus. NULL forcé quand H1 ≠ Chord_Atomic.

**H3 — Pitch (utilisation étendue)**
Quand H1=Chord_Atomic, H3 code la **fondamentale absolue** de l'accord. Vocab inchangé (282 tokens).

**Autres heads quand H1=Chord_Atomic :**
- H2 (Duration) : durée commune
- H4-H8 (Ornement, Articulation, Tie, Slur, Phrase) : **forcés NULL**
- H9, H10 (Voicing stems, Beam) : valeur commune à toutes les notes
- H11 (Spine), H12 (Dynamic) : comme d'habitude

**Règle de fallback (préservation totale d'info)**
Un accord est tokenisé en `Chord_Atomic` ssi :
1. Sa qualité harmonique matche une des 8 entrées de H_chord_quality
2. Son voicing matche une des 10 entrées de H_chord_voicing
3. Aucun membre ne porte d'ornement, articulation, tie ou slur per-note (H4-H8 NULL)
4. Tous les membres ont la même durée (garanti par kern)

Sinon → fallback Chord_Start/Cont/End (path v1 inchangé).

### Fichiers à modifier (chemins relatifs)

| Fichier | Modification |
|---|---|
| `music_cwt/v3/  (à créer le moment venu)/tokenization_rules_reviewed.md` | Documenter Chord_Atomic, H_chord_quality, H_chord_voicing |
| `music_cwt/v3/  (à créer le moment venu)/tokenizers/multihead_tokenizer.py` | Ajouter détection qualité/voicing + émission atomique vs séquentielle |
| `music_cwt/v3/  (à créer le moment venu)/postprocess.py` | Ajouter expansion `(quality + voicing + root) → kern chord` |
| `music_cwt/v3/  (à créer le moment venu)/cwt_model.py` | Ajouter embeddings + prediction heads + masques de loss conditionnels |

### Étapes (référence existante : `music_cwt/v1/tokenizers/multihead_tokenizer.py:571-627`)

1. **Tokeniseur** : fonctions `_detect_chord_quality`, `_detect_root_and_quality`, `_detect_voicing`, `_try_atomize_chord`. Si retour non-None → 1 compound token avec H1=Chord_Atomic. Sinon → émission Chord_Start/Cont/End actuelle.

2. **Postprocesseur** (référence : `music_cwt/v1/postprocess.py:75-159`) : tables `_QUALITY_TO_INTERVALS` et `_VOICING_BUILDERS`, fonction `_expand_atomic_chord`. Dans `_group_to_kern`, si H1=Chord_Atomic → déléguer à l'expansion.

3. **Modèle** : embeddings input pour H_chord_quality et H_chord_voicing (sommés aux autres), output heads en Stage 2, masques de loss (NULL forcé quand H1 ≠ Chord_Atomic).

4. **Tests round-trip** : `music_cwt/v3/  (à créer le moment venu)/tests/test_roundtrip_chords.py` — kern → tokenize → detokenize → comparer. Zéro différence requise.

5. **Stats de couverture** : sur le corpus complet, % atomique vs séquentiel, top-20 des fallbacks.

### Risques

- Confusion modèle entre paths atomique et séquentiel : si déséquilibre extrême (>10x), envisager pondération de loss différentielle.
- Bug de reconstruction : round-trip tests bloquent la mise en production.
- Coût en paramètres : ~20K params total → négligeable.

---

## Critère de décision pour reprendre ce projet

À évaluer après l'entraînement v3 :

- **Reprendre** si la v3 montre :
  - Une cohérence harmonique faible dans les accords générés (sons ambigus, mal résolus)
  - Une exploitation faible de la fenêtre de contexte (saturation à 1024-1500 tokens malgré max_seq_len=2048)
  - Une difficulté à apprendre les transitions d'accords idiomatiques

- **Abandonner définitivement** si la v3 :
  - Génère déjà des accords musicalement satisfaisants
  - A une bonne mémoire long-terme (RoPE + Stage 3 suffisent)
  - Le gain estimé (~11-13% de compression) ne justifie pas la complexité

---

## Fichiers liés conservés dans ce dossier

- `accords_atomiques.md` (ce fichier)
- `chord_shape_stats.py` — script d'analyse (à relancer sur les corpus reconvertis si besoin)
- `_analyze_chord_stats.py` — helper d'analyse approfondie des CSV produits
- `chord_stats_chopin.csv` — résultats Chopin first editions (214k events)
- `chord_stats_kernscores.csv` — résultats KernScores (156k events)
