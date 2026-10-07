# Datasets — Piano CWT Project

Vue d'ensemble de toutes les sources de données utilisées pour la v2 et leur statut.

---

## 1. Humdrum Chopin First Editions
**Chemin local :** `kern_scrapping/humdrum-chopin-first-editions/kern/`
**Chemin serveur :** `~/taff/kern_all/chopin_first/`
**Source :** https://github.com/pl-wnifc/humdrum-chopin-first-editions
**Format :** Humdrum `**kern` (prêt à tokeniser directement)
**Licence :** CC BY
**État :** ✅ Uploadé sur le serveur, intégré dans `kern_all/`

| Stat | Valeur |
|---|---|
| Fichiers | 512 |
| Notes totales | ~1 140 000 |
| Qualité Humdrum (`!!!COM:`) | 100 % |
| Fichiers avec ornements (T/M/t/m/W/S…) | 243 / 512 (47 %) |
| Tokens ornements | 1 200 (T=480, M=405, t=170, m=46, w=37, s=34, S=12) |
| Fichiers avec notes de grace | 361 / 512 (70 %) |
| Tokens grace notes | 6 614 |

**Contenu :** Œuvre complète de Chopin (~200 pièces), multiple éditions par œuvre (Breitkopf, Peters, Schlesinger, Pleyel, Wessel…) — premières éditions diplomatiques.
**Genres :** Mazurkas (58), Études (27), Préludes (26), Nocturnes (21), Valses (18), Polonaises (16), Ballades (4), Scherzos (4), Sonates (3), Rondos, Concertos…
**Note :** Les fichiers ont 2-11 spines (`**kern` + `**dynam` + `**fing`). Le tokeniseur v2 gère correctement les spines extra.

---

## 2. KernScores — Composers
**Chemin local :** `kern_scrapping/kernscores_composers/`
**Chemin serveur :** `~/taff/kern_all/kernscores/`
**Source :** kern.humdrum.org (scrapé manuellement)
**Format :** Humdrum `**kern`
**État :** ✅ Uploadé sur le serveur (28.8 MB, 5219 fichiers)

**Total : 5219 fichiers** (incluant tous styles : classique, baroque, romantique, ballades, chorales, motets…).

Distribution principale par compositeur (top 30) :

| Compositeur | Fichiers | Notes |
|---|---|---|
| Ballads (anonymes) | 2 565 | Ballades traditionnelles — peuvent biaiser vers monophonie |
| J.S. Bach | 839 | Choral, fugues, inventions |
| Mozart | 453 | Sonates, divertimentos |
| Haydn | 362 | Sonates, quatuors |
| Corelli | 245 | Sonates baroques |
| Beethoven | 195 | Sonates, quatuors |
| Chopin | 92 | Mazurkas, nocturnes, préludes, études, scherzos, valses, ballade |
| Schubert | 63 | — |
| Scarlatti | 60 | Sonates pour clavier |
| Joplin | 45 | Ragtime |
| Frescobaldi | 40 | Toccates, ricercares baroques |
| Vivaldi | 35 | — |
| Hummel | 24 | — |
| Buxtehude | 21 | Préludes, fugues d'orgue |
| Monteverdi | 18 | Madrigaux |
| Clementi | 17 | Sonatines |
| Grieg | 16 | Pièces lyriques |
| Brahms | 14 | — |
| Schumann | 11 | — |
| Sousa | 10 | Marches |
| MacDowell | 9 | — |
| Mendelssohn | 7 | — |
| Weber | 7 | — |
| Giovannelli | 6 | Madrigaux |
| Alkan | 4 | Études virtuoses |
| Himmel | 5 | — |
| Prokofiev | 3 | — |
| Josquin | 3 | Renaissance |
| Victoria | 3 | Motets renaissance |
| Madrigaux | 3 | — |
| **Liszt** | **1** | ⚠️ Très peu de Liszt direct ici |
| ... | ... | (Pachelbel, Ravel, Scriabin, Ives, Field, Byrd, Lassus, … : 1-2 chacun) |

**Note :** Chevauchement probable avec `humdrum-chopin-first-editions` sur les Chopin (92 ici + 512 là). Pas dédupliqué — le modèle verra ces œuvres plusieurs fois (effet d'oversampling sur Chopin, voulu).

**Note historique :** la doc précédente listait "~100 fichiers". C'était une sous-estimation grossière — c'est 50x plus.

---

## 3. ASAP Dataset — MusicXML extraits
**Chemin local (MXL) :** `kern_scrapping/asap_mxl/`
**Chemin local (kern) :** `kern_scrapping/asap_kern/`
**Chemin serveur :** `~/taff/kern_all/asap/`
**Source :** https://github.com/fosfrancesco/asap-dataset
**Format :** MusicXML (`.musicxml`) → converti en kern via `mxl_to_kern.py` v2
**Licence :** Public domain
**État :** ✅ Converti (200/235 piano) + uploadé sur le serveur

Distribution prévue (235 MXL au total) :

| Compositeur | Fichiers MXL | Œuvres notables |
|---|---|---|
| Bach | 59 | WTC préludes+fugues complets (BWV 846-893), Concerto italien |
| Beethoven | 63 | 30+ sonates pour piano (op.2 à op.111) |
| Chopin | 36 | Ballades, Études op.10+25, Scherzos, Sonate op.35 |
| Liszt | 17 | Transcendental Études (7), Paganini Études (Campanella !), Ballade 2, Sonate, Mephisto Waltz |
| Schubert | 15 | Impromptus op.90, Sonates, Wanderer Fantasie |
| Haydn | 12 | Sonates pour clavier |
| Schumann | 11 | Kreisleriana, Arabesque, Toccata |
| Mozart | 6 | Sonates + Fantaisie |
| Ravel | 4 | Gaspard de la Nuit (Ondine !), Miroirs, Pavane |
| Rachmaninoff | 4 | Préludes op.23 (n°4, 6), op.32 (n°5, 10) |
| Debussy | 2 | Images Livre 1 (Reflets), Pour le Piano |
| Scriabin | 2 | Étude op.8 n°11, Sonate n°5 |
| Brahms | 1 | 6 Pièces op.118 n°2 |
| Balakirev | 1 | Islamey |
| Prokofiev | 1 | Toccata |
| Glinka | 1 | The Lark |
| **TOTAL MXL** | **235** | |
| **TOTAL .krn convertis** | **200** | (35 fichiers sans tag instrument piano explicite, filtrés par `--piano-only`) |

**Conversion réalisée (2026-05-19) :**
```bash
python pdmx_scrapper/mxl_to_kern.py \
  --input kern_scrapping/asap_mxl/ \
  --output kern_scrapping/asap_kern/ \
  --jobs 4 --piano-only
# Resultat : 200 OK, 35 non-piano filtres, 0 erreurs
```

**Ornements dans les MXL :** Trilles, arpèges, grace notes — correctement extraits par `mxl_to_kern.py` v2.

---

## 4. PDMX — Corpus principal
**Chemin local (kern) :** `pdmx_scrapper/pdmx_data/kern/` (213 369 fichiers)
**Chemin local (MXL) :** ❌ supprimés après conversion (pour libérer le quota d'inodes)
**Chemin serveur (kern) :** `~/taff/pdmx_scrapper/pdmx_data/kern/` (213 369 fichiers)
**Chemin serveur (symlink) :** `~/taff/kern_all/pdmx/` → `~/taff/pdmx_scrapper/pdmx_data/kern/`
**Source :** https://github.com/pnlong/PDMX — scrape MuseScore public domain
**Format :** kern (reconverti depuis MXL via `mxl_to_kern.py` v2)
**Licence :** MIT (repo) ; ~12 % de fichiers avec statut copyright incertain
**Métadonnées :** `pdmx_scrapper/pdmx_data/PDMX.csv` (215 MB, 254 077 entrées : composer, song_name, license…)
**État :** ✅ Reconverti (2026-05-17/19), inclus dans `kern_all/` via symlink

| Stat | Valeur |
|---|---|
| Fichiers MXL d'origine | 254 035 |
| Fichiers .krn convertis | **213 369** (84.0 %) |
| Non-piano filtrés (skip) | 39 864 (15.7 %) |
| Erreurs de conversion | 802 (0.3 %) — notation shape-note, MXL corrompus |
| % avec < 5 % rests | 88 % (bonne qualité, mesuré sur v1) |
| % avec > 10 mesures répétées | 1 % (dégénéré, à filtrer avec `diagnose_kern.py`) |

**Bug spine swap (v1)** : `mxl_to_kern.py` swappait RH/LH si `parts[0]` était en clé de basse — ~7 % des fichiers affectés en v1.
✅ **Corrigé en v2** (détection `isinstance(clef, BassClef)`) + reconversion complète.

**Ornements / grace notes / arpèges** : perdus en v1 (conversion music21 simple).
✅ **Extraits en v2** : T, t, M, m, W, w, S, s, : (arpège), ; (fermata), q (acciaccatura), Q (appoggiatura).

**Stats compositeur (via PDMX.csv croisé avec les hashes) :**

| Compositeur | Fichiers PDMX |
|---|---|
| ⚠️ `NA` (non identifié) | 163 067 (64 % du PDMX) |
| anon. / Trad. / Traditional / Composer | ~7 500 (~3 %) |
| J.S. Bach | ~880 |
| William Marshall (folk écossais) | 1 185 |
| Yohei Kato (jap.) | 966 |
| Alexander Walker | 343 |
| Niel Gow / Nathaniel Gow (folk écossais) | ~460 |
| Charles Hutchinson Gabriel (hymnes) | 262 |
| John Robson Sweney / William James Kirkpatrick (hymnes) | ~450 |
| **Franz Liszt** (et variantes) | **~70** |
| Beethoven, Chopin, Mozart, etc. | quelques dizaines chacun |

→ **Conclusion** : PDMX est dominé par de l'anonyme/folk/hymnes (>70 %), avec peu de répertoire pianistique romantique pur. Utile en pre-training pour la diversité harmonique, mais le fine-tuning sur Chopin/ASAP est indispensable pour le style.

**Renommage (à faire après Phase 1) :** script `pdmx_scrapper/rename_pdmx_kern.py` + mapping `rename_mapping.tsv` prêt → renommera en `Qm_<composer>_<title>_<hash>.krn` (le prefix `Qm_` garde `is_not_pdmx()` fonctionnel).

---

## Corpus unifié pour la Phase 1 d'entraînement v2

Structure sur le serveur : `~/taff/kern_all/` (lu par `train_cwt.py` via `rglob` + `followlinks`).

| Source | Fichiers | Qualité | Statut |
|---|---|---|---|
| PDMX (symlink) | 213 369 | Variable (64 % anonymes/folk) | ✅ |
| KernScores | 5 219 | Haute (Humdrum) | ✅ |
| Chopin first editions | 512 | Maximale (Humdrum manuel) | ✅ |
| ASAP | 200 | Haute (MXL annoté) | ✅ |
| **TOTAL** | **219 300** | | |

Confirmé par `[Dataset] 219300 fichiers (recursif+symlinks)` dans `train_v2.log`.

---

## Corpus pour la Phase 2 (fine-tuning v2)

Filtre `is_not_pdmx()` (exclut tout fichier dont le nom commence par `Qm`) sur la même `kern_all/` :

| Source | Fichiers |
|---|---|
| KernScores | 5 219 |
| Chopin first editions | 512 |
| ASAP | 200 |
| **TOTAL** | **5 931** |

Plus haute densité de Chopin/Liszt/Bach/Beethoven/Schubert/etc. → focalise le style sur le répertoire pianistique romantique/classique.

---

## Versions de `mxl_to_kern.py`

| Version | Spine swap | Grace notes | Ornements (T/M/S…) | Arpèges (:) | Slurs/Phrases | Dynamics | Tempo text | Pédale | .musicxml |
|---|---|---|---|---|---|---|---|---|---|
| v1 (originale) | Bug (7%) | Non | Non | Non | Partiel | Partiel | Non | Non | Non |
| v2 (corrigée) | ✓ | ✓ | ✓ | ✓ | ✓ (spannerBundle) | ✓ (par mesure) | ✓ (rall/accel/atempo/rubato + poco/molto) | partiel | ✓ |

Le script v2 est utilisé pour les **3 reconversions** : PDMX (213k), ASAP (200), et toute reconversion future.

---

## Stats du répertoire pianistique de référence (Chopin/Liszt)

| Compositeur | Total .krn (3 sources confondues) | Notes |
|---|---|---|
| **Chopin** | 512 + 92 + 36 + ~dizaines PDMX ≈ **~660+** | Couverture complète, multiples éditions (1ère ed.) |
| **Liszt** | 17 (ASAP) + 1 (KernScores) + ~70 (PDMX) ≈ **~88** | Beaucoup mieux qu'estimé initialement, mais incluant des transcriptions Beethoven/Paganini |
| **Beethoven** | 195 (KernScores) + 63 (ASAP) + ?(PDMX) ≈ **260+** | Sonates pour piano essentiellement (KernScores+ASAP) |
| **J.S. Bach** | 839 (KernScores) + 59 (ASAP) + ~880 (PDMX) ≈ **~1800** | Choral + WTC + autres |
| **Mozart** | 453 + 6 + ?(PDMX) ≈ **460+** | Sonates surtout |

Le corpus est très dominé par Bach (1800) et Mozart/Haydn/Beethoven du classique. Chopin a une présence modérée (660+), Liszt minoritaire (88). Pour générer du Liszt convaincant, faudra probablement un fine-tuning par style (v3+).

---

## Multi-staff dans le corpus (audit 2026-06-01)

Tous les corpora ont été scannés pour compter combien de fichiers ont **3 portées musicales ou plus** (`**kern` spines, hors `**dynam`/`**fing`/`**pedal`). C'est important car `mxl_to_kern.py` v2 et `_mxl_to_kern()` v2 **squeezent** systématiquement les scores multi-portées en 2 portées RH/LH (heuristique fragile, perte d'info).

### Distribution par corpus

| Corpus | Total | 1 spine | 2 spines | **3+ spines** | % squeezé |
|---|---|---|---|---|---|
| Chopin first editions | 512 | 35 (6.8%) | 407 (79.5%) | **70** | **13.7%** |
| KernScores | 5 219 | 2 669 (51.1%) | 909 (17.4%) | **1 641** | **31.4%** |
| ASAP MXL (avant conv.) | 235 | 0 | 228 (97.0%) | **7** | 3.0% — Liszt Trans. n°4, Ravel (Ondine, Miroirs 3+4), Scriabin Sonate n°5 |
| ASAP-kern (après conv. squeezée) | 200 | 0 | 200 | 0 | (artéfact du squeeze) |
| PDMX (déjà squeezé par converter) | 213 369 | — | 100% | 0 | (artéfact du squeeze) |

### Impact estimé sur le corpus qualité

**~1 716 fichiers / 5 931** (~**29%**) du corpus qualité avaient ≥3 portées source et ont donc été partiellement déformés par le squeeze v2.

Détail par sous-corpus :
- KernScores : 31.4 % concerné — beaucoup de Bach choral 4-voix, fugues, polyphonie baroque/Renaissance
- Chopin first editions : 13.7 % — pièces de jeunesse avec accomp. orchestral (Op.2 Variations, concerto), trios
- ASAP : 3 % — surtout Liszt virtuose (Jeux d'eaux, Transcendental n°4) + Ravel impressionniste

### Observation annexe — fichiers monophoniques KernScores

**51 %** de KernScores (2 669 / 5 219) ont **1 seule spine** : chants traditionnels (`deut*.krn`, `eng*.krn`, `oest*.krn`...), hymnes, ballades populaires monophoniques. Ces fichiers entraînent le modèle à de la monophonie, ce qui biaise vers un style melody-only qui n'a pas de sens pour du piano 2-mains.

→ Seulement **17.4 %** de KernScores (909 fichiers) sont des **vrais 2-spines pianistique propres**.

### Décision v2 vs v3

- **v2 (en cours)** : statu quo. Tous les fichiers passent par le squeeze, on accepte ~29 % de contenu un peu déformé. Pas la peine d'éjecter 30 % du corpus.
- **v3** : extension H11 à `Staff_1..Staff_N` pour récupérer la fidélité totale. Aussi : filtrer les monophoniques (ou les tagger explicitement). Voir todo.md `## v3 — à intégrer`.

---

## Tri qualité du corpus v3 — déductions & décisions (itération 2026-06)

Travail de conception du filtrage corpus pour le pre-training/fine-tuning v3. Point de départ : `music_cwt/data_analysis/results/analysis_recap.md` (filtre structurel v1 = `diagnose_density.py`, 8 296 rejetés / 3.8 %, validé sur 18 fichiers inspectés à la main).

### Constat sur le filtre structurel seul

Le filtre structurel (densité/skew/rests/longueur) a deux catégories parfaitement précises (`empty`, `too_short`, `density_skew` : 9/9 confirmés mauvais) et trois problématiques :
- **`hyper_dense` (npm > 24, p99)** : faux positifs sur de la virtuosité légitime (Bach BWV 564, Beethoven Op.78). Le vrai bug "mesure sur-condensée" est `max >> mean`, déjà capturé par `density_skew`. → relâcher à npm > ~50 (garde-fou).
- **`long_rest` (compte de lignes)** : ignore la POSITION. Du silence en tête/queue est trimmable, pas une raison de jeter.
- **`mostly_rests` (pct global)** : conflate "stave déclarée jamais jouée" (à jeter) et "stave intermittente musicale" (Liszt Jeux d'eau, légitime).

### Axe métadonnées — `PDMX.csv` (signaux qualité MuseScore)

`PDMX.csv` (254 077 lignes) contient bien plus que composer/title. Signaux qualité exploitables :
- `rating`, `n_ratings`, `is_rated` : étoiles utilisateurs MuseScore.
- `subset:rated` / `subset:rated_deduplicated` : sous-ensembles curés par les auteurs PDMX (bien noté + dédupliqué).
- `is_original` : composition originale vs arrangement.
- `genres`, `tags`, `composer_name`, `complexity`, `n_tracks`.
- `scale_consistency`, `pitch_class_entropy`, `groove_consistency` : cohérence musicale calculée sur le MIDI (proxy "vraie musique").
- `is_user_pro` / `is_user_publisher` / `is_user_staff` : fiabilité de la source.

**Couverture mesurée (sur les 254 077 lignes) :**

| Signal | Compte | % |
|---|---:|---:|
| `n_ratings ≥ 1` | 14 182 | 5.6 % |
| `subset:rated_deduplicated` | 13 187 | 5.2 % |
| `is_original` | 9 745 | 3.8 % |
| valeur rating | 3★:27 · 4★:2 374 · 5★:11 781 | — |
| buckets n_ratings | 0:239 895 · 3-9:8 562 · 10+:5 620 | — |

**Déductions clés :**
1. **Le rating ne peut PAS être un filtre en dur** — 94 % du corpus n'a aucun vote, on détruirait le corpus.
2. **La valeur du rating ne discrimine presque rien** (quasi tout 4-5★ : les gens ne notent que ce qu'ils aiment). Le signal discriminant est la **PRÉSENCE** de votes (`n_ratings ≥ 3`), pas la valeur. → rating = signal POSITIF (tier), jamais couperet.
3. **PDMX fournit un tier "or" prêt à l'emploi** : `subset:rated_deduplicated`.

**Comptes Tier A (join `PDMX.csv` complet, à reconfirmer sur `kern_all` réel) :**
- Tier A (`rated_deduplicated`) : **13 181** fichiers.
- Tier A ∩ (classique OU original) : **5 929** fichiers.

### Le join est invariant à la reconversion

Le join `kern` ↔ `PDMX.csv` se fait par **hash IPFS (CID Qm...)** = stem du nom de fichier = stem de la colonne `mxl`. Le hash est l'identité de la source ; reconvertir ne change pas le nom de fichier. Les métadonnées viennent du CSV, pas du contenu kern. → `tier_A.txt`, `tier_A_classical.txt`, `meta_flags.csv` sont **permanents**, produisables maintenant, valides après n'importe quelle reconversion.

### Décision tiers d'échantillonnage v3

- **Tier A classique** (~6k, `tier_A_classical.txt`) → **AJOUTER au fine-tuning** (`is_not_pdmx` actuel ~6k → ~12k). Filtré sur classique/original pour ne pas diluer le style avec pop/jeux vidéo (même bien notés). Décision validée avec l'utilisateur.
- **Tier A complet** (13k) → sur-échantillonnage fin de pre-training.
- **Tier B** (reste passant les filtres structurels) → pre-training de base.
- **Tier C** (jeté) → échoue au structurel OU flag métadonnées (`is_original=0` + genre non-classique, `scale_consistency` bas).

### Proxy "silence tête/queue = qualité" — validée par décomposition

Hypothèse utilisateur : un transcripteur sérieux ne laisse pas de silence en début/fin → proxy de qualité. **Bonne mais à éclater en 3** (mesuré en MESURES, indép. de la subdivision et du nb de spines) :
- `leading_silent` / `trailing_silent` (bord) → **trimmable**, on récupère le fichier.
- `max_internal_silent` (hors bords) → vrai défaut structurel (fragments collés) → rejet.

Smoke-test sur les `bad_sample/` : la décomposition sépare exactement les cas décrits à la main — `long_rest_3` (lead=21, pause au début), `long_rest_2` (trail=11, ligne vide à la fin), `long_rest_1` (int=7, concaténation à jeter). Détail : `lead=1` / `trail=1` sont **ubiquitaires** (anacrouse / en-tête avant 1ère barline) → seuils `≥ 2`. La proxy est **testable contre `is_rated`** (vérité-terrain) via `validate_silence_proxy.py` — le silence de bord étant une propriété de la source (pas un artefact du squeeze), le verdict transfère à la v3.

### Décision de séquençage : multi-staff D'ABORD, tri structurel ENSUITE

Les résultats du filtre **structurel** dépendent du contenu kern, donc de la conversion. **Le squeeze 3→2 portées de l'ancienne pipeline produit lui-même des `hyper_dense`/`density_skew`** (= le symptôme "mesures sur-condensées"). Figer le blacklist structurel sur l'ancien kern jetterait des fichiers 3-portées légitimes qui se convertiraient proprement en v3.

→ **Décision (utilisateur)** : implémenter le multi-staff, reconvertir tout, PUIS faire le tri structurel sur le kern propre. Le tri métadonnées (hash-based) tourne en parallèle dès maintenant. Re-run du filtre structurel = cheap (~min de parsing) ; seuls la logique et les seuils transfèrent. Le test `dead_spine` doit être écrit pour **N portées** (H11 = `Staff_1..N`) → couplé au chantier multi-staff. Séquence détaillée : `todo.md` > "▶ PROCHAINES ÉTAPES — chantier reconversion v3".

### Scripts produits (`music_cwt/data_analysis/`)

| Script | Rôle | Statut |
|---|---|---|
| `pdmx_metadata.py` | Join `kern_all` ↔ `PDMX.csv` par hash → `meta_flags.csv`, `tier_A.txt`, `tier_A_classical.txt` | écrit, smoke-testé local, à lancer serveur |
| `validate_silence_proxy.py` | Teste proxy silence bord ↔ `is_rated`, décompose leading/trailing/internal | écrit, smoke-testé local, à lancer serveur |
| `diagnose_density.py` (existant) | Filtre structurel — à REFONDRE (point 2) après reconversion | refonte gated multi-staff |
