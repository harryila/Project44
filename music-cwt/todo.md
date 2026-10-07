# TODO — Projet CWT Piano

> Règle : dès qu'une piste intéressante est évoquée, l'ajouter ici.
> Ce fichier est la source de vérité des tâches en attente. Les autres fichiers (README, CLAUDE.md) peuvent référencer ces tâches mais ne les dupliquent pas.

---

## En cours

- [~] **Pre-training v2 (Phase 1)** sur `kern_all/` (219 300 fichiers : PDMX + ASAP + Chopin first + KernScores).
  Machine : **portugal** (RTX A4000 16GB, GPU libre), ~10.2 it/s, 50 epochs ≈ 70h (3 jours).
  Resume automatique via `auto_train.sh` + cron `@reboot`. Loss à ep002 step 11k+ = 0.076.
  Surveillance : `ssh poly-albatros 'tail -20 ~/taff/v2/train_v2.log'`. Journal : `music_cwt/v2/docs/training_strategy.md`.

- [ ] **Resync server → local** des 213k .krn reconvertis si besoin de tester en local (`rsync -avzP poly-albatros:~/taff/pdmx_scrapper/pdmx_data/kern/ local/`). Optionnel — pas nécessaire pour l'entraînement (tout est sur le serveur).

---

## ▶ PROCHAINES ÉTAPES — chantier reconversion v3 (décidé 2026-06)

**Décision actée :** multi-staff D'ABORD → reconversion complète → tri structurel sur le kern propre. Le tri métadonnées (hash-based) est permanent et peut tourner à tout moment en parallèle. Détails et justification : section `## v3 — à intégrer` > "Axe qualité par métadonnées" et "Support multi-staff".

Ordre d'exécution (dépendances) :

0. **[parallèle, non bloquant] Tri métadonnées sur le serveur** — uploader `PDMX.csv` (215 MB) + `pdmx_metadata.py` + `validate_silence_proxy.py` sur `poly-albatros`, lancer les deux. Sorties permanentes (`tier_A.txt`, `tier_A_classical.txt`, `meta_flags.csv`, `silence_proxy_report.txt`). Survit à la reconversion (join par hash IPFS). Donne les vrais comptes Tier A présents + verdict de la proxy silence.

1. **Implémenter le multi-staff** (cœur du chantier, ~3-5j) — voir les 7 changements dans "Support multi-staff (3+ portées)" ci-dessous. Inclut la suppression du squeeze dans `pdmx_scrapper/mxl_to_kern.py` ET `_mxl_to_kern()` de `generate.py`, l'extension H11 → `Staff_1..N`, le tokeniseur, le postprocess, le meta `N_STAVES`, les loss masks.

2. **Re-download des MXL PDMX** — PRÉREQUIS de la reconversion PDMX : `pdmx_data/mxl/` a été vidé (`--delete-mxl`). Re-télécharger les 254k MXL via `pdmx_scrapper/download_pdmx.py`. ⚠️ Vérifier le quota d'inodes (limite 300k) AVANT — refaire la conversion en streaming (`--delete-mxl` au fur et à mesure) comme en 2026-05. Les sources non-PDMX (ASAP MXL, Chopin Humdrum, KernScores) existent toujours → pas de re-download pour elles.

3. **Reconvertir tout le corpus** avec la pipeline multi-staff → nouveau `kern_all/` (PDMX + ASAP + Chopin + KernScores).

4. **Filtre structurel sur le kern reconverti** (point 2 ci-dessous) — `diagnose_density.py` refondu : silence positionnel (trim vs rejet), `dead_spine` N-portées, `hyper_dense` relâché. Produit `blacklist.txt` + `trim_list.tsv`.

5. **Blacklist métadonnées** (point 4) — croiser `meta_flags.csv` (de l'étape 0) : rejeter `is_original=0` + genre non-classique, `scale_consistency` bas.

6. **Construire les tiers d'échantillonnage** pour le DataLoader v3 : Tier A classique ajouté au fine-tuning ; Tier A complet sur-échantillonné fin de pre-training ; Tier B = base.

---

## v3 — à intégrer


- [~] **▶ Chantier multi-staff / multi-voix / Sustain** — plan : `music_cwt/v3/docs/plan_multivoix_sustain.md`. **Phases 1 (head H17 voix) et 2 (Sustain) FAITES (2026-06).**
  Fait : (b) multi-VOIX intra-stave = head **H17 (voix V1-3)** + reconstruction par colonnes `*staffN` (inflation de lignes corrigée) ; (c) **Sustain** = ancre Staff_1/V1, −23 % tokens, 140/140 idempotent ; + **bug réparé** : `cwt_model.py` n'importait pas (`H11_VOCAB.index("LH")` mort).
  **Fait aussi (2026-06)** : (d) `generate.py _mxl_to_kern` **dé-squeezé** → émet N portées (parties → Staff_1..4, capé à 4). Testé : Liszt Jeux d'eaux 3-portées → kern 3-portées → round-trip 0.00 % de diff, idempotent. + bug réparé : `_barline_seed` référençait `"RH"` (mort). (e) Round-trip kern→token→kern validé sur 400 fichiers (médiane 0.00 % hauteurs, 400/400 idempotent) ; **3 bugs corrigés** : `**fing`/`**text` lus comme notes (filtre type sur `staff_cols`), notes `c` inventées (parse `\`-préfixe + filet `_extract_pitch_str`), fallback no-`*staff` (lit toutes les colonnes `**kern`, 4-portées 18.6 %→0.26 %).
  **Fait aussi (2026-06, perfectionnement lossless du convertisseur EXHAUSTIF) :** (f) `pdmx_scrapper/mxl_to_kern.py` **dé-squeezé** → `_score_to_kern_lines` émet **une spine `**kern` par part music21** (N portées) + `**dynam`. (g) **Voix intra-portée préservées** via splits `*^` (`_voice_split_seq`/`_voice_merge_seq`/`_measure_voice_streams`), **cap à 2 voix** (`_part_voice_count`) car music21 ré-importe mal les splits 3-4 profonds (mélange portées + perte de notes) → voix excédentaires fusionnées en accords sans perte de hauteur. (h) `kern_to_mxl.py` restaure l'ordre des N portées en relisant **l'index de spine stocké dans `part.id`** (`spine_0`=portée 1) — music21 inverse `.parts` mais préserve l'ordre vrai dans l'id ; repli registre si non parseable. Le tri par registre seul se trompait (Liszt : staff2 registre moyen > staff1). **Résultat : MXL→kern→MXL ~100 % pitch** sur op.2, Liszt 3-portées, 14 fugues Bach (avant : Bach perdait 20-33 % des notes via squeeze-en-accords). Delta de nb de notes restant = liaisons/silences (frontière music21), pas de hauteurs.
  **Fait aussi (2026-06, bugs slurs `kern_to_mxl.py`) :** 3 bugs corrigés lors du round-trip MXL→kern→tok→kern→MXL sur op9no2 : (i) **offset drift des rests** dans `_rebuild_slurs`/`_rebuild_ornaments`/`_rebuild_beams` (rest skippé sans avancer `col_offset` → tous les lookups suivants décalés) ; (ii) **`set` → `Counter`** dans `_collect_slurs` de `mxl_to_kern.py` (deux slurs convergeant sur la même note perdaient 1 `)`) ; (iii) **ordre end-avant-start** pour token `Slur_EndStart` dans `_rebuild_slurs` (push-avant-pop créait une auto-boucle et perdait 2 slurs par event). Résultat : 95→98 slurs reconstruits (−10→−7). Idempotence kern→tok→kern confirmée 200/200 après fixes.
  **Reste** : (a) reconvertir PDMX/ASAP avec le convertisseur dé-squeezé (cf. § reconversion) ; (b) entraînement v3 ; (c) rebuilds slurs/dynamics/tempo de `kern_to_mxl.py` encore limités à 2 portées (staff 3+ : notes OK, mais slurs/nuances/tempo non rebâtis) ; (d) extraire staccatissimo + turn inversé. **Résidu slurs −7 (non bloquant) :**
  - **−3 : H7 mono-valeur** — double-close/double-start sur même note (kern `))` / `((`) : H7 ne peut en porter qu'un. Fix = `Slur_End2`/`Slur_Start2` dans H7 vocab (+2 tokens) ; décision à prendre avant pré-training v3.
  - **−1 : LIFO orphan** — slur croisé non-imbriqué (F6→E-6, m30) : reste sur la pile car le double-close ne génère qu'une fermeture. Fix = algo non-LIFO. Effort ~1j.
  - **−3 : divers** — grace-note-to-main-note slurs + décalage comptage par mesure (anacrouse) ; probablement ≤1 perte réelle.

- [ ] **A/B accords atomiques : entraîner deux v3 (atomisée vs non-atomisée)** — expérience post-pipeline
  Splitter la v3 en deux variantes pour mesurer empiriquement l'apport des accords atomiques : (1) **atomisée** = chemin actuel (`Chord_Atomic` + H15/H16 quand l'atomisation est lossless, séquentiel sinon) ; (2) **non-atomisée** = flag qui force TOUS les accords en séquentiel (`Chord_Start/Cont/End`). Le flag est cheap à ajouter (un bool dans le tokeniseur, ~rien à coder). Coût réel = le 2ᵉ entraînement (~70h).
  ⚠️ **Métrique** : les deux tokenisations ont des longueurs de séquence différentes → la **loss brute n'est pas comparable**. Comparer via éval downstream : qualité de génération (cohérence harmonique), et/ou perplexité normalisée par contenu musical (par note/mesure, pas par token). À cadrer avant de lancer.
  État accords atomiques (2026-06) : reconstruction 100% kern (sans MIDI), garde lossless (0 partition altérée, vérifié kern→token→kern sur 24.6k accords), 91% des accords reconnus atomisés. 3 défauts de reconstruction trouvés+corrigés via le round-trip (dbX doublait la mauvaise note ; open montait la mauvaise note).

- [x] **Token `Sustain`** (remplacer `SEP`) — ✅ IMPLÉMENTÉ (2026-06, Phase 2)
  Ancre = **Staff_1 / voix V1** : toujours émise par ligne (événement si active, sinon token `Sustain`, tous heads NULL sauf H11/H17) ; elle délimite la ligne (plus de `SEP` émis). Les autres (portée, voix) skippent leurs `.`. Modèle : h17 passé en STAGE1, `Sustain` ajouté au masque NULL (comme Barline). **Gain mesuré : −23 % de tokens** (001-BRZ 9173→7063, 002-Hat 18038→13898). Round-trip 140/140 idempotent (Chopin+KernScores), hauteurs préservées à 0.17 %. `SEP` reste dans le vocab pour un futur mécanisme harmonisation (`SEP_HARM`).

- [ ] **Filtrage corpus par diag densité/pauses** — priorité haute
  Constat 2026-06 : génération v2 produit (a) mesures sur-condensées et (b) longues pauses bizarres. Scan corpus complet via `music_cwt/data_analysis/kern_distribution.py` (219 300 fichiers analysés, métriques par fichier dans `kern_distribution.csv`).
  Quantiles serveur (p50/p75/p90/p95/p98/p99/max) — seuils retenus après revue des plots :
  - `density_skew`           : 1.43 / 1.65 / 1.95 / 2.23 / —    / **3.93** / 267  → seuil **p99 = 4**
  - `max_notes_per_measure`  : 7    / 9    / 11   / 13   / —    / **24**   / 533  → seuil **p99 = 24**
  - `max_rest_streak`        : 0    / 1    / 2    / 3    / **7** / 15      / 1841 → seuil **p98 = 7** (meilleur elbow)
  - `pct_rests`              : 0    / 0.016 / 0.057 / 0.099 / **0.17** / 0.25 / 1 → seuil **p98 = 0.17** (cible pauses inopinées)
  → p99 vs max explose × 20-120 → la queue contient des fichiers catastrophiques (rest_streak=1841 lignes !).
  Action : relancer `diagnose_density.py` avec les seuils ajustés (`--max-npm 24 --max-skew 4 --max-rest 7 --max-pct-rests 0.17 --min-measures 4`), inspecter ~10 fichiers du `bad_files.csv` pour valider, puis exclure du DataLoader v3 via filtre nom-de-fichier (la liste va dans `music_cwt/v3/`).
  Scripts : `music_cwt/data_analysis/kern_distribution.py` (distribution), `diagnose_density.py` (filtre avec seuils), `plot_distributions.py` (graphes). Rapport visuel : `music_cwt/data_analysis/results/REPORT.md`.

- [ ] **Axe qualité par métadonnées + tiers d'échantillonnage** — priorité haute (itération 2026-06)
  Constat : le filtre structurel seul a des faux positifs (`hyper_dense` rejette Bach BWV 564 / Beethoven Op.78) et ne peut pas juger le "moche mais pas buggé" (Donkey Kong Country, npm=28). Solution : croiser avec les métadonnées MuseScore de `PDMX.csv`.
  Chiffres établis : sur 254k PDMX, seuls **5.6% ont un rating** (valeur quasi toujours 4-5★ → c'est la PRÉSENCE de votes `n_ratings≥3` qui discrimine, pas la valeur). PDMX fournit `subset:rated_deduplicated` = **13 181** fichiers (tier A "or"), dont **5 929** classique/original.
  **Fait :**
  - [x] `music_cwt/data_analysis/pdmx_metadata.py` — join `kern_all` ↔ `PDMX.csv` par hash IPFS (= stem du nom de fichier = stem colonne `mxl`). Sort `meta_flags.csv`, `tier_A.txt`, `tier_A_classical.txt`. Smoke-testé en local (load OK, hash connu résout).
  - [x] `music_cwt/data_analysis/validate_silence_proxy.py` — teste la proxy "silence tête/queue ↔ qualité" contre `is_rated`. Décompose le silence en `leading`/`trailing`/`internal` (mesures, indép. de la subdivision). Smoke-testé sur `bad_sample/` : sépare bien trimmable (lead/trail) vs concaténation à jeter (internal). NB : `lead=1`/`trail=1` ubiquitaires (anacrouse/en-tête) → seuils `≥2`.
  **À lancer sur le serveur :**
  - [ ] Uploader `PDMX.csv` (215 MB) + les 2 scripts sur `poly-albatros:~/taff/`. Lancer `pdmx_metadata.py` puis `validate_silence_proxy.py` sur `~/taff/kern_all` → vrais comptes Tier A présents + verdict proxy silence.
  **DÉCISION 2026-06 (utilisateur) : multi-staff D'ABORD, puis tri structurel sur le kern reconverti.** Raison : le squeeze 3→2 de l'ancienne pipeline produit lui-même des `hyper_dense`/`density_skew` (faux positifs Bach/Liszt). Figer le blacklist structurel sur l'ancien kern jetterait des fichiers 3-portées qui se convertiraient proprement en v3. Le re-run du filtre structurel est cheap (~min de parsing) ; seuls la logique et les seuils transfèrent, pas les fichiers parsés.
  **Point 2 — refonte `diagnose_density.py`** (APRÈS reconversion multi-staff — couplé au chantier multi-staff car `dead_spine` doit gérer N portées) :
  - [ ] Remplacer `max_rest_streak` (lignes) par silence positionnel en mesures : `leading`/`trailing` → **trim-list** (récupère le fichier au lieu de le jeter) ; `max_internal_silent > 2-3` → rejet (fragments collés).
  - [ ] Remplacer `mostly_rests` (pct global) par test `dead_spine` **par colonne, N portées** : spine `**kern` < ~5% de notes = portée morte → rejet. Distingue "stave jamais jouée" (jeter) de "stave intermittente musicale" (Liszt Jeux d'eau, garder). À écrire contre le format kern v3 (H11 = `Staff_1..N`).
  - [ ] Relâcher `hyper_dense` à npm > ~50 (garde-fou ; le vrai bug est déjà pris par `density_skew > 4`).
  - [ ] Sortie DataLoader v3 : `blacklist.txt` + `trim_list.tsv`.
  **Point 4 — blacklist métadonnées** (orthogonale au structurel) :
  - [ ] Via `meta_flags.csv` : rejeter `is_original=0` + genre non-classique (pop/jeux vidéo) et/ou `scale_consistency` très bas. Attrape le "moche mais pas buggé".
  **Stratégie tiers v3 :**
  - [ ] **Tier A classique** (~6k présents, `tier_A_classical.txt`) → AJOUTER au fine-tuning (`is_not_pdmx` actuel ~6k → ~12k). Décision validée avec l'utilisateur (filtrer le style pour ne pas diluer le classique).
  - [ ] **Tier A complet** (13k) → sur-échantillonnage fin de pre-training. **Tier B** (reste passant les filtres) → pre-training de base.

- [ ] **Support multi-staff (3+ portées)** — priorité haute
  Constat v2 : `_mxl_to_kern()` et `pdmx_scrapper/mxl_to_kern.py` **squeezent** systématiquement les scores 3+ portées en 2 portées (RH/LH). Coût mesuré sur le corpus qualité :
  - Chopin first editions : 70 / 512 = **13.7%** à 3+ spines
  - KernScores : 1641 / 5219 = **31.4%** à 3+ spines
  - ASAP MXL : 7 / 235 = 3.0% (Liszt Trans 4, Ravel, Scriabin)
  - Total : ~1716 fichiers / 5931 = **~29% du corpus qualité partiellement déformé**

  **Solution v3** : étendre H11 (Spine) de `RH`/`LH` à `Staff_1..Staff_N` (N=4 ou 6 max). Le même modèle gère 2-stave (Chopin), 3-stave (Liszt), 4-voix (Bach choral). Pas d'overhead pour les fichiers 2-staves (séquence inchangée), juste un vocab H11 plus riche et ~1.3K params en plus (invisible sur 16M).

  **Changements requis** :
  1. Vocab H11 : `Staff_1, Staff_2, Staff_3, Staff_4` (+ NULL si besoin)
  2. Tokeniseur `multihead_tokenizer.py` : lire N spines, labelliser H11=`Staff_X`
  3. Postprocess : émettre N colonnes kern à la reconstruction
  4. `mxl_to_kern.py` (pdmx) + `_mxl_to_kern()` (generate.py) : **supprimer le squeeze**, garder les N portées
  5. Meta prefix : ajouter `N_STAVES_2/3/4/5/6` pour conditionner le modèle (évite qu'il génère du `Staff_3` quand on lui demande du 2-staves)
  6. Loss masks : généraliser la règle `H12/H13/H14 attached to RH` → `attached to Staff_1` (ou plus subtil : à la portée la plus haute active)
  7. Reconvertir tout le corpus (PDMX + ASAP + Chopin + KernScores) depuis les sources MXL/kern originales avec le nouveau pipeline
  Effort estimé : ~3-5 jours d'ingénierie. Aucune nouveauté algorithmique, juste de la cohérence.

- [ ] **Filtre monophonique sur KernScores** (à faire au moment du multi-staff)
  KernScores contient **51% de fichiers 1-spine** (2669/5219) : chants traditionnels (`deut*.krn`), hymnes, melody-only. Ces fichiers conditionnent le modèle à de la monophonie qui n'a pas de sens pour du piano 2-mains. Les filtrer du corpus de pretrain pour éviter de biaiser le modèle vers du contenu mono. (Ou garder mais avec un tag meta `MONOPHONIC` pour conditionnement explicite.)

- [ ] **Vrai set test OOS** (à isoler dès le début de v3)
  Constat v2 : la val loss du finetune (5% du corpus qualité) **n'est PAS indépendante du pretrain** — ces fichiers ont été vus pendant la Phase 1. La val loss du finetune est donc optimiste, et on n'a aucune mesure fiable de la qualité OOS du modèle.
  Pour v3 : réserver un sous-ensemble (~300 fichiers, seed fixe) tenu à l'écart de **train ET de val** dans les deux phases. À ranger dans `kern_all_test/` séparé. Sert uniquement à : (a) mesurer la vraie loss OOS, (b) faire les tests qualitatifs (générations sur prompts jamais vus).
  **Difficulté : overlap PDMX.** PDMX contient ~213k fichiers auto-convertis avec composer/title inconnus pour 64% (NA). Il peut contenir des doublons ou des transcriptions alternatives de pièces présentes dans Chopin first editions / ASAP / KernScores → même si on isole 300 fichiers de qualité comme test, le modèle a peut-être vu la même musique via PDMX.
  Plan de mitigation (à raffiner) :
  1. Choisir le test set (300 fichiers de qualité)
  2. Extraire composer + title de chacun
  3. Croiser avec `pdmx_scrapper/rename_mapping.tsv` (déjà généré) pour identifier les PDMX matchant ces titres
  4. Exclure ces PDMX du pretrain
  5. Accepter qu'on ne dédoublonne pas parfaitement (titres ambigus, transcriptions non-attribuées) — un test "majoritairement OOS" est déjà beaucoup mieux que rien

  **Pièce à inclure obligatoirement dans le test set : Chopin Op. 2 n°1.** À identifier précisément dans Chopin first editions / KernScores / ASAP, puis exclure systématiquement de tous les corpus train+val (et chercher activement ses doublons potentiels dans PDMX via le mapping titre).

---

## Projets en standby (à reprendre en parallèle de v3)

- [ ] **Accords atomiques en tokenisation** — voir `music_cwt/projet_a_pt_etre_reprendre/accords_atomiques.md`
  Idée : compresser les accords harmoniques fréquents en 1 compound token (H1=Chord_Atomic, H_chord_quality, H_chord_voicing) au lieu de N tokens via Chord_Start/Cont/End. Garantie zéro perte d'info via fallback séquentiel.
  Étude empirique faite (Chopin + KernScores) : gain de compression réel ~11-13% global, ~35-47% sur accords 3+ notes. Listes recommandées : 8 qualités (Maj/min/dim/aug/dom7/dim7/half_dim7/sus4) et 10 voicings.
  **À tester en parallèle de la v3** pour comparer empiriquement la cohérence harmonique générée. Critère de reprise : si v3 montre faiblesse harmonique ou saturation de contexte, implémenter cette compression.

---

## Critique — Prochaine étape

- [ ] **Cadrage article / representation score-level** — utiliser music_cwt/article/idees.txt comme base. Angle : representation score-complete compacte (`**kern` compound) + petit Transformer ~16M ; baseline a reprendre : modele texte pur kern/ABC ~16M pour verifier que la representation structuree ameliore syntaxe valide, parsabilite et coherence a budget comparable.

- [ ] **Review complète du papier "Nested Music Transformer"** (Yoo, Dong, Jung, Jeong — ISMIR 2024, arXiv:2408.01180)
  Lire l'article en entier (pas que l'abstract/HTML) et identifier précisément ce qu'on apporte de nouveau par rapport à eux.
  Points à creuser :
  - Sub-decoder cross-attention autoregressif sur les sub-tokens — à reprendre pour la v3 ?
  - Expérience d'ordre des heads (Metric-First vs Pitch-First) — tester sur nos 14 heads
  - Embedding Enricher — équivalent de notre "concat + projection" déjà implémenté en v2
  - Loss moving-window NLL — comparer à notre loss masquée
  Notre apport probable : heads de notation expressive (ornements, articulations, slurs, phrases, ties, beams, pédale, tempo) absentes du NMT, format kern natif (pas de quantification Beat artificielle), spine RH/LH spécifique piano.
  Sortie attendue : note de positionnement « ce que NMT apporte vs ce qu'on apporte » + décision sur les éléments NMT à reprendre en v3.

- [ ] **Lancer fine-tuning v2** (Phase 2) — APRÈS la fin de la Phase 1 — sur corpus qualité (ASAP + Chopin + KernScores ≈ 6k, exclusion PDMX via filtre `is_not_pdmx`). 50 epochs depuis le dernier checkpoint pre-training (`ckpt_ep050.pt`). À lancer depuis `portugal` (ou autre machine libre) avec un nouveau wrapper / tmux.

- [ ] **Renommer les fichiers PDMX** avec composer + titre (après la phase 1 d'entraînement, pour ne pas invalider les paths cachés du DataLoader).
  Script prêt : `pdmx_scrapper/rename_pdmx_kern.py` + mapping `rename_mapping.tsv` (33 MB) déjà généré localement.
  Format cible : `Qm_<composer>_<title>_<hash>.krn` — le prefix `Qm_` est conservé pour que le filtre `is_not_pdmx` continue de marcher.
  ~213k renames sur le serveur en ~3-5 min. Permettra ensuite de grep par compositeur (Liszt, Chopin, Bach, etc.). Stats du mapping : ~70 fichiers Liszt PDMX (vs 17 ASAP), 163k fichiers `composer_name=NA` (64% du PDMX).

- [ ] **Diagnostic des données dégénérées** sur `pdmx_data/kern/` après reconversion.
  Script `pdmx_scrapper/diagnose_kern.py` déjà créé : génère stats CSV (% rests, mesures répétées, courtes boucles).
  À lancer : `python diagnose_kern.py --input pdmx_data/kern` puis filtrer manuellement avec seuils à définir.

---

## Haute priorité — Génération

- [ ] **Repetition penalty dans `generate.py`**
  Pénaliser les tokens H3 (pitch) récemment générés pour réduire les boucles.
  Paramètre `--repetition-penalty` (ex: 1.2). (~1h)

- [ ] **Fine-tuning par style** (`ST_chopin`, `ST_liszt`)
  Conditionner la génération sur le compositeur via le token de style dans MetaPrefix.
  Nécessite d'ajouter le champ style au préfixe meta dans le tokeniseur + encoder le compositeur depuis le nom de fichier PDMX (CSV disponible).

---

## Haute priorité — Architecture

- [ ] **GPT+BPE baseline**
  Entraîner un transformer standard avec tokenisation BPE sur le même corpus pour comparer avec le CWT.
  Donne une référence objective sur l'apport des 14 têtes composites.

- [ ] **Flash Attention**
  Remplacement drop-in, ~3x plus rapide sur GPU. Utile si on passe à un GPU cloud ou A100.

---

## Priorité haute — Tokenisation v3 (expressions musicales, à faire avant le pre-training v3)

- [x] **H18 — Caractère / Expression** — ✅ IMPLÉMENTÉ (2026-06)
  18e tête, 11 tokens : `Dolce`, `Cantabile`, `Espr`, `Leggiero`, `Marcato_P`, `Sostenuto`, `Tranquillo`, `Agitato`, `Sotto_Voce`, `Pesante`, `NULL`. Event-based (comme H12/H13/H14). Kern : `*dolce`, `*espr`, etc.
  Fusions : `appassionato/con passione/con anima` → `Espr` ; `scherzando/giocoso` → `Leggiero`.
  Collecte : `_collect_carac()` dans `mxl_to_kern.py` + `_CARAC_INTERP` dans tokenizer + `_H18_TO_KERN` dans postprocess.
  **Bug fix** : interp events dont l'offset dépasse le dernier note-offset d'une mesure sont clippés au dernier offset → consommés par la dernière data line, plus écrasés par le tandem de la mesure suivante.
  **Limitation one-slot** : si deux expressions carac sont consécutives sans note entre elles (ex. "dolce espr" en début de morceau = deux TextExpression), la seconde écrase la première. Acceptable.
  Round-trip : 200/200 idempotents. Dolce/Espr capturés sur fichier de test, idempotence kern2==kern3.

- [x] **`Calando` / `Poco_Calando` dans H14** — ✅ IMPLÉMENTÉ (2026-06)
  Tokens ajoutés à H14_VOCAB avant `NULL`. Kern : `*calando` / `*pococal` (pocalando aurait été produit par le pattern générique — pré-check spécifique ajouté dans `_tempo_text_to_interp`). Source words : `calando`, `perdendosi`, `smorzando` retirés de `_DYN_TEXT_MAP`, ajoutés à `_TEMPO_TEXT_MAP`.

- [x] **`Una_Corda` / `Tre_Corde` dans H13** — ✅ IMPLÉMENTÉ (2026-06)
  Tokens ajoutés à H13_VOCAB avant `NULL`. Kern : `*unacorda` / `*trecorde`. Collecte : textes `una corda`, `u.c`, `uc`, `tre corde`, `t.c`, `tc`, `tutte le corde` dans `_collect_pedal()` de `mxl_to_kern.py`. Postprocess : émission dans le loop H13.

---

## Priorité basse — Tokenisation (à évaluer après l'entraînement v2)

- [ ] **H10 (Beam)** : vérifier si le modèle l'apprend bien, sinon supprimer et reconstruire déterministiquement depuis la durée + signature rythmique.

- [ ] **H9 (Voicing `/` `\`)** : si le modèle ne l'apprend pas → fallback heuristique (note la plus haute = hampe haute, autres = hampe basse).

- [ ] **P/p (Appoggiature)** : retiré du tokeniseur v2 (cf. `music_cwt/v2/docs/tokenization_rules_reviewed.md` §Changes from v1, item 1) — les fioritures vivent maintenant en H1 (`Fioritura_Q`/`Fioritura_q`). Vérifier que le modèle les apprend correctement.

- [ ] **Beam search / nucleus sampling** dans `generate.py` en alternative au top-k pur.

---

## Pipeline de données — MIDI/MXL

- [ ] **Séparation main gauche / main droite pour MIDI mono-track**
  Options à évaluer et benchmarker :
  1. Seuil de pitch fixe (C4 comme pivot) — baseline simple
  2. Heuristique music21 native (solution actuelle)
  3. Clustering par continuité de voix (skyline algorithm)
  4. Classifieur ML supervisé
  Prévoir évaluation quantitative sur dataset annoté avant de figer la pipeline.
  Point de vigilance : les MIDI du pipeline audio-to-MIDI peuvent arriver en 1 ou 2 tracks selon le modèle.

- [ ] **midi2kern : support robuste des changements de tempo et d'armure** en cours de morceau.

- [ ] **midi2kern : extraction des pédales et ornements** depuis les événements MIDI CC.

- [ ] **midi2kern : tests round-trip** (MIDI → kern → score) pour valider la fidélité de la conversion.

---

## Évaluation

- [ ] **Métriques objectives de qualité musicale** : implémenter ou brancher des métriques standard (pitch class entropy, rhythmic consistency, chord coherence) pour comparer les checkpoints sans écoute manuelle.

---

## Fait

- [x] **3 bugs `kern_to_mxl.py` slurs corrigés (2026-06)** — (1) offset drift des rests dans `_rebuild_slurs/ornaments/beams` ; (2) `set`→`Counter` dans `_collect_slurs` de `mxl_to_kern.py` ; (3) ordre end-avant-start pour `Slur_EndStart`. Résultat : 95→98 slurs reconstruits sur op9no2, idempotence 200/200 confirmée.
- [x] **Bug spine swap dans `pdmx_scrapper/mxl_to_kern.py`** corrigé (détection BassClef, ligne 84). v2 du script.
- [x] **Format de clef `*clefG2` / `*clefF4`** corrigé dans `mxl_to_kern.py` (suppression de l'espace).
- [x] **Ornements / grace notes / arpèges extraits** par `mxl_to_kern.py` v2 (T, t, M, m, W, S, q, Q, :).
- [x] **Pré-training v1** terminé (ep50, loss ~0.084).
- [x] **Fine-tuning Humdrum v1** terminé (ft50).
- [x] **Postprocesseur** : beam/tie/slur/phrase markers interdits sur les rests — corrigé.
- [x] **Split v1 / v2** : `music_cwt/v1/` figé, `music_cwt/v2/` créé pour la prochaine itération.
- [x] **Conversion ASAP** : 235 MXL → 200 .krn piano (15% non-piano filtrés). Dans `kern_scrapping/asap_kern/`.
- [x] **Code v2 implémenté** : RoPE, Stage 3 (H4/H5|H2+H3), concat+projection, cleanup_checkpoints, filtre `is_not_pdmx`. Config 16.2M params.
- [x] **`music_cwt/v2/docs/training_strategy.md`** : plan complet de l'entraînement v2 (corpus, hyperparams, checkpoints, journal de bord).
- [x] **Setup serveur v2** : code uploadé sur `~/taff/v2/`, corpus unifié dans `~/taff/kern_all/` (4 sous-dossiers + symlink PDMX), smoke-test OK.
- [x] **Wrapper `auto_train.sh`** + cron `@reboot` configurés sur `portugal` pour survivre crashs/reboots.
- [x] **Mapping PDMX hash → composer/title** généré en local (`pdmx_scrapper/rename_mapping.tsv`, 33 MB, 254k entrées). Stats : ~70 Liszt PDMX, 163k composer=NA. Script `rename_pdmx_kern.py` (modes --build/--apply) prêt — application reportée à après la Phase 1.
