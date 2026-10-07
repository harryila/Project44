# Plan d'implémentation v3 — multi-staff / multi-voix / Sustain

> Plan opérationnel issu de la session 2026-06.
>
> **✅ ÉTAT (2026-06) : Phase 1 (head H17 voix) et Phase 2 (Sustain) FAITES et validées.**
> - Phase 1 : head H17 (V1-3), reconstruction par colonnes `*staffN`, inflation de lignes corrigée. + bug réparé : `cwt_model.py` n'importait pas (`H11_VOCAB.index("LH")` mort).
> - Phase 2 : ancre Staff_1/V1 (toujours émise, `Sustain` si tenue), retrait du délimiteur `SEP`, **−23 % tokens**, 140/140 idempotent.
> - **Reste : Phase 3** (dé-squeezer les 2 convertisseurs mxl→kern pour le multi-staff PHYSIQUE 3+ portées sur PDMX/ASAP ; round-trip `*^`/`*v` exact si on veut le split dynamique plutôt que les colonnes fixes), puis entraînement.

---

## 0. État actuel vérifié (point de départ)

### Multi-staff PHYSIQUE (3+ portées, ex. Liszt) — **côté ML déjà fait**
| Composant | État |
|---|---|
| Tokeniseur lit N portées (`H11_VOCAB=[Staff_1..4]`, `_SpineTracker.staff_cols(n)`, `staves_raw`, Meta `n_staves`+clefs/portée, H12/13/14 sur Staff_1) | ✅ fait |
| Modèle (`cwt_model.py`, H11 vocab=4) | ✅ fait |
| Postprocess (`_group_timesteps` → `per_stave`, `_build_header(meta, n_staves)`) | ✅ fait |
| `pdmx_scrapper/mxl_to_kern.py` (corpus) — squeeze `parts[0]/parts[1]` | ❌ à dé-squeezer |
| `generate.py _mxl_to_kern` (prompts) — `"kern 2-spines"` | ❌ à dé-squeezer |

→ Conséquence : le corpus **Humdrum** (déjà N-spines) gagne le multi-staff **sans reconversion**. Seuls PDMX/ASAP (squeezés à la conversion) restent à 2.

### Multi-VOIX intra-stave (polyphonie sur une même portée, ex. Chopin) — **PAS géré**
- Code actuel : un `*^` (split) → `staff_cols(n)` renvoie 2 colonnes même `Staff_n` → `_collect_tokens` → 2 raw tokens → `_process_spine_tokens` émet **2 `Note_Single` distincts, tous deux H11=Staff_n** (pas un accord).
- `_group_timesteps` casse sur "même stave réapparaît" (Note_Single ∈ `_KERN_EVENT_END`) → les 2 voix co-attaquées sont reconstruites sur **2 lignes séparées** → **simultanéité perdue**. Bug latent aujourd'hui.

### Sustain — **PAS implémenté**
- Le code utilise encore `SEP` comme délimiteur de fin de ligne (tokeniseur l.~1285, postprocess `_group_timesteps`/`_NON_DATA_H1`).

---

## 1. Décisions de design actées (session 2026-06)

### a) Découplage PORTÉE / VOIX sur deux axes
- **H11 = portée** (`Staff_1..4`) — inchangé.
- **NOUVEAU head H17 = voix dans la portée** (`V1`, `V2`, `V3`, `NULL`).
- Raison : la profondeur de voix est **quasi symétrique** entre portées (mesuré : MD 24 % / MG 14 % atteignent 3 voix). Un axe voix partagé est plus efficace en paramètres que des labels par sous-stave (`Staff_1a/b/c`, `Staff_2a/b/c`…) et garde H11 propre. (H9 Voicing `/\` est lié mais insuffisant : 2 valeurs, sémantique hampe ≠ index de voix.)

### b) Caps de profondeur (couvre ~99 % du corpus)
- Portées **1-2** : voix **1-3**.
- Portées **3-4** : voix **1-2**.
- Reste (Chopin 4-voix ≈ 0.4 %) → **dégradation locale** (fusionner la voix excédentaire dans la plus proche, ou drop des voix rest-only invisibles `yy`).

### c) Découpage de ligne ancré sur Staff_1 / voix 1 (remplace SEP)
- **Staff_1 voix 1 est TOUJOURS émis** à chaque ligne de données : son événement, OU un token **`Sustain`** s'il porte `.`.
- **Toutes les autres portées/voix** : on **skippe leurs `.`** (pas de Sustain), émis seulement si actifs.
- **Frontière de ligne = réapparition du token Staff_1/voix-1** (événement ou Sustain).
- `SEP` est **retiré** du rôle délimiteur. Réservé/renommé `SEP_HARM` si le mécanisme harmonisation mélodie→partition revient.
- **`Sustain` n'est émis QUE pour Staff_1 voix 1** (décision user). C'est l'ancre, jamais absente → découpage non ambigu, coût minimal (≤ SEP partout, et ne paie jamais le coût "1 Sustain par portée" du Sustain-partout).

### Justification chiffrée (mesures de la session)
- Multi-staff physique : Humdrum 13-31 % à 3+ portées ; PDMX/ASAP squeezés.
- Intra-stave **3 voix sonnantes** (notes réelles, rest-padding exclu) : Chopin **28 %** des fichiers (MD 24 %, MG 14 %), KernScores **1.7 %**. Norme = 2 voix.

---

## 2. Phases d'implémentation (ordre + fichiers)

### Phase 0 — Valider le multi-staff existant (½ j, AVANT tout code)
- [ ] Round-trip `tokenize → postprocess` sur 2-3 fichiers Humdrum 3-portées réels + 1 fichier 2-portées (régression).
- [ ] Confirmer que `Staff_3` ressort correctement et que `_build_header` émet 3 colonnes. Débusquer les bugs avant d'empiler le multi-voix.

### Phase 1 — Axe VOIX (H17) + round-trip `*^`/`*v` (le gros morceau)
**Tokeniseur (`v3/tokenizers/multihead_tokenizer.py`) :**
- [ ] Ajouter `H17_VOCAB = ["V1","V2","V3","NULL"]`, l'entrée dans `HEADS`/`ALL_HEAD_VOCABS`, le champ `CompoundToken.h17`, encode/decode.
- [ ] **Assignation de l'index de voix** : pour un stave n, `_SpineTracker.staff_cols(n)` renvoie ses sous-colonnes ; voix = **position (gauche→droite) dans ces colonnes** (col la plus à gauche = V1). Caper : portées 1-2 → V1-3, portées 3-4 → V1-2 (voix au-delà du cap = fusion/drop local).
- [ ] Émettre H17 sur chaque token note/accord/rest selon sa sous-colonne.
- [ ] **Détecter les transitions de profondeur** pour reconstruire les marqueurs : nb voix d'un stave ↑ entre 2 lignes → `*^` ; ↓ → `*v`. (Implicite via le head, ou token structurel — voir micro-décision §3.)

**Modèle (`v3/cwt_model.py`) :**
- [ ] Ajouter `h17` aux `HEADS` (→ 17 heads), embedding + output head, `VOCAB_SIZES`.
- [ ] **Stage** : mettre H17 en **Stage 1** avec H1/H11 (axe structurel, prédit tôt).
- [ ] **Masques de loss** : H17 = `NULL` forcé quand `H1 ∈ {Barline}` ; pour `Sustain` voir Phase 2. H17 ∈ {V1..V3} sur les events musicaux.

**Postprocess (`v3/postprocess.py`) :**
- [ ] `_group_timesteps` : regrouper par (portée, voix) ; reconstruire les sous-colonnes par voix.
- [ ] Émettre `*^` / `*v` aux bons endroits depuis les changements de nb de voix par portée.
- [ ] Round-trip test : un fichier Chopin 3-voix doit revenir identique (ou dégradation locale documentée).

### Phase 2 — Sustain / ancre Staff_1-voix1 (remplace SEP)
**Tokeniseur :**
- [ ] Ajouter `Sustain` à `H1_VOCAB`.
- [ ] Boucle data-line : émettre **Staff_1/V1 systématiquement** (event OU `Sustain` si `.`). Les autres portées/voix : skip les `.`.
- [ ] **Supprimer l'émission `SEP`** de fin de ligne (l.~1285-1290).
- [ ] Garder `make_sep()`/`SEP` uniquement pour la frontière harmonisation (ou renommer `SEP_HARM`).

**Modèle :**
- [ ] Ajouter `Sustain` à l'ensemble qui force H2-H17 = `NULL` (comme `Barline`). H11=Staff_1, H17=V1 sur le Sustain (ou NULL — fixer en §3).

**Postprocess :**
- [ ] Nouvelle frontière de ligne = apparition d'un token Staff_1/V1 (event ou Sustain) → retirer la dépendance à `SEP` dans `_group_timesteps`/`_NON_DATA_H1`.
- [ ] `Sustain` → `.` dans la colonne Staff_1.
- [ ] Reconstruire les `.` des portées/voix absentes d'une ligne.

### Phase 3 — Dé-squeezer les convertisseurs mxl→kern
- [ ] `generate.py _mxl_to_kern` : émettre N spines (prompts 3-portées Liszt). Cheap, zéro impact corpus.
- [ ] `pdmx_scrapper/mxl_to_kern.py` : idem — **seulement si** on décide de reconvertir PDMX (sinon Humdrum suffit pour valider le multi-staff).

### Phase 4 — Round-trip combiné + doc
- [ ] Tests round-trip sur un panel : 2-staves dense, 2-staves + 3-voix (Chopin), 3-portées (Humdrum/Liszt).
- [ ] Mettre à jour `v3/docs/tokenization_rules_reviewed.md` (H17, Sustain implémenté, multi-staff, retrait SEP-délimiteur) — la doc est actuellement périmée (dit "strictement 2-spines").

### Hors plan (séparé, plus tard)
- Re-download MXL PDMX + reconversion complète (uniquement si on veut le multi-staff PDMX + le tri structurel sur kern propre).
- Entraînement v3 (repart de zéro : H17 + Sustain changent le vocab → pas de chargement strict du checkpoint v2).

---

## 3. Micro-décisions à trancher en début d'implémentation
- [ ] **`*^`/`*v` implicite (déduit du head H17) vs explicite (tokens structurels dédiés)**. Implicite = plus léger mais suppose que le modèle réactive proprement les voix. À trancher Phase 1.
- [ ] **H11/H17 du token `Sustain`** : `Staff_1`/`V1` (cohérent avec l'ancre) vs `NULL`. Préférence : `Staff_1`/`V1` pour que la règle d'ancre soit uniforme.
- [ ] **Dégradation au-delà du cap** (4ᵉ voix, 0.4 %) : fusion dans la voix la plus proche vs drop des voix rest-only invisibles (`yy`). Mesurer la fréquence réelle si besoin.
- [ ] **Edge case** : ligne sans Staff_1 du tout (stave 1 terminée `*-`, rare en piano) → ancrer sur la voix 1 de la portée active la plus basse. Vérifier la fréquence.
- [ ] **Profondeur portée 3** : non mesurée ; supposée négligeable → cap V1-2. Confirmer si un fichier 3-portées pose problème en Phase 0.

---

## 4. Récap "ce qui reste" (réponse à la question)
- **Multi-staff physique** : il ne reste QUE dé-squeezer les 2 convertisseurs (Phase 3). Le ML est fait.
- **Multi-voix intra-stave** : tout à faire (Phase 1) — head H17 + assignation voix + round-trip `*^`/`*v`.
- **Sustain** : tout à faire (Phase 2) — ancre Staff_1/V1, retrait SEP-délimiteur.
