# Tokenization Rules — Reviewed (v3)

> Revised version based on design discussions. v2 spec in `../../v2/docs/tokenization_rules_reviewed.md`.
> Scope: full-score transformer on complete kern files (2 spines: RH + LH, plus optional dynam/pedal).

> **⚠️ v3 — non figé.** Deux changements de design actés :
> 1. Remplacement du token `SEP` (délimiteur de ligne) par un token `Sustain` explicite (le `.` kern).
>    **✅ STATUT : IMPLÉMENTÉ (2026-06).** L'ancre = **Staff_1 / voix V1** : toujours émise par ligne
>    (événement si active, sinon `Sustain`) ; elle délimite la ligne (plus de `SEP`). Les autres
>    (portée, voix) skippent leurs `.`. Gain mesuré : **−23 %** de tokens. Round-trip 140/140 idempotent.
> 2. **Accords atomiques** : compression optionnelle des accords harmoniques fréquents en un seul
>    compound token (H1=`Chord_Atomic` + heads H15/H16), avec fallback séquentiel garantissant
>    zéro perte d'info. **✅ STATUT : IMPLÉMENTÉ** (tokeniseur + modèle 16 heads + postprocess, round-trip OK).
> Voir la section *Changes from v2* ci-dessous. Le reste du fichier est hérité de v2 ;
> les autres pistes v3 (loss pondérée par fréquence, multi-staff…) ne sont PAS encore intégrées ici.

---

## Head Overview

| # | Name | Content | Vocab |
|---|---|---|---|
| H1 | **Type** | `Note_Single`, `Chord_Start`, `Chord_Cont`, `Chord_End`, `Chord_Atomic` (v3), `Fioritura_Q`, `Fioritura_q`, `Rest`, `Barline`, `Clef_G2`, `Clef_F4`, `Clef_G1`, `Sustain` (v3), `SEP` | 14 |
| H2 | **Duration** | Standard recip + dots + tuplets (no special grace marker — see note) | ~22 |
| H3 | **Pitch** | Fused pitch+octave+accidental + `r`, `rr` | ~282 |
| H4 | **Ornament** | Trills (`T`,`t`), trill+nachschlag (`TR`,`tR`), mordents (`M`,`m`,`W`,`w`), turns (`S`,`$`), arpeggio (`:`), trill-end marker (`Tr]`), generic (`O`,`I`) + `NULL` | 16 |
| H5 | **Articulation** | `'`, `` ` ``, `~`, `^`, `I`, `;` (fermata) + `NULL` | 7 |
| H6 | **Tie** | `Tie_Start`, `Tie_Continue`, `Tie_End`, `NULL` | 4 |
| H7 | **Slur** | `Slur_Start`, `Slur_End`, `Slur_EndStart`, `NULL` | 4 |
| H8 | **Phrase** | `Phrase_Start`, `Phrase_End`, `&Phrase_Start`, `&Phrase_End`, `NULL` | 5 |
| H9 | **Voicing** | `/` (stem up), `\` (stem down), `NULL` | 3 |
| H10 | **Beam** | `L`,`LL`,`LLL`,`J`,`JJ`,`JJJ`,`K`,`k`, `NULL` | 9 |
| H11 | **Spine / Portée** (v3) | `Staff_1`, `Staff_2`, `Staff_3`, `Staff_4` | 4 |
| H12 | **Dynamic** | Levels (`pp`→`ffff`), accents (`fz`,`sfz`...), hairpins (`<`,`>`,`(`,`)`) + `NULL` | 18 |
| | | **EVENT-based** (2026-06) : H12 ne porte la nuance QU'à la ligne du marquage explicite (sinon `NULL`), plus de carry-over sticky. Préserve les re-statements (`mf mf`) et hairpins répétés (`< <`) ; le postprocess émet la nuance dès que H12≠NULL (plus de dédup sur changement). | |
| H13 | **Pedal** | `Ped_Down`, `Ped_Up`, `Ped_Change`, `NULL` | 4 |
| H14 | **Tempo** | `Rall`/`Poco_Rall`/`Molto_Rall`, `Accel`/`Poco_Accel`/`Molto_Accel`, `A_Tempo`, `Rubato`/`Poco_Rubato`/`Molto_Rubato`, `NULL` | 11 |
| H15 | **Chord Quality** (v3) | `Maj`, `min`, `dim`, `aug`, `dom7`, `dim7`, `half_dim7`, `sus4`, `NULL` — actif seulement si H1=`Chord_Atomic` | 9 |
| H16 | **Chord Voicing** (v3) | 10 layouts (inversion + spacing + doublures) + `NULL` — actif seulement si H1=`Chord_Atomic` | 11 |
| H17 | **Voix** (v3) | `V1`, `V2`, `V3` — voix dans la portée (polyphonie intra-stave). `NULL` sur barline/clef | 4 |
| — | **Meta (prefix)** | `K_[sig]_maj/min`, `M_[ts]`, `MM_[bpm]`, `CLEF_G`, `CLEF_F`, `N_STAVES_[2-4]`, `BRACKET_PIANO` | ~46 |

**Total : 17 heads** (v2 = 14 ; v3 ajoute H15 Chord Quality, H16 Chord Voicing, H17 Voix).

---

## Changes from v1 (token-level fixes after first round-trip review)

These changes address bugs discovered when round-tripping a Chopin Nocturne from MXL → kern → tokens → kern → MXL.

| # | Change | Reason |
|---|---|---|
| 1 | Add `Fioritura_q` to H1 (and keep `Fioritura_Q`); **drop** the `q` and `8P`/`8p`/…/`16p` markers from H2 | Both grace types live in H1; H2 just carries the visual notated duration (`8`, `16`, `32`…) for both. The compound `P`/`p` scheme was never implemented and is redundant with `Fioritura_Q` + standard duration. |
| 2 | Add concrete clef tokens to H1: `Clef_G2`, `Clef_F4`, `Clef_G1`. H11 picks the staff. All other heads = `NULL`. | Mid-piece clef changes were entirely lost. Putting the full clef literal in H1 (rather than splitting H1 family + H3 line number) keeps H3's semantics consistent ("always pitch") and avoids the model having to learn that H3 means something different conditional on H1. Vocab cost is minimal (~3 entries for piano repertoire). |
| 3 | Add `;` to H5 vocabulary | Fermatas were silently dropped by the tokenizer suffix loop. |
| 4 | Add `Tr]` to H4 (trill-end marker) | Distinguishes `4cT` (single trill) from extended `4cT` … `4cTr]` (trill-line group). |
| 5 | Add `H13` (Pedal) reading from `**pedal` spine | Pedaling was unrepresented (no head, no spine). New head with sticky semantics. |
| 6 | Add `BRACKET_PIANO` to Meta | System bracket (`{` joining two staves) lost in round-trip; encoded once in the prefix. |
| 7 | Add `H14` (Tempo) reading from tempo-text directions (`rall.`, `accel.`, `a tempo`, `rubato`…) | Local tempo modifications (rallentando, accelerando, return-to-tempo, rubato) were entirely lost — the model had only the global `MM_` prefix and `*MM` markers, nothing for *gradual / transient* tempo changes. New head with a small closed vocabulary; orthogonal to H12 (a `rall.` can occur at any dynamic). Strict synonyms are merged (`ritardando`/`ritenuto`/`allargando` → `Rall`, etc.) but the **intensity adverb** (`poco`/`molto`) is kept as a vocab variant (`Poco_Rall`, `Molto_Rall`…) — it carries real interpretive weight and costs only a few extra tokens. Dynamic-style text (`cres.`, `dim.`, `smorz.`) is **not** included here — it is redundant with the H12 hairpins. Pure-character text (`dolce`, `espr.`, `dolcissimo`) is dropped (interpretive, no effect on notes). |

---

## Changes from v2 (acté pour v3)

| # | Change | Reason |
|---|---|---|
| 1 | **Remplacer `SEP` (délimiteur de ligne) par un token `Sustain`** dans H1, qui encode explicitement le `.` du kern (continuation nulle d'un spine). | En v2, le tokeniseur **skippe** les `.` et émet un `SEP` en fin de chaque ligne kern, uniquement pour que le post-processeur sache où finit une ligne. Problèmes : (a) `SEP` ne porte **aucune information musicale** — pur bruit structurel ; (b) il coûte **+1 token par ligne** — soit ~33 % de tokens en plus dans les passages denses (les deux mains jouant à chaque ligne) ; (c) il était à l'origine prévu pour la frontière mélodie→partition, un usage différent → sémantique confuse. **Solution v3 :** chaque ligne kern émet **exactement 1 groupe par spine** (RH puis LH), où un groupe = une note, OU un bloc d'accord `Chord_Start…Chord_End`, OU un token `Sustain` si ce spine porte `.`. La séquence devient auto-délimitée : `[groupe-RH][groupe-LH][groupe-RH][groupe-LH]…`. Le tag H11 (spine) + l'invariant « un groupe par spine par ligne » permettent au post-processeur de re-découper les lignes sans ambiguïté. **Gain :** −33 % de longueur de séquence en texture dense (→ plus de musique dans les 2048 tokens de contexte → meilleure cohérence long terme), et `Sustain` porte une vraie info (« cette main tient / est silencieuse ici ») que le modèle apprend explicitement. |

**Note sur `SEP`** : le token `SEP` est **retiré** du vocabulaire H1 en v3. S'il faut un jour ré-introduire le mécanisme d'harmonisation mélodie→partition, utiliser un token dédié et distinct (`SEP_HARM`) pour ne pas re-confondre les deux usages.

| # | Change | Reason |
|---|---|---|
| 2 | **Accords atomiques** : nouveau type H1 `Chord_Atomic` + deux nouveaux heads `H15` (Chord Quality) et `H16` (Chord Voicing). Un accord harmonique reconnu est encodé en **un seul compound token** au lieu de N tokens via `Chord_Start…Chord_End`. | En v2, un accord de N notes occupe N positions du contexte. L'atomisation des accords fréquents : (a) **raccourcit les séquences** (un accord 4 notes = 1 token) → plus de musique dans les 2048 tokens ; (b) donne un **inductive bias harmonique** — le modèle apprend « Do majeur » comme une unité, pas 3 hauteurs indépendantes. **Garantie zéro perte d'info :** un accord n'est atomisé que s'il matche le schéma (qualité + voicing connus, aucune décoration per-note) ; sinon il bascule sur le path séquentiel `Chord_Start/Cont/End` de v2, inchangé. Étude empirique (Chopin + KernScores) : ~35-47 % des accords 3+ notes atomisables, soit ~11-13 % de compression globale. Détail complet et étude : `../../projet_a_pt_etre_reprendre/accords_atomiques.md`. |

**Comptage de tokens — exemple (ligne kern avec événement RH ET LH) :**
- v2 (SEP) : `[RH] [LH] [SEP]` = 3 tokens
- v3 (`.`) : `[RH] [LH]` = 2 tokens

**Ligne avec seulement RH (LH tient sa note) :**
- v2 (SEP) : `[RH] [SEP]` = 2 tokens
- v3 (`.`) : `[RH] [Sustain]` = 2 tokens

→ v3 est toujours ≤ v2, strictement meilleur en texture dense.

**Lignes entièrement vides (`. .` sur les deux spines)** : aucun token émis (skippées), comme en v2.

---

**General rules:**
- When `H1 = Barline`: all heads H2–H14 are forced to `NULL`.
- When `H1 = Rest`: H3 = `r`/`rr`, H4–H8 = `NULL` (ornaments and articulations are forbidden on rests).
- When `H1 = Sustain` (v3): H2–H11, H15–H16 forcés `NULL`. **H11 = `Staff_1`, H17 = `V1`**. **H12/H13/H14 portent la valeur active au moment du Sustain** (même règle que pour tout token : la dynamique/pédale/tempo du moment musical). Reconstruit en `.` dans la colonne Staff_1.
- When `H1 = Chord_Atomic` (v3): H15 (qualité) et H16 (voicing) sont **actifs** (non-`NULL`) ; H3 porte la fondamentale ; H2 la durée ; **H4/H5 portent l'ornement/articulation communs à l'accord** (ou `NULL`, relaxation 2026-06 §4bis) ; H6/H7/H8 forcés `NULL` ; H9/H10/H11/H12/H13/H14 gardent leur sémantique habituelle.
- When `H1 ≠ Chord_Atomic`: H15 et H16 sont forcés `NULL`.
- **Invariant de ligne (v3, ANCRE)** : chaque ligne kern de données émet en **premier** le groupe **Staff_1 / voix V1** (événement si actif, sinon un token `Sustain`), puis les autres (portée, voix) **actives** dans l'ordre (leurs `.` sont skippés). L'ancre Staff_1/V1 — toujours présente exactement une fois par ligne — **délimite la ligne** au postprocess (nouvelle ligne = réapparition de Staff_1/V1). Cet invariant remplace le délimiteur `SEP` de v2 (qui n'est plus émis).

---

## H1 — Type

Manages the structure of musical events and chord linearization.

| Token | Meaning |
|---|---|
| `Note_Single` | Standalone note |
| `Chord_Start` | First note of a chord |
| `Chord_Cont` | Middle note of a chord |
| `Chord_End` | Last note of a chord — signals end of chord block, acts as hand-switch cue |
| `Chord_Atomic` | **(v3)** Accord harmonique entier encodé en UN seul compound token. H15 donne la qualité, H16 le voicing, H3 la fondamentale, H2 la durée commune. H4/H5 = ornement/articulation communs à l'accord (sinon `NULL`, §4bis) ; H6/H7/H8 forcés `NULL`. Émis seulement si l'accord matche le schéma (voir *règle de fallback* ci-dessous) ; sinon fallback `Chord_Start…Chord_End`. |
| `Fioritura_Q` | Appoggiatura (`Q` in kern, **unslashed**). H2 = visual duration (`8`, `16`, `32`…). Same metric offset as next note (durationless on the grid). |
| `Fioritura_q` | Acciaccatura (`q` in kern, **slashed**). H2 = visual duration (`8`, `16`, `32`…). Same metric offset as next note (durationless on the grid). |
| `Rest` | Silence (`r` or `rr`) |
| `Sustain` | **(v3)** Ancre de ligne : la voix 1 de la portée 1 tient son `.` (note tenue / silence en cours). **H11 = `Staff_1`, H17 = `V1`**. H2–H10, H15–H16 = `NULL`. **H12/H13/H14 portent la valeur active** (dynamique/pédale/tempo du moment). Toujours émis quand Staff_1/V1 n'a pas d'événement mais que la ligne est non-vide → délimite la ligne (remplace `SEP`). Reconstruit en `.` dans la colonne Staff_1. |
| `Barline` | Measure barline |
| `Clef_G2` | Mid-piece clef change to **treble clef** (G on 2nd line). H11 picks the staff. All other heads = `NULL`. |
| `Clef_F4` | Mid-piece clef change to **bass clef** (F on 4th line). H11 picks the staff. All other heads = `NULL`. |
| `Clef_G1` | Mid-piece clef change to **French violin clef** (G on 1st line). Rare; appears in some 19th-century editions (Wessel, Schlesinger). H11 picks the staff. All other heads = `NULL`. |

**Règle de fallback `Chord_Atomic` (v3) — garantit zéro perte d'info :**
Un accord est tokenisé en `Chord_Atomic` **si et seulement si** :
1. Sa qualité harmonique matche une des 8 entrées de H15 (`Maj`, `min`, `dim`, `aug`, `dom7`, `dim7`, `half_dim7`, `sus4`)
2. Son voicing matche une des 10 entrées de H16
3. **Aucun membre** ne porte de tie/slur/phrase (H6/H7/H8 tous `NULL`). **H4 (ornement) et H5 (articulation)** sont autorisés **uniquement s'ils sont communs à tous les membres** (relaxation 2026-06) ; valeurs différentes → fallback. La valeur commune est portée par le `Chord_Atomic` et ré-émise sur chaque membre.
4. Tous les membres partagent la même durée (H2), les mêmes hampes/beam (H9/H10)
5. **(garde lossless hauteurs, 2026-06)** la reconstruction depuis `(H15, H16, H3)` reproduit **exactement** les notes écrites (orthographe + octave). `_try_atomize_chord` appelle `_atomic_chord_pitches` et compare aux notes source.
6. **(garde lossless décoration, 2026-06)** le suffixe ré-émis pour (H4, H5) se re-parse exactement vers (H4, H5).

Si l'une de ces conditions échoue → fallback sur `Chord_Start/Cont/End` (path v2 inchangé). **Aucun accord atomisé ne change jamais la partition.** Vérifié kern→token→kern : **0 partition altérée** sur 24.6 k accords (hauteurs) ; **1959/1959 accords décorés + 24559/24559 nus** round-trip exact (relaxation H4/H5). Détail : [`atomisation_accords.md`](atomisation_accords.md) §4bis.

> Pourquoi la condition 5 est nécessaire : H15/H16 sont des catégories **grossières** (qualité + classe de voicing). Plusieurs accords écrits différemment (enharmonie, registre d'un voicing ouvert, note doublée d'un `dbX`, fondamentale d'un accord symétrique) peuvent matcher la même paire (H15, H16). Sans la garde, la reconstruction ré-épellerait ou re-placerait des notes → changement de partition. La garde renvoie ces cas vers le séquentiel (qui préserve l'écrit exact).

**Post-processing:**
- `Chord_Start → ... → Chord_End`: merge into space-separated kern token (`4c 4e 4g`)
- `Chord_Atomic`: expansion `(H15 qualité + H16 voicing + H3 fondamentale + H2 durée) → token kern d'accord` via `_atomic_chord_pitches` (voir sections H15/H16). **Reconstruction 100 % kern** (épellation diatonique par degré + octave par répétition de lettre) — **aucun détour par MIDI** (le MIDI introduisait une dérive d'octave et de mauvaises épellations enharmoniques, supprimé 2026-06). La fonction est **partagée** entre tokeniseur (garde lossless) et postprocess, garantissant détection et reconstruction cohérentes par construction.
- `Fioritura_Q` / `Fioritura_q` block: extract and place **before** the principal note in kern; emit `8Q` for `Fioritura_Q` (uppercase, unslashed) and `8q` for `Fioritura_q` (lowercase, slashed). The `8`/`16`/… comes from H2.
- Rests cannot appear inside a Chord block
- `Clef_G2` / `Clef_F4` / `Clef_G1`: emit a tandem `*clefG2\t*` (or `*\t*clefF4`, etc.) line at the corresponding position. The clef string `G2`/`F4`/`G1` is taken from the H1 token name itself; H11 dictates the column.

---

## H2 — Duration (Recip System)

| Category | Tokens |
|---|---|
| Standard | `0` (breve), `1`, `2`, `4`, `8`, `16`, `32`, `64` |
| Dotted | `2.`, `4.`, `8.`, `16.`, `4..`, `8..` |
| Tuplets | `3` (half triplet), `6` (quarter triplet), `12` (8th triplet), `24` (16th triplet) |

**Grace notes (acciaccatura `q` / appoggiature `Q`):** no special H2 marker. Both use the standard duration tokens above (`8`, `16`, `32`…) representing the **visual** notated duration shown in the score. The grace-vs-main and slashed-vs-unslashed semantics are entirely encoded by H1 (`Fioritura_q`, `Fioritura_Q`, or any of the regular `Note_*` types).

---

## H3 — Pitch (Fused)

Each token is an atomic entity fusing pitch + octave + accidental.

**Kern convention:**
- Lowercase = C4 and above: `c`(C4), `cc`(C5), `ccc`(C6), `cccc`(C7)
- Uppercase = below C4: `C`(C3), `CC`(C2), `CCC`(C1), `CCCC`(C0)
- Octave boundary is between B and C

**Accidentals included:** `#`, `##`, `-`, `--` (all variants needed for Chopin/Liszt repertoire)

**Natural sign (`n`): post-processing only**
- Per-measure, per-voice accidental state tracker
- If the model predicts a natural pitch while a sharp/flat is active for that note → append `n`
- Automatic reset at every barline

**Special tokens:** `r` (rest), `rr` (whole-measure rest)

**Vocabulary size: ~282 tokens** (56 base pitches × 5 accidental states + `r` + `rr`)

**Usage étendu en v3 — quand `H1 = Chord_Atomic` :** H3 code la **fondamentale absolue** de l'accord (ex. `c` pour un accord dont la fondamentale est C4). La basse réellement jouée se déduit de (fondamentale + voicing H16) — elle peut différer de la fondamentale en cas d'inversion. Le vocabulaire H3 reste inchangé (~282 tokens).

---

## H4 — Ornament

| Token | Kern meaning |
|---|---|
| `T` | Trill (half-step above) |
| `t` | Trill (whole-step above) |
| `TR` | Trill + nachschlag |
| `tR` | Trill + nachschlag (whole-step) |
| `Tr]` | Trill-line END marker — closes a trill that was opened earlier (extends the wavy line from a previous `T`/`t` to this note's position). |
| `M` | Upper mordent |
| `m` | Lower mordent |
| `W` | Inverted mordent |
| `w` | Short lower mordent |
| `S` | Turn (standard, gruppetto) |
| `$` | Inverted turn |
| `:` | Arpeggio (applied to each chord member) |
| `O` | Generic ornament |
| `I` | Generic articulation |
| `NULL` | No ornament |

**Rules:** `NULL` is mandatory when `H1 = Rest` or any `Fioritura_*`.
In a chord, each member can independently carry its own ornament.
**Trill extension lines:** A note marked `T` opens a trill; the trill's end position is signaled by a later note carrying `Tr]`. If no `Tr]` follows, the trill is treated as ornamenting only the originating note.

---

## H5 — Articulation

| Token | Kern meaning |
|---|---|
| `'` | Staccato |
| `` ` `` | Staccatissimo |
| `~` | Tenuto |
| `^` | Accent (sfz, >, <) |
| `;` | Fermata (point d'orgue) |
| `I` | Generic articulation |
| `NULL` | No articulation |

**Rule:** `NULL` is mandatory when `H1 = Rest`. Fermatas on rests are encoded via a separate `Rest`-typed token whose H5 still carries `;` (exception to the rest-NULL rule).

---

## H6 — Tie

| Token | Kern meaning |
|---|---|
| `Tie_Start` | `[` — start of harmonic tie |
| `Tie_Continue` | `_` — continuation |
| `Tie_End` | `]` — end of tie |
| `NULL` | No tie |

---

## H7 — Slur

| Token | Kern meaning |
|---|---|
| `Slur_Start` | `(` — start of legato |
| `Slur_End` | `)` — end of legato |
| `NULL` | No slur |

**Why separate from H6 and H8:** a slur and a phrase can end on the same note (very common in Chopin). H6, H7, and H8 are fully independent and can all be active simultaneously on the same note.

---

## H8 — Phrase

| Token | Kern meaning |
|---|---|
| `Phrase_Start` | `{` |
| `Phrase_End` | `}` |
| `&Phrase_Start` | `&{` — elided start (new phrase before previous one ends) |
| `&Phrase_End` | `&}` — elided end |
| `NULL` | |

---

## H9 — Voicing (Stems)

| Token | Kern meaning |
|---|---|
| `/` | Stem up — upper voice |
| `\` | Stem down — lower inner voice |
| `NULL` | Voicing unspecified |

**Important:** voicing is **not deducible** from pitch alone. It encodes voice ownership in intra-spine polyphonic textures (e.g. RH melody + inner accompaniment in Chopin).

**Post-processing fallback:** if the model does not learn it reliably → heuristic: highest note = stem up, others = stem down.

---

## H10 — Beam

| Token | Kern meaning |
|---|---|
| `L` / `LL` / `LLL` | Beam start (1, 2, 3 beams) |
| `J` / `JJ` / `JJJ` | Beam end |
| `K` | Partial beam (right-extending) |
| `k` | Partial beam (left-extending) |
| `NULL` | |

**Note:** beaming is **fully reconstructible** from (duration + time signature + position in measure). It can therefore be **removed from the model** and recomputed deterministically at post-processing with zero musical information loss.

**Decision:** include H10 in v1 so the model generates syntactically complete kern. Drop it and reconstruct deterministically if it harms convergence.

---

## H11 — Spine / Portée (v3, multi-staff)

Une partition piano kern a N colonnes (portées). On linéarise en une seule séquence ; H11
indique la **portée** de chaque token. **v3 supporte jusqu'à 4 portées** (au-delà = capé).

| Token | Meaning |
|---|---|
| `Staff_1` | portée 1 (souvent main droite / aiguë au piano) |
| `Staff_2` | portée 2 (souvent main gauche / grave) |
| `Staff_3`, `Staff_4` | portées 3-4 (Liszt grand système, Bach 4 voix, orgue…) |

Les portées sont lues depuis les marqueurs `*staffN` du kern (sinon fallback : col0→Staff_1,
col1→Staff_2). H11 est en **STAGE1** (identité structurelle, prédite tôt). Le tokeniseur,
le modèle et le postprocess **gèrent N portées** ; la reconstruction émet une colonne
`**kern` par (portée, voix) avec son `*staffN`.

**Conditionnement :** le Meta préfixe inclut `N_STAVES_[2-4]` (nombre de portées attendu) +
une clef par portée → évite que le modèle génère du `Staff_3` en mode 2-portées.

**Polyphonie intra-stave** : gérée par l'axe orthogonal **H17 (voix)**, voir sa section.

**Ordre d'émission par ligne** : ancre Staff_1/V1 d'abord (cf. *Invariant de ligne*), puis
les autres (portée, voix) actives en ordre croissant.

> **⚠️ Seul le SQUEEZE des convertisseurs MXL→kern reste à retirer.** `pdmx_scrapper/mxl_to_kern.py`
> (corpus PDMX/ASAP) et `_mxl_to_kern()` (`generate.py`, prompts) fusionnent encore les scores
> 3+ portées en 2. Le corpus **Humdrum** (KernScores, Chopin first ed.) a déjà ses N portées
> dans le kern → le multi-staff y fonctionne sans reconversion. Pour PDMX/ASAP : dé-squeezer
> les 2 convertisseurs (cf. `todo.md`, `plan_multivoix_sustain.md` Phase 3).

---

## H12 — Dynamic

Extracted from the `**dynam` spine, which is syntactically attached to `*staff1` (RH) in kern.

| Token | Meaning |
|---|---|
| `pppp`, `ppp`, `pp`, `p` | Piano |
| `mp`, `mf` | Mezzo |
| `f`, `ff`, `fff`, `ffff` | Forte |
| `fp`, `fz`, `sfz`, `sfp` | Special accents |
| `<` | Crescendo start (hairpin) |
| `>` | Decrescendo start |
| `(` | Hairpin continuation |
| `)` | Hairpin end |
| `NULL` | Aucune dynamique établie (début de pièce, ou pièce sans `**dynam`) |

**Attachment rule (v3) :** la dynamique est une propriété du **moment musical** — elle est portée par **tous les tokens** du timestep (toutes portées, toutes voix, Sustain inclus). La valeur est `current_dyn` : la dernière dynamique lue dans le flux kern, propagée de façon sticky. Le postprocess lit H12 **uniquement depuis l'ancre Staff_1/V1** pour reconstruire la colonne `**dynam`, en n'émettant que sur changement.

**Missing spine rule:** if the file has no `**dynam` spine, all tokens receive `NULL`.

**Sticky semantics (read AND write):**
- Reading: a non-NULL value persists on subsequent tokens until a new value or `NULL` resets it. The H12 of every note represents the dynamic *during* that note.
- Writing back to kern: only emit the value to the `**dynam` spine when it **changes** from the previous one. This avoids the explosion of `<` repeats observed in v1 round-trips.

---

## H13 — Pedal (sustain pedal markings)

Extracted from a dedicated `**pedal` spine, or from kern tandem markers `*ped`/`*Xped` if present.

| Token | Meaning |
|---|---|
| `Ped_Down` | Pedal pressed (start) — `*ped` in kern, `<sustain type="start">` in MXL. |
| `Ped_Up`   | Pedal released (end) — `*Xped` in kern, `<sustain type="stop">` in MXL. |
| `Ped_Change` | Pedal change (release+press at the same beat) — `<sustain type="change">` in MXL. |
| `Una_Corda` | Left pedal pressed — `una corda`, `u.c.` |
| `Tre_Corde` | Left pedal released — `tre corde`, `t.c.`, `tutte le corde` |
| `NULL` | No pedal event at this token (~98% of tokens). |

**Attachment (v3):** comme H12, les événements de pédale sont portés par **tous les tokens** du timestep. Le postprocess lit H13 uniquement depuis l'ancre Staff_1/V1.
**Sticky:** the pedal state persists between `Ped_Down` and `Ped_Up` even though intermediate tokens carry `NULL`. The model only emits the *event*, not the state.

---

## H14 — Tempo (local tempo-modification directions)

Extracted from textual tempo directions in the score (MXL `<words>` / music21 `TextExpression`), encoded in kern as tandem interpretations (`*rall`, `*pocorall`, `*moltorall`, `*accel`, …).

Each token is a **direction category** optionally prefixed by an **intensity** (`Poco_` / `Molto_`). `A_Tempo` takes no intensity (a return-to-tempo is binary).

| Token | Meaning | Source words (case-insensitive, normalized) |
|---|---|---|
| `Rall` / `Poco_Rall` / `Molto_Rall` | Slowing down | `rall.`, `rallentando`, `ritard.`, `ritardando`, `riten.`, `ritenuto`, `rit.`, `slargando`, `allargando` |
| `Accel` / `Poco_Accel` / `Molto_Accel` | Speeding up | `accel.`, `accelerando`, `stretto`, `string.`, `stringendo`, `affrettando` |
| `A_Tempo` | Return to the base tempo | `a tempo`, `tempo I`, `tempo primo`, `in tempo` |
| `Rubato` / `Poco_Rubato` / `Molto_Rubato` | Free / flexible timing | `rubato`, `senza tempo`, `ad lib.`, `ad libitum` |
| `Calando` / `Poco_Calando` | Slowing down **and** fading simultaneously | `calando`, `perdendosi`, `smorzando` (when used as tempo+dyn direction). **Double-nature token** : implique à la fois le ralentissement (≈ Rall) et la diminution sonore (≈ H12 `>`). Ni l'un ni l'autre seul ne suffit à l'encoder. `smorz.` peut aussi apparaître en H12 `>` si le contexte ne comporte pas de composante tempo. |
| `NULL` | No tempo direction at this token (~99.5% of tokens) | |

**Normalization:** strict synonyms collapse to one category (e.g. `ritardando` ≈ `ritenuto` ≈ `allargando` → `Rall`). The **intensity adverb** is detected and kept: `poco` / `un poco` → `Poco_`, `molto` / `assai` → `Molto_`; other adverbs (`sempre`…) are dropped. Words that are *dynamic-only* directions (`cres.`, `dim.`) are **not** mapped here (→ H12). Words that are purely *character / touch* indications (`dolce`, `espressivo`, `cantabile`…) are mapped to **H18** (not here).

**Attachment (v3):** comme H12 et H13, les événements de tempo sont portés par **tous les tokens** du timestep. Le postprocess lit H14 uniquement depuis l'ancre Staff_1/V1.

**Event semantics (not sticky-state):** H14 marks the *onset* of a tempo direction, exactly like H13 pedal events. The model emits the event token once; the "until when" is implicit (an `A_Tempo` later cancels a `Rall`). Intermediate tokens carry `NULL`.

**Kern representation:** a tandem interpretation line placed at the event's position, on the `RH` (staff1) column. The kern token is `*` + intensity-prefix + category: `*rall`, `*pocorall`, `*moltorall`, `*accel`, `*pocoaccel`, `*moltoaccel`, `*Atempo`, `*rubato`, `*pocorubato`, `*moltorubato`. The other spines carry `*`. Mid-measure directions are interleaved between data lines at the right offset (same mechanism as mid-piece clef changes).

**Known limitation:** if two tempo directions sit at the *exact same musical instant* — e.g. `rall.` at the end of a measure and `a tempo` at the downbeat of the next — only the second survives the tokenizer (a single token's H14 cannot hold two events, and there is no note between them to carry the first). This is rare and analogous to the `fz`+`p` dynamic-overlap limitation of the `**dynam` spine.

---

> **Mécanisme complet de l'atomisation** (détection, reconstruction kern, garde lossless,
> bilan empirique, où vit le code) : voir [`atomisation_accords.md`](atomisation_accords.md).

## H15 — Chord Quality (v3)

Actif **uniquement** quand `H1 = Chord_Atomic` ; sinon forcé `NULL`.
Encode la qualité harmonique de l'accord atomisé. 8 qualités + `NULL`.

| Token | Intervalles (demi-tons depuis la fondamentale) | Exemple sur C |
|---|---|---|
| `Maj` | 0, 4, 7 | C–E–G |
| `min` | 0, 3, 7 | C–E♭–G |
| `dim` | 0, 3, 6 | C–E♭–G♭ |
| `aug` | 0, 4, 8 | C–E–G♯ |
| `dom7` | 0, 4, 7, 10 | C–E–G–B♭ |
| `dim7` | 0, 3, 6, 9 | C–E♭–G♭–A |
| `half_dim7` | 0, 3, 6, 10 | C–E♭–G♭–B♭ |
| `sus4` | 0, 5, 7 | C–F–G |
| `NULL` | — | (H1 ≠ Chord_Atomic) |

**Choix de la liste** : étude empirique sur Chopin + KernScores (cf. `accords_atomiques.md`). Maj/min/dim couvrent ~85 % des accords matchés, dom7 ~5 %. Les qualités plus rares (`Maj7`, `min7`, `aug7`, `dom9`, `sus2`, `minMaj7` — chacune <1 %) ne sont **pas** atomisées : elles tombent en fallback séquentiel. Garder la liste courte évite de raréfier le path séquentiel.

---

## H16 — Chord Voicing (v3)

Actif **uniquement** quand `H1 = Chord_Atomic` ; sinon forcé `NULL`.
Encode en une seule catégorie l'arrangement physique de l'accord : inversion + spacing + doublures. 10 voicings + `NULL`.

| Token | Description |
|---|---|
| `close_root_nodb` | Position fondamentale, voicing serré (toutes les notes dans une octave), sans doublure |
| `close_inv1_nodb` | 1ʳᵉ inversion (3ce à la basse), serré, sans doublure |
| `close_inv2_nodb` | 2ⁿᵈᵉ inversion (5te à la basse), serré, sans doublure |
| `close_inv3_nodb` | 3ᵉ inversion (7e à la basse — accords de 7e uniquement), serré, sans doublure |
| `open_root_nodb` | Position fondamentale, voicing ouvert (≥1 note déplacée d'une octave), sans doublure |
| `open_root_dbR` | Position fondamentale, ouvert, fondamentale doublée |
| `open_inv1_nodb` | 1ʳᵉ inversion, ouvert, sans doublure |
| `open_inv1_dbX` | 1ʳᵉ inversion, ouvert, une note non-fondamentale doublée |
| `open_inv2_nodb` | 2ⁿᵈᵉ inversion, ouvert, sans doublure |
| `open_inv2_dbX` | 2ⁿᵈᵉ inversion, ouvert, une note non-fondamentale doublée |
| `NULL` | (H1 ≠ Chord_Atomic) |

**Choix de la liste** : ces 10 voicings couvrent **97.7 %** des accords matchés du corpus (couverture cumulative mesurée). Les voicings restants (`wide_*`, `*_dbMulti`, `*_other`, doublures exotiques) représentent <3 % et tombent en fallback séquentiel.

**Reconstruction (post-processing) — 100 % kern, sans MIDI** : pour chaque degré de la qualité (H15 → liste `(pas diatonique, intervalle demi-tons)`), on épelle la note (lettre + accidentel) relativement à la fondamentale H3, on l'ancre à l'octave de H3, puis on applique le voicing H16 :
> - **inversion** : la note de basse passe sous la fondamentale (qui reste à son octave H3) ; les autres s'empilent en montant.
> - **open** : on monte d'une octave la **plus basse note non-fondamentale** au-dessus de la basse (3ce en position fonda, 5te en inv1, 3ce en inv2…) — la fonda n'est **jamais** déplacée.
> - **doublure** : `dbR` double la fondamentale une octave au-dessus ; `dbX` double la **note de basse de l'inversion** (la plus souvent doublée en pratique).
>
> L'octave est gérée en **diatonique** (répétition de lettre), jamais en pitch-class — sinon un C♭ (pc 11, note basse) casserait le registre.

**Garde lossless** : ces choix canoniques sont des **points fixes de la détection** (100/100 voicings × racines testés). Mais un voicing étant une **catégorie** (plusieurs registres/doublures/épellations possibles), la reconstruction canonique ne coïncide pas toujours avec l'écriture source → la garde lossless (cf. règle de fallback, condition 5) renvoie alors l'accord vers le séquentiel.

> ⚠️ Les listes H15/H16 sont validées empiriquement. À affiner si la reconversion PDMX v3 montre une distribution différente.

---

## H17 — Voix (v3)

Encode la **voix dans la portée** — la polyphonie intra-stave (plusieurs lignes indépendantes
sur une même portée, ex. mélodie + voix interne à la main droite chez Chopin : **28 %** des
fichiers atteignent 3 voix sonnantes sur une portée).

| Token | Meaning |
|---|---|
| `V1` | voix 1 de la portée (la principale / la plus basse colonne) |
| `V2`, `V3` | voix internes supplémentaires |
| `NULL` | barline / clef (pas de voix) |

**Axe orthogonal à H11** : `(H11 portée, H17 voix)` identifie chaque colonne `**kern`. Choix
de découplage (vs labels `Staff_1a/b`) : la profondeur de voix est ~symétrique entre portées,
donc un axe voix **partagé** est plus efficace (le modèle apprend "voix interne" une fois).

**Caps** (couvrent ~99 % du corpus) : **V1-3 sur portées 1-2**, **V1-2 sur portées 3-4**.
Au-delà → fusion en accord (dégradation locale, ~0.4 %).

**STAGE1** : H17 est structural (comme H11) — prédit tôt, et **jamais forcé NULL sur un token
`Sustain`** (l'ancre Sustain porte H17=`V1`). Voir *Invariant de ligne*.

**Reconstruction** : chaque (portée, voix) rencontrée → une colonne `**kern` fixe avec `*staffN`
(convention Humdrum multi-spine-par-portée, relue à l'identique par le tokeniseur).

---

## H18 — Caractère / Expression (v3, non implémenté)

Direction de **caractère ou de toucher** s'appliquant à un passage entier. Orthogonale à H12 (nuance) : un même token peut porter `H12=p` ET `H18=Dolce`. Orthogonale à H14 (tempo) : un `Calando` en H14 peut coexister avec `H18=Tranquillo`. Extraite des `TextExpression` music21 ne relevant ni du tempo (H14) ni des nuances (H12) ni de la pédale (H13).

| Token | Meaning | Source words |
|---|---|---|
| `Dolce` | Doux, son rond et chaleureux | `dolce`, `dolcissimo`, `dolciss.`, `dol.` |
| `Cantabile` | Chantant, ligne liée et soutenue | `cantabile`, `cantando` |
| `Espr` | Expressif, liberté interprétative ; aussi : passionné | `espressivo`, `espress.`, `espr.`, `con espressione`, `appassionato`, `con passione`, `con anima` |
| `Leggiero` | Léger, toucher discret ; aussi : espiègle | `leggiero`, `leggierissimo`, `legg.`, `scherzando`, `giocoso` |
| `Marcato_P` | Passage marqué, chaque note appuyée (≠ H5 `^` qui est par note) | `marcato`, `marcatissimo` (comme TextExpression de passage) |
| `Sostenuto` | Tenu généralisé sur le passage, son généreux | `sostenuto`, `tenuto` (comme TextExpression de passage) |
| `Tranquillo` | Calme, son posé, sans agitation | `tranquillo`, `quieto` |
| `Agitato` | Nerveux, tendu, légèrement précipité | `agitato`, `precipitato` |
| `Sotto_Voce` | Murmuré, son intérieur | `sotto voce` |
| `Pesante` | Lourd, avec poids ; aussi : avec force/énergie | `pesante`, `con forza`, `energico` |
| `NULL` | Pas de direction de caractère (~97 % des tokens) | |

**Vocab size : 11** (10 tokens + NULL).

**Attachment :** comme H12/H13/H14, émis sur tous les tokens du timestep, lu uniquement depuis l'ancre Staff_1/V1 en postprocess.

**Kern representation :** tandem `*dolce`, `*cantabile`, `*espr`, `*leggiero`, `*marcato`, `*sostenuto`, `*tranquillo`, `*agitato`, `*sottovoce`, `*pesante` émis à l'onset de la direction ; `*Xcarac` à la fin si explicitement indiquée (rare — souvent la direction s'éteint à la prochaine direction de caractère ou à la fin de phrase).

**Pertes connues dans H18 :**
- `simile` — instruction de répétition ("continuer comme avant"), pas un caractère sonore. **Perdu** — ne rentre dans aucune head.
- `sempre` seul — modificateur sans continuation ("sempre p" devrait propager H12=p, pas une head séparée). **Perdu** en tant que token autonome.
- `poco a poco` seul — modificateur de progression. **Perdu.**
- Termes rares (< ~500 occ. corpus complet) : `grandioso`, `maestoso`, `brillante`, `imperioso`, `fuocoso`, `tempestuoso`, `trionfante`, `languido`, `teneramente`, `accarezzevole`, `recitativo`, `grazioso`, `con grazia`, `disperato`, `vivamente`, `capricciosamente`, `con luminosità`… **Perdus** — fréquence insuffisante pour l'apprentissage.
- Termes français/allemands (ASAP uniquement) : `expressif`, `cédez`, `retenez`, `en dehors`, `sehr lebhaft`, `zurückhaltend`… **Perdus** — hors vocabulaire, corpus trop marginal.
- `tenuto` / `staccato` comme direction de passage (≠ H5 note par note). **Perdus** — le postprocess ne peut pas reconstruire une direction de passage depuis H5.
- `rinforzando`, `rinf.` — accent dynamique, déjà absorbé en H12 comme sfz. La composante "renforcement graduel" est perdue.
- La **double nature exacte de `calando`** (ralentissement + diminuendo simultanés et couplés) est approximée par H14=`Calando` mais le modèle doit apprendre seul que cela implique aussi une diminution sonore — cette co-contrainte n'est pas explicite.

**État : spécifié, non implémenté.** Voir `todo.md`.

---

## Meta — Conditioning Tokens (Prefix, never predicted)

These tokens occupy the **first N positions** of every sequence. They condition the model but are **never predicted** (masked out of the loss).

| Category | Tokens | Examples |
|---|---|---|
| Key signature | `K_[notes]_maj` / `K_[notes]_min` | `K_f#c#_maj`, `K_b-e-a-_min`, `K_0_maj` |
| Time signature | `M_[num/den]` | `M_4/4`, `M_3/4`, `M_6/8`, `M_complex` |
| Tempo | `MM_[value]` | `MM_60`, `MM_120`, `MM_200` |
| Initial clef | `CLEF_G`, `CLEF_F` | (initial clef of each staff; mid-piece changes via `Clef_G2`/`Clef_F4`/`Clef_G1` H1 tokens) |
| Style | `ST_chopin`, `ST_liszt`, `ST_other` | |
| System bracket | `BRACKET_PIANO`, `BRACKET_NONE` | Tells the post-processor to emit `!!!system-decoration: {(s1,s2)}` so MXL renders the keyboard brace. |

**At inference:** the user provides these tokens to steer the model toward a specific key, meter, tempo, or style.

---

## Post-Processing Summary

| Operation | Trigger | Action |
|---|---|---|
| Natural sign insertion | Natural pitch + active accidental in state tracker | Append `n` to kern token |
| Accidental state reset | `Barline` token | Clear per-measure tracker |
| Chord merge | `Chord_Start → ... → Chord_End` | Merge into `4c 4e 4g` |
| Sustain → `.` (v3) | `H1 = Sustain` | Émet `.` dans la colonne Staff_1. C'est l'**ancre** : le découpage en lignes suit la réapparition de Staff_1/V1 (plus de délimiteur `SEP`) |
| Atomic chord expansion (v3) | `H1 = Chord_Atomic` | `(H15 qualité + H16 voicing + H3 fondamentale + H2 durée)` → token kern d'accord espacé `4c 4e 4g`. Table qualité→intervalles et constructeurs de voicing dans `postprocess.py` |
| Grace note split | `Fioritura_q` or `Fioritura_Q` in block | Move before principal note; emit `<dur>q` (slashed) or `<dur>Q` (unslashed) where `<dur>` comes from H2 |
| Beam reconstruction | (if H10 removed) | Recompute from duration + time signature |
| Voicing fallback | (if H9 unreliable) | Highest note → `/`, others → `\` |
| Measure renumbering | Always | Increment `=N` sequentially; **skip the first barline if the opening tokens form a pickup** (shorter than the time signature) |
| Clef change emission | `Clef_G2` / `Clef_F4` / `Clef_G1` | Emit `*clef{G2\|F4\|G1}` on the column dictated by H11 |
| Pedal emission | `H13 = Ped_Down` / `Ped_Up` / `Ped_Change` / `Una_Corda` / `Tre_Corde` | Emit `*ped` / `*Xped` / `*unacorda` / `*trecorde` on a `**pedal` spine, or as tandem if no spine |
| Tempo emission | `H14 = Rall` / `Accel` / `A_Tempo` / `Rubato` / `Calando` | Emit tandem `*rall` / `*accel` / `*Atempo` / `*rubato` / `*calando` on the RH column at the token's position |
| Caractère emission | `H18 ≠ NULL` *(non implémenté)* | Emit tandem `*dolce` / `*cantabile` / … on the RH column at the token's position |
| Dynamic deduplication | Sticky H12 values | Only emit to `**dynam` when value changes from the previous one |

---

## Deferred Decisions

- **H10 (Beam):** included in v1; remove and reconstruct deterministically if convergence is poor
- **Perte connue — groupement de tuplet (sextolet vs deux triolets) :** le recip kern `48` (1/12 QN) est identique pour une 32nd note en triolet (Tuplet 3:2) et en sextolet (Tuplet 6:4), car les deux ont le même ratio `actual/normal = 1.5`. La distinction — interprétativement importante (un seul geste continu vs deux sous-groupes avec micro-accent) — est perdue dès `mxl_to_kern.py`. Le pipeline kern→tokens→kern→MXL ne peut donc pas restituer le crochet "6" : music21 reconstruit deux "3" par défaut. **Solution envisagée : head H19 `TupletGroup`** (`NULL` / `TG6_start` / `TG6_mid` / `TG6_stop`, extensible à `TG5_*` / `TG7_*`), orthogonale à H10 (une note peut porter simultanément un beam marker ET un marqueur de groupe de tuplet). Non implémentée — voir `todo.md`.
- **H18 (Caractère / Expression) :** spécifiée (voir section ci-dessus), non implémentée. Étendre `mxl_to_kern.py` (`_collect_carac`), kern tandem `*dolce` etc., tokeniseur H18, postprocess. À faire avant le pre-training v3 si la notation exacte est requise — voir `todo.md`.
- **H9 (Voicing):** included in v1; heuristic fallback available
- **Style tokens:** optional; add to Meta prefix if composer-conditioned generation is desired
- **C clefs (alto, tenor, soprano):** not currently representable — H1 only has `Clef_G2`, `Clef_F4`, `Clef_G1`. Add `Clef_C1`…`Clef_C5` if needed (none in Chopin/Liszt piano repertoire).
- **(v3) H15/H16 — accords atomiques — bilan empirique (corpus local, 2026-06) :** le taux d'atomisation réel est **21.4 % des accords 3+ notes = 1.25 % de tous les évènements** → compression de séquence **~2 %** seulement. Le plafond est irréductible : accords décorés (ties/ornements/slurs) 32 %, **clusters non-tertiens 22 %**, hampes/beams mixtes 20 %. Ajouter des qualités ne sert quasiment à rien (**98 %** des accords hors-liste sont non-tertiens, pas Maj7/min7/9es) ; ajouter des voicings non plus (+1 %). Donc la valeur des accords atomiques **ne peut venir que du biais inductif harmonique**, pas de la compression. **Décision : on garde tel quel (propre, lossless), on ne touche plus**, et on tranche la valeur via une **expérience A/B** (deux v3, atomisée vs non-atomisée — un flag force tout en séquentiel). Cf. `todo.md` § A/B accords atomiques.
- **(v3) Sustain vs SEP :** `Sustain` remplace `SEP`. Si le mécanisme d'harmonisation mélodie→partition est ré-introduit, prévoir un token distinct `SEP_HARM`.

