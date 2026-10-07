# Tokenization Rules — Reviewed (v2)

> Revised version based on design discussions. Original v1 in `../../v1/tokenization_rules_reviewed.md`.
> Scope: full-score transformer on complete kern files (2 spines: RH + LH, plus optional dynam/pedal).

---

## Head Overview

| # | Name | Content | Vocab |
|---|---|---|---|
| H1 | **Type** | `Note_Single`, `Chord_Start`, `Chord_Cont`, `Chord_End`, `Fioritura_Q`, `Fioritura_q`, `Rest`, `Barline`, `Clef_G2`, `Clef_F4`, `Clef_G1` | 11 |
| H2 | **Duration** | Standard recip + dots + tuplets (no special grace marker — see note) | ~22 |
| H3 | **Pitch** | Fused pitch+octave+accidental + `r`, `rr` | ~282 |
| H4 | **Ornament** | Trills (`T`,`t`), trill+nachschlag (`TR`,`tR`), mordents (`M`,`m`,`W`,`w`), turns (`S`,`$`), arpeggio (`:`), trill-end marker (`Tr]`), generic (`O`,`I`) + `NULL` | 16 |
| H5 | **Articulation** | `'`, `` ` ``, `~`, `^`, `I`, `;` (fermata) + `NULL` | 7 |
| H6 | **Tie** | `Tie_Start`, `Tie_Continue`, `Tie_End`, `NULL` | 4 |
| H7 | **Slur** | `Slur_Start`, `Slur_End`, `NULL` | 3 |
| H8 | **Phrase** | `Phrase_Start`, `Phrase_End`, `&Phrase_Start`, `&Phrase_End`, `NULL` | 5 |
| H9 | **Voicing** | `/` (stem up), `\` (stem down), `NULL` | 3 |
| H10 | **Beam** | `L`,`LL`,`LLL`,`J`,`JJ`,`JJJ`,`K`,`k`, `NULL` | 9 |
| H11 | **Spine** | `RH`, `LH` | 2 |
| H12 | **Dynamic** | Levels (`pp`→`ffff`), accents (`fz`,`sfz`...), hairpins (`<`,`>`,`(`,`)`) + `NULL` | 18 |
| H13 | **Pedal** | `Ped_Down`, `Ped_Up`, `Ped_Change`, `NULL` | 4 |
| H14 | **Tempo** | `Rall`/`Poco_Rall`/`Molto_Rall`, `Accel`/`Poco_Accel`/`Molto_Accel`, `A_Tempo`, `Rubato`/`Poco_Rubato`/`Molto_Rubato`, `NULL` | 11 |
| — | **Meta (prefix)** | `K_[sig]_maj/min`, `M_[ts]`, `MM_[bpm]`, `CLEF_G`, `CLEF_F`, `BRACKET_PIANO` | ~42 |

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

**General rules:**
- When `H1 = Barline`: all heads H2–H14 are forced to `NULL`.
- When `H1 = Rest`: H3 = `r`/`rr`, H4–H8 = `NULL` (ornaments and articulations are forbidden on rests).

---

## H1 — Type

Manages the structure of musical events and chord linearization.

| Token | Meaning |
|---|---|
| `Note_Single` | Standalone note |
| `Chord_Start` | First note of a chord |
| `Chord_Cont` | Middle note of a chord |
| `Chord_End` | Last note of a chord — signals end of chord block, acts as hand-switch cue |
| `Fioritura_Q` | Appoggiatura (`Q` in kern, **unslashed**). H2 = visual duration (`8`, `16`, `32`…). Same metric offset as next note (durationless on the grid). |
| `Fioritura_q` | Acciaccatura (`q` in kern, **slashed**). H2 = visual duration (`8`, `16`, `32`…). Same metric offset as next note (durationless on the grid). |
| `Rest` | Silence (`r` or `rr`) |
| `Barline` | Measure barline |
| `Clef_G2` | Mid-piece clef change to **treble clef** (G on 2nd line). H11 picks the staff. All other heads = `NULL`. |
| `Clef_F4` | Mid-piece clef change to **bass clef** (F on 4th line). H11 picks the staff. All other heads = `NULL`. |
| `Clef_G1` | Mid-piece clef change to **French violin clef** (G on 1st line). Rare; appears in some 19th-century editions (Wessel, Schlesinger). H11 picks the staff. All other heads = `NULL`. |

**Post-processing:**
- `Chord_Start → ... → Chord_End`: merge into space-separated kern token (`4c 4e 4g`)
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

## H11 — Spine (Multi-Spine Management)

A complete piano kern file has 2 tab-separated columns (RH + LH). We linearize them into a single sequence by interleaving events in chronological order (same offset → RH first).

| Token | Meaning |
|---|---|
| `RH` | This event belongs to the right hand (staff1) |
| `LH` | This event belongs to the left hand (staff2) |

**Key design choice:** the model itself declares which hand it is generating at each step by predicting H11. It learns the interleaving pattern from training data. Null continuations (`.` in kern, when one hand sustains while the other plays) are **skipped** — only actual events (notes, rests, barlines) are emitted.

**`Chord_End` as hand-switch cue:** after a `Chord_End` token for RH, the model learns to emit the corresponding LH event at the same offset. The `Chord_End` token acts as an implicit synchronization signal.

**Inference safeguard:** if the model generates N consecutive RH tokens without switching to LH, a decoding constraint forces an LH token.

**Compatibility:** monophonic files (extracted melodies) all use `H11 = RH`.

**Multi-staff scores (3+ portées) — squeeze automatique en 2 spines :**
Le format est strictement **2-spines**. À l'inférence, quand un prompt MXL a 3 portées ou plus (typique chez Liszt : mélodie + arpèges + basse), `_mxl_to_kern()` dans `generate.py` fusionne en 2 par cette heuristique :
1. Clef d'abord : portées en clef de basse → LH, autres → RH
2. Fallback par pitch moyen (quand toutes les portées sont en clé de sol) : portée la plus grave → LH, autres mergées dans RH
3. Notes simultanées au même offset depuis plusieurs portées source → empilées en accord
4. Rest vs Note au même offset → la note l'emporte

**Limitations connues du squeeze** :
- Changements de clef mid-piece dans une portée source : ambigus après merge
- Durées différentes au même offset entre 2 portées : accord avec la durée de la 1ère lue (kern valide mais musicalement imprécis)

Pour le corpus d'entraînement, le script `pdmx_scrapper/mxl_to_kern.py` ne fait PAS encore ce squeeze (il prend `parts[0]` et `parts[1]` seulement) — à patcher pour v3 si on veut intégrer des pièces 3-portées au corpus.

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
| `NULL` | No dynamic (~95% of tokens) |

**Attachment rule:** dynamics are always carried by the `RH` token at the corresponding timestep. The `LH` token at the same timestep receives `NULL`. The transformer learns via attention that RH dynamics condition the LH playing as well.

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
| `NULL` | No pedal event at this token (~98% of tokens). |

**Attachment:** like H12, pedal events are carried by `RH` tokens. `LH` tokens receive `NULL`.
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
| `NULL` | No tempo direction at this token (~99.5% of tokens) | |

**Normalization:** strict synonyms collapse to one category (e.g. `ritardando` ≈ `ritenuto` ≈ `allargando` → `Rall`). The **intensity adverb** is detected and kept: `poco` / `un poco` → `Poco_`, `molto` / `assai` → `Molto_`; other adverbs (`sempre`…) are dropped. Words that are *dynamic* directions (`cres.`, `dim.`, `smorz.`, `con forza`) are **not** mapped here (they overlap H12 hairpins). Words that are purely *character / touch* indications (`dolce`, `espressivo`, `dolcissimo`) are dropped entirely.

**Attachment:** like H12 and H13, tempo events are carried by the `RH` token at the corresponding timestep; the `LH` token receives `NULL`.

**Event semantics (not sticky-state):** H14 marks the *onset* of a tempo direction, exactly like H13 pedal events. The model emits the event token once; the "until when" is implicit (an `A_Tempo` later cancels a `Rall`). Intermediate tokens carry `NULL`.

**Kern representation:** a tandem interpretation line placed at the event's position, on the `RH` (staff1) column. The kern token is `*` + intensity-prefix + category: `*rall`, `*pocorall`, `*moltorall`, `*accel`, `*pocoaccel`, `*moltoaccel`, `*Atempo`, `*rubato`, `*pocorubato`, `*moltorubato`. The other spines carry `*`. Mid-measure directions are interleaved between data lines at the right offset (same mechanism as mid-piece clef changes).

**Known limitation:** if two tempo directions sit at the *exact same musical instant* — e.g. `rall.` at the end of a measure and `a tempo` at the downbeat of the next — only the second survives the tokenizer (a single token's H14 cannot hold two events, and there is no note between them to carry the first). This is rare and analogous to the `fz`+`p` dynamic-overlap limitation of the `**dynam` spine.

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
| Grace note split | `Fioritura_q` or `Fioritura_Q` in block | Move before principal note; emit `<dur>q` (slashed) or `<dur>Q` (unslashed) where `<dur>` comes from H2 |
| Beam reconstruction | (if H10 removed) | Recompute from duration + time signature |
| Voicing fallback | (if H9 unreliable) | Highest note → `/`, others → `\` |
| Measure renumbering | Always | Increment `=N` sequentially; **skip the first barline if the opening tokens form a pickup** (shorter than the time signature) |
| Clef change emission | `Clef_G2` / `Clef_F4` / `Clef_G1` | Emit `*clef{G2\|F4\|G1}` on the column dictated by H11 |
| Pedal emission | `H13 = Ped_Down` / `Ped_Up` / `Ped_Change` | Emit `*ped` / `*Xped` on a `**pedal` spine, or as tandem if no spine |
| Tempo emission | `H14 = Rall` / `Accel` / `A_Tempo` / `Rubato` | Emit tandem `*rall` / `*accel` / `*Atempo` / `*rubato` on the RH column at the token's position |
| Dynamic deduplication | Sticky H12 values | Only emit to `**dynam` when value changes from the previous one |

---

## Deferred Decisions

- **H10 (Beam):** included in v1; remove and reconstruct deterministically if convergence is poor
- **H9 (Voicing):** included in v1; heuristic fallback available
- **Style tokens:** optional; add to Meta prefix if composer-conditioned generation is desired
- **C clefs (alto, tenor, soprano):** not currently representable — H1 only has `Clef_G2`, `Clef_F4`, `Clef_G1`. Add `Clef_C1`…`Clef_C5` if needed (none in Chopin/Liszt piano repertoire).

