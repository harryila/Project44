# Tokenization Rules — Reviewed

> Revised version based on design discussions. Original file preserved in `tokenization_rules.md`.
> Scope: full-score transformer on complete kern files (2 spines: RH + LH).

---

## Head Overview

| # | Name | Content | Vocab |
|---|---|---|---|
| H1 | **Type** | `Note_Single`, `Chord_Start`, `Chord_Cont`, `Chord_End`, `Fioritura_Q`, `Rest`, `Barline` | 7 |
| H2 | **Duration** | Standard recip + dots + tuplets + grace (`q`) + appoggiatura (`P`/`p`) | ~25 |
| H3 | **Pitch** | Fused pitch+octave+accidental + `r`, `rr` | ~282 |
| H4 | **Ornament** | Trills, mordents, turns, arpeggio, generic + `NULL` | 14 |
| H5 | **Articulation** | `'`, `` ` ``, `~`, `^`, `I` + `NULL` | 6 |
| H6 | **Tie** | `Tie_Start`, `Tie_Continue`, `Tie_End`, `NULL` | 4 |
| H7 | **Slur** | `Slur_Start`, `Slur_End`, `NULL` | 3 |
| H8 | **Phrase** | `Phrase_Start`, `Phrase_End`, `&Phrase_Start`, `&Phrase_End`, `NULL` | 5 |
| H9 | **Voicing** | `/` (stem up), `\` (stem down), `NULL` | 3 |
| H10 | **Beam** | `L`,`LL`,`LLL`,`J`,`JJ`,`JJJ`,`K`,`k`, `NULL` | 9 |
| H11 | **Spine** | `RH`, `LH` | 2 |
| H12 | **Dynamic** | Levels (`pp`→`ffff`), accents (`fz`,`sfz`...), hairpins (`<`,`>`,`(`,`)`) + `NULL` | 18 |
| — | **Meta (prefix)** | `K_[sig]_maj/min`, `M_[ts]`, `MM_[bpm]`, `CLEF_G`, `CLEF_F` | ~40 |

**General rules:**
- When `H1 = Barline`: all heads H2–H12 are forced to `NULL`.
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
| `Fioritura_Q` | Fioritura note (gruppetto, `Q` in kern) |
| `Rest` | Silence (`r` or `rr`) |
| `Barline` | Measure barline |

**Post-processing:**
- `Chord_Start → ... → Chord_End`: merge into space-separated kern token (`4c 4e 4g`)
- `Fioritura_Q` or `q` in a block: extract and place **before** the principal note
- Rests cannot appear inside a Chord block

---

## H2 — Duration (Recip System)

| Category | Tokens |
|---|---|
| Standard | `0` (breve), `1`, `2`, `4`, `8`, `16`, `32`, `64` |
| Dotted | `2.`, `4.`, `8.`, `16.`, `4..`, `8..` |
| Tuplets | `3` (half triplet), `6` (quarter triplet), `12` (8th triplet), `24` (16th triplet) |
| Grace note | `q` (acciaccatura, durationless) |
| Appoggiatura | `8P`, `8p`, `16P`, `16p` (performed duration; P = leaning note, p = resolution) |

**Note on appoggiaturas:** the model must learn that `P` is strictly followed by `p`. If this does not converge, the constraint can be hardcoded at post-processing.

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
| `M` | Upper mordent |
| `m` | Lower mordent |
| `W` | Inverted mordent |
| `w` | Short lower mordent |
| `S` | Turn (standard) |
| `$` | Inverted turn |
| `:` | Arpeggio (applied to each chord member) |
| `O` | Generic ornament |
| `I` | Generic articulation |
| `NULL` | No ornament |

**Rules:** `NULL` is mandatory when `H1 = Rest` or `Fioritura_Q`.
In a chord, each member can independently carry its own ornament.

---

## H5 — Articulation

| Token | Kern meaning |
|---|---|
| `'` | Staccato |
| `` ` `` | Staccatissimo |
| `~` | Tenuto |
| `^` | Accent (sfz, >, <) |
| `I` | Generic articulation |
| `NULL` | No articulation |

**Rule:** `NULL` is mandatory when `H1 = Rest`.

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

---

## Meta — Conditioning Tokens (Prefix, never predicted)

These tokens occupy the **first N positions** of every sequence. They condition the model but are **never predicted** (masked out of the loss).

| Category | Tokens | Examples |
|---|---|---|
| Key signature | `K_[notes]_maj` / `K_[notes]_min` | `K_f#c#_maj`, `K_b-e-a-_min`, `K_0_maj` |
| Time signature | `M_[num/den]` | `M_4/4`, `M_3/4`, `M_6/8`, `M_complex` |
| Tempo | `MM_[value]` | `MM_60`, `MM_120`, `MM_200` |
| Clef | `CLEF_G`, `CLEF_F` | |
| Style | `ST_chopin`, `ST_liszt`, `ST_other` | |

**At inference:** the user provides these tokens to steer the model toward a specific key, meter, tempo, or style.

---

## Post-Processing Summary

| Operation | Trigger | Action |
|---|---|---|
| Natural sign insertion | Natural pitch + active accidental in state tracker | Append `n` to kern token |
| Accidental state reset | `Barline` token | Clear per-measure tracker |
| Chord merge | `Chord_Start → ... → Chord_End` | Merge into `4c 4e 4g` |
| Grace note split | `q` or `Fioritura_Q` in block | Move before principal note |
| Beam reconstruction | (if H10 removed) | Recompute from duration + time signature |
| Voicing fallback | (if H9 unreliable) | Highest note → `/`, others → `\` |
| Measure renumbering | Always | Increment `=N` sequentially |
| Appoggiatura constraint | `P` token | Force `p` as next duration if model diverges |

---

## Deferred Decisions

- **H10 (Beam):** included in v1; remove and reconstruct deterministically if convergence is poor
- **H9 (Voicing):** included in v1; heuristic fallback available
- **Appoggiaturas P/p:** included in v1; hard constraint at post-processing if needed
- **Style tokens:** optional; add to Meta prefix if composer-conditioned generation is desired

