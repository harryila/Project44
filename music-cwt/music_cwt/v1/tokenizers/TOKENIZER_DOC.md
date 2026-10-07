# Multi-Head Tokenizer — Technical Documentation

> Oral presentation reference.
> Spec: `tokenization_rules_reviewed.md` · Implementation: `multihead_tokenizer.py`

---

## 1. Motivation: why multi-head?

### The flat vocabulary problem

A raw kern token looks like `(8.cc#/L` — it fuses duration, pitch, articulation, voicing, and beam into a single string.

**Flat vocabulary** (one token = one unique string):
- Very large vocab → ~13 T tokens for full Humdrum kern
- Inter-attribute relationships are opaque to the model
- A rare token (`8.cc#/L`) is treated as completely unrelated to `8.cc#/J`, even though they differ only in beam direction

**Multi-head vocabulary** (one token = 12 categorical values):
- Total vocab: **~378 values** split across 12 heads
- The model learns pitch, duration, and articulation distributions independently
- Better generalization: `4c/` and `4d/` share the same duration and voicing heads

### The bekern alternative (Ríos-Vila 2023)

Decomposes tokens with a separator `·`: `8e-J` → `8·e·-·J`.
- Vocab reduced from ~20,000 (observed corpus) → ~184 tokens
- But sequences are 3–4× longer
- No explicit structure between attributes

**Our choice: multi-head** — explicit structure, short sequences, designed for the Compound Word Transformer.

---

## 2. The 12 heads at a glance

| # | Name | Content | Vocab size |
|---|------|---------|:---:|
| H1 | **Type** | `Note_Single`, `Chord_Start`, `Chord_Cont`, `Chord_End`, `Fioritura_Q`, `Rest`, `Barline`, `SEP` | 8 |
| H2 | **Duration** | Recip values + dots + tuplets + grace (`q`) + appoggiatura (`P`/`p`) | 24 |
| H3 | **Pitch** | Fused pitch token (letter + octave + accidental) + `r`, `rr` | 282 |
| H4 | **Ornament** | Trills, mordents, turns, arpeggio, generic + `NULL` | 14 |
| H5 | **Articulation** | `'` `\`` `~` `^` `I` + `NULL` | 6 |
| H6 | **Tie** | `Tie_Start`, `Tie_Continue`, `Tie_End`, `NULL` | 4 |
| H7 | **Slur** | `Slur_Start`, `Slur_End`, `NULL` | 3 |
| H8 | **Phrase** | `Phrase_Start`, `Phrase_End`, `&Phrase_Start`, `&Phrase_End`, `NULL` | 5 |
| H9 | **Voicing** | `/` (stem up), `\` (stem down), `NULL` | 3 |
| H10 | **Beam** | `L` `LL` `LLL` `J` `JJ` `JJJ` `K` `k` + `NULL` | 9 |
| H11 | **Spine** | `RH`, `LH` | 2 |
| H12 | **Dynamic** | `pp`→`ffff`, accents (`fz`, `sfz`…), hairpins (`<`, `>`, `(`, `)`) + `NULL` | 19 |
| — | **Meta (prefix)** | `K_[sig]_maj/min`, `M_[ts]`, `MM_[bpm]`, `CLEF_G/F`, `ST_style` | ~40 |

**Total: ~378 distinct values** (vs ~10¹³ theoretical / ~20,000 observed in a homogeneous corpus)

---

## 3. Key design decisions per head

### H1 — Type (event structure)

- Encodes the **structural role** of the event, not its musical content
- Enables **chord linearization** without vocabulary explosion
- `Chord_End` also acts as an implicit **hand-switch cue** for the model
- `Fioritura_Q` = written-out small note (kern gruppetto `Q`), distinct from grace note `q`

### H2 — Duration (kern recip system)

```
Standard    : 0 (breve), 1, 2, 4, 8, 16, 32, 64
Dotted      : 2.  4.  8.  16.  4..  8..
Tuplets     : 3 (half triplet), 6 (quarter triplet), 12 (8th triplet), 24 (16th triplet)
Grace note  : q   (acciaccatura, no metric duration)
Appoggiatura: 8P, 8p, 16P, 16p   (performed duration; P = leaning note, p = resolution)
```

- The model must learn that `P` is **strictly followed** by `p` → hard constraint in post-processing if needed

### H3 — Pitch (fused token)

**Kern convention:**
- Lowercase = C4 and above: `c`(C4), `cc`(C5), `ccc`(C6), `cccc`(C7)
- Uppercase = below C4: `C`(C3), `CC`(C2), `CCC`(C1), `CCCC`(C0)

**Vocabulary construction:**
```
7 letters × 8 octave levels × 5 accidental states = 280 tokens
+ r (rest) + rr (whole-measure rest) = 282 tokens
```

Accidentals covered: `#`, `##`, `-`, `--` (all required for Chopin/Liszt repertoire)

### H6 / H7 / H8 — Tie, Slur, Phrase (three separate heads)

**Why 3 independent heads?**
In Chopin, it is common for a note to simultaneously end a slur AND a phrase:

```kern
4c)})    ← slur end + phrase end on the same note
```

H6, H7, and H8 are **fully independent** and can all be active at the same time.
A single "grouping markers" head would require 4 × 3 × 5 = 60 combinations.

### H11 — Spine (multi-spine management)

- The model **declares its own hand** at each step (RH or LH)
- It learns the RH/LH interleaving pattern from training data
- Null continuations (`.` in kern, when one hand sustains while the other plays) are **skipped** — only real events are emitted
- **Inference safeguard**: if the model generates N consecutive RH tokens without switching → force an LH token

### H12 — Dynamic

- Extracted from the `**dynam` spine if present, otherwise from tandem interpretations (`*p`, `*mf`…)
- **Attachment rule**: always carried by the RH token at the corresponding timestep; the LH token at the same timestep receives `NULL`
- **Sticky value**: the last seen dynamic carries forward until the next change
- The transformer learns via attention that RH dynamics also condition LH playing

---

## 4. Multi-spine management (2-spine files)

### Problem

A piano kern file = 2 tab-separated columns. But:
- Column order is not fixed (`*staff2` can be column 0, `*staff1` can be column 1)
- Spines can **split** (`*^`) or **merge** (`*v`) mid-piece (polyphonic passages)
- A `**dynam` spine may appear as a 3rd column

### Solution: SpineTracker

`_SpineTracker` maintains `_staves[col] = {1, 2, None}` line by line:

| Kern token | Action |
|---|---|
| `**kern` | Initialize column slots |
| `*staff1` / `*staff2` | Assign staff 1 or 2 to current column |
| `*^` | Duplicate a column (voice split) |
| `*v` | Merge two adjacent columns |
| `*x` | Exchange two adjacent columns |
| `*-` | Remove a column |

At each data line: `staff1_cols` → RH tokens, `staff2_cols` → LH tokens.

### Linearization order

```
Kern line :  [RH_token]  \t  [LH_token]
Output    :  RH events, then LH events  (same temporal offset)
```

Emission order: **RH before LH** for the same timestep.

---

## 5. Chord linearization

A kern chord = space-separated token: `4c 4e 4g`

```
Input  : "4c 4e 4g"    (1 kern token, 3 notes)
Output :
  CompoundToken(H1=Chord_Start, H2=4, H3=c, …)
  CompoundToken(H1=Chord_Cont,  H2=4, H3=e, …)
  CompoundToken(H1=Chord_End,   H2=4, H3=g, …)
```

- N=1 → `Note_Single`
- N≥2 → first=`Chord_Start`, last=`Chord_End`, middle=`Chord_Cont`
- Each chord member independently carries its own ornament, articulation, tie, slur
- Post-processing: `Chord_Start → … → Chord_End` is re-merged into kern token `4c 4e 4g`

**Why not one token per chord?**
```
Vocabulary size = Σ C(282, k)  for k = 1..N
For N = 5  →  several billion combinations
```
Linearization keeps the pitch vocabulary fixed at **282 tokens**.

---

## 6. Post-processing (model output → valid kern file)

| Operation | Trigger | Action |
|---|---|---|
| Natural sign insertion | Natural pitch + active accidental in measure tracker | Append `n` to kern token |
| Accidental reset | `Barline` token | Clear per-measure tracker |
| Chord merge | `Chord_Start → … → Chord_End` | Re-merge into `4c 4e 4g` |
| Grace note extraction | `q` or `Fioritura_Q` in a block | Move **before** the principal note |
| Beam reconstruction | (if H10 removed) | Recompute from duration + time signature |
| Voicing fallback | (if H9 unreliable) | Highest note → `/`, others → `\` |
| Measure renumbering | Always | Increment `=N` sequentially |
| Appoggiatura constraint | `P` token | Force `p` as next duration if model diverges |

### Natural sign logic

The natural sign `n` is **not in the model's vocabulary**.
- Per-measure, per-voice accidental state tracker
- If the model predicts `f` while `f#` is active in that measure → append `fn`
- Automatic reset at every `Barline` token

---

## 7. Meta prefix (conditioning tokens)

Meta tokens are **never predicted** (masked from the loss). Each sequence contains **two** Meta blocks separated by `SEP`.

| Category | Format | Examples |
|---|---|---|
| Key signature | `K_[notes]_maj/min` | `K_f#c#_maj`, `K_b-e-a-_min`, `K_0_maj` |
| Time signature | `M_[n/d]` | `M_3/4`, `M_6/8`, `M_complex` |
| Tempo | `MM_[bpm]` | `MM_60`, `MM_72`, `MM_120` |
| Clef | `CLEF_G`, `CLEF_F` | |
| Style | `ST_[composer]` | `ST_chopin`, `ST_liszt` |

**At inference**: the user provides these tokens to steer the model toward a specific key, meter, tempo, or style.

---

## 8. SEP token and sequence layout

`SEP` is a structural token in H1 (same convention as `Barline`: all other heads NULL).  
It marks the boundary between the **melody prompt** (1 spine) and the **full-score generation** (2 spines).

```
[Meta_melody: K, M, MM, CLEF_G]            ← 1-spine conditioning
  [melody tokens — H11 = RH only]
  SEP                                       ← H1=SEP, H2–H12=NULL
[Meta_score: K, M, MM, CLEF_G, CLEF_F]     ← 2-spine conditioning
  [full score tokens — H11 = RH | LH interleaved →]
```

- The second Meta prefix re-conditions the model for 2-spine generation
- Key/time/tempo can differ between the two blocks (transposition, tempo change)
- At inference: provide melody + SEP + score Meta prefix → model continues with the full score

---

## 9. Structural constraints (applied in the loss)

| Condition | Heads forced to NULL |
|---|---|
| `H1 = Barline` | H2–H12 all NULL |
| `H1 = SEP` | H2–H12 all NULL |
| `H1 = Rest` | H4 (ornament), H5 (articulation), H8 (phrase) |
| `H1 = Fioritura_Q` | H4 (ornament), H8 (phrase) |
| `H11 = LH` | H12 (dynamic) always NULL |

---

## 9. Deferred decisions (v2)

| Head | Status | Fallback |
|---|---|---|
| H10 Beam | Included in v1; fully reconstructible from duration + time sig | Drop and recompute if it hurts convergence |
| H9 Voicing | Included in v1; not deducible from pitch alone | Highest note = `/`, others = `\` |
| P/p appoggiatura | Included in v1 | Hard constraint in post-processing if needed |
| Style tokens | Optional | Add to Meta prefix for composer-conditioned generation |
