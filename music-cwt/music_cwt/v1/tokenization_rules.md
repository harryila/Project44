# Tokenization Rules

### Meta/Header (Contextual Prompting)

**Strategy:** These tokens are used to condition the model at the start of the sequence. They are NOT predicted during generation but serve as a global state for the transformer's attention.

**1. Key Signatures (Total: 30 tokens)**
- **Format:** `K_[Notes]_[Type]`
- **Examples:** 
    - `K_f#c#_maj` (D major)
    - `K_b-e-a-_maj` (Eb major)
    - `K_0_maj` (C major / A minor)
- **Note:** Covers 0 to 7 sharps/flats for both Major and Minor.

**2. Time Signatures (Common in Chopin/Liszt)**
- **Tokens:** `M_2/2`, `M_2/4`, `M_3/4`, `M_4/4`, `M_6/8`, `M_9/8`, `M_12/8`.
- **Note:** Include a `M_complex` token for irregular meters found in Liszt.

**3. Clef Configuration**
- **Tokens:** `CLEF_G` (Treble), `CLEF_F` (Bass). 
- **Note:** In a multi-spine (two-hand) setup, this prompt defines the default range for the sequence.

**4. Performance Metadata**
- **Tempo:** `MM_[value]` (e.g., `MM_120`).
- **Style:** `ST_chopin`, `ST_liszt`.

**Usage Rule:**
- These tokens occupy the **first N positions** of the sequence.
- During inference, the user provides these tokens to "force" the model into a specific key or meter. 
- The model starts generating from the first **Barline** (`=`) or **Note**.


### Pitch Tokenization Strategy (**kern)

**Mapping:** Each token represents an atomic "Fused Pitch" (Pitch + Accidental + Octave) following the absolute **\*\*kern** scheme (C4 = `c`) or a silence.

- **Standard Range:** 8 octaves from `CCCC` (C0) to `cccc` (C8), covering the full MIDI/Piano range.
- **Octave Logic:** Case-sensitive repetition (Lower-case for C4+; Upper-case for below C4). The octave shift occurs between **B** and **C**.
- **Accidentals:** Includes all variations required for Liszt/Chopin repertoire: Sharp (`#`), Double-Sharp (`##`), Flat (`-`), Double-Flat (`--`), and Natural (`n`).

**Vocabulary:** 
- ~280 tokens (56 base pitches × 5 accidental states + 1 silence).
- Each token is a standalone entity (e.g., `f#`, `ee##`, `BBB--`, `cn`).

**Goal:** Maintaining atomic tokens for pitches ensures harmonic integrity and simplifies the multi-head prediction by avoiding "floating" accidentals.

### Natural Sign (`n`) Logic

**Status:** Hardcoded (Post-Processing). Do not include `n` in the model's vocabulary.

**Rules:**
1. **Context:** Use a per-measure state tracker for accidentals.
2. **Becarre (Natural) Insertion:** If the model predicts a natural pitch but a sharp/flat is active for that specific note in the current measure, append `n` (e.g., `f` -> `fn`).
3. **Implicit Reset:** All accidentals reset automatically at the barline (`=`).

**Goal:** Ensure `**kern` compliance without increasing token complexity.


### Structural Pairing Modifiers

**Scope:** Managing paired musical markings: **Slurs** `( )`, **Ties** `[ ] _`, and **Phrases** `{ }`.

**State-Based Tokens:**
- **Ties (`[ ] _`):** `Tie_Start`, `Tie_Continue`, `Tie_End`. (Harmonic duration)
- **Slurs (`( )`):** `Slur_Start`, `Slur_End`. (Legato articulation)
- **Phrases (`{ }`):** `Phrase_Start`, `Phrase_End`. (Structural breathing)

**Three different head ?**


### Chord & Fioritura (Gruppetto) Management

**Strategy:** Sequential Flattening (Linearization). Musical events are decomposed into a sequence of individual notes, then reconstructed (merged or split) during export based on their structural type.

**Logic & Tokenization**
- **Head 1 (Type) Tokens:** `Note_Single`, `Chord_Start`, `Chord_Cont`, `Chord_End`, `Fioritura_Q`.
- **FioriturasQ = Gruppettos:** These are written-out small notes (fiorituras). The `Fioritura_Q` token acts as a structural lock, ensuring each note is treated as a distinct melodic step and preventing illegal vertical stacking.
- **Consistency:** For **Chords**, the model uses self-attention to ensure all members between `Chord_Start` and `Chord_End` inherit the same **Duration** (Head 3) and **Structural** (Head 4) markers from the initial `Chord_Start` token.

**Post-Processing Rules (The "Split & Merge" Engine)**
1. **Vertical Merging (Chords):** The script identifies sequences from `Chord_Start` to `Chord_End` and merges them into a single space-separated `**kern` string (e.g., `4c 4e 4g`).
2. **Vertical Splitting (Grace Notes/Fiorituras):** To optimize generation length, the model may predict Grace notes (`q` in Head 3) or Fiorituras (`Fioritura_Q` in Head 1) within a chord block. The script MUST extract them and create **individual** records (lines) placed immediately **before** the principal note or chord to comply with `**kern` syntax.
3. **Selective Muting (Safety Layer):** When `Fioritura_Q` or `q` is detected, the post-processor filters other heads to ensure musicological validity:
    - **FORBIDDEN (Force NULL):** Ornaments (`T`, `M`, `S`, `$`), Arpeggios (`:`), and Meta-data changes. These are non-sensical on extra-metrical small notes.
    - **ALLOWED:** Articulations (`'`, `^`, `~`) and Slur/Phrase markers (e.g., `(`, `}`, `&{`). This preserves Chopin's expressive touch and phrasing flow.

**Note on Vocabulary Explosion**
We use linearization specifically to avoid the exponential growth of the vocabulary. If each possible chord were a unique token, the size would follow the formula:
$$Vocabulary\ Size = \sum_{i=1}^{N} \binom{T}{i}$$
*(Where $T$ is the number of base pitch tokens [~280] and $N$ is the maximum number of notes in a chord).*

For $N=5$, this would result in **billions of tokens**, making the model impossible to train. Our sequential approach maintains a fixed, manageable vocabulary of **~280 tokens** while allowing the model to generate any possible harmonic combination.


### Ornaments, Arpeggiation & Rests

**Tokens:**
- **Trills:** `T`, `t`, `TR`, `tR`.
- **Mordents:** `M`, `m` (upper), `W`, `w` (lower/inverted).
- **Turns:** `S` (standard), `$` (inverted).
- **Arpeggio:** `:` (Applied to each chord member).
- **Rests:** `r` (standard), `rr` (whole-measure rest).
- **Other:** `O` (generic ornament), `I` (generic articulation).

**Logic:**
- **Note-Linked:** These tokens are predicted for each pitch member. 
- **Rest Exclusion:** If a **Rest** (`r` or `rr`) is predicted in the Pitch Head, this Decorator Head MUST be forced to `NULL` (or a dedicated "rest-safe" state). Articulations and ornaments are syntactically forbidden on rests.
- **Chord Incompatibility:** Rests cannot be part of a chord stack (`Chord_Start/Cont/End`). If Head 1 = `Rest`, the model must bypass chord logic.
- **Positioning:** For rests, the Pitch Head may provide a position (e.g., `rg`), but the Decorator Head remains empty.

**Chopin/Liszt Specifics:** Since these composers often place an ornament on only one note of a chord, keeping this head independent for each chord member is essential.


### Articulations (Toucher)
- **Staccato:** `'` (short/detached).
- **Staccatissimo:** <kbd>`</kbd> (very short/sharp attack).
- **Tenuto:** `~` (held for full value/emphasis).
- **Accents:** `^` (includes all dynamic markings like `sfz`, `>`, `<`).
- **Generic:** `I` (other articulations).

**Note** we kept only the one regarding the piano.


### Voicing (Stems)
- **Up-stem (`/`):** Note belongs to the upper voice.
- **Down-stem (`\`):** Note belongs to the lower voice.
*Note: Essential for polyphonic piano textures (Liszt/Chopin).*

### Beaming (Ligatures)
- **Start Beam:** `L` (repeat for multiple beams: `LL`, `LLL`).
- **End Beam:** `J` (repeat for multiple beams: `JJ`, `JJJ`).
- **Partial Beams:** `K` (right-extending), `k` (left-extending).
*Example:* `16..LL` tied to `32JJkk`.

### Nested & Elided Markings (Advanced Structures)
- **Nesting:** Standard pairs `( )` or `{ }` can be nested (e.g., `( ( ) )`).
- **Elision (Overlapping) `&`:** Used when a new phrase/slur starts before the previous one ends.
  - *Format:* `{` (Start 1), `&{` (Start 2), `}` (End 1), `&}` (End 2).
  - *Multiple levels:* Use `&&{` for the third overlapping layer.


### Duration & Rhythmic Nature (Recip System)

**Logic:** Absolute reciprocal values based on the whole note (`1`) OR status markers for grace notes (`q`), fiorituras (`Q`), and appoggiaturas (`P`, `p`).

1. **Standard:** `0` (Breve), `1` (Whole), `2` (Half), `4` (Quarter), `8` (8th), etc.
2. **Dots:** Suffix `.` for 1.5x duration (e.g., `4.` or `8..`).
3. **Tuplets:** Reciprocal integers (e.g., `6` for triplet quarter, `12` for triplet 8th).
4. **Grace Notes (Acciaccaturas):** Token `q`. Represents a "durationless" note.
5. **Appoggiaturas (P/p):** Fused tokens (e.g., `8P`, `8p`, `16P`).
   - Durations are encoded "as performed" (already subdivided in the dataset).
   - `P` (Upper-case) marks the leaning note; `p` (lower-case) marks the resolution.
   - **Dependency:** The model must learn that a `P` duration token is strictly followed by a `p` token.

**Post-Processing Rules (Split Strategy):**
To optimize sequence length during generation, the model is allowed to predict `q` or `Q` (from Head 1) notes simultaneously with standard notes (within the same chord block). The export script must then:

- **Scan & Extract:** Identify any note with a `q` or `Q` marker within a multi-note group.
- **Vertical Split:** Extract these notes and create **individual** data records (lines) placed immediately **before** the main note or chord.
- **Ordering:** If multiple grace notes are predicted at the same time-step, they are serialized one after another (suggested order: by pitch) before the principal note.
- **Note on Appoggiaturas:** Unlike `q`, appoggiaturas (`P/p`) have real performed durations and are **not split**; they remain in the normal chronological flow.

**Benefits:** Drastically reduces generated time-steps while ensuring strict `**kern` rhythmic and syntactical validity.


**Barlines & Measure Markers**

**Logic:** These tokens act as global "resets" for the accidental state tracker and measure count.

1. **Tokens:** 
    - Standard: `=` (followed by measure number, e.g., `=1`).
    - Double: `==` (section end) or `==;` (with pause).
    - With Pause: `=;` (fermata on the barline).
2. **Post-Processing Rules:**
    - **Global Reset:** Every time a token starting with `=` is encountered, the state tracker for sharps/flats (Head 3) must be cleared.
    - **Single Record:** A barline always occupies its own record (line). When **Head 1 (Type)** = `barline`, all other heads (Pitch, Duration, Decorators) are forced to `NULL`.
3. **Numbering:** For ML simplicity, the measure number can be treated as a separate sub-token or ignored during generation and re-inserted via an incremental script.

**Impact:** Ensures the model understands the structural boundaries of the piece, especially for Chopin/Liszt where measure lengths can be complex.


