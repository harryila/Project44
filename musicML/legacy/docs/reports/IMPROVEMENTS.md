# Pipeline Improvements & Optimizations

Comprehensive catalog of improvements across all three phases of the pipeline:
Audio → MIDI → Score → PDF/PNG.

Current best pipeline: **MT3 → MIDI2ScoreTransformer → MuseScore**.

---

## Table of Contents

- [Phase 1: Audio to MIDI](#phase-1-audio-to-midi)
  - [MT3 Improvements](#mt3-improvements)
  - [Basic Pitch Improvements](#basic-pitch-improvements)
  - [Post-MIDI Filtering (applies to both)](#post-midi-filtering-applies-to-both)
- [Phase 2: MIDI to Score](#phase-2-midi-to-score)
  - [MIDI2ScoreTransformer Improvements](#midi2scoretransformer-improvements)
  - [music21 Backend Improvements](#music21-backend-improvements)
- [Phase 3: Score to PDF/PNG](#phase-3-score-to-pdfpng)
  - [MuseScore Rendering](#musescore-rendering)
  - [LilyPond Rendering](#lilypond-rendering)
- [Cross-Cutting Concerns](#cross-cutting-concerns)
- [Evaluation Framework](#evaluation-framework)

---

## Phase 1: Audio to MIDI

### MT3 Improvements

#### 1. Try the `mt3` model type for sustain-heavy pieces

**File:** `mt3_inference.py` line 244, `model_type` parameter

Currently we use `ismir2021` (piano-specific, 127 velocity bins, no tie
encoding). The alternative `mt3` model uses `NoteEncodingWithTiesSpec`, which
encodes notes that sustain across segment boundaries via tie tokens. The
`ismir2021` model has no mechanism for this — notes at chunk boundaries simply
lose their offset events.

**What to change:**
```python
# In transcribe_audio(), allow model_type selection:
transcribe_audio(audio_path, midi_path, model_type="mt3")
```

**Trade-off:** `mt3` model lacks velocity (1 bin) and uses shorter input
segments (256 frames vs 512). For simple pieces, `ismir2021` is better. For
pieces with lots of sustain pedal (Chopin, ballads), `mt3` may produce more
accurate note durations.

**Effort:** Low — just a parameter change. Needs A/B testing.

---

#### 2. Expose decoder beam search parameters

**File:** `mt3_inference.py` line 148

The model uses T5X's `beam_search` internally but the wrapper passes
`decoder_params={"decode_rng": None}` with no beam width control. The default
T5X beam width is typically 4.

**What to change:**
```python
# In partial_predict_fn:
return self.model.predict_batch_with_aux(
    params, batch, decoder_params={
        "decode_rng": None,
        "beam_size": 8,      # default is ~4
        "alpha": 0.6,        # length normalization
    }
)
```

**Trade-off:** Larger beam = better output quality but slower inference. On CPU,
this could double inference time.

**Effort:** Medium — need to trace T5X's `predict_batch_with_aux` to confirm
which decoder_params are actually accepted.

---

#### 3. Segment boundary handling is lossy

**File:** `mt3/mt3/metrics_utils.py` lines 98-111,
`mt3/mt3/run_length_encoding.py` lines 406-411

Audio is split into non-overlapping ~4.1-second chunks (512 frames at 125
frames/sec). At each boundary, a hard `max_decode_time` cut discards any events
(including note-offs) that fall past the boundary. This means:

- Notes sustaining across chunk boundaries can lose their offset
- Events near the boundary are dropped, counted as `dropped_events`
- The `ismir2021` model has no tie encoding to recover

**What to change (conceptual):**

Option A: Use overlapping audio segments with deduplication. Split audio with
50% overlap, run inference on both, and merge predictions (keeping the "inner"
portion of each segment where the model has full context).

Option B: Add a small time buffer to `max_decode_time` (e.g. +0.5 seconds) and
deduplicate any notes that overlap between segments.

**Trade-off:** More computation; deduplication logic needs careful handling of
near-duplicate notes.

**Effort:** High — requires modifying the preprocessing and postprocessing
pipeline, and thorough testing.

---

#### 4. Model caching across multiple files

**File:** `mt3_inference.py` line 269

Currently `InferenceModel` is constructed fresh for every `transcribe_audio()`
call. Model loading + checkpoint restoration takes ~1-2 seconds. When processing
multiple files (e.g. all 3 test pieces), the model is loaded 3 times.

**What to change:**
```python
_cached_model = None

def get_model(model_type="ismir2021"):
    global _cached_model
    if _cached_model is None or _cached_model[0] != model_type:
        checkpoint_path = str(CHECKPOINT_DIR / model_type)
        _cached_model = (model_type, InferenceModel(checkpoint_path, model_type))
    return _cached_model[1]
```

**Effort:** Low.

---

### Basic Pitch Improvements

#### 5. Expose Basic Pitch confidence thresholds

**File:** `transcribe.py` line 54

Basic Pitch's `predict()` function accepts several parameters we're not using:

```python
model_output, midi_data, note_events = predict(
    str(audio_path),
    onset_threshold=0.5,         # default 0.5, higher = fewer false onsets
    frame_threshold=0.3,         # default 0.3, higher = fewer phantom notes
    minimum_note_length=58,      # default 58ms
    minimum_frequency=32.7,      # default C1
    maximum_frequency=4186.0,    # default C8
)
```

Raising `onset_threshold` to 0.6 and `frame_threshold` to 0.4 would reduce the
569-note over-detection problem significantly.

**Effort:** Low — just pass parameters through.

---

### Post-MIDI Filtering (applies to both)

#### 6. Add MIDI post-processing filters

**File:** New function in `transcribe.py` or `mt3_inference.py`

Neither transcriber applies any filtering after MIDI generation. Adding filters
between Phase 1 and Phase 2 would clean up the MIDI before it reaches the score
backend.

**Filters to add (in order):**

```python
def filter_midi(midi_path, output_path=None,
                min_duration_ms=50,
                min_velocity=10,
                pitch_range=(21, 108),   # A0 to C8 (piano range)
                debounce_ms=30):
    """Post-process MIDI to remove spurious notes."""

    pm = pretty_midi.PrettyMIDI(str(midi_path))
    for inst in pm.instruments:
        filtered = []
        for note in inst.notes:
            duration_ms = (note.end - note.start) * 1000

            # Filter 1: Minimum duration
            if duration_ms < min_duration_ms:
                continue

            # Filter 2: Minimum velocity
            if note.velocity < min_velocity:
                continue

            # Filter 3: Piano pitch range
            if note.pitch < pitch_range[0] or note.pitch > pitch_range[1]:
                continue

            filtered.append(note)

        # Filter 4: Debounce duplicate onsets (same pitch within debounce_ms)
        filtered.sort(key=lambda n: (n.pitch, n.start))
        deduped = []
        for note in filtered:
            if deduped and deduped[-1].pitch == note.pitch:
                gap_ms = (note.start - deduped[-1].start) * 1000
                if gap_ms < debounce_ms:
                    # Keep the louder one
                    if note.velocity > deduped[-1].velocity:
                        deduped[-1] = note
                    continue
            deduped.append(note)

        inst.notes = deduped

    out = output_path or midi_path
    pm.write(str(out))
```

**Expected impact:** For Twinkle with MT3 (429 notes), this would likely bring
it down to ~100-150 notes. For Basic Pitch (569 notes), probably ~200-250.

**Effort:** Low-medium. The function itself is simple. Tuning the thresholds
requires testing against multiple pieces.

---

## Phase 2: MIDI to Score

### MIDI2ScoreTransformer Improvements

#### 7. Increase inference overlap from 64 to 128+

**File:** `transcribe.py` line 279, the `infer()` call

Currently:
```python
y_hat = infer(x, model, verbose=False, kv_cache=True)
# infer() defaults: overlap=64, chunk=512
```

With 64 notes of overlap, the decoder has minimal context when continuing from
the previous chunk. Increasing overlap gives the decoder more "memory" of what
came before, leading to more consistent voice assignment, hand splitting, and
barline placement across chunk boundaries.

**What to change:**
```python
y_hat = infer(x, model, verbose=False, kv_cache=True, overlap=128)
# or even overlap=256 for complex pieces
```

**Trade-off:** More overlap = more redundant computation per chunk (the overlap
region is decoded twice and discarded from the second pass). For 512-note
chunks, overlap=128 means 25% more compute; overlap=256 means 50% more.

**Effort:** Low — single parameter change.

---

#### 8. Lower the pad threshold from 0.5 to 0.3

**File:** `MIDI2ScoreTransformer/midi2scoretransformer/tokenizer.py` line 404

```python
mask = token_dict["pad"].squeeze() > 0.5  # current
mask = token_dict["pad"].squeeze() > 0.3  # proposed
```

The `pad` stream is a sigmoid output that gates whether each position is a real
note or empty. At 0.5, borderline notes (where the model is ~50% confident) get
dropped. Lowering to 0.3 rescues notes the model is slightly unsure about.

**Trade-off:** May introduce some false-positive notes that the model was
correctly uncertain about. Test on Twinkle first — count notes at 0.5 vs 0.3
vs 0.4 and compare score quality.

**Effort:** Low — single constant change.

---

#### 9. Try `top_k=5` with `temperature=0.8` in generation

**File:** `MIDI2ScoreTransformer/midi2scoretransformer/utils.py` line 62-63,
`models/model.py` line 33

Currently hard-coded to greedy decoding:
```python
# In infer():
y_hat = model.generate(x=x_chunk, top_k=1, max_length=chunk, kv_cache=True)
```

With `top_k=1`, every token is the single most likely prediction. This is safe
but potentially suboptimal — the model might have two nearly-equal options for
voice assignment or accidental spelling, and always picking the top one can lead
to monotonous or incorrect choices.

**What to change:**
```python
y_hat = model.generate(x=x_chunk, top_k=5, temperature=0.8,
                       max_length=chunk, kv_cache=True)
```

**Trade-off:** Non-deterministic output; may occasionally produce garbage.
Should be A/B tested and possibly used with a seed for reproducibility.

**Effort:** Low — parameter change. But needs careful evaluation.

---

#### 10. Raise measure merge threshold in postprocessing

**File:** `MIDI2ScoreTransformer/midi2scoretransformer/score_utils.py` line 143

```python
if len(next_m.flatten().notes) > 6:   # current: skip merge if > 6 notes
    not_candidates.add((i, i+1))
    continue
```

This prevents merging adjacent measures when the next measure has more than 6
notes. For dense passages, this creates many tiny measures. Raising the
threshold allows more aggressive merging.

**What to change:**
```python
if len(next_m.flatten().notes) > 12:  # or 15 for very dense passages
```

**Trade-off:** Too aggressive merging can create overly long measures that are
hard to read. The visual result depends on the piece.

**Effort:** Low.

---

#### 11. Fix voice padding bug in postprocessing

**File:** `MIDI2ScoreTransformer/midi2scoretransformer/score_utils.py` line 247

```python
if m.highestTime < m.barDuration.quarterLength:
    quarterLength = m.barDuration.quarterLength - v.highestTime
    rest = note.Rest(quarterLength=quarterLength)
    v.append(rest.splitAtDurations())
```

The condition checks `m.highestTime` (the measure's global highest time) but
pads based on `v.highestTime` (the individual voice). If one voice fills the
bar but another is short, the short voice won't get padded because the measure
condition fails. Should check per-voice:

```python
if v.highestTime < m.barDuration.quarterLength:
    quarterLength = m.barDuration.quarterLength - v.highestTime
    rest = note.Rest(quarterLength=quarterLength)
    v.append(rest.splitAtDurations())
```

**Effort:** Low — single line fix. Needs testing to confirm it doesn't break
other cases.

---

#### 12. Dead code cleanup in generate()

**File:** `MIDI2ScoreTransformer/midi2scoretransformer/models/model.py`
lines 109-115

```python
if k == "accidental":
    {
        0: 'double-flat',
        1: 'flat',
        2: 'natural',
        3: 'sharp',
        4: 'double-sharp',
    }
```

This dict literal is a no-op (not assigned to anything). It's a leftover
comment-as-code. Harmless but should be removed or converted to a comment.

**Effort:** Trivial.

---

### music21 Backend Improvements

#### 13. Improve key detection with windowed analysis

**File:** `transcribe.py` line 143

Currently: `key = score.analyze("key")` runs Krumhansl-Schmuckler on the
entire piece. For pieces that modulate (most non-trivial music), this gives a
"best average" key that may not match any specific section.

**What to change:** Run key analysis on the first N measures (e.g. 8-16) to get
the opening key, which is more likely to be the "correct" key signature for the
score. Optionally detect key changes at section boundaries.

**Effort:** Medium.

---

#### 14. Add hand splitting heuristic for single-track MIDI

**File:** `transcribe.py`, `backend_music21()` function

Basic Pitch and MT3 both produce single-track MIDI. music21 imports this as one
part. A simple split at middle C (MIDI 60) would assign low notes to bass clef
and high notes to treble clef, creating a two-staff piano score.

```python
from music21 import stream
treble = stream.Part()
bass = stream.Part()
for note in score.flatten().notes:
    if note.pitch.midi >= 60:
        treble.append(note)
    else:
        bass.append(note)
```

**Trade-off:** Fails when hands cross (left hand plays above middle C).
MIDI2ScoreTransformer handles this better since it has a learned `hand` stream.

**Effort:** Medium.

---

## Phase 3: Score to PDF/PNG

### MuseScore Rendering

#### 15. Handle MuseScore CLI crashes gracefully

**File:** `transcribe.py` lines 119-130, 293

MuseScore CLI (`mscore -o`) crashes with exit code 40 on large/complex
MusicXML files (observed with Chopin at 1977 notes and Park Blvd 2 at 6477
notes). Currently this crashes the entire pipeline.

**What to change:**

```python
def run_musescore(mscore_path, input_path, output_path):
    result = subprocess.run(
        [mscore_path, "-o", output_path, input_path],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=env,
    )
    if result.returncode != 0:
        log.warning("MuseScore CLI failed (exit %d)", result.returncode)
        return False
    return True
```

Then in the transformer backend, fall back to saving the MusicXML when MuseScore
fails:

```python
if not run_musescore(mscore, tmp_mxl, str(output_path)):
    log.warning("Falling back to MusicXML export")
    shutil.copy(tmp_mxl, str(output_path.with_suffix(".musicxml")))
```

**Effort:** Low.

---

#### 16. Save intermediate MusicXML from transformer backend

**File:** `transcribe.py` line 287-294

Currently the transformer backend writes MusicXML to a temp file and deletes it
after MuseScore renders. For debugging and for opening in MuseScore's GUI (which
handles complex files better than the CLI), the MusicXML should be preserved.

**What to change:** Save it alongside the PDF:
```python
mxl_path = output_path.with_suffix(".musicxml")
mxl.write("musicxml", fp=str(mxl_path))
log.info("      MusicXML saved: %s", mxl_path)
```

**Effort:** Trivial.

---

### LilyPond Rendering

#### 17. Fix space-in-filename issue

**File:** `transcribe.py` lines 170-188

LilyPond rendering fails when the output path contains spaces (e.g. "Park Blvd
2_mt3.pdf"). The `score.write(lily_fmt, fp=stem)` call passes the stem directly
to LilyPond which chokes on spaces.

**What to change:** Use a temp path without spaces for LilyPond rendering, then
rename the output:

```python
import tempfile
with tempfile.NamedTemporaryFile(suffix="." + ext[1:], delete=False) as tmp:
    tmp_stem = tmp.name.rsplit(".", 1)[0]
written = Path(str(score.write(lily_fmt, fp=tmp_stem)))
written.rename(output_path)
```

**Effort:** Low.

---

## Cross-Cutting Concerns

#### 18. Unify venvs or add venv auto-detection

Currently the pipeline requires 3 different venvs:
- `venv` (Python 3.9): Basic Pitch + music21/musescore backends
- `venv311` (Python 3.11): transformer backend (PyTorch)
- `venv_mt3` (Python 3.11): MT3 transcriber (JAX/TensorFlow)

Running MT3 → MIDI2ScoreTransformer requires two separate commands with
different venvs. Options:

**Option A:** Subprocess dispatch — `transcribe.py` detects which venv to use
for each phase and shells out to the correct Python interpreter.

**Option B:** Consolidate `venv311` and `venv_mt3` into one environment.
Requires resolving JAX + PyTorch coexistence (they can coexist but need careful
pinning).

**Option C:** Docker container with all dependencies pre-installed.

**Effort:** High for any option. Option A is the most practical.

---

#### 19. Add `--skip-phase1` flag for pre-existing MIDI

Currently there's no way to skip Phase 1 and feed a pre-existing MIDI directly
into a backend. This would be useful for:
- Running the same MIDI through multiple backends for comparison
- Using MIDI from external tools
- The two-venv workflow (MT3 generates MIDI, then transformer backend processes
  it)

**What to change:** Add `--midi-input` flag that skips audio-to-MIDI:
```python
parser.add_argument("--midi-input", type=Path, default=None,
    help="Skip Phase 1 and use this MIDI file directly.")
```

**Effort:** Low.

---

#### 20. Batch processing mode

Currently the CLI processes one file at a time. Adding glob/directory support
would allow processing all files in `samples/` in one command:

```bash
python transcribe.py samples/*.mp3 -t mt3 -b transformer
```

**Effort:** Medium.

---

## Evaluation Framework

#### 21. Build quantitative comparison tooling

Currently comparisons are manual (note counts, visual PDF inspection). An
evaluation script should:

- Count notes per transcriber per piece
- Compare pitch distributions (histogram)
- Measure timing accuracy if ground-truth MIDI is available
- Generate a comparison table automatically

```bash
python evaluate.py samples/TwinkleTwinkle.mp3 \
    --ground-truth ground_truth/TwinkleTwinkle.mid \
    --transcribers basic-pitch,mt3 \
    --backends music21,musescore,transformer
```

**Effort:** Medium-high.

---

## Priority Matrix

| # | Improvement | Effort | Predicted Impact | Actual Impact (measured) | Risk |
|---|-------------|--------|------------------|--------------------------|------|
| 6 | Post-MIDI filtering | Low | High | (shipped, not re-measured) | Low |
| 7 | Increase MIDI2ScoreTF overlap | Low | Medium | (shipped, not re-measured) | Low |
| 5 | Expose Basic Pitch thresholds | Low | Medium | (shipped) | Low |
| 15 | Handle MuseScore crashes | Low | Medium | (shipped, partial PDFs save) | Low |
| 19 | Add --skip-phase1 flag | Low | Medium | (shipped) | Low |
| 8 | **A1: Lower pad threshold** | Low | Medium | **±0 F1, 0 measure change on 4 pieces** | Low |
| 9 | **A2: top_k=5, T=0.8** | Low | Medium | **+0.001 F1 (Op10), no other change** | Low |
| 11 | **A3: Voice padding fix** | Low | Low | **No measurable change on 4 pieces** | Low |
| — | **A4: chunk_size > 512** | Low | Medium | **Cannot use — model trained at 512, IndexError** | High |
| — | **A5: Audio mono+resample+peak normalize** | Low | Medium | **Mixed. Mazeppa: TS still wrong, F1 dropped 0.959 → 0.936** | Medium |
| 4 | MT3 model caching | Low | Low | (shipped) | Low |
| 16 | Save intermediate MusicXML | Trivial | Low | (shipped) | None |
| 12 | Dead code cleanup | Trivial | None | (shipped) | None |
| 17 | Fix LilyPond space-in-filename | Low | Low | (shipped) | Low |
| 1 | Try mt3 model for sustain pieces | Low | Medium | (not tested) | Medium |
| 10 | Raise measure merge threshold | Low | Low | (not tested) | Low |
| 2 | Expose beam search params | Medium | Medium | (not tested) | Medium |
| 13 | Windowed key detection | Medium | Medium | (deprioritized) | Low |
| 14 | Hand splitting heuristic | Medium | Medium | (deprioritized — TF backend has hand stream) | Medium |
| 3 | Overlapping audio segments | High | High | (deprioritized — MT3 not primary) | High |
| 18 | Unify venvs | High | Medium | (partially via direct hFT handoff) | Medium |
| 20 | Batch processing | Medium | Low | (deferred) | Low |
| 21 | Evaluation framework | Medium-high | High | **A1-A5 framework shipped at `benchmark/eval_improvements.py`** | Low |

## A1-A5 measured results (2026-04)

Tested on Twinkle Twinkle plus 3 MAESTRO benchmark pieces (Chopin Op.10 No.4, Op.25 No.11, Liszt Mazeppa). Each improvement was tested individually and combined. See [benchmark/IMPROVEMENT_RESULTS.md](benchmark/IMPROVEMENT_RESULTS.md) for the full table and analysis.

**Top-line: inference-time tweaks barely move the needle.** Note F1 swings are within ±0.005 across all variants; measure counts within ±2; time signature predictions are unchanged from baseline. Mazeppa stays broken (3/4 instead of 4/4) regardless of any inference flag combination. This empirically confirms Phase 2 is **data-bound** rather than hyperparameter-bound, justifying the synthetic pretrain plan.
