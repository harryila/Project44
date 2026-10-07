# Piano Audio → Sheet Music: Project Journey Report

A full retrospective of the building blocks evaluated, decisions made, bugs
hunted, and experiments run on the way to a working audio-to-score pipeline.

---

## 1. The problem and the architecture

The goal: take an audio recording of someone playing piano and produce
engraved sheet music (PDF) for the piece. This is "automatic music
transcription" — a problem with two distinct ML sub-problems chained
together, plus a deterministic rendering step on the end.

```
Audio (.wav/.mp3)
        │
        │  Phase 1: Audio → MIDI
        │  "What notes were played and when?"
        ▼
   MIDI (.mid)
        │
        │  Phase 2: MIDI → MusicXML
        │  "How should those notes be written on a staff?"
        ▼
  MusicXML
        │
        │  Phase 3: MusicXML → PDF
        │  Pure rendering, no ML.
        ▼
  Sheet music PDF
```

Each phase is a different research community with different tools. Phase 1 is
"music information retrieval / automatic music transcription." Phase 2 is
"symbolic music processing / score engraving." Phase 3 is engineering on top
of LilyPond or MuseScore. We tried multiple options for each.

---

## 2. Phase 1: Audio → MIDI (the transcribers)

A "transcriber" takes raw audio and outputs MIDI: a list of `(pitch, onset,
offset, velocity)` events. Different transcribers use very different
architectures and have very different strengths.

### 2.1 Basic Pitch (Spotify, 2022)

A small CNN that processes mel-spectrograms of audio and outputs three
probability maps per time-frame: onset probability, frame (sustain)
probability, and contour probability. Notes are extracted by thresholding.

**What it does well:** fast, simple to install, low VRAM, works on any
instrument. Ships as a `pip install basic-pitch`. CPU inference is fine.

**What it does poorly:** over-detects on dense piano. On Twinkle Twinkle, it
emitted 569 notes for a piece with ~150 actual notes. The thresholds are
exposed (`onset_threshold`, `frame_threshold`) but tuning them is per-piece.

**What we did with it:** wired it in as the original Phase-1 default. Later
exposed `--onset-threshold` and `--frame-threshold` as CLI flags so you can
tune at inference time.

### 2.2 Onset and Frames (Google Magenta, 2018)

The conceptual ancestor of Basic Pitch. Two-stack CNN+BiLSTM: one tower
predicts note onsets (sharp transients), another predicts note frames
(sustained pitch). The output of the onset tower gates the frame tower —
a frame only counts as part of a note if there was a corresponding onset.

This architecture is the foundation that almost every modern piano
transcriber is built on. We didn't actually integrate Onset and Frames
directly because newer models (Basic Pitch, MT3, hFT) all have it as their
ancestor and outperform it. But it's worth naming as the lineage.

### 2.3 MT3 (Google Magenta, 2021)

A T5-based encoder-decoder transformer for music transcription. Takes
spectrogram features → outputs a sequence of MIDI-like events. Trained on
multiple instrument datasets including MAESTRO. Uses the `ismir2021` model
for piano-specific transcription (with velocity), and the `mt3` model for
multi-instrument with sustain ties.

**What it does well:** very expressive output, handles polyphony better
than CNN-based approaches.

**What it does poorly:**

1. Massive dependency footprint: requires JAX, t5x, seqio, gin, airio,
   tensorflow. Different versions of each fight each other.
2. 4.1-second non-overlapping segments lose notes that sustain across
   boundaries.
3. The default model (`ismir2021`) has no tie-encoding, so notes that span
   chunk boundaries lose their offsets.

**What we did with it:**

- Spent significant time getting it to install. Required pinning `t5x` to a
  specific commit (`0a5677649d2affbda128f4744610f950222c6392`) predating an
  `airio` dependency, pinning `optax==0.1.9`, and monkey-patching `jax.config`
  and `jax.tree.*` API changes that broke `t5x` at runtime.
- Wrote `mt3_inference.py` as a local wrapper (Magenta only ships a Colab
  notebook).
- Added module-level model caching so multiple files don't reload the model.
- Cached the preprocessed dataset (`ds.cache()`) to skip a redundant
  spectrogram pass.
- Used a separate venv (`venv_mt3`) because JAX and PyTorch don't share well.
- Documented in `IMPROVEMENTS.md` that for sustain-heavy pieces (Chopin etc.)
  the `mt3` model with tie encoding might outperform `ismir2021`. Did not
  end up testing this on the benchmark since hFT outperformed both.

### 2.4 Transkun (Yan & Lu, 2022)

An event-based piano transcriber written in PyTorch, pip-installable. Predicts
note events (onset, offset, pitch, velocity) directly with continuous (not
quantized) timestamps. Key advantage: continuous timestamps mean no rounding
error at frame boundaries.

**What it does well:** Simple to install (`pip install transkun`), pure
PyTorch, runs in `venv311`, continuous timestamps preserve micro-timing.

**What it does poorly:** On Twinkle Twinkle it produced 345 notes (closer
to right than Basic Pitch's 569 but with a 3/4 time signature that
MIDI2ScoreTransformer then propagated as wrong meter — though this turned
out to be Phase 2's fault, not Transkun's).

**What we did with it:** integrated as `--transcriber transkun`, ran it on
benchmark pieces, confirmed it was a viable alternative but not our top pick.

### 2.5 hFT-Transformer (Sony, ICASSP 2023) — current best

Hierarchical frequency-temporal transformer for piano transcription. Two
stages: a frame-level spectrogram→pitch encoder, then a note-level decoder
that aggregates frames into notes with sub-frame timing precision (~1-2 ms,
much better than Basic Pitch's 12 ms or MT3's 10 ms grid).

**What it does well:**

- Best timing precision of anything we tested.
- Produces 359 notes on Twinkle (closest to ground truth of any transcriber).
- Pure PyTorch, runs in `venv311` alongside MIDI2ScoreTransformer.

**What it does poorly:**

- Distributed as a private GitHub repo with a `.pkl` checkpoint. The
  checkpoint was pickled with CUDA tensor references, so loading it on a
  CPU-only machine requires a custom unpickler.
- Some internal modules have hardcoded `device="cuda"` references that
  break on CPU.
- Config file is missing a key (`min_value`) that has to be computed at load
  time.

**What we did with it:**

- Cloned the repo, downloaded the `MAESTRO-V3/model_016_003.pkl` checkpoint.
- Wrote a `CPUUnpickler` subclass to override `torch.storage._load_from_bytes`
  to force `map_location="cpu"`.
- Patched `min_value = float(np.log(config["feature"]["log_offset"]))` into
  the config dict before model construction.
- Iterated through `amtobj.model.modules()` to override every hardcoded
  `device="cuda"` to `"cpu"`.
- Wired it in as `--transcriber hft`.
- **Optimization**: rather than write hFT's note list to a `.mid` file and
  then re-parse it for the tokenizer, we pass the raw note dicts directly to
  `MultistreamTokenizer.tokenize_notes()` (a new method we added). This
  preserves hFT's sub-millisecond timing precision through the entire
  pipeline.

### 2.6 Comparison summary

| Transcriber | Notes on Twinkle | Time-sig output | Speed | Install difficulty |
|---|---|---|---|---|
| Basic Pitch | 569 (over-detects) | 4/4 ✓ | Fast | Easy |
| MT3 | 423 | 4/4 ✓ | Slow (JAX) | **Hard** (deps) |
| Transkun | 345 | 3/4 ✗ | Medium | Easy |
| **hFT-Transformer** | **359** | 4/4 ✓ | Medium | Medium (CUDA pkl) |

We also added a `filter_midi()` post-processing step that runs after every
transcriber: drops notes shorter than 50 ms, velocity below 10, outside
piano range A0-C8, and debounces same-pitch notes within 30 ms.

---

## 3. Phase 2: MIDI → MusicXML (the score backends)

This is the harder ML problem. Given a MIDI file (which says only "this pitch
turned on at this time"), produce a MusicXML score (which has barlines,
voices, hand assignments, key signatures, accidental spellings, ties,
dynamics, etc.). This requires understanding *musical structure*, not just
acoustic events.

### 3.1 music21 (MIT, 2010+) — heuristics

A symbolic music library that includes a rule-based MIDI-to-score converter.
Reads MIDI, runs Krumhansl-Schmuckler key analysis, infers a time signature
from beat clustering, splits notes between treble and bass clef by middle-C,
emits MusicXML.

**What it does well:** zero ML, works on anything that parses, produces
*something* readable. Free.

**What it does poorly:**

- Single global key signature — modulating pieces get the "best average."
- Hand splitting is just "is the pitch above middle C?" — fails when hands
  cross.
- Time signature inference is fragile.
- No voice separation.

**What we did with it:** wired in as `--backend music21` rendering via
LilyPond. Fixed a filename-with-spaces bug in our pipeline (LilyPond chokes
on paths like "Park Blvd 2.mp3" — we render to a temp path and rename).

### 3.2 MuseScore CLI — heuristic + GUI engine

MuseScore (the popular open-source notation editor) ships a command-line
mode that does its own MIDI import. Internally it uses a more sophisticated
heuristic engine than music21's. Hand splitting is per-track, voices are
inferred, accidentals are spelled musically.

**What it does well:** the most "musically polished" output of any
non-neural approach. Time signatures are usually right. Accidental spelling
is sensible.

**What it does poorly:** the CLI converter is fragile. On any complex
MusicXML input (>3000 notes, dense polyphony) it crashes with exit code 40.
The GUI app handles the same files fine — it's specifically the headless
CLI converter that's flaky.

**What we did with it:**

- Wired in as `--backend musescore`.
- When MuseScore CLI crashes on a complex MusicXML, fall back to a
  **chunked renderer**: split the score into 10-measure chunks, render each
  separately, and merge the resulting PDFs with `pypdf`. Skip chunks that
  individually fail. This produces a partial-but-usable PDF rather than
  total failure.
- Saved intermediate MusicXML alongside the PDF for debugging and so users
  can open it in the MuseScore GUI manually if the CLI fails completely.

### 3.3 MIDI2ScoreTransformer (Beyer, ISMIR 2024) — current best, neural

A Roformer (Rotary positional encoding transformer) encoder-decoder model
trained specifically on MIDI→MusicXML. Encoder takes MIDI tokens (pitch,
onset-delta, duration, velocity); decoder produces 13 parallel output
streams (offset, downbeat, duration, pitch, accidental, key signature,
voice, stem, hand, grace, trill, staccato, plus a binary pad mask).

**What it does well:**

- Real voice separation (one of 9 voices per note).
- Real hand assignment (learned from data, not heuristic).
- Stems and beams.
- Trained on aligned (MIDI, MusicXML) pairs, so it understands engraving
  conventions a heuristic can't.

**What it does poorly:**

- Trained on only ~200 ASAP pieces. Doesn't generalize to repertoire ASAP
  doesn't include (Liszt Mazeppa, advanced Rachmaninoff).
- Uses MuseScore CLI for final rendering — inherits the crash issue.
- Has a numerical instability on Apple Silicon MPS (predicts pad=0 for every
  position when run on MPS, producing empty scores).
- Time signature inference is unreliable on dense pieces — gets Mazeppa as
  3/4 instead of 4/4.

**What we did with it:**

- Wired in as `--backend transformer`.
- Fixed the MPS bug by forcing `device="cpu"` for the transformer backend.
  (Lots of debugging time spent figuring out why output was empty —
  eventually traced to numerically incorrect attention pad logits on MPS.)
- Increased decoder chunk overlap from 64 to 128 so cross-chunk consistency
  improves (voice/hand assignment, barline placement).
- Added a post-hoc time signature correction (`_fix_time_signatures`):
  collects per-measure predicted bar lengths during detokenization, finds
  the majority vote, and rewrites the score's time signature accordingly.
  Documented the assumption: this assumes a single predominant meter for
  the whole piece, fine for 95% of repertoire but would steamroll genuine
  meter changes (Bartók etc.).
- Direct note-array handoff: when the transcriber is hFT (which we have),
  we pass its raw note dicts directly to `MultistreamTokenizer.tokenize_notes()`
  bypassing the MIDI file round-trip entirely. Preserves sub-millisecond
  timing precision.
- Bug fix: `parse_mxl` crashed on MusicXML with integer Part IDs (auto-
  generated from the model's own output). Fixed by coercing `Part.id` to
  `str` before `.lower()`.

### 3.4 Comparison summary

| Backend | Voices | Hands | Time sig | Quality on simple piano | Quality on complex piano |
|---|---|---|---|---|---|
| music21 | None | Middle-C split | Heuristic | OK | Bad |
| MuseScore CLI | Heuristic | Per-track | Usually right | Good | Crashes on dense scores |
| **MIDI2ScoreTransformer** | **Learned** | **Learned** | Wrong on Liszt | **Best** | **Best (when it works)** |

The transformer is the clear winner on simple-to-medium pieces. On the
hardest cases (Liszt Mazeppa) it fails — and we discovered this is the
real bottleneck of the whole pipeline, not Phase 1.

---

## 4. Phase 3: MusicXML → PDF (the renderers)

### 4.1 LilyPond

Text-based engraver via `score.write("lily.pdf", fp=...)` from music21. The
oldest and most beautiful music engraver available. Used for the music21
backend.

**What it does well:** beautiful output; widely respected in music
notation.

**What it does poorly:** crashes on filenames with spaces; slow on
complex scores; the `score.write` route through music21 sometimes fails
silently. Required the temp-path-and-rename trick to handle space-in-filename
inputs reliably.

### 4.2 MuseScore CLI

Used by both the `musescore` backend (directly converts MIDI) and the
`transformer` backend (renders the model's MusicXML output). Cross-platform
(installable via Homebrew on Mac, apt on Linux).

**What it does well:** matches what users see in the MuseScore GUI app.

**What it does poorly:** crashes on complex inputs (the chunked-rendering
fallback is our workaround).

### 4.3 Chunked rendering (our addition)

When MuseScore CLI fails on the full score, our `run_musescore_chunked()`:

1. Splits the MusicXML by measure into 10-measure chunks.
2. Renders each chunk to its own temp PDF.
3. Merges successful chunks with `pypdf`.
4. Logs which chunk ranges failed (so you know what's missing).

Crude but effective. On Mazeppa for example, ~20 of 29 chunks render
successfully; the resulting PDF is missing some sections but is otherwise
readable.

---

## 5. Datasets we used (and didn't use)

### 5.1 MAESTRO (Magenta, 2018)

200+ hours of piano performances recorded on a Disklavier (a piano that
records its own MIDI as it plays). Each piece has matched audio + MIDI.
This is the gold-standard training set for **Phase 1** (audio → MIDI). All
of our transcribers were trained on it (or similar).

**Critical fact for our project:** MAESTRO has no MusicXML scores. So you
can use MAESTRO for Phase-1 training but not Phase-2 training.

We used 3 specific pieces from MAESTRO as our benchmark:
- Chopin Op.10 No.4 (C# minor) — Etude
- Chopin Op.25 No.11 (A minor) — "Winter Wind"
- Liszt Transcendental Etude No.4 — "Mazeppa"

### 5.2 ASAP (CPJKU/Aligned Scores And Performances, 2020)

~200 pieces × ~5 performances each (~1000 total recordings) with
performance MIDI aligned to engraved MusicXML scores. This is the gold-
standard training set for **Phase 2** (MIDI → score). MIDI2ScoreTransformer
was trained on ASAP.

Critical fact: ASAP is small (~200 unique pieces) because creating each
pair requires manually aligning a performance MIDI to its published score.
That's the labor cost behind why ASAP doesn't have ten thousand pairs.

### 5.3 ACPAS (Aligned Classical Piano Audio + Score, 2021)

A metadata layer that joins ASAP performances to MAESTRO recordings (via
checksum matching) and adds train/val/test split annotations. We download
two CSV files (`metadata_R.csv`, `metadata_S.csv`) which `ASAPDataset`
uses to filter pieces.

### 5.4 PDMX (UCSD, 2024) — our pretrain corpus

~250,000 public-domain MusicXML scores scraped from MuseScore.com. Of these,
~181,000 are piano. About 24,000 are tagged classical. NO audio, NO
performance MIDI — just engraved scores.

This is what we plan to use for the **synthetic pretrain**: render perturbed
performance MIDIs from the scores, train MIDI2ScoreTransformer on those
synthetic pairs, then fine-tune on real ASAP pairs.

### 5.5 Things we considered but did not use

- **Pianoroll RNN-NADE** — a *generative* model (creates new music). Not
  applicable to our transcription task.
- **Score2Perf / Music Transformer** — goes Score → Performance MIDI, the
  exact opposite direction. Worth knowing about as related work.
- **Onsets and Frames** dataset variants — superseded by MAESTRO + Basic
  Pitch.

---

## 6. The improvements we shipped (chronological)

A long list. Each came out of either a benchmark observation, a bug
investigation, or an empirical experiment.

### 6.1 Wins from benchmark/visual inspection

| What | Where | Why |
|---|---|---|
| Post-MIDI filtering (debounce, min duration, pitch range) | `transcribe.py:filter_midi` | Basic Pitch over-detected by 4×; filter cleans up before Phase 2 |
| Increased MIDI2ScoreTransformer overlap 64→128 | `transcribe.py` | Better cross-chunk consistency for long pieces |
| Save intermediate MusicXML alongside PDF | `transcribe.py` | Debug + manual GUI render fallback |
| MuseScore crash → chunked render fallback | `transcribe.py:run_musescore_chunked` | Don't lose entire output when one measure breaks the renderer |
| `--midi-input` CLI flag | `transcribe.py` | Skip Phase 1, run Phase 2 only on a known-good MIDI |
| Output path standardization (`outputs/<backend>/<stem>.pdf`) | `transcribe.py` | No more `_v1`, `_v2`, `_eval` suffix chaos |

### 6.2 Bug fixes (the long list)

| What | Symptom | Cause |
|---|---|---|
| MIDI2ScoreTransformer empty PDF on Mac | 0 notes in output | MPS produces NaN attention pad logits |
| MT3 install failures | `ImportError`, `AttributeError` cascades | t5x pulls from HEAD, JAX API changed |
| hFT load failure on CPU | "deserialize CUDA on non-CUDA" | Pickled checkpoint references CUDA tensors |
| hFT runtime CUDA assertion | `Torch not compiled with CUDA enabled` | Model submodules have hardcoded `device="cuda"` |
| LilyPond render fails on "Park Blvd 2.mp3" | LilyPond errors on filename with spaces | LilyPond CLI doesn't quote |
| `parse_mxl` crashes on auto-generated XML | `'int' object has no attribute 'lower'` | music21 `Part.id` is sometimes int, not str |

### 6.3 The A1-A5 empirical study

After all the above was working, we did a controlled study to see if
inference-time tweaks could move the needle on the 3 benchmark pieces.

| Tag | Improvement | Result |
|---|---|---|
| **A1** | Lower pad threshold from 0.5 to 0.4 (`--pad-threshold` flag) | ±0 measure change, +0.000 F1 |
| **A2** | top-k=5, temperature=0.8 instead of greedy (`--top-k`, `--temperature`) | +0.001 F1 on Op10 only |
| **A3** | Per-voice padding bug fix in `score_utils.py` | No measurable effect on 4 test pieces |
| **A4** | Larger chunk size (1024 instead of 512) | **Cannot use** — model trained at 512, position embeddings break |
| **A5** | Audio mono + 16 kHz + peak normalize (`--normalize-audio`) | Mixed: shifts predicted TS but drops F1 |

The combined `_combined.pdf` and `_full.pdf` outputs in
`benchmark/<piece>/transformer/pdf/` show the difference visually. They're
side-by-side with the original baseline (no suffix) and `_hft.pdf` (audio
path baseline) so you can compare.

**Verdict from A1-A5:** inference flags don't fix the structural problem.
Mazeppa stays broken (3/4 instead of 4/4) on every variant. That's the
empirical evidence that motivated the next experiment.

---

## 7. The synthetic pretrain experiment (current frontier)

### 7.1 Why

A1-A5 proved that hyperparameter tuning won't fix Mazeppa. The problem is
that MIDI2ScoreTransformer was trained on only ~200 pieces and its training
distribution doesn't include Liszt-density polyphony. More data is the
intervention.

But more `(MIDI, MusicXML)` pairs at scale don't exist — that's why ASAP
is small. So we synthesize them.

### 7.2 How (the pipeline)

```
PDMX MusicXML scores (181K piano pieces)
        │
        │ scripts/expressive_render.py
        │ - render each score to a perturbed performance MIDI
        │ - per-bar tempo walk (~±10%)
        │ - per-note velocity wiggle (~N(0, 10))
        │ - per-chord onset jitter (~±20 ms)
        │ - duration noise (~±5%)
        │ - phrase-level loudness envelope
        │ - pair maintains exact 1:1 alignment by construction
        ▼
50,000 (perturbed-MIDI, MusicXML) pairs
        │
        │ scripts/make_pairs.py
        │ - tokenize MIDI + MXL
        │ - build per-beat chunks.json (matching ASAP chunker schema)
        │ - cache as pickle for fast loading
        ▼
data/pairs/, data/cache_pdmx/
        │
        │ MIDI2ScoreTransformer/midi2scoretransformer/train.py
        │ Stage A: pretrain on synthetic
        │ - 8 epochs over 50K pairs
        │ - lr=1e-4 (calibrated via LR-range test)
        │ - bf16 mixed precision on CUDA
        ▼
pretrain_pdmx.ckpt
        │
        │ Stage B: fine-tune on real ASAP
        │ - 5 epochs over ~800 ASAP pairs
        │ - lr=1e-5 (0.1× the Stage A LR)
        │ - warm-start from Stage A
        ▼
finetune_asap.ckpt
        │
        │ benchmark/eval_baseline.py
        ▼
Compare {baseline, pretrain, finetune} × {3 benchmark pieces, 3 held-out}
on time-sig-correct, measure-count-error, note-F1.
```

### 7.3 What's done so far (Mac CPU pilot)

- Phases R0-R2 (data pipeline): code works, validated with a 5K-pair pilot
- Stage A trainer: works mechanically, val loss converges 1.85 → 0.244 over 4 epochs
- One bug found and fixed: decoder input wasn't being dropped during
  training, so the model trivially learned "decoder input = decoder output"
  and minimized loss without learning anything generalizable.
- Bug fix to upstream tokenizer: `Part.id` int → str coercion.

### 7.4 What's blocking (the from-zero generation issue)

Even after fixing the decoder dropout, the trained checkpoint produces
**empty output** at inference. The model predicts `pad=0` for every position
when given the all-zero start token that the inference loop uses.

Why this happens: the model is trained "non-autoregressively" — it sees the
full GT score as decoder input and predicts the same score. At inference,
the decoder has to bootstrap from an all-zero start token, a regime it
never saw during training.

Why this works for the released checkpoint: unclear. Same architecture,
same hyperparameters, but the released `MIDI2ScoreTF.ckpt` generates fine.
The training-time mechanism that bridges teacher-forced training to
zero-start inference isn't in the upstream code we have. We have three
calibration experiments documented to figure this out:

1. Warm-start from `MIDI2ScoreTF.ckpt`, run 200 steps with our trainer,
   check if it still generates. Tests whether our trainer destabilizes a
   working model.
2. Try `is_autoregressive=True` so decoder input is shifted by one position
   during training (simulates the inference loop's autoregressive
   appending).
3. Try lower dropout (0.5 instead of 0.75). The high rate may be too
   aggressive for from-scratch training.

### 7.5 What's expected after R3 succeeds

The plan target on the 3 benchmark pieces:

| Piece | Baseline TS | Baseline meas err | Goal |
|---|---|---|---|
| Op10 No.4 | 4/4 ✓ | 6.8% | Maintain or improve |
| Op25 No.11 | 4/4 ✓ | 2.0% | Maintain |
| Mazeppa | **3/4 ✗** | **92.8%** | **4/4, <30% meas err** |

The bar is: post-pretrain Mazeppa gets the time signature right and the
measure count within ~50 of the reference's 167.

---

## 8. The four virtual environments and why we have them

| Venv | Python | Contents | Used for |
|---|---|---|---|
| `venv` | 3.9 | basic-pitch, music21 | Original setup; deprecated |
| `venv311` | 3.11 | torch, hFT, MIDI2ScoreTransformer, transkun | **Current default** for hFT → transformer |
| `venv_mt3` | 3.11 | jax, t5x, mt3, tensorflow | MT3 only — JAX and PyTorch don't share well |

The split exists because MT3's JAX dependency conflicts with PyTorch's
versions of numpy and CUDA libraries. Eventually one could consolidate
`venv311` and `venv_mt3` (jax + pytorch can coexist with careful pinning),
but it wasn't worth the engineering time given hFT outperforms MT3
empirically.

---

## 9. Where we are today

**Pipeline architecture (works end-to-end):**

```
audio.wav  →  hFT-Transformer  →  MIDI  →  MIDI2ScoreTransformer  →  MusicXML  →  MuseScore  →  PDF
              (Phase 1)                    (Phase 2)                            (Phase 3)
              best of 4 transcribers       best of 3 backends                   with chunked-render fallback
              tested                       tested
```

**Production CLI:**

```bash
venv311/bin/python transcribe.py audio.wav \
  --transcriber hft \
  --backend transformer \
  --pad-threshold 0.4 --top-k 5 --temperature 0.8 \
  --normalize-audio \
  --output sheet_music.pdf
```

**Empirical performance** (from `benchmark/baseline_metrics.json`):

| Piece | Time sig correct? | Measure error | Note F1 |
|---|---|---|---|
| Op10 No.4 (Chopin) | YES | 6.8% | 0.971 |
| Op25 No.11 (Chopin) | YES | 2.0% | 0.978 |
| Mazeppa (Liszt) | NO | 92.8% | 0.981 |

**Open question being addressed:** can synthetic pretraining on PDMX fix
Mazeppa-class pieces? Pipeline ready, calibration experiments queued, will
run on a 4090 GPU.

---

## 10. Lessons from the journey

1. **Phase 1 transcribers are good enough.** hFT gets 0.97+ note F1 on
   real performances. Phase 1 is not the bottleneck.

2. **Phase 2 is where the value is.** Voice separation, hand splitting,
   meter inference, accidental spelling — these are the things that turn a
   list of (pitch, time) tuples into actual sheet music. Heuristics
   (music21, MuseScore CLI) get 70% of the way; the neural model
   (MIDI2ScoreTransformer) gets the remaining 25%; the last 5% is data-
   bound on hard repertoire.

3. **Inference-time tuning has diminishing returns once the easy bugs are
   fixed.** The A1-A5 study showed pad threshold, sampling temperature, and
   voice padding are all near-zero impact on benchmark metrics. Don't burn
   time tuning hyperparameters when the architecture has fundamental
   limits.

4. **MPS is unsafe for any non-trivial transformer.** We hit two MPS-
   specific numerical bugs (the pad logit collapse and various other
   silent NaNs). Force CPU on Apple Silicon for anything important.

5. **Renderer fragility is real.** MuseScore CLI crashing on its own
   MusicXML output is a known issue; the chunked-render fallback is
   essentially a hack to work around upstream's bug. LilyPond is more
   robust but has its own quirks (filename quoting). Plan for the renderer
   to fail.

6. **Most "ML pipelines" are 80% engineering.** Look at the bug fixes
   list in section 6.2 — that's MPS bugs, JAX deps, pickled CUDA tensors,
   filename quoting. Almost no actual model-architecture changes. The
   model code is short; the surrounding infrastructure is long.

7. **Data is harder to scale than compute.** ASAP has 200 pieces because
   200 humans aligned 200 scores. We have 50K synthetic pairs because we
   wrote a 200-line renderer. The bet of the synthetic-pretrain experiment
   is whether 50K synthetic pairs can substitute for the 1000+ real pairs
   we don't have. We'll know after R3-R7 on the GPU.

---

## 11. Repository layout (for the report reader)

```
musicML/
├── transcribe.py                     # main CLI entry point
├── README.md                         # user-facing usage docs
├── IMPROVEMENTS.md                   # full priority matrix of optimizations
├── STATUS.md                         # what's done vs pending
├── MODEL_IMPROVEMENT.md              # deep-dive on dataset / training improvements
├── SYNTHETIC_PRETRAIN_STATUS.md      # state of the pretrain experiment
├── PROJECT_REPORT.md                 # this document
│
├── samples/                          # input audio for demos (TwinkleTwinkle, Chopin etc.)
├── outputs/                          # CLI outputs by backend
│   ├── music21/
│   ├── musescore/
│   ├── transformer/
│   └── improvements/                 # A1-A5 study results on Twinkle
│
├── benchmark/                        # MAESTRO benchmark on the 3 hard pieces
│   ├── chopin_op10/
│   ├── chopin_op25/
│   ├── liszt_transcendental/
│   ├── eval_baseline.py              # structural metrics computer
│   ├── eval_improvements.py          # A1-A5 sweep
│   ├── baseline_metrics.json         # numbers to beat
│   ├── improvement_metrics.json      # A1-A5 results
│   └── IMPROVEMENT_RESULTS.md        # writeup of A1-A5 findings
│
├── scripts/                          # synthetic pretrain pipeline
│   ├── filter_pdmx.py                # piano subset filter
│   ├── expressive_render.py          # score → perturbed MIDI
│   ├── make_pairs.py                 # bulk pair generation
│   ├── eyeball_check.py              # synthetic distribution sanity check
│   └── infer_with_ckpt.py            # eval helper
│
├── MIDI2ScoreTransformer/            # cloned upstream + our additions
│   └── midi2scoretransformer/
│       ├── train.py                  # OUR Lightning trainer (upstream has none)
│       ├── pdmx_dataset.py           # OUR synthetic dataset wrapper
│       ├── tokenizer.py              # bug-fixed for int Part IDs
│       ├── score_utils.py            # bug-fixed for per-voice padding
│       └── ...                       # rest is upstream
│
├── hFT-Transformer/                  # cloned upstream Sony repo
├── mt3/                              # cloned upstream Magenta repo (deprecated path)
│
├── data/
│   ├── preflight/                    # findings docs, ASAP overlap, held-out plan
│   ├── pdmx_piano_subset.csv         # filtered PDMX (regenerable, gitignored)
│   ├── pairs/                        # 5K synthetic pairs (regenerable, gitignored)
│   └── cache_pdmx/                   # tokenized cache (regenerable, gitignored)
│
├── venv/, venv311/, venv_mt3/        # the three environments (gitignored)
└── .gitignore                        # excludes regenerable data + venvs
```

---

## 12. Acknowledgments / external work

Without these public projects this would not have been possible:

- **Basic Pitch** — Spotify (MIT License)
- **MT3** — Google Magenta (Apache 2.0)
- **Transkun** — Yan & Lu (MIT)
- **hFT-Transformer** — Sony (proprietary research code; checkpoint
  redistribution rights as per their license)
- **MIDI2ScoreTransformer** — Beyer et al. ISMIR 2024 (MIT)
- **music21** — MIT
- **MuseScore** — GPL
- **LilyPond** — GPL
- **MAESTRO** — Magenta, CC-BY-NC-SA-4.0
- **ASAP** — CPJKU (research use)
- **PDMX** — UCSD (CC-BY-4.0; with `no_license_conflict` subset for safe use)
- **PyTorch, JAX, TensorFlow** — for being there
