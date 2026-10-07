# Pipeline Status — Complete Improvement Tracker

Everything we've discussed, implemented, and still need to do.

Last updated after MAESTRO ground-truth benchmark.

---

## Current Best Pipeline

**hFT-Transformer → MIDI2ScoreTransformer → MuseScore** (all in venv311)

4 transcribers available: `basic-pitch`, `mt3`, `transkun`, `hft`
3 backends available: `music21`, `musescore`, `transformer`

---

## DONE (Implemented)

### Phase 1 improvements

| # | What | Files changed |
|---|------|---------------|
| 4 | MT3 model caching (`_get_model()` singleton) | `mt3_inference.py` |
| 5 | Basic Pitch thresholds exposed (`--onset-threshold`, `--frame-threshold`) | `transcribe.py` |
| 6 | Post-MIDI filtering (min duration, min velocity, pitch range, debounce) | `transcribe.py` |
| — | Integrated MT3 as `--transcriber mt3` | `mt3_inference.py`, `transcribe.py` |
| — | Integrated Transkun as `--transcriber transkun` | `transcribe.py` |
| — | Integrated hFT-Transformer as `--transcriber hft` | `transcribe.py` |
| — | Fixed MPS numerical bug in MIDI2ScoreTF (forced CPU) | `transcribe.py` |
| — | Fixed MT3 double dataset iteration (`ds.cache()`) | `mt3_inference.py` |
| — | Scoped warning suppression to Basic Pitch only | `transcribe.py` |
| — | JAX 0.9 compatibility shims for MT3 (`jax.tree_map` etc.) | `mt3_inference.py` |
| — | hFT CPU unpickler + device override for macOS | `transcribe.py` |

### Phase 2 improvements

| # | What | Files changed |
|---|------|---------------|
| 7 | Increased MIDI2ScoreTF overlap from 64 to 128 | `transcribe.py` |
| 8 | A1: Lower pad threshold via `--pad-threshold` flag (default 0.5) | `tokenizer.py`, `transcribe.py` |
| 9 | A2: top-k / temperature flags (`--top-k`, `--temperature`) | `utils.py`, `transcribe.py` |
| 11 | A3: Fixed per-voice padding bug in `score_utils.py` | `score_utils.py` |
| 12 | Removed dead code (no-op dict in `generate()`) | `MIDI2ScoreTransformer/.../model.py` |
| — | A5: `--normalize-audio` flag (mono+resample+peak norm for hFT) | `transcribe.py` |
| — | A1-A5 benchmark eval framework | `benchmark/eval_improvements.py`, `benchmark/IMPROVEMENT_RESULTS.md` |

### Phase 3 improvements

| # | What | Files changed |
|---|------|---------------|
| 15 | MuseScore crash handling (returns bool, fallback to MusicXML) | `transcribe.py` |
| 16 | Save intermediate MusicXML permanently | `transcribe.py` |
| 17 | Fixed LilyPond space-in-filename bug (temp path rename) | `transcribe.py` |

### CLI / infrastructure

| # | What | Files changed |
|---|------|---------------|
| 19 | Added `--midi-input` flag to skip Phase 1 | `transcribe.py` |
| — | Added `--transcriber` flag with 4 options | `transcribe.py` |
| — | Auto-output to `outputs/<backend>/<stem>.pdf` | `transcribe.py` |
| — | Organized outputs into `outputs/music21/`, `musescore/`, `transformer/` | file moves |
| — | Set up benchmark folder with MAESTRO catalogs | `benchmark/` |
| — | Connected to GitHub repo | `.gitignore`, git init |

### Documentation

| What | File |
|------|------|
| Documented MIDI round-trip quantization limitation (#7 in Known Limitations) | `README.md` |
| Documented Pianoroll RNN-NADE (not applicable) | `README.md` |
| Documented Score2Perf / Music Transformer (not applicable, related work) | `README.md` |
| Full improvement catalog with priority matrix | `IMPROVEMENTS.md` |

---

## NOT DONE — Pending Improvements

### High priority (low effort, high/medium impact)

| # | What | Effort | Why not done yet |
|---|------|--------|------------------|
| — | hFT model caching (same pattern as MT3) | Low | Planned, not executed |
| — | hFT threshold tuning + CLI flags (`--hft-onset-threshold`, `--hft-mpe-threshold`) | Low | Planned, not executed |
| — | Document MIDI tick precision loss for hFT/Transkun | Low | Planned, not executed |
| — | Update IMPROVEMENTS.md to reflect current state (hFT as best, new transcribers) | Low | File is stale |
| — | Update README with Transkun, hFT docs, benchmark results | Low | README is stale |
| — | Git commit + push all recent changes | Low | Haven't committed since initial push |

### Medium priority (low effort, medium impact, needs testing)

| # | What | Effort | Risk |
|---|------|--------|------|
| 1 | Try `mt3` model type (with ties) for sustain-heavy pieces | Low | Medium |
| 10 | Raise measure merge threshold from 6 to 12 | Low | Low |

### Medium priority (medium effort)

| # | What | Effort | Notes |
|---|------|--------|-------|
| 2 | Expose MT3 beam search parameters | Medium | Need to trace T5X internals |
| 13 | Windowed key detection in music21 backend | Medium | Run on first N measures instead of full piece |
| 14 | Hand splitting heuristic for music21 backend (split at middle C) | Medium | MIDI2ScoreTF already does this via learned `hand` stream |
| 20 | Batch processing mode (`samples/*.mp3`) | Medium | Quality-of-life |
| — | MAESTRO Path A benchmark (needs 101GB audio download or selective download) | Medium | Blocked on audio file access |
| — | Fix MIDI2ScoreTF time signature inference (got 3/4 instead of 4/4 on benchmark) | Medium | Discovered in benchmark; root cause in tokenizer downbeat logic |

### Low priority (high effort)

| # | What | Effort | Notes |
|---|------|--------|-------|
| 3 | Overlapping audio segments for MT3 (dedup at boundaries) | High | Would fix cross-boundary note loss |
| 18 | Unify venvs (JAX + PyTorch coexistence) | High | Would eliminate the two-command MT3→transformer workflow |
| 21 | Full evaluation framework script | Medium-high | Automated comparison tables, ground-truth metrics |
| — | NoteSequence → MIDI direct handoff (skip file I/O) | High | Blocked on venv unification |

### Discovered in benchmark (new findings)

| Finding | Impact | What to do |
|---------|--------|------------|
| MIDI2ScoreTF gets time signature wrong on 2/3 GT pieces (3/4 instead of 4/4) | High | Investigate `db_config` / downbeat logic in tokenizer.py |
| MIDI2ScoreTF correctly splits into 2 hands; MuseScore only gives 1 part | High (positive) | This is MIDI2ScoreTF's strongest advantage — document it |
| MIDI2ScoreTF measure count is closer to published scores than MuseScore | Medium (positive) | Document as validation of the model's value |
| MuseScore CLI crashes on complex MIDI2ScoreTF MusicXML (>3000 notes) | Medium | Already handled with fallback; could investigate chunked rendering |
| Liszt Mazeppa: MIDI2ScoreTF produced 322 measures vs 167 expected | High | Likely related to time sig issue; too many barlines from wrong meter |
| **A1-A5 inference-time tweaks: ±0.005 F1 swing, no time-sig fix on Mazeppa** | High | Empirically confirmed Phase 2 is data-bound; pretrain plan justified. See `benchmark/IMPROVEMENT_RESULTS.md` |
| **A4 (chunk_size > 512) crashes the model**: trained at `seq_length=512`, position embeddings break above that | Medium | Chunk-size flag exists but cannot exceed 512 on this checkpoint. Fix requires retraining |

---

## Benchmark Results Summary

### Transcriber Comparison (Twinkle Twinkle, all → MIDI2ScoreTF)

| Transcriber | Notes (filtered) | Time sig | Correct 4/4? | Venv |
|---|---|---|---|---|
| Basic Pitch | 569 | 4/4 | Yes | venv |
| MT3 | 423 | 4/4 | Yes | venv_mt3 |
| Transkun | 345 | 3/4 | No | venv311 |
| hFT | 359 | 4/4 | Yes | venv311 |

### Ground-Truth MIDI Benchmark (MAESTRO → Score)

| Piece | GT Notes | MIDI2ScoreTF measures | MuseScore measures | Expected |
|---|---|---|---|---|
| Chopin Op.10 No.4 | 2306 | 82 | 60 | ~88 |
| Chopin Op.25 No.11 | 3241 | 100 (PDF crashed) | 109 | ~102 |
| Liszt Mazeppa | 7158 | 322 | 248 | ~167 |

**Conclusion**: Phase 2 (MIDI2ScoreTF) is a bottleneck in its own right — even with perfect MIDI, time signature inference is unreliable and measure counts drift on complex pieces. Hand splitting is its strongest feature.
