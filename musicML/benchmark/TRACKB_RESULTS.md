# Track B — Cheap Inference Wins: Measured Results

**Date:** 2026-05-30 · **Verdict: inference tuning is null within noise — the
out-of-distribution failure is data-bound, not knob-tunable.** This is a measured
negative result, and it's the strongest justification yet for moving to retraining
(Track C/D).

## What was implemented (in `transcribe.py`, all opt-in, defaults unchanged)

| Flag | Purpose |
|---|---|
| `--hft-onset-threshold` / `--hft-mpe-threshold` / `--hft-offset-threshold` | Phase-1 recall knobs (were hardcoded 0.5) |
| `--hft-offset-mode {shorter,offset,longer}` | note-duration estimation (`shorter` clips sustained) |
| `--no-render` | skip MuseScore for fast eval (the scored artifact is the MusicXML) |

Production behavior is **unchanged** — every flag defaults to the prior hardcoded
value. These are reversible additions; nothing was silently re-tuned.

## What was deliberately NOT done (with reasoning)

- **Tempo recovery — skipped.** Passing `midi_sequence` to `detokenize_mxl` inserts
  `MetronomeMark(number=note.start_seconds)` as timing *scaffolding*, but there is no
  downstream code converting it to real BPM — so enabling it risks emitting garbage
  tempo marks, and it cannot move MUSTER (which scores beats, not seconds).
- **Meter generalization — deprioritized.** Mazeppa's meter is *fundamentally*
  mispredicted by the model (57% of bars vote 3/4), so no post-hoc meter logic fixes
  it; the Chopin demos are already correct. It would only help 6/8 repertoire, which
  the benchmark doesn't contain. (The `n/4`-only limitation in `_fix_time_signatures`
  remains documented as a real-but-undemonstrable-here gap.)

## Measurement: hFT-threshold sweep on Mazeppa (the headroom case)

End-to-end audio→score, MUSTER vs the 2-staff GT
([trackb_mazeppa_sweep.json](trackb_mazeppa_sweep.json)):

| config | MeanER | MissRate | ExtraRate | OnsetER | PitchER |
|---|---|---|---|---|---|
| baseline 0.50 | 34.89 | 39.56 | 35.19 | 35.01 | 16.91 |
| onset/mpe 0.35 | 34.36 | 38.34 | 34.34 | 31.70 | 17.75 |
| onset/mpe 0.25 | 34.13 | 39.34 | 34.59 | 33.14 | 18.20 |
| offset-mode `offset` | 34.96 | 38.91 | 33.95 | 36.00 | 15.52 |

## Why this is a null result (rigorously)

1. **The effect is within noise.** MeanER moves 34.9 → 34.1 across the whole sweep
   — smaller than the GT's self-vs-self noise floor (~2.1 for this community score).
2. **Run-to-run variance already exceeds it.** The earlier Tier-2 baseline measured
   Mazeppa at **34.23**; this run's identical-config baseline measured **34.89** — a
   **0.66 swing from nondeterminism alone**, larger than the entire threshold effect.
3. **Recall isn't the bottleneck.** Lowering onset/mpe to 0.25 did *not* reduce
   MissRate (39.3 vs 39.6) — the missing 38% of notes are not soft notes thresholded
   out; they're unrecoverable from the dense fff audio (or absent from this
   performance). Lowering the threshold instead makes **PitchER worse** (16.9 → 18.2),
   adding wrong notes. Classic precision/recall wash with no net gain.

## Conclusion

Three independent lines of evidence now agree — the A1–A5 study, the architecture
analysis, and this rigorous MUSTER sweep — that **the out-of-distribution failure
(Mazeppa-class) cannot be fixed by inference-time tuning.** The cheap no-GPU levers
are exhausted. The remaining accuracy is gated on **retraining / data scaling
(Track C/D)**: a model that has actually seen dense late-Romantic repertoire.

The implemented flags remain available for users who want to tune for a specific
recording, and `--no-render` is a genuine eval-speed utility — but none of them
move the benchmark, and we don't claim otherwise.
