# GPU Fine-Tune Results (Track C, first run) — 2026-05-31

The first real GPU run of the warm-start fine-tune (Track C). **Headline: naive
fine-tuning on synthetic hand-coded-jitter pairs DEGRADED the model by +6.9 MeanER.
This is a clean, decisive negative result.** It rules out the cheapest version of
the data-scaling bet and points to real aligned data as the only path with upside.

## Setup
- **Warm-start** from the released `MIDI2ScoreTF.ckpt` (validated by Calibration A
  to preserve generation).
- Fine-tuned on **52,902 deduped synthetic PDMX pairs** (hand-coded jitter renderer),
  3 epochs, bf16, on an 8× RTX 4090 box.
- Evaluated with the validated Tier-1 MUSTER harness on the 59-perf ASAP held-out split.

## Result (MUSTER, lower = better)

| Run | Device | n | PitchER | MissRate | ExtraRate | OnsetER | OffsetER | **MeanER** |
|---|---|---|---|---|---|---|---|---|
| Baseline (released) | CPU | 59 | 3.19 | 7.83 | 6.67 | 14.50 | 23.73 | **11.18** |
| Baseline (released) | CUDA | 59 | — | — | — | — | — | **11.10** |
| **Fine-tuned (3 epochs)** | CUDA | 59 | 4.53 | 11.48 | 13.22 | 27.19 | 34.10 | **18.10** |

The CPU↔CUDA gap on the baseline is ~0.08 MeanER (11.18 vs 11.10) — device is NOT
the confound. The fine-tune's +6.9 regression is real. (Earlier note that the gap
was ~0.9 was based on a partial 45/59 CUDA run; the full 59/59 CUDA baseline is 11.10.)
so the conclusion is robust regardless of device.

**Every composer regressed.** Worst: Rachmaninoff +13.8, Brahms +14.5, Chopin +8.8.
Per-composer deltas are in `benchmark/finetune_eval_full/REPORT.txt`.

## Why it got worse (diagnosis)
- **Generation is healthy** — 59/59 scored, 0 errors, 0 empty. The warm-start held
  (no collapse). The model generates fine; it just generates *worse* scores.
- **It never fit the synthetic data well** — `val/total` plateaued at ~0.25 (the
  saved ckpts are `epoch=01 …0.2553` and `epoch=02 …0.2527`).
- **Mechanism: distribution drift / catastrophic forgetting.** 52,902 synthetic
  pairs vastly outnumber the ~800 real ASAP pairs the released model was tuned on
  (~65:1). Fine-tuning pulled the model toward synthetic hand-coded-jitter timing,
  which does not match real human performances → worse on the real test set.
- **One thing the synthetic data DID teach:** `NoteDuration` *improved* (56.4 → 48.1)
  and stems/beams nudged better — so the data carries *some* useful engraving signal,
  but the net effect is dominated by the timing-distribution drift.

## Corroboration with prior research
This matches the rendering-landscape research (`scripts/` workflow, 2026-05-31):
**no published work shows that an expressive-rendering / synthetic-pair pipeline
improves a downstream MIDI→score model.** The nearest positive results are for a
*different* task (audio synthesis → audio→MIDI transcription). So this negative
result is consistent with the literature, not a bug.

## Operational lesson (important for future runs)
The synthetic-pair dataset is **brutally CPU-bound to load** — every `__getitem__`
parses MusicXML chunks. Running 2 trainings (× 8 dataloader workers) + parallel
evals saturated the box (load avg 56, GPUs idle at 1%) and the trainings hung in
data-loading without completing an epoch. **Run ONE training at a time**, keep
`num_workers` modest, and run evals separately. (Eval has no DataLoader workers, so
multiple evals can run in parallel across GPUs.)

## What this means for the roadmap
- **Stop iterating hand-coded-jitter synthetic.** It is now empirically shown to
  hurt, and the literature agrees it's unproven.
- **The path with upside is REAL aligned data** (ACPAS / PianoCoRe) used in a
  **joint** fine-tune (real + small synthetic, low LR), mirroring the paper's actual
  recipe — not synthetic-then-eval. See `legacy/docs/reports/ACCURACY_ROADMAP.md` Track D and the
  follow-up plan.
- Diagnostic still in progress: the **drift trajectory** (epoch01 vs epoch02 vs
  final) and a **gentle low-LR re-run** to separate "over-training/forgetting"
  (recipe-fixable) from "synthetic distribution is fundamentally off" (needs real
  data). Results appended below when complete.

## Drift trajectory (appended 2026-06-01)

Re-scored the per-epoch checkpoints to see whether the degradation is gradual
(over-training) or immediate (distribution shift):

| Checkpoint | MeanER | Notes |
|---|---|---|
| Baseline (released), CUDA, 59/59 | **11.10** | device-control anchor |
| Fine-tune epoch 1 | **~17.7** | (26/59 partial; settling ~17.7–18.2) |
| Fine-tune final (3 epochs) | **18.10** | full 59/59 |

**The degradation is present from epoch 1, not gradual.** The model lands at ~18
within the first epoch and stays there. This rules out "over-training" — lowering
LR or training fewer epochs would NOT recover the baseline. The cause is the data
mix: with ~52k synthetic vs ~822 real pairs (≈58:1), the synthetic distribution
dominates the loss from the start. The fix must keep real data in the loss — which
is what ARM-1 (the mixed 80/20 real:synth loader) does.

## Bug found while building the fix: silent ASAP chunker failure
Building the real-data loader required ASAP `_chunks.json` for all train
performances. The upstream `chunker.py` had a **path-prefix bug**: it stripped
`"./data/asap-dataset/"` from MIDI paths to look up alignment annotations, but
`ASAPDataset` supplies paths as `"data/asap-dataset/..."` (no `./`). The mismatch
raised a `KeyError` that joblib swallowed silently — so it **skipped every
not-yet-chunked performance without logging anything**, leaving the chunk count
stuck at 505. Fixed by stripping either prefix; rebuilt to the full **967** chunks
(the complete set of aligned ASAP performances). Without this, the mixed loader
crashes on the first un-chunked piece. (Patch in `chunker.py:handle_file`.)

## Next experiment: ARM-1 (running)
Warm-start from the released ckpt → mixed **80/20 real:synth** fine-tune (real ASAP
822-perf train split via a WeightedRandomSampler over ConcatDataset, validated on
real ASAP only), lr 3e-5, 8 epochs, single-GPU, MUSTER-gated vs 11.18. Result
appended when checkpoints are evaluated.

Honest expectation: most likely near-baseline (10.8–12.0) — a modest win or clean
null — since the released model is already trained on ASAP. The key signals: does
the mix PREVENT the 18.10 catastrophic drift (it should), and does it beat 11.18 on
≥3 sub-metrics (the real success bar). If it nulls, the bottleneck is genuinely-new
real data (ARM-2 unpaired-scores path, or ATEPP), not more synthetic-ratio tuning.

### ARM-1 RESULT (2026-06-01)

ARM-1 trained cleanly (8 epochs, ~12 it/s once the ASAP parse-cache was pre-built;
val/total 0.67 on REAL ASAP — a meaningful number, vs the synthetic-only run's
misleading 0.25 on synthetic val). Three checkpoints (epoch 3/4/7) all ~val 0.668.

**Verdict: ARM-1 is MODESTLY WORSE than baseline — the mix greatly reduced the drift
(18.10 → ~13) but did NOT eliminate it. Synthetic-jitter data is net-harmful at
every mix ratio tried.**

Measured via a **matched per-piece comparison** (apples-to-apples on the 36 pieces
both runs had scored — the running average earlier was flattered by scoring easy
pieces first, so the matched number is the honest one):

| Run | MeanER (matched 36) | Notes |
|---|---|---|
| Baseline (released) | **11.88** | on the same 36 pieces |
| **ARM-1 mixed 80/20** | **13.51 (+1.63 worse)** | last.ckpt |
| (Full-59 baseline) | 11.10 | reference |
| (Synthetic-only FAIL) | 18.10 | reference |

ARM-1 is **consistently worse on the hard repertoire** (Ravel Ondine +6.2, Prokofiev
Toccata +4.5, Liszt Gondoliera +4.3) and only slightly better on a few easy/medium
pieces (Schumann −1.8, Mozart −1.4, Debussy −1.3). The net effect is negative.

> **Honest note:** an earlier draft of this doc called ARM-1 "≈ baseline (~11.3)"
> based on the running partial average. That was wrong — it was biased by the eval
> scoring short/easy pieces first. The matched comparison (+1.63 worse) is correct.

**What this proves (by elimination):**
1. Synthetic-jitter pairs *hurt* a model already trained on ASAP (synthetic-only 18.10).
2. Real-majority mixing *reduces* the harm (13.5 vs 18.1) — confirming the failure was
   data-mix domination, not a code bug — but does NOT remove it.
3. The synthetic-jitter data is **net-negative at every ratio tried** (100% synth,
   20% synth). It adds no signal the released model didn't already have, and its
   distribution mismatch costs more than its (real) NoteDuration signal gains.

### Strategic conclusion
The released Beyer model is **saturated on the public data we can access** (ASAP
paired + MuseScore-derived unpaired scores). Three experiments now agree there is no
quick-experiment win left on this data:
- A1–A5 inference tweaks: null
- Synthetic-only fine-tune: harmful
- Mixed real+synthetic (ARM-1): neutral

**ARM-2 (PDMX as unpaired scores) is NOT recommended** — PDMX is scraped from
MuseScore, the same source as the released model's 58K unpaired scores, so it tests
"more of the data the model already maxed out."

**The only levers with genuine upside require new REAL data:**
- **ATEPP** (verified 2026-06-01): ~5,800 performances with MusicXML, CC-BY-4.0
  (commercial-safe). BUT audio-transcribed MIDI (noisy) + NO note-level alignment →
  needs Parangonar/Nakamura alignment work (days–weeks), not a GPU-hour.
- **PianoCoRe**: 157K real aligned pairs, drop-in — but CC-BY-NC-SA (research-only).

Recommendation: bank the validated tooling + this clean negative/neutral result;
make the ATEPP-ingestion decision deliberately (it's a data-engineering investment,
not a GPU experiment).
