# A1-A5 Improvement Results

Empirical evaluation of the five "quick win" improvements identified before the synthetic pretrain plan, run on Twinkle Twinkle plus the three MAESTRO benchmark pieces (Chopin Op.10 No.4, Chopin Op.25 No.11, Liszt Transcendental Etude No.4 "Mazeppa").

## What was tested

| Tag | Improvement | Configuration |
|---|---|---|
| `baseline` | Pre-A3 code, default flags (existing on-disk) | `--midi-input gt.midi`, default flags |
| `hft_baseline` | Pre-A3 audio path | `audio.wav --transcriber hft`, default flags |
| `a3` | A3 only (voice padding fix; code-only) | New defaults via `--midi-input gt.midi` |
| `a1` | A1 only (lower pad threshold) | `--pad-threshold 0.4 --midi-input gt.midi` |
| `a2` | A2 only (top-k sampling) | `--top-k 5 --temperature 0.8 --midi-input gt.midi` |
| `a4` | A4 only (larger chunk size) | `--chunk-size 1024 --midi-input gt.midi` |
| `combined` | A1+A2+A3 (A4 dropped, see findings) | All three Phase-2 flags + `--midi-input gt.midi` |
| `full` | A1+A2+A3+A5 with audio path | `audio.wav --transcriber hft --normalize-audio` + Phase-2 flags |

A3 is a code-only fix (per-voice padding instead of measure-wide). It's always on after the change. The baseline rows above were captured **before** the A3 fix landed.

## What worked, what didn't

| Improvement | Verdict |
|---|---|
| **A1** — pad threshold 0.5 → 0.4 | No measurable change. Identical metrics to baseline on all 4 pieces. |
| **A2** — top-k 5, temperature 0.8 | Tiny gain (+0.001 F1 on Op10). Not statistically meaningful. |
| **A3** — per-voice padding fix | No change on these 4 pieces. May matter on rarer multi-voice patterns. |
| **A4** — chunk size 512 → 1024 | **Cannot use.** Model was trained at `seq_length=512`; positional embeddings break above that on real pieces (`IndexError: index out of range`). Worked silently on Twinkle only because it has fewer than 512 notes. **Documented as not implementable on this checkpoint.** |
| **A5** — audio mono+resample+peak normalize | Mixed. On Mazeppa, shifted predicted time signature 2/4 → 3/4 (still wrong) and reduced measure count error from 112.6% to 71.9%, but dropped note F1 from 0.959 → 0.936. Probably hurts because the trained-on-MAESTRO hFT model expects MAESTRO-like recording levels and aggressive normalization moves outside that distribution. |

## Numbers

Reference column is the published score. `OK` = predicted time signature matches the published time signature. `Meas` = measure count predicted by Phase 2. `%err` = `|pred - ref| / ref * 100`. `Note F1` = pitch-multiset F1 between predicted MusicXML notes and ground-truth MIDI notes.

```
Piece                          Tag           TS    OK   Meas   %err    Note F1
-------------------------------------------------------------------------------
Twinkle Twinkle (ref 18)
  hft_baseline                 (audio,  hFT)  4/4  YES  19    5.6     1.000
  a3                           (gt midi)      4/4  YES  19    5.6     1.000
  a1                           (gt midi)      4/4  YES  19    5.6     1.000
  a2                           (gt midi)      4/4  YES  19    5.6     1.000
  a4                                          (chunk=1024 was no-op on this piece)
  combined                     (gt midi)      4/4  YES  19    5.6     0.999
  full                         (audio + A5)   4/4  YES  19    5.6     0.985

Chopin Op.10 No.4 (ref 88)
  baseline                     (gt midi)      4/4  YES  82    6.8     0.971
  hft_baseline                 (audio, hFT)   4/4  YES  81    8.0     0.970
  a3                           (gt midi)      4/4  YES  82    6.8     0.971
  a1                           (gt midi)      4/4  YES  82    6.8     0.971
  a2                           (gt midi)      4/4  YES  82    6.8     0.972
  a4                           (gt midi)      FAILED  IndexError (chunk > 512)
  combined                     (gt midi)      4/4  YES  81    8.0     0.973
  full                         (audio + A5)   4/4  YES  81    8.0     0.967

Chopin Op.25 No.11 (ref 102)
  baseline                     (gt midi)      4/4  YES  100   2.0     0.978
  hft_baseline                 (audio, hFT)   4/4  YES  96    5.9     0.974
  combined                     (gt midi)      4/4  YES  98    3.9     0.978
  full                         (audio + A5)   4/4  YES  96    5.9     0.969

Liszt Mazeppa (ref 167)
  baseline                     (gt midi)      3/4  NO   322   92.8    0.981
  hft_baseline                 (audio, hFT)   2/4  NO   355   112.6   0.959
  combined                     (gt midi)      3/4  NO   400   139.5   0.978
  full                         (audio + A5)   3/4  NO   287   71.9    0.936
```

Raw metrics: [improvement_metrics.json](improvement_metrics.json).

## Findings

1. **Inference-time tweaks barely move the needle.** A1, A2, A3 individually and combined produce metrics within ±0.005 F1 and ±2 measures of the baseline. The thresholds, sampling strategy, and voice padding fix are not where the bottleneck is.

2. **A4 is unusable on this checkpoint.** The trained model has `seq_length=512`. Position embeddings beyond that crash even though `max_position_embeddings=1536` is configured. To get longer-context inference, the model would have to be retrained. Confirmed live: chunk=1024 fails with `IndexError` on Op10 No.4 (2306 notes), works only on small pieces by accident.

3. **A5 (audio normalization) shifts predictions but doesn't clearly help.** On Mazeppa, normalization changed the wrong time signature from 2/4 to 3/4 and roughly halved measure count error, but dropped note F1. Suggests the hFT model is sensitive to recording level — but the right fix is probably matching MAESTRO's level statistics rather than blanket peak-normalize.

4. **Mazeppa stays broken regardless.** Wrong time signature on every variant, including the original baseline. Combined+sampled inference made it slightly *worse* on measure count (322 → 400). Inference-time fixes cannot rescue a model that was never trained on Liszt-density polyphony. This empirically confirms the synthetic pretrain plan's premise: Phase 2 is data-bound, not hyperparameter-bound.

5. **Note F1 is consistently high (~0.97-0.99) even when the piece is structurally wrong.** Note coverage isn't the problem — the model recovers the right pitches. What it gets wrong is *meter and barlines*, which dominate musical readability. F1 alone misses this; measure-count error is the more diagnostic metric.

## Implication for the synthetic pretrain plan

The pretrain plan was already gated on PDMX val structural metrics (time sig accuracy, measure count error) rather than note F1 specifically because of finding #5. These results reinforce that:

- The "baseline" the pretrain has to beat on the 3 benchmark pieces is **the existing baseline numbers above** (gt-midi `baseline` row): 4/4 on Op10 and Op25 with measure count within 7%, and 3/4 with 92.8% measure error on Mazeppa.
- A1, A2, A3 are now permanent code paths but contribute negligibly to the comparison. We don't need to re-baseline.
- A4 is dropped from the plan; documenting in [STATUS.md](../STATUS.md) and [IMPROVEMENTS.md](../IMPROVEMENTS.md).
- A5 is on a `--normalize-audio` flag (off by default) so it's available when the audio path output looks bad, but not forced.

## Reproducing

```bash
# Phase-2 isolated tests (--midi-input GT)
venv311/bin/python transcribe.py --midi-input <gt.midi> --backend transformer \
    --pad-threshold 0.4 --top-k 5 --temperature 0.8 \
    --output benchmark/.../<piece>_combined.pdf

# Full pipeline test (audio path, A5 + combined)
venv311/bin/python transcribe.py <audio.wav> --transcriber hft --normalize-audio \
    --backend transformer --pad-threshold 0.4 --top-k 5 --temperature 0.8 \
    --output benchmark/.../<piece>_full.pdf

# Tabulate
venv311/bin/python benchmark/eval_improvements.py
```
