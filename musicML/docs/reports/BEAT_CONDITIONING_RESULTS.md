# Beat-Conditioning Experiment — Results (2026-06-01, GPU box A100)

**Bottom line (honest):** Beat-conditioning, run as a warm-start fine-tune on ASAP,
does **not** improve accuracy — at best it's neutral, and with the runbook's recipe it
is **destructive**. Even fed **gold** beats (the ceiling), the signal moves the MUSTER
aggregate by ≈0. This is a real negative result that re-scopes the lever: the published
gain (Wachter/Klangio) almost certainly requires **full-regime retraining with beats**,
not a cheap fine-tune. Details below.

## What I ran (end-to-end on the new A100 box)
Full setup from a bare repo: venv + deps (pinned `transformers==4.42.4`/`tokenizers
0.19.1` — the latest 5.x breaks the custom RoFormer config), transferred the released
ckpt + ASAP (967 chunks) from the Mac, warmed the parse-cache (967/967), and validated
the warm-start identity on GPU (beat Linear zero-init, forward/backward OK).

Then two training runs + MUSTER evals on the 59-performance ASAP test split.

## Run v1 — the runbook recipe (lr 3e-4, 12 epochs, ASAP-only) — FAILED
Trained cleanly (beat Linear learned, norm 0.35). But the 3-way eval was decisive:

| config | what | MeanER | OnsetER |
|---|---|---|---|
| **A** | baseline released ckpt | **11.16** | 14.68 |
| **C** | beat ckpt, *no-beat* input | 15.54 | 19.40 |
| **B** | beat ckpt, *gold beats* | 15.44 | 18.99 |

Decomposition (mean over 59 perfs):
- **C−A = +4.39** — the fine-tune **wrecked the base model even without beats**
  (catastrophic forgetting: ASAP is only ~14 distinct pieces, so lr-3e-4×12ep overfit
  and lost the released model's PDMX-pretrained generality).
- **B−C = −0.10** — the gold-beat signal is **~neutral on average**. It *does* help a
  few of the hardest pieces (Ravel *Ondine* 31.79→**22.58**, −9.21; Debussy *Reflets*
  −3.05) but loses on others — no systematic win.
- **B−A = +4.28**, per-piece wins=2 / losses=54. Net clearly worse.

A reproduced published SOTA (11.16 ≈ paper 11.30) → the harness is trustworthy. The
`val/total` creeping up during training (0.674→0.688) was the early warning.

**Why it failed:** the released model's lr-3e-4 peak was for its *full PDMX+ASAP*
training regime. Continue-training a *converged* model at that LR on ASAP-only is
destructive. The runbook recipe was wrong for a warm-start.

## CONFIRMED (v2ep4 3-way eval): beat-conditioning fails as a fine-tune
The least-drifted beat ckpt (v2 epoch-4, val 0.6739 ≈ baseline) on the full 59-perf test:
A baseline **11.16** → C (beat, no-beat) **14.76** → B (beat, gold) **14.54**. So C−A=+3.61
(fine-tune drift persists even at the best val), B−C=−0.22 (gold-beat signal ~neutral),
B−A=+3.38 (wins 7 / losses 30). **The gold-beat ceiling is flat; the chapter is closed.**
Note: val/total≈baseline but MUSTER +3.6 → val-loss is a poor proxy for MUSTER (select on MUSTER).

## Run v2 — corrected, non-destructive (lr 2e-5, 8 epochs) — no aggregate gain
`val/total` per epoch vs baseline **0.6736**:
```
ep0 0.700  ep1 0.711  ep2 0.677  ep3 0.685  ep4 0.674(best)  ep5 0.684  ep6 0.684  ep7 0.683
```
Even at 15× lower LR, **val/total never dips below baseline** — the best epoch (4) only
*matches* it. Beat-conditioning does not reduce aggregate validation loss at any LR I
tried. (A clean 3-way MUSTER eval of the least-drifted epoch-4 ckpt is running to
confirm C′≈A and quantify B′ — the honest "beats neither help nor hurt when the base is
preserved" number, with OnsetER broken out since that's the metric the literature gain
is reported on. Appended when done.)

## Interpretation — why this differs from the published gain
The Klangio result (MUSTER e_onset 12.30 vs Beyer 15.55) added beat-conditioning to a
Beyer baseline and *retrained*, not warm-start-fine-tuned. The most likely reasons my
cheap version doesn't replicate it:
1. **Full-regime training, not fine-tune.** A model trained *from the pretraining stage*
   with the beat input learns to exploit it everywhere; bolting a beat Linear onto a
   converged model and briefly fine-tuning on ASAP-only can't.
2. **ASAP-only is tiny + redundant** (~14 pieces) → fine-tuning overfits/forgets,
   swamping any beat benefit. The released model's strength is PDMX breadth.
3. **Gold-beat ceiling is already ≈neutral** — even *perfect* beats don't help the
   warm-started model, so a real tracker (PM2S) certainly wouldn't. This is a strong
   negative signal for the cheap path.

## Recommendation
- **Do NOT ship beat-conditioning as a fine-tune.** It's neutral-to-harmful here.
- The published gain, if reachable, needs **full retraining with the beat input from the
  PDMX pretraining stage** — a much larger, data-heavy experiment (PDMX manifest + long
  multi-GPU training), with uncertain payoff given the gold-beat ceiling is already flat.
- The **clean-piano audio→score result already beats Songscription's published number**
  (MeanER 2.75 vs 11.30, see SONGSCRIPTION_PARITY.md). The remaining gap is the
  hard-virtuoso tail, which beat-conditioning was meant to address but (as tested)
  doesn't. The tuplet/meter limit looks like a deeper model-capacity issue than a missing
  beat input can fix cheaply.

## Artifacts (nothing committed)
- Box: `MIDI2ScoreTransformer/checkpoints/beat_asap` (v1), `beat_asap_v2` (v2);
  `benchmark/eval_A_baseline.json`, `eval_B_beat_gold.json`, `eval_C_beat_nobeat.json`,
  and the pending `eval_Bp_v2ep4_gold.json` / `eval_Cp_v2ep4_nobeat.json`.
- Eval harness: `benchmark/eval_tier1_asap.py` gained an opt-in `--use-beat-conditioning`
  (threads ASAP gold beats; default-off path byte-identical to the validated baseline)
  and a `--jobs` parallel-scoring path (works serially; the parallel path hit
  fork-after-CUDA issues and is not needed for the conclusion).
- `benchmark/compare_beat_eval.py` — the 3-way decomposition tool.
- `scripts/build_asap_cache.py` — parallel parse-cache warmer (recreated; 64 workers).
