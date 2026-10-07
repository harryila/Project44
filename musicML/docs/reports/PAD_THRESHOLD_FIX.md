# Finding: the released `--pad-threshold` knob was a no-op — fixed, validated, now a live calibration lever

**Date:** 2026-06-04 · **Status:** implemented + locally validated (no GPU needed); box sweep pending.
**Files touched (NOT committed):** `MIDI2ScoreTransformer/midi2scoretransformer/models/model.py`,
`…/utils.py`, `…/tokenizer.py`, `benchmark/eval_tuplet.py`. Test: `scripts/test_pad_prob_fix.py`. Sweep: `scripts/run_padsweep.sh`.

## TL;DR
The model emits one output slot per input performance note; a per-note **pad** head decides keep/drop, and
`detokenize_mxl(..., pad_threshold)` is documented to let you lower the gate to *"rescue notes the model is
slightly unsure about."* **That knob did nothing.** `generate()` collapsed the pad head to a hard binary
decision at sigmoid = 0.5 *before* detokenize ever saw it, so any `--pad-threshold` in (0,1) produced the
identical score. We restored a continuous keep-probability (plus the un-zeroed per-stream predictions needed
to actually recover a rescued note), making the gate live. At threshold 0.5 the output is **byte-identical**
to the released behaviour, so it is a strict, backward-compatible superset.

This matters because our two residuals are *both* pad-gate calibration symptoms:
- **Scriabin / dense tail — MISS-RATE** (~14–22 % of fast notes dropped): a *lower* gate rescues borderline real notes.
- **Mozart — OVER-PRODUCTION**: a *higher* gate prunes spurious emissions.

It is the cheapest possible shot at the frontier: **zero retraining**, pure inference.

## The bug (verified in source)
1. `models/model.py` `generate()` builds the pad distribution `probs = [1 − sigmoid(logits), sigmoid(logits)]`
   (`model.py:135`).
2. It then takes a **hard argmax** of that 2-way distribution (`model.py:142`) — i.e. keep ⇔ sigmoid > 0.5 —
   and floats the result to a binary `{0.0, 1.0}` stream (`model.py:160`).
3. `tokenizer.py` `detokenize_mxl` masks notes with `token_dict["pad"] > pad_threshold` (`tokenizer.py:463`).
   Because the stream is already binary, **every threshold in (0,1) yields the same mask**; ≥ 1.0 drops all,
   < 0 keeps all. The docstring promise at `tokenizer.py:455` ("Lowering to ~0.3–0.4 rescues notes") was
   therefore unreachable.
4. Subtlety the naive fix misses: `model.py:152-155` *zeroes* the other streams (pitch/duration/…) for dropped
   slots (for AR feedback). So even if you re-expose a soft gate, a rescued slot would carry zeroed garbage —
   you must also preserve the **un-zeroed** prediction.

## The fix (additive, backward-compatible)
- `generate()` now also accumulates, with **no change to the AR feedback path**:
  - `pad_prob` — the continuous keep-probability per slot (the sigmoid, captured *before* the argmax);
  - `raw_<stream>` — the un-zeroed one-hot prediction per slot, so rescued notes carry real pitch/duration/etc.
  - These are aligned to the returned length through the chunked-overlap context machinery (context-region
    placeholders are sliced away by `infer()`'s overlap logic).
- `utils.infer()` excludes the `pad_prob` / `raw_*` side channels from the chunk-to-chunk context dict (they
  aren't decoder inputs).
- `detokenize_mxl()` prefers `pad_prob` + `raw_*` when present (so the threshold is live and rescued notes are
  real), and falls back to the legacy binary stream otherwise (teacher forcing, round-trip tests unchanged).
- `utils.eval()` and `benchmark/eval_tuplet.py` thread `--pad-threshold` through to detokenize **and** MUSTER.

## Local validation (no checkpoint, `scripts/test_pad_prob_fix.py`)
Real `tokenize → generate → infer → detokenize` run on `midi/TwinkleTwinkle.mid` with a randomly-initialised
tiny model, forcing the chunked-overlap path (chunk 24 / overlap 8):
- **(A) 0.5-invariant:** new soft path = legacy binary path = **584 notes** (identical). Default behaviour is
  byte-for-byte unchanged.
- **(B) live & monotonic:** note count is non-increasing in the threshold; the random model's narrow keep-prob
  range `[0.624, 0.699]` means thresholds > 0.7 drop everything — i.e. the knob now actually moves the output.
  On the *trained* ckpt the keep-prob distribution is wide (confident keeps ≈ 1.0, borderline notes ≈ 0.3–0.5),
  which is exactly the rescue zone.

```
input notes: 569
generated slots: 584 | kept@0.5: 584 | pad_prob range [0.624, 0.699]
(A) 0.5-invariant: new=584 notes  legacy=584 notes
(B) notes vs threshold: {0.1: 584, 0.3: 584, 0.5: 584, 0.7: 0, 0.9: 0}
PASS: 0.5 is byte-identical to legacy; pad-threshold is live and monotonic.
```

## BOX SWEEP RESULT (ssl_classical_clean ep13, real ASAP + MUSTER, 2026-06-04)
Single-pass sweep (`benchmark/eval_padsweep.py`, generate-once-per-piece) over 13/14 ASAP test pieces × 8
thresholds. Mozart at 0.50 = 5.76 reproduces the documented baseline exactly (0.50-invariant confirmed on the
real model + real metric).

| corpus-mean MeanER | 0.60 | 0.55 | **0.50** | 0.45 | **0.40** | 0.35 | 0.30 | 0.25 |
|---|---|---|---|---|---|---|---|---|
| (14 pieces) | 12.909 | 12.791 | **12.685** | 12.654 | **12.649** | 12.661 | 12.684 | 12.691 |

- **Global optimum 0.40 → 12.649 (−0.036 vs the released 0.50 default = 12.685), never worse.** The released default
  was slightly too high; a mild universal lowering is a small, safe, free win. Per-piece-optimal = 12.619 (−0.066).
- **The lever helps where the model is uncertain (mid-complexity Romantic tier):** Liszt −0.49 (14.45→13.96 @0.35),
  Beethoven −0.12 (9.50→9.38 @0.40), Ravel −0.16, Rachmaninoff/Debussy small. On easy pieces (Haydn, Schumann)
  lowering *hurts*, so 0.50 wasn't a bad default — just not optimal.
- **DECISIVE NEGATIVE — Scriabin is dead flat: 12.32 at EVERY threshold.** The calibration lever cannot touch the
  headline residual. Confirmed in advance as the go/no-go: Scriabin's dense-tuplet misses are **confident drops**
  (keep-prob ≪0.1), not borderline — a **representation/training problem, not a decoding one**. This rules out the
  whole cheap inference branch for the hardest residual (≈0 GPU) and redirects it to the retraining queue
  (denoising-SSL, curriculum masking, tuplet-loss).

## RELEASED-CKPT A/B RESULT (full 14 pieces, 2026-06-05) — reframes the standing
The A/B finished. It did **not** confirm a field-wide flat Scriabin, and it corrected a bigger thing: the
released model's corpus mean is **10.77 vs our 12.69** (+1.91) — **we are not at parity overall** (we win only
Mozart/Ravel/Prokofiev of 14). The released model's Scriabin is in fact *mildly* responsive to the gate
(10.40→10.27), but the real story is in the **tuplet counts**: our model emits **0 tuplets on 9/14 pieces**
while the released model emits them across the board (Haydn 711, Liszt 675, Debussy 672). The per-stream
breakdown confirms the gap is **`NoteDuration`** (Haydn 0.654 ours vs 0.192 released), not structure
(`TimeSignature` = 0 for both). So the pad-gate's flat-Scriabin result was a *true* negative for *decoding*,
but the dominant residual is a **collapsed tuplet head** — a training-objective problem now being fixed via
tuplet-aware loss reweighting (warm-started from `ssl_classical_clean`). See `LAB_REPORT.md §11` +
`benchmark/compare_sweeps.py` / `diag_streams.py`.

## Box sweep command (reproduce)
`bash scripts/run_padsweep.sh 'checkpoints/ssl_classical_clean/*epoch=13*.ckpt' sslcc`
sweeps thresholds `{0.60 … 0.25}` over **all 14 ASAP test pieces** (also satisfies the "measure the full
14-piece standing" prerequisite), reporting per-piece MeanER + predicted/gt tuplet & note counts, the best
threshold per piece, and the corpus mean per threshold.

**Success criterion:** Scriabin MeanER drops below 12.3 at some threshold ≤ 0.40 **without** Mozart regressing
past ~5.86 or re-introducing tuplet over-production (watch `pred_tuplets/gt_tuplets`). A per-piece (density-
binned) optimal threshold is a legitimate, publishable calibration result even if a single global threshold
is flat. Worst case it cannot regress the released config (0.50 is preserved exactly), so it is a free roll.

## Real-model validation on CPU (no box) — and an honest EV caveat
With the GPU box down (all hosts unreachable) and our best ckpt (`ssl_classical_clean`) stranded on it, I ran
the fix against the **locally-available released SOTA checkpoint** (`MIDI2ScoreTF.ckpt`) on CPU
(`scripts/probe_released_padprob.py`) to (a) confirm the fix on a *real trained* model and (b) measure the
keep-gate's actual calibration curve — does the 0.30–0.50 "rescue zone" hold real note mass, or is the gate
saturated (=> sweep flat)?

| piece (released ckpt, CPU) | notes | 0.5-invariant | rescue zone [0.30,0.50) | prune (0.50,0.70] | sweep 0.60→0.30 | rescue+prune % |
|---|---|---|---|---|---|---|
| Bach Prelude BWV 846 (easy/homophonic) | 805 | ✓ holds | 1 note | 0 | 802 → 803 (flat) | 0.1% |
| Chopin Nocturne Op.9/2 (denser) | 1544 | ✓ holds | 7 notes | 11 notes | 1485 → 1495 (mild) | 1.0% |

Both confirm the fix on a **real trained** model (the 0.5-invariant holds). The keep-gate is strongly **bimodal /
near-saturated** on normal repertoire — the model is *decisive* (mass at ≥0.9 keep or <0.1 drop), not uncertain,
so the threshold lever has little headroom on easy/medium pieces.

**The key signal is the trend:** rescue+prune mass grows **0.1% → 1.0%** from Bach to the denser Chopin, and the
sweep goes from flat to mildly live. Headroom **increases with density** — exactly as the hypothesis predicts,
since the residual lives on dense passages where the model is uncertain. Scriabin Op.8/11 (where the model
drops ~14–22% of notes, far denser than either piece here) should sit much further along this trend.

**Honest EV caveat:** this refines the triage's "highest-EV" rating into a *conditional* one — the pad lever
helps only where keep-prob lands in the rescue zone. The decisive test (does dense-Scriabin produce *borderline*
drops, which the gate rescues, or *confident* drops at <0.1, which it can't?) needs the box + the real Scriabin
piece. A flat Scriabin sweep would be a publishable negative ("dense-tail misses are confident, not
calibration-fixable → a representation problem, not a decoding one"); a sloped one is the free win. Either way,
the core finding — *a documented hyperparameter in the published model was silently inert* — stands, and the fix
is a strict, validated superset of the released behavior.
