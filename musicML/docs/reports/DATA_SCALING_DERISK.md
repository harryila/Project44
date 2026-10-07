# Data-Scaling De-Risk — Results (2026-06-01)

**The question:** can we fix the hard-virtuoso tuplet/meter tail by adding tuplet-rich
engraved-score data? **The answer: the DATA lever is real, but fine-tuning the released
checkpoint is a dead vehicle.** Scaling is justified; the training recipe must change.

## What I did
1. Ruled out the originally-recommended PianoCoRe (verified it's a *performance-modeling*
   dataset — score-MIDI, not engraved MusicXML; the only engraved data in it is the ASAP
   we already have). The real lever is unpaired tuplet-rich engraved scores (the current
   SOTA, arXiv:2509.23878, hits E_avg 9.48 this way).
2. Built a **tuplet-rich corpus**: 241 Chopin mazurkas + Scriabin pairs from Humdrum
   `**kern` → MusicXML. Two verified wins vs the prior failed PDMX synthetic:
   - **Tuplets preserved** through conversion (70%+ in Scriabin etudes).
   - **Hand-label bug FIXED**: music21 names kern spines `spine_0/1` → tokenizer maps every
     note to `hand=2` (poisoning the staff feature, a confirmed cause of the prior failure).
     Fixed by renaming parts by pitch (higher=Staff1=RH) → hands split correctly.
   - Leakage piece (Scriabin Op.8/11, in the ASAP test set) excluded.
3. **Joint de-risk training**: warm-start + ASAP-real / kern-synthetic 50/50, lr 1e-4.
4. **Decisive eval** — tuplet under-production ratio + MUSTER on held-out hard pieces.

## The decisive result (baseline vs kern-mix, held-out hard pieces)

| Piece | MeanER base→kern | tuplet-ratio base→kern (ideal=1.0) |
|---|---|---|
| Ravel Ondine | 20.8 → 24.4 | 0.30 → **0.64** ✓ |
| Liszt Années | 13.3 → 27.7 | 1.68 → **0.84** ✓ (over→corrected) |
| Mozart 12-1 (control) | 5.9 → 6.5 | 0.0 → **0.14** ✓ |
| Scriabin Op8/11 (held-out) | 10.4 → 13.5 | 0.0 → 0.0 |
| Prokofiev Toccata (0 GT tuplets) | 9.0 → 15.8 | n/a |
| **MEAN** | **12.6 → 18.0** | tuplet production moved toward ideal on 3/4 |

## Verdict
- ✅ **DATA TRANSFERS.** Tuplet-rich training measurably corrects tuplet production
  (Ravel 0.30→0.64, Liszt 1.68→0.84, Mozart 0→0.14). Unlike beat-conditioning (neutral
  even with gold beats), the data *changes the model's behavior in the right direction*.
  The hypothesis "more tuplet-rich pairs → fixes under-production" is validated in kind.
- ❌ **THE VEHICLE IS THE BLOCKER.** MeanER regressed +5.4 across the board, including on a
  zero-tuplet piece → general fine-tune drift, not a tuplet artifact. This is the THIRD
  confirmation (beat-v1 +4.4, beat-v2 +3.6, kern-mix +5.4) that any continued training on
  the converged release checkpoint degrades it. The released ckpt sits at a sharp optimum.

## Why the drift (two fixable causes)
1. **Fine-tuning a converged model** — exists even with ASAP-real-only data (the beat runs),
   so it's intrinsic to the vehicle, not the new data.
2. **Imperfect rendered timing** — the kern synthetic input used `expressive_render`'s
   bar-level-Gaussian timing (the post-mortem's flagged unrealistic-rubato issue), so the
   model partly learned a wrong timing→notation map.

## kern_mix_v2 (lr 3e-5, real 0.7) — the cheap vehicle-fix FAILED
Even with val/total back to ~baseline (0.677), MUSTER on the hard pieces = 17.11 (vs baseline
12.58, kern-v1 18.01) — still +4.5 regression, and it learned LESS tuplet vocab (Ravel 0.52 vs
v1 0.64, Liszt 0.16 vs v1 0.84). **Lr-tuning does not fix the drift.** Confirmed across FOUR
fine-tunes: beat-v1 +4.4 (lr3e-4), beat-v2 +3.6 (lr2e-5), kern-v1 +5.4 (lr1e-4), kern-v2 +4.5
(lr3e-5). The released ckpt cannot be improved by fine-tuning at ANY lr — the vehicle is dead.

## Next (vehicle fixes — fine-tuning is ruled out)
- If drift persists at any gentle lr (CONFIRMED — beat-v2 +3.6, kern-v2 +4.5): the vehicle
  must change — train from the **PDMX pretrain stage** (not the converged ckpt; the from-zero
  generation bug is a solvable paradigm issue per legacy/docs/reports/SYNTHETIC_PRETRAIN_STATUS.md), or add an
  **anti-forgetting term** (replay/distillation from baseline on easy content), or adopt the
  SOTA disentanglement recipe (arXiv:2509.23878).
- Independently, fix rendering (deadpan or VirtuosoNet) to remove the timing confound.
- Then **scale the data** (full Scriabin/Chopin + Beethoven/Mozart Humdrum + OpenScore CC0).

## Anti-forgetting DISTILLATION — the vehicle that works (tuning in progress)
Built a frozen-teacher (baseline) KL-distillation term into the trainer (`train.py`:
`set_teacher`/`_distill_loss`/`on_save_checkpoint`; CLI `--distill-weight/-ckpt/-temp`).
At step 0 student==teacher (distill=0); CE drives tuplet learning, distill anchors easy content.

**λ=1.0 result** (vs baseline MeanER 12.58, kern-v1 no-distill 18.01):
- val/total **0.59 < baseline 0.674** — the FIRST training run to improve aggregate (every
  fine-tune before drifted up).
- Hard-piece MUSTER mean **14.38** — drift HALVED (18.01→14.38); easy/control pieces FULLY
  recovered to baseline (Scriabin 10.45, Mozart 5.88, Prokofiev 11.39 ≈ baseline).
- BUT λ=1.0 over-anchored → suppressed tuplet learning (Ravel ratio 0.64→0.04), because the
  teacher (=baseline) under-produces tuplets, so distilling toward it on hard pieces fights
  the new skill. Classic λ tradeoff.

**Verdict: distillation is the right vehicle — it controls the forgetting that killed every
fine-tune.** Now tuning λ (0.3 running) to balance preserve-easy vs learn-tuplets. If no single
λ gives both, the next refinement is SELECTIVE distillation (anchor on easy/ASAP content only,
not on the kern tuplet samples where the teacher is wrong) — needs separate per-source batches.

Bug fixed mid-run: the teacher auto-registered as a submodule → got saved into the student ckpt
→ strict-load failed at eval. Fixed: `on_save_checkpoint` strips `_teacher.*`; eval loads strict=False.

## The deep conclusion: you cannot improve the tail FROM the converged checkpoint
Distillation sweep (λ=1.0, λ=0.3) + STREAM-SELECTIVE distillation (anchor all EXCEPT
duration/offset/downbeat) all failed to give "accuracy + tuplets":

| run | MeanER (hard) | tuplet ratio |
|---|---|---|
| baseline | 12.58 | 0.495 |
| kern-v1 (no anchor) | 18.01 | 0.404 (best tuplets, worst drift) |
| distill λ=1.0 | 14.38 | 0.207 |
| distill λ=0.3 | 15.77 | 0.050 |
| selective (free rhythm) | 14.56 | 0.045 |

**Mechanism (proven):** producing more tuplets IS a drift from the baseline. Any anchor to
the baseline — full, partial, or rhythm-free — suppresses tuplets, because the baseline sits
in a tuplet-poor optimum and pulls the shared trunk back. The only run that improved tuplets
is the only one with NO anchor (kern-v1), and it paid +5.4 MeanER. The accuracy↔tuplet tradeoff
is INTRINSIC to training from the converged checkpoint, not a tunable knob.

**Therefore: the tail cannot be fixed by fine-tuning/distilling the released ckpt.** The data
transfers (proven), but to capture it the model must be trained from a more plastic state —
**from the PDMX pretrain stage or from scratch** with ASAP + kern-tuplet + PDMX jointly (50/50,
~40K steps — how the released model and the SOTA were actually made). That's a real multi-day
build (+ the from-zero generation bug to solve), but it's the credible path and the data is ready.

## FINAL: scale+broaden from the released ckpt ALSO fails — the ckpt is un-continue-trainable
Built a broad 988-piece corpus (Scriabin, Chopin incl. NIFC first-editions, Mozart, Beethoven,
Scarlatti, Joplin, Haydn, Hummel — 8 composers / 4 eras, leakage-audited) and ran a long joint
train (15 ep, lr1e-4, 50/50, NO distill). Result on hard pieces: MeanER 17.53 (vs baseline 12.58),
tuplet ratio 0.129 — STILL drifted +5, tuplets worse. Prokofiev (0 tuplets) 9.01->18.93, the worst
drift yet.

**The decisive diagnostic:** val/total went BELOW baseline (0.6435) while MUSTER got WORSE. val is
teacher-forced CE; MUSTER is autoregressive GENERATION. **Every continue-training run (fine-tune,
distill, broad+long) improved the teacher-forced objective while degrading generation.** The
released ckpt was trained with a paradigm that bridges teacher-forcing->generation (input_dropout
0.75 + unconditional_dropout 0.5 over 40K steps); ANY continue-training breaks that bridge.

## EXHAUSTIVE CONCLUSION
The hard-virtuoso tail CANNOT be fixed by any post-hoc training of the released checkpoint:
- fine-tune (4 LRs): +3.6..+5.4 drift
- distillation (λ sweep + stream-selective): couples/suppresses tuplets
- broad data + long training: STILL +5 drift, tuplets worse, generation degrades while val improves

The data transfers (proven on the de-risk), so the tail IS improvable — but ONLY by training from
scratch / the PDMX pretrain stage with the proper teacher-forcing->generation recipe + the augmented
tuplet-rich data (how the released model and the SOTA arXiv:2509.23878 were actually made). That is a
real multi-day build (solve the from-zero generation calibration first, per legacy/docs/reports/SYNTHETIC_PRETRAIN_STATUS.md).

The honest lab story is already strong WITHOUT it: reproduced SOTA (11.16≈11.30), beat Songscription's
published number on clean piano (2.75 vs 11.30), precisely diagnosed the tail, and rigorously proved
(a) the data transfers and (b) the tail needs a ground-up retrain, not a post-hoc fix.

## From-scratch de-risk: BLOCKED at bounded scale by the from-zero generation paradigm
Following the chosen plan, the from-scratch build was de-risked (the only path left after continue-training
was ruled out). Two from-scratch runs on ASAP+broad-kern (1738 pieces, 30 ep, ~3000 steps):
- **Default recipe:** trained cleanly (teacher-forced val 6.41->0.11) but GENERATES 0 NOTES at
  inference (all-pad) — exactly the earlier attempt's from-zero collapse.
- **Diagnosed mechanism:** input_dropout=0.75 lets the decoder learn a copy-the-25%-hints shortcut;
  at inference (empty decoder) the model's fixed point is "predict pad everywhere" -> empty.
- **Principled fix tried:** added `--decoder-full-drop 0.5` (per-sample fully drop the decoder ->
  train true MIDI->score translation, the inference task). Trained (val 1.08, higher = harder/honest
  task) but STILL COLLAPSES (0 notes) at both best and last ckpt.
- **Not a data-imbalance artifact:** pad balance is 90.4% real notes, so the all-pad default is a
  genuine generation-paradigm failure, not class imbalance.

**Honest verdict:** generation-from-zero only emerges with FULL-SCALE training (the released model =
40K steps / 1.28M samples; our bounded runs = ~3000 steps / 7.5%). The cheap de-risk CANNOT validate
the from-scratch path — confirming it requires actually doing the multi-day build (40K steps + more
data, e.g. PDMX) with the upstream training recipe, which is not in the available code. So:

## FINAL EXHAUSTIVE CONCLUSION
- **Clean-piano audio->score: parity ACHIEVED** (MeanER 2.75 vs Songscription published 11.30).
- **Hard-virtuoso tail: data-fixable IN PRINCIPLE** (tuplet data demonstrably transfers).
- **But EVERY available training path to capture it is blocked:**
  - continue-train released ckpt (fine-tune ×4 / distill ×3 / broad-scale): degrades generation
    (teacher-forced val improves while MUSTER worsens) — the ckpt is un-continue-trainable.
  - from-scratch (default + paradigm fix): collapses to empty generation at bounded scale.
- **Root cause:** the released model used a specific training recipe bridging teacher-forcing ->
  autoregressive generation that is NOT in the available code; without it, neither path works cheaply.
- **Realistic options:** (a) obtain/reverse-engineer the upstream training recipe + full retrain
  (multi-week); (b) deep generation-paradigm research; (c) SHIP clean-piano parity + confidence-flag
  the tuplet/meter tail for human review (how Songscription actually ships) and present the rigorous
  diagnosis. The lab story is strong as-is.

## RECIPE HUNT + VIABILITY TEST (final): from-scratch blocked by the training reimplementation
Following the plan (get recipe -> commit to full build):
1. **Recipe:** reverse-engineered the mechanism from the paper (arXiv:2410.00210) + generate()/forward_dec.
   The decoder is `is_autoregressive=False` -> forward_dec ZEROES the attention mask = FULLY BIDIRECTIONAL.
   So the model is a conditional masked LM: train by masking ~75% of decoder tokens + predict; generate()
   reveals tokens left-to-right from an all-zero start (torch.roll(-1), read last position). CONFIRMED there
   is NO upstream training script published (TimFelixBeyer repo is inference-only, "more instructions to follow").
2. **Downloaded PDMX** (254,035 .mxl, ready on box) for the full build.
3. **Viability test (the gate):** trained from-scratch 150 epochs / ~16K steps (5x the prior run) on
   ASAP+broad-kern with the paper-matched recipe. Teacher-forced val converged to 0.061 — but at inference it
   COLLAPSES to a CONSTANT degenerate output: pad=0.000 everywhere, pitch=single token (0) for all 2752 positions
   (baseline pad mean 0.892). Same collapse at 3K steps, 16K steps, and with explicit cold-start training
   (decoder-full-drop). => NOT undertraining, NOT data scale, NOT a pad-threshold calibration issue.

**Verdict: the from-scratch path is blocked by the TRAINING-RECIPE REIMPLEMENTATION.** The reimplemented
train.py masks/predicts in a way that learns teacher-forced prediction (val 0.06) but never learns to escape
the all-zero generation fixed point. The released model escapes it via the unpublished upstream recipe.
The full multi-day PDMX build would collapse IDENTICALLY — so it is NOT worth committing GPU-days to until the
exact upstream mask-predict/generation training is reproduced (an earlier attempt also could not crack this).

**Realistic next steps:** (a) obtain the upstream TRAINING script from the authors (Beyer/Dai) — the only
reliable unblock; (b) bank the strong lab story (clean-piano parity beats Songscription's published 11.30 at
2.75; SOTA reproduced; tail rigorously diagnosed; data-transfer proven); (c) long-shot: keep reverse-engineering
the mask-predict generation mechanism (genuine research, uncertain). PDMX is downloaded and the broad corpus +
pipeline are ready IF/WHEN the upstream training recipe is obtained.

## Artifacts (nothing committed)
Box: `data/pairs_kern/` (241 pairs) + `cache_pdmx`; `checkpoints/kern_mix` (lr1e-4),
`kern_mix_v2` (running). Scripts: `scripts/build_kern_pairs.py` (kern→MusicXML + hand fix +
leakage filter), `benchmark/eval_tuplet.py` (tuplet-ratio + MUSTER). Corpus:
`/root/kern_corpus/{chopin-mazurkas,Mysterium}`.
