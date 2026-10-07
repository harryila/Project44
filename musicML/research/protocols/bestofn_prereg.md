# Pre-registration: verifier-guided best-of-n, 2026-08-17

Written BEFORE building V3b and BEFORE any N=32 run. All thresholds below are frozen now.

## Protocol

- **Dev pool (tuning allowed):** Scriabin op.8/11 only, N=8, seeds 1000-1007, temp 0.7,
  top-k 15, sampled streams = offset+duration, released ckpt, CPU/FP32. Already decoded;
  candidate MUSTER numbers are known (benchmark/verifier_bestofn_t07_v3.json).
- **True eval (NO tuning):** N=32, seeds 1000-1031, same sampling config (frozen now, no
  further config sweep), pieces = Scriabin, Ravel, Liszt (hard tail) + Mozart K.332
  (do-no-harm control). Verifier logic and all weights frozen before launch.
- **Reporting:** baseline argmax, verifier-pick, oracle, mean-candidate, per piece and
  aggregate; scaling curve at N in {1,2,4,8,16,32} by seed-prefix subsets.

## V3b (the evidence term, hypothesis-test form; fixed design)

Per decoded (measure, quarter) group with >=3 distinct decoded phases and a usable time
anchor (first onset of the NEXT adjacent decoded quarter; groups without an anchor or with
non-monotonic times are skipped): normalize input onset times to the quarter span
u_i = (t_i - t_q0) / (t_anchor - t_q0). Fit u against the duple grid {0, 1/4, 1/2, 3/4} and
the triple grid {0, 1/3, 2/3} (nearest-point mean absolute error each). Evidence margin
m = err_duple - err_triple. Decode-side label: group "says triple" iff it contains any
triplet-phase note. Violations:
- m > theta (evidence favors triple) and decode says duple: penalty m
- m < -theta (evidence favors duple) and decode says triple: penalty |m|
Fixed: theta = 0.02. Verifier total = (V1 + 0.5*V2)/n + w_ev * (sum of V3b penalties)/n_kept,
initial w_ev = 4.0. V1/V2 unchanged from the committed script.

## The ONE permitted adjustment (predeclared)

If the dev gate (below) fails on the first V3b run, exactly one adjustment is allowed:
re-pick w_ev from {1, 2, 8, 16}. No formula changes, no new terms, no theta change, no grid
change, no sampling-config change. After that, verifier logic is FROZEN regardless of outcome.

## Dev gate (pass/fail, frozen)

V3b passes dev iff the verifier-pick on the frozen N=8 Scriabin pool satisfies BOTH:
- tuplet-onset error strictly below baseline (43.59)
- MeanER <= baseline + 1.0 (i.e. <= 11.37)
(cand02 with onT 39.53 / MeanER 10.70 satisfies this; the gate is stated metric-wise, not
candidate-wise.) If dev fails after the one adjustment: STOP, report, do not run N=32.

## True-eval claim thresholds (frozen cliff edges)

**Strong claim** ("verifier-guided selection recovers part of the oracle gap") requires ALL:
1. Aggregate tuplet-onset error over Scriabin+Ravel+Liszt improves by >= 5.0 percentage
   points vs baseline.
2. Aggregate MeanER over those three pieces degrades by <= +0.75 vs baseline.
3. Mozart do-no-harm: MeanER delta <= +0.50; non-tuplet onset error delta <= +1.0 pp;
   decoded triplet count <= 2x GT triplet count (GT ~36).
4. Verifier-pick recovers >= 30% of the oracle gap on the tuplet-onset metric
   (gap = baseline onT minus oracle onT, aggregated over the three hard pieces).

**Fallback claim** (clean negative: likelihood/coherence-selection mismatch + oracle
existence) is what is reported if any threshold fails. Its own requirement: the oracle at
N=32 must beat baseline tuplet-onset by >= 5.0 pp on the hard-piece aggregate; if even the
oracle fails that, the best-of-n direction is reported as exhausted and we STOP (one honest
paragraph, nothing more).

---

# Addendum (2026-08-18): V3b DEAD; fork to V4 local repair. Pre-registered BEFORE any V4 run.

## V3b outcome (kept as ablation/motivation)
V3b FAILED the dev gate at w_ev=4.0 (pick = cand07: onT 42.17 passes, MeanER 11.84 > 11.37
fails) and, by arithmetic over the frozen pool, fails at every permitted w_ev in {1,2,8,16}
(cand02's V1/V2 penalty dominates and its evidence score is not lower than cand07's; the
verifier-pick is cand07 at all permitted weights). Per protocol, no N=32 was run. V3b and the
whole-sequence-selection family are DEAD. The failure is diagnostic: whole-sequence selection
rewards symbolic cleanliness; the real gain in the pool is COVERAGE in hard passages.

## Corrected target metric (frozen)
**TPF (tuplet-passage failures) = missed GT tuplet notes + tuplet onset-error events.**
Verified on the frozen dev pool: baseline 21+34=55, cand02 13+34=47, cand07 16+35=51.
(GT tuplets Scriabin op.8/11 = 99; matched counts from MTD decomposition.)

## V4: LOCAL REPAIR (design frozen now)
Start from the argmax decode; patch only hard windows using sampled candidates as donors;
never touch the rest.
- **Donor pool:** dev = the frozen 8 candidates (seeds 1000-1007, temp 0.7, top-k 15,
  offset+duration). Held-out = N=32 donors, seeds 1000-1031, same config.
- **Hard-window detection (on baseline + donors only, no GT):** per-note disagreement
  d_i = fraction of donors whose offset bucket differs from baseline at note i; evidence flag
  e_i = note lies in a (measure, quarter) whose V3b margin (theta=0.02, same grids) favors
  triple while the baseline group is duple. Window = maximal runs where (d_i >= 0.25 or e_i),
  padded +/-2 notes, min length 3, gaps < 4 merged.
- **Variants:** each donor's (offset, duration, pad) slice over the window; downbeat stays
  baseline. Baseline slice itself always competes.
- **V4 local score (frozen weights):** 2.0 * kept-note fraction in window (pad>0.5)
  + 1.0 * timing-fit = negative mean nearest-grid distance of decoded phases vs input onsets
  normalized on quarter anchors within the window. w_lik = 0.
- **Acceptance:** take the variant with the highest V4 score (ties -> baseline). Compose the
  repaired sequence; ONE MUSTER eval.
- **Global drift gate:** repaired MeanER <= baseline + 0.5, else the whole repair is rejected.

## Dev gate for V4 (frozen)
On Scriabin dev: repaired TPF <= 50 AND repaired MeanER <= 10.87 (baseline + 0.5).
**One permitted adjustment if it fails:** enable the teacher-forced likelihood term
(w_lik 0 -> 1) in the V4 local score. Nothing else. If still failing: STOP (final).

## Held-out protocol (frozen V4, no peeking)
Pieces: Ravel + Liszt (efficacy) and Mozart K.332 (do-no-harm), N=32 donors each.
**Strong claim requires:** relative TPF improvement >= 15% on Ravel+Liszt aggregate;
per-piece MeanER delta <= +0.5; Mozart: MeanER delta <= +0.5, TPF not worsened, decoded
triplet count <= 2x GT (~36).

---

# Addendum 2 (2026-08-18): V10.1 freeze + confirmatory protocol. Written BEFORE the run.

## Status declaration
V5 through V10 were EXPLORATORY: designed and calibrated with eval feedback on the four
development pieces (Scriabin op.8/11, Mozart K.332/12-1, Liszt Gondoliera, Ravel Ondine).
The four-piece result (benchmark/routed_selector_v10_n32_final.json: pooled TPF 2773->2667,
macro MeanER 12.5424->12.5388) is a DEVELOPMENT result. This addendum cannot make V5-V10
pre-registered; it makes the ten-piece run below a legitimate confirmatory test.

## V10.1 freeze (by content hash, 2026-08-18)
- scripts/routed_selector_v10.py  cb9556b057abd85906eafcf66be3a6ef3dd84e911c2a9a3d32c67241b61d230d
- scripts/decode_lever_sweep_v9.py 18fd0c62482dbfd4e806277e23155f75e98e3df480aca4e0c6b4dd0b44fc40ac
- scripts/guarded_selector_v8.py  65e183aace68c385e8a1f5cba58a892942afae9de94c77150b07d5636ed4e0b3
- scripts/likelihood_timing_selector_v6.py 2c7fbf435bbe5eca5035298a71cc0bf44956e280adbea050a6257d295d077fb2
- scripts/local_repair_v4.py      370cf01a2e529b0e314c65a4550033e47231cfdd551ed5a771308e966d2776f4
- scripts/verifier_bestofn.py     eace77dca21dd49b14f92e72be3af89f88c3209eeee0b8bf3bcc0600ca8197dd
- data/duration_priors.pt         7e0b8c2543362e75f70f42b46d93832db983c4f41692320a3771b0cee47710e4
  (fit on ASAP train+val only; leakage-safe)
- MIDI2ScoreTransformer/checkpoints/MIDI2ScoreTF.ckpt 7b8ec6e3da365b97443fb67a8f0b37d63997e93c152d665d43cb2011245db638
- Env: python 3.11.14, torch 2.11.0, CPU, FP32 (autocast disabled), deterministic seeds.
- Parameters (ALL V10 defaults, none changed): n=32 donors (seeds 1000-1031), topk=15,
  temp=0.7, streams=offset+duration; baseline_timing_min=0.001; deterministic branch:
  offset_mode=eighth, offset_lambda=1.0, dur_metrical_lambda=0.25, det_loss_improve=0.01,
  det_min_base_trip=50; sample guards: screen_frac=0.5, max_keep_delta_frac=0.05,
  max_model_delta=0.03, min_model_delta=0.02, target_model_delta=0.025,
  trip_floor_min_base=50, min_trip_ratio=0.5. NOTE: the published four-piece JSON predates
  the det_min_base_trip guard; V10.1 (this hash) is the frozen method.

## The ten untouched confirmatory pieces (every remaining test piece; no subset selection)
Bach Fugue BWV846; Beethoven Sonata 10-1; Brahms op.118/2; Chopin Ballade 1; Debussy
Reflets dans l'Eau; Haydn Sonata 31-1; Prokofiev Toccata; Rachmaninoff Prelude op.23/4;
Schubert Impromptu op.90/1; Schumann Arabeske. These are called the "ten untouched
confirmatory pieces" (NOT an untouched test split: four test pieces became development data).

## Two-phase execution (selection locked before any candidate MUSTER)
- **Phase 1 (select):** for all ten pieces, run V10.1 routing end-to-end WITHOUT any MUSTER
  call: baseline decode, deterministic decode, donors as routed, guard records, final route
  + selected candidate identity, and sha256 of the selected AND baseline decoded streams
  (offset/duration argmax + pad mask bytes). Output benchmark/confirmatory_v10_1_selections.json
  with a UTC timestamp; its sha256 is recorded in the run log. No oracle, no per-candidate
  MUSTER, no GT contact in phase 1 (GT tuplet counts are computed in phase 2 only).
- **Phase 2 (eval):** re-decode baseline + selected per piece, VERIFY stream hashes match
  the locked selections (mismatch = abort), then run MUSTER + TPF on baseline and selected
  only. Oracle and prefix-scaling analyses afterward are SECONDARY and labeled as such.

## Decision rule (frozen; piece-normalized, controls separated)
Tuplet-bearing piece = gt_tuplets >= 20 (determined in phase 2). Control = gt_tuplets < 20.
A no-op route (baseline_easy / baseline_fallback) counts as delta 0.
**"Constructive result confirmed" requires ALL of:**
- P1: >= 3 tuplet-bearing pieces with relative TPF reduction >= 5%.
- P2: mean per-piece relative TPF delta over tuplet-bearing pieces <= -5%.
- P3: no tuplet-bearing piece worsens by more than max(5% relative, 3 absolute) TPF.
- G1: every piece (bearing + control): MeanER delta <= +0.50.
- G2: every control: kept tuplet-phase emissions increase <= 5 notes vs baseline, and TPF
  worsens by <= 1.
**Secondary summaries (reported, never gating):** pooled TPF, macro MeanER, oracle gap,
prefix scaling curve, per-route breakdown. Every piece is reported regardless of outcome.
If the rule fails, the writeup reports the four-piece development result AS development-only
plus this confirmatory failure, and the headline reverts to the diagnosis claims.

---

# Post-run audit correction (2026-08-21; NOT pre-registered)

This section was appended after the confirmation. The file immediately before this append
had sha256 `bece5ab4b62ba6e21bfd986bc0edb9a9a4aebfdeeeee76964f5a9d879fba5af1`.
It does not alter the frozen method, population, selections, or decision rule.

The original phase-two evaluator incorrectly assumed Fmt3x TPQN=24 for every score. The
confirmatory scores use multiple resolutions, so its GT tuplet labels and TPF values are
invalid. That result is preserved at
`benchmark/confirmatory_v10_1_results_invalid_tpqn24.json` with sha256
`85de60af449c9270b84e0f45c3494044ccce729df7bbf705990ca05efb786f9e`.

Phase two was replayed against the unchanged locked selections (sha256
`f1c4a6f184bc05724c4584864440149296f339fa03a1fb931178223c1fd7b9cc`) using exact
MusicXML `<time-modification>` tags joined by Fmt3x GtID. The corrected result is
`benchmark/confirmatory_v10_1_results.json`: P1=false (1 qualifying improvement),
P2=false (mean relative TPF delta +19.39%), P3=false, G1=false, G2=true, and
`CONFIRMED=false`. Pooled TPF is 1457->1553 and macro MeanER delta is +0.4162.

The same correction changes the four-piece exploratory total from the invalid 2773->2667
to 1380->1320; per-piece values are Scriabin 55->47, Mozart 15->15, Liszt 231->191,
Ravel 1079->1067. Full evidence and hashes are in `research/protocols/V10_1_CONFIRMATORY_AUDIT.md`.

---

# Secondary oracle/prefix audit closeout (2026-08-22; NOT pre-registered)

This section was appended after every candidate was evaluated. The file immediately before
this append had sha256
`08765beb2bc14cf22bab020fa7201bbb7fee5fdb51c63d42ce622dd6621f6032`.
It changes no method, selector, sample, candidate order, or confirmatory decision.

## Final exact-metric reconciliation

The exact evaluator now takes matched GT IDs from MUSTER's `ScoreMatchEvaluation`
GT-to-estimate correspondence block. This reproduces the scored `nGT - nMiss` denominator
exactly. The ten-piece confirmatory values and `CONFIRMED=false` decision are unchanged.
The final four-piece exploratory total is 1380->1321 (-59), not 1380->1320: Liszt is
231->192, not 231->191. This one-count correction does not change its exploratory status.

## Frozen N=32 secondary analysis

After phase-two selection was locked and the confirmatory decision was known, all 32 sampled
candidates were evaluated with the corrected metric for each of the eight pieces whose
frozen V10.1 route generated a sample pool. This is a GT-using, post-selection oracle
analysis. It cannot validate the GT-blind V10.1 selector and is never presented as such.

The primary oracle includes the baseline and restricts candidate eligibility to
`MeanER(candidate) <= MeanER(baseline) + 0.50`. Over the six sampled tuplet-bearing pieces:

| Prefix N | TPF baseline | Guarded oracle TPF | Delta | Relative delta | Macro MeanER delta | Pieces improved |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1248 | 1215 | -33 | -2.64% | -0.009 | 1/6 |
| 2 | 1248 | 1215 | -33 | -2.64% | -0.009 | 1/6 |
| 4 | 1248 | 1212 | -36 | -2.88% | +0.055 | 2/6 |
| 8 | 1248 | 1212 | -36 | -2.88% | +0.055 | 2/6 |
| 16 | 1248 | 1140 | -108 | -8.65% | -0.459 | 5/6 |
| 32 | 1248 | 1129 | -119 | -9.54% | -0.558 | 6/6 |

At N=32 every sampled tuplet-bearing pool contains at least one candidate that lowers TPF
within the MeanER guard. The unconstrained baseline-inclusive oracle is 1248->1121 (-10.18%)
with macro MeanER delta -0.005. All 32 Prokofiev control candidates have TPF=0.
Because Prokofiev has zero GT tuplet notes, that TPF value is automatic and is not evidence
about false tuplet emissions.

This distinguishes two hypotheses cleanly: sampling did not fail on these held-out pools;
selection/routing failed to identify the available improvements without GT. That is the
supported sequence-level claim. It is not evidence that the V10.1 repair itself succeeded.
Accordingly, the pre-run anticipated claim that an evidence-conditioned verifier recovers
the oracle gap is rejected.

Canonical result: `benchmark/oracle_prefix_v10_1.json` (sha256
`4bb023cc3714880a480a488e4f185802cc30c05543f7785108f0714e8cc77e59`). Independent
reconstruction: `scripts/audit_oracle_prefix_v10_1.py`. Full audit table and provenance:
`research/protocols/V10_1_ORACLE_AUDIT.md`.
