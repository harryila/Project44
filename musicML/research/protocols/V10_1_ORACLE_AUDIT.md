# V10.1 Frozen-Pool Oracle and Prefix Audit

**Audit date:** 2026-08-22  
**Status:** Complete secondary analysis. The frozen V10.1 selector remains unconfirmed.  
**Question:** Did sampling fail on the held-out pieces, or did GT-blind selection fail to
identify better candidates that were already present?

## Answer

Sampling succeeded on every sampled tuplet-bearing held-out piece. At N=32, all six pools
contain at least one candidate with lower TPF and
`MeanER(candidate) <= MeanER(baseline) + 0.50`. The guarded baseline-inclusive oracle lowers
pooled TPF from 1248 to 1129 (-119, -9.54%) while macro MeanER changes by -0.558. V10.1 did
not select those candidates.

This supports a sequence-level existence and selection-gap claim. It does not rescue the
constructive selector claim: the oracle uses GT after selection, and the frozen V10.1
confirmatory decision remains `CONFIRMED=false`.

## N=32 piece audit

| Piece | Route | Baseline TPF | V10.1 TPF | Guarded oracle | Candidate | Delta TPF | Delta MeanER | Qualifying candidates |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Haydn | best-of-N | 41 | 92 | 34 | 15 | -7 | -0.291 | 3 |
| Brahms | baseline fallback | 16 | 16 | 13 | 8 | -3 | -0.518 | 1 |
| Debussy | best-of-N | 411 | 439 | 351 | 27 | -60 | -2.768 | 10 |
| Beethoven | baseline fallback | 201 | 201 | 168 | 0 | -33 | -0.055 | 18 |
| Schubert | best-of-N | 224 | 258 | 212 | 13 | -12 | +0.132 | 2 |
| Chopin | best-of-N | 355 | 366 | 351 | 31 | -4 | +0.151 | 1 |

`Qualifying candidates` counts sampled candidates that beat baseline TPF while satisfying
the +0.50 MeanER guard. Candidate IDs follow the frozen seed order, `seed = 1000 + ID`.

## Prefix scaling

The primary curve includes the baseline and applies the same +0.50 MeanER eligibility guard
at every prefix.

| Prefix N | Baseline TPF | Oracle TPF | Delta | Relative delta | Macro MeanER delta | Pieces improved |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1248 | 1215 | -33 | -2.64% | -0.009 | 1/6 |
| 2 | 1248 | 1215 | -33 | -2.64% | -0.009 | 1/6 |
| 4 | 1248 | 1212 | -36 | -2.88% | +0.055 | 2/6 |
| 8 | 1248 | 1212 | -36 | -2.88% | +0.055 | 2/6 |
| 16 | 1248 | 1140 | -108 | -8.65% | -0.459 | 5/6 |
| 32 | 1248 | 1129 | -119 | -9.54% | -0.558 | 6/6 |

The unconstrained baseline-inclusive N=32 oracle is 1248 to 1121 (-10.18%) with macro
MeanER delta -0.005. The guarded curve is the defensible headline because it prevents TPF
gains bought through more than +0.50 MeanER degradation on any selected candidate.

## Post-selection robustness checks

These checks were added after the oracle result was known. They use only the frozen candidate
rows in `benchmark/oracle_prefix_v10_1.json`; no model run, selector change, threshold tuning,
or new piece choice is involved.

| MeanER guard | Oracle TPF | Relative delta | Pieces improved | Macro MeanER delta |
|---:|---:|---:|---:|---:|
| 0.00 | 1145 | -8.25% | 4/6 | -0.605 |
| +0.25 | 1129 | -9.54% | 6/6 | -0.558 |
| +0.50 | 1129 | -9.54% | 6/6 | -0.558 |

The zero guard requires every selected candidate to have MeanER no higher than its own
baseline. The main aggregate reduction therefore does not depend on permitting overall-error
damage. Schubert and Chopin require a positive guard for a strict TPF improvement.

The numbers of +0.50-guarded TPF-improving candidates are 3, 1, 10, 18, 2, and 1 for Haydn,
Brahms, Debussy, Beethoven, Schubert, and Chopin. Brahms and Chopin are singleton-support
cases. For each piece with k qualifying candidates, a uniform size-N subset has support with
probability `1 - C(32-k, N) / C(32, N)`. Exact convolution of the six finite-pool minimum
distributions gives:

| N | Expected pooled TPF delta | Central 90% range | Expected pieces with support |
|---:|---:|---:|---:|
| 1 | -1.85% | [-5.61%, 0.00%] | 1.09 |
| 2 | -3.11% | [-6.65%, 0.00%] | 1.78 |
| 4 | -4.71% | [-7.37%, -1.84%] | 2.60 |
| 8 | -6.37% | [-8.17%, -2.96%] | 3.50 |
| 16 | -7.90% | [-9.13%, -6.49%] | 4.65 |
| 32 | -9.54% | [-9.54%, -9.54%] | 6.00 |

This is a seed-order robustness reference over the observed pools, not a confidence interval
over new pieces. `scripts/audit_oracle_prefix_v10_1.py` independently derives and asserts all
values above.

## Controls and scope

- The sampled populations are six pieces with at least 20 GT tuplet notes and two controls.
- Prokofiev has zero GT tuplet notes, so all 32 candidates have TPF=0 by definition. This is
  an arithmetic control, not evidence that candidates emit no false tuplet symbols.
- Schumann has only eight GT tuplet notes and is reported with controls, not folded into the
  six-piece efficacy aggregate.
- Bach and Rachmaninoff produced no N=32 pool under the frozen routes, so they are not added
  to the sampling-oracle denominator after the fact.
- The four former development pieces are excluded from this held-out oracle headline.

## Integrity checks

- Frozen selector/source/checkpoint/prior hashes match the V10.1 preregistration addendum.
- The locked selection artifact hash is
  `f1c4a6f184bc05724c4584864440149296f339fa03a1fb931178223c1fd7b9cc`.
- Candidate IDs and seeds match all 256 locked sample records.
- Selected candidate stream hashes and metrics reproduce the confirmatory artifact.
- 163 candidates were reconstructed from the complete locked model-score record; the other
  93 reproduce the locked keep/rhythm/timing fingerprint and selected-stream identity.
- An independent read-only script reconstructs every piece-level prefix winner and every
  aggregate, then hard-checks the N=32 totals and Prokofiev control.

## Artifacts

- Canonical result: `benchmark/oracle_prefix_v10_1.json` (sha256
  `4bb023cc3714880a480a488e4f185802cc30c05543f7785108f0714e8cc77e59`)
- Analysis: `scripts/oracle_prefix_v10_1.py`
- Independent audit: `scripts/audit_oracle_prefix_v10_1.py`
- Confirmatory audit: `research/protocols/V10_1_CONFIRMATORY_AUDIT.md`
- Append-only protocol history: `research/protocols/bestofn_prereg.md`

## Binding writeup language

Allowed: "Across all six sampled tuplet-bearing held-out pieces, N=32 contains a
MeanER-guarded candidate with lower TPF; the aggregate guarded oracle reduces TPF by 9.54%."

Not allowed: "V10.1 improves held-out transcription," "the selector recovers the oracle,"
or any wording that presents GT-using oracle selection as a deployable method.
