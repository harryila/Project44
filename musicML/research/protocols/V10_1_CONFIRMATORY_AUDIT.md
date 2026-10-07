# V10.1 Confirmatory Audit

**Audit date:** 2026-08-22  
**Question:** Was the frozen V10.1 ten-piece confirmation executed correctly, and does it
support the constructive held-out claim?  
**Assessment:** The phase-one selection protocol is valid and reproducible. The original
phase-two tuplet labels were invalid. After a metric-only repair and a replay against the
unchanged locked selections, V10.1 fails the frozen confirmatory rule.

## Critical metric finding

The old tuplet decomposer assumed every MUSTER Fmt3x score used TPQN=24 and tagged a note
with a modulo-3 grid rule. Confirmatory scores actually range from TPQN 12 to 1024 (and the
development Ravel score uses TPQN 7168). This created false GT tuplets: the invalid result
reported Bach=728 and Prokofiev=4649 although both scores contain zero explicit tuplets, and
Chopin=5092 instead of 500.

The repaired evaluator joins each Fmt3x GtID (`P<part>-<measure>-<note ordinal>`) to the
corresponding MusicXML note and uses exact `<time-modification>` membership. It matched all
25,987 confirmatory Fmt3x GtIDs with zero missing joins. Its scored denominator is the exact
GT-to-estimate correspondence block emitted by MUSTER's `ScoreMatchEvaluation`, which
reproduces MUSTER's `nGT - nMiss` denominator rather than inferring correspondence from the
detail rows. TPF remains the preregistered candidate-independent quantity:

`TPF = all explicit GT tuplet notes - matched GT tuplet notes + tuplet onset-error events`.

The original invalid result is preserved as
`benchmark/confirmatory_v10_1_results_invalid_tpqn24.json` (sha256
`85de60af449c9270b84e0f45c3494044ccce729df7bbf705990ca05efb786f9e`).

## Protocol integrity

- The pre-correction preregistration file hash was
  `bece5ab4b62ba6e21bfd986bc0edb9a9a4aebfdeeeee76964f5a9d879fba5af1`.
- All eight frozen V10.1 source, prior, and checkpoint hashes still match Addendum 2.
- The selection artifact is unchanged: sha256
  `f1c4a6f184bc05724c4584864440149296f339fa03a1fb931178223c1fd7b9cc`.
- Selection contains all ten declared pieces and no MUSTER, MeanER, TPF, GT-tuplet, or
  decomposed error fields.
- Phase two re-decodes and verifies baseline/selected stream hashes before evaluation.
- The corrected work tree contains only `baseline` and `selected` evaluations per piece;
  no candidate oracle evaluation was introduced.
- Filesystem chronology and embedded UTC timestamps put the addendum before selection and
  selection before evaluation. There is no external immutable timestamp, so this remains a
  provenance caveat rather than a cryptographic proof of chronology.

## Corrected confirmatory result

| Piece | GT tuplets | Route | TPF baseline | TPF selected | Delta TPF | Delta MeanER |
|---|---:|---|---:|---:|---:|---:|
| Bach Fugue BWV846 | 0 | baseline easy | 0 | 0 | 0 | +0.000 |
| Rachmaninoff Prelude op.23/4 | 678 | deterministic | 204 | 176 | -28 | -1.191 |
| Haydn Sonata 31-1 | 856 | best-of-N | 41 | 92 | +51 | +2.695 |
| Brahms op.118/2 | 153 | baseline fallback | 16 | 16 | 0 | +0.000 |
| Debussy Reflets dans l'Eau | 894 | best-of-N | 411 | 439 | +28 | +1.578 |
| Schumann Arabeske | 8 | best-of-N | 5 | 5 | 0 | -1.841 |
| Beethoven Sonata 10-1 | 364 | baseline fallback | 201 | 201 | 0 | +0.000 |
| Prokofiev Toccata | 0 | best-of-N | 0 | 0 | 0 | +0.616 |
| Schubert Impromptu op.90/1 | 1144 | best-of-N | 224 | 258 | +34 | +0.965 |
| Chopin Ballade 1 | 500 | best-of-N | 355 | 366 | +11 | +1.340 |

Independent arithmetic reproduces the stored rule:

- P1: 1 piece improves by at least 5%; required 3. **Fail.**
- P2: mean per-piece relative TPF delta is +19.39%; required at most -5%. **Fail.**
- P3: Haydn, Debussy, and Schubert exceed the TPF no-harm limit. **Fail.**
- G1: Haydn, Debussy, Prokofiev, Schubert, and Chopin exceed +0.50 MeanER. **Fail.**
- G2: all three low-tuplet controls satisfy the emission and TPF limits. **Pass.**
- Pooled TPF is 1457 to 1553 (+96); macro MeanER delta is +0.4162.

Therefore `CONFIRMED=false`. The paper cannot claim held-out confirmation of V10.1.

## Corrected development result

The old four-piece `2773->2667` TPF total must not be cited. Identity-checked re-evaluation
of the same already-selected outputs gives:

| Piece | TPF baseline | TPF selected | Delta TPF | Delta MeanER |
|---|---:|---:|---:|---:|
| Scriabin | 55 | 47 | -8 | +0.327 |
| Mozart | 15 | 15 | 0 | +0.000 |
| Liszt | 231 | 192 | -39 | +0.157 |
| Ravel | 1079 | 1067 | -12 | -0.498 |

The corrected exploratory total is 1380 to 1321 (-59), with macro MeanER delta -0.00365.
This remains an exploratory development result, not confirmatory evidence.

## Artifacts and checks

- Correct confirmatory result: `benchmark/confirmatory_v10_1_results.json` (sha256
  `e9a3be41ec4e10f0661ea4bfec7bcffa9388b2da718d0e742c9d648910fc035a`)
- Locked selections: `benchmark/confirmatory_v10_1_selections.json`
- Corrected development result: `benchmark/routed_selector_v10_n32_final_metric_v2.json`
  (sha256 `de3b87d1bf786ef72703fc10573c722f4cb0acf999d55cc333816c8221c9d759`)
- Exact metric adapter: `scripts/tuplet_metrics_v2.py` (sha256
  `46c539170918c57ea8360c63ef850d6e419c60ff81e3a96343511484a4809a1b`)
- Exact GT tagger: `scripts/muster_tuplet_decompose.py` (sha256
  `01a979984c58423d9a5f59d67d4c88015028e0917dcd9b37f53f455d7762ac10`)
- Read-only integrity checker: `scripts/audit_confirmatory_v10_1.py`
- Development re-scorer: `scripts/reevaluate_v10_dev_metric_v2.py`

The ten confirmatory pieces are now consumed. Any V11 changes informed by these outcomes
must treat them as development data and require a new untouched confirmation set.

## Secondary frozen-pool diagnosis

The preregistered post-selection oracle/prefix analysis was subsequently completed without
changing the selector or candidate pools. All six sampled tuplet-bearing held-out pools
contain a lower-TPF candidate within +0.50 MeanER; the guarded N=32 oracle is 1248 to 1129
TPF (-9.54%) with macro MeanER delta -0.558. This establishes a sampling-versus-selection
diagnosis but does not alter `CONFIRMED=false`. See `research/protocols/V10_1_ORACLE_AUDIT.md`.
