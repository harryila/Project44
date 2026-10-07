# Support-selection audit supplement

This directory contains the machine-readable evidence used by the writeup. It is designed to
let a reader verify the reported arithmetic without downloading the model checkpoint or
rerunning score transcription.

## Verify the two headline sequence results

Run from this directory with Python 3.11 or later:

```bash
python3 scripts/audit_confirmatory_v10_1.py
python3 scripts/audit_oracle_prefix_v10_1.py
python3 scripts/audit_token_and_tail.py
python3 scripts/audit_failure_record.py
```

The first command reconstructs the frozen routes, confirms that the phase-one selection
record contains no ground-truth evaluation fields, and reproduces `CONFIRMED=false`, pooled
TPF `1457 -> 1553`, and macro MeanER delta `+0.416228`.

The second command reconstructs every piece-level prefix oracle and every aggregate. It
reproduces the guarded N=32 result `1248 -> 1129` TPF, macro MeanER delta `-0.558135`, and
the finding that all six sampled tuplet-bearing efficacy pools contain a guarded improving
candidate. It also derives the post-selection guard sensitivity (`1248 -> 1145` at a zero
guard) and the exact uniform-subset reference from the same frozen candidate rows.

The third command reproduces the 1,184-note teacher-forced population, hit@1/5/15 values,
all four keep-masked greedy-rollout counts, and both checkpoints' hidden-tail piece counts.

The fourth command checks the exact-label, keep-masked boost sweep, reconstructs
the winner of every permitted V3b evidence weight, verifies the saturated V4 repair window,
and confirms that the discarded 24-TPQN artifacts fail the Bach and Prokofiev controls.

## Contents

- `artifacts/muster_tuplet_decomposed.json`: exact MusicXML-tuplet error decomposition.
- `artifacts/topk_offset_diag.json`: teacher-forced local-rank and keep-masked rollout diagnostic.
- `artifacts/confirmatory_v10_1_selections.json`: frozen, ground-truth-blind phase-one record.
- `artifacts/confirmatory_v10_1_results.json`: corrected ten-piece confirmation.
- `artifacts/oracle_prefix_v10_1.json`: all 256 candidate evaluations and prefix oracles.
- `artifacts/routed_selector_v10_n32_final_metric_v2.json`: corrected exploratory result.
- `artifacts/reranker_doseresponse_exact.json`: exact-label, keep-masked boost sweep.
- `artifacts/verifier_bestofn_dev_v3b.json`: frozen V3b development pool and scores.
- `artifacts/local_repair_v4_dev.json`: frozen V4 development result and drift gate.
- `artifacts/*invalid_tpqn24.json`: retained outputs from the discarded labeler.
- `artifacts/*unmasked_positions.json`: superseded prevalence records retained for audit.
- `scripts/tuplet_metrics_v2.py`: exact TPF adapter.
- `scripts/muster_tuplet_decompose.py`: MusicXML label and MUSTER correspondence join.
- `scripts/audit_confirmatory_v10_1.py`: standalone confirmatory arithmetic audit.
- `scripts/audit_oracle_prefix_v10_1.py`: standalone candidate and prefix audit.
- `scripts/audit_token_and_tail.py`: standalone local-rank and decomposed-tail audit.
- `scripts/audit_failure_record.py`: standalone intervention-record audit.
- `../make_figures_bestofn.py`: figure builder for the writeup, kept one directory up.
- `bestofn_prereg.md`: full design, addenda, frozen thresholds, and failure record.
- `PROVENANCE.md`: source identities, stream coverage, chronology, and rerun boundary.
- `MANIFEST.sha256`: full hashes for every other packaged file.

Run `python3 ../make_figures_bestofn.py` from this directory to regenerate all figures for
the writeup. Matplotlib is the only figure dependency.

## Hashes and rerun boundary

Run `shasum -a 256 -c MANIFEST.sha256` after extraction to verify every packaged file.
`PROVENANCE.md` records full checkpoint, prior, and experiment-source identities and explains
which stream identities were captured at runtime.

The standalone audits reproduce the writeup's arithmetic from the packaged records. This is
not a self-contained candidate-generation environment. A full model and MUSTER rerun also
requires omitted upstream assets, the duration prior, frozen experiment source, and decoded
streams. See `PROVENANCE.md` for the exact boundary.

## Data and licensing

The pinned ASAP snapshot is based on v1.1 and is distributed by its creators under
CC BY-NC-SA 4.0. The dataset, MUSTER, and released model checkpoint are not redistributed
here. The public model and MUSTER repository snapshots did not declare standalone repository
licenses. See `LICENSE.md` for the licenses applied to the new scripts and derived numerical
records in this supplement.
