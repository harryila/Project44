# legacy/ — archived, never deleted

Everything here was moved out of the active tree in the 2026-08 cleanup. Paths mirror the
original layout (`legacy/benchmark/...` was `benchmark/...`). Nothing was deleted; if a file
is not here, it did not move. Rationale per cluster:

## checkpoints_backup/gpu_full/ (~16G)
Run directories of concluded GPU experiments (all documented in the staying
`docs/reports/AR_BUILD_RESULTS.md`, `DATA_SCALING_DERISK.md`, `BEAT_CONDITIONING_RESULTS.md`):
the causal-AR from-scratch series (`ar_*`), pre-SSL scratch runs (`scratch_*`),
beat-conditioning (`beat_asap*`), kern-data fine-tunes (`kern_*`, `broad_mix`),
`ssl_reshape_g1`, and the run dirs of `ssl_classical_clean` / `ssl_bigc` / `ssl_tuplet20`
(their best epochs remain ACTIVE as flat copies in `../checkpoints_backup/`).
**Before the move, every checkpoint referenced by a staying benchmark JSON was extracted to a
flat copy in `checkpoints_backup/` (21 files), and every run's `lightning_logs/metrics.csv`
was copied to `benchmark/curves/<run>.metrics.csv`.** Also here: a third redundant copy of the
released `MIDI2ScoreTF.ckpt`.

## MIDI2ScoreTransformer/checkpoints/
The Apr-30 from-zero local pretrain leftovers (`last.ckpt`, `pretrain_pdmx-epoch=*`,
`lightning_logs/`, `lr_range_test.csv`). Zero references repo-wide; superseded by the SSL line.

## data/
- `pairs/` + `cache_pdmx/`: the April pre-dedup PDMX render + its tokenizer cache, fully
  superseded by the staying `data/pairs_deduped_full/`. Its `_manifest.csv`/`_errors.log`
  were copied out to `data/pairs_april_manifest.csv` / `data/pairs_april_errors.log`.
- `render_test/`: one-off April render sanity check.

## venv/
The pre-migration Python 3.9 env (April). Nothing references it. NOTE: venvs are not
relocatable; this one is preserved bytes, not a runnable env. Recreate from
requirements if ever needed (active envs: `venv311`, `venv_mt3`).

## benchmark/
Pre-MUSTER-era eval (`eval_baseline.py`, `eval_improvements.py`, their metrics JSONs,
`IMPROVEMENT_RESULTS.md`), one-off GPU sweep outputs (`a2_released/`, `a2reshape_full/`,
`stack20/`, `pretrain_eval/`), scratch/temp JSONs (`_tmp_*`, `_beat_smoke`), and 57 GPU-box
logs (`box_logs/`). The tiered harness, all headline-result JSONs, and all negative-result evidence
sets (beat-conditioning, GPU-finetune, TrackB, `_ar*`) STAYED in `benchmark/`.

## scripts/
Concluded one-offs with no staying references: `build_upweighted.py`, `rebuild_and_combine.py`,
`eyeball_check.py`, `smoke_ssl.py`, `smoke_beat_gpu.py`, `calibration_a.py`.
(The GPU box launchers formerly at repo root now live ACTIVE in `scripts/gpu_launchers/`.)

## docs/
Superseded planning/status docs in `reports/` (`ACCURACY_ROADMAP`, `AUDIO_SCORE_ROADMAP`, `NEXT_EXPERIMENTS`,
runbooks, April-June status files, `PROJECT_REPORT`, `GPU_CHECKPOINT_INVENTORY`; the GPU box
is terminated), and the ideation archive (`archive/`: harvest, flat list, digest, triage result).
`LIT_TUPLET_FINDINGS.json` was promoted out of the archive and now lives at `docs/related_work/`.
Day-to-day working logs are not part of this export.

## root prose
Presentation scripts and decks were moved out of the root in the cleanup; they are not part of this
export (only the figures under `prez/figs/` ship).
