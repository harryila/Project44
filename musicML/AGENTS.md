# Working in this repository (for people and coding agents)

Read `README.md` first. This file is the short list of facts that prevent wasted hours.

## What is current

- Pipeline entry point: `transcribe.py` (`-t hft -b transformer` is the best configuration).
- Block 2 model and our training stack: `MIDI2ScoreTransformer/midi2scoretransformer/` (`train.py`,
  `dataset.py`, `pdmx_dataset.py`, `pathmap.py`, `tokenizer.py`, `models/`).
- Evaluation harness: `benchmark/eval_tier1_asap.py` (paper-comparable), `benchmark/eval_padsweep.py`
  (14-piece corpus sweep), `benchmark/eval_tuplet.py`, `benchmark/eval_decomposed.py`,
  `scripts/muster_tuplet_decompose.py`. Commands that work are in `README.md` and `benchmark/README.md`.
- Training launchers that run anywhere: `scripts/run_ssl_classical_clean.sh` and
  `scripts/run_ssl_tuplet.sh` (both source `scripts/_env.sh`). The best own model was
  `TUPLET_WEIGHT=2.0 STAGE=ssl_tuplet20 bash scripts/run_ssl_tuplet.sh`.
- Locations: `musicml_paths.py` (Python), `scripts/_env.sh` (shell), overrides in `.env.example`.
- Indexes: `RUNS.md` (runs), `DATA.md` (datasets), `MODELS.md` (checkpoints), `scripts/README.md`,
  `benchmark/README.md`, `assets/MANIFEST.md` (large files).

## What is historical

- Every launcher with a `# HISTORICAL:` header ran only on a 2026 rented GPU machine with fixed GPU
  ids and `pkill` lines. They are records, not tools. Do not run them as they are.
- `LAB_REPORT_COMPLETE.md` (June 2026) and `legacy/` predate the self-supervised pretraining results.
  Where they conflict with `docs/reports/LAB_REPORT.md` or `README.md`, the newer file wins.
- Reports with a `HISTORICAL` or date banner at the top describe the state at that date.

## Commands never to run

- `pip install -r hFT-Transformer/requirements.txt` or `pip install -e mt3` into `venv311`
  (torch 1.10 and TensorFlow pins break the environment). Only `requirements.lock`.
- Anything with `--device mps` or `.to("mps")`: Apple GPU corrupts the model's pad logits.
- `git add -A`, `git add .`, `git commit -a`: stage by path. Checkpoints, data packs, caches and
  outputs are gitignored on purpose; do not force-add them.
- Edits to the frozen research scripts listed in `research/supplement/PROVENANCE.md`
  (`verifier_bestofn.py`, `local_repair_v4.py`, `likelihood_timing_selector_v6.py`,
  `guarded_selector_v8.py`, `decode_lever_sweep_v9.py`, `routed_selector_v10.py`,
  `confirmatory_v10_1.py`, `oracle_prefix_v10_1.py`, and the model sources it pins). Their hashes
  are the provenance of the recorded results. Write a wrapper or a new file.
- Rewriting the `/root/Music-ML-BayenLab` paths inside `data/*.csv` manifests. They are cache keys.
  `pathmap.py` resolves them.

## Where large files come from

`scripts/fetch_assets.sh quickstart | data | archive | all`, verified against `assets/SHA256SUMS`,
extracted into this project root. `assets/RELEASE` names the hosting repository and tag.
Checkpoints land in `MIDI2ScoreTransformer/checkpoints/`, data packs in `data/` and `legacy/data/`,
the ASAP dataset (pinned commit plus 967 generated chunk files plus cache) in
`MIDI2ScoreTransformer/data/`, PDMX under `${MUSICML_DATASETS:-~/datasets}/pdmx`.

## Facts that are easy to get wrong

- The two live own checkpoints: `MIDI2ScoreTransformer/checkpoints/ssl_tuplet20/last.ckpt` (11.87)
  and `MIDI2ScoreTransformer/checkpoints/ssl_classical_clean/ssl_classical_clean-epoch=13-val/total=0.5125.ckpt`
  (the Lightning filename contains `/`, so it is a directory plus a file).
- `ssl_tuplet20/last.ckpt` is epoch 3 of a 15-epoch run (val 0.5388); the epoch-1 checkpoint has the
  best validation loss (0.5234) but worse MUSTER (12.20). Validation loss and MUSTER disagree; select
  by MUSTER.
- The released model is a causal autoregressive decoder (checkpoint config `is_autoregressive: True`
  on the decoder). Older notes calling it bidirectional or mask-predict are wrong.
- The ASAP cache is only hit when the loader sees the literal data dir `./data/` after `cd
  MIDI2ScoreTransformer` and dataset id `diffusion_2024_04_18` (the docstring in `dataset.py` says
  `diffusion_2023_10_13`; the code is right, the docstring is stale).
- `eval_tier1_asap.py --out` is required; its `--ckpt` default is relative to
  `MIDI2ScoreTransformer/`. `eval_padsweep.py` needs `--ckpt` and `--out`.
- The 14-piece test set is small and single performances are noisy (one Schumann performance moved
  the per-piece number by 8 points). Compare on the same pieces; average several performances.
- Numbers quoted as "99 percent vs 0 percent" in older documents are superseded by hit@1 36.66
  percent and hit@15 98.48 percent on the same 1,184 targets.

## Environments

- `requirements.lock`: the Mac/CPU environment the evaluations were run in (Python 3.11.14, torch
  2.11.0, lightning 2.6.1, transformers 4.42.4, tokenizers 0.19.1, numpy 1.26.4, music21 and
  muster from pinned git commits).
- The GPU training runs used torch 2.12.0, lightning 2.6.5, transformers 5.10.2, tokenizers 0.22.2,
  numpy 2.4.6 (`legacy/benchmark/box_logs/pip_install.log`). A CUDA box needs a fresh environment;
  the lock file is evidence of what worked on CPU, not a tested GPU environment.
- `requirements-mt3.lock`: the separate MT3 environment (TensorFlow, JAX). Optional.
