# MODELS.md: the model registry

Every checkpoint, weight file and learned prior this project uses or has produced, with where it
comes from, where it lands on disk, what it scored, and how to load it. Paths are relative to the
project root (the directory holding `musicml_paths.py`) unless stated otherwise. Every sha256 in
sections 1 and 2 is copied from `assets/SHA256SUMS`; the 56 archived checkpoints of section 4 are
hashed in the same file under their `ckpt-archive/` names and the hashes are not repeated here. Sizes
are from `assets/ASSETS.csv`.

Conventions used throughout:

- **corpus MeanER**: mean MUSTER MeanER over the 14 ASAP test pieces, one performance per piece
  (`--limit-per 1`), pad threshold 0.50, as produced by `benchmark/eval_padsweep.py` or
  `benchmark/eval_tuplet.py`. Lower is better.
- **tier-1 MeanER**: mean over all 59 performances of the ASAP test split, `benchmark/eval_tier1_asap.py`.
  This is the number comparable with the paper (Beyer and Dai, ISMIR 2024, arXiv:2410.00210).
- **val/total**: the teacher-forced validation loss that Lightning logged during training, on the ASAP
  validation split. It does not track MUSTER: the released model scores 0.674 on that split and 10.77
  on the corpus, while every own SSL checkpoint with a corpus evaluation scores 0.50 to 0.57 on the
  split and 11.8 to 16.1 on the corpus (more cases in sections 1 and 4). Pick checkpoints by MUSTER.
- **epoch / step**: as stored inside the checkpoint (`ckpt["epoch"]`, `ckpt["global_step"]`). Epoch
  numbers are zero-based. The training curves in `benchmark/curves/*.metrics.csv` log the step at
  validation time, one less than the saved `global_step`.
- The eval JSONs name the checkpoint they scored in their `ckpt` field. Box-side runs recorded the
  GPU machine's absolute repository prefix (a directory under `/root/`, spelled out in `pathmap.py`);
  Mac-side runs recorded a path relative to `MIDI2ScoreTransformer/` (`checkpoints/...`).
  `MIDI2ScoreTransformer/midi2scoretransformer/pathmap.py` maps the recorded prefixes to this machine.

A warning that applies to every own checkpoint, live or archived: **a file called `last.ckpt` is not
necessarily the final epoch of its run.** Verified by loading: `ssl_tuplet20/last.ckpt` holds epoch 3,
step 2780 (val 0.5388), although its curve runs to epoch 14. The asset list confirms the same pattern
ten more times, where a `last.ckpt` is byte-identical to a named epoch checkpoint that is not the
final one (`beat_asap` epoch 10 of 12, `beat_asap_v2` 4 of 8, `kern_mix` 4 of 8, `kern_mix_v2` 4 of 6,
`kern_distill` 3 of 8, `kern_distill03` 5 of 8, `kern_distill_sel` 5 of 8, `scratch_fulldrop` 10 of 30,
`scratch_long` 35 of 150, `ssl_reshape_g1` 2 of 15). The rule that fits all 17 surviving `last.ckpt`
files (checked against `benchmark/curves/*.metrics.csv` with `save_top_k=2`, mode min, strict
improvement): **`last.ckpt` is the save made at the most recent epoch that entered the top-2 by
`val/total`**, so it equals the final epoch only when the final epoch was itself a top-2 epoch. The
Lightning code path that produced this on the box was not established from files, but the pattern has
no exception. Always read `ckpt["epoch"]` and `ckpt["global_step"]`; never infer the epoch from the
file name.

---

## 1. Live checkpoints

The three checkpoints that scripts expect to find, installed by `scripts/fetch_assets.sh quickstart`
into `MIDI2ScoreTransformer/checkpoints/` (`CHECKPOINTS_DIR` in `musicml_paths.py`, overridable with
`MUSICML_CHECKPOINTS_DIR`). `python musicml_paths.py` prints whether each one is present.

| | Released model | Best own model | Own SSL baseline |
|---|---|---|---|
| Asset name | `MIDI2ScoreTF.ckpt` | `ssl_tuplet20_last.ckpt` | `ssl_classical_clean_epoch13.ckpt` |
| Install path | `MIDI2ScoreTransformer/checkpoints/MIDI2ScoreTF.ckpt` | `MIDI2ScoreTransformer/checkpoints/ssl_tuplet20/last.ckpt` | `MIDI2ScoreTransformer/checkpoints/ssl_classical_clean/ssl_classical_clean-epoch=13-val/total=0.5125.ckpt` |
| `musicml_paths` constant | `RELEASED_CKPT` | `BEST_OWN_CKPT` | `SSL_CLASSICAL_CLEAN_CKPT` |
| sha256 | `7b8ec6e3da365b97443fb67a8f0b37d63997e93c152d665d43cb2011245db638` | `8e52c3574507b26ef79d6f7a8ad7b842746533ed97bc2a111246e48736e5a34a` | `666bf3264d42d982766e8a9c14b6b07579bbe27d46ffc132c6391eafbdf4798f` |
| Size (bytes) | 389,829,880 | 389,828,159 | 389,828,159 |
| Origin | Upstream release of github.com/TimFelixBeyer/MIDI2ScoreTransformer (Beyer and Dai, ISMIR 2024) | Warm start from the SSL baseline (right column), same masked-SSL recipe, tuplet loss weight 2.0 | From scratch, masked SSL on the leak-filtered classical corpus |
| Epoch / step | 21 / 40,000 | 3 / 2,780 | 13 / 9,730 |
| val/total | 0.674 when scored by this project's trainer on the ASAP validation split (`docs/reports/BEAT_CONDITIONING_RESULTS.md`); the file itself stores no validation score | 0.5388 | 0.5125 |
| Corpus MeanER (14 pieces, thr 0.50) | **10.77** (`benchmark/padsweep_released.json`) | **11.87** (`benchmark/ssl_tuplet20_last.json`); 11.79 on re-evaluation (`benchmark/rerank/ssl_tuplet20__ssl_tuplet20_last.json`) | **12.69** (`benchmark/padsweep_sslcc.json`); 12.77 on re-evaluation (`benchmark/rerank/ssl_classical_clean__ssl_classical_clean-epoch_13-val_total_0.5125.json`) |
| Tier-1 MeanER (59 performances) | **11.18** on CPU (`benchmark/tier1_baseline.json`), 11.16 on CUDA (`benchmark/eval_A_baseline.json`); the paper reports 11.30 | not run | not run |

### 1.1 `MIDI2ScoreTF.ckpt` (released model)

- Recipe saved inside the checkpoint (`ckpt["hyper_parameters"]`, listed in
  `legacy/docs/reports/SYNTHETIC_PRETRAIN_STATUS.md`): AdamW, lr 3e-4, betas (0.9, 0.999), weight
  decay 0.2, 40,000 steps, 4,000 warmup, batch 32, sequence 512, input_dropout 0.75,
  unconditional_dropout 0.5, two datasets weighted [0.5, 0.5] (paired ASAP plus unpaired scores).
- Decoder is causal (`is_autoregressive=True` in the saved decoder config). The earlier belief that
  it was bidirectional was wrong; `docs/reports/AR_BUILD_RESULTS.md` records the correction.
- Other numbers that belong to this checkpoint: the rank diagnostic on 1,184 teacher-forced
  off-dyadic targets, hit@1 36.66 percent and hit@15 98.48 percent (`benchmark/topk_offset_diag.json`);
  the clean-Chopin full audio-to-score result of about 2.75 MeanER (`docs/reports/SONGSCRIPTION_PARITY.md`);
  every best-of-N / support-selection result under `benchmark/*selector*`, `benchmark/verifier_*`,
  `benchmark/local_repair_*`, `benchmark/reranker_*` and `benchmark/decode_lever_sweep_*`, all of which
  record `ckpt: checkpoints/MIDI2ScoreTF.ckpt`.
- Pad threshold: `generate()` was fixed so that `--pad-threshold` is live (`docs/reports/PAD_THRESHOLD_FIX.md`).
  At 0.50 the output is byte-identical to the released behaviour, so 0.50 is the only threshold at
  which numbers are comparable with the paper.
- Scripts that expect this exact path: `transcribe.py` (`CHECKPOINT_PATH`), `benchmark/eval_tier1_asap.py`
  (default `--ckpt checkpoints/MIDI2ScoreTF.ckpt`, relative to `MIDI2ScoreTransformer/` because the
  script changes directory there), `benchmark/eval_decomposed.py`, the frozen best-of-N scripts
  (`scripts/routed_selector_v10.py`, `verifier_bestofn.py`, `guarded_selector_v8.py`,
  `decode_lever_sweep_v9.py`, `local_repair_v4.py`, `local_repair_v5.py`,
  `likelihood_timing_selector_v6.py`, `evidence_screen_selector_v7.py`, `rerank_offset.py`,
  `diag_topk_offset.py`, `muster_tuplet_decompose.py`, `confirmatory_v10_1.py`),
  `scripts/probe_released_padprob.py`, the GPU launchers (`scripts/gpu_finetune.sh`,
  `scripts/run_arm1_mixed.sh`, and `diag_clean.sh`, `diag_launch.sh`, `launch_gentle.sh` under
  `scripts/gpu_launchers/`). `scripts/audit_confirmatory_v10_1.py`
  and `research/protocols/bestofn_prereg.md` pin its sha256; a different file fails the audit by design.

### 1.2 `ssl_tuplet20/last.ckpt` (best own model)

- Produced by `TUPLET_WEIGHT=2.0 STAGE=ssl_tuplet20 bash scripts/run_ssl_tuplet.sh`: warm start from
  the SSL baseline, `--dataset-type ssl --real-fraction 0.5 --autoregressive`, manifest
  `data/pairs_classical_clean_manifest.csv`, lr 2e-4, 15 epochs, warmup 300, batch 32, sequence 512,
  bf16. The training inputs it needs (the classical_clean renders and cache) are not in this
  repository; see `DATA.md`.
- The checkpoint is epoch 3, step 2780, val 0.5388. Its own callback state lists the two best-by-val
  epochs of the run as epoch 1 (0.5234) and epoch 3 (0.5388) and records `last_model_path` as this
  file. The run's curve (`benchmark/curves/ssl_tuplet20.metrics.csv`) continues to epoch 14, and no
  later epoch beat 0.5388, so epoch 3 was the last top-2 save (the rule stated at the top of this
  file). `docs/reports/LAB_REPORT.md` describes the 11.87 checkpoint as "last/ep14"; the file
  contradicts that wording; trust the file. Directory listings in `legacy/benchmark/box_logs/rerank.log`
  show the box held exactly `last.ckpt`, epoch 1 and epoch 3 for this run.
- `ckpt["hyper_parameters"]` of this checkpoint is inherited from the warm-start source (it says lr
  3e-4, warmup 1000, tuplet_weight 1.0, total_steps 40000). The logger copy
  `legacy/checkpoints_backup/gpu_full/ssl_tuplet20/lightning_logs/version_0/hparams.yaml` repeats the
  same inherited values. None of them describe this run. The real settings are in the launcher and in
  `docs/reports/LAB_REPORT.md`. This applies to every warm-started run (the `hparams.yaml` of every
  fine-tune of the released model likewise says lr 3e-4, warmup 1000, 40000 steps).
- Per-piece numbers at threshold 0.50 are in section 1.4. The val-best epoch of the same run (epoch 1,
  archived, section 4) scores worse on MUSTER (12.20 and 12.15), one of the proofs that val is not MUSTER.
- Scripts that expect the path: `musicml_paths.BEST_OWN_CKPT`; `scripts/rerank_by_muster.sh`
  enumerates `checkpoints/ssl_tuplet20/*.ckpt`; `scripts/run_ssl_tuplet.sh` writes to
  `checkpoints/<STAGE>/`.

### 1.3 `ssl_classical_clean/.../total=0.5125.ckpt` (own SSL baseline)

- Trained from scratch with the causal-AR decoder and the masked-SSL data strategy: 50/50 real ASAP
  pairs and 23,783 leak-filtered genre=classical PDMX scores (`data/pairs_classical_clean_manifest.csv`),
  lr 3e-4, batch 32, 30 epochs at about 695 steps per epoch, warmup 416 steps
  (`legacy/checkpoints_backup/gpu_full/ssl_classical_clean/lightning_logs/version_0/hparams.yaml`;
  the curve is `benchmark/curves/ssl_classical_clean.metrics.csv`). Best val 0.5125 at epoch 13.
  The story is in `docs/reports/AR_BUILD_RESULTS.md` and `docs/reports/LAB_REPORT.md` sections 4 to 6.
- The path contains a slash and two equals signs: the checkpoint is a file named `total=0.5125.ckpt`
  inside a directory named `ssl_classical_clean-epoch=13-val/`, because Lightning formats the monitor
  key `val/total` into the file name. Quote it in shells.
- It is the warm-start source of all tuplet-lever runs (`scripts/run_ssl_tuplet.sh` `INIT_CKPT`
  default) and the "epoch-13 replication checkpoint" of the best-of-N / support-selection study
  (`research/supplement/PROVENANCE.md` pins its sha256; `benchmark/routed_selector_v10_ep13_*.json`).
- Scripts that expect the path: `musicml_paths.SSL_CLASSICAL_CLEAN_CKPT`, `scripts/run_ssl_tuplet.sh`,
  `scripts/run_padsweep.sh` (default glob `checkpoints/ssl_classical_clean/*.ckpt`),
  `scripts/build_and_train_st.sh`.

### 1.4 Per-piece MeanER of the three live checkpoints (threshold 0.50)

From `benchmark/padsweep_released.json`, `benchmark/padsweep_sslcc.json` and
`benchmark/ssl_tuplet20_last.json`. In parentheses: predicted tuplet notes versus ground truth.

| Piece | Released | ssl_classical_clean ep13 | ssl_tuplet20 last |
|---|---|---|---|
| Bach, Fugue BWV 846 (0 tuplets) | 5.62 (0) | 7.20 (0) | 5.15 (0) |
| Scriabin, Etude Op.8 No.11 (99) | 10.37 (0) | 12.32 (0) | 11.69 (1) |
| Rachmaninoff, Prelude Op.23 No.4 (641) | 14.44 (134) | 16.02 (0) | 14.95 (6) |
| Haydn, Sonata Hob.XVI:31 i (860) | 4.92 (711) | 10.56 (21) | 9.97 (0) |
| Brahms, Op.118 No.2 (154) | 6.63 (224) | 11.11 (363) | 9.67 (0) |
| Debussy, Reflets dans l'eau (804) | 12.72 (672) | 13.21 (321) | 11.83 (636) |
| Mozart, Sonata K.332 i (36) | 5.86 (0) | 5.76 (2) | 6.49 (0) |
| Schumann, Arabeske (8) | 11.15 (9) | 19.28 (0) | 11.42 (0) |
| Beethoven, Sonata Op.10 No.1 i (364) | 8.82 (91) | 9.50 (0) | 8.81 (0) |
| Liszt, Gondoliera (402) | 13.28 (675) | 14.45 (0) | 13.42 (0) |
| Ravel, Ondine (1281) | 20.80 (384) | 20.47 (0) | 21.64 (0) |
| Prokofiev, Toccata (0) | 9.32 (21) | 8.82 (0) | 11.51 (0) |
| Schubert, Impromptu D.899 No.1 (1059) | 18.31 (1072) | 19.20 (2151) | 20.39 (1419) |
| Chopin, Ballade No.1 (432) | 8.55 (260) | 9.71 (0) | 9.25 (104) |
| **Corpus mean** | **10.77** | **12.69** | **11.87** |

Single-performance numbers are noisy (`docs/reports/LAB_REPORT.md` shows the Schumann value moving by
several points between performances). The corpus mean is the robust figure.

---

## 2. Upstream weights (Block 1 transcribers)

These are other projects' models. They are not retrained here and never were. Two ship as release
assets, two ship inside pip packages that `requirements.lock` pins.

| Model | Where the weights live | Installed by | Loaded by |
|---|---|---|---|
| hFT-Transformer, MAESTRO-V3 (ISMIR 2023 release) | `hFT-Transformer/checkpoint/MAESTRO-V3/model_016_003.pkl` (22,136,000 bytes) plus `parameter.json` (957 bytes, tracked in git); model config `hFT-Transformer/corpus/config.json` | asset `hft-checkpoint.tar.zst`, 20,522,914 bytes, sha256 `97f7499b370f828879fb66c41d26a2ec99083f31563aa12c38ccd329ea33663e` | `transcribe.py -t hft` (`audio_to_midi_hft`, `torch.load(..., weights_only=False)`) |
| MT3, `ismir2021` piano checkpoint (ISMIR 2021) | `mt3/checkpoints/ismir2021/` (t5x format: a `checkpoint` file plus one directory per parameter, 148 entries, about 165 MB) | asset `mt3-checkpoints.tar.zst`, 171,987,569 bytes, sha256 `7452df75ed12968c7ae1433d1ac7980e2656a5d7e10a816183c07efc3aec27da` | `mt3_inference.py` (`InferenceModel(checkpoint_path, "ismir2021")`, t5x `RestoreCheckpointConfig`) via `transcribe.py -t mt3`; needs the separate `venv_mt3` built from `requirements-mt3.lock` |
| Transkun 2.0.1 | inside the wheel: `<site-packages>/transkun/pretrained/2.0.pt` (56,408,978 bytes) and `2.0.conf` | `requirements.lock` pins `transkun==2.0.1` | `transcribe.py -t transkun` (`pkg_resources.resource_filename("transkun", "pretrained/2.0.pt")`) |
| Basic Pitch 0.4.0 (ICASSP 2022 model) | inside the wheel: `<site-packages>/basic_pitch/saved_models/icassp_2022/` with four formats: `nmp/` (TensorFlow SavedModel), `nmp.mlpackage/` (CoreML), `nmp.tflite`, `nmp.onnx` | `requirements.lock` pins `basic-pitch==0.4.0` | `transcribe.py -t basic-pitch` (the default transcriber) through `basic_pitch.inference.predict` |

Notes:

- Basic Pitch picks its backend at import time in the order tensorflow, coremltools, tflite-runtime,
  onnxruntime. `requirements.lock` contains `coremltools==9.0` and none of the other three, so the
  pinned environment uses `nmp.mlpackage`, which only runs on macOS. On Linux use `-t hft` (what the
  README quickstart does), or install `tensorflow` or `onnxruntime` and remove `coremltools`.
- hFT and MT3 weights are release assets because they are too large for git and are not on PyPI. The
  tracked `parameter.json` is the hFT training config, not weights.
- **Hash-check once in a fresh install.** The lock pins package versions, not the bytes of the weight
  files inside them, and the two assets are verified by `fetch_assets.sh` only at extraction time. After
  `pip install -r requirements.lock` and `scripts/fetch_assets.sh quickstart`, run the commands below,
  compare with the reference values, and record the result in this file.

```bash
shasum -a 256 hFT-Transformer/checkpoint/MAESTRO-V3/model_016_003.pkl
python -c "import transkun, os; print(os.path.join(os.path.dirname(transkun.__file__), 'pretrained', '2.0.pt'))" | xargs shasum -a 256
BP=$(python -c "import basic_pitch, os; print(os.path.dirname(basic_pitch.__file__))")/saved_models/icassp_2022
shasum -a 256 "$BP/nmp.onnx" "$BP/nmp.tflite" "$BP/nmp/saved_model.pb" "$BP/nmp.mlpackage/Data/com.apple.CoreML/weights/weight.bin"
find mt3/checkpoints/ismir2021 -type f | sort | xargs shasum -a 256 > /tmp/mt3.sha256   # keep this list with the install
```

Reference values, measured on the original Mac environment (Python 3.11, `requirements.lock`) on
2026-10-06. They are reference points for the check above, not part of `assets/SHA256SUMS`:

| File | sha256 |
|---|---|
| `hFT-Transformer/checkpoint/MAESTRO-V3/model_016_003.pkl` | `f3a90cf0b82895f9923f3e766bff7e0a4759bc477370f015496ded2ad63e127f` |
| `transkun/pretrained/2.0.pt` | `50a80010effc2a59ffcd068a95cd2b29bd7f23a27a3515bc3ccd209c89a3d44c` |
| `basic_pitch/saved_models/icassp_2022/nmp.onnx` | `2c3c1d144bfa61ad236e92e169c13535c880469a12a047d4e73451f2c059a0ec` |
| `basic_pitch/saved_models/icassp_2022/nmp.tflite` | `3db297d54af8e01c6e5618245c956b1d71b6a2b978cb2dedb527173186552676` |
| `basic_pitch/saved_models/icassp_2022/nmp/saved_model.pb` | `eaa25c91c431c91100c416a2c018663f4c635f28fa19529c4ff5e14c18aa29c9` |
| `basic_pitch/saved_models/icassp_2022/nmp.mlpackage/Data/com.apple.CoreML/weights/weight.bin` | `691a6b63c7ddcdde0ee131ff3986dcb1250df47cd738612efde966ba9b4c99cd` |
| `mt3/checkpoints/ismir2021/checkpoint` (the index file only) | `2fe909f0fd8052d7b86ff4e0e23630bdc20d932586e36024c9e72fa0dbb974b9` |

---

## 3. Learned priors

Two small torch files in `data/`, both tracked in git, both fitted on ASAP train plus validation
engraved scores only (never the test split), both used only at inference time.

### 3.1 `data/duration_priors.pt`

- 11,685 bytes, written 2026-06-07, sha256
  `7e0b8c2543362e75f70f42b46d93832db983c4f41692320a3771b0cee47710e4` (pinned by
  `research/protocols/bestofn_prereg.md`, `research/supplement/bestofn_prereg.md`,
  `scripts/audit_confirmatory_v10_1.py` and `research/supplement/scripts/audit_confirmatory_v10_1.py`).
- Contents: a dict with `log_pi_dur` (97,) the log marginal over the 97 duration buckets (the A1
  logit-adjustment prior), `log_p_dur_given_phase` (24, 97) the log conditional of duration given the
  sub-quarter metrical phase `offset_bucket % 24` (the A2 metrical prior), `n_phase` 24, `vocab_dur` 97,
  `n_notes` 514,645, `tuplet_frac` 0.1129.
- Produced by `scripts/compute_duration_priors.py` (`--out data/duration_priors.pt`, splits train and
  validation, scores collected with `collect_paths` from `benchmark/eval_tier1_asap.py`, tokenized with
  `MultistreamTokenizer.tokenize_mxl`). Rebuilding needs the ASAP asset; a rebuilt file must hash to the
  value above or the frozen audits fail, which is intended.
- Consumed through `model.generate(dur_log_pi=..., dur_tau=..., dur_metrical=..., dur_metrical_lambda=...)`
  (`MIDI2ScoreTransformer/midi2scoretransformer/models/model.py`, passed through `utils.infer`) by
  `benchmark/eval_padsweep.py` (`--prior-path` with `--dur-tau` / `--dur-metrical-lambda`),
  `scripts/decode_lever_sweep_v9.py`, `scripts/routed_selector_v10.py`, `scripts/confirmatory_v10_1.py`,
  `scripts/oracle_prefix_v10_1.py`, `scripts/reevaluate_v10_dev_metric_v2.py`,
  `scripts/test_duration_priors.py` and `scripts/run_dur_prior_sweep.sh`. With `dur_tau=0` and
  `dur_metrical_lambda=0` the decode is byte-identical to plain argmax.

### 3.2 `data/offset_phase_prior.pt`

- 1,782 bytes, written 2026-06-25, sha256
  `3729b6a0c64f55cbebdb66f82f417f1d6b51aa744fcc104d31f7322385564f02` (measured 2026-10-06; this file is
  not pinned by any audit).
- Contents: `log_p_phase` (24,) the empirical log probability of each sub-quarter offset phase, and
  `n_scores` 201 (the number of engraved scores it was accumulated from).
- Produced by `build_phase_prior()` in `scripts/rerank_offset.py`, the offset-phase reranker that grew
  out of the phase diagnostic (`scripts/diag_offset_phase.py` is the diagnostic itself; it reads
  checkpoints and prints phase histograms but does not write this file). `build_phase_prior()` returns
  the cached file when it exists and otherwise recomputes it from ASAP train plus validation and writes
  it, so deleting the file and running `scripts/rerank_offset.py` regenerates it (ASAP asset required).
- Consumed only by `OffsetReranker` in `scripts/rerank_offset.py`, which is passed to
  `model.generate(offset_rerank=...)` and applied to the offset head's logits at each decode step.
  `scripts/verifier_bestofn.py` mentions `rerank_offset.py` in its docstring but does not import it or
  read this file; no other script in the export references `offset_phase_prior.pt`. `lambda=0` is the
  identity.

---

## 4. Archived checkpoints (56 files, tier `archive-checkpoints`, 22.1 GB)

Every non-live checkpoint that survived, deduplicated by content: when two original paths held
byte-identical files, one asset carries both paths. `scripts/fetch_assets.sh archive` installs each file
at the first original path listed (the `extracts_to` column of `assets/ASSETS.csv`). Asset names below
are the file names under `ckpt-archive/` in the release. Epoch and val/total are parsed from the names
(`-epNN-valX.XXXX` or `-epoch-NN-val__total-X.XXXX`); for a `last.ckpt` they come from the aliased
original path, cross-checked against `benchmark/curves/<run>.metrics.csv`. Status is one of
"superseded epoch of a live run" or "ruled-out run", with the report that rules it out.

Sizes are informative: 389.8 MB is the standard RoFormer checkpoint; the two 520.3 MB `kern_distill`
files also carry the frozen teacher (`_teacher.*` keys, a bug fixed before the later distillation runs);
the `beat_asap*` files are about 88 KB larger because of the extra beat input layer.

### 4.1 April 2026 Mac pretraining (bidirectional decoder, empty generation)

Stage A synthetic pretraining on 5,248 rendered PDMX pairs, Mac CPU, lr 1e-4, batch 4, four epochs.
Loss fell cleanly but the checkpoints generate empty scores. Ruled out by
`legacy/docs/reports/SYNTHETIC_PRETRAIN_STATUS.md`.

| Asset file | Epoch | val/total | Bytes | Original path(s) | Status |
|---|---|---|---|---|---|
| `legacy__MIDI2ScoreTransformer__checkpoints__pretrain_pdmx-epoch-02-val__total-0.2650.ckpt` | 2 | 0.2650 | 389,814,725 | `legacy/MIDI2ScoreTransformer/checkpoints/pretrain_pdmx-epoch=02-val/total=0.2650.ckpt` | ruled-out run, SYNTHETIC_PRETRAIN_STATUS.md |
| `legacy__MIDI2ScoreTransformer__checkpoints__last.ckpt` | 3 | 0.2440 | 389,814,725 | `legacy/MIDI2ScoreTransformer/checkpoints/last.ckpt` and `legacy/MIDI2ScoreTransformer/checkpoints/pretrain_pdmx-epoch=03-val/total=0.2440.ckpt` | ruled-out run (the "best ckpt" of that report), SYNTHETIC_PRETRAIN_STATUS.md |

### 4.2 Warm-start fine-tunes of the released checkpoint (all ruled out)

Every attempt to continue training `MIDI2ScoreTF.ckpt` degraded MUSTER, often while val/total
improved. Beat conditioning: `docs/reports/BEAT_CONDITIONING_RESULTS.md`. Kern mixes, distillation and
the broad corpus: `docs/reports/DATA_SCALING_DERISK.md`. The hard-piece evaluations
(`benchmark/tuplet_*.json`, five pieces) have the released model at 11.87 on the same five pieces
(`benchmark/tuplet_baseline.json`).

| Asset file | Epoch | val/total | Bytes | Original path(s) | Status |
|---|---|---|---|---|---|
| `legacy__checkpoints_backup__gpu_full__beat_asap__beat_asap-epoch-00-val__total-0.6736.ckpt` | 0 | 0.6736 | 389,915,765 | `legacy/checkpoints_backup/gpu_full/beat_asap/beat_asap-epoch=00-val/total=0.6736.ckpt` | ruled-out run (beat conditioning v1, lr 3e-4, 12 epochs, ASAP only), BEAT_CONDITIONING_RESULTS.md |
| `checkpoints_backup__beat_asap-last.ckpt` | 10 | 0.6872 | 389,915,957 | `checkpoints_backup/beat_asap-last.ckpt` and `legacy/checkpoints_backup/gpu_full/beat_asap/beat_asap-epoch=10-val/total=0.6872.ckpt` | ruled-out run; tier-1 15.44 with gold beats and 15.54 without (`benchmark/eval_B_beat_gold.json`, `eval_C_beat_nobeat.json`) versus 11.16, BEAT_CONDITIONING_RESULTS.md |
| `legacy__checkpoints_backup__gpu_full__beat_asap_v2__beat_asap_v2-epoch-02-val__total-0.6768.ckpt` | 2 | 0.6768 | 389,915,957 | `legacy/checkpoints_backup/gpu_full/beat_asap_v2/beat_asap_v2-epoch=02-val/total=0.6768.ckpt` | ruled-out run (beat conditioning v2, lr 2e-5, 8 epochs), BEAT_CONDITIONING_RESULTS.md |
| `checkpoints_backup__beat_asap_v2-ep04-val0.6739.ckpt` | 4 | 0.6739 | 389,916,021 | `checkpoints_backup/beat_asap_v2-ep04-val0.6739.ckpt` and `legacy/checkpoints_backup/gpu_full/beat_asap_v2/last.ckpt` | ruled-out run; tier-1 13.40 with gold beats, 13.59 without, 59/59 scored (`benchmark/eval_Bp_v2ep4_gold.json`, `eval_Cp_v2ep4_nobeat.json`; the text of BEAT_CONDITIONING_RESULTS.md quotes 14.54 and 14.76 for the same comparison, a discrepancy the report does not explain; the JSONs are the primary record), BEAT_CONDITIONING_RESULTS.md |
| `legacy__checkpoints_backup__gpu_full__kern_mix__kern_mix-epoch-00-val__total-0.6911.ckpt` | 0 | 0.6911 | 389,827,269 | `legacy/checkpoints_backup/gpu_full/kern_mix/kern_mix-epoch=00-val/total=0.6911.ckpt` | ruled-out run (50/50 ASAP and kern renders, lr 1e-4), DATA_SCALING_DERISK.md |
| `checkpoints_backup__kern_mix-last.ckpt` | 4 | 0.7038 | 389,827,525 | `checkpoints_backup/kern_mix-last.ckpt` and `legacy/checkpoints_backup/gpu_full/kern_mix/kern_mix-epoch=04-val/total=0.7038.ckpt` | ruled-out run; hard pieces 17.58 (`benchmark/tuplet_kernmix.json`), tuplet production improved but MeanER drifted, DATA_SCALING_DERISK.md |
| `checkpoints_backup__kern_mix_v2-ep03-val0.6770.ckpt` | 3 | 0.6770 | 389,827,525 | `checkpoints_backup/kern_mix_v2-ep03-val0.6770.ckpt` | ruled-out run (lr 3e-5, real fraction 0.7); hard pieces 16.81 (`benchmark/tuplet_kernv2.json`), DATA_SCALING_DERISK.md |
| `legacy__checkpoints_backup__gpu_full__kern_mix_v2__last.ckpt` | 4 | 0.6856 | 389,827,525 | `legacy/checkpoints_backup/gpu_full/kern_mix_v2/last.ckpt` and `legacy/checkpoints_backup/gpu_full/kern_mix_v2/kern_mix_v2-epoch=04-val/total=0.6856.ckpt` | ruled-out run, DATA_SCALING_DERISK.md |
| `checkpoints_backup__kern_distill-ep02-val0.5901.ckpt` | 2 | 0.5901 | 520,294,067 | `checkpoints_backup/kern_distill-ep02-val0.5901.ckpt` | ruled-out run (distillation lambda 1.0; carries `_teacher.*` keys, load with `strict=False`); hard pieces 13.78 (`benchmark/tuplet_distill.json`), tuplets suppressed, DATA_SCALING_DERISK.md |
| `legacy__checkpoints_backup__gpu_full__kern_distill__last.ckpt` | 3 | 0.6054 | 520,294,067 | `legacy/checkpoints_backup/gpu_full/kern_distill/last.ckpt` and `legacy/checkpoints_backup/gpu_full/kern_distill/kern_distill-epoch=03-val/total=0.6054.ckpt` | ruled-out run (carries `_teacher.*` keys), DATA_SCALING_DERISK.md |
| `checkpoints_backup__kern_distill03-ep04-val0.6074.ckpt` | 4 | 0.6074 | 389,842,693 | `checkpoints_backup/kern_distill03-ep04-val0.6074.ckpt` | ruled-out run (distillation lambda 0.3); hard pieces 15.58 (`benchmark/tuplet_distill03.json`), DATA_SCALING_DERISK.md |
| `legacy__checkpoints_backup__gpu_full__kern_distill03__last.ckpt` | 5 | 0.6074 | 389,842,693 | `legacy/checkpoints_backup/gpu_full/kern_distill03/last.ckpt` and `legacy/checkpoints_backup/gpu_full/kern_distill03/kern_distill03-epoch=05-val/total=0.6074.ckpt` | ruled-out run, DATA_SCALING_DERISK.md |
| `checkpoints_backup__kern_distill_sel-ep04-val0.6169.ckpt` | 4 | 0.6169 | 389,842,693 | `checkpoints_backup/kern_distill_sel-ep04-val0.6169.ckpt` | ruled-out run (stream-selective distillation); hard pieces 13.82 (`benchmark/tuplet_distill_sel.json`), DATA_SCALING_DERISK.md |
| `legacy__checkpoints_backup__gpu_full__kern_distill_sel__last.ckpt` | 5 | 0.6169 | 389,842,693 | `legacy/checkpoints_backup/gpu_full/kern_distill_sel/last.ckpt` and `legacy/checkpoints_backup/gpu_full/kern_distill_sel/kern_distill_sel-epoch=05-val/total=0.6169.ckpt` | ruled-out run, DATA_SCALING_DERISK.md |
| `legacy__checkpoints_backup__gpu_full__broad_mix__broad_mix-epoch-11-val__total-0.6441.ckpt` | 11 | 0.6441 | 389,827,525 | `legacy/checkpoints_backup/gpu_full/broad_mix/broad_mix-epoch=11-val/total=0.6441.ckpt` | ruled-out run (988-piece broad corpus, 15 epochs, lr 1e-4, 50/50), DATA_SCALING_DERISK.md |
| `checkpoints_backup__broad_mix-ep14-val0.6435.ckpt` | 14 | 0.6435 | 389,827,525 | `checkpoints_backup/broad_mix-ep14-val0.6435.ckpt` and `legacy/checkpoints_backup/gpu_full/broad_mix/last.ckpt` | ruled-out run; val below the released model's 0.674 yet hard pieces 17.81 (`benchmark/tuplet_broad.json`), DATA_SCALING_DERISK.md |

### 4.3 From-scratch de-risk with the bidirectional recipe (generation collapse)

Three runs on ASAP plus the broad kern corpus (1,738 pieces) that trained the released model's
non-causal decoder configuration from scratch. Teacher-forced loss converged; inference produced no
notes (constant pad = 0). Ruled out by the "From-scratch de-risk" and "RECIPE HUNT + VIABILITY TEST"
sections of `docs/reports/DATA_SCALING_DERISK.md`. These checkpoints are useful only as negative
examples; `benchmark/_scratch_gen.json` is the generation check (MeanER null on all three pieces).

| Asset file | Epoch | val/total | Bytes | Original path(s) | Status |
|---|---|---|---|---|---|
| `checkpoints_backup__scratch_calib-ep24-val0.1104.ckpt` | 24 | 0.1104 | 389,823,173 | `checkpoints_backup/scratch_calib-ep24-val0.1104.ckpt` | ruled-out run (default recipe, 30 epochs; generates 0 notes, `benchmark/_scratch_gen.json`), DATA_SCALING_DERISK.md |
| `legacy__checkpoints_backup__gpu_full__scratch_calib__last.ckpt` | 29 | 0.1111 | 389,823,173 | `legacy/checkpoints_backup/gpu_full/scratch_calib/last.ckpt` and `legacy/checkpoints_backup/gpu_full/scratch_calib/scratch_calib-epoch=29-val/total=0.1111.ckpt` | ruled-out run, DATA_SCALING_DERISK.md |
| `legacy__checkpoints_backup__gpu_full__scratch_fulldrop__scratch_fulldrop-epoch-09-val__total-1.1831.ckpt` | 9 | 1.1831 | 389,823,173 | `legacy/checkpoints_backup/gpu_full/scratch_fulldrop/scratch_fulldrop-epoch=09-val/total=1.1831.ckpt` | ruled-out run (`--decoder-full-drop 0.5`; still collapses), DATA_SCALING_DERISK.md |
| `legacy__checkpoints_backup__gpu_full__scratch_fulldrop__last.ckpt` | 10 | 1.0828 | 389,823,173 | `legacy/checkpoints_backup/gpu_full/scratch_fulldrop/last.ckpt` and `legacy/checkpoints_backup/gpu_full/scratch_fulldrop/scratch_fulldrop-epoch=10-val/total=1.0828.ckpt` | ruled-out run, DATA_SCALING_DERISK.md |
| `legacy__checkpoints_backup__gpu_full__scratch_long__scratch_long-epoch-34-val__total-0.0614.ckpt` | 34 | 0.0614 | 389,823,173 | `legacy/checkpoints_backup/gpu_full/scratch_long/scratch_long-epoch=34-val/total=0.0614.ckpt` | ruled-out run (150-epoch viability test, about 16K steps; constant degenerate output), DATA_SCALING_DERISK.md |
| `legacy__checkpoints_backup__gpu_full__scratch_long__last.ckpt` | 35 | 0.0622 | 389,823,173 | `legacy/checkpoints_backup/gpu_full/scratch_long/last.ckpt` and `legacy/checkpoints_backup/gpu_full/scratch_long/scratch_long-epoch=35-val/total=0.0622.ckpt` | ruled-out run (the curve continues to epoch 149 at val 6.38; this `last.ckpt` is the epoch-35 save), DATA_SCALING_DERISK.md |

### 4.4 Causal-AR from scratch on synthetic renders (the road to the SSL runs)

The `--autoregressive` trainer that solved the generation collapse, trained on ASAP plus rendered
PDMX or kern pairs. Accuracy improved from about 50 to about 20 MeanER across the builds but never
reached the released model, and three data levers (more PDMX, kern repetition, rubato augmentation)
failed. All superseded by the masked-SSL runs of 4.5. Report: `docs/reports/AR_BUILD_RESULTS.md`.
Hard-piece evaluations: `benchmark/_ar*_eval.json`, `benchmark/_arfull_best.json`.

| Asset file | Epoch | val/total | Bytes | Original path(s) | Status |
|---|---|---|---|---|---|
| `legacy__checkpoints_backup__gpu_full__scratch_ar__scratch_ar-epoch-28-val__total-2.6320.ckpt` | 28 | 2.6320 | 389,823,173 | `legacy/checkpoints_backup/gpu_full/scratch_ar/scratch_ar-epoch=28-val/total=2.6320.ckpt` | ruled-out run (first causal-AR build, ASAP and broad kern, 30 epochs, before the double-shift and dropout fixes), AR_BUILD_RESULTS.md |
| `checkpoints_backup__scratch_ar-ep29-val2.6270.ckpt` | 29 | 2.6270 | 389,823,173 | `checkpoints_backup/scratch_ar-ep29-val2.6270.ckpt` and `legacy/checkpoints_backup/gpu_full/scratch_ar/last.ckpt` | ruled-out run; Scriabin 66.48 (`benchmark/_ar_eval.json`), AR_BUILD_RESULTS.md |
| `checkpoints_backup__ar_fix-ep28-val2.5528.ckpt` | 28 | 2.5528 | 389,823,109 | `checkpoints_backup/ar_fix-ep28-val2.5528.ckpt` | ruled-out run (same data with the two training bugs fixed; three pieces mean 27.95, `benchmark/_arfix_eval.json`), superseded by ar_full3, AR_BUILD_RESULTS.md |
| `legacy__checkpoints_backup__gpu_full__ar_fix__last.ckpt` | 29 | 2.5528 | 389,823,109 | `legacy/checkpoints_backup/gpu_full/ar_fix/last.ckpt` and `legacy/checkpoints_backup/gpu_full/ar_fix/ar_fix-epoch=29-val/total=2.5528.ckpt` | ruled-out run, AR_BUILD_RESULTS.md |
| `legacy__checkpoints_backup__gpu_full__ar_long__ar_long-epoch-00-val__total-7.0072.ckpt` | 0 | 7.0072 | 389,822,917 | `legacy/checkpoints_backup/gpu_full/ar_long/ar_long-epoch=00-val/total=7.0072.ckpt` | abandoned after one epoch (`global_step` 129; the curve logs step 128); no report mentions this run, only `benchmark/curves/ar_long.metrics.csv` and its `hparams.yaml` (lr 3e-4, warmup 256, total_steps 12,840) survive |
| `legacy__checkpoints_backup__gpu_full__ar_long__last.ckpt` | 0 (verified by loading: epoch 0, `global_step` 129, the same epoch and step as the row above; the 64-byte size difference is consistent with the `last_model_path` entry in the callback state, which is empty in the named file and set in this one) | 7.0072 | 389,822,981 | `legacy/checkpoints_backup/gpu_full/ar_long/last.ckpt` | abandoned run, no report |
| `legacy__checkpoints_backup__gpu_full__ar_full__ar_full-epoch-04-val__total-1.2821.ckpt` | 4 | 1.2821 | 389,823,173 | `legacy/checkpoints_backup/gpu_full/ar_full/ar_full-epoch=04-val/total=1.2821.ckpt` | ruled-out run (ASAP, 31,777 PDMX renders, 988 kern; buggy trainer), AR_BUILD_RESULTS.md |
| `checkpoints_backup__ar_full-ep10-val1.2629.ckpt` | 10 | 1.2629 | 389,823,173 | `checkpoints_backup/ar_full-ep10-val1.2629.ckpt` | ruled-out run; five pieces mean 53.80, 0 tuplets on the four pieces that contain them and 388 spurious tuplets on Prokofiev (`benchmark/_arfull_best.json`), AR_BUILD_RESULTS.md |
| `legacy__checkpoints_backup__gpu_full__ar_full2__ar_full2-epoch-01-val__total-1.6004.ckpt` | 1 | 1.6004 | 389,823,173 | `legacy/checkpoints_backup/gpu_full/ar_full2/ar_full2-epoch=01-val/total=1.6004.ckpt` | ruled-out run (85K pairs, buggy trainer), AR_BUILD_RESULTS.md |
| `checkpoints_backup__ar_full2-ep02-val1.4901.ckpt` | 2 | 1.4901 | 389,823,173 | `checkpoints_backup/ar_full2-ep02-val1.4901.ckpt` | ruled-out run; five pieces mean 45.67 (`benchmark/_ar2_eval.json`), AR_BUILD_RESULTS.md |
| `legacy__checkpoints_backup__gpu_full__ar_full3__ar_full3-epoch-04-val__total-0.9835.ckpt` | 4 | 0.9835 | 389,823,173 | `legacy/checkpoints_backup/gpu_full/ar_full3/ar_full3-epoch=04-val/total=0.9835.ckpt` | superseded epoch of ar_full3, AR_BUILD_RESULTS.md |
| `checkpoints_backup__ar_full3-ep09-val0.9696.ckpt` | 9 | 0.9696 | 389,823,173 | `checkpoints_backup/ar_full3-ep09-val0.9696.ckpt` | best synthetic-render AR checkpoint (85K pairs, fixed trainer); five pieces mean 19.97, Mozart 12.45 (`benchmark/_ar3_eval.json`); superseded by the SSL runs, AR_BUILD_RESULTS.md |
| `legacy__checkpoints_backup__gpu_full__ar_full4__ar_full4-epoch-01-val__total-1.1352.ckpt` | 1 | 1.1352 | 389,823,173 | `legacy/checkpoints_backup/gpu_full/ar_full4/ar_full4-epoch=01-val/total=1.1352.ckpt` | ruled-out run (171K PDMX renders; regressed), AR_BUILD_RESULTS.md |
| `checkpoints_backup__ar_full4-ep02-val1.1188.ckpt` | 2 | 1.1188 | 389,823,173 | `checkpoints_backup/ar_full4-ep02-val1.1188.ckpt` | ruled-out run; five pieces mean 23.43 (`benchmark/_ar4_eval.json`), AR_BUILD_RESULTS.md |
| `legacy__checkpoints_backup__gpu_full__ar_full5__ar_full5-epoch-03-val__total-1.0112.ckpt` | 3 | 1.0112 | 389,823,173 | `legacy/checkpoints_backup/gpu_full/ar_full5/ar_full5-epoch=03-val/total=1.0112.ckpt` | ruled-out run (kern repeated 30 times; val worse than ar_full3), AR_BUILD_RESULTS.md |
| `legacy__checkpoints_backup__gpu_full__ar_full5__ar_full5-epoch-04-val__total-0.9895.ckpt` | 4 | 0.9895 | 389,823,173 | `legacy/checkpoints_backup/gpu_full/ar_full5/ar_full5-epoch=04-val/total=0.9895.ckpt` | ruled-out run, AR_BUILD_RESULTS.md |
| `legacy__checkpoints_backup__gpu_full__ar_rubato__ar_rubato-epoch-00-val__total-1.0908.ckpt` | 0 | 1.0908 | 389,822,917 | `legacy/checkpoints_backup/gpu_full/ar_rubato/ar_rubato-epoch=00-val/total=1.0908.ckpt` | ruled-out run (warm start from ar_full3 on the rubato-augmented mix; diverged, no tail gain), AR_BUILD_RESULTS.md |
| `legacy__checkpoints_backup__gpu_full__ar_rubato__last.ckpt` | 1 | 1.2508 | 389,823,173 | `legacy/checkpoints_backup/gpu_full/ar_rubato/last.ckpt` and `legacy/checkpoints_backup/gpu_full/ar_rubato/ar_rubato-epoch=01-val/total=1.2508.ckpt` | ruled-out run (killed at epoch 2), AR_BUILD_RESULTS.md |

### 4.5 Masked-SSL from scratch (the runs that closed the gap, except the one kept live)

Same causal-AR trainer, data strategy switched to 50/50 real ASAP pairs and real unpaired scores with
masked self-supervision. Each run isolated one lever; `ssl_classical_clean` (live, section 1.3) is the
leak-filtered rerun of `ssl_classical`. Reports: `docs/reports/AR_BUILD_RESULTS.md` (sections from
"THE SOTA GAP IS THE DATA STRATEGY" onward) and `docs/reports/LAB_REPORT.md` sections 4 to 6. Training
configs and curves for `ssl_unpaired`, `ssl_v2`, `ssl_classical` and `ssl_recipe` are in
`checkpoints_backup/gpu_full/<run>/lightning_logs/version_0/`; `ssl_bigc` has
`legacy/checkpoints_backup/gpu_full/ssl_bigc/lightning_logs/version_0/` (lr 3e-4, warmup 507,
total_steps 25,380) and `benchmark/curves/ssl_bigc.metrics.csv`. No per-piece JSON survives for these
five runs; their numbers live only in the report text.

| Asset file | Epoch | val/total | Bytes | Original path(s) | Status |
|---|---|---|---|---|---|
| `checkpoints_backup__gpu_full__ssl_unpaired__ssl_unpaired-epoch-00-val__total-0.5952.ckpt` | 0 | 0.5952 | 389,827,903 | `checkpoints_backup/gpu_full/ssl_unpaired/ssl_unpaired-epoch=00-val/total=0.5952.ckpt` | ruled-out run (170k pop PDMX unpaired; overfits after epoch 0; Mozart 8.85, Scriabin 12.68), superseded by ssl_v2 and ssl_classical, AR_BUILD_RESULTS.md |
| `checkpoints_backup__gpu_full__ssl_unpaired__ssl_unpaired-epoch-01-val__total-0.6033.ckpt` | 1 | 0.6033 | 389,828,095 | `checkpoints_backup/gpu_full/ssl_unpaired/ssl_unpaired-epoch=01-val/total=0.6033.ckpt` | ruled-out run, AR_BUILD_RESULTS.md |
| `checkpoints_backup__gpu_full__ssl_v2__ssl_v2-epoch-04-val__total-0.5645.ckpt` | 4 | 0.5645 | 389,828,095 | `checkpoints_backup/gpu_full/ssl_v2/ssl_v2-epoch=04-val/total=0.5645.ckpt` | ruled-out run (58k pop PDMX, exposure-matched; Mozart 7.37 but Scriabin 15.15), superseded by ssl_classical, AR_BUILD_RESULTS.md |
| `checkpoints_backup__gpu_full__ssl_v2__ssl_v2-epoch-06-val__total-0.5714.ckpt` | 6 | 0.5714 | 389,828,095 | `checkpoints_backup/gpu_full/ssl_v2/ssl_v2-epoch=06-val/total=0.5714.ckpt` | ruled-out run, AR_BUILD_RESULTS.md |
| `checkpoints_backup__gpu_full__ssl_classical__ssl_classical-epoch-10-val__total-0.5186.ckpt` | 10 | 0.5186 | 389,828,095 | `checkpoints_backup/gpu_full/ssl_classical/ssl_classical-epoch=10-val/total=0.5186.ckpt` | superseded run (genre=classical corpus before the leak filter; Mozart 5.72, Scriabin 12.67, Ravel 20.14); replaced by ssl_classical_clean after the leak was found, LAB_REPORT.md section 5 and AR_BUILD_RESULTS.md |
| `checkpoints_backup__gpu_full__ssl_classical__ssl_classical-epoch-11-val__total-0.5232.ckpt` | 11 | 0.5232 | 389,828,095 | `checkpoints_backup/gpu_full/ssl_classical/ssl_classical-epoch=11-val/total=0.5232.ckpt` | superseded run, LAB_REPORT.md section 5 |
| `checkpoints_backup__gpu_full__ssl_recipe__ssl_recipe-epoch-12-val__total-0.5194.ckpt` | 12 | 0.5194 | 389,828,095 | `checkpoints_backup/gpu_full/ssl_recipe/ssl_recipe-epoch=12-val/total=0.5194.ckpt` | ruled-out run (released schedule: 4,000 warmup, 40,252-step cosine; worse minimum than 0.5125, then overfit), AR_BUILD_RESULTS.md and LAB_REPORT.md section 6 |
| `checkpoints_backup__gpu_full__ssl_recipe__ssl_recipe-epoch-14-val__total-0.5263.ckpt` | 14 | 0.5263 | 389,828,095 | `checkpoints_backup/gpu_full/ssl_recipe/ssl_recipe-epoch=14-val/total=0.5263.ckpt` | ruled-out run, AR_BUILD_RESULTS.md |
| `checkpoints_backup__ssl_bigc-ep11-val0.4996.ckpt` | 11 | 0.4996 | 389,828,095 | `checkpoints_backup/ssl_bigc-ep11-val0.4996.ckpt` | ruled-out run (29,181-score merged classical corpus; the epoch the reports evaluated: best val at the time of writing, 0.4996, but over-produces tuplets: Mozart 7.6 with 166 tuplets against 36, Ravel 24.74, Scriabin 11.81), AR_BUILD_RESULTS.md and LAB_REPORT.md section 6 |
| `checkpoints_backup__ssl_bigc-ep14-val0.4975.ckpt` | 14 | 0.4975 | 389,828,095 | `checkpoints_backup/ssl_bigc-ep14-val0.4975.ckpt` and `legacy/checkpoints_backup/gpu_full/ssl_bigc/ssl_bigc-epoch=14-val/total=0.4975.ckpt` | ruled-out run (the lowest val/total of any run, 0.4975, reached after the report's evaluation; no MUSTER record of this epoch survives, the run's MUSTER numbers belong to epoch 11; `legacy/docs/reports/GPU_CHECKPOINT_INVENTORY.md` lists it as "Best val (over-produces on MUSTER)"), AR_BUILD_RESULTS.md |

### 4.6 Superseded epochs of the two live own runs

| Asset file | Epoch | val/total | Bytes | Original path(s) | Status |
|---|---|---|---|---|---|
| `checkpoints_backup__ssl_classical_clean-ep06-val0.5335.ckpt` | 6 | 0.5335 | 389,828,159 | `checkpoints_backup/ssl_classical_clean-ep06-val0.5335.ckpt` | superseded epoch of the live run (epoch 13 is live); corpus 13.54 (`benchmark/rerank/ssl_classical_clean__ssl_classical_clean-epoch_06-val_total_0.5335.json`), LAB_REPORT.md |
| `checkpoints_backup__ssl_tuplet20-ep01-val0.5234.ckpt` | 1 | 0.5234 | 389,828,159 | `checkpoints_backup/ssl_tuplet20-ep01-val0.5234.ckpt` and `legacy/checkpoints_backup/gpu_full/ssl_tuplet20/ssl_tuplet20-epoch=01-val/total=0.5234.ckpt` | superseded epoch of the live run: the val-best epoch, but corpus 12.20 (`benchmark/ssl_tuplet20_best.json`) and 12.15 on re-evaluation (`benchmark/rerank/ssl_tuplet20__ssl_tuplet20-epoch_01-val_total_0.5234.json`) against 11.87 for the live epoch-3 file, LAB_REPORT.md section 11 |

### 4.7 The corpus-reshape lever

`ssl_reshape_g1`: warm start from the SSL baseline with tuplet-rich scores upsampled (gamma 1.0,
manifest `data/pairs_classical_clean_tuplrate.csv`). Best reshape result, 12.08 (val-best epoch), still
short of 11.87; gamma 1 fixed the Ravel blowup but did not recover Haydn. Ruled out by
`docs/reports/LAB_REPORT.md` ("The corpus-reshape lever").

| Asset file | Epoch | val/total | Bytes | Original path(s) | Status |
|---|---|---|---|---|---|
| `checkpoints_backup__ssl_reshape_g1-ep00-val0.5322.ckpt` | 0 | 0.5322 | 389,827,903 | `checkpoints_backup/ssl_reshape_g1-ep00-val0.5322.ckpt` | ruled-out run; corpus 12.08 (`benchmark/ssl_reshape_g1_best.json`), 12.01 on re-evaluation (`benchmark/rerank/ssl_reshape_g1__ssl_reshape_g1-epoch_00-val_total_0.5322.json`), LAB_REPORT.md |
| `checkpoints_backup__ssl_reshape_g1-last.ckpt` | 2 | 0.5460 | 389,828,159 | `checkpoints_backup/ssl_reshape_g1-last.ckpt` and `checkpoints_backup/ssl_reshape_g1-ep02-val0.5460.ckpt` | ruled-out run; corpus 12.69 (`benchmark/ssl_reshape_g1_last.json`), 12.73 and 12.92 on re-evaluation (`benchmark/rerank/ssl_reshape_g1__ssl_reshape_g1_last.json`, `..._epoch_02_...json`), LAB_REPORT.md |

### 4.8 Runs with evaluation records but no surviving checkpoint

Ten runs were evaluated on the box but no checkpoint of theirs exists anywhere. For four of them
`legacy/docs/reports/GPU_CHECKPOINT_INVENTORY.md` records that the directory was removed in a disk
cleanup; the other six were never copied off the box. Their only trace is the eval JSON named below,
corpus MeanER at threshold 0.50 over 14 pieces (one file has 13 scored), plus, for the last five, the
training log in `legacy/benchmark/box_logs/<run>.log` from which the configuration column is taken.
All of them ran the masked-SSL recipe with warmup 300 and 15 epochs. The first five and `ssl_pseudo`
warm-started from the SSL baseline (the `INIT_CKPT` default of `scripts/run_ssl_tuplet.sh` and the
explicit `INIT` of `scripts/build_and_train_st.sh`); for `ssl_bal_g0.5`, `ssl_bal_g1.0`, `ssl_balanced`
and `ssl_real07` the init path is not logged and the same source is inferred.

| Run | What it was | Evidence (corpus MeanER) |
|---|---|---|
| `ssl_tuplet5` | tuplet weight 5.0, 15 epochs | `benchmark/tuplet5_ep06.json` 13.04, `tuplet5_last.json` 13.81 |
| `ssl_tuplet25` | tuplet weight 2.5, 15 epochs | `benchmark/ssl_tuplet25_best.json` 12.45, `ssl_tuplet25_last.json` 12.46 |
| `ssl_tuplet20e30` | tuplet weight 2.0, 30 epochs | `benchmark/ssl_tuplet20e30_best.json` 12.24, `ssl_tuplet20e30_last.json` 12.33 |
| `ssl_combo` | tuplet weight 2.0 plus reshape gamma 1.0 | `benchmark/ssl_combo_best.json` 12.30, `ssl_combo_last.json` 12.25 |
| `ssl_reshape_g2` | reshape gamma 2.0, tuplet weight 1.0 | `benchmark/ssl_reshape_g2_best.json` 12.51, `ssl_reshape_g2_last.json` 12.32 |
| `ssl_bal_g0.5` | reshape gamma 0.5 with tuplet weight 20.0, real fraction 0.5, 21,402 unpaired | `benchmark/rerank/ssl_bal_g0.5__*.json` 13.46 / 13.10 / 13.41 |
| `ssl_bal_g1.0` | reshape gamma 1.0 with tuplet weight 20.0, real fraction 0.5, 21,402 unpaired | `benchmark/rerank/ssl_bal_g1.0__*.json` 14.82 (13 pieces) / 13.53 / 13.74 |
| `ssl_balanced` | reshape gamma 1.5 with tuplet weight 20.0, real fraction 0.5, 21,402 unpaired | `benchmark/rerank/ssl_balanced__*.json` 16.00 / 16.07 / 15.68 |
| `ssl_pseudo` | tuplet weight 5.0, real fraction 0.5, 23,562 unpaired including MAESTRO pseudo-pairs built from 200 MAESTRO pieces (`pseudo_sh0.log`, `pseudo_sh1.log`); that data is gone too, see `DATA.md` | `benchmark/rerank/ssl_pseudo__*.json` 13.56 / 13.32 / 13.31 |
| `ssl_real07` | real fraction 0.7 with tuplet weight 20.0, 21,402 unpaired | `benchmark/rerank/ssl_real07__*.json` 15.02 / 13.65 / 13.94 |

None of these beat the live `ssl_tuplet20` file. No B2 beat-relative checkpoint survives either
(`legacy/benchmark/box_logs/ssl_b2*.log` are the only record of those runs).

---

## 5. Architecture and checkpoint anatomy

### 5.1 The model

All own checkpoints and the released checkpoint are the same network, defined in
`MIDI2ScoreTransformer/midi2scoretransformer/models/roformer.py` (class `Roformer`, a Lightning module
over a modified HuggingFace RoFormer: pre-norm, SwiGLU, fused QKV, RoPE in cross-attention) with the
configuration class `MyModelConfig` in `config.py` and the stream embeddings in `models/embedding.py`.
`docs/reports/AR_BUILD_RESULTS.md` ("Diagnosis (direct checkpoint inspection)") compared the released
file and an own file parameter by parameter.

- **32,595,514 parameters** (201 tensors in `state_dict`), encoder about 13.76M and decoder about 17.97M.
  Re-verified on `ssl_tuplet20/last.ckpt` while writing this file.
- RoFormer encoder, 4 layers; RoFormer decoder, 4 layers with cross-attention; hidden size 512,
  8 attention heads, feed-forward 1536, RoPE positions, max positions 1536, sequence length 512 at
  training and inference (`--chunk-size 512`, overlap 64 in the evaluation harness, 128 in
  `scripts/infer_with_ckpt.py`).
- The decoder is **causal and autoregressive** (`is_autoregressive=True` in the decoder config of the
  released checkpoint and of every own run from `scratch_ar` onward; the encoder keeps
  `is_autoregressive=False`). The 4.1 and 4.3 checkpoints have a bidirectional decoder and do not generate.
- **Inputs**: 4 MIDI streams (onset 200 buckets, duration 200, pitch 128, velocity 8) plus a 1-dim
  `unconditional` conditioning input (the masked-SSL flag). Beat-conditioned checkpoints (4.2) add a
  13-bucket `beat` stream on the encoder.
- **Outputs**: 13 score streams, each its own linear head: offset (145), downbeat (146), duration (97),
  pitch (128), accidental (7), keysignature (16), velocity (8), grace (2), trill (2), staccato (2),
  voice (9), stem (4), hand (3); plus a 1-dim pad (keep or drop) head, `mask_embeddings`. Offsets,
  downbeats and durations live on a 1/24-quarter grid, which is why phase 24 appears in the priors.
- Loss weights per stream are the `loss_weight` entries of `FEATURES` in `config.py`; the tuplet
  weight of `ssl_tuplet20` multiplies the non-dyadic buckets of duration, offset and downbeat inside
  that cross-entropy (`train.py`, `_tuplet_weight`).

### 5.2 Lightning checkpoint anatomy

Measured on `MIDI2ScoreTransformer/checkpoints/ssl_tuplet20/last.ckpt`; the other files differ by a
few hundred bytes of metadata unless noted in section 4.

| Key | What it holds | Size |
|---|---|---|
| `state_dict` | the 201 weight tensors, float32 | about 130 MB (130,459,373 bytes serialised) |
| `optimizer_states` | AdamW first and second moments for every parameter | about 259 MB (259,350,293 bytes serialised) |
| `lr_schedulers`, `loops`, `callbacks` | cosine schedule state, Lightning loop counters, `ModelCheckpoint` state including `best_k_models` and `best_model_path` as recorded on the training machine | small |
| `hyper_parameters`, `hparams_name` | the `enc_configuration`, `dec_configuration` and the recipe dict; for warm-started runs inherited from the init checkpoint (see 1.2) | small |
| `epoch`, `global_step`, `pytorch-lightning_version` | the ground truth for "which epoch is this" | tiny |

Rules that follow:

- **Checkpoints do not compress.** Float32 weights and moments are incompressible, which is why the
  release ships every `.ckpt` raw while everything else is a `.tar.zst`.
- **Never strip optimizer state from the archived files.** The archive exists so that any run can be
  resumed or re-examined exactly; the two thirds of each file that is optimizer state is the part a
  resume needs. Make a stripped copy elsewhere if you need a small inference-only file.
- Trainer settings on the GPU box: `ModelCheckpoint(monitor="val/total", mode="min", save_top_k=2,
  save_last=True)` (`train.py`). That is why each run left two named epoch files plus `last.ckpt`, and
  why the first-paragraph warning about `last.ckpt` matters.
- Environment the files were written under: the GPU box ran torch 2.12.0, lightning 2.6.5,
  transformers 5.10.2, tokenizers 0.22.2, numpy 2.4.6 (`legacy/benchmark/box_logs/pip_install.log`);
  the April 2026 files were written on the Mac. The pinned Mac environment (`requirements.lock`: torch
  2.11.0, lightning 2.6.1, transformers 4.42.4, tokenizers 0.19.1, numpy 1.26.4) loads them; the
  compatibility shim in section 6 is what makes the transformers difference harmless.

---

## 6. Loading any checkpoint

### 6.1 Command line

`scripts/infer_with_ckpt.py` runs one MIDI through one checkpoint on CPU and writes MusicXML. The
example input below is the MAESTRO performance MIDI installed by the `benchmark-pieces.tar.zst`
quickstart asset; before that asset is fetched the only file in `benchmark/chopin_op10/midi/` is the
hFT transcription `Op10_No4_CsharpMinor_hft.mid`, which works as input too.

```bash
python scripts/infer_with_ckpt.py benchmark/chopin_op10/midi/Op10_No4_CsharpMinor.midi \
    --ckpt MIDI2ScoreTransformer/checkpoints/ssl_tuplet20/last.ckpt \
    --out outputs/op10_tuplet20.musicxml \
    --pad-threshold 0.5 --top-k 1 --temperature 1.0 --chunk-size 512 --overlap 128
```

The evaluation scripts take `--ckpt` too, but `benchmark/eval_tier1_asap.py`, `eval_padsweep.py`,
`eval_tuplet.py` and `eval_decomposed.py` change directory into `MIDI2ScoreTransformer/` first, so pass
either an absolute path or one relative to that directory (`checkpoints/MIDI2ScoreTF.ckpt`).
`transcribe.py -b transformer` always uses the released checkpoint. Run on `cpu` or `cuda`; never on
Apple `mps` (it corrupts the pad logits, see the README).

### 6.2 In Python

```python
import sys, torch
sys.path.insert(0, "MIDI2ScoreTransformer/midi2scoretransformer")

from config import MyModelConfig
if not hasattr(MyModelConfig, "_attn_implementation_internal"):   # transformers version shim
    MyModelConfig._attn_implementation_internal = None
torch.serialization.add_safe_globals([MyModelConfig])              # the config is pickled inside the file

from models.roformer import Roformer
model = Roformer.load_from_checkpoint(
    "MIDI2ScoreTransformer/checkpoints/MIDI2ScoreTF.ckpt",
    map_location="cpu", weights_only=False, strict=False,
).eval()
```

`Roformer.load_from_checkpoint` rebuilds the network from the `enc_configuration` and
`dec_configuration` stored in the file, so beat-conditioned and SSL checkpoints come back with the
right extra layers without any flag. `strict=False` is needed for the two `kern_distill` files that
carry `_teacher.*` keys and is harmless otherwise. Checkpoints written by the trainer are instances of
`TrainableRoformer` (`train.py`); if the base class refuses one, `load_any_checkpoint` in
`benchmark/eval_tier1_asap.py` falls back to `TrainableRoformer.load_from_checkpoint` with the same
arguments, and that helper is the one to reuse. `load_pretrained_init` in `train.py` is the warm-start
path (loads a checkpoint, optionally adds the beat or beat-relative layers, filters shape mismatches).

To inspect without building the model:

```python
ck = torch.load(path, map_location="cpu", weights_only=False)
ck["epoch"], ck["global_step"]                      # the real epoch and step
ck["callbacks"]                                     # ModelCheckpoint state: best_k_models, best_model_path
sum(t.numel() for t in ck["state_dict"].values())   # 32595514
```

Inference goes through `utils.infer(x, model, kv_cache=True, overlap=..., chunk=..., top_k=...,
temperature=..., dur_log_pi=..., dur_metrical=..., offset_rerank=...)`, which calls `model.generate`
chunk by chunk, then `MultistreamTokenizer.detokenize_mxl(y_hat, pad_threshold=...)` and
`score_utils.postprocess_score`. `scripts/infer_with_ckpt.py` is the shortest complete example;
`benchmark/eval_padsweep.py` is the complete example with the priors.
