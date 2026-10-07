# benchmark/: the evaluation harness and every result it produced

Everything that measures a model lives here: the harness scripts, the three audio test pieces, and one JSON per evaluation. Each evaluation JSON names the checkpoint it scored in its `ckpt` field (or `meta.ckpt` for tier-1 files, or `models.<name>.ckpt` for the decomposition), so a number can always be traced to weights. The frozen-study records that have no `ckpt` field (`confirmatory_v10_1_*.json`, `oracle_prefix_v10_1.json`, `routed_selector_v10_n32_final_metric_v2.json`, `bestofn_secondary_audit.json`, `v8_*_metrics.json`) instead carry the sha256 of their input records; `oracle_prefix_v10_1.json` and `revision_locality_v10_1.json` also carry the released checkpoint's hash (`7b8ec6e3...`). The tables below list every tracked JSON in `benchmark/`, `benchmark/rerank/` and `benchmark/diag/` with the experiment, the checkpoint, the headline number read from the file, and the report that discusses it. Where no report cites a file, the table says so; the number is still in the file.

Checkpoint availability is marked in the checkpoint column: **quickstart** (`scripts/fetch_assets.sh quickstart`), **archive** (the `archive-checkpoints` tier, original path in `assets/ASSETS.csv`), or **no checkpoint** (the run's weights survive nowhere; only this evaluation and, for some, a training curve or a GPU box log remain). Ten runs are in that last state: `ssl_tuplet5`, `ssl_tuplet25`, `ssl_tuplet20e30`, `ssl_combo`, `ssl_reshape_g2`, `ssl_bal_g0.5`, `ssl_bal_g1.0`, `ssl_balanced`, `ssl_pseudo`, `ssl_real07`. The June 2026 fine-tuning runs (`finetune_full`, `arm1_mixed8020`, `ft_gentle_*`) are also absent from the asset list.

MeanER is MUSTER's mean error rate (mean of PitchER, MissRate, ExtraRate, OnsetER, OffsetER, each in percent); lower is better. TPF (tuplet-passage failure) is the exact count of missed explicit-tuplet ground-truth notes plus onset-error events on matched tuplet notes, defined in `scripts/tuplet_metrics_v2.py`.

## The test pieces

**Tier 1 (MIDI to score).** The ASAP test split: 14 scores, 59 performances. `eval_tier1_asap.py` scores all 59; `eval_padsweep.py` and `eval_tuplet.py` take one performance per score (`--limit-per 1`, the one with the fewest tokenized input notes), which is what "14-piece corpus mean" means everywhere in the reports. The best-of-N / support-selection study used four of the fourteen as development pieces (Scriabin, Mozart, Liszt, Ravel) and the other ten as the untouched confirmatory set; its confirmatory runner picks the first aligned performance in the dataset metadata rather than the shortest, so the diagnostic and confirmatory performances differ on six pieces (the exact MIDI names are tabulated in `research/supplement/PROVENANCE.md`).

| Composer | Score (path under `asap-dataset/`) | Role in the study |
|---|---|---|
| Bach | `Bach/Fugue/bwv_846` | confirmatory (control, 0 tuplets) |
| Beethoven | `Beethoven/Piano_Sonatas/10-1` | confirmatory |
| Brahms | `Brahms/Six_Pieces_op_118/2` | confirmatory |
| Chopin | `Chopin/Ballades/1` | confirmatory |
| Debussy | `Debussy/Images_Book_1/1_Reflets_dans_lEau` | confirmatory |
| Haydn | `Haydn/Keyboard_Sonatas/31-1` | confirmatory |
| Liszt | `Liszt/Annees_de_pelerinage_2/1_Gondoliera` | development (hard tail) |
| Mozart | `Mozart/Piano_Sonatas/12-1` | development (sparse-tuplet control) |
| Prokofiev | `Prokofiev/Toccata` | confirmatory (control, 0 tuplets) |
| Rachmaninoff | `Rachmaninoff/Preludes_op_23/4` | confirmatory (control) |
| Ravel | `Ravel/Gaspard_de_la_Nuit/1_Ondine` | development (densest tuplets) |
| Schubert | `Schubert/Impromptu_op.90_D.899/1` | confirmatory |
| Schumann | `Schumann/Arabeske` | confirmatory |
| Scriabin | `Scriabin/Etudes_op_8/11` | development (the main dev piece) |

The "hard 5" used by `eval_tuplet.py` in the data-scaling work are Scriabin, Mozart, Liszt, Ravel, Prokofiev; the earliest from-scratch evaluations used Bach, Scriabin, Mozart, Ravel, Prokofiev.

**Tier 2 (audio to score).** Three MAESTRO performances (CC BY-NC-SA 4.0), audio and MIDI installed by the quickstart tier into `benchmark/<piece>/audio/` and `midi/`; the ground-truth scores are in ASAP (`Chopin/Etudes_op_10/4`, `Chopin/Etudes_op_25/11`) and, for Mazeppa, `benchmark/liszt_transcendental/gt_score.musicxml` (sourced from PDMX, two-staff version; the raw three-staff file is kept beside it). `chopin_op10/`, `chopin_op25/` and `liszt_transcendental/` also hold `catalog.json`, MuseScore renders and the transformer outputs.

| Piece | Why it is here |
|---|---|
| Chopin Op. 10 No. 4 | in distribution (its score is in ASAP train) |
| Chopin Op. 25 No. 11 | in distribution |
| Liszt Transcendental Etude No. 4 "Mazeppa" | dense, the honest out-of-distribution test for the score model |

## Harness scripts and the commands that work

Run everything from the repository root with the `venv311` interpreter (`python` below). `eval_tier1_asap.py`, `eval_padsweep.py`, `eval_tuplet.py` and `diag_streams.py` all `os.chdir` into `MIDI2ScoreTransformer/` on import, so **`--ckpt` is relative to that directory** (`checkpoints/MIDI2ScoreTF.ckpt`, not `MIDI2ScoreTransformer/checkpoints/...`); absolute paths also work and are what the shell drivers use. A relative `--out`, by contrast, is resolved against the repository root (`REPO_ROOT / out`), so `--out benchmark/x.json` lands in this directory. Never pass `--device mps`.

| Script | What it measures | Working command |
|---|---|---|
| `eval_tier1_asap.py` | Paper-comparable tier 1: all 59 ASAP test performances, MUSTER plus the notation error rates (`score_similarity`). `--out` is required; `--ckpt` defaults to the released model. `--limit N` takes the N shortest performances for a smoke test; `--use-beat-conditioning` is for the beat-conditioned checkpoints; `--jobs` parallelises on CPU. | `python benchmark/eval_tier1_asap.py --ckpt checkpoints/MIDI2ScoreTF.ckpt --out benchmark/tier1_smoke.json --limit 3` |
| `eval_padsweep.py` | The standing 14-piece measurement: generate once per piece, then detokenize and score at every pad threshold (default `"0.60 0.55 0.50 0.45 0.40 0.35 0.30 0.25"`). `--ckpt` and `--out` are required. `--pieces` filters by composer substring; `--prior-path`, `--dur-tau`, `--dur-metrical-lambda` switch on the A1/A2 duration levers. Reports corpus mean per threshold. | `python benchmark/eval_padsweep.py --ckpt checkpoints/MIDI2ScoreTF.ckpt --device cpu --limit-per 1 --thresholds 0.50 --out benchmark/padsweep_smoke.json --pieces Mozart` |
| `eval_tuplet.py` | One threshold, chosen pieces: MeanER plus predicted versus ground-truth tuplet counts (the under-production ratio). `--pad-threshold`, `--pitch-top-k`, `--pitch-temp` are available. Per-piece SIGALRM guard against MUSTER hangs. | `python benchmark/eval_tuplet.py --ckpt checkpoints/MIDI2ScoreTF.ckpt --device cpu --pieces Scriabin Mozart --out benchmark/tuplet_smoke.json` |
| `diag_streams.py` | Full per-stream error decomposition (NoteDuration, NoteDeletion, TimeSignature, ...) for chosen pieces and one checkpoint; separates a tuplet collapse from a structural failure. | `python benchmark/diag_streams.py --ckpt checkpoints/MIDI2ScoreTF.ckpt --pieces Haydn Schumann` |
| `eval_tier2_e2e.py` | End to end audio to score through the real `transcribe.py -t hft -b transformer`, scored with MUSTER against the ground-truth MusicXML. Arguments after `--` go to `transcribe.py`. | `python benchmark/eval_tier2_e2e.py --out benchmark/tier2_smoke.json --tag smoke` |
| `eval_decomposed.py` | Splits the end-to-end error into Stage A (audio to MIDI, hFT scored with mir_eval note F1) and Stage B (MIDI to score on the ground-truth MIDI, MUSTER). | `python benchmark/eval_decomposed.py --out benchmark/decomposed_smoke.json --stage both` |
| `trackb_sweep.py` | Sweeps hFT onset and frame thresholds on Mazeppa and scores each end-to-end output with MUSTER. No arguments. | `python benchmark/trackb_sweep.py` |
| `compare_sweeps.py` | Prints per-piece MeanER, gap and over-production proxies for `padsweep_sslcc.json` versus `padsweep_released.json` at one threshold. | `python benchmark/compare_sweeps.py 0.50` |
| `compare_tuplet5.py` | Per-piece comparison of a best-val and a last checkpoint evaluation against `padsweep_sslcc.json` and `padsweep_released.json` at threshold 0.50. | `python benchmark/compare_tuplet5.py ssl_tuplet20_best.json ssl_tuplet20_last.json best last` |
| `compare_beat_eval.py` | Aligns the beat-conditioning arms A, B, C per performance and prints drift, beat-signal and net effects, hardest first. | `python benchmark/compare_beat_eval.py --a benchmark/eval_A_baseline.json --c benchmark/eval_C_beat_nobeat.json --b benchmark/eval_B_beat_gold.json` |
| `auto_eval_stage.sh` | HISTORICAL (box banner): waited for a training stage to finish, then evaluated its best-val and last checkpoints with `eval_padsweep.py` at 0.50, producing the `benchmark/<stage>_best.json` and `<stage>_last.json` pairs. | do not run; the equivalent today is two `eval_padsweep.py` calls |

Every tier-1 and 14-piece script needs the quickstart tier (the checkpoint, ASAP with its chunks, the tokenization cache) and the `muster` package from `requirements.lock`. On a Mac the released model takes roughly 40 minutes for the 59 performances on CPU (`tier1_baseline.log`).

## How to read each JSON family

| Family (producer) | Shape | Where the headline is |
|---|---|---|
| tier-1 (`eval_tier1_asap.py`) | `{meta, aggregate, per_performance[59]}` | `aggregate.muster.MeanER`; `per_performance[i].sim.muster` and `.sim["mxl <-> gt_mxl"]` |
| 14-piece sweep (`eval_padsweep.py`, also via `rerank_by_muster.sh` and `auto_eval_stage.sh`) | `{ckpt, thresholds, results[14]}`; each result has `per_threshold["0.50"].MeanER`, `.pred_tuplets`, `.pred_notes`, plus `gt_tuplets`, `gt_notes` | corpus mean = mean of `per_threshold[t].MeanER` over the 14 results (`compare_sweeps.py` does it) |
| tuplet check (`eval_tuplet.py`) | `{ckpt, results[n]}` with `MeanER`, `pred_tuplets`, `gt_tuplets` per piece | mean over results; `pred_tuplets / gt_tuplets` |
| best-of-N selector records (`scripts/*_v4..v10.py`) | `{ckpt, n, topk, temp, ..., pieces{key: {baseline, candidates, selector_pick or selected, route, gate_ok}}}` | `pieces[k].baseline.meaner_overall` and `.tpf` versus the pick's `metrics` |
| confirmatory and oracle (`confirmatory_v10_1.py`, `oracle_prefix_v10_1.py`) | documented inline: `decision_rule`, `n32_diagnosis`, `aggregate_prefix_curve`, `provenance` with every input hash | see the table |

## Every tracked JSON

### Tier 1, all 59 performances (`eval_tier1_asap.py`)

| File | Experiment | Checkpoint (`meta.ckpt`) | Headline | Report |
|---|---|---|---|---|
| `tier1_baseline.json` | Reproduction of the published number, CPU | `checkpoints/MIDI2ScoreTF.ckpt` (quickstart) | MeanER **11.18** (paper: 11.30); per-performance log in `tier1_baseline.log` | `BASELINE_RESULTS.md`, `README.md` |
| `eval_A_baseline.json` | Beat-conditioning arm A: released model, no beats, CUDA | `checkpoints/MIDI2ScoreTF.ckpt` (quickstart) | 11.16 | `docs/reports/BEAT_CONDITIONING_RESULTS.md` |
| `eval_B_beat_gold.json` | Arm B: beat-conditioned fine-tune with gold beats | `checkpoints/beat_asap/last.ckpt` (archive: `checkpoints_backup/beat_asap-last.ckpt`) | 15.44 | same |
| `eval_C_beat_nobeat.json` | Arm C: same checkpoint, no beats (drift test) | `checkpoints/beat_asap/last.ckpt` (archive) | 15.54 | same |
| `eval_Bp_v2ep4_gold.json` | Arm B', second beat run with gold beats | `checkpoints/beat_asap_v2/beat_asap_v2-epoch=04-val/total=0.6739.ckpt` (archive) | 13.40 | same |
| `eval_Cp_v2ep4_nobeat.json` | Arm C', second beat run, no beats | same (archive) | 13.59 | same |

### 14-piece corpus sweeps (`eval_padsweep.py`), corpus mean MeanER at threshold 0.50

| File | Experiment | Checkpoint (`ckpt`) | Headline | Report |
|---|---|---|---|---|
| `padsweep_released.json` | Released model across 8 pad thresholds; the standing reference | `MIDI2ScoreTF.ckpt` (quickstart) | **10.77** at 0.50 (10.88 at 0.60 down to 10.81 at 0.25; the gate is nearly flat) | `docs/reports/PAD_THRESHOLD_FIX.md`, `docs/reports/LAB_REPORT.md` sections 10 and 11, `README.md` |
| `padsweep_sslcc.json` | `ssl_classical_clean` epoch 13 across 8 thresholds | `ssl_classical_clean-epoch=13-val/total=0.5125.ckpt` (quickstart) | **12.69** at 0.50 (best 12.65 at 0.40 and 0.45) | same |
| `padsweep_preflight.json` | Preflight of the sweep on Mozart only, thresholds 0.5 and 0.4 | same (quickstart) | 5.76 / 5.75 | none |
| `ssl_tuplet20_last.json` | Tuplet loss weight 2.0, warm start from epoch 13; the `last` checkpoint of a run whose curve (`curves/ssl_tuplet20.metrics.csv`) spans epochs 0 to 14 | `ssl_tuplet20/last.ckpt` (quickstart; the file header says epoch 3, step 2780, val 0.5388, which matches the curve's epoch-3 row, step 2779, val 0.5388). `LAB_REPORT.md` section 11 calls this checkpoint "`last`/ep14"; which training state the 11.87 was scored on cannot be settled from the files, but the later re-evaluation of `last.ckpt` (11.79) sits next to that of the epoch-3 file (11.81) | **11.87**, the best own number; re-evaluated at 11.79 in `rerank/` | `LAB_REPORT.md` section 11, `legacy/docs/reports/GPU_CHECKPOINT_INVENTORY.md`, `README.md`, `RUNS.md` section 10 |
| `ssl_tuplet20_best.json` | Same run, best-validation epoch | `ssl_tuplet20-epoch=01-val/total=0.5234.ckpt` (archive) | 12.20 (validation loss and MUSTER disagree) | `LAB_REPORT.md` section 11 |
| `ssl_tuplet25_best.json` | Tuplet weight 2.5, best-val epoch | `ssl_tuplet25-epoch=05-val/total=0.5373.ckpt` (**no checkpoint**) | 12.45 | `LAB_REPORT.md` section 11 |
| `ssl_tuplet25_last.json` | Tuplet weight 2.5, last | `ssl_tuplet25/last.ckpt` (**no checkpoint**) | 12.46 | same |
| `tuplet5_ep06.json` | Tuplet weight 5.0, epoch 6 | `ssl_tuplet5-epoch=06-val/total=0.5209.ckpt` (**no checkpoint**) | 13.04 (over-corrected; Ravel 31.74) | same |
| `tuplet5_last.json` | Tuplet weight 5.0, last | `ssl_tuplet5/last.ckpt` (**no checkpoint**) | 13.81 | same |
| `ssl_tuplet20e30_best.json` | Weight 2.0 for 30 epochs, best-val | `ssl_tuplet20e30-epoch=00-val/total=0.5195.ckpt` (**no checkpoint**) | 12.24 (longer training did not help) | `LAB_REPORT.md` section 11, `GPU_CHECKPOINT_INVENTORY.md` |
| `ssl_tuplet20e30_last.json` | Weight 2.0 for 30 epochs, last | `ssl_tuplet20e30/last.ckpt` (**no checkpoint**) | 12.33 | same |
| `ssl_reshape_g1_best.json` | Corpus reshape gamma 1.0 (data lever), best-val | `ssl_reshape_g1-epoch=00-val/total=0.5322.ckpt` (archive) | 12.08 (Ravel 18.54 against the released model's 20.80; `ssl_classical_clean` itself scores 20.47 on Ravel and `ssl_tuplet20e30_best.json` 18.09, so this is not the only own model below the released number there) | `LAB_REPORT.md` section 11 |
| `ssl_reshape_g1_last.json` | Reshape gamma 1.0, last | `ssl_reshape_g1/last.ckpt` (archive, same bytes as the epoch 02 file) | 12.69 | same |
| `ssl_reshape_g2_best.json` | Reshape gamma 2.0, best-val | `ssl_reshape_g2-epoch=03-val/total=0.5699.ckpt` (**no checkpoint**) | 12.51 | same |
| `ssl_reshape_g2_last.json` | Reshape gamma 2.0, last | `ssl_reshape_g2/last.ckpt` (**no checkpoint**) | 12.32 (Haydn 5.12, Ravel 24.55) | same |
| `ssl_combo_best.json` | Weight 2.0 plus reshape gamma 1.0, best-val | `ssl_combo-epoch=04-val/total=0.5296.ckpt` (**no checkpoint**) | 12.30 | same |
| `ssl_combo_last.json` | Same, last | `ssl_combo/last.ckpt` (**no checkpoint**) | 12.25 (the levers do not compose) | same |

### `rerank/` (every saved epoch re-ranked by MUSTER, `rerank_by_muster.sh`), threshold 0.50, 14 pieces

File names are `<run>__<checkpoint name with / and = replaced by _>.json`. No report cites these by file name; `LAB_REPORT.md` section 11 covers the `ssl_tuplet20` and `ssl_reshape_g1` numbers, and the five balance / pseudo-label runs appear only here and in `legacy/benchmark/box_logs/` (`ssl_bal_g0.5.log`, `ssl_bal_g1.0.log`, `ssl_balanced.log`, `ssl_pseudo.log`, `ssl_real07.log`, `balanced_driver.log`, `rerank*.log`).

| File | Run and epoch | Checkpoint | Corpus mean |
|---|---|---|---|
| `ssl_tuplet20__ssl_tuplet20_last.json` | weight 2.0, last | quickstart (`ssl_tuplet20/last.ckpt`) | **11.79** |
| `ssl_tuplet20__ssl_tuplet20-epoch_03-val_total_0.5388.json` | weight 2.0, epoch 3 | the same training state as `last.ckpt` per `assets/ASSETS.csv` | 11.81 |
| `ssl_tuplet20__ssl_tuplet20-epoch_01-val_total_0.5234.json` | weight 2.0, epoch 1 (best val) | archive | 12.15 |
| `ssl_classical_clean__ssl_classical_clean-epoch_13-val_total_0.5125.json` | base model, epoch 13 | quickstart | 12.77 (12.69 in `padsweep_sslcc.json`; evaluation noise between runs) |
| `ssl_classical_clean__ssl_classical_clean-epoch_06-val_total_0.5335.json` | base model, epoch 6 | archive | 13.54 |
| `ssl_reshape_g1__ssl_reshape_g1-epoch_00-val_total_0.5322.json` | reshape gamma 1, epoch 0 | archive | 12.01 |
| `ssl_reshape_g1__ssl_reshape_g1-epoch_02-val_total_0.5460.json` | reshape gamma 1, epoch 2 | archive (`ssl_reshape_g1-last.ckpt`) | 12.92 |
| `ssl_reshape_g1__ssl_reshape_g1_last.json` | reshape gamma 1, last | archive | 12.73 |
| `ssl_bal_g0.5__ssl_bal_g0.5-epoch_03-val_total_0.5043.json` | balanced sampling gamma 0.5, epoch 3 | **no checkpoint** | 13.46 |
| `ssl_bal_g0.5__ssl_bal_g0.5-epoch_07-val_total_0.5031.json` | same, epoch 7 | **no checkpoint** | 13.10 |
| `ssl_bal_g0.5__ssl_bal_g0.5_last.json` | same, last | **no checkpoint** | 13.41 |
| `ssl_bal_g1.0__ssl_bal_g1.0-epoch_00-val_total_0.5049.json` | balanced sampling gamma 1.0, epoch 0 | **no checkpoint** | 14.82 (13 of 14 pieces scored) |
| `ssl_bal_g1.0__ssl_bal_g1.0-epoch_03-val_total_0.5070.json` | same, epoch 3 | **no checkpoint** | 13.53 |
| `ssl_bal_g1.0__ssl_bal_g1.0_last.json` | same, last | **no checkpoint** | 13.74 |
| `ssl_balanced__ssl_balanced-epoch_00-val_total_0.5089.json` | balanced corpus, epoch 0 | **no checkpoint** | 16.00 |
| `ssl_balanced__ssl_balanced-epoch_01-val_total_0.5090.json` | same, epoch 1 | **no checkpoint** | 16.07 |
| `ssl_balanced__ssl_balanced_last.json` | same, last | **no checkpoint** | 15.68 |
| `ssl_pseudo__ssl_pseudo-epoch_00-val_total_0.5232.json` | MAESTRO pseudo-label self-training (Track ST), epoch 0 | **no checkpoint**; training data **MISSING** (MAESTRO) | 13.56 |
| `ssl_pseudo__ssl_pseudo-epoch_04-val_total_0.5375.json` | same, epoch 4 | **no checkpoint** | 13.32 |
| `ssl_pseudo__ssl_pseudo_last.json` | same, last | **no checkpoint** | 13.31 |
| `ssl_real07__ssl_real07-epoch_00-val_total_0.5126.json` | real fraction 0.7, epoch 0 | **no checkpoint** | 15.02 |
| `ssl_real07__ssl_real07-epoch_01-val_total_0.5121.json` | same, epoch 1 | **no checkpoint** | 13.65 |
| `ssl_real07__ssl_real07_last.json` | same, last | **no checkpoint** | 13.94 |

### `diag/` (June 2026 fine-tuning diagnosis, `eval_tier1_asap.py` via `scripts/gpu_launchers/`)

The `.log` files beside them are the launcher logs; `gentle_chain.log`, `gentle_chain.done` and `GENTLE_REPORT.txt` record a chain whose evaluation JSONs (`gentle_2e6.json`, `gentle_1e6.json`) were never produced (the report file is the traceback).

| File | Experiment | Checkpoint (`meta.ckpt`) | Headline | Report |
|---|---|---|---|---|
| `baseline_cuda.json` | Released model on CUDA, all 59 | `MIDI2ScoreTF.ckpt` (quickstart) | 11.10 (CPU gives 11.18; device gap 0.08) | `GPU_FINETUNE_RESULTS.md` |
| `ft_epoch01.json` | Fine-tune on synthetic pairs, epoch 1, partial (26 of 59) | `finetune_full/pretrain_pdmx-epoch=01-val/total=0.2553.ckpt` (**not in the asset list**) | 17.73 (degradation present from epoch 1) | `GPU_FINETUNE_RESULTS.md` |
| `arm1_ep03.json` | ARM-1 80/20 real/synthetic mix, epoch 3, 2 performances | `arm1_mixed8020/arm1_mixed-epoch=03-val/total=0.6683.ckpt` (**not in the asset list**) | 8.41 on 2 | `GPU_FINETUNE_RESULTS.md` |
| `arm1_ep04.json` | ARM-1 epoch 4, 2 performances | `...epoch=04-val/total=0.6682.ckpt` (**not in the asset list**) | 8.49 on 2 | same |
| `arm1_last.json` | ARM-1 last, 39 of 59 scored | `arm1_mixed8020/last.ckpt` (**not in the asset list**) | 14.11 on 39 (13.51 on the 36 performances both runs had scored, against the released model's 11.88 on the same 36) | same |

The full 59-performance fine-tune evaluations are in `finetune_eval_full/eval_last.json` (MeanER 18.10, +6.92) and `finetune_eval_smoke/eval_last.json`, each with a `REPORT.txt`; both checkpoints are absent from the asset list.

### Hard-piece tuplet checks (`eval_tuplet.py`, single threshold 0.50)

Five pieces (Scriabin, Mozart, Liszt, Ravel, Prokofiev); headline is the mean MeanER over the five and the pooled predicted / ground-truth tuplet count (1,818 ground-truth tuplets). `docs/reports/DATA_SCALING_DERISK.md` quotes four-piece means that leave Prokofiev out (baseline 12.58, kern mix 18.01, distillation 14.38, broad 17.53); the table gives each file's five-piece mean, and both can be recomputed from `results[*].MeanER`.

| File | Experiment | Checkpoint (`ckpt`) | Headline | Report |
|---|---|---|---|---|
| `tuplet_baseline.json` | Released model on the hard 5 | `MIDI2ScoreTF.ckpt` (quickstart) | 11.87 mean; 1064 / 1818 tuplets | `docs/reports/DATA_SCALING_DERISK.md` |
| `tuplet_kernmix.json` | Warm start plus 50/50 ASAP and kern pairs, lr 1e-4 | `kern_mix/last.ckpt` (archive) | 17.58; 1160 / 1818 (more tuplets, +5.7 drift) | same |
| `tuplet_kernv2.json` | Same at lr 3e-5, real fraction 0.7 | `kern_mix_v2-epoch=03-val/total=0.6770.ckpt` (archive) | 16.81; 730 / 1818 | same |
| `tuplet_distill.json` | Kern mix with frozen-teacher distillation, lambda 1.0 | `kern_distill-epoch=02-val/total=0.5901.ckpt` (archive) | 13.78; 278 / 1818 | same |
| `tuplet_distill03.json` | Distillation lambda 0.3 | `kern_distill03-epoch=04-val/total=0.6074.ckpt` (archive) | 15.58; 81 / 1818 | same |
| `tuplet_distill_sel.json` | Stream-selective distillation | `kern_distill_sel-epoch=04-val/total=0.6169.ckpt` (archive) | 13.82; 82 / 1818 | same |
| `tuplet_broad.json` | Broad 988-piece kern corpus, 15 epochs, no distillation | `broad_mix-epoch=14-val/total=0.6435.ckpt` (archive) | 17.81; 230 / 1818 | same |

### From-scratch causal-AR builds (`eval_tuplet.py`, leading underscore files)

Bach, Scriabin, Mozart, Ravel, Prokofiev (1,416 ground-truth tuplets) unless noted. All in `docs/reports/AR_BUILD_RESULTS.md`.

| File | Experiment | Checkpoint (`ckpt`) | Headline |
|---|---|---|---|
| `_scratch_gen.json` | Mask-predict from-scratch run: generation check on 3 pieces | `scratch_calib-epoch=24-val/total=0.1104.ckpt` (archive) | `MeanER` null on every piece: 0 predicted notes (the all-pad collapse that motivated the causal-AR rebuild) |
| `_ar_eval.json` | First causal-AR from scratch, Scriabin only | `scratch_ar-epoch=29-val/total=2.6270.ckpt` (archive) | 66.48; 0 tuplets |
| `_arfix_eval.json` | After the two bug fixes, 3 pieces | `ar_fix-epoch=28-val/total=2.5528.ckpt` (archive) | 27.95 mean; 0 tuplets |
| `_arfull_best.json` | First full-PDMX AR build (buggy) | `ar_full-epoch=10-val/total=1.2629.ckpt` (archive) | 53.80; 388 tuplets |
| `_ar2_eval.json` | Second build | `ar_full2-epoch=02-val/total=1.4901.ckpt` (archive) | 45.67; 0 tuplets |
| `_ar3_eval.json` | Corrected 85K build, the best from-scratch AR checkpoint | `ar_full3-epoch=09-val/total=0.9696.ckpt` (archive) | **19.97**; 53 tuplets (Mozart 12.45 versus released 5.86) |
| `_ar4_eval.json` | 171K PDMX: more data regressed | `ar_full4-epoch=02-val/total=1.1188.ckpt` (archive) | 23.43; 675 tuplets |

### Hard-tail diagnostics (released model unless noted)

| File | Experiment | Checkpoint | Headline | Report |
|---|---|---|---|---|
| `topk_offset_diag.json` | Teacher-forced rank of the correct off-dyadic offset bucket on Liszt, Ravel, Scriabin, Mozart (222 + 872 + 66 + 24 = 1,184 targets), plus keep-masked free-running rollouts. Produced by `scripts/diag_topk_offset.py`. | `MIDI2ScoreTF.ckpt` (quickstart; sha256 recorded) | hit@1 **36.66 percent**, hit@5 88.94, hit@15 **98.48**; mean probability on the correct bucket 0.32; free-running greedy emits 0 off-dyadic argmax on Scriabin and Mozart, 494 on Liszt, 266 on Ravel | `README.md`, `research/supplement/README.md`, `AGENTS.md` |
| `topk_offset_diag_unmasked_positions.json` | Same teacher-forced numbers; the rollout prevalence denominators include decoder slots rejected by the keep stream (the superseded form, kept for audit) | same | same hit rates; rollout counts over all decoder positions (Liszt 2,752, Ravel 4,544, Scriabin 1,408, Mozart 2,752) instead of kept notes, which puts Ravel's free-running off-dyadic argmax count at 1,135 instead of 266 | `research/supplement/PROVENANCE.md` (chronology) |
| `topk_offset_diag_legacy_wording.json` | The June version of the same diagnostic with the older "triplet" wording and a Mac checkpoint path | same weights (Mac path) | same aggregate numbers; wording superseded ("99 percent vs 0 percent" is not the right reading, see `AGENTS.md`) | none |
| `muster_tuplet_decomposed.json` | MUSTER onset and offset error rates split by tuplet versus non-tuplet ground-truth notes, 14 pieces, two models, exact MusicXML tuplet labels (recomposed from `decomp_work/` by `scripts/recompose_muster_tuplet_decomposed.py`) | released (`checkpoints/MIDI2ScoreTF.ckpt`) and `ssl_classical_clean` epoch 13 (`../checkpoints_backup/ssl_classical_clean-ep13-val0.5125.ckpt`; both quickstart) | released: tuplet onset error 35.7 percent versus 9.2 non-tuplet (ratio 3.9, micro pooled over 5,786 tuplet notes), macro MeanER 10.73; ours: 34.9 versus 12.8 (ratio 2.7), 12.72 | `README.md`, `research/supplement/README.md`, `research/protocols/V10_1_CONFIRMATORY_AUDIT.md` |
| `muster_tuplet_decomposed_raw_exact.json` | The direct output of the exact-label run that the canonical file was built from | same | same aggregates (5,786 / 5,661 tuplet notes) | none (listed as `source_json` in the canonical file's provenance) |
| `muster_tuplet_decomposed_invalid_tpqn24.json` | The discarded version with the TPQN=24 grid labeler; kept because the audit cites its false tuplet counts | same | 17,103 "tuplet" notes for released (versus 5,786 exact); onset ratio 1.66 | `research/protocols/V10_1_CONFIRMATORY_AUDIT.md` |
| `corpus_tuplet_stats.json` | Tuplet rate of the classical SSL training corpus (`scripts/corpus_tuplet_stats.py` over `data/pairs_classical_clean_tuplrate.csv`) | no model | 23,783 scores; mean per-score tuplet rate **1.71 percent**; 86.49 percent of scores tuplet-free; note-level 3.95 percent | `LAB_REPORT.md` section 11 ("1.7 percent"), `LAB_REPORT_COMPLETE.md` |
| `reranker_doseresponse.json` | **June version** of the uniform top-k offset boost sweep on Scriabin (`scripts/rerank_offset.py`, lambda 0 to 8); its decoded-triplet rate was computed over all 1,408 decoder positions | `MIDI2ScoreTF.ckpt` (quickstart) | MeanER 10.37 at lambda 0 rising to 14.06 at 3 and 19.99 at 8 while decoded triplet rate goes 0 to 0.46: the boost moves production, not placement. **Superseded by `reranker_doseresponse_exact.json`.** | `research/supplement/README.md` (as the superseded record) |
| `reranker_doseresponse_exact.json` | The corrected successor: exact MusicXML tuplet labels and the keep-masked population (1,307 generated notes with pad probability above 0.5), recomposed from `rerank_work/` by `scripts/recompose_reranker_doseresponse.py` | same (sha256 recorded) | same MeanER curve; decoded triplet rate 0 to 0.49 over kept notes; lambda 0 byte-identical to baseline | `research/supplement/README.md`, `research/protocols/bestofn_prereg.md` |
| `reranker_doseresponse_unmasked_positions.json` | Exact labels but the unmasked-position denominators; retained for audit | same | same MeanER; rates over 1,408 positions | `PROVENANCE.md` (chronology) |
| `reranker_results_smoke.json` | Smoke run of the reranker (lambda 0, 8, 16) before the sweep; lambda 0 was not yet byte-identical (35 differing notes) | same | 10.48 at lambda 0 | none |

### Best-of-N / support-selection study (all on the released checkpoint, CPU FP32, unless noted)

Development pieces are Scriabin (main), Mozart (control), Liszt and Ravel (hard tail). Baseline Scriabin is MeanER 10.37, TPF 55; Mozart 5.86 / 15; Liszt 13.14 / 231; Ravel 20.79 / 2472. Protocols: `research/protocols/bestofn_prereg.md` (V3b, V4, V10.1 addenda, decision rule, post-run corrections), `V10_1_CONFIRMATORY_AUDIT.md`, `V10_1_ORACLE_AUDIT.md`, `REVISION_LOCALITY_PROTOCOL.md`. The pinned copies of the key records are under `research/supplement/artifacts/` with hashes in `research/supplement/MANIFEST.sha256`. Candidate pools and MUSTER intermediates for all of these are in the `benchmark-work-dirs` archive asset (`benchmark/*_work*/`), without which a script can only regenerate, not re-read.

| File | Experiment | Headline | Report |
|---|---|---|---|
| `verifier_bestofn_smoke.json` | V1+V2 verifier, n=2, temperature 1.0, Scriabin | pick 11.25 versus baseline 10.37 | none |
| `verifier_bestofn_t07.json` | V1+V2 verifier, n=8, temperature 0.7 | pick 11.89; oracle among candidates 10.53; mean candidate 11.67 | `bestofn_prereg.md` (V3b outcome) |
| `verifier_bestofn_v3smoke.json` | V3 timing-evidence term, n=2 | pick 12.23 | none |
| `verifier_bestofn_t07_v3.json` | V3, n=8 | pick 11.89; oracle 10.53 | `bestofn_prereg.md` |
| `verifier_bestofn_dev_v3b.json` | V3b (the pre-registered evidence form), n=8; the frozen development pool | pick 11.84 versus baseline 10.37 (V3b declared dead) | `bestofn_prereg.md`, `research/supplement/README.md` (pinned copy) |
| `local_repair_v4_dev.json` | V4 local repair on Scriabin, n=8 | repaired 16.63, TPF 65 versus 10.37 / 55; drift gate failed (window saturated to the whole piece) | `bestofn_prereg.md` addendum, `research/supplement/README.md` (pinned) |
| `local_repair_v5_dev.json` | Exploratory V5 repair | 10.97, TPF 58 versus 55 | none |
| `likelihood_timing_selector_v6_dev.json` | V6 donor-only selector, Scriabin, n=8 | pick candidate 2: 10.70, TPF 47 (gate ok) | none (`bestofn_prereg.md` mentions V6 in passing) |
| `likelihood_timing_selector_v6_dev_chunked.json` | V6 with chunked teacher forcing (needed for long pieces) | pick candidate 5: 11.89, TPF 60 (gate failed; chunking compresses the loss signal) | none |
| `evidence_screen_selector_v7_dev.json` | V7 evidence screen, Scriabin, n=8 | 10.70, TPF 47 | none |
| `evidence_screen_selector_v7_mozart_n8.json` | V7 on the Mozart control | 6.12, TPF 10 versus 5.86 / 15 | none |
| `evidence_screen_selector_v7_hardtail_n8.json` | V7 on Liszt and Ravel | Liszt 26.48 (TPF 373), Ravel 22.38 (2627): timing evidence alone picks damaged candidates | none (motivates V8) |
| `guarded_selector_v8_dev_mozart_n8.json` | V8 guards, Scriabin and Mozart, n=8 | Scriabin 10.70 / 47, Mozart 6.12 / 10 | `bestofn_prereg.md` (V8 freeze) |
| `guarded_selector_v8_hardtail_n8.json` | V8 on Liszt and Ravel, n=8 | both fall back to baseline (no donor passes the guards) | same |
| `guarded_selector_v8_heldout_n32.json` | V8 at n=32 on Mozart, Liszt, Ravel | Mozart 6.46 / 16, Liszt 13.03 / 207, Ravel 22.23 / 2547; gate not met | same |
| `guarded_selector_v8_scriabin_n32_evalall.json` | V8 n=32 Scriabin with every candidate scored | pick 10.69 / 54 | none |
| `guarded_selector_v8_liszt_n32_evalall.json` | V8 n=32 Liszt, every candidate scored | pick 13.03 / 207 | none |
| `guarded_selector_v8_ravel_offset_only_n16.json` | V8 Ravel sampling the offset stream only, n=16, top-k 15 | pick 30.10 / 3162 (worse) | none |
| `guarded_selector_v8_ravel_offset_only_k10_n16.json` | same with top-k 10 | identical pick | none |
| `v8_eligible_metrics.json` | MUSTER metrics of the guard-eligible candidates of `guarded_selector_v8_heldout_n32.json` (`scripts/eval_v8_eligible.py`; its `source` field is an absolute path on the original machine) | per-candidate TPF and MeanER for Mozart, Liszt, Ravel | none |
| `v8_ravel_all_candidate_metrics.json` | All 32 Ravel candidates of the same pool scored | shows the Ravel pool had no non-worse candidate at that time | `bestofn_prereg.md` (V9 motivation) |
| `decode_lever_sweep_v9_ravel_offset.json` | V9: offset phase boost on Ravel, 19 configurations | best TPF 2454 versus 2472 (eighth mode, lambda 1.0), MeanER 20.58 | `bestofn_prereg.md` |
| `decode_lever_sweep_v9_ravel_duration_cross.json` | V9: offset boost crossed with the A2 metrical prior on Ravel, 12 rows | best TPF **2414**, MeanER 20.29 (eighth 1.0 plus metrical 0.25): the deterministic phase-duration candidate V10 uses | same |
| `decode_lever_sweep_v9_all_eighth.json` | Offset boost only, all four dev pieces, 3 rows each | helps Ravel only | same |
| `decode_lever_sweep_v9_all_eighth_metrical.json` | Offset boost plus metrical prior, four pieces, 6 rows each | Liszt TPF 182 at metrical 0.25 but MeanER 14.87 (too much MeanER); Mozart unchanged | same |
| `routed_selector_v10_n8_smoke.json` | V10 routing smoke, n=8, four pieces | routes: Scriabin best-of-N, Mozart baseline, Liszt baseline, Ravel deterministic | none |
| `routed_selector_v10_n32.json` | V10 first n=32 run | Scriabin 10.69 / 54, Liszt 13.03 / 207, Ravel 20.29 / 2414 | none |
| `routed_selector_v10_scriabin_n32_counts.json` | V10 Scriabin with the triplet-count floor variant | 10.69 / 54 | none |
| `routed_selector_v10_scriabin_n32_margin.json` | V10 Scriabin with the model-loss margin variant (`min_model_delta` 0.02, `target_model_delta` 0.025) | 10.70 / 47 | none |
| `routed_selector_v10_n32_margin.json` | V10 margin variant, four pieces | Scriabin 10.70 / 47, Liszt 13.03 / 207, Ravel 20.29 / 2414 | none |
| `routed_selector_v10_liszt_n32_dense_timing.json` | V10 Liszt with the dense-timing rule and keep delta 0.05 | 13.30 / 191 | none |
| `routed_selector_v10_n32_final.json` | **The frozen V10.1 development result** (guards: keep delta 0.05, model delta 0.03, margin 0.02 to 0.025, triplet floor 50) | Scriabin 10.70 / 47, Mozart baseline 5.86 / 15, Liszt 13.30 / 191, Ravel 20.29 / 2414 (TPQN=24 labels) | `bestofn_prereg.md` Addendum 2 |
| `routed_selector_v10_n32_final_metric_v2.json` | The same four picks re-scored with exact tuplet labels (`scripts/reevaluate_v10_dev_metric_v2.py`); choices unchanged | pooled TPF 1380 to 1321, macro MeanER delta -0.004 | `V10_1_CONFIRMATORY_AUDIT.md`, `research/supplement/README.md` (pinned) |
| `routed_selector_v10_ep13_n32.json` | V10 applied to the own model (`ssl_classical_clean` epoch 13, quickstart) on the four dev pieces | Liszt takes the deterministic route: MeanER 15.78 to 13.04 but TPF 226 to 243; the other three return baseline | none |
| `routed_selector_v10_ep13_liszt_guarded.json` | Same checkpoint, Liszt, with the deterministic route additionally guarded by a 50-triplet floor | falls back to baseline 15.78 / 226 | none |
| `confirmatory_v10_1_selections.json` | Phase one of the confirmation: ground-truth-blind V10.1 routing on the ten untouched pieces with stream hashes, 2026-08-18 | per piece: route, selected candidate, guards, baseline and selected stream hashes; no evaluation fields | `bestofn_prereg.md` Addendum 2, `V10_1_CONFIRMATORY_AUDIT.md`, supplement (pinned, sha256 f1c4a6f1...) |
| `confirmatory_v10_1_results.json` | Phase two with the exact metric, replayed against the unchanged locked selections | pooled TPF **1457 to 1553**, macro MeanER **+0.416**, 7 tuplet-bearing and 3 control pieces, **CONFIRMED=false** | same, plus `README.md` |
| `confirmatory_v10_1_results_invalid_tpqn24.json` | The original phase two with the invalid TPQN=24 labeler (Bach 728 and Prokofiev 4649 false tuplets) | also CONFIRMED=false; kept as evidence for the audit | `V10_1_CONFIRMATORY_AUDIT.md` |
| `oracle_prefix_v10_1.json` | Secondary analysis: all 256 locked candidates (8 sampled pieces times 32) scored, prefix oracles at N = 1, 2, 4, 8, 16, 32, with every input hash | guarded baseline-inclusive oracle: pooled TPF **1248 to 1129** (-9.54 percent), macro MeanER -0.558; all 6 sampled tuplet-bearing pools contain a guarded improving candidate; V10.1 did not pick them | `V10_1_ORACLE_AUDIT.md`, `REVISION_LOCALITY_PROTOCOL.md`, `README.md`, supplement (pinned) |
| `revision_locality_v10_1.json` | How far the six guarded-oracle candidates are from the greedy decode in aligned streams (`scripts/revision_locality_v10_1.py`) | 17,921 positions, 72.6 percent changed (micro; per-piece 23.7 to 94.5 percent), change span at least 98 percent of each piece, 22 to 91 change runs: the better candidate is a global revision, not a local patch | `REVISION_LOCALITY_PROTOCOL.md` (no result document; the numbers live here) |
| `bestofn_secondary_audit.json` | Post-selection secondary audit written by `scripts/audit_bestofn_secondary.py` from the three records above (their hashes recorded; the file itself carries no date, late August 2026 by file date) | sealed V10.1 TPF 1372 versus greedy 1248; oracle at guard 0.0 / 0.25 / 0.5: 1145 / 1129 / 1129; minimum-model-loss selector 1239; of 192 saved samples 35 are guarded improvements and none survive the model-loss band | none yet |

### Audio to score (tier 2)

| File | Experiment | Model | Headline | Report |
|---|---|---|---|---|
| `tier2_baseline.json` | End to end through `transcribe.py -t hft -b transformer` on the three audio pieces (`tier2_baseline.log` beside it) | hFT plus released | Chopin Op. 10 No. 4 **2.75**, Op. 25 No. 11 2.76, Mazeppa 34.23 | `BASELINE_RESULTS.md`, `docs/reports/SONGSCRIPTION_PARITY.md` |
| `decomposed.json` | Stage A (hFT, mir_eval onset F1) versus Stage B (released model on the ground-truth MIDI, MUSTER) | hFT; released | onset F1 0.97 / 0.98 / 0.95; Stage B MeanER 1.64 / 1.40 / 32.97: Mazeppa fails in the score model, not the transcriber | `DECOMPOSED_FINDINGS.md`, `docs/reports/SONGSCRIPTION_PARITY.md` (`legacy/docs/reports/AUDIO_SCORE_ROADMAP.md` only proposes this evaluation; it predates the numbers) |
| `trackb_mazeppa_sweep.json` | hFT detection-threshold sweep on Mazeppa, end to end | hFT plus released | 34.89 baseline, 34.36 at onset/mpe 0.35, 34.13 at 0.25, 34.96 offset mode: no lever moves it | `TRACKB_RESULTS.md` |

## Other files here

- `*.md`: `BASELINE_RESULTS.md` (May 2026 tier-1 and tier-2 baselines), `DECOMPOSED_FINDINGS.md`, `TRACKB_RESULTS.md`, `LEAKAGE_AUDIT.md` (why `scripts/content_dedup.py` exists), `GPU_FINETUNE_RESULTS.md` (June 2026 fine-tuning diagnosis). Later topic reports are in `docs/reports/`.
- `curves/`: 24 training curves (`<run>.metrics.csv`, Lightning CSV logger output: `epoch`, `step`, `train/*`, `val/*`), including runs whose checkpoints are gone. Configs for some runs are in `checkpoints_backup/**/hparams.yaml`.
- `b2_eval/` and `seg_eval/`: `eval_padsweep.py` outputs of the beat-relative tokenization (B2) and segment experiments (`ssl_b2*`, `seg_b2*`, `seg_ctrl*`); their checkpoints are not in the asset list; discussed in `LAB_REPORT_COMPLETE.md` and the `legacy/benchmark/box_logs/` logs (`ssl_b2*.log`, `seg_*.log`, `b2_eval*.log`).
- `finetune_eval_full/`, `finetune_eval_smoke/`: the June 2026 `gpu_finetune.sh` outputs (tier-1 JSON plus `REPORT.txt`).
- `tier2_out/baseline/`: the MusicXML that `tier2_baseline.json` scored.
- `maestro-v3.0.0.csv`: the MAESTRO metadata table (the audio itself is not here beyond the three benchmark pieces).
- `*_work*/` (gitignored): candidate pools and MUSTER intermediates, from the `benchmark-work-dirs` archive asset.
