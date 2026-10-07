# GPU Runbook — Track C/D Fine-Tune

The turnkey, **validated** procedure to actually improve Block-2 accuracy: warm-start
from the released checkpoint, fine-tune on the deduped synthetic PDMX pairs, and
measure the result against the reproduced SOTA baseline with real MUSTER.

> This **supersedes** the from-scratch R3–R8 plan in `SYNTHETIC_PRETRAIN_STATUS.md`.
> That plan collapsed to empty scores because it trained from random init. We proved
> on the Mac (Calibration A) that **warm-starting** from the released checkpoint
> preserves generation — so this runbook warm-starts. Everything here is grounded in
> work already validated locally; see `ACCURACY_ROADMAP.md` for the full story.

## What's already proven (so you can trust this)
- **Eval is validated** — `benchmark/eval_tier1_asap.py` reproduces the paper's SOTA
  (MeanER **11.18** vs 11.30). That 11.18 is your baseline to beat. See
  `benchmark/BASELINE_RESULTS.md`.
- **Warm-start works** — `scripts/calibration_a.py` warm-started + fine-tuned 250
  steps and generation was preserved (no collapse). The cold-start bug does not apply.
- **Data is clean** — `scripts/content_dedup.py` removed the 23 eval-piece leaks from
  PDMX → `data/pdmx_piano_subset.deduped.csv`. Training on it does not contaminate
  the held-out eval. See `benchmark/LEAKAGE_AUDIT.md`.
- **Inference tuning is exhausted** — Track B measured null (`benchmark/TRACKB_RESULTS.md`).
  Retraining is the remaining lever.

## Prerequisites (one-time machine setup)
```bash
# 1. Repo + Python 3.11 env with CUDA torch
git clone <this repo> && cd musicML
python3.11 -m venv venv311 && source venv311/bin/activate
pip install -r MIDI2ScoreTransformer/requirements.txt
pip install pytorch-lightning>=2.0 tensorboard pandas joblib pretty_midi soxr soundfile
pip install torch --index-url https://download.pytorch.org/whl/cu121   # if needed
python -c "import torch; print('cuda:', torch.cuda.is_available())"      # expect True

# 2. Released checkpoint (warm-start source + baseline)  -> ~372 MB
#    From https://github.com/TimFelixBeyer/MIDI2ScoreTransformer/releases
#    Save as MIDI2ScoreTransformer/checkpoints/MIDI2ScoreTF.ckpt

# 3. PDMX corpus (score source for pair generation)  -> ~7 GB
#    See SYNTHETIC_PRETRAIN_STATUS.md "Phase R1" (Zenodo download + extract).
#    Result: ~/datasets/pdmx/mxl/ with 254,035 .mxl files.

# 4. ASAP + ACPAS (needed by the eval + the dedup needles)
#    See SYNTHETIC_PRETRAIN_STATUS.md "Phase R1": clone asap-dataset (correct commit)
#    into MIDI2ScoreTransformer/data/, add ACPAS metadata_R/S.csv.

# 5. The deduped training pool + eval GT scores. Two options:
#    (a) transfer from the Mac: data/pdmx_piano_subset.deduped.csv,
#        data/pdmx_eval_leak_blocklist.csv, benchmark/liszt_transcendental/gt_score*.musicxml
#    (b) regenerate on the GPU box once PDMX + ASAP are present:
#        python scripts/content_dedup.py    # rebuilds the deduped csv + blocklist
```

## Run it (one command)
```bash
# DE-RISK FIRST: ~20 min end-to-end on a GPU (2k pairs, 1 epoch).
MODE=smoke bash scripts/gpu_finetune.sh

# If the smoke run completes and the report prints, launch the real run:
MODE=full bash scripts/gpu_finetune.sh        # 50k pairs, 3 epochs

# Tunables (env overrides): PAIRS_N, EPOCHS, LR, BATCH, DEVICE, PDMX_ROOT, PY
LR=2e-5 EPOCHS=4 bash scripts/gpu_finetune.sh
```

## What the script does (`scripts/gpu_finetune.sh`)
| Phase | Action | Gate |
|---|---|---|
| 0 | Env check: CUDA, released ckpt, deduped csv, PDMX present | fails fast with a clear message |
| 1 | Generate `PAIRS_N` synthetic pairs from the **deduped** subset (skips if present) | needs ≥100 pairs |
| 2 | **Warm-start** fine-tune (`train.py fit --init-ckpt <released>`), bf16, saves epoch + last ckpts | — |
| 3 | Score **every** saved ckpt with Tier-1 MUSTER (on GPU), plus the baseline | — |
| 4 | `compare_eval.py`: best ckpt vs baseline, overall + hard-composer subset | prints VERDICT |

## Acceptance criteria & interpretation
- **Generation sanity:** every evaluated ckpt should produce non-empty scores (n_scored ≈ 59).
  If a ckpt scores 0/59, it collapsed — discard it (and prefer ckpts selected by MUSTER, not `val/total`).
- **Did it help?** The report flags a ckpt **PASS** only if overall MeanER beats the
  baseline by more than the eval noise floor (~0.28). Watch the **hard-composer**
  MeanER (Liszt/Schubert/Ravel/Rachmaninoff/Debussy/Schumann) — that's the OOD tail
  the synthetic data is meant to fix.
- **CRITICAL caveat:** `train.py`'s `ModelCheckpoint` selects on `val/total`
  (teacher-forced CE), which the analysis showed is **misleading** — it does not
  measure generation. **Always pick the final ckpt by the Phase-4 MUSTER report, not
  by `val/total`.**

## If it does NOT beat baseline (likely on the first try)
This is a real possibility — synthetic hand-coded-jitter pairs may not transfer to
real performances (domain gap), and the released model already maxes ASAP. In order:
1. **Sweep LR / steps** — try `LR=2e-5 EPOCHS=4`, then `LR=5e-5`. Too low = no
   learning; too high = drifts off the ASAP-real distribution.
2. **Upgrade the renderer to learned rendering** (the biggest lever — `ACCURACY_ROADMAP.md`
   Track D): swap VirtuosoNet (`dasaem/virtuosonet`) into `scripts/expressive_render.py`
   so synthetic MIDI is human-realistic, and break the 1:1 alignment. A 2025 paper hit
   MUSTER 9.48 vs 11.30 doing exactly this.
3. **Add real aligned pairs** — ingest ACPAS / PianoCoRe (`ACCURACY_ROADMAP.md` Track D)
   and mix them in, jointly with the unpaired PDMX score-prior (the paper's actual recipe).
4. **Curate the corpus** — oversample canonical classical (the "24K classical" is ~93%
   folk/hymn); use `--classical-only`-style filtering on the subset before pair-gen.

## Resume
`train.py fit` does not auto-resume optimizer state. To continue from a saved ckpt,
warm-start from it instead of the released one:
```bash
RELEASED=MIDI2ScoreTransformer/checkpoints/finetune_full/last.ckpt \
  EPOCHS=2 bash scripts/gpu_finetune.sh    # (override RELEASED via the script's var)
```
Phases 1/3 are idempotent (skip existing pairs / cached evals).

## Cost
A single RTX 4090 / A100 handles this in ~1–4 h for the full run (pair-gen is the
CPU-bound part; training 50k pairs × 3 epochs is fast). The smoke run is ~20 min.
