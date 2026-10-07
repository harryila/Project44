#!/usr/bin/env bash
# ============================================================================
# Track C/D — GPU fine-tune runbook (one command).
#
# Warm-starts from the released MIDI2ScoreTF.ckpt (validated by Calibration A to
# preserve generation), fine-tunes on DEDUPED synthetic PDMX pairs, then scores
# every saved checkpoint with the validated Tier-1 MUSTER harness and reports the
# best vs the reproduced baseline (MeanER 11.18).
#
#   MODE=smoke  bash scripts/gpu_finetune.sh    # ~20 min: 2k pairs, 1 epoch — DE-RISK FIRST
#   MODE=full   bash scripts/gpu_finetune.sh    # the real run: 50k pairs, 3 epochs
#
# Override anything via env: PAIRS_N, EPOCHS, LR, BATCH, DEVICE, PY, PDMX_ROOT.
# See GPU_RUNBOOK.md for the full walkthrough, prerequisites, and acceptance gates.
# ============================================================================
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
MODE="${MODE:-full}"
PY="${PY:-$REPO/venv311/bin/python}"
DEVICE="${DEVICE:-cuda}"
PDMX_ROOT="${PDMX_ROOT:-$HOME/datasets/pdmx}"
RELEASED="$REPO/MIDI2ScoreTransformer/checkpoints/MIDI2ScoreTF.ckpt"
DEDUPED_CSV="$REPO/data/pdmx_piano_subset.deduped.csv"
TF_SRC="$REPO/MIDI2ScoreTransformer/midi2scoretransformer"

if [ "$MODE" = "smoke" ]; then
  PAIRS_N="${PAIRS_N:-2000}"; EPOCHS="${EPOCHS:-1}"; BATCH="${BATCH:-8}"
  TAG="smoke"
else
  PAIRS_N="${PAIRS_N:-50000}"; EPOCHS="${EPOCHS:-3}"; BATCH="${BATCH:-16}"
  TAG="full"
fi
LR="${LR:-1e-5}"
PAIRS_DIR="$REPO/data/pairs_deduped_$TAG"
MANIFEST="$PAIRS_DIR/_manifest.csv"
OUT_DIR="$REPO/MIDI2ScoreTransformer/checkpoints/finetune_$TAG"
EVAL_DIR="$REPO/benchmark/finetune_eval_$TAG"
mkdir -p "$PAIRS_DIR" "$OUT_DIR" "$EVAL_DIR"

say() { echo -e "\n=== $* ==="; }

# ---------------------------------------------------------------- Phase 0: env
say "Phase 0: environment"
[ -f "$RELEASED" ]   || { echo "MISSING released ckpt: $RELEASED (download from the upstream GitHub Releases)"; exit 1; }
[ -f "$DEDUPED_CSV" ]|| { echo "MISSING deduped subset: $DEDUPED_CSV (run scripts/content_dedup.py)"; exit 1; }
[ -d "$PDMX_ROOT/mxl" ] || { echo "MISSING PDMX mxl/ at $PDMX_ROOT (see GPU_RUNBOOK.md Phase R1 download)"; exit 1; }
"$PY" -c "import torch; assert torch.cuda.is_available(), 'CUDA not visible'; print('CUDA ok:', torch.cuda.get_device_name(0))" \
  || { echo "CUDA not available — set DEVICE=cpu to run anyway (slow)"; [ "$DEVICE" = "cuda" ] && exit 1; }
echo "mode=$MODE pairs=$PAIRS_N epochs=$EPOCHS batch=$BATCH lr=$LR device=$DEVICE"

# ------------------------------------------------------ Phase 1: synthetic data
say "Phase 1: synthetic pairs from the DEDUPED subset"
have=0; [ -f "$MANIFEST" ] && have=$(($(wc -l < "$MANIFEST") - 1)) || true
if [ "$have" -ge "$PAIRS_N" ]; then
  echo "manifest already has $have >= $PAIRS_N pairs — skipping generation"
else
  echo "generating $PAIRS_N pairs (CPU-bound; ~minutes-hours by core count)..."
  "$PY" "$REPO/scripts/make_pairs.py" \
    --subset-csv "$DEDUPED_CSV" --mxl-root "$PDMX_ROOT" \
    --out-dir "$PAIRS_DIR" --cache-dir "$REPO/data/cache_pdmx_$TAG" \
    --manifest "$MANIFEST" --errors "$PAIRS_DIR/_errors.log" \
    --n "$PAIRS_N" --n-jobs -1 --prefer-multi-track
fi
rows=$(($(wc -l < "$MANIFEST") - 1)); echo "pairs available: $rows"
[ "$rows" -ge 100 ] || { echo "too few pairs generated — check $PAIRS_DIR/_errors.log"; exit 1; }

# ------------------------------------------------ Phase 2: warm-start fine-tune
say "Phase 2: warm-start fine-tune (init from released ckpt)"
# NOTE: stage name reuses 'pretrain_pdmx' (the only PDMX-manifest stage in train.py);
# --init-ckpt makes it a WARM-START fine-tune, not from-scratch.
PYTHONPATH="$TF_SRC" "$PY" "$TF_SRC/train.py" fit \
  --stage pretrain_pdmx --manifest "$MANIFEST" \
  --lr "$LR" --max-epochs "$EPOCHS" --batch-size "$BATCH" \
  --init-ckpt "$RELEASED" --precision bf16-mixed --num-workers 8 \
  --out-dir "$OUT_DIR"

# ------------------------------------------------------------ Phase 3: evaluate
say "Phase 3: evaluate every saved checkpoint (real MUSTER, not val/total)"
# baseline (reproduce once if not cached)
BASE_JSON="$REPO/benchmark/tier1_baseline.json"
if [ ! -f "$BASE_JSON" ]; then
  "$PY" "$REPO/benchmark/eval_tier1_asap.py" --ckpt "$RELEASED" --device "$DEVICE" --out "$BASE_JSON"
fi
for ckpt in "$OUT_DIR"/*.ckpt; do
  [ -e "$ckpt" ] || continue
  name=$(basename "$ckpt" .ckpt | tr '/=' '__')
  out="$EVAL_DIR/eval_${name}.json"
  [ -f "$out" ] && { echo "cached: $out"; continue; }
  echo "evaluating $ckpt ..."
  "$PY" "$REPO/benchmark/eval_tier1_asap.py" --ckpt "$ckpt" --device "$DEVICE" --out "$out"
done

# -------------------------------------------------------------- Phase 4: report
say "Phase 4: comparison (best fine-tuned vs baseline, overall + hard composers)"
"$PY" "$REPO/scripts/compare_eval.py" "$BASE_JSON" "$EVAL_DIR"/eval_*.json | tee "$EVAL_DIR/REPORT.txt"
echo -e "\nDone. Full report: $EVAL_DIR/REPORT.txt"
