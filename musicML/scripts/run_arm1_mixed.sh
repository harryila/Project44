#!/usr/bin/env bash
# ARM-1: the corrected fine-tune. Warm-start from the released checkpoint and
# continue-train on a REAL-MAJORITY mix (80% real ASAP / 20% synthetic) so the
# real-performance distribution is never abandoned — the direct fix for the
# 65:1 synthetic domination that degraded MUSTER 11.18 -> 18.10.
#
# Run ON THE GPU BOX from the repo root. Single-GPU (eval is CPU; training is the
# only GPU job — the synthetic dataset is CPU-IO-bound, so ONE training at a time).
set -euo pipefail
cd "$(dirname "$0")/.." || exit 1
REPO="$PWD"
PY="$REPO/venv311/bin/python"
TFDIR="$REPO/MIDI2ScoreTransformer"
REL="$TFDIR/checkpoints/MIDI2ScoreTF.ckpt"
# synthetic manifest (deduped); adjust if the box stores it elsewhere
MAN="$REPO/data/pairs_deduped_full/_manifest.csv"
OUT="$TFDIR/checkpoints/arm1_mixed8020"
mkdir -p "$OUT"

# guardrails
[ -f "$REL" ] || { echo "MISSING released ckpt: $REL"; exit 1; }
[ -f "$MAN" ] || { echo "MISSING synthetic manifest: $MAN"; exit 1; }
chunks=$(find "$TFDIR/data/asap-dataset" -name '*_chunks.json' | wc -l)
echo "ASAP _chunks.json present: $chunks (need most of ~1300 for the full train split)"
pkill -9 -f 'train.py fit' 2>/dev/null || true; sleep 2

# Pin to ONE GPU *before* python starts so Lightning doesn't auto-DDP across all 8.
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$TFDIR/midi2scoretransformer"
cd "$TFDIR"   # so ./data/ resolves for ASAPDataset

nohup "$PY" midi2scoretransformer/train.py fit \
  --stage arm1_mixed --dataset-type mixed --real-fraction 0.8 \
  --manifest "$MAN" --data-dir ./data/ \
  --lr 3e-5 --max-epochs 8 --batch-size 16 --seq-length 512 \
  --init-ckpt "$REL" --precision bf16-mixed --num-workers 8 \
  --out-dir "$OUT" > "$OUT/train.log" 2>&1 &
echo "ARM-1 launched (pid $!) -> $OUT/train.log"
sleep 12
echo "train procs: $(pgrep -fc 'train.py fit')"
grep -aE 'used:|\[mixed\]|GPU available|Error|Traceback' "$OUT/train.log" | tail -5 || true
