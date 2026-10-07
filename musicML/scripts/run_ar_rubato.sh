#!/bin/bash
# HISTORICAL: this launcher ran on the 2026 rented GPU box (/root/Music-ML-BayenLab, fixed GPU ids,
# pkill of other runs). Kept as the record of how that run was launched. Do not run as is; see
# scripts/README.md and use scripts/_env.sh conventions (run_ssl_classical_clean.sh, run_ssl_tuplet.sh).
# Warm-start finetune of ar_full3 (best general ckpt, val 0.9696) on the RUBATO mix:
# 84K PDMX passthrough + 241 kern x30 DISTINCT rubato realizations (augmentation, not the
# exact-repeat upweighting that regressed ar_full5). Tests whether realistic structured
# rubato on the tuplet corpus improves the hard-virtuoso tail.
# Run from the REPO ROOT so manifest 'data/...' paths + the cache_pdmx lookup resolve;
# ASAP is found via --data-dir MIDI2ScoreTransformer/data/ (contains asap-dataset).
set -uo pipefail
cd /root/Music-ML-BayenLab || exit 1
REPO="$PWD"
MAN="$REPO/data/pairs_rubato_manifest.csv"
INIT="$REPO/MIDI2ScoreTransformer/checkpoints/ar_full3/last.ckpt"
OUT="$REPO/MIDI2ScoreTransformer/checkpoints/ar_rubato"
LOG="/root/ar_rubato.log"
[ -f "$MAN" ]  || { echo "MISSING manifest: $MAN"; exit 1; }
[ -f "$INIT" ] || { echo "MISSING init ckpt: $INIT"; exit 1; }
mkdir -p "$OUT"
# bracket trick so this pkill can't match the ssh shell / this script's own cmdline
pkill -9 -f '[t]rain.py fit' 2>/dev/null; sleep 2
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$REPO/MIDI2ScoreTransformer/midi2scoretransformer"
setsid nohup "$REPO/venv311/bin/python" -u "$REPO/MIDI2ScoreTransformer/midi2scoretransformer/train.py" fit \
  --stage ar_rubato --dataset-type mixed --real-fraction 0.4 \
  --manifest "$MAN" --data-dir "$REPO/MIDI2ScoreTransformer/data/" \
  --autoregressive --lr 3e-5 --max-epochs 5 --batch-size 16 --seq-length 512 \
  --init-ckpt "$INIT" --precision bf16-mixed --num-workers 8 \
  --out-dir "$OUT" > "$LOG" 2>&1 < /dev/null &
echo "ar_rubato launched (pid $!) -> $LOG"
