#!/bin/bash
# HISTORICAL: this launcher ran on the 2026 rented GPU box (/root/Music-ML-BayenLab, fixed GPU ids,
# pkill of other runs). Kept as the record of how that run was launched. Do not run as is; see
# scripts/README.md and use scripts/_env.sh conventions (run_ssl_classical_clean.sh, run_ssl_tuplet.sh).
# Masked-SSL with a CLASSICAL unpaired corpus (the lever after ssl_v2, where pop-heavy PDMX sharpened
# the common case but REGRESSED the hard classical tail / Scriabin). Unpaired = ~24k genre=classical
# PDMX scores. Tests whether a genre-matched unpaired corpus recovers the tail while keeping the broad
# gains. Same exposure-matched recipe as ssl_v2 (real_fraction 0.5, batch 32, lr 3e-4), from scratch.
set -uo pipefail
cd /root/Music-ML-BayenLab || exit 1
REPO="$PWD"
MAN="$REPO/data/pairs_classical_clean_manifest.csv"
OUT="$REPO/MIDI2ScoreTransformer/checkpoints/ssl_recipe"
LOG="/root/ssl_recipe.log"
[ -f "$MAN" ] || { echo "MISSING manifest: $MAN"; exit 1; }
mkdir -p "$OUT"
pkill -9 -f '[t]rain.py fit' 2>/dev/null; sleep 2
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$REPO/MIDI2ScoreTransformer/midi2scoretransformer"
setsid nohup "$REPO/venv311/bin/python" -u \
  "$REPO/MIDI2ScoreTransformer/midi2scoretransformer/train.py" fit \
  --stage ssl_recipe --dataset-type ssl --real-fraction 0.5 \
  --manifest "$MAN" --data-dir "$REPO/MIDI2ScoreTransformer/data/" \
  --autoregressive --lr 3e-4 --max-epochs 58 --warmup-steps 4000 --batch-size 32 --seq-length 512 \
  --precision bf16-mixed --num-workers 10 \
  --out-dir "$OUT" > "$LOG" 2>&1 < /dev/null &
echo "ssl_recipe launched (pid $!) -> $LOG"
