#!/bin/bash
# HISTORICAL: this launcher ran on the 2026 rented GPU box (/root/Music-ML-BayenLab, fixed GPU ids,
# pkill of other runs). Kept as the record of how that run was launched. Do not run as is; see
# scripts/README.md and use scripts/_env.sh conventions (run_ssl_classical_clean.sh, run_ssl_tuplet.sh).
# Masked-self-supervised UNPAIRED-score training = the released model's data strategy.
# 50/50 mix of real ASAP pairs and unpaired engraved scores (surrogate score-pitch input +
# masked timing + conditioning token; decoder reconstructs the score). This is the lever the
# SOTA-gap diagnosis identified: same architecture/recipe as ours, the difference is DATA STRATEGY
# (real-unpaired masked-SSL vs our synthetic rendered pairs). From scratch, recipe matched.
# Run from REPO ROOT so manifest data/... paths + cache_pdmx lookups resolve; ASAP via --data-dir.
set -uo pipefail
cd /root/Music-ML-BayenLab || exit 1
REPO="$PWD"
MAN="$REPO/data/pairs_unpaired_ssl_manifest.csv"   # PDMX-only (real engraved scores), built in prep
INIT_NONE=""                                        # from scratch (no warm-start; SSL is from-scratch)
OUT="$REPO/MIDI2ScoreTransformer/checkpoints/ssl_unpaired"
LOG="/root/ssl_unpaired.log"
[ -f "$MAN" ] || { echo "MISSING manifest: $MAN (run the prep step first)"; exit 1; }
mkdir -p "$OUT"
pkill -9 -f '[t]rain.py fit' 2>/dev/null; sleep 2
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$REPO/MIDI2ScoreTransformer/midi2scoretransformer"
setsid nohup "$REPO/venv311/bin/python" -u \
  "$REPO/MIDI2ScoreTransformer/midi2scoretransformer/train.py" fit \
  --stage ssl_unpaired --dataset-type ssl --real-fraction 0.5 \
  --manifest "$MAN" --data-dir "$REPO/MIDI2ScoreTransformer/data/" \
  --autoregressive --lr 3e-4 --max-epochs 5 --batch-size 16 --seq-length 512 \
  --precision bf16-mixed --num-workers 8 \
  --out-dir "$OUT" > "$LOG" 2>&1 < /dev/null &
echo "ssl_unpaired launched (pid $!) -> $LOG"
