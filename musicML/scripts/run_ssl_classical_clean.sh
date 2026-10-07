#!/bin/bash
# Masked-SSL with a CLASSICAL unpaired corpus (the lever after ssl_v2, where pop-heavy PDMX sharpened
# the common case but REGRESSED the hard classical tail / Scriabin). Unpaired = ~24k genre=classical
# PDMX scores. Tests whether a genre-matched unpaired corpus recovers the tail while keeping the broad
# gains. Same exposure-matched recipe as ssl_v2 (real_fraction 0.5, batch 32, lr 3e-4), from scratch.
#
# This is the recipe that produced ssl_classical_clean epoch 13 (corpus MeanER 12.69). Portable version:
# paths come from scripts/_env.sh; the ASAP data dir is passed as the literal "./data/" after cd into
# MIDI2ScoreTransformer so the shipped tokenization cache (keyed on that string) is hit.
#
# INPUTS NOT IN THIS REPOSITORY: data/pairs_classical_clean_manifest.csv is tracked, but the rendered
# .mid files and cache pickles it points to were never retrieved from the training machine. See DATA.md
# ("classical_clean") for how to rebuild them from PDMX with scripts/build_classical_unpaired.py.
set -uo pipefail
. "$(dirname "$0")/_env.sh"
MAN="${MANIFEST:-$REPO/data/pairs_classical_clean_manifest.csv}"
OUT="$REPO/MIDI2ScoreTransformer/checkpoints/ssl_classical_clean"
LOG="$LOG_DIR/ssl_classical_clean.log"
[ -f "$MAN" ] || { echo "MISSING manifest: $MAN"; exit 1; }
mkdir -p "$OUT"
kill_existing_train || exit 1
export CUDA_VISIBLE_DEVICES="$MUSICML_GPU"
cd "$REPO/MIDI2ScoreTransformer" || exit 1
setsid nohup "$PY" -u midi2scoretransformer/train.py fit \
  --stage ssl_classical_clean --dataset-type ssl --real-fraction 0.5 \
  --manifest "$MAN" --data-dir ./data/ \
  --autoregressive --lr 3e-4 --max-epochs 30 --batch-size 32 --seq-length 512 \
  --precision bf16-mixed --num-workers 10 \
  --out-dir "$OUT" > "$LOG" 2>&1 < /dev/null &
echo "ssl_classical_clean launched (pid $!) -> $LOG"
