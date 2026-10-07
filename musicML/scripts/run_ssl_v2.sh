#!/bin/bash
# HISTORICAL: this launcher ran on the 2026 rented GPU box (/root/Music-ML-BayenLab, fixed GPU ids,
# pkill of other runs). Kept as the record of how that run was launched. Do not run as is; see
# scripts/README.md and use scripts/_env.sh conventions (run_ssl_classical_clean.sh, run_ssl_tuplet.sh).
# Follow-up to ssl_unpaired (which closed 55-61% of the SOTA gap but overfit after ep0 because each
# unpaired score was seen only ~0.5x while 822 real pieces were hammered ~94x/epoch). This run MATCHES
# the released model's per-score exposure: ~58k unpaired scores (like their 58,646) + ~40k steps at
# batch 32, so each unpaired score is seen ~12x (their ~11x) and the SSL signal accumulates to counter
# the real-set overfit. Everything else identical (real_fraction 0.5, lr 3e-4, masked-SSL surrogate).
set -uo pipefail
cd /root/Music-ML-BayenLab || exit 1
REPO="$PWD"
MAN="$REPO/data/pairs_unpaired_ssl_58k.csv"
OUT="$REPO/MIDI2ScoreTransformer/checkpoints/ssl_v2"
LOG="/root/ssl_v2.log"
[ -f "$MAN" ] || { echo "MISSING manifest: $MAN"; exit 1; }
mkdir -p "$OUT"
pkill -9 -f '[t]rain.py fit' 2>/dev/null; sleep 2
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$REPO/MIDI2ScoreTransformer/midi2scoretransformer"
setsid nohup "$REPO/venv311/bin/python" -u \
  "$REPO/MIDI2ScoreTransformer/midi2scoretransformer/train.py" fit \
  --stage ssl_v2 --dataset-type ssl --real-fraction 0.5 \
  --manifest "$MAN" --data-dir "$REPO/MIDI2ScoreTransformer/data/" \
  --autoregressive --lr 3e-4 --max-epochs 24 --batch-size 32 --seq-length 512 \
  --precision bf16-mixed --num-workers 10 \
  --out-dir "$OUT" > "$LOG" 2>&1 < /dev/null &
echo "ssl_v2 launched (pid $!) -> $LOG"
