#!/bin/bash
# HISTORICAL: this launcher ran on the 2026 rented GPU box (/root/Music-ML-BayenLab, fixed GPU ids,
# pkill of other runs). Kept as the record of how that run was launched. Do not run as is; see
# scripts/README.md and use scripts/_env.sh conventions (run_ssl_classical_clean.sh, run_ssl_tuplet.sh).
# Box-side orchestrator: wait for <STAGE> training to finish, then eval its best-val + last checkpoints
# on all 14 ASAP test pieces (threshold 0.50), then print the comparison vs ssl_classical_clean + released.
# Uses ABSOLUTE ckpt paths (eval_padsweep.py chdirs into MIDI2ScoreTransformer/, so relative paths double).
set -uo pipefail
cd /root/Music-ML-BayenLab || exit 1
R=/root/Music-ML-BayenLab
STAGE="${1:?usage: auto_eval_stage.sh <STAGE>}"
CKDIR="$R/MIDI2ScoreTransformer/checkpoints/$STAGE"

echo "[autoeval] waiting for $STAGE training to finish..."
while pgrep -f "[t]rain.py fit.*$STAGE" >/dev/null 2>&1; do sleep 30; done
sleep 5

# Best-val ckpt = smallest total=NN under the stage dir (filename is .../<stage>-epoch=NN-val/total=0.XXXX.ckpt)
BEST=$(ls "$CKDIR"/*/total=*.ckpt 2>/dev/null | awk -F'total=' '{print $2" "$0}' | sort -n | head -1 | cut -d' ' -f2-)
LAST="$CKDIR/last.ckpt"
echo "[autoeval] best=$BEST"
echo "[autoeval] last=$LAST"

venv311/bin/python benchmark/eval_padsweep.py --ckpt "$BEST" --device cuda --limit-per 1 \
  --thresholds 0.50 --out "benchmark/${STAGE}_best.json"
venv311/bin/python benchmark/eval_padsweep.py --ckpt "$LAST" --device cuda --limit-per 1 \
  --thresholds 0.50 --out "benchmark/${STAGE}_last.json"

echo "=== COMPARISON ($STAGE) ==="
python3 benchmark/compare_tuplet5.py "${STAGE}_best.json" "${STAGE}_last.json" best last
echo "AUTO_EVAL_DONE_${STAGE}"
