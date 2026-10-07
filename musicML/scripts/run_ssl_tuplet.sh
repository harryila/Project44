#!/bin/bash
# Tuplet-aware continue-train: warm-start from ssl_classical_clean ep13 and continue the SAME masked-SSL
# recipe (50/50 real ASAP / unpaired classical scores), but upweight the rare NON-DYADIC (tuplet) buckets
# of the duration/offset/downbeat cross-entropy. Diagnosis (docs/reports/LAB_REPORT.md section 11):
# ssl_classical_clean's tuplet head COLLAPSED (emits 0 tuplets on 9/14 ASAP pieces; NoteDuration is the
# dominant error stream, 0.654 vs the released model's 0.192 on Haydn). The released model emits tuplets
# across the board, so the gap is a training-objective problem, not data-ceiling and not field-wide-hard.
# Upweighting tuplet buckets un-collapses the head while warm-start preserves the pitch/structure streams.
#
# THE BEST OWN MODEL (ssl_tuplet20, corpus MeanER 11.87) was produced with:
#     TUPLET_WEIGHT=2.0 STAGE=ssl_tuplet20 bash scripts/run_ssl_tuplet.sh
# The defaults below (weight 5.0, stage ssl_tuplet5) are the sweep's starting point, not the best run.
# Weight 5.0 gave 13.04, 2.5 gave 12.45, 2.0 gave 11.87 (U-shaped; see docs/reports/LAB_REPORT.md).
#
# Portable version: paths from scripts/_env.sh; ASAP data dir passed as the literal "./data/" after cd
# into MIDI2ScoreTransformer so the shipped tokenization cache is hit. The classical_clean inputs this
# needs are not in this repository (see DATA.md, "classical_clean").
set -uo pipefail
. "$(dirname "$0")/_env.sh"

TUPLET_WEIGHT="${TUPLET_WEIGHT:-5.0}"
TUPLET_GAMMA="${TUPLET_GAMMA:-0.0}"
LR="${LR:-2e-4}"
EPOCHS="${EPOCHS:-15}"
WARMUP="${WARMUP:-300}"
STAGE="${STAGE:-ssl_tuplet5}"

# Reshape runs need the tuplet_rate-augmented manifest; default to the plain one otherwise.
MAN="${MANIFEST:-$REPO/data/pairs_classical_clean_manifest.csv}"
INIT="${INIT_CKPT:-$REPO/MIDI2ScoreTransformer/checkpoints/ssl_classical_clean/ssl_classical_clean-epoch=13-val/total=0.5125.ckpt}"
OUT="$REPO/MIDI2ScoreTransformer/checkpoints/$STAGE"
LOG="$LOG_DIR/$STAGE.log"

[ -f "$MAN" ]  || { echo "MISSING manifest: $MAN"; exit 1; }
[ -f "$INIT" ] || { echo "MISSING init ckpt: $INIT (run scripts/fetch_assets.sh quickstart)"; exit 1; }
# Guard: --tuplet-gamma>0 silently NO-OPS unless the manifest carries a tuplet_rate column
# (the dataset's `'tuplet_rate' in columns` check). Fail loudly so a "reshape" run can't
# secretly degrade to a plain warm-start (this ambiguity affected ssl_reshape_g1/g2).
if [ "$TUPLET_GAMMA" != "0.0" ] && [ "$TUPLET_GAMMA" != "0" ]; then
  head -1 "$MAN" | grep -q "tuplet_rate" || {
    echo "ERROR: --tuplet-gamma=$TUPLET_GAMMA but manifest has no 'tuplet_rate' column: $MAN"
    echo "       Use MANIFEST=\$REPO/data/pairs_classical_clean_tuplrate.csv for reshape runs."; exit 1; }
fi
mkdir -p "$OUT"
kill_existing_train || exit 1
export CUDA_VISIBLE_DEVICES="$MUSICML_GPU"
cd "$REPO/MIDI2ScoreTransformer" || exit 1
setsid nohup "$PY" -u midi2scoretransformer/train.py fit \
  --stage "$STAGE" --dataset-type ssl --real-fraction 0.5 \
  --manifest "$MAN" --data-dir ./data/ \
  --init-ckpt "$INIT" --autoregressive \
  --tuplet-weight "$TUPLET_WEIGHT" --tuplet-gamma "$TUPLET_GAMMA" \
  --lr "$LR" --max-epochs "$EPOCHS" --warmup-steps "$WARMUP" \
  --batch-size 32 --seq-length 512 --precision bf16-mixed --num-workers 10 \
  --out-dir "$OUT" > "$LOG" 2>&1 < /dev/null &
echo "$STAGE launched (pid $!) -> $LOG   [tuplet_weight=$TUPLET_WEIGHT tuplet_gamma=$TUPLET_GAMMA lr=$LR epochs=$EPOCHS warmup=$WARMUP manifest=$(basename "$MAN")]"
