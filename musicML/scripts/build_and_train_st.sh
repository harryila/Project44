#!/usr/bin/env bash
# HISTORICAL: this launcher ran on the 2026 rented GPU box (/root/Music-ML-BayenLab, fixed GPU ids,
# pkill of other runs). Kept as the record of how that run was launched. Do not run as is; see
# scripts/README.md and use scripts/_env.sh conventions (run_ssl_classical_clean.sh, run_ssl_tuplet.sh).
# Track ST chain: wait for MAESTRO pseudo-labeling -> tokenize pseudo-scores as unpaired ->
# merge (upweighted) into the classical unpaired corpus -> warm-start SSL retrain.
# Injects tuplet-CORRECT (released-teacher) scores at natural distribution to fix the tuplet-poor corpus.
set -uo pipefail
cd /root/Music-ML-BayenLab; REPO=$PWD
export PYTHONPATH=$REPO/MIDI2ScoreTransformer/midi2scoretransformer
REP="${REP:-12}"; TW="${TW:-5}"
INIT=$REPO/MIDI2ScoreTransformer/checkpoints/ssl_classical_clean/ssl_classical_clean-epoch=13-val/total=0.5125.ckpt

echo "[ST] waiting for pseudo-labeling shards..."
while pgrep -f "[p]seudo_label_maestro" >/dev/null; do sleep 30; done
find /root/datasets/maestro_pseudo -name "*.musicxml" > /root/pseudo_list.txt
echo "[ST] pseudo scores: $(wc -l < /root/pseudo_list.txt)"
[ "$(wc -l < /root/pseudo_list.txt)" -lt 20 ] && { echo "[ST] too few pseudo scores, abort"; exit 1; }

echo "[ST] tokenizing pseudo-scores (unpaired cache)..."
venv311/bin/python scripts/build_classical_unpaired.py /root/pseudo_list.txt pairs_maestro_pseudo data/pairs_maestro_pseudo_manifest.csv 2>&1 | tail -3

echo "[ST] merging classical + pseudo x$REP ..."
venv311/bin/python - "$REP" <<'PY'
import csv, sys, random
REP=int(sys.argv[1])
base=list(csv.reader(open('data/pairs_classical_rebuilt_manifest.csv')))
ps=list(csv.reader(open('data/pairs_maestro_pseudo_manifest.csv')))
hdr=base[0]; rows=base[1:]+ps[1:]*REP
random.seed(0); random.shuffle(rows)
w=csv.writer(open('data/pairs_st_merged_manifest.csv','w')); w.writerow(hdr); w.writerows(rows)
print(f"  classical={len(base)-1} pseudo={len(ps)-1} x{REP} -> merged={len(rows)}")
PY

echo "[ST] warm-start SSL retrain (ssl_pseudo, tuplet_weight=$TW)..."
CUDA_VISIBLE_DEVICES=0 venv311/bin/python -u $REPO/MIDI2ScoreTransformer/midi2scoretransformer/train.py fit \
  --stage ssl_pseudo --dataset-type ssl --real-fraction 0.5 \
  --manifest data/pairs_st_merged_manifest.csv --data-dir $REPO/MIDI2ScoreTransformer/data/ \
  --init-ckpt "$INIT" --autoregressive --tuplet-weight "$TW" --tuplet-gamma 0 \
  --lr 2e-4 --max-epochs 15 --warmup-steps 300 --batch-size 32 --seq-length 512 \
  --precision bf16-mixed --num-workers 16 \
  --out-dir $REPO/MIDI2ScoreTransformer/checkpoints/ssl_pseudo > /root/ssl_pseudo.log 2>&1
echo "ST_DONE best=$(grep -oE 'Best ckpt.*' /root/ssl_pseudo.log | tail -1)"
