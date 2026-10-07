#!/bin/bash
# HISTORICAL: this launcher ran on the 2026 rented GPU box (/root/Music-ML-BayenLab, fixed GPU ids,
# pkill of other runs). Kept as the record of how that run was launched. Do not run as is; see
# scripts/README.md and use scripts/_env.sh conventions (run_ssl_classical_clean.sh, run_ssl_tuplet.sh).
cd /root/Music-ML-BayenLab || exit 1
PY=venv311/bin/python ; EV=benchmark/eval_tier1_asap.py
R="$PWD/MIDI2ScoreTransformer/checkpoints"
REL="$R/MIDI2ScoreTF.ckpt"
E1="$R/finetune_full/pretrain_pdmx-epoch=01-val/total=0.2553.ckpt"
E2="$R/finetune_full/pretrain_pdmx-epoch=02-val/total=0.2527.ckpt"
# clean any leftovers
pkill -9 -f eval_tier1_asap.py 2>/dev/null; sleep 2
CUDA_VISIBLE_DEVICES=0 nohup $PY $EV --ckpt "$REL" --device cuda --out benchmark/diag/baseline_cuda.json > benchmark/diag/bl.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup $PY $EV --ckpt "$E1"  --device cuda --out benchmark/diag/ft_epoch01.json  > benchmark/diag/e1.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup $PY $EV --ckpt "$E2"  --device cuda --out benchmark/diag/ft_epoch02.json  > benchmark/diag/e2.log 2>&1 &
sleep 5
echo "launched 3 evals: baseline(GPU0) epoch01(GPU1) epoch02(GPU2)"
echo "eval procs: $(pgrep -fc eval_tier1_asap.py)"
