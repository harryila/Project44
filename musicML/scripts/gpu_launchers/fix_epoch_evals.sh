#!/bin/bash
# HISTORICAL: this launcher ran on the 2026 rented GPU box (/root/Music-ML-BayenLab, fixed GPU ids,
# pkill of other runs). Kept as the record of how that run was launched. Do not run as is; see
# scripts/README.md and use scripts/_env.sh conventions (run_ssl_classical_clean.sh, run_ssl_tuplet.sh).
cd /root/Music-ML-BayenLab || exit 1
PY=venv311/bin/python ; EV=benchmark/eval_tier1_asap.py
R="$PWD/MIDI2ScoreTransformer/checkpoints/finetune_full"
E1="$R/pretrain_pdmx-epoch=01-val/total=0.2553.ckpt"
E2="$R/pretrain_pdmx-epoch=02-val/total=0.2527.ckpt"
CUDA_VISIBLE_DEVICES=1 nohup $PY $EV --ckpt "$E1" --device cuda --out benchmark/diag/ft_epoch01.json > benchmark/diag/ft_epoch01.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup $PY $EV --ckpt "$E2" --device cuda --out benchmark/diag/ft_epoch02.json > benchmark/diag/ft_epoch02.log 2>&1 &
sleep 4
echo "relaunched epoch01 (GPU1) + epoch02 (GPU2) with correct paths"
ps aux | grep -c "[e]val_tier1"
