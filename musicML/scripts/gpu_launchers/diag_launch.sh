#!/bin/bash
# HISTORICAL: this launcher ran on the 2026 rented GPU box (/root/Music-ML-BayenLab, fixed GPU ids,
# pkill of other runs). Kept as the record of how that run was launched. Do not run as is; see
# scripts/README.md and use scripts/_env.sh conventions (run_ssl_classical_clean.sh, run_ssl_tuplet.sh).
cd /root/Music-ML-BayenLab || exit 1
mkdir -p benchmark/diag
PY=venv311/bin/python ; EV=benchmark/eval_tier1_asap.py ; R="$PWD/MIDI2ScoreTransformer/checkpoints"
A="$R/MIDI2ScoreTF.ckpt"
B="$R/finetune_full/pretrain_pdmx-epoch=00-val/total=0.1657.ckpt"
C="$R/finetune_full/pretrain_pdmx-epoch=01-val/total=0.1545.ckpt"
CUDA_VISIBLE_DEVICES=0 nohup $PY $EV --ckpt "$A" --device cuda --out benchmark/diag/baseline_cuda.json  > benchmark/diag/baseline_cuda.log  2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup $PY $EV --ckpt "$B" --device cuda --out benchmark/diag/ft_epoch00.json     > benchmark/diag/ft_epoch00.log     2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup $PY $EV --ckpt "$C" --device cuda --out benchmark/diag/ft_epoch01.json     > benchmark/diag/ft_epoch01.log     2>&1 &
sleep 3
echo "launched 3 parallel evals (GPUs 0,1,2)"
ls -la benchmark/diag/*.log
