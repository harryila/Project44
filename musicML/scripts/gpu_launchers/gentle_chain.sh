#!/bin/bash
# HISTORICAL: this launcher ran on the 2026 rented GPU box (/root/Music-ML-BayenLab, fixed GPU ids,
# pkill of other runs). Kept as the record of how that run was launched. Do not run as is; see
# scripts/README.md and use scripts/_env.sh conventions (run_ssl_classical_clean.sh, run_ssl_tuplet.sh).
cd /root/Music-ML-BayenLab || exit 1
PY=venv311/bin/python ; EV=benchmark/eval_tier1_asap.py
# 1) wait for both gentle trainings (they use the gentle5k manifest) to finish
while pgrep -f pairs_gentle5k >/dev/null 2>&1; do sleep 30; done
echo "trainings done $(date)" >> benchmark/diag/gentle_chain.log
# 2) eval both gentle ckpts in parallel on free GPUs 5,6
G2="$PWD/MIDI2ScoreTransformer/checkpoints/ft_gentle_2e6/last.ckpt"
G1="$PWD/MIDI2ScoreTransformer/checkpoints/ft_gentle_1e6/last.ckpt"
CUDA_VISIBLE_DEVICES=5 $PY $EV --ckpt "$G2" --device cuda --out benchmark/diag/gentle_2e6.json > benchmark/diag/eval_g2.log 2>&1 &
P2=$!
CUDA_VISIBLE_DEVICES=6 $PY $EV --ckpt "$G1" --device cuda --out benchmark/diag/gentle_1e6.json > benchmark/diag/eval_g1.log 2>&1 &
P1=$!
wait $P2 $P1
# 3) report vs the cuda baseline
$PY scripts/compare_eval.py benchmark/diag/baseline_cuda.json benchmark/diag/gentle_2e6.json benchmark/diag/gentle_1e6.json > benchmark/diag/GENTLE_REPORT.txt 2>&1
echo DONE > benchmark/diag/gentle_chain.done
