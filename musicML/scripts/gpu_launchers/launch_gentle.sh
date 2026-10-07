#!/bin/bash
# HISTORICAL: this launcher ran on the 2026 rented GPU box (/root/Music-ML-BayenLab, fixed GPU ids,
# pkill of other runs). Kept as the record of how that run was launched. Do not run as is; see
# scripts/README.md and use scripts/_env.sh conventions (run_ssl_classical_clean.sh, run_ssl_tuplet.sh).
cd /root/Music-ML-BayenLab || exit 1
mkdir -p benchmark/diag
head -5001 data/pairs_deduped_full/_manifest.csv > data/pairs_gentle5k_manifest.csv
PY=venv311/bin/python ; TR=MIDI2ScoreTransformer/midi2scoretransformer/train.py
REL="$PWD/MIDI2ScoreTransformer/checkpoints/MIDI2ScoreTF.ckpt"
export PYTHONPATH=MIDI2ScoreTransformer/midi2scoretransformer
CUDA_VISIBLE_DEVICES=3 nohup $PY $TR fit --stage pretrain_pdmx --manifest data/pairs_gentle5k_manifest.csv \
  --lr 2e-6 --max-epochs 2 --batch-size 16 --init-ckpt "$REL" --precision bf16-mixed --num-workers 8 \
  --out-dir MIDI2ScoreTransformer/checkpoints/ft_gentle_2e6 > benchmark/diag/train_gentle_2e6.log 2>&1 &
CUDA_VISIBLE_DEVICES=4 nohup $PY $TR fit --stage pretrain_pdmx --manifest data/pairs_gentle5k_manifest.csv \
  --lr 1e-6 --max-epochs 2 --batch-size 16 --init-ckpt "$REL" --precision bf16-mixed --num-workers 8 \
  --out-dir MIDI2ScoreTransformer/checkpoints/ft_gentle_1e6 > benchmark/diag/train_gentle_1e6.log 2>&1 &
sleep 6
echo "launched gentle 2e-6 (GPU3) + 1e-6 (GPU4) on 5k pairs"
echo "manifest rows: $(($(wc -l < data/pairs_gentle5k_manifest.csv)-1))"
ls -la benchmark/diag/train_gentle_*.log
