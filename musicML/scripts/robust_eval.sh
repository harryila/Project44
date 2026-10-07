#!/bin/bash
# HISTORICAL: this launcher ran on the 2026 rented GPU box (/root/Music-ML-BayenLab, fixed GPU ids,
# pkill of other runs). Kept as the record of how that run was launched. Do not run as is; see
# scripts/README.md and use scripts/_env.sh conventions (run_ssl_classical_clean.sh, run_ssl_tuplet.sh).
# Per-piece eval with a hard OS-level timeout (kills music21/MUSTER C-hangs that SIGALRM
# can't interrupt). Usage: robust_eval.sh <ckpt> <outfile> <piece1> <piece2> ...
cd /root/Music-ML-BayenLab
CK="$1"; OUT="$2"; shift 2
: > "$OUT"
for pc in "$@"; do
  res=$(timeout 220 env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=MIDI2ScoreTransformer/midi2scoretransformer \
        ./venv311/bin/python benchmark/eval_tuplet.py --ckpt "$CK" --device cuda \
        --pieces "$pc" --limit-per 1 --out "benchmark/_tmp_${pc}.json" 2>/dev/null | grep -E "^\[")
  echo "${res:-[$pc] TIMEOUT/skip}" >> "$OUT"
done
echo "__DONE__" >> "$OUT"
