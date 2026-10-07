#!/bin/bash
# HISTORICAL: this launcher ran on the 2026 rented GPU box (/root/Music-ML-BayenLab, fixed GPU ids,
# pkill of other runs). Kept as the record of how that run was launched. Do not run as is; see
# scripts/README.md and use scripts/_env.sh conventions (run_ssl_classical_clean.sh, run_ssl_tuplet.sh).
cd /root/Music-ML-BayenLab || exit 1
PY=venv311/bin/python ; EV=benchmark/eval_tier1_asap.py
# paths RELATIVE TO MIDI2ScoreTransformer/ (eval chdir's there). The '=' and '/' in
# the ckpt names are fine as long as the whole arg is one quoted string.
declare -A CK
CK[ep03]='checkpoints/arm1_mixed8020/arm1_mixed-epoch=03-val/total=0.6683.ckpt'
CK[ep04]='checkpoints/arm1_mixed8020/arm1_mixed-epoch=04-val/total=0.6682.ckpt'
CK[last]='checkpoints/arm1_mixed8020/last.ckpt'
pkill -9 -f eval_tier1_asap.py 2>/dev/null; sleep 2
for tag in ep03 ep04 last; do
  nohup $PY $EV --ckpt "${CK[$tag]}" --device cpu --out "benchmark/diag/arm1_${tag}.json" \
    > "benchmark/diag/arm1_${tag}.log" 2>&1 < /dev/null &
  echo "launched $tag (pid $!): ${CK[$tag]}"
done
sleep 10
echo "running evals: $(pgrep -fc eval_tier1_asap.py)"
for tag in ep03 ep04 last; do echo "$tag log: $(tail -1 benchmark/diag/arm1_${tag}.log 2>/dev/null | grep -aviE 'warn|deprecat' | cut -c1-60)"; done
