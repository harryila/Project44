#!/bin/bash
# HISTORICAL: this launcher ran on the 2026 rented GPU box (/root/Music-ML-BayenLab, fixed GPU ids,
# pkill of other runs). Kept as the record of how that run was launched. Do not run as is; see
# scripts/README.md and use scripts/_env.sh conventions (run_ssl_classical_clean.sh, run_ssl_tuplet.sh).
# Render the rubato-augmented kern corpus via N INDEPENDENT OS processes (not joblib/loky,
# which deadlocks because torch is imported in the parent before forking workers). Each shard
# is a fresh interpreter that imports torch once and renders its slice serially. Then merge.
set -uo pipefail
cd /root/Music-ML-BayenLab || exit 1
export PYTHONPATH=MIDI2ScoreTransformer/midi2scoretransformer
PY=./venv311/bin/python
N=${1:-32}
echo "=== cleaning prior partial output ==="
rm -f data/pairs_kern_rubato/*.mid data/pairs_kern_rubato/*.json \
      data/pairs_rubato_manifest.csv.shard*.csv data/pairs_rubato_manifest.csv 2>/dev/null
echo "=== launching $N detached shards ==="
for i in $(seq 0 $((N-1))); do
  setsid $PY scripts/rerender_rubato.py --shard "$i/$N" --kern-mult 30 \
      > /root/rubato_shard_$i.log 2>&1 < /dev/null &
done
sleep 3
echo "launched; shard procs now: $(pgrep -f 'rerender_rubato.py --shard' | wc -l)"
echo "(poll: find data/pairs_kern_rubato -name '*.mid' | wc -l ; target 7230)"
echo "(merge when shards gone: $PY scripts/rerender_rubato.py --merge --kern-mult 30)"
