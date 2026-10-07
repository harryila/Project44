#!/usr/bin/env bash
# Re-rank EVERY saved epoch of the given runs by true MUSTER (14-piece ASAP), not val/total.
# Motivation: val != MUSTER is proven repeatedly (ssl_tuplet20 'last'=11.87 beat its val-best
# 'ep01'=12.20; ssl_bigc best-val was worse on MUSTER). Best epoch on disk may beat 11.87 with
# ZERO new training. Pure eval — cheap on GPU, mostly CPU (music21/MUSTER).
#
# Usage (on the box, repo root):
#   bash scripts/rerank_by_muster.sh ssl_tuplet20 ssl_reshape_g1 ssl_reshape_g2 ssl_combo ssl_tuplet25
#   (no args -> defaults to the tuplet-lever runs below)
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
REPO="$PWD"
PY="${PY:-venv311/bin/python}"
# Absolute, because eval_padsweep.py chdir's into MIDI2ScoreTransformer (relative paths would double).
CKPT_ROOT="$REPO/MIDI2ScoreTransformer/checkpoints"
THR="${THR:-0.50}"
OUTDIR="benchmark/rerank"
mkdir -p "$OUTDIR"

RUNS=("$@")
[ ${#RUNS[@]} -eq 0 ] && RUNS=(ssl_tuplet20 ssl_tuplet25 ssl_reshape_g1 ssl_reshape_g2 ssl_combo)

echo "Re-ranking runs: ${RUNS[*]}  (threshold $THR, all 14 ASAP pieces)"
for run in "${RUNS[@]}"; do
  # every epoch ckpt for this run (the val/total= subdir holds the actual .ckpt)
  mapfile -t CKPTS < <(find "$CKPT_ROOT/$run" -name "*.ckpt" 2>/dev/null | sort)
  [ ${#CKPTS[@]} -eq 0 ] && { echo "  (no ckpts found for $run)"; continue; }
  for ck in "${CKPTS[@]}"; do
    tag="${run}__$(basename "$(dirname "$ck")")_$(basename "$ck" .ckpt)"
    tag="$(echo "$tag" | tr '/=' '__')"
    out="$OUTDIR/${tag}.json"
    [ -f "$out" ] && { echo "  skip (done): $tag"; continue; }
    echo "  eval: $run :: $ck"
    "$PY" benchmark/eval_padsweep.py --ckpt "$ck" --device cuda \
        --limit-per 1 --thresholds "$THR" --out "$out" 2>&1 | tail -2
  done
done

echo; echo "=== RE-RANK SUMMARY (corpus-mean MeanER @ thr $THR, lower=better) ==="
"$PY" - "$OUTDIR" "$THR" <<'PY'
import json, sys, glob, os
outdir, thr = sys.argv[1], sys.argv[2]
rows = []
for f in glob.glob(os.path.join(outdir, "*.json")):
    try:
        d = json.load(open(f))
    except Exception:
        continue
    vals = []
    for r in d.get("results", []):
        v = (r.get("per_threshold", {}).get(thr, {}) or {}).get("MeanER")
        if isinstance(v, (int, float)):
            vals.append(v)
    if vals:
        rows.append((sum(vals)/len(vals), len(vals), os.path.basename(f)[:-5]))
rows.sort()
print(f"{'rank':<5}{'MeanER':<10}{'n':<4}ckpt")
for i, (m, n, name) in enumerate(rows, 1):
    flag = "  <-- beats 11.87" if m < 11.87 else ""
    print(f"{i:<5}{m:<10.3f}{n:<4}{name}{flag}")
print("\nbaseline ssl_classical_clean=12.685 | best-so-far ssl_tuplet20_last=11.87 | released=10.77")
PY
echo "Per-ckpt JSON in $OUTDIR/"
