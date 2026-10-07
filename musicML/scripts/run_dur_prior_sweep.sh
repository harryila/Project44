#!/usr/bin/env bash
# Sweep the inference-time DURATION placement levers (A1 logit-adjust tau, A2 metrical prior lambda)
# over the 14 ASAP pieces by real MUSTER. Each (tau,lambda) is a fresh generation pass (the levers act
# during AR decoding). Pad threshold fixed at 0.50. Reports corpus-mean MeanER + per-config ranking.
#   bash scripts/run_dur_prior_sweep.sh <CKPT> <TAG>
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
REPO="$PWD"; PY="${PY:-venv311/bin/python}"
CKPT="$1"; TAG="${2:-durprior}"; PRIORS="${PRIORS:-$REPO/data/duration_priors.pt}"
OUT="benchmark/durprior_$TAG"; mkdir -p "$OUT"
# (tau lambda): baseline, A1-only sweep, A2-only sweep, combos
CONFIGS=("0 0" "0.5 0" "1.0 0" "1.5 0" "2.0 0" "0 0.5" "0 1.0" "0 2.0" "1.0 1.0" "1.5 0.5")
echo "CKPT=$CKPT  PRIORS=$PRIORS  configs=${#CONFIGS[@]}"
for cfg in "${CONFIGS[@]}"; do
  tau=${cfg% *}; lam=${cfg#* }
  out="$OUT/t${tau}_l${lam}.json"; [ -f "$out" ] && { echo "skip $tau/$lam"; continue; }
  echo "=== tau=$tau lambda=$lam -> $out ==="
  "$PY" benchmark/eval_padsweep.py --ckpt "$CKPT" --device cuda --limit-per 1 \
      --thresholds 0.50 --prior-path "$PRIORS" --dur-tau "$tau" --dur-metrical-lambda "$lam" \
      --out "$out" 2>&1 | tail -2
done
echo; echo "=== SUMMARY (corpus-mean MeanER @0.50, lower better; baseline=tau0/lam0) ==="
"$PY" - "$OUT" <<'PYEOF'
import json, sys, glob, os
od = sys.argv[1]; rows = []
for f in sorted(glob.glob(od + "/*.json")):
    try: d = json.load(open(f))
    except Exception: continue
    vals, ptup, gtup = [], 0, 0
    for r in d.get("results", []):
        v = (r.get("per_threshold", {}).get("0.50", {}) or {}).get("MeanER")
        if isinstance(v, (int, float)): vals.append(v)
        pt = (r.get("per_threshold", {}).get("0.50", {}) or {}).get("pred_tuplets")
        if isinstance(pt, int): ptup += pt
        if isinstance(r.get("gt_tuplets"), int): gtup += r["gt_tuplets"]
    if vals: rows.append((sum(vals)/len(vals), len(vals), ptup, gtup, os.path.basename(f)[:-5]))
rows.sort()
print(f"{'MeanER':<9}{'n':<4}{'pred_tup':<9}{'gt_tup':<7}config")
for m, n, pt, gt, nm in rows:
    flag = "  <-- beats 11.87" if m < 11.87 else ""
    print(f"{m:<9.3f}{n:<4}{pt:<9}{gt:<7}{nm}{flag}")
PYEOF
echo "JSON in $OUT/"
