#!/usr/bin/env bash
# Box-side pad-threshold sweep for the now-LIVE soft keep-gate (see PAD_THRESHOLD_FIX.md).
#
# Before the fix, --pad-threshold was a no-op (generate() hard-argmaxed pad to binary at
# sigmoid=0.5). It now thresholds a continuous keep-probability with un-zeroed predictions,
# so lowering it rescues borderline dropped notes (attacks the Scriabin/dense MISS-RATE) and
# raising it prunes over-emission (attacks the Mozart OVER-PRODUCTION). At 0.50 the output is
# byte-identical to the released behaviour (locally verified by scripts/test_pad_prob_fix.py).
#
# This single sweep doubles as the "measure all 14 ASAP test pieces" prerequisite: with no
# --pieces filter, eval_tuplet.py scores every test piece (--limit-per 1 = shortest perf each),
# emitting per-piece MeanER + predicted/gt tuplet+note counts at every threshold.
#
# Usage (on the GPU box, repo root):
#   bash scripts/run_padsweep.sh [CKPT_GLOB] [TAG]
# Examples:
#   bash scripts/run_padsweep.sh 'checkpoints/ssl_classical_clean/*epoch=13*.ckpt' sslcc
#   bash scripts/run_padsweep.sh 'checkpoints/released/*.ckpt' released   # A/B the SOTA ckpt
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

CKPT_GLOB="${1:-checkpoints/ssl_classical_clean/*.ckpt}"
TAG="${2:-sslcc}"
PY="${PY:-venv311/bin/python}"
THRESHOLDS=(0.60 0.55 0.50 0.45 0.40 0.35 0.30 0.25)
OUTDIR="benchmark/padsweep_${TAG}"
mkdir -p "$OUTDIR"

# Resolve the checkpoint (newest match if the glob is broad).
CKPT="$(ls -t $CKPT_GLOB 2>/dev/null | head -1)"
if [[ -z "${CKPT}" ]]; then
  echo "ERROR: no checkpoint matched: ${CKPT_GLOB}" >&2
  echo "Available under checkpoints/:" >&2
  ls -1 checkpoints/ 2>/dev/null >&2
  exit 1
fi
echo "CKPT = ${CKPT}"
echo "TAG  = ${TAG}   OUT = ${OUTDIR}"
echo "THRESHOLDS = ${THRESHOLDS[*]}"

for T in "${THRESHOLDS[@]}"; do
  OUT="${OUTDIR}/padsweep_${T}.json"
  echo "=== pad_threshold=${T}  ->  ${OUT} ==="
  "${PY}" benchmark/eval_tuplet.py \
      --ckpt "${CKPT}" --device cuda \
      --pad-threshold "${T}" \
      --limit-per 1 \
      --overlap 64 --chunk 512 \
      --out "${OUT}"
done

echo
echo "=== SUMMARY (MeanER per piece x threshold) ==="
"${PY}" - "${OUTDIR}" <<'PYEOF'
import json, sys, glob, os
outdir = sys.argv[1]
rows = {}          # piece -> {thr: MeanER}
tup  = {}          # piece -> {thr: "pred/gt"}
thrs = []
for f in sorted(glob.glob(os.path.join(outdir, "padsweep_*.json"))):
    thr = os.path.basename(f).replace("padsweep_", "").replace(".json", "")
    thrs.append(thr)
    data = json.load(open(f))
    for r in data.get("results", []):
        name = f"{r.get('composer','?')[:10]}/{r.get('piece','?')[:22]}"
        rows.setdefault(name, {})[thr] = r.get("MeanER")
        if r.get("gt_tuplets"):
            tup.setdefault(name, {})[thr] = f"{r.get('pred_tuplets','?')}/{r.get('gt_tuplets','?')}"
hdr = "piece".ljust(34) + "".join(t.rjust(8) for t in thrs)
print(hdr); print("-" * len(hdr))
for name in sorted(rows):
    cells = "".join((f"{rows[name].get(t):.2f}".rjust(8) if isinstance(rows[name].get(t), (int, float)) else "—".rjust(8)) for t in thrs)
    print(name.ljust(34) + cells)
# best threshold per piece + corpus mean at each threshold
print("\nBest pad_threshold per piece (min MeanER):")
for name in sorted(rows):
    vals = {t: v for t, v in rows[name].items() if isinstance(v, (int, float))}
    if vals:
        bt = min(vals, key=vals.get)
        print(f"  {name.ljust(34)} best={bt} ({vals[bt]:.2f})  vs 0.50=({rows[name].get('0.50','—')})")
print("\nCorpus-mean MeanER per threshold:")
for t in thrs:
    vals = [rows[n][t] for n in rows if isinstance(rows[n].get(t), (int, float))]
    if vals:
        print(f"  thr={t}: mean={sum(vals)/len(vals):.3f}  (n={len(vals)})")
PYEOF
echo "Done. Per-threshold JSON in ${OUTDIR}/"
