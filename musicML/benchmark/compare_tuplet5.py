"""Compare ssl_classical_clean (baseline) vs ssl_tuplet5 (ep06 + last) vs released SOTA at thr 0.50.

Shows, per piece: MeanER for each model, predicted tuplet counts (the un-collapse test), and the
ground-truth tuplet count. The verdict on the tuplet-loss lever:
  - did the 9 zero-tuplet pieces start emitting tuplets (toward gtT)?
  - did corpus-mean MeanER move from ~12.69 toward the released 10.77?
  - did Mozart/Ravel/Prokofiev (our 3 wins) hold (no over-production blowup)?
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
T = "0.50"
# Optional argv: <best_json> <last_json> <label_best> <label_last>
F_BEST = sys.argv[1] if len(sys.argv) > 1 else "tuplet5_ep06.json"
F_LAST = sys.argv[2] if len(sys.argv) > 2 else "tuplet5_last.json"
L_BEST = sys.argv[3] if len(sys.argv) > 3 else "tup06"
L_LAST = sys.argv[4] if len(sys.argv) > 4 else "tupL"


def load(name):
    p = HERE / name
    if not p.exists():
        return {}
    return {(x["composer"], x["piece"]): x for x in json.load(open(p))["results"]}


base = load("padsweep_sslcc.json")       # ssl_classical_clean ep13
ep06 = load(F_BEST)                       # candidate best-val
last = load(F_LAST)                       # candidate last
rel = load("padsweep_released.json")     # released SOTA


def cell(d, k, key):
    return d.get(k, {}).get("per_threshold", {}).get(T, {}).get(key)


keys = list(base or rel)
hdr = (f"{'piece':24s}{'base':>7}{L_BEST:>7}{L_LAST:>7}{'rel':>7}"
       f"{'  | tuplets pred (base/best/last/rel) vs gt':<40}")
print(f"threshold {T}  — MeanER + predicted tuplet counts")
print(hdr)
print("-" * len(hdr))
sums = {"base": [0.0, 0], "ep06": [0.0, 0], "last": [0.0, 0], "rel": [0.0, 0]}
for k in keys:
    bm, e6, lm, rm = cell(base, k, "MeanER"), cell(ep06, k, "MeanER"), cell(last, k, "MeanER"), cell(rel, k, "MeanER")
    bt = cell(base, k, "pred_tuplets"); e6t = cell(ep06, k, "pred_tuplets")
    lt = cell(last, k, "pred_tuplets"); rt = cell(rel, k, "pred_tuplets")
    gt = base.get(k, {}).get("gt_tuplets") or rel.get(k, {}).get("gt_tuplets")

    def f(x):
        return f"{x:7.2f}" if isinstance(x, (int, float)) else f"{'—':>7}"
    for tag, v in (("base", bm), ("ep06", e6), ("last", lm), ("rel", rm)):
        if isinstance(v, (int, float)):
            sums[tag][0] += v
            sums[tag][1] += 1
    tup = f"   {str(bt):>4}/{str(e6t):>4}/{str(lt):>4}/{str(rt):>4} vs {str(gt):>4}"
    name = f"{k[0][:9]}/{k[1][:13]}"
    print(f"{name:24s}{f(bm)}{f(e6)}{f(lm)}{f(rm)}{tup}")
print("-" * len(hdr))


def mean(tag):
    s, n = sums[tag]
    return s / n if n else float("nan")


print(f"{'CORPUS MEAN':24s}{mean('base'):7.2f}{mean('ep06'):7.2f}{mean('last'):7.2f}{mean('rel'):7.2f}")
print(f"\nbase=ssl_classical_clean  {L_BEST}={F_BEST}  {L_LAST}={F_LAST}  rel=released SOTA")
print(f"gap to released: base {mean('base')-mean('rel'):+.2f} | {L_BEST} {mean('ep06')-mean('rel'):+.2f} | "
      f"{L_LAST} {mean('last')-mean('rel'):+.2f}")
