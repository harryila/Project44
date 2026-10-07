"""Compare ssl_classical_clean vs released pad-threshold sweeps at a chosen threshold.

Prints per-piece MeanER (ours vs released), the gap, and over-production proxies
(predicted/ground-truth note + tuplet counts) sorted by gap descending — so we can
see WHERE the gap is and whether it tracks over-production (n_ratio >> 1) vs the
dense-tuplet residual.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
T = sys.argv[1] if len(sys.argv) > 1 else "0.50"
ours = json.load(open(HERE / "padsweep_sslcc.json"))["results"]
rel = json.load(open(HERE / "padsweep_released.json"))["results"]
rd = {(x["composer"], x["piece"]): x for x in rel}

def cell(d, key):
    return d.get("per_threshold", {}).get(T, {}).get(key)


rows = []
for x in ours:
    y = rd.get((x["composer"], x["piece"]), {})
    om, rm = cell(x, "MeanER"), cell(y, "MeanER")
    if om is None or rm is None:
        continue
    gn, gt = x.get("gt_notes"), x.get("gt_tuplets")          # gt identical for both models
    o_pn, o_pt = cell(x, "pred_notes"), cell(x, "pred_tuplets")
    r_pn, r_pt = cell(y, "pred_notes"), cell(y, "pred_tuplets")
    rows.append((om - rm, x["composer"][:9], x["piece"][:16], om, rm,
                 o_pn, r_pn, gn, o_pt, r_pt, gt))

rows.sort(reverse=True)
print(f"threshold={T}   (note/tuplet columns are ours|rel vs gt)")
hdr = (f"{'piece':26s}{'ours':>6}{'rel':>6}{'gap':>7}"
       f"{'oN':>6}{'rN':>6}{'gtN':>6}{'oT':>6}{'rT':>6}{'gtT':>6}")
print(hdr)
print("-" * len(hdr))
om_sum = rm_sum = 0.0
for gap, c, p, om, rm, o_pn, r_pn, gn, o_pt, r_pt, gt in rows:
    name = c + "/" + p
    print(f"{name:26s}{om:6.2f}{rm:6.2f}{gap:+7.2f}"
          f"{str(o_pn):>6}{str(r_pn):>6}{str(gn):>6}{str(o_pt):>6}{str(r_pt):>6}{str(gt):>6}")
    om_sum += om
    rm_sum += rm
n = len(rows)
print("-" * len(hdr))
print(f"{'CORPUS MEAN':26s}{om_sum/n:6.2f}{rm_sum/n:6.2f}{(om_sum-rm_sum)/n:+7.2f}   (n={n})")
print("\nN = note count, T = tuplet count;  o=ours r=released gt=ground-truth")
