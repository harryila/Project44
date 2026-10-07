"""Compare Tier-1 eval JSONs (baseline vs fine-tuned checkpoints) and pick the
best by real MUSTER MeanER — NOT by val/total (which the analysis showed is a
misleading teacher-forced metric). Reports overall + the hard-composer subset,
which is where Track C/D is supposed to help.

Usage:
    python scripts/compare_eval.py benchmark/tier1_baseline.json benchmark/eval_*.json
"""
import json
import sys
from collections import defaultdict

# the out-of-distribution-difficulty composers (baseline MeanER >= ~12)
HARD = {"Schumann", "Debussy", "Rachmaninoff", "Liszt", "Schubert", "Ravel"}


def load(path):
    d = json.load(open(path))
    agg = d.get("aggregate", {})
    mean = (agg.get("muster") or {}).get("MeanER")
    byc = defaultdict(list)
    for r in d.get("per_performance", []):
        s = r.get("sim") or {}
        m = (s.get("muster") or {}).get("MeanER")
        if m is not None:
            byc[r["composer"]].append(m)
    per = {c: sum(v) / len(v) for c, v in byc.items()}
    hard = [m for c, v in byc.items() if c in HARD for m in v]
    return {
        "path": path, "label": d.get("meta", {}).get("ckpt", path),
        "MeanER": mean, "per_composer": per,
        "hard_MeanER": sum(hard) / len(hard) if hard else None,
        "n": agg.get("n_scored"),
    }


def main():
    paths = sys.argv[1:]
    if not paths:
        print("usage: compare_eval.py <baseline.json> <finetuned.json> ...")
        sys.exit(1)
    runs = [load(p) for p in paths]
    base = runs[0]
    print("=" * 84)
    print(f"{'run':42s} {'n':>4s} {'MeanER':>8s} {'hardMeanER':>11s}  vs baseline")
    print("=" * 84)
    for r in runs:
        d_all = (r["MeanER"] - base["MeanER"]) if (r["MeanER"] and base["MeanER"]) else None
        tag = "  (baseline)" if r is base else (
            f"  Δ={d_all:+.2f} {'BETTER' if d_all and d_all < -0.28 else ('worse' if d_all and d_all > 0.28 else 'noise')}")
        print(f"{str(r['label'])[-42:]:42s} {str(r['n']):>4s} "
              f"{(r['MeanER'] or float('nan')):8.2f} {(r['hard_MeanER'] or float('nan')):11.2f}{tag}")

    # best by MeanER (excluding baseline)
    ft = [r for r in runs if r is not base and r["MeanER"] is not None]
    if ft:
        best = min(ft, key=lambda r: r["MeanER"])
        print("\nBEST fine-tuned by MeanER:", best["path"])
        print(f"  overall MeanER {best['MeanER']:.2f} vs baseline {base['MeanER']:.2f} "
              f"({best['MeanER']-base['MeanER']:+.2f})")
        if best["hard_MeanER"] and base["hard_MeanER"]:
            print(f"  hard-composer MeanER {best['hard_MeanER']:.2f} vs baseline "
                  f"{base['hard_MeanER']:.2f} ({best['hard_MeanER']-base['hard_MeanER']:+.2f})")
        # per-composer deltas
        print("\n  per-composer Δ (negative = improved):")
        for c in sorted(set(best["per_composer"]) & set(base["per_composer"]),
                        key=lambda c: best["per_composer"][c] - base["per_composer"][c]):
            d = best["per_composer"][c] - base["per_composer"][c]
            flag = "  <- HARD" if c in HARD else ""
            print(f"    {c:14s} {base['per_composer'][c]:6.2f} -> {best['per_composer'][c]:6.2f}  ({d:+.2f}){flag}")
        verdict = "PASS — beats baseline" if best["MeanER"] < base["MeanER"] - 0.28 else \
                  "NULL — within noise (try higher LR / more steps / learned rendering)"
        print(f"\n  VERDICT: {verdict}")


if __name__ == "__main__":
    main()
