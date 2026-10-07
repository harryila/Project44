"""Augment an SSL manifest with a per-score `tuplet_rate` column.

tuplet_rate = fraction of a score's notes whose DURATION is a tuplet (non-dyadic on the 1/24
grid: round(quarterLength*24) not divisible by 3). This is the signal whose collapse drives the
NoteDuration error (LAB_REPORT §11). The reshape lever upweights tuplet-rich scores in sampling
so the model's tuplet prior is calibrated by the data distribution, not a blunt global loss weight.

Run (on the box):
  venv311/bin/python scripts/compute_tuplet_rates.py \
    --manifest data/pairs_classical_clean_manifest.csv \
    --out data/pairs_classical_clean_tuplrate.csv
"""
import argparse
import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed


def score_tuplet_rate(cache_path):
    try:
        ins, outs = torch.load(cache_path, weights_only=False)
        d = outs["duration"]
        if d is None or len(d) == 0:
            return (float("nan"), 0)
        t24 = torch.round(d.float() * 24).long()
        is_tup = ((t24 % 3) != 0) & (t24 > 0)
        return (float(is_tup.float().mean()), int(len(d)))
    except Exception:
        return (float("nan"), 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-jobs", type=int, default=-1)
    args = ap.parse_args()

    df = pd.read_csv(args.manifest)
    print(f"scoring {len(df)} pairs for tuplet_rate...", flush=True)
    res = Parallel(n_jobs=args.n_jobs, batch_size=64, verbose=5)(
        delayed(score_tuplet_rate)(c) for c in df["cache"].tolist()
    )
    df["tuplet_rate"] = [r[0] for r in res]
    df["n_notes_cache"] = [r[1] for r in res]
    df.to_csv(args.out, index=False)

    tr = df["tuplet_rate"].dropna().values
    print(f"\nwrote {args.out}  ({len(df)} rows, {np.isnan(df['tuplet_rate'].values).sum()} NaN)")
    print("tuplet_rate distribution:")
    for q in [0, 10, 25, 50, 75, 90, 95, 99, 100]:
        print(f"  p{q:>3}: {np.percentile(tr, q):.4f}")
    print(f"  mean: {tr.mean():.4f}")
    print(f"  fraction of scores with ZERO tuplets:    {(tr == 0).mean():.3f}")
    print(f"  fraction with tuplet_rate > 0.05:        {(tr > 0.05).mean():.3f}")
    print(f"  fraction with tuplet_rate > 0.15:        {(tr > 0.15).mean():.3f}")
    # corpus-mean tuplet exposure under weight ~ (rate+floor)^gamma, for sizing gamma
    for floor in (0.05,):
        for gamma in (0.0, 1.0, 2.0, 3.0):
            w = np.power(tr + floor, gamma)
            w = w / w.sum()
            eff = float((w * tr).sum())
            print(f"  exposure-weighted mean tuplet_rate (floor={floor}, gamma={gamma}): {eff:.4f}")


if __name__ == "__main__":
    main()
