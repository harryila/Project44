"""Aggregate the per-score tuplet rates of the classical training corpus into a
single, reproducible artifact — so the headline "~1.7% tuplets" figure (the
leading explanation for the placement gap) is grounded in a committed JSON
rather than living only in session-log narrative.

Source: data/pairs_classical_clean_tuplrate.csv — the per-score `tuplet_rate`
column produced by scripts/compute_tuplet_rates.py.
tuplet_rate = fraction of a score's notes whose DURATION is a tuplet
(non-dyadic on the 1/24-quarter grid).

Usage:  venv311/bin/python scripts/corpus_tuplet_stats.py
Writes: benchmark/corpus_tuplet_stats.json   (CPU only; no GPU, no model)
"""
import json
from datetime import date

import numpy as np
import pandas as pd

CSV = "data/pairs_classical_clean_tuplrate.csv"
OUT = "benchmark/corpus_tuplet_stats.json"

df = pd.read_csv(CSV).dropna(subset=["tuplet_rate"])
tr = df["tuplet_rate"].astype(float).to_numpy()
nn = df["n_notes"].astype(float).to_numpy()

total_notes = int(nn.sum())
total_tuplet_notes = int(round(float((tr * nn).sum())))

stats = {
    "corpus": "PDMX classical (the from-scratch / SSL training corpus)",
    "source_csv": CSV,
    "definition": (
        "tuplet_rate = fraction of a score's notes whose duration is a tuplet "
        "(non-dyadic on the 1/24-quarter grid); per scripts/compute_tuplet_rates.py"
    ),
    "computed": str(date.today()),
    "n_scores": int(len(df)),
    "total_notes": total_notes,
    "total_tuplet_notes": total_tuplet_notes,
    # the two headline figures, now grounded:
    "mean_per_score_tuplet_rate_pct": round(100.0 * float(tr.mean()), 3),
    "note_level_tuplet_fraction_pct": round(100.0 * total_tuplet_notes / total_notes, 3),
    "pct_scores_tuplet_free": round(100.0 * float((tr == 0).mean()), 2),
    # context for the data-engine "tuplet-rich subset" lever:
    "median_per_score_tuplet_rate_pct": round(100.0 * float(np.median(tr)), 3),
    "pct_scores_ge_10pct_tuplets": round(100.0 * float((tr >= 0.10).mean()), 2),
    "p90_per_score_tuplet_rate_pct": round(100.0 * float(np.percentile(tr, 90)), 3),
}

with open(OUT, "w") as f:
    json.dump(stats, f, indent=2)
print(json.dumps(stats, indent=2))
print(f"\nwrote {OUT}")
