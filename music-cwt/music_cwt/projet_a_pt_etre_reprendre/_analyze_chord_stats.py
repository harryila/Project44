"""Helper d'analyse approfondie des CSV de chord_shape_stats."""
import csv
from collections import Counter

from pathlib import Path

HERE = Path(__file__).parent.resolve()
DATASETS = [
    ("Chopin first editions", HERE / "chord_stats_chopin.csv"),
    ("KernScores composers",  HERE / "chord_stats_kernscores.csv"),
]

for name, path in DATASETS:
    print(f"\n=== {name} ===")
    total = 0
    n_by_size = Counter()
    matched_by_size = Counter()
    atom_by_size = Counter()
    decor_by_size = Counter()
    q_3plus = Counter()
    v_3plus = Counter()
    qv_3plus = Counter()

    with open(path, "r", encoding="utf-8") as fh:
        r = csv.DictReader(fh)
        for row in r:
            total += 1
            n = int(row["n_notes"])
            n_by_size[n] += 1
            q = row["quality"]
            v = row["voicing"]
            atom = int(row["atomizable"])
            dec = int(row["has_hard_decoration"])
            if q != "UNMATCHED":
                matched_by_size[n] += 1
            if atom:
                atom_by_size[n] += 1
            if dec:
                decor_by_size[n] += 1
            if n >= 3 and q != "UNMATCHED":
                q_3plus[q] += 1
                v_3plus[v] += 1
                qv_3plus[(q, v)] += 1

    print(f"Total events : {total:,}")
    print()
    print(f"  {'size':>5} {'count':>10} {'%total':>7} {'matched':>10} {'%match':>7} {'atomiz':>10} {'%atom':>7} {'decor':>10}")
    for n in sorted(n_by_size):
        c = n_by_size[n]
        m = matched_by_size.get(n, 0)
        a = atom_by_size.get(n, 0)
        d = decor_by_size.get(n, 0)
        pct_total = 100 * c / total
        pct_match = 100 * m / c if c else 0
        pct_atom = 100 * a / c if c else 0
        print(f"  {n:>5} {c:>10,} {pct_total:>6.1f}% {m:>10,} {pct_match:>6.1f}% "
              f"{a:>10,} {pct_atom:>6.1f}% {d:>10,}")

    matched_3plus = sum(q_3plus.values())
    total_3plus = sum(c for n, c in n_by_size.items() if n >= 3)
    atom_3plus  = sum(c for n, c in atom_by_size.items() if n >= 3)

    print(f"\n  >> 3+ notes total: {total_3plus:,}  matched: {matched_3plus:,} "
          f"({100*matched_3plus/total_3plus:.1f}%)  atomizable: {atom_3plus:,} "
          f"({100*atom_3plus/total_3plus:.1f}%)")

    if matched_3plus == 0:
        continue

    print(f"\n  Top 12 qualites (sur 3+ matched) :")
    for q, c in q_3plus.most_common(12):
        print(f"    {q:<14} {c:>8,} ({100*c/matched_3plus:.1f}%)")

    print(f"\n  Top 25 voicings (sur 3+ matched) — couverture cumulative :")
    cum = 0
    for v, c in v_3plus.most_common(25):
        cum += c
        print(f"    {v:<28} {c:>8,} ({100*c/matched_3plus:5.1f}%)  cum:{100*cum/matched_3plus:5.1f}%")

    print(f"\n  Top 25 (qualite, voicing) — couverture cumulative :")
    cum = 0
    for (q, v), c in qv_3plus.most_common(25):
        cum += c
        print(f"    {q:<10} {v:<28} {c:>8,} ({100*c/matched_3plus:5.1f}%)  cum:{100*cum/matched_3plus:5.1f}%")
