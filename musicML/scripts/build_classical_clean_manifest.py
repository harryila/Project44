"""Leak-filter the classical unpaired manifest: drop any PDMX score matching an ASAP TEST piece,
so the eval (which uses ASAP test pieces) is not contaminated by the unpaired training corpus.
The released model filtered its unpaired data for overlap with the labeled set; we must too for a
fair comparison. Over-exclusion is safe (lose a few training scores); leakage is not.

Matches on composer_name + title + song_name (lowercased). Conservative-aggressive per ASAP test piece.
Writes data/pairs_classical_clean_manifest.csv + reports how many rows dropped."""
from __future__ import annotations
import re
import pandas as pd
import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
from musicml_paths import PDMX_ROOT as _PDMX

SUBSET = "data/pdmx_piano_subset.csv"
MANIFEST = "data/pairs_classical_manifest.csv"
OUT = "data/pairs_classical_clean_manifest.csv"
PDMX_ROOT = str(_PDMX) + "/"

# (label, regex over "composer | title | song_name" lowercased). Aggressive but composer-anchored
# where the title words are common. The 14 ASAP test pieces:
PATTERNS = [
    ("bach_bwv846",     r"bwv\.?\s?846|(bach).*(wtc|well.temper).*(prelude|fug).*(\b1\b|no\.?\s?1|c\b)"),
    ("beethoven_op14",  r"(beethoven).*(op\.?\s?14\b|sonat.*(\bno\.?\s?9\b|\bno\.?\s?10\b|\b9\b|\b10\b))"),
    ("chopin_ballade1", r"(chopin).*ballad.*(\b1\b|no\.?\s?1|op\.?\s?23)"),
    ("debussy_reflets", r"reflets|(debussy).*(reflet|images.*(eau|water))"),
    ("haydn_31",        r"(haydn).*(sonat.*\b31\b|hob.*xvi.?\s?31)"),
    ("liszt_annees2",   r"(liszt).*(gondol|venezia|napoli|annees.*(2|ii|deux)|pelerinage.*(2|ii|second|italie|italy|deux))"),
    ("mozart_k332",     r"k\.?\s?332\b|(mozart).*sonat.*(\bno\.?\s?12\b|\b12\b|\bf\b.?\s?(major|maj))"),
    ("rach_op23_4",     r"(rachman).*(op\.?\s?23.*(\b4\b|no\.?\s?4)|prelude.*\b23\b.*\b4\b)"),
    ("schubert_imp",    r"(schubert).*impromptu.*(op\.?\s?90|d\.?\s?899|\b1\b|c\b.?minor)"),
    ("schumann_arab",   r"arabeske|(schumann).*arabesqu"),
    ("scriabin_op8",    r"(scriabin).*(op\.?\s?8\b|etude.*op\.?\s?8)"),
    ("brahms_op118_2",  r"(brahms).*(op\.?\s?118.*(\b2\b|no\.?\s?2)|intermezzo.*118)"),
    ("prokofiev_tocc",  r"(prokofiev).*toccata|toccata.*op\.?\s?11"),
    ("ravel_ondine",    r"ondine|gaspard"),
]
RES = [(lab, re.compile(rx)) for lab, rx in PATTERNS]


def main():
    df = pd.read_csv(SUBSET, low_memory=False)
    for c in ("composer_name", "title", "song_name"):
        if c not in df.columns:
            df[c] = ""
    text = (df["composer_name"].fillna("").astype(str) + " | "
            + df["title"].fillna("").astype(str) + " | "
            + df["song_name"].fillna("").astype(str)).str.lower()
    leaked_paths = set()
    counts = {}
    for lab, rx in RES:
        hit = df[text.apply(lambda s: bool(rx.search(s)))]
        counts[lab] = len(hit)
        for p in hit["mxl"].astype(str):
            leaked_paths.add(PDMX_ROOT + p.lstrip("./"))
    print("leak matches per test piece:", counts, flush=True)
    print(f"total distinct leaked PDMX mxl paths: {len(leaked_paths)}", flush=True)

    man = pd.read_csv(MANIFEST, dtype={"id": str}, low_memory=False)
    before = len(man)
    clean = man[~man["src_mxl"].astype(str).isin(leaked_paths)]
    dropped = before - len(clean)
    clean.to_csv(OUT, index=False)
    print(f"manifest: {before} -> {len(clean)} rows ({dropped} leaked rows dropped) -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
