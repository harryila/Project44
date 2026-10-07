"""Filter PDMX.csv to a piano-only subset.

Filter chain (priority order):
1. subset:no_license_conflict == True  (PDMX authors' recommendation)
2. subset:all_valid == True           (has the .mxl file we need)
3. tracks string is all "0" tokens    (every track is MIDI program 0 = piano)

Single-track piano (tracks == "0") is mostly lead-sheet-style and includes a lot
of hymn / popular arrangements. Two-track piano (tracks == "0-0") and beyond
are classic LH+RH or 4-voice arrangements. We keep both.

Output: pdmx_piano_subset.csv with the path/mxl/composer/title/n_notes/etc that
we need for the renderer.
"""

import argparse
from pathlib import Path

import pandas as pd
import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
from musicml_paths import PDMX_ROOT as _PDMX


def is_all_piano(tracks: str) -> bool:
    if pd.isna(tracks):
        return False
    parts = str(tracks).split("-")
    return all(p == "0" for p in parts) and len(parts) >= 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(_PDMX / "PDMX.csv"),
                    help="Path to PDMX.csv")
    ap.add_argument("--out", default="data/pdmx_piano_subset.csv",
                    help="Output path for the filtered subset")
    ap.add_argument("--min-notes", type=int, default=32,
                    help="Drop very short pieces (default 32 notes)")
    ap.add_argument("--max-notes", type=int, default=10000,
                    help="Drop very long pieces (default 10000 notes)")
    args = ap.parse_args()

    df = pd.read_csv(args.csv, low_memory=False)
    n0 = len(df)
    print(f"Total rows: {n0}")

    # 1. License safety
    df = df[df["subset:no_license_conflict"].astype(bool)]
    print(f"After no_license_conflict: {len(df)}  ({100*len(df)/n0:.1f}%)")

    # 2. All files valid
    df = df[df["subset:all_valid"].astype(bool)]
    print(f"After all_valid:           {len(df)}")

    # 3. Piano only (every track is program 0)
    is_piano = df["tracks"].apply(is_all_piano)
    df = df[is_piano]
    print(f"After all-piano tracks:    {len(df)}")

    # 4. Reasonable length
    df = df[(df["n_notes"] >= args.min_notes) & (df["n_notes"] <= args.max_notes)]
    print(f"After {args.min_notes}\u2264n_notes\u2264{args.max_notes}:   {len(df)}")

    # Keep only useful columns
    keep_cols = [
        "path", "mxl", "mid",
        "title", "song_name", "composer_name", "artist_name",
        "license", "genres", "tracks", "n_tracks",
        "n_notes", "song_length.bars", "song_length.beats",
        "complexity", "rating", "n_ratings",
        "subset:rated", "subset:deduplicated",
        "subset:rated_deduplicated", "subset:no_license_conflict",
        "subset:all_valid",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols]

    # Distribution sanity check
    print("\nTrack-shape distribution within piano subset:")
    print(df["tracks"].value_counts().head(8))
    print("\ngenres (top 8):")
    print(df["genres"].value_counts().head(8))
    print(f"\nclassical-tagged: {df['genres'].astype(str).str.contains('classical', na=False).sum()}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nSaved {len(df)} rows to {out_path}")


if __name__ == "__main__":
    main()
