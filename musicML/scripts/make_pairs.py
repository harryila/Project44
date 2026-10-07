"""Generate synthetic (perturbed-MIDI, MusicXML) training pairs from PDMX.

For each .mxl file in the piano subset:
    1. Render to perturbed MIDI via expressive_render.render
    2. Build _chunks.json from the alignment.json + score measure structure
       (1:1 mapping by construction, no need for ASAP-style beat swaps)
    3. Tokenize MIDI + MXL via MIDI2ScoreTransformer's MultistreamTokenizer
       and save to cache_pdmx/<sha>.pkl in the same format ASAPDataset expects.

Layout:
    pairs/{id}.mid              perturbed performance MIDI
    pairs/{id}.alignment.json   per-MIDI-note alignment (renderer's output)
    pairs/{id}_chunks.json      per-measure index lists (chunker schema)
    pairs/{id}.musicxml         copy of source PDMX MusicXML (for parse_mxl)
    data/cache_pdmx/<sha>.pkl   pickled (input_stream, output_stream)
    pairs/_errors.log           one line per skipped file
    pairs/_manifest.csv         {id, mxl_src, midi, mxl, n_notes, n_measures}
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import multiprocessing as mp
import shutil
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from joblib import Parallel, delayed

SCRIPT_DIR = Path(__file__).resolve().parent.parent
TOKENIZER_DIR = SCRIPT_DIR / "MIDI2ScoreTransformer" / "midi2scoretransformer"
if str(TOKENIZER_DIR) not in sys.path:
    sys.path.insert(0, str(TOKENIZER_DIR))
SCRIPTS_DIR = SCRIPT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from expressive_render import render
import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
from musicml_paths import PDMX_ROOT as _PDMX

log = logging.getLogger(__name__)


def sha256(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def build_chunks_from_alignment(alignment_path: Path) -> dict:
    """Build a chunks.json with per-beat midi/mxl index lists.

    Matches the format produced by `MIDI2ScoreTransformer/midi2scoretransformer/
    chunker.py` (which chunks per beat via ASAP's `midi_score_beats` and
    `performance_beats`). Because we render MIDI 1:1 from the score (no missing/
    extra notes, identical canonical sort order), midi[b] == mxl[b] for every
    beat b.

    Schema (matches chunker.py output):
        {
            "midi":    list[list[int]],  # per-beat sorted indices
            "mxl":     list[list[int]],  # per-beat sorted indices
            "swapped": bool,             # always False for synthetic
        }
    """
    align = json.loads(alignment_path.read_text())
    n_beats = align["n_beats"]
    midi_chunks = [[] for _ in range(n_beats)]
    mxl_chunks = [[] for _ in range(n_beats)]
    for entry in align["alignment"]:
        b = entry["beat_idx"]
        if 0 <= b < n_beats:
            midi_chunks[b].append(int(entry["midi_idx"]))
            mxl_chunks[b].append(int(entry["score_idx"]))
    midi_chunks = [sorted(c) for c in midi_chunks]
    mxl_chunks = [sorted(c) for c in mxl_chunks]
    return {"midi": midi_chunks, "mxl": mxl_chunks, "swapped": False}


def parse_and_cache(mxl_path: Path, midi_path: Path, cache_dir: Path) -> tuple[Path, dict]:
    """Run parse_midi + parse_mxl, save (input_stream, output_stream) to a sha256-keyed pkl.

    Returns (cache_pkl_path, stats_dict).
    """
    from tokenizer import MultistreamTokenizer
    input_stream = MultistreamTokenizer.parse_midi(str(midi_path))
    output_stream = MultistreamTokenizer.parse_mxl(str(mxl_path))
    n_in = int(input_stream["pitch"].shape[0])
    n_out = int(output_stream["pitch"].shape[0])
    if n_in != n_out:
        raise ValueError(f"Token count mismatch: midi={n_in}, mxl={n_out}")
    # Use the midi path string as the cache key so a single file maps to a single pkl
    cache_key = sha256(str(midi_path))
    pkl_path = cache_dir / f"{cache_key}.pkl"
    cache_dir.mkdir(parents=True, exist_ok=True)
    torch.save((input_stream, output_stream), pkl_path)
    return pkl_path, {"n_in": n_in, "n_out": n_out}


def process_one(idx: int, src_mxl: Path, dst_dir: Path, cache_dir: Path,
                seed: int) -> dict:
    """Render + chunkify + cache one PDMX piece.

    Returns a result dict:
        {ok: True, manifest_row: {...}}
        {ok: False, error: "stage:msg", id: pid, src: str}
    Designed for joblib parallelism — no shared file handles inside this fn.
    """
    pid = f"{idx:06d}"
    out_midi = dst_dir / f"{pid}.mid"
    src_ext = "".join(src_mxl.suffixes) or src_mxl.suffix or ".mxl"
    if src_ext.endswith(".gz") or src_ext.endswith(".zip"):
        src_ext = ".mxl"
    out_mxl = dst_dir / f"{pid}{src_ext}"
    out_chunks = dst_dir / f"{pid}_chunks.json"

    if out_midi.exists() and out_chunks.exists():
        return {"ok": False, "error": "already_done", "id": pid, "src": str(src_mxl)}

    if not src_mxl.exists():
        return {"ok": False, "error": "missing_src", "id": pid, "src": str(src_mxl)}

    try:
        shutil.copy(str(src_mxl), str(out_mxl))
    except Exception as exc:
        return {"ok": False, "error": f"copy:{exc}", "id": pid, "src": str(src_mxl)}

    result = render(out_mxl, out_midi, seed=seed)
    if not result.get("ok"):
        out_mxl.unlink(missing_ok=True)
        return {"ok": False, "error": f"render:{result.get('error', 'unknown')}",
                "id": pid, "src": str(src_mxl)}

    try:
        align_path = out_midi.with_suffix(".alignment.json")
        chunks = build_chunks_from_alignment(align_path)
        out_chunks.write_text(json.dumps(chunks))
    except Exception as exc:
        return {"ok": False, "error": f"chunks:{type(exc).__name__}: {exc}",
                "id": pid, "src": str(src_mxl)}

    try:
        pkl_path, stats = parse_and_cache(out_mxl, out_midi, cache_dir)
    except Exception as exc:
        return {"ok": False, "error": f"cache:{type(exc).__name__}: {exc}",
                "id": pid, "src": str(src_mxl)}

    return {
        "ok": True,
        "manifest_row": {
            "id": pid,
            "src_mxl": str(src_mxl),
            "midi": str(out_midi),
            "mxl": str(out_mxl),
            "chunks": str(out_chunks),
            "cache": str(pkl_path),
            "n_notes": result["n_notes"],
            "n_measures": result["n_measures"],
            "n_in_tokens": stats["n_in"],
            "n_out_tokens": stats["n_out"],
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset-csv", default="data/pdmx_piano_subset.csv")
    ap.add_argument("--mxl-root", default=str(_PDMX),
                    help="Root where ./mxl/.../*.mxl is found (paths in subset csv "
                         "are relative to this root, e.g. './mxl/1/11/...mxl')")
    ap.add_argument("--out-dir", default="data/pairs",
                    help="Output dir for {id}.mid, {id}.musicxml, {id}_chunks.json")
    ap.add_argument("--cache-dir", default="data/cache_pdmx",
                    help="Cache dir for tokenized .pkl files")
    ap.add_argument("--manifest", default="data/pairs/_manifest.csv")
    ap.add_argument("--errors", default="data/pairs/_errors.log")
    ap.add_argument("--n", type=int, default=5000, help="Pairs to attempt")
    ap.add_argument("--seed", type=int, default=42, help="Sampling seed for piece selection")
    ap.add_argument("--prefer-multi-track", action="store_true",
                    help="Prefer 2+ track piano (LH+RH) over single-track")
    ap.add_argument("--n-jobs", type=int, default=-1,
                    help="Parallel workers. -1 = all cores. Default: -1.")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    df = pd.read_csv(args.subset_csv)
    log.info("Loaded subset: %d rows", len(df))

    if args.prefer_multi_track:
        df["_tk"] = df["tracks"].apply(lambda s: str(s).count("-"))
        df = df.sort_values(["_tk", "n_notes"], ascending=[False, True])
        df = df.drop(columns=["_tk"])
    else:
        df = df.sample(n=min(args.n * 2, len(df)), random_state=args.seed)

    # 10% buffer to absorb parse failures
    n_candidates = int(args.n * 1.1) + 50
    df = df.head(n_candidates).reset_index(drop=True)

    out_dir = Path(args.out_dir)
    cache_dir = Path(args.cache_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    mxl_root = Path(args.mxl_root)
    manifest_path = Path(args.manifest)
    error_log_path = Path(args.errors)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    error_log_path.parent.mkdir(parents=True, exist_ok=True)

    fields = ["id", "src_mxl", "midi", "mxl", "chunks", "cache",
              "n_notes", "n_measures", "n_in_tokens", "n_out_tokens"]
    write_header = not manifest_path.exists()

    # Build the worklist
    work_items = []
    for seen, (_, row) in enumerate(df.iterrows()):
        mxl_rel = str(row["mxl"])
        if mxl_rel.startswith("./"):
            mxl_rel = mxl_rel[2:]
        src_mxl = mxl_root / mxl_rel
        work_items.append((seen, src_mxl, args.seed + seen))
    log.info("Dispatching %d items across %d workers...", len(work_items), args.n_jobs)

    t0 = time.time()
    # joblib.Parallel + threading-based progress reporting
    results = Parallel(n_jobs=args.n_jobs, verbose=10, batch_size=4)(
        delayed(process_one)(idx, src, dst_dir=out_dir, cache_dir=cache_dir, seed=seed)
        for idx, src, seed in work_items
    )

    ok = 0
    skipped = 0
    with manifest_path.open("a", newline="") as mf, error_log_path.open("a") as ef:
        writer = csv.DictWriter(mf, fieldnames=fields)
        if write_header:
            writer.writeheader()
        for r in results:
            if r.get("ok"):
                writer.writerow(r["manifest_row"])
                ok += 1
            else:
                ef.write(f"{r.get('id', '?')}\t{r.get('error', 'unknown')}\t{r.get('src', '?')}\n")
                skipped += 1

    dt = time.time() - t0
    log.info("Done. ok=%d skipped=%d total_seen=%d in %.1fs (%.2f items/s)",
             ok, skipped, len(work_items), dt, len(work_items) / max(dt, 1))
    log.info("Manifest: %s", manifest_path)
    log.info("Errors:   %s", error_log_path)


if __name__ == "__main__":
    main()
