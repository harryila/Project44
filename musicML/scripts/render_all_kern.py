"""Robust rubato re-render of the kern corpus, self-contained.

Why this exists: joblib/loky deadlocks here (torch is imported in the parent before workers
fork), and shell-backgrounded shards kept dying with the flaky SSH. This script owns its own
parallelism via a `spawn` ProcessPoolExecutor (fresh interpreters => no fork-after-torch
deadlock; workers are reused => no 240-way import storm) and is launched as ONE process.

Renders each kern score K times with distinct rubato seeds (augmentation of the scarce tuplet
corpus), then writes data/pairs_rubato_manifest.csv = 84K PDMX passthrough + kern rubato rows.
Progress is logged every score so it can be polled. Run from the repo root.
"""
from __future__ import annotations

import concurrent.futures as cf
import csv
import sys
import time
from multiprocessing import get_context
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
for p in (REPO / "MIDI2ScoreTransformer" / "midi2scoretransformer", REPO / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

FIELDS = ["id", "src_mxl", "midi", "mxl", "chunks", "cache",
          "n_notes", "n_measures", "n_in_tokens", "n_out_tokens"]
OUT_DIR = REPO / "data" / "pairs_kern_rubato"
CACHE_DIR = REPO / "data" / "cache_pdmx"
KERN_GLOB = "data/pairs_kern/*.musicxml"
PDMX_MANIFEST = "data/pairs_upweighted_manifest.csv"
MANIFEST = "data/pairs_rubato_manifest.csv"
K = 30
SEED = 1000


def render_one_score(mxl_str: str, base_seed: int) -> list:
    """Spawned-worker entrypoint: parse one score once, emit K rubato realizations."""
    import json as _json
    import torch
    from tokenizer import MultistreamTokenizer
    from expressive_render import render_from_parsed, Perturbations
    from make_pairs import build_chunks_from_alignment, sha256
    mxl = Path(mxl_str)
    try:
        notes_list, score = MultistreamTokenizer.mxl_to_list(mxl_str)
        if not notes_list:
            return []
        output_stream = MultistreamTokenizer.parse_mxl(mxl_str)
        n_out = int(output_stream["pitch"].shape[0])
    except Exception:
        return []
    rows = []
    for j in range(K):
        pid = f"{mxl.stem}_r{j:02d}"
        out_midi = OUT_DIR / f"{pid}.mid"
        try:
            res = render_from_parsed(notes_list, score, out_midi, base_seed + j,
                                     Perturbations(use_beat_rubato=True))
            if not res.get("ok"):
                continue
            (OUT_DIR / f"{pid}_chunks.json").write_text(_json.dumps(
                build_chunks_from_alignment(out_midi.with_suffix(".alignment.json"))))
            input_stream = MultistreamTokenizer.parse_midi(str(out_midi))
            n_in = int(input_stream["pitch"].shape[0])
            if n_in != n_out:
                continue
            pkl = CACHE_DIR / f"{sha256(str(out_midi))}.pkl"
            torch.save((input_stream, output_stream), pkl)
            rows.append({"id": pid, "src_mxl": mxl_str, "midi": str(out_midi),
                         "mxl": mxl_str, "chunks": str(OUT_DIR / f"{pid}_chunks.json"),
                         "cache": str(pkl), "n_notes": res["n_notes"],
                         "n_measures": res["n_measures"], "n_in_tokens": n_in,
                         "n_out_tokens": n_out})
        except Exception:
            continue
    return rows


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    kerns = [str(p) for p in sorted(Path().glob(KERN_GLOB))]
    print(f"[render_all_kern] {len(kerns)} kern scores x{K} = {len(kerns)*K} renders", flush=True)

    kern_rows = []
    ctx = get_context("spawn")
    t0 = time.time()
    done = 0
    with cf.ProcessPoolExecutor(max_workers=24, mp_context=ctx) as ex:
        futs = {ex.submit(render_one_score, k, SEED + i * K): k for i, k in enumerate(kerns)}
        for fut in cf.as_completed(futs):
            try:
                rows = fut.result()
            except Exception as e:
                rows = []
                print(f"  worker error on {futs[fut]}: {e}", flush=True)
            kern_rows.extend(rows)
            done += 1
            if done % 10 == 0 or done == len(kerns):
                print(f"  {done}/{len(kerns)} scores, {len(kern_rows)} rows, "
                      f"{time.time()-t0:.0f}s", flush=True)

    # merge with PDMX passthrough
    import pandas as pd
    df = pd.read_csv(PDMX_MANIFEST, dtype={"id": str}, low_memory=False)
    pdmx = df[~df["id"].astype(str).str.startswith("k")]
    with open(MANIFEST, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for _, row in pdmx.iterrows():
            w.writerow({k: row.get(k, "") for k in FIELDS})
        for row in kern_rows:
            w.writerow(row)
    total = len(pdmx) + len(kern_rows)
    print(f"[render_all_kern] WROTE {MANIFEST}: {total} rows "
          f"({len(pdmx)} PDMX + {len(kern_rows)} rubato-kern, "
          f"kern={100.0*len(kern_rows)/max(total,1):.1f}%)", flush=True)


if __name__ == "__main__":
    main()
