"""Build a RUBATO training manifest for the realistic-rendering experiment.

Motivation: the legacy renderer held tempo constant within each bar, so tuplets landed at
exactly 1/N of the beat (deadpan-within-bar). Real test performances have structured sub-beat
rubato + cadential ritardando, so the timing->notation mapping the model must invert differs at
test time. expressive_render.py now defaults to a beat-resolution AR(1) rubato curve. This driver:

  1. KERN (tuplet-rich, 241 converted scores): re-render each one K times with DIFFERENT rubato
     seeds. Because each render is a distinct plausible performance of the same score, this is
     genuine data AUGMENTATION of the scarce tuplet corpus -- NOT the exact-repeat upweighting that
     made ar_full5 (kern x30 deadpan) overfit and regress (val 0.9895 vs ar_full3 0.9696).
  2. PDMX (the bulk): pass the existing 84K-cap rows through unchanged. They already get
     onset_jitter=0.05 at train time (pdmx_dataset augmentation), so they carry timing variance and
     the "rubato => tuplet" confound is mitigated without a (riskier, slower) 84K re-render.

Caches land in data/cache_pdmx keyed by sha256(midi_path) -- exactly where pdmx_dataset._load_pair
looks (cache_root = midi.parents[1]/"cache_pdmx"). Output: data/pairs_rubato_manifest.csv.
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed

REPO = Path(__file__).resolve().parent.parent
for p in (REPO / "MIDI2ScoreTransformer" / "midi2scoretransformer", REPO / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import json as _json  # noqa: E402

from expressive_render import render_from_parsed, Perturbations  # noqa: E402
from make_pairs import build_chunks_from_alignment, sha256  # noqa: E402

log = logging.getLogger("rerender_rubato")
FIELDS = ["id", "src_mxl", "midi", "mxl", "chunks", "cache",
          "n_notes", "n_measures", "n_in_tokens", "n_out_tokens"]


def render_kern_score(mxl: Path, K: int, out_dir: Path, cache_dir: Path, base_seed: int) -> list:
    """Parse one kern score ONCE, then emit K distinct rubato realizations.

    The music21 parse (mxl_to_list + parse_mxl) dominates cost and is identical across
    realizations, so we do it once and reuse both the note list AND the tokenized score
    (output_stream). Each realization only rebuilds the cheap tempo curve, writes a MIDI,
    and tokenizes that (small) MIDI. ~30x less parse work than re-rendering from scratch.
    """
    import torch
    from tokenizer import MultistreamTokenizer
    try:
        notes_list, score = MultistreamTokenizer.mxl_to_list(str(mxl))
        if not notes_list:
            return [{"ok": False, "error": "no_notes", "src": str(mxl)}]
        output_stream = MultistreamTokenizer.parse_mxl(str(mxl))  # shared across realizations
        n_out = int(output_stream["pitch"].shape[0])
    except Exception as exc:
        return [{"ok": False, "error": f"parse:{type(exc).__name__}: {exc}", "src": str(mxl)}]

    rows = []
    for j in range(K):
        pid = f"{mxl.stem}_r{j:02d}"
        out_midi = out_dir / f"{pid}.mid"
        out_chunks = out_dir / f"{pid}_chunks.json"
        try:
            result = render_from_parsed(notes_list, score, out_midi, base_seed + j,
                                        Perturbations(use_beat_rubato=True))
            if not result.get("ok"):
                rows.append({"ok": False, "error": f"render:{result.get('error')}", "src": str(mxl)})
                continue
            out_chunks.write_text(_json.dumps(
                build_chunks_from_alignment(out_midi.with_suffix(".alignment.json"))))
            input_stream = MultistreamTokenizer.parse_midi(str(out_midi))
            n_in = int(input_stream["pitch"].shape[0])
            if n_in != n_out:
                rows.append({"ok": False, "error": f"tok_mismatch {n_in}!={n_out}", "src": str(mxl)})
                continue
            pkl = cache_dir / f"{sha256(str(out_midi))}.pkl"
            torch.save((input_stream, output_stream), pkl)
            rows.append({"ok": True, "row": {
                "id": pid, "src_mxl": str(mxl), "midi": str(out_midi), "mxl": str(mxl),
                "chunks": str(out_chunks), "cache": str(pkl),
                "n_notes": result["n_notes"], "n_measures": result["n_measures"],
                "n_in_tokens": n_in, "n_out_tokens": n_out}})
        except Exception as exc:
            rows.append({"ok": False, "error": f"{type(exc).__name__}: {exc}", "src": str(mxl)})
    return rows


def run_shard(args) -> None:
    """Render kern scores[i::N] SEQUENTIALLY in this (clean, single) process.

    No joblib/loky: torch is imported at module load (via make_pairs), which deadlocks
    loky's forked workers. Instead we shard across independent OS processes (launched by
    a shell loop), each a fresh interpreter that imports torch once and works serially.
    Each shard writes its kern rows (no header) to <out>.shard<i>.csv.
    """
    i, n = (int(x) for x in args.shard.split("/"))
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir); cache_dir.mkdir(parents=True, exist_ok=True)
    kerns = sorted(Path().glob(args.kern_glob))
    mine = kerns[i::n]
    log.info("shard %d/%d: %d of %d kern scores", i, n, len(mine), len(kerns))
    shard_csv = Path(f"{args.manifest}.shard{i}.csv")
    t0 = time.time(); ok = fail = 0
    with shard_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        for j, k in enumerate(mine):
            global_idx = i + j * n  # stable seed regardless of shard count
            rows = render_kern_score(k, args.kern_mult, out_dir, cache_dir,
                                     args.seed + global_idx * args.kern_mult)
            for r in rows:
                if r.get("ok"):
                    w.writerow(r["row"]); ok += 1
                else:
                    fail += 1
            f.flush()
    log.info("shard %d/%d DONE: %d rows ok, %d failed, %.1fs -> %s",
             i, n, ok, fail, time.time() - t0, shard_csv)


def run_merge(args) -> None:
    """Combine PDMX passthrough + all kern shard CSVs into the final manifest."""
    df = pd.read_csv(args.pdmx_manifest, dtype={"id": str}, low_memory=False)
    pdmx = df[~df["id"].astype(str).str.startswith("k")].copy()
    shards = sorted(Path(".").glob(f"{Path(args.manifest).name}.shard*.csv")
                    if "/" not in args.manifest else Path(args.manifest).parent.glob(
                        f"{Path(args.manifest).name}.shard*.csv"))
    n_kern = 0
    man = Path(args.manifest)
    with man.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for _, row in pdmx.iterrows():
            w.writerow({k: row.get(k, "") for k in FIELDS})
        for sc in shards:
            with sc.open() as sf:
                r = csv.DictReader(sf, fieldnames=FIELDS)
                for row in r:
                    w.writerow(row); n_kern += 1
    total = len(pdmx) + n_kern
    log.info("WROTE %s : %d rows (%d PDMX + %d rubato-kern from %d shards, kern=%.1f%%)",
             man, total, len(pdmx), n_kern, len(shards), 100.0 * n_kern / max(total, 1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdmx-manifest", default="data/pairs_upweighted_manifest.csv",
                    help="Source of the 84K-cap PDMX rows (kern rows are dropped + replaced).")
    ap.add_argument("--kern-glob", default="data/pairs_kern/*.musicxml")
    ap.add_argument("--kern-mult", type=int, default=30, help="rubato realizations per kern score")
    ap.add_argument("--out-dir", default="data/pairs_kern_rubato")
    ap.add_argument("--cache-dir", default="data/cache_pdmx",
                    help="MUST be cache_pdmx: pdmx_dataset derives it from midi.parents[1].")
    ap.add_argument("--manifest", default="data/pairs_rubato_manifest.csv")
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--shard", default=None, help="i/N: render kern scores[i::N] in this process")
    ap.add_argument("--merge", action="store_true", help="merge PDMX + all kern shard CSVs")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    if args.merge:
        run_merge(args)
    elif args.shard:
        run_shard(args)
    else:
        ap.error("pass --shard i/N (render) or --merge")


if __name__ == "__main__":
    main()
