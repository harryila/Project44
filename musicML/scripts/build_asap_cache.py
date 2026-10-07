#!/usr/bin/env python3
"""Parallel ASAP parse-cache warmer.

The ASAPDataset caches `(input_stream, output_stream)` — the raw parsed MIDI/MXL
streams, BEFORE bucketing and BEFORE beat-phase — to data/cache/*.pkl on first
__getitem__. music21's parse_mxl is the slow step; warming the cache once with many
processes makes every subsequent training epoch fast. The cache is identical whether
or not beat-conditioning is on (beat-phase + bucketing happen post-load), so a single
warm serves both baseline and --use-beat-conditioning runs.

Usage:
  PYTHONPATH=MIDI2ScoreTransformer/midi2scoretransformer \
    venv311/bin/python scripts/build_asap_cache.py --data-dir ./MIDI2ScoreTransformer/data/ --workers 32
"""
import argparse
import os
import sys
import time
from multiprocessing import Pool

# return_continous=True so __getitem__ writes the pkl (lines 165-169) then returns
# early — it never touches chunks/beats, so warming needs no beat annotations.
_DS = None


def _init(data_dir):
    global _DS
    import warnings
    warnings.simplefilter("ignore")
    from dataset import ASAPDataset
    _DS = ASAPDataset(
        data_dir=data_dir,
        split="all",
        seq_length=None,
        cache=True,
        padding=None,
        return_continous=True,
    )


def _warm(idx):
    try:
        _DS[idx]
        return (idx, True, "")
    except Exception as e:  # noqa: BLE001 — report per-item, don't kill the pool
        return (idx, False, f"{type(e).__name__}: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./data/")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 8)
    args = ap.parse_args()

    import warnings
    warnings.simplefilter("ignore")
    from dataset import ASAPDataset

    ds = ASAPDataset(
        data_dir=args.data_dir, split="all", seq_length=None,
        cache=True, padding=None, return_continous=True,
    )
    n = len(ds)
    del ds
    print(f"Warming ASAP cache: {n} samples, {args.workers} workers", flush=True)

    t0 = time.time()
    ok = 0
    fail = 0
    with Pool(args.workers, initializer=_init, initargs=(args.data_dir,)) as pool:
        for i, (idx, success, err) in enumerate(pool.imap_unordered(_warm, range(n)), 1):
            if success:
                ok += 1
            else:
                fail += 1
                print(f"  [FAIL {idx}] {err}", flush=True)
            if i % 50 == 0 or i == n:
                print(f"  {i}/{n}  ok={ok} fail={fail}  {time.time()-t0:.0f}s", flush=True)
    print(f"DONE: ok={ok} fail={fail} in {time.time()-t0:.0f}s", flush=True)
    sys.exit(1 if fail and ok == 0 else 0)


if __name__ == "__main__":
    main()
