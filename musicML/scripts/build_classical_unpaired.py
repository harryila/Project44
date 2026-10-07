"""Build a CLASSICAL unpaired-score corpus for masked-SSL (the lever after ssl_v2: pop-heavy PDMX
regressed the classical tail). Tokenizes the ~24k genre=classical PDMX scores into the same cache
format the UnpairedScoreDataset consumes. We reuse expressive_render (it produces cache + per-beat
chunks in one proven step) but DELETE the throwaway MIDI/alignment afterwards — the unpaired branch
masks the performance timing anyway, so only the tokenized cache + chunks are needed (saves disk).

Spawn ProcessPoolExecutor (joblib/loky deadlocks: torch imported pre-fork). Incremental manifest write
so a partial run is still usable. Run from repo root. Output: data/pairs_classical_manifest.csv."""
from __future__ import annotations
import concurrent.futures as cf
import csv
import hashlib
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
# argv: [1]=list-file of .musicxml paths, [2]=out-subdir under data/, [3]=manifest path. Defaults = the
# original genre=classical PDMX run. UnpairedScoreDataset derives cache from midi.parents[1]/cache_pdmx.
_LIST = sys.argv[1] if len(sys.argv) > 1 else "/root/classical_mxl_list.txt"
_SUB = sys.argv[2] if len(sys.argv) > 2 else "pairs_classical"
_MAN = sys.argv[3] if len(sys.argv) > 3 else "data/pairs_classical_manifest.csv"
OUT_DIR = REPO / "data" / _SUB
CACHE_DIR = REPO / "data" / "cache_pdmx"
LIST = _LIST
MANIFEST = _MAN


def build_one(mxl_str: str, seed: int):
    import json as _json
    from expressive_render import render
    from make_pairs import build_chunks_from_alignment, parse_and_cache
    mxl = Path(mxl_str)
    pid = "c" + hashlib.sha1(mxl_str.encode()).hexdigest()[:14]
    out_midi = OUT_DIR / f"{pid}.mid"
    out_chunks = OUT_DIR / f"{pid}_chunks.json"
    align = out_midi.with_suffix(".alignment.json")
    try:
        res = render(mxl, out_midi, seed=seed)
        if not res.get("ok"):
            return None
        out_chunks.write_text(_json.dumps(build_chunks_from_alignment(align)))
        pkl, stats = parse_and_cache(mxl, out_midi, CACHE_DIR)
        # we only need the cache + chunks; the masked unpaired branch never uses the perf timing.
        out_midi.unlink(missing_ok=True)
        align.unlink(missing_ok=True)
        return {"id": pid, "src_mxl": mxl_str, "midi": str(out_midi), "mxl": mxl_str,
                "chunks": str(out_chunks), "cache": str(pkl), "n_notes": res["n_notes"],
                "n_measures": res["n_measures"], "n_in_tokens": stats["n_in"],
                "n_out_tokens": stats["n_out"]}
    except Exception:
        out_midi.unlink(missing_ok=True)
        align.unlink(missing_ok=True)
        return None


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    mxls = [ln.strip() for ln in open(LIST) if ln.strip()]
    print(f"[classical] {len(mxls)} classical scores to tokenize", flush=True)
    ctx = get_context("spawn")
    t0 = time.time()
    ok = done = 0
    with open(MANIFEST, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        with cf.ProcessPoolExecutor(max_workers=24, mp_context=ctx) as ex:
            futs = {ex.submit(build_one, m, 1000 + i): m for i, m in enumerate(mxls)}
            for fut in cf.as_completed(futs):
                done += 1
                try:
                    row = fut.result()
                except Exception:
                    row = None
                if row:
                    w.writerow(row); ok += 1
                if done % 500 == 0:
                    f.flush()
                    print(f"  {done}/{len(mxls)} processed, {ok} ok, {time.time()-t0:.0f}s", flush=True)
    print(f"[classical] DONE: {ok}/{len(mxls)} tokenized -> {MANIFEST}, {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
