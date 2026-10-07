"""Build (perturbed-MIDI, engraved-MusicXML) training pairs from a Humdrum **kern
corpus (tuplet-rich classical: Scriabin, Chopin, ...).

This is the data-scaling de-risk: feed the model tuplet-rich engraved scores it
currently under-produces. Reuses the existing synthetic pipeline (expressive_render +
chunker + cache + manifest = PDMXDataset format) so make_mixed_loaders can train on it
jointly with real ASAP. The ONLY new step vs make_pairs.py is:
  (1) **kern -> MusicXML conversion via music21, and
  (2) the HAND-LABEL FIX: music21 names kern spines "spine_0/1", which the tokenizer
      maps to hand=2 for EVERY note (tokenizer.py:359-364 only matches staff1/staff2).
      We rename the two parts by average pitch (higher=Staff1=RH, lower=Staff2=LH) so
      parse_mxl assigns hands correctly. Verified: fixes {2:769} -> {0:389,1:380}.

Leakage: excludes any piece that is in the ASAP held-out test set (by name match).

Output mirrors make_pairs.py: pairs_kern/{id}.mid/.musicxml/_chunks.json + cache + manifest.
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
import warnings
from pathlib import Path

from joblib import Parallel, delayed

SCRIPT_DIR = Path(__file__).resolve().parent.parent
TOKENIZER_DIR = SCRIPT_DIR / "MIDI2ScoreTransformer" / "midi2scoretransformer"
SCRIPTS_DIR = SCRIPT_DIR / "scripts"
for p in (str(TOKENIZER_DIR), str(SCRIPTS_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from expressive_render import render  # noqa: E402
from make_pairs import build_chunks_from_alignment, parse_and_cache  # noqa: E402

log = logging.getLogger(__name__)

# ASAP held-out test pieces — exclude any kern file whose FULL PATH matches, to prevent
# train/test leakage. Path-aware (NOT filename-only) because "sonata12-1.krn" means K.332
# in the Mozart repo but a different sonata in Beethoven/Haydn. Patterns are path substrings,
# lowercased, '/'-normalized. Covers the 14-piece ASAP test split that appears in these corpora.
LEAKAGE_SUBSTRINGS = [
    "mysterium/op08/scriabin-op08_no11",      # Scriabin Etude Op.8/11
    "op08_no11",                               # (belt-and-suspenders for ccarh_kern copy)
    "mozart-piano-sonatas/kern/sonata12",      # K.332   (ASAP Mozart 12-1) — header-verified
    # ASAP Beethoven "10-1" = Op.14 (score-verified). No.1 vs No.2 ambiguous in the score,
    # so exclude BOTH Op.14 sonatas (sonata09=Op.14/1, sonata10=Op.14/2) to be leak-safe.
    "beethoven-piano-sonatas/kern/sonata09",
    "beethoven-piano-sonatas/kern/sonata10",
    "chopin-first-editions/kern/023-",         # Chopin Ballade 1 Op.23 (ASAP Chopin test)
    # Haydn XVI:31 absent from haydn-piano-sonatas (verified) -> no Haydn leak.
    # No Bach repo in this corpus -> no BWV846 leak.
]


def _avg_pitch(part) -> float:
    ps = []
    for n in part.recurse().notes:
        if n.isNote:
            ps.append(n.pitch.midi)
        elif n.isChord:
            ps.extend(p.midi for p in n.pitches)
    return sum(ps) / max(len(ps), 1)


def convert_kern_to_musicxml(krn_path: Path, out_path: Path) -> dict:
    """Parse **kern -> rename parts by pitch (hand fix) -> write MusicXML."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from music21 import converter
            s = converter.parse(str(krn_path))
            # music21's kern import leaves Dynamic/Dynamics objects without the
            # humdrumPosition attr the MusicXML writer expects -> AttributeError on
            # write. We don't use dynamics (not a tokenizer stream), so strip them.
            for cls in ("Dynamic", "DynamicWedge", "Crescendo", "Diminuendo"):
                try:
                    s.recurse().removeByClass(cls)
                except Exception:
                    pass
            parts = list(s.parts)
            if len(parts) >= 2:
                order = sorted(range(len(parts)), key=lambda i: -_avg_pitch(parts[i]))
                for rank, i in enumerate(order):
                    name = f"P{rank + 1}-Staff{rank + 1}"  # Staff1=RH(top), Staff2=LH
                    parts[i].id = name
                    parts[i].partName = name
                    try:
                        parts[i].partAbbreviation = name
                    except Exception:
                        pass
            out_path.parent.mkdir(parents=True, exist_ok=True)
            s.write("musicxml", str(out_path))
        return {"ok": True, "n_parts": len(parts)}
    except Exception as exc:
        return {"ok": False, "error": f"convert:{type(exc).__name__}: {exc}"}


def process_one(idx: int, krn: Path, dst_dir: Path, cache_dir: Path, seed: int) -> dict:
    import hashlib
    pid = "k" + hashlib.sha1(str(krn).encode()).hexdigest()[:12]  # path-derived: collision-free + idempotent
    out_mxl = dst_dir / f"{pid}.musicxml"
    out_midi = dst_dir / f"{pid}.mid"
    out_chunks = dst_dir / f"{pid}_chunks.json"
    if out_midi.exists() and out_chunks.exists():
        return {"ok": False, "error": "already_done", "id": pid, "src": str(krn)}

    conv = convert_kern_to_musicxml(krn, out_mxl)
    if not conv.get("ok"):
        return {"ok": False, "error": conv["error"], "id": pid, "src": str(krn)}

    result = render(out_mxl, out_midi, seed=seed)
    if not result.get("ok"):
        out_mxl.unlink(missing_ok=True)
        return {"ok": False, "error": f"render:{result.get('error')}", "id": pid, "src": str(krn)}

    try:
        chunks = build_chunks_from_alignment(out_midi.with_suffix(".alignment.json"))
        import json as _json
        out_chunks.write_text(_json.dumps(chunks))
    except Exception as exc:
        return {"ok": False, "error": f"chunks:{type(exc).__name__}: {exc}", "id": pid, "src": str(krn)}

    try:
        pkl_path, stats = parse_and_cache(out_mxl, out_midi, cache_dir)
    except Exception as exc:
        return {"ok": False, "error": f"cache:{type(exc).__name__}: {exc}", "id": pid, "src": str(krn)}

    return {"ok": True, "manifest_row": {
        "id": pid, "src_mxl": str(krn), "midi": str(out_midi), "mxl": str(out_mxl),
        "chunks": str(out_chunks), "cache": str(pkl_path),
        "n_notes": result["n_notes"], "n_measures": result["n_measures"],
        "n_in_tokens": stats["n_in"], "n_out_tokens": stats["n_out"],
    }}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kern-dirs", nargs="+", required=True,
                    help="Dirs to glob *.krn from (recursive).")
    ap.add_argument("--out-dir", default="data/pairs_kern")
    ap.add_argument("--cache-dir", default="data/cache_kern")
    ap.add_argument("--manifest", default="data/pairs_kern/_manifest.csv")
    ap.add_argument("--errors", default="data/pairs_kern/_errors.log")
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    krns = []
    for d in args.kern_dirs:
        krns.extend(sorted(Path(d).rglob("*.krn")))
    # leakage filter
    kept, leaked = [], []
    for k in krns:
        path_norm = str(k).replace("\\", "/").lower()
        if any(sub in path_norm for sub in LEAKAGE_SUBSTRINGS):
            leaked.append(k)
        else:
            kept.append(k)
    log.info("kern files: %d total, %d kept, %d EXCLUDED for leakage: %s",
             len(krns), len(kept), len(leaked), [k.name for k in leaked])

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir); cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.manifest); manifest_path.parent.mkdir(parents=True, exist_ok=True)
    error_log_path = Path(args.errors)

    fields = ["id", "src_mxl", "midi", "mxl", "chunks", "cache",
              "n_notes", "n_measures", "n_in_tokens", "n_out_tokens"]
    write_header = not manifest_path.exists()

    t0 = time.time()
    results = Parallel(n_jobs=args.n_jobs, verbose=10, batch_size=2)(
        delayed(process_one)(i, k, out_dir, cache_dir, args.seed + i)
        for i, k in enumerate(kept)
    )

    ok = skipped = 0
    tuplet_notes = total_notes = 0
    with manifest_path.open("a", newline="") as mf, error_log_path.open("a") as ef:
        writer = csv.DictWriter(mf, fieldnames=fields)
        if write_header:
            writer.writeheader()
        for r in results:
            if r.get("ok"):
                writer.writerow(r["manifest_row"]); ok += 1
            else:
                ef.write(f"{r.get('id','?')}\t{r.get('error','')}\t{r.get('src','?')}\n"); skipped += 1
    log.info("Done. ok=%d skipped=%d in %.1fs", ok, skipped, time.time() - t0)
    log.info("Manifest: %s", manifest_path)


if __name__ == "__main__":
    main()
