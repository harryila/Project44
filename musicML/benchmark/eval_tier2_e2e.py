"""Tier-2 accuracy eval: END-TO-END audio -> score, scored with MUSTER +
score_similarity vs ground-truth MusicXML.

This measures the REAL product quality of the full production pipeline
(audio -> hFT -> MIDI2ScoreTransformer -> MusicXML), as opposed to Tier-1 which
isolates the MIDI->score model. It is necessarily worse than Tier-1 / the paper's
11.30 because Block-1 transcription error stacks on top (see ACCURACY_ROADMAP.md
section 2). Use it for the lab "before/after" on the cheap inference fixes.

It shells out to the real production CLI (transcribe.py -t hft -b transformer) so
it measures exactly what users get, then scores the saved intermediate MusicXML.

Default pieces (the 3 benchmark demos):
  - Chopin Op.10 No.4   (GT in ASAP — IN-DISTRIBUTION for the released model)
  - Chopin Op.25 No.11  (GT in ASAP — IN-DISTRIBUTION)
  - Liszt Mazeppa       (GT sourced from PDMX — OUT-OF-DISTRIBUTION, the honest test)

Run AFTER the Tier-1 baseline finishes to avoid CPU contention.

Usage:
    venv311/bin/python benchmark/eval_tier2_e2e.py --out benchmark/tier2_baseline.json
    # extra args after -- are forwarded to transcribe.py (e.g. inference fixes):
    ...  --tag fixed -- --normalize-audio
"""
import argparse
import json
import os
import subprocess
import sys
import time
import warnings
from pathlib import Path

warnings.simplefilter("ignore")

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
from musicml_paths import REPO_ROOT as _ROOT
REPO = Path(_ROOT)
TF = REPO / "MIDI2ScoreTransformer" / "midi2scoretransformer"
sys.path.insert(0, str(TF))

from muster import muster  # noqa: E402
from utils import score_similarity_normalized  # noqa: E402
import music21  # noqa: E402

ASAP = REPO / "MIDI2ScoreTransformer/data/asap-dataset"

# name, audio, GT score, distribution flag
PIECES = [
    {"name": "Chopin_Op10_No4", "dist": "in-dist (ASAP train)",
     "audio": REPO / "benchmark/chopin_op10/audio/Op10_No4_CsharpMinor.wav",
     "gt": ASAP / "Chopin/Etudes_op_10/4/xml_score.musicxml"},
    {"name": "Chopin_Op25_No11", "dist": "in-dist (ASAP train)",
     "audio": REPO / "benchmark/chopin_op25/audio/Op25_No11_Aminor.wav",
     "gt": ASAP / "Chopin/Etudes_op_25/11/xml_score.musicxml"},
    {"name": "Liszt_Mazeppa", "dist": "dense (in ASAP train)",
     "audio": REPO / "benchmark/liszt_transcendental/audio/Transcendental_No4_Mazeppa.wav",
     # ASAP edition (model trained on it) — the PDMX community score inflated MUSTER
     # ~2x via edition mismatch (34 vs ~14), not real model error. See DECOMPOSED_FINDINGS.md.
     "gt": ASAP / "Liszt/Transcendental_Etudes/4/xml_score.musicxml"},
]

MUSTER_KEYS = ["PitchER", "MissRate", "ExtraRate", "OnsetER", "OffsetER", "MeanER"]
NOTATION_KEYS = ["NoteDeletion", "NoteInsertion", "NoteDuration",
                 "StaffAssignment", "StemDirection", "NoteSpelling"]


def transcribe_to_mxl(audio: Path, out_pdf: Path, extra_args):
    """Run the real production pipeline; return the saved intermediate .musicxml."""
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    cmd = [str(REPO / "venv311/bin/python"), str(REPO / "transcribe.py"),
           str(audio), "-t", "hft", "-b", "transformer", "-o", str(out_pdf)] + list(extra_args)
    subprocess.run(cmd, cwd=str(REPO), check=False,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    mxl = out_pdf.with_suffix(".musicxml")
    return mxl if mxl.exists() else None


def score(est_mxl: Path, gt_mxl: Path):
    est = music21.converter.parse(str(est_mxl))
    return {
        "muster": muster(str(est_mxl), str(gt_mxl)),
        "mxl <-> gt_mxl": score_similarity_normalized(est, str(gt_mxl), full=False),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="baseline", help="label for the transcribe output dir")
    ap.add_argument("--skip-transcribe", action="store_true",
                    help="score pre-existing <tag> MusicXML without re-running the pipeline")
    ap.add_argument("transcribe_args", nargs="*",
                    help="args after -- forwarded to transcribe.py")
    args = ap.parse_args()

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = REPO / out_path
    work = REPO / "benchmark" / "tier2_out" / args.tag

    results = []
    t0 = time.time()
    for p in PIECES:
        rec = {"name": p["name"], "dist": p["dist"],
               "audio": str(p["audio"]), "gt": str(p["gt"])}
        if not p["gt"].exists():
            rec["error"] = f"GT score missing: {p['gt']}"
            print(f"  SKIP {p['name']}: {rec['error']}", flush=True)
            results.append(rec); continue
        if not p["audio"].exists():
            rec["error"] = f"audio missing: {p['audio']}"
            print(f"  SKIP {p['name']}: {rec['error']}", flush=True)
            results.append(rec); continue

        est = work / f"{p['name']}.musicxml"
        if not args.skip_transcribe:
            ti = time.time()
            print(f"  transcribing {p['name']} ...", flush=True)
            est = transcribe_to_mxl(p["audio"], work / f"{p['name']}.pdf", args.transcribe_args)
            rec["transcribe_s"] = round(time.time() - ti, 1)
        if est is None or not Path(est).exists():
            rec["error"] = "pipeline produced no MusicXML"
            print(f"  FAIL {p['name']}: {rec['error']}", flush=True)
            results.append(rec); continue
        try:
            rec["sim"] = score(Path(est), p["gt"])
            mer = (rec["sim"].get("muster") or {}).get("MeanER")
            print(f"  {p['name']:18s} [{p['dist']}] MeanER={mer if mer is None else round(mer,2)}", flush=True)
        except Exception as e:
            rec["error"] = f"score: {e}"
            print(f"  FAIL {p['name']}: {rec['error']}", flush=True)
        results.append(rec)
        json.dump({"meta": {"tag": args.tag, "elapsed_s": round(time.time()-t0, 1)},
                   "per_piece": results}, open(out_path, "w"), indent=2)

    print("\n" + "=" * 72)
    print(f"TIER-2 (end-to-end audio->score), tag={args.tag}")
    print("=" * 72)
    for r in results:
        if "sim" not in r:
            print(f"  {r['name']:18s} -> {r.get('error','?')}"); continue
        m = r["sim"].get("muster") or {}
        print(f"  {r['name']:18s} [{r['dist']:22s}] " +
              "  ".join(f"{k}={(m.get(k) or float('nan')):.1f}" for k in ["MeanER", "OnsetER", "PitchER"]))
    json.dump({"meta": {"tag": args.tag, "elapsed_s": round(time.time()-t0, 1)},
               "per_piece": results}, open(out_path, "w"), indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
