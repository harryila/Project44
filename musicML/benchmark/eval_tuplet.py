"""De-risk metric for the data-scaling experiment: does adding tuplet-rich engraved
scores (Humdrum Scriabin/Chopin) make the model PRODUCE more tuplets on held-out hard
pieces? Reports, per piece + per checkpoint:
  - MUSTER MeanER (accuracy)
  - n_pred_tuplets / n_gt_tuplets  (the tuplet UNDER-PRODUCTION ratio; baseline ~0.5x)

Run once per checkpoint, then diff. Pieces are matched by substring against the ASAP
test split. Scriabin Op.8/11 (EXCLUDED from kern training) is the key same-style held-out.

Usage:
  venv311/bin/python benchmark/eval_tuplet.py --ckpt <ckpt> --device cuda \
     --pieces Scriabin Ravel Prokofiev Liszt Mozart --out benchmark/tuplet_<tag>.json
"""
import argparse
import json
import os
import signal
import sys
import warnings
from pathlib import Path


class _Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise _Timeout()


signal.signal(signal.SIGALRM, _alarm)  # per-piece guard: MUSTER/music21 hang on rough outputs

warnings.simplefilter("ignore")
REPO_ROOT = Path(__file__).resolve().parent.parent
TF_ROOT = REPO_ROOT / "MIDI2ScoreTransformer"
sys.path.insert(0, str(TF_ROOT / "midi2scoretransformer"))
os.chdir(TF_ROOT)

import torch  # noqa: E402
from config import MyModelConfig  # noqa: E402
if not hasattr(MyModelConfig, "_attn_implementation_internal"):
    MyModelConfig._attn_implementation_internal = None
torch.serialization.add_safe_globals([MyModelConfig])

from tokenizer import MultistreamTokenizer  # noqa: E402
from utils import infer, eval as eval_pair  # noqa: E402
from score_utils import postprocess_score  # noqa: E402

# reuse the test-split path collection + checkpoint loader from the tier-1 harness
sys.path.insert(0, str(REPO_ROOT / "benchmark"))
from eval_tier1_asap import collect_paths, load_any_checkpoint  # noqa: E402
import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
from musicml_paths import device as _device


def count_tuplets(score) -> tuple[int, int]:
    """(n_tuplet_notes, n_notes) in a music21 score."""
    if score is None:
        return 0, 0
    nt = nn = 0
    for n in score.recurse().notes:
        nn += 1
        try:
            if n.duration.tuplets:
                nt += 1
        except Exception:
            pass
    return nt, nn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--device", default=_device())
    ap.add_argument("--pieces", nargs="*", default=None,
                    help="substring filters on composer/piece; default = all test pieces")
    ap.add_argument("--limit-per", type=int, default=1,
                    help="max performances per matched piece (default 1, the shortest)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--overlap", type=int, default=64)
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--pad-threshold", type=float, default=0.5,
                    help="soft keep-prob gate; <0.5 rescues dropped notes, >0.5 prunes (live since the pad_prob fix)")
    ap.add_argument("--pitch-top-k", type=int, default=None,
                    help="per-head decoding: top_k for the PITCH head only (others stay greedy)")
    ap.add_argument("--pitch-temp", type=float, default=None,
                    help="per-head decoding: temperature for the PITCH head only")
    args = ap.parse_args()

    # Per-head decoding override (None unless a pitch knob is set -> identical to greedy).
    head_overrides = None
    if args.pitch_top_k is not None or args.pitch_temp is not None:
        head_overrides = {"pitch": (args.pitch_top_k if args.pitch_top_k is not None else 1,
                                    args.pitch_temp if args.pitch_temp is not None else 1.0)}

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path

    paths = collect_paths("test")
    # tokenize + length for sorting
    for p in paths:
        try:
            p["_x"] = MultistreamTokenizer.tokenize_midi(p["midi"])
            p["n_notes"] = int(p["_x"]["pitch"].shape[0])
        except Exception as e:
            p["_x"] = None
            p["n_notes"] = 1 << 30
    if args.pieces:
        paths = [p for p in paths
                 if any(s.lower() in (p["composer"] + "/" + p["piece"]).lower() for s in args.pieces)]
    # keep the shortest --limit-per performances per (composer,piece)
    paths.sort(key=lambda p: p["n_notes"])
    seen = {}
    kept = []
    for p in paths:
        key = (p["composer"], p["piece"])
        if seen.get(key, 0) < args.limit_per and p["_x"] is not None:
            seen[key] = seen.get(key, 0) + 1
            kept.append(p)
    paths = kept

    model = load_any_checkpoint(args.ckpt, args.device)
    model.eval(); model.to(args.device)

    results = []
    for i, p in enumerate(paths):
        rec = {k: p[k] for k in ("composer", "piece", "n_notes")}
        try:
            y_hat = infer(p["_x"], model, overlap=args.overlap, chunk=args.chunk,
                          verbose=False, kv_cache=True, head_overrides=head_overrides)
            # Tuplet counts (cheap) first, then MUSTER under a timeout (it can hang on
            # malformed scores from rough models). On timeout, keep the tuplet counts.
            signal.alarm(120)
            pred = postprocess_score(
                MultistreamTokenizer.detokenize_mxl(y_hat, pad_threshold=args.pad_threshold),
                inPlace=True)
            from music21 import converter
            gt = converter.parse(p["score"])
            ptup, pn = count_tuplets(pred)
            gtup, gn = count_tuplets(gt)
            signal.alarm(0)
            rec.update({"pred_tuplets": ptup, "pred_notes": pn, "gt_tuplets": gtup,
                        "gt_notes": gn, "tuplet_ratio": (ptup / gtup) if gtup else None})
            mer = None
            try:
                signal.alarm(180)
                sim = eval_pair(y_hat, p["score"], pad_threshold=args.pad_threshold)
                mer = (sim.get("muster") or {}).get("MeanER")
                signal.alarm(0)
            except _Timeout:
                signal.alarm(0)
                rec["muster_timeout"] = True
            rec["MeanER"] = mer
            print(f"[{i+1}/{len(paths)}] {p['composer']}/{p['piece'][:28]:28s} "
                  f"MeanER={None if mer is None else round(mer,2)} "
                  f"pred_tup={ptup} gt_tup={gtup} ratio="
                  f"{None if not gtup else round(ptup/gtup,2)}", flush=True)
        except _Timeout:
            signal.alarm(0)
            rec["error"] = "timeout"
            print(f"[{i+1}/{len(paths)}] {p['composer']}/{p['piece'][:28]} TIMEOUT", flush=True)
        except Exception:
            signal.alarm(0)
            import traceback
            rec["error"] = traceback.format_exc().splitlines()[-1]
            print(f"[{i+1}/{len(paths)}] {p['composer']}/{p['piece'][:28]} ERROR {rec['error']}", flush=True)
        results.append(rec)
        json.dump({"ckpt": args.ckpt, "results": results}, open(out_path, "w"), indent=2)

    # summary
    scored = [r for r in results if r.get("tuplet_ratio") is not None]
    if scored:
        mr = sum(r["tuplet_ratio"] for r in scored) / len(scored)
        me = [r["MeanER"] for r in scored if isinstance(r.get("MeanER"), (int, float))]
        print(f"\nMEAN tuplet_ratio={mr:.3f}  MEAN MeanER={sum(me)/len(me):.2f}  (n={len(scored)})")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
