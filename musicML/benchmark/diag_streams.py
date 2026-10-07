"""Per-stream MUSTER/score_similarity breakdown for selected pieces + a checkpoint.

The pad-sweep only kept MeanER; this dumps the full per-stream error decomposition
(TimeSignature, NoteDeletion/Insertion, NoteSpelling, NoteDuration, Tie, Beams, ...)
so we can separate a tuplet/note-value collapse (-> NoteDuration) from a structural
failure (-> TimeSignature / NoteDeletion). Run the same pieces on ours vs released.

Usage:
  venv311/bin/python benchmark/diag_streams.py --ckpt <ckpt> --pieces Schumann Haydn Rachmaninoff
"""
import argparse
import json
import os
import sys
from pathlib import Path

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

sys.path.insert(0, str(REPO_ROOT / "benchmark"))
from eval_tier1_asap import collect_paths, load_any_checkpoint  # noqa: E402
import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
from musicml_paths import device as _device

STREAMS = ["TimeSignature", "KeySignature", "Clef", "NoteDeletion", "NoteInsertion",
           "NoteSpelling", "NoteDuration", "Tie", "Beams", "StemDirection",
           "StaffAssignment", "Voice"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--device", default=_device())
    ap.add_argument("--pieces", nargs="+", required=True)
    ap.add_argument("--pad-threshold", type=float, default=0.5)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    paths = collect_paths("test")
    sel = []
    seen = set()
    for p in paths:
        name = (p["composer"] + "/" + p["piece"]).lower()
        if any(s.lower() in name for s in args.pieces) and (p["composer"], p["piece"]) not in seen:
            seen.add((p["composer"], p["piece"]))
            sel.append(p)

    model = load_any_checkpoint(args.ckpt, args.device)
    model.eval(); model.to(args.device)

    print(f"\n### {args.tag or args.ckpt}")
    print(f"{'piece':22s}{'MeanER':>8}" + "".join(s[:8].rjust(9) for s in STREAMS))
    for p in sel:
        try:
            x = MultistreamTokenizer.tokenize_midi(p["midi"])
            y = infer(x, model, overlap=64, chunk=512, verbose=False, kv_cache=True)
            sim = eval_pair(y, p["score"], pad_threshold=args.pad_threshold)
            mer = (sim.get("muster") or {}).get("MeanER")
            ss = sim.get("mxl <-> gt_mxl") or {}
            cells = ""
            for s in STREAMS:
                v = ss.get(s)
                cells += (f"{v:.3f}".rjust(9) if isinstance(v, (int, float)) else "—".rjust(9))
            mers = f"{mer:.2f}" if isinstance(mer, (int, float)) else "—"
            print(f"{p['composer'][:10]+'/'+p['piece'][:10]:22s}{mers:>8}{cells}")
        except Exception as e:
            print(f"{p['composer'][:20]:22s}  ERROR {repr(e)[:80]}")


if __name__ == "__main__":
    main()
