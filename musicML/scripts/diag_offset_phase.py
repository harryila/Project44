"""Decisive diagnostic for the placement levers: does the model EVER place notes at triplet
metrical phases, or does it quantize everything to binary phases?

If the predicted OFFSET stream has ~no mass at triplet phases {8,16} (while GT does), then the
metrical-prior (A2) / grammar (A3) inference levers CANNOT help (they only reshape the DURATION
choice at a given offset), and the structural fix (beat-relative OUTPUT retokenization, B2) is
required. If the model DOES place notes at triplet phases, positional levers have headroom.

Compares model prediction vs ground-truth score on a dense piece (Scriabin). CPU (no GPU needed).

Run (box):  PYTHONPATH=MIDI2ScoreTransformer/midi2scoretransformer \
  venv311/bin/python scripts/diag_offset_phase.py --ckpt <ckpt> --piece Scriabin
"""
import argparse, os, sys, warnings
from pathlib import Path
warnings.simplefilter("ignore")
REPO = Path(__file__).resolve().parent.parent
TF = REPO / "MIDI2ScoreTransformer"
sys.path.insert(0, str(TF / "midi2scoretransformer")); sys.path.insert(0, str(REPO / "benchmark"))
os.chdir(TF)
import torch
from config import MyModelConfig
if not hasattr(MyModelConfig, "_attn_implementation_internal"):
    MyModelConfig._attn_implementation_internal = None
torch.serialization.add_safe_globals([MyModelConfig])
from tokenizer import MultistreamTokenizer
from utils import infer
from eval_tier1_asap import collect_paths, load_any_checkpoint

N_PHASE = 24
TRIPLET = {8, 16, 4, 20}  # non-multiple-of-3 sub-quarter phases (triplet/sextuplet)


def phase_hist(off_idx):
    import numpy as np
    ph = (off_idx % N_PHASE)
    tot = len(ph)
    tup = int(sum(1 for p in ph if int(p) % 3 != 0))
    return tot, tup, (tup / tot if tot else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--piece", default="Scriabin")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    paths = [p for p in collect_paths("test")
             if args.piece.lower() in (p["composer"] + "/" + p["piece"]).lower()]
    paths = sorted(paths, key=lambda p: 0)[:1]
    assert paths, f"no test piece matching {args.piece}"
    p = paths[0]
    print(f"piece: {p['composer']}/{p['piece']}")
    x = MultistreamTokenizer.tokenize_midi(p["midi"])
    model = load_any_checkpoint(args.ckpt, args.device); model.eval(); model.to(args.device)
    with torch.no_grad():
        y = infer(x, model, overlap=64, chunk=512, verbose=False, kv_cache=True)
    pred_off = y["offset"].argmax(-1).reshape(-1).cpu().numpy()
    # ground-truth score offsets
    gt = MultistreamTokenizer.tokenize_mxl(p["score"])
    gt_off = gt["offset"].argmax(-1).reshape(-1).numpy()
    pt, ptup, pfrac = phase_hist(pred_off)
    gt_t, gtup, gfrac = phase_hist(gt_off)
    print(f"PRED offsets: {pt} notes, {ptup} at triplet phases ({pfrac:.3%})")
    print(f"GT   offsets: {gt_t} notes, {gtup} at triplet phases ({gfrac:.3%})")
    import numpy as np
    print("PRED phase histogram (phase:count) for triplet phases 4,8,16,20:")
    for ph in (4, 8, 16, 20):
        print(f"  phase {ph}: pred={int((pred_off % N_PHASE == ph).sum())}  gt={int((gt_off % N_PHASE == ph).sum())}")
    verdict = ("POSITIONAL LEVERS VIABLE — model places notes at triplet phases" if pfrac > 0.3 * gfrac and ptup > 5
               else "POSITIONAL INFERENCE LEVERS DOOMED — model quantizes offsets to binary; needs beat-relative OUTPUT (B2)")
    print("VERDICT:", verdict)


if __name__ == "__main__":
    main()
