"""Validate the pad_prob fix on the REAL released SOTA checkpoint (CPU, no box) and measure
its actual keep-gate calibration curve.

We can't compute MUSTER locally (ASAP test data lives on the box), but we CAN answer the
question that decides whether the now-live --pad-threshold has any headroom: on a real trained
model, is there real note mass in the 0.30-0.50 "rescue zone", or is the keep-gate saturated
near 1.0 (=> sweep would be flat)? Also re-confirms the 0.5-invariant on a real model.

Run:  venv311/bin/python scripts/probe_released_padprob.py [midi_path]
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TF = REPO / "MIDI2ScoreTransformer" / "midi2scoretransformer"
sys.path.insert(0, str(TF))

import torch
from config import MyModelConfig
if not hasattr(MyModelConfig, "_attn_implementation_internal"):
    MyModelConfig._attn_implementation_internal = None
torch.serialization.add_safe_globals([MyModelConfig])

from tokenizer import MultistreamTokenizer
from utils import infer
from models.roformer import Roformer

CKPT = REPO / "MIDI2ScoreTransformer" / "checkpoints" / "MIDI2ScoreTF.ckpt"  # released SOTA
MIDI = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "midi" / "Bach, Prelude in C major, BWV 846 .mid"


def n_notes(d, thr):
    return int((d["pad_prob"].squeeze() > thr).sum())


def main():
    print(f"ckpt: {CKPT.name}\nmidi: {MIDI.name}")
    model = Roformer.load_from_checkpoint(str(CKPT), map_location="cpu", weights_only=False)
    model.eval(); model.to("cpu")

    x = MultistreamTokenizer.tokenize_midi(str(MIDI))
    print(f"input notes: {x['pitch'].shape[0]}  (CPU inference, please wait...)")
    with torch.no_grad():
        y = infer(x, model, overlap=64, chunk=512, verbose=False, kv_cache=True)

    assert "pad_prob" in y, "pad_prob missing — fix not active"
    pp = y["pad_prob"].squeeze().float()
    T = pp.numel()

    # 0.5-invariant on the REAL model: soft mask == legacy binary argmax mask
    soft = (pp > 0.5)
    hard = (y["pad"].squeeze() > 0.5)
    inv = bool(torch.equal(soft, hard))
    print(f"\n0.5-invariant (soft==legacy binary): {inv}  (kept@0.5 = {int(hard.sum())}/{T})")

    # keep-probability distribution
    import numpy as np
    a = pp.numpy()
    qs = np.percentile(a, [0, 5, 10, 25, 50, 75, 90, 95, 100])
    print("\nkeep-prob percentiles  p0/p5/p10/p25/p50/p75/p90/p95/p100:")
    print("  " + "  ".join(f"{q:.3f}" for q in qs))
    bins = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.01]
    hist, _ = np.histogram(a, bins=bins)
    print("\nhistogram (keep-prob bucket -> #notes):")
    for i in range(len(bins) - 1):
        bar = "#" * int(40 * hist[i] / max(hist.max(), 1))
        print(f"  [{bins[i]:.1f},{bins[i+1]:.1f}) {hist[i]:5d} {bar}")

    rescue = int(((a >= 0.30) & (a < 0.50)).sum())   # would be ADDED by lowering to 0.30
    prune  = int(((a > 0.50) & (a <= 0.70)).sum())   # would be REMOVED by raising to 0.70
    print(f"\nRESCUE ZONE  keep-prob in [0.30,0.50): {rescue} notes  (lowering the gate adds these)")
    print(f"PRUNE  ZONE  keep-prob in (0.50,0.70]: {prune} notes  (raising the gate removes these)")

    print("\nnote count vs threshold:")
    for thr in (0.60, 0.55, 0.50, 0.45, 0.40, 0.35, 0.30):
        print(f"  thr={thr:.2f}: {n_notes(y, thr)} notes")

    verdict = "LIVE — the keep-gate has headroom" if (rescue + prune) > 0.02 * T else \
              "near-saturated — sweep likely flat on this piece"
    print(f"\nVERDICT: {verdict}  (rescue+prune = {rescue+prune} of {T} = {100*(rescue+prune)/T:.1f}%)")


if __name__ == "__main__":
    main()
