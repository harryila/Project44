"""Local validation for tuplet-aware loss reweighting (no GPU, no data).

Checks:
  (A) tuplet_weight=1.0 -> _stream_class_weights returns None, and _compute_loss is byte-identical
      to a manual standard cross-entropy (zero regression risk when off).
  (B) the per-class weight vector is CORRECT: on the 1/24 grid, dyadic buckets (multiples of 3 for
      duration/offset; i-1 multiple of 3 for downbeat) get weight 1.0; tuplet buckets get the multiplier.
      Spot-check musical values: quarter=24, eighth=12, 16th=6 -> dyadic; triplet-8th=8, triplet-4=16,
      triplet-16th=4 -> tuplet.
  (C) tuplet_weight=5.0 changes ONLY the rhythm-stream losses (duration/offset/downbeat) and leaves
      pitch/pad/etc untouched; no NaN.

Run:  venv311/bin/python scripts/test_tuplet_loss.py
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TF = REPO / "MIDI2ScoreTransformer" / "midi2scoretransformer"
sys.path.insert(0, str(TF))

import torch
import torch.nn.functional as F
from config import MyModelConfig, FEATURES
if not hasattr(MyModelConfig, "_attn_implementation_internal"):
    MyModelConfig._attn_implementation_internal = None
from train import _new_model

torch.manual_seed(0)


def synth_batch(B=2, T=12):
    """Build synthetic (pred, target) dicts with one-hot targets + random logits per stream."""
    pred, target = {}, {}
    real = torch.ones(B, T)
    real[:, T - 3:] = 0  # last 3 positions are padding
    target["pad"] = real
    pred["pad"] = torch.randn(B, T, 1)
    for s, conf in FEATURES.items():
        if s == "pad":
            continue
        V = conf["vocab_size"]
        idx = torch.randint(0, V, (B, T))
        if s == "duration":          # force a tuplet target (bucket 8 = triplet-eighth) to be present
            idx[0, 0] = 8
            idx[0, 1] = 24           # quarter (dyadic)
        target[s] = F.one_hot(idx, num_classes=V).float()
        pred[s] = torch.randn(B, T, V)
    return pred, target


def main():
    model = _new_model(learning_rate=3e-4)
    model.eval()
    pred, target = synth_batch()

    # ---- (B) weight-vector correctness ----
    model._tuplet_weight = 5.0
    model._tuplet_weight_cache = {}
    wd = model._stream_class_weights("duration", FEATURES["duration"]["vocab_size"], "cpu")
    wo = model._stream_class_weights("offset", FEATURES["offset"]["vocab_size"], "cpu")
    wdb = model._stream_class_weights("downbeat", FEATURES["downbeat"]["vocab_size"], "cpu")
    wp = model._stream_class_weights("pitch", FEATURES["pitch"]["vocab_size"], "cpu")
    assert wp is None, "pitch is not a rhythm stream -> must be unweighted"
    for name, val, w in [("quarter", 24, wd), ("eighth", 12, wd), ("16th", 6, wd),
                         ("trip-8th", 8, wd), ("trip-4", 16, wd), ("trip-16th", 4, wd)]:
        got = float(w[val])
        exp = 1.0 if val % 3 == 0 else 5.0
        assert got == exp, f"duration[{name}={val}] weight {got} != {exp}"
    # downbeat: bucket i -> value*24 = i-1, so dyadic <=> (i-1)%3==0
    assert float(wdb[25]) == 1.0, "downbeat bucket 25 (val24=24, quarter) should be dyadic (1.0)"
    assert float(wdb[9]) == 5.0, "downbeat bucket 9 (val24=8, triplet) should be tuplet (5.0)"
    print("(B) weight vectors correct: dyadic=mult-of-3 -> 1.0, tuplet -> 5.0 "
          f"(duration tuplet buckets: {int((wd == 5.0).sum())}/{wd.numel()})")

    # ---- (A) off == standard CE ----
    model._tuplet_weight = 1.0
    model._tuplet_weight_cache = {}
    assert model._stream_class_weights("duration", 97, "cpu") is None
    total_off, parts_off = model._compute_loss(pred, target)
    # manual standard CE for the duration stream (boolean-mask branch; ignore_index=-100<0)
    V = FEATURES["duration"]["vocab_size"]
    logits = pred["duration"].reshape(-1, V)
    tgt = target["duration"].argmax(-1).reshape(-1)
    m = target["pad"].reshape(-1) > 0
    manual = F.cross_entropy(logits[m], tgt[m])
    assert torch.allclose(parts_off["duration"], manual, atol=1e-6), \
        f"weight=1.0 not identical to standard CE: {parts_off['duration']} vs {manual}"
    print(f"(A) weight=1.0 == standard CE on duration ({float(manual):.5f}); total={float(total_off):.4f}")

    # ---- (C) weight=5.0 changes only rhythm streams, no NaN ----
    model._tuplet_weight = 5.0
    model._tuplet_weight_cache = {}
    total_on, parts_on = model._compute_loss(pred, target)
    assert not torch.isnan(total_on), "NaN in total loss"
    changed = {k for k in parts_on if not torch.allclose(parts_on[k], parts_off[k], atol=1e-6)}
    print(f"(C) streams changed by tuplet_weight=5.0: {sorted(changed)}")
    assert changed <= {"duration", "offset", "downbeat"}, f"unexpected streams changed: {changed}"
    assert "duration" in changed, "duration loss should change (batch has a tuplet target)"
    print("\nPASS: tuplet-aware loss is off-identical at 1.0, correct grid, rhythm-only when on.")


if __name__ == "__main__":
    main()
