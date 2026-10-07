"""Local validation for the tuplet-rate RESHAPE (no GPU, no caches).

Builds a tiny synthetic manifest with a tuplet_rate column and checks PDMXDataset's resample
weights:
  (A) gamma=0 -> _resample_w == lengths (byte-identical to the original length-only resampling).
  (B) gamma>0 -> _resample_w == lengths * (tuplet_rate+floor)^gamma, so tuplet-rich scores are
      upweighted; verify the effective (sampling-weighted) mean tuplet_rate rises vs uniform.

Run:  venv311/bin/python scripts/test_tuplet_reshape.py
"""
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TF = REPO / "MIDI2ScoreTransformer" / "midi2scoretransformer"
sys.path.insert(0, str(TF))

import numpy as np
import pandas as pd
import torch
from pdmx_dataset import PDMXDataset

# Synthetic manifest: 1000 scores, 86% tuplet-free (mirrors the real corpus), rest 0.1-0.9.
rng = np.random.default_rng(0)
n = 1000
tr = np.where(rng.random(n) < 0.86, 0.0, rng.uniform(0.1, 0.9, n))
df = pd.DataFrame({
    "id": [f"s{i}" for i in range(n)],
    "midi": ["x.mid"] * n, "chunks": ["x.json"] * n, "cache": ["x.pkl"] * n,
    "n_in_tokens": rng.integers(50, 500, n).astype(float),
    "tuplet_rate": tr,
})
tmp = Path(tempfile.mkdtemp()) / "m.csv"
df.to_csv(tmp, index=False)


def eff_mean_tuplet(ds):
    w = ds._resample_w.numpy()
    w = w / w.sum()
    tr_all = ds.metadata["tuplet_rate"].values
    return float((w * tr_all).sum())


d0 = PDMXDataset(str(tmp), split="all", tuplet_gamma=0.0)
assert torch.allclose(d0._resample_w, d0.lengths), "gamma=0 must equal length-only weights"
print(f"(A) gamma=0: _resample_w == lengths  (uniform-by-length, eff tuplet_rate={eff_mean_tuplet(d0):.4f})")

for g in (1.0, 2.0):
    d = PDMXDataset(str(tmp), split="all", tuplet_gamma=g, tuplet_floor=0.05)
    tr_t = torch.FloatTensor(d.metadata["tuplet_rate"].values.astype(float))
    expect = d.lengths * torch.pow(tr_t + 0.05, g)
    assert torch.allclose(d._resample_w, expect, atol=1e-5), f"gamma={g} weight mismatch"
    eff = eff_mean_tuplet(d)
    print(f"(B) gamma={g}: eff sampling-weighted tuplet_rate = {eff:.4f}  (vs {eff_mean_tuplet(d0):.4f} uniform)")
    assert eff > eff_mean_tuplet(d0), "reshape must raise effective tuplet exposure"

print("\nPASS: gamma=0 identical; gamma>0 upweights tuplet-rich scores (exposure rises).")
