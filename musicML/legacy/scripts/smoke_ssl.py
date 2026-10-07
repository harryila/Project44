import warnings, sys
warnings.simplefilter("ignore")
sys.path.insert(0, "MIDI2ScoreTransformer/midi2scoretransformer")
import torch
import train

tl, vl = train.make_ssl_loaders(
    "data/pairs_unpaired_ssl_manifest.csv",
    seq_length=512, batch_size=8, num_workers=0, real_fraction=0.5,
    data_dir="MIDI2ScoreTransformer/data/")
m = train._new_model(autoregressive=True)
m._ssl_mode = True
m.train()
it = iter(tl)
for b in range(3):
    inp, out = next(it)
    uc = inp.get("unconditional")
    has = "unconditional" in inp
    shape = tuple(uc.shape) if uc is not None else None
    uniq = sorted(set(uc[:, 0, 0].tolist())) if uc is not None else None
    print("batch", b, "has_uc", has, "uc_shape", shape, "uc_vals", uniq,
          "pitch", tuple(inp["pitch"].shape), "out_pitch", tuple(out["pitch"].shape))
    loss = m.training_step((inp, out), b)
    print("   loss", round(float(loss), 4), "finite", bool(torch.isfinite(loss)))
print("SMOKE OK")
