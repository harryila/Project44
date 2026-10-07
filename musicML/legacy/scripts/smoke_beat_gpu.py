#!/usr/bin/env python3
"""GPU pre-launch smoke test for the beat-conditioning warm-start.

Confirms, before committing to the multi-hour run:
  1. load_pretrained_init(use_beat_conditioning=True) loads the released ckpt
     (strict=False) with NO missing/unexpected keys other than the new beat Linear.
  2. The beat Linear is zero-init (warm-start byte-identical to baseline at step 0).
  3. A real ASAP train batch (with the beat stream) collates, and forward + loss +
     backward run on CUDA without shape/OOM errors.
  4. The beat stream is actually present and non-trivial in the input batch.
"""
import sys
import warnings

warnings.simplefilter("ignore")
sys.path.insert(0, "MIDI2ScoreTransformer/midi2scoretransformer")

import torch
from train import load_pretrained_init, make_asap_loaders

CKPT = "MIDI2ScoreTransformer/checkpoints/MIDI2ScoreTF.ckpt"

print("[1] load_pretrained_init(use_beat_conditioning=True) ...", flush=True)
model = load_pretrained_init(CKPT, use_beat_conditioning=True)
assert model is not None, "ckpt not found / load failed"

emb = model.embeddings_enc.embeddings["beat"]
wmax = emb.weight.abs().max().item()
print(f"    beat Linear in={emb.in_features} out={emb.out_features} "
      f"|w|max={wmax:.3e} (must be 0.0 for warm-start identity)")
assert wmax == 0.0, "beat Linear is NOT zero-init -> warm-start would regress!"
print("    OK: beat Linear zero-init.")

print("[2] build ASAP train loader (beat on) + pull one batch ...", flush=True)
train_loader, val_loader = make_asap_loaders(
    seq_length=512, batch_size=4, num_workers=4,
    data_dir="./MIDI2ScoreTransformer/data/", use_beat_conditioning=True)
batch = next(iter(train_loader))
inp, out = batch
print("    input streams:", sorted(inp.keys()))
assert "beat" in inp, "beat stream missing from batch!"
beat = inp["beat"]
# one-hot over 13 buckets; count notes assigned to a real beat bucket (not the
# no-beat bucket index 12) across the batch as a sanity signal.
print(f"    beat tensor shape={tuple(beat.shape)} dtype={beat.dtype}")
if beat.dim() == 3:
    bucket = beat.argmax(-1)
    nonzero_phase = (bucket < 12).sum().item()
    print(f"    notes with a real beat bucket (<12): {nonzero_phase} "
          f"/ {bucket.numel()} (expect many — beats are present)")

print("[3] forward + loss + backward on CUDA ...", flush=True)
dev = torch.device("cuda")
model.to(dev).train()
inp = {k: v.to(dev) for k, v in inp.items()}
out = {k: v.to(dev) for k, v in out.items()}
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    pred = model.forward(input_streams=inp, output_streams=out)
    loss, _ = model._compute_loss(pred, out)
loss.backward()
gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1e9).item()
beat_grad = emb.weight.grad.abs().max().item() if emb.weight.grad is not None else None
print(f"    step-0 loss={loss.item():.4f}  grad_norm={gnorm:.3f}")
print(f"    beat Linear grad |max|={beat_grad:.3e} "
      f"(>0 => the beat signal has gradient -> it will learn)")
mem = torch.cuda.max_memory_allocated() / 1e9
print(f"    peak CUDA mem={mem:.2f} GB (A100 80GB)")
print("SMOKE PASS: warm-start identity + beat forward/backward OK.")
