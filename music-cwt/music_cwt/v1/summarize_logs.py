"""
summarize_logs.py — Résume les logs d'entraînement CWT.
Usage : python3 summarize_logs.py [train.log]
"""
import re
import sys
from pathlib import Path

log_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("train.log")
text = log_path.read_text(errors="replace")

# Ligne de fin d'epoch : [ep 001/050] train=0.1234  val=0.1456  time=2700s
epoch_re = re.compile(
    r"\[ep\s*(\d+)/\d+\]\s+train=([\d.]+)\s+val=([\d.]+)\s+time=([\d.]+)s"
)

# Ligne mid-epoch : ep001 | step  600/53556 | loss 0.1234 | 20.0 it/s
mid_re = re.compile(
    r"ep(\d+)\s*\|\s*step\s*(\d+)/(\d+)\s*\|\s*loss\s*([\d.]+)\s*\|\s*([\d.]+)\s*it/s"
)

epochs: dict[int, dict] = {}

for line in text.splitlines():
    m = epoch_re.search(line)
    if m:
        ep = int(m.group(1))
        epochs.setdefault(ep, {})
        epochs[ep]["train"] = float(m.group(2))
        epochs[ep]["val"]   = float(m.group(3))
        epochs[ep]["time"]  = float(m.group(4))
        continue

    m = mid_re.search(line)
    if m:
        ep   = int(m.group(1))
        step = int(m.group(2))
        total= int(m.group(3))
        loss = float(m.group(4))
        speed= float(m.group(5))
        epochs.setdefault(ep, {})
        # Garde seulement le dernier step mid de chaque epoch
        epochs[ep]["last_step"]  = step
        epochs[ep]["total_steps"]= total
        epochs[ep]["last_loss"]  = loss
        epochs[ep]["speed"]      = speed

if not epochs:
    print("Aucune donnée trouvée dans", log_path)
    sys.exit(1)

print(f"{'Epoch':>5}  {'Train':>8}  {'Val':>8}  {'Mid loss':>9}  {'Step':>12}  {'it/s':>6}  {'Time':>7}")
print("-" * 65)
for ep in sorted(epochs):
    d = epochs[ep]
    train = f"{d['train']:.4f}" if "train" in d else "  —   "
    val   = f"{d['val']:.4f}"   if "val"   in d else "  —   "
    mid   = f"{d['last_loss']:.4f}" if "last_loss" in d else "  —   "
    step  = f"{d.get('last_step','?')}/{d.get('total_steps','?')}" if "last_step" in d else "  —   "
    speed = f"{d['speed']:.1f}"     if "speed"     in d else "  — "
    time  = f"{d['time']:.0f}s"     if "time"      in d else "  —  "
    print(f"  {ep:3d}  {train:>8}  {val:>8}  {mid:>9}  {step:>12}  {speed:>6}  {time:>7}")

# Résumé rapide
losses = [d["train"] for d in epochs.values() if "train" in d]
if losses:
    best_ep = min((d["train"], ep) for ep, d in epochs.items() if "train" in d)
    print(f"\nMeilleure train loss : {best_ep[0]:.4f}  (ep {best_ep[1]})")
    print(f"Dernière train loss  : {losses[-1]:.4f}")
    print(f"Delta total          : {losses[0] - losses[-1]:.4f}")
