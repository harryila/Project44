"""Build a kern-UPWEIGHTED training manifest: PDMX (sweet-spot 84K) + the tuplet-rich kern
corpus repeated KERN_MULT times, so tuplet exposure goes from <1% to ~25% of synthetic and the
tail can actually learn. PDMX capped at 84K (more regressed it via pop-dilution; see ar_full4)."""
import os
import pandas as pd

os.chdir("/root/Music-ML-BayenLab")
PDMX_CAP = 84000
KERN_MULT = 30

pdmx = pd.read_csv("data/pairs_pdmx/_manifest.csv").head(PDMX_CAP)
kern = pd.read_csv("data/pairs_broad/_manifest.csv")
kern_up = pd.concat([kern] * KERN_MULT, ignore_index=True)
combined = pd.concat([pdmx, kern_up], ignore_index=True)
combined.to_csv("data/pairs_upweighted_manifest.csv", index=False)
pct = 100 * len(kern_up) / len(combined)
print(f"PDMX {len(pdmx)} + kern {len(kern)}x{KERN_MULT}={len(kern_up)} "
      f"= {len(combined)} total; kern is {pct:.0f}% of synthetic", flush=True)
print("__DONE__", flush=True)
