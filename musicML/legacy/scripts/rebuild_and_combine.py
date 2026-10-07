"""Reconstruct the PDMX pairs manifest from disk (the render hung on its tail without writing
it) and combine with the broad-kern manifest into the training manifest. Idempotent."""
import csv
import hashlib
import os
from pathlib import Path

# MUST match how make_pairs created paths: it ran from the repo root with relative out/cache
# dirs, and cached by sha256(str(relative_midi_path)). So use RELATIVE paths from repo root.
os.chdir("/root/Music-ML-BayenLab")
ROOT = Path(".")
PDMX = Path("data/pairs_pdmx")
CACHE = Path("data/cache_pdmx")
FIELDS = ["id", "src_mxl", "midi", "mxl", "chunks", "cache",
          "n_notes", "n_measures", "n_in_tokens", "n_out_tokens"]


def sha(s):
    return hashlib.sha256(s.encode()).hexdigest()


rows = []
mc = cc = mx = 0
for mid in PDMX.glob("*.mid"):
    pid = mid.stem
    # PDMX pairs use .mxl (compressed); kern used .musicxml — accept either.
    mxl = next((PDMX / (pid + e) for e in (".mxl", ".musicxml") if (PDMX / (pid + e)).exists()), None)
    ch = PDMX / (pid + "_chunks.json")
    ck = CACHE / (sha(str(mid)) + ".pkl")
    if not ch.exists():
        mc += 1; continue
    if not ck.exists():
        cc += 1; continue
    if mxl is None:
        mx += 1; continue
    rows.append({"id": pid, "src_mxl": "", "midi": str(mid), "mxl": str(mxl),
                 "chunks": str(ch), "cache": str(ck), "n_notes": 0, "n_measures": 0,
                 "n_in_tokens": 256, "n_out_tokens": 256})

pdmx_manifest = PDMX / "_manifest.csv"
with open(pdmx_manifest, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=FIELDS)
    w.writeheader()
    w.writerows(rows)
print(f"PDMX manifest rebuilt: {len(rows)} rows (skipped: chunks {mc}, cache {cc}, mxl {mx})", flush=True)

# combine with broad-kern
import pandas as pd
parts = []
for f in [str(pdmx_manifest), str(ROOT / "data/pairs_broad/_manifest.csv")]:
    try:
        d = pd.read_csv(f); parts.append(d); print(f"{f}: {len(d)}", flush=True)
    except Exception as e:
        print(f"skip {f}: {e}", flush=True)
combined = pd.concat(parts, ignore_index=True)
combined.to_csv(ROOT / "data/pairs_combined_manifest.csv", index=False)
print(f"COMBINED {len(combined)}", flush=True)
print("__DONE__", flush=True)
