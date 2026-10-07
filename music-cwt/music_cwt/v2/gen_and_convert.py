"""Script one-shot : génère avec un checkpoint et convertit en MXL."""
import sys
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
BASE = str(Path(__file__).parent)
sys.path.insert(0, BASE + "/tokenizers")
sys.path.insert(0, BASE)

from generate import generate  # noqa
from music21 import converter  # noqa
import argparse

# ── Args ──────────────────────────────────────────────────────────────────────
p = argparse.ArgumentParser()
p.add_argument("--checkpoint",      required=True)
p.add_argument("--output",          required=True)
p.add_argument("--prompt-krn",      default="")
p.add_argument("--prompt-measures", type=int,   default=6)
p.add_argument("--max-tokens",      type=int,   default=1200)
p.add_argument("--temperature",     type=float, default=1.0)
p.add_argument("--top-k",           type=int,   default=50)
p.add_argument("--key",             default="K_0_maj")
p.add_argument("--time",            default="M_4/4")
p.add_argument("--tempo",           default="MM_120")
p.add_argument("--style",           default="")
p.add_argument("--seed",            type=int,   default=42)
args = p.parse_args()

# ── Génération ────────────────────────────────────────────────────────────────
generate(args)

# ── Conversion MXL ───────────────────────────────────────────────────────────
krn_path = args.output
mxl_path = krn_path.replace(".krn", ".mxl")
s = converter.parse(krn_path)
# Humdrum place *staff1 en bas et *staff2 en haut — inverser pour MuseScore
parts = list(s.parts)
if len(parts) == 2:
    s.remove(parts[0])
    s.remove(parts[1])
    s.insert(0, parts[1])
    s.insert(0, parts[0])
s.write("musicxml", mxl_path)
print(f"MXL OK : {mxl_path}  ({len(list(s.flatten().notes))} notes)")
