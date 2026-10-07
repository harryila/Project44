"""One place for every machine-specific location in this project. Torch-free.

Usage from any script:
    import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    from musicml_paths import REPO_ROOT, DATA_DIR, CHECKPOINTS_DIR, OUTPUTS_DIR, PDMX_ROOT, device, musescore

Every value can be overridden by an environment variable (see .env.example). Defaults are relative to
the project root, which is the directory containing this file, so the project works from any location.

Recorded manifest paths (the /root/Music-ML-BayenLab strings inside data/*.csv) are NOT rewritten:
MIDI2ScoreTransformer/midi2scoretransformer/pathmap.py maps them when files are opened and keeps the
raw strings as cache keys.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

REPO_ROOT = Path(os.environ.get("MUSICML_ROOT") or Path(__file__).resolve().parent)
DATA_DIR = Path(os.environ.get("MUSICML_DATA_DIR") or REPO_ROOT / "data")
CHECKPOINTS_DIR = Path(os.environ.get("MUSICML_CHECKPOINTS_DIR") or REPO_ROOT / "MIDI2ScoreTransformer" / "checkpoints")
OUTPUTS_DIR = Path(os.environ.get("MUSICML_OUTPUTS_DIR") or REPO_ROOT / "outputs")
DATASETS_ROOT = Path(os.environ.get("MUSICML_DATASETS") or os.path.expanduser("~/datasets"))
PDMX_ROOT = Path(os.environ.get("MUSICML_PDMX_ROOT") or DATASETS_ROOT / "pdmx")
TF_ROOT = REPO_ROOT / "MIDI2ScoreTransformer"
TF_PKG = TF_ROOT / "midi2scoretransformer"

# The three live checkpoints, at the paths every script and report expects.
RELEASED_CKPT = CHECKPOINTS_DIR / "MIDI2ScoreTF.ckpt"
BEST_OWN_CKPT = CHECKPOINTS_DIR / "ssl_tuplet20" / "last.ckpt"
SSL_CLASSICAL_CLEAN_CKPT = CHECKPOINTS_DIR / "ssl_classical_clean" / "ssl_classical_clean-epoch=13-val" / "total=0.5125.ckpt"


def _binary(env: str, names, fallbacks) -> str | None:
    p = os.environ.get(env)
    if p and Path(p).is_file():
        return p
    for n in names:
        w = shutil.which(n)
        if w:
            return w
    for f in fallbacks:
        if Path(f).is_file():
            return f
    return None


def musescore() -> str | None:
    return _binary("MUSICML_MUSESCORE", ["mscore", "musescore", "mscore4", "MuseScore4"],
                   ["/Applications/MuseScore 4.app/Contents/MacOS/mscore", "/opt/homebrew/bin/mscore",
                    "/usr/local/bin/mscore", "/usr/bin/mscore"])


def lilypond() -> str | None:
    return _binary("MUSICML_LILYPOND", ["lilypond"], ["/opt/homebrew/bin/lilypond", "/usr/local/bin/lilypond"])


def device() -> str:
    """'cuda' when available, otherwise 'cpu'. Never 'mps': Apple GPU corrupts the model's pad logits."""
    forced = os.environ.get("MUSICML_DEVICE")
    if forced:
        return forced
    try:
        import torch  # imported lazily so this module stays torch-free
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


if __name__ == "__main__":
    for k in ("REPO_ROOT", "DATA_DIR", "CHECKPOINTS_DIR", "OUTPUTS_DIR", "DATASETS_ROOT", "PDMX_ROOT"):
        print(f"{k:18s} {globals()[k]}")
    print(f"{'musescore':18s} {musescore()}")
    print(f"{'lilypond':18s} {lilypond()}")
    print(f"{'device':18s} {device()}")
    for k in ("RELEASED_CKPT", "BEST_OWN_CKPT", "SSL_CLASSICAL_CLEAN_CKPT"):
        p = globals()[k]
        print(f"{k:18s} {'present' if p.is_file() else 'MISSING'}  {p}")
