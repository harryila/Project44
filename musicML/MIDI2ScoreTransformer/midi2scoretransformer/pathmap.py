"""Portable resolution of the machine paths recorded in the data manifests.

The manifests under data/ (and the April manifest for legacy/data) record absolute paths from the
machines the corpora were built on, for example

    /root/Music-ML-BayenLab/data/pairs_deduped_full/<id>.mid
    /Users/harry/datasets/pdmx/mxl/...

Those strings are ALSO the cache keys: every pickle in data/cache_pdmx_* is named sha256(<raw string>).
So the raw string must never be rewritten (hash it as recorded) while the file must be opened at its
real location on this machine. This module does only that mapping. It is torch-free on purpose.

Overrides (semicolon separated "old_prefix=new_prefix" pairs):
    MUSICML_PATH_MAP   prefix rewrites applied when OPENING a file
    MUSICML_CACHE_MAP  "<resolved pairs dir>=<cache dir>" pairs for the tokenizer cache
    MUSICML_ROOT       project root (default: two levels above this file's package)
    MUSICML_DATASETS   where external datasets live (default: ~/datasets)
"""
from __future__ import annotations

import os
from pathlib import Path

_PKG = Path(__file__).resolve().parent                 # .../MIDI2ScoreTransformer/midi2scoretransformer
ROOT = Path(os.environ.get("MUSICML_ROOT") or _PKG.parents[1])
DATASETS = Path(os.environ.get("MUSICML_DATASETS") or os.path.expanduser("~/datasets"))

_DEFAULT_PATH_MAP = [
    ("/root/Music-ML-BayenLab", str(ROOT)),
    ("/Users/harry/Desktop/temp/musicML", str(ROOT)),
    ("/root/datasets", str(DATASETS)),
    ("/Users/harry/datasets", str(DATASETS)),
    ("data/pairs/", str(ROOT / "legacy" / "data" / "pairs") + "/"),   # April 2026 manifest, relative paths
]
_DEFAULT_CACHE_MAP = [
    (str(ROOT / "data" / "pairs_deduped_full"), str(ROOT / "data" / "cache_pdmx_full")),
    (str(ROOT / "data" / "pairs_deduped_smoke"), str(ROOT / "data" / "cache_pdmx_smoke")),
    (str(ROOT / "legacy" / "data" / "pairs"), str(ROOT / "legacy" / "data" / "cache_pdmx")),
]


def _parse(env: str, default):
    raw = os.environ.get(env)
    if not raw:
        return list(default)
    pairs = []
    for item in raw.split(";"):
        if "=" in item:
            old, new = item.split("=", 1)
            pairs.append((old.strip(), new.strip()))
    return pairs + list(default)


PATH_MAP = _parse("MUSICML_PATH_MAP", _DEFAULT_PATH_MAP)
CACHE_MAP = _parse("MUSICML_CACHE_MAP", _DEFAULT_CACHE_MAP)


def resolve_path(recorded: str) -> str:
    """Return the location on this machine of a path string recorded in a manifest.

    The first matching prefix wins; strings that already exist are returned unchanged.
    """
    if os.path.exists(recorded):
        return recorded
    for old, new in PATH_MAP:
        if recorded.startswith(old):
            return new + recorded[len(old):]
    return recorded


def cache_dir_for(recorded_sample_path: str) -> Path:
    """Directory holding sha256(<recorded path>).pkl for a manifest sample.

    Mirrors the original rule (<pairs parent>/cache_pdmx) after path resolution, with per-pack
    overrides because the shipped packs are named cache_pdmx_full / cache_pdmx_smoke.
    """
    resolved = Path(resolve_path(recorded_sample_path))
    pairs_dir = str(resolved.parent)
    for pairs, cache in CACHE_MAP:
        if pairs_dir == pairs:
            return Path(cache)
    return resolved.parents[1] / "cache_pdmx"
