"""Synthetic PDMX dataset wrapper.

Mirrors the on-disk layout that ASAPDataset uses but reads from a flat manifest
(`data/pairs/_manifest.csv`) and a parallel cache directory. Avoids the
ACPAS-metadata join entirely — the manifest is the source of truth.

A pair on disk:
    pairs/<id>.mid
    pairs/<id>.musicxml
    pairs/<id>_chunks.json
    cache_pdmx/<sha256(midi_path)>.pkl   -> (input_stream, output_stream)

Yields the same (input_stream, output_stream) shapes as ASAPDataset for any
seq_length / padding combination. Augmentations are reused from there.
"""

from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import pandas as pd
import torch
from torch.utils.data import Dataset

# Need MultistreamTokenizer for bucket_midi / bucket_mxl + cut_pad/cat_dict utilities
from utils import cat_dict, cut_pad
from tokenizer import MultistreamTokenizer
from dataset import sha256
# Manifests record paths from the machines the corpora were built on; those strings are the cache
# keys and must be hashed verbatim, while files are opened at their location on this machine.
from pathmap import resolve_path, cache_dir_for


class PDMXDataset(Dataset):
    """Reads tokenised pairs from data/pairs/ + data/cache_pdmx/.

    Augmentations supported (subset of ASAPDataset):
        transpose, tempo_jitter, onset_jitter, velocity_jitter,
        random_crop, random_shift
    """
    def __init__(
        self,
        manifest_csv: str,
        split: str = "train",
        val_fraction: float = 0.10,
        seq_length: Optional[int] = None,
        cache: bool = True,
        padding: str = "per-beat",
        augmentations: Dict[str, Union[float, Dict[str, float]]] | None = None,
        return_continous: bool = False,
        return_paths: bool = False,
        seed: int = 42,
        tuplet_gamma: float = 0.0,
        tuplet_floor: float = 0.05,
    ):
        super().__init__()
        assert split in ("all", "train", "validation", "val")
        if split == "val":
            split = "validation"
        assert padding in ("per-beat", "end", None)
        self.split = split
        self.seq_length = seq_length
        self.cache = cache
        self.padding = padding
        self.augmentations = augmentations or {}
        self.return_continous = return_continous
        self.return_paths = return_paths

        df = pd.read_csv(manifest_csv)
        # Deterministic train/val split
        rng = random.Random(seed)
        idx = list(range(len(df)))
        rng.shuffle(idx)
        n_val = max(1, int(len(idx) * val_fraction))
        val_set = set(idx[:n_val])

        if split == "all":
            keep = idx
        elif split == "validation":
            keep = sorted(val_set)
        else:
            keep = [i for i in range(len(df)) if i not in val_set]
        self.metadata = df.iloc[keep].reset_index(drop=True)

        # For weighted sampling like ASAPDataset, populate self.lengths
        try:
            self.lengths = torch.FloatTensor(self.metadata["n_in_tokens"].values.astype(float))
        except Exception:
            self.lengths = torch.ones(len(self.metadata))

        # Tuplet-rate RESHAPE: the unpaired classical corpus is ~86% tuplet-free, which collapses
        # the model's tuplet prior. With tuplet_gamma>0, upweight tuplet-rich scores in the
        # length-based resampling so the effective corpus tuplet-rate is lifted (calibrate the
        # prior via the data distribution rather than a blunt global loss weight). gamma=0 => the
        # original length-only weighting (byte-identical). Requires a `tuplet_rate` manifest column.
        self.tuplet_gamma = float(tuplet_gamma)
        self._resample_w = self.lengths
        if self.tuplet_gamma > 0 and "tuplet_rate" in self.metadata.columns:
            tr = torch.FloatTensor(self.metadata["tuplet_rate"].fillna(0.0).values.astype(float))
            self._resample_w = self.lengths * torch.pow(tr + float(tuplet_floor), self.tuplet_gamma)

    def __len__(self) -> int:
        return len(self.metadata)

    def _load_pair(self, sample_path: str) -> Tuple[Dict, Dict]:
        """Load (input_stream, output_stream) from cache, falling back to parse."""
        # Match ASAPDataset's hashing convention (which uses sample_path + dataset id),
        # but for synthetic we use just the midi path.
        cache_root = cache_dir_for(sample_path)          # was: Path(sample_path).parents[1] / "cache_pdmx"
        pkl = cache_root / f"{sha256(sample_path)}.pkl"  # key = the RECORDED string, never the resolved one
        if self.cache and pkl.is_file():
            return torch.load(str(pkl), weights_only=False)
        # Fallback: parse from the .mid + .mxl/.musicxml beside it
        midi_path = resolve_path(sample_path)
        mxl_candidates = [midi_path.replace(".mid", ext) for ext in (".mxl", ".musicxml")]
        mxl_path = next((p for p in mxl_candidates if Path(p).is_file()), mxl_candidates[0])
        input_stream = MultistreamTokenizer.parse_midi(midi_path)
        output_stream = MultistreamTokenizer.parse_mxl(mxl_path)
        if self.cache:
            cache_root.mkdir(parents=True, exist_ok=True)
            torch.save((input_stream, output_stream), str(pkl))
        return input_stream, output_stream

    def __getitem__(self, idx: int):
        if self.split == "train":
            idx = int(torch.multinomial(self._resample_w, 1, replacement=True).item())
        sample = self.metadata.iloc[idx]
        sample_path = sample["midi"]
        chunks_path = resolve_path(sample["chunks"])

        input_stream, output_stream = self._load_pair(sample_path)

        # Augmentations (mirroring ASAPDataset's behaviour)
        if self.augmentations.get("transpose", False):
            from dataset import ASAPDataset
            shift = random.randint(-6, 6)
            in_p, out_p, out_a, out_k = ASAPDataset._transpose(
                shift,
                midi_stream=input_stream["pitch"],
                mxl_stream=output_stream["pitch"],
                accidental_stream=output_stream["accidental"],
                keysignature_stream=output_stream["keysignature"],
            )
            input_stream["pitch"] = in_p
            output_stream["pitch"] = out_p
            output_stream["accidental"] = out_a
            output_stream["keysignature"] = out_k

        if (v := self.augmentations.get("tempo_jitter", False)):
            jitter_onset = random.uniform(*v)
            jitter_duration = jitter_onset + random.uniform(-0.05, 0.05)
            input_stream["onset"] = input_stream["onset"] * jitter_onset
            input_stream["duration"] = input_stream["duration"] * jitter_duration
        if (v := self.augmentations.get("onset_jitter", False)):
            jitter = 1 + torch.randn(input_stream["onset"].shape) * v
            inter_note_intervals = torch.diff(input_stream["onset"], prepend=torch.tensor([0]), dim=0)
            input_stream["onset"] = torch.cumsum(inter_note_intervals * jitter, dim=0)
        if (v := self.augmentations.get("velocity_jitter", False)):
            input_stream["velocity"] = input_stream["velocity"] + torch.round(
                torch.randn(input_stream["velocity"].shape) * v
            ).long()
            input_stream["velocity"] = torch.clamp(input_stream["velocity"], 1, 127)

        if self.return_continous:
            return input_stream, output_stream

        input_stream = MultistreamTokenizer.bucket_midi(input_stream)
        output_stream = MultistreamTokenizer.bucket_mxl(output_stream)

        if self.seq_length is not None:
            seq_length = self.seq_length
        else:
            seq_length = max(len(input_stream["onset"]), len(output_stream["offset"])) + 256

        with open(chunks_path) as f:
            chunk_annots = json.load(f)

        # Random crop start
        if (v := self.augmentations.get("random_crop", False)):
            min_beats = 16
            n_chunks = len(chunk_annots["midi"])
            if v is True:
                n_0 = random.randint(0, max(n_chunks - min_beats, 0))
            elif isinstance(v, int):
                avg = sum(len(x) for x in chunk_annots["midi"]) / max(n_chunks, 1)
                step = max(1, int(v / max(avg, 1)))
                n_0 = random.choice(range(0, max(n_chunks - min_beats, 1), step))
            else:
                n_0 = 0
        else:
            n_0 = 0

        def process_chunk(stream, chunk, padding, length):
            if padding == "per-beat":
                return {k: cut_pad(v[chunk], length, 0) for k, v in stream.items()}
            return {k: v[chunk] for k, v in stream.items()}

        new_input_stream = None
        new_output_stream = None
        for midi_chunk, mxl_chunk in zip(chunk_annots["midi"][n_0:],
                                         chunk_annots["mxl"][n_0:]):
            length = max(len(midi_chunk), len(mxl_chunk))
            if new_input_stream is not None and len(new_input_stream["onset"]) + length > seq_length + self.augmentations.get("random_shift", 0):
                break
            in_chunk = process_chunk(input_stream, midi_chunk, self.padding, length)
            out_chunk = process_chunk(output_stream, mxl_chunk, self.padding, length)
            if new_input_stream is None:
                new_input_stream = in_chunk
                new_output_stream = out_chunk
            else:
                new_input_stream = cat_dict(new_input_stream, in_chunk)
                new_output_stream = cat_dict(new_output_stream, out_chunk)

        if new_input_stream is None:
            # Edge case: all measures empty / cropped out
            new_input_stream = {k: v[:0] for k, v in input_stream.items()}
            new_output_stream = {k: v[:0] for k, v in output_stream.items()}

        if (v := self.augmentations.get("random_shift", False)):
            shift = random.randint(0, v - 1)
            for k, val in new_input_stream.items():
                new_input_stream[k] = val[shift:]
            for k, val in new_output_stream.items():
                new_output_stream[k] = val[shift:]

        if self.padding is not None:
            for k, val in new_input_stream.items():
                new_input_stream[k] = cut_pad(val, seq_length, 0)
            for k, val in new_output_stream.items():
                new_output_stream[k] = cut_pad(val, seq_length, 0)

        if self.return_paths:
            mxl_path = resolve_path(sample["mxl"] if "mxl" in sample else sample_path.replace(".mid", ".musicxml"))
            return new_input_stream, new_output_stream, sample_path, mxl_path
        return new_input_stream, new_output_stream


class UnpairedScoreDataset(PDMXDataset):
    """Masked self-supervision over real engraved scores (the released model's recipe).

    Reuses the existing PDMX caches but treats each score as UNPAIRED (no real performance):
    the encoder receives a SURROGATE input = the score's own pitch (which, in our 1:1 rendered
    pairs, IS the cached input pitch) with onset/duration/velocity MASKED to constants. The
    decoder target is the score itself. The conditioning token c_i is added by the loader's
    _WithConditioning wrapper (=1 here). This is the path the released checkpoint used for its
    58k unpaired MuseScore scores (Beyer & Dai 2024, sec 3.1.2; surrogate pitch is worth
    +1.84 E_avg vs zeroing the encoder, Table 6 row 8).

    Only _load_pair changes; all of PDMXDataset's chunking / crop / pad / augmentation machinery
    is reused unchanged (transposition still shifts input+output pitch together, keeping them
    consistent; timing augmentations are no-ops on the masked streams).
    """

    def _load_pair(self, sample_path: str) -> Tuple[Dict, Dict]:
        input_stream, output_stream = super()._load_pair(sample_path)
        # Keep the (score) pitch; mask the performance-specific channels so the model must
        # recover the score from pitch + musical context, not invert a real performance.
        input_stream = dict(input_stream)
        input_stream["onset"] = torch.zeros_like(input_stream["onset"])
        input_stream["duration"] = torch.zeros_like(input_stream["duration"])
        input_stream["velocity"] = torch.full_like(input_stream["velocity"], 64)
        return input_stream, output_stream
