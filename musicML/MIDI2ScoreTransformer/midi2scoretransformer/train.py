"""PyTorch Lightning trainer for MIDI2ScoreTransformer.

Upstream MIDI2ScoreTransformer ships dataset/inference code but no training
loop. This module provides:
    - A training_step / validation_step that adds weighted CE losses across
      all output streams + BCE on the pad stream.
    - configure_optimizers with AdamW + cosine schedule + linear warmup.
    - LR-range test helper (LRFinder-style sweep) for the sanity check before
      committing to a long pretrain run.
    - A `fit_pdmx` and `fit_asap` driver to run Stage A and Stage B.

Loss weights and ignore-index defaults come from `config.FEATURES`. The pad
stream is special-cased as binary (sigmoid -> BCE) since it gates whether each
output position is a real note.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Lightning
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

# Local imports
from config import FEATURES, MyModelConfig
from models.roformer import Roformer
from utils import pad_batch

# Compat shim (newer transformers): the released ckpt's MyModelConfig predates
# the `_attn_implementation_internal` attribute that current transformers expect
# to read during model construction. Provide a default + allow safe unpickling.
if not hasattr(MyModelConfig, "_attn_implementation_internal"):
    MyModelConfig._attn_implementation_internal = None
torch.serialization.add_safe_globals([MyModelConfig])

log = logging.getLogger(__name__)

# cuda or cpu only: Apple MPS silently corrupts this model's pad logits (see benchmark/eval_tier1_asap.py).
device_str = os.environ.get("MUSICML_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")


def collate_fn(batch):
    """Pad a list of (input_stream, output_stream) tuples to a common length."""
    inputs = [b[0] for b in batch]
    outputs = [b[1] for b in batch]
    return pad_batch([{k: v.unsqueeze(0) for k, v in s.items()} for s in inputs]), \
           pad_batch([{k: v.unsqueeze(0) for k, v in s.items()} for s in outputs])


class TrainableRoformer(Roformer):
    """Adds Lightning training methods to the existing Roformer model."""

    def __init__(self, *args, learning_rate: float = 3e-4, weight_decay: float = 0.2,
                 warmup_steps: int = 1000, total_steps: int = 40000,
                 betas: tuple = (0.9, 0.999), input_dropout: float = 0.75,
                 unconditional_dropout: float = 0.5, tuplet_weight: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        # Save extra training hparams
        self._lr = learning_rate
        self._weight_decay = weight_decay
        self._warmup_steps = warmup_steps
        self._total_steps = total_steps
        self._betas = betas
        self._input_dropout = input_dropout
        self._unconditional_dropout = unconditional_dropout
        # Tuplet-aware loss reweighting: multiplier applied to the rare NON-DYADIC (tuplet)
        # buckets of the rhythm streams (duration/offset/downbeat). These streams live on a
        # 1/24-quarter grid where dyadic note values land on multiples of 3 (quarter=24,
        # eighth=12, 16th=6) and triplets/sextuplets land OFF them (triplet-eighth=8,
        # triplet-quarter=16, triplet-16th=4). The base model's tuplet head collapses to ~0
        # (emits 0 tuplets on 9/14 ASAP pieces) because tuplet buckets are rare and the
        # duration/offset loss-weights are low; upweighting their CE un-collapses it. 1.0 = off.
        self._tuplet_weight = float(tuplet_weight)
        self._tuplet_weight_cache: dict = {}
        # Fraction of samples to FULLY drop the decoder input on (per-sample). Forces true
        # generation/translation (predict the whole score from the encoder, zero decoder hints)
        # instead of the copy-25%-hints shortcut that collapses to all-pad at inference.
        # Training-only; set in fit() (not a saved hyperparameter — inference is a no-op).
        self._decoder_full_drop_prob = 0.0
        # Masked-SSL (unpaired-score) mode: when True, the DATASET already builds the encoder
        # input (surrogate pitch + masked timing for unpaired; real perf for paired), so
        # _maybe_drop_input is a no-op, and decoder prior-token dropout is per-branch (75%
        # paired / 50% unpaired, paper sec 3.1.2). Set in fit() for --dataset-type ssl.
        self._ssl_mode = False
        # Anti-forgetting distillation: a frozen teacher (the released baseline) anchors
        # the student's outputs where the teacher is good (easy/non-tuplet content) while
        # CE drives new (tuplet) learning. Set via set_teacher(); off by default.
        self._teacher = None
        self._distill_weight = 0.0
        self._distill_temp = 2.0
        self._distill_free = set()

    def set_teacher(self, teacher: "TrainableRoformer", weight: float, temp: float = 2.0,
                    free_streams: tuple = ()):
        """Attach a FROZEN teacher for learning-without-forgetting distillation.
        NOTE: assigning an nn.Module to self._teacher AUTO-REGISTERS it as a submodule
        (nn.Module.__setattr__). We rely on that: on_fit_start moves it to device, and
        on_save_checkpoint explicitly strips the `_teacher.*` keys so it is not saved (which
        would break strict eval-time loading). It is never *trained* because we freeze its
        params below (requires_grad_(False)), so the optimizer's param list excludes it.
        Do NOT "fix" this into a plain attribute — that would break the device move + strip.

        free_streams: stream names EXCLUDED from the distill anchor (the student may relearn
        them freely). For the tuplet/meter task, set {"duration","offset","downbeat"} so the
        teacher preserves pitch/hand/voice accuracy while rhythm is free to improve."""
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        self._teacher = teacher
        self._distill_weight = weight
        self._distill_temp = temp
        self._distill_free = set(free_streams)

    def on_fit_start(self):
        if self._teacher is not None:
            self._teacher.to(self.device)
            self._teacher.eval()

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        """Keep the frozen teacher OUT of the saved checkpoint (it's a registered
        submodule, so it would otherwise be saved + break strict loading at eval)."""
        sd = checkpoint.get("state_dict")
        if sd:
            for k in [k for k in sd if k.startswith("_teacher.")]:
                del sd[k]

    def _distill_loss(self, student_pred: dict, teacher_pred: dict, target: dict) -> torch.Tensor:
        """Masked, temperature-scaled KL(student || teacher) summed over streams with the
        same per-stream weights as the CE loss. Only real (non-pad) positions count."""
        real_mask = target["pad"].float()
        if real_mask.dim() == 3:
            real_mask = real_mask.squeeze(-1)
        T = self._distill_temp
        total = torch.tensor(0.0, device=real_mask.device)
        denom = real_mask.sum().clamp(min=1.0)
        for stream, conf in FEATURES.items():
            if stream == "pad" or stream not in student_pred or stream not in teacher_pred:
                continue
            if stream in self._distill_free:
                continue  # rhythm streams: let the student relearn them freely (no anchor)
            s = student_pred[stream]
            t = teacher_pred[stream]
            if s.dim() < 3 or t.dim() < 3:
                continue
            logp_s = F.log_softmax(s / T, dim=-1)
            p_t = F.softmax(t / T, dim=-1)
            kl = F.kl_div(logp_s, p_t, reduction="none").sum(-1)  # (B, T)
            kl = (kl * real_mask).sum() / denom
            total = total + conf["loss_weight"] * kl * (T * T)
        return total

    def _maybe_drop_input(self, input_streams: dict) -> dict:
        """Apply token-position dropout (input_dropout) and full-context dropout
        (unconditional_dropout) during training. Inference: no-op."""
        if not self.training:
            return input_streams
        if getattr(self, "_ssl_mode", False):
            # The dataset already constructed the encoder input: surrogate score-pitch +
            # masked onset/duration/velocity for unpaired, real performance for paired. Zeroing
            # or position-dropping here would destroy the surrogate pitch (= the paper's
            # "no surrogate pitch" ablation, +1.84 E_avg worse). So leave it untouched.
            return input_streams
        if self._unconditional_dropout > 0 and torch.rand(1).item() < self._unconditional_dropout:
            zeroed = {k: torch.zeros_like(v) for k, v in input_streams.items()}
            zeroed["pad"] = input_streams["pad"]  # keep mask so attention works
            return zeroed
        if self._input_dropout > 0:
            mask_shape = (input_streams["pitch"].shape[0], input_streams["pitch"].shape[1], 1)
            keep = (torch.rand(mask_shape, device=input_streams["pitch"].device) > self._input_dropout).float()
            return {k: (v * keep if v.dim() == 3 else v) for k, v in input_streams.items()}
        return input_streams

    def _maybe_drop_decoder(self, output_streams: dict, unpaired_mask=None) -> dict:
        """Drop tokens fed to the decoder (prior-token / exposure-bias dropout). Without this,
        the decoder learns the trivial identity (decoder_in == target), and at inference time
        when the start token is all-zero it predicts all-zeros for everything. Match the released
        checkpoint's input_dropout=0.75. In SSL mode use 75% for paired / 50% for unpaired rows
        (paper sec 3.1.2: the surrogate input is information-poor, so ease the unpaired objective)."""
        if not self.training or self._input_dropout <= 0:
            return output_streams
        T = output_streams["pitch"].shape[1]
        B = output_streams["pitch"].shape[0]
        device = output_streams["pitch"].device
        rate = self._input_dropout
        if getattr(self, "_ssl_mode", False) and unpaired_mask is not None:
            # per-row: unpaired -> 0.50, paired -> 0.75
            rate = torch.where(unpaired_mask.view(B, 1, 1).to(device),
                               torch.full((), 0.50, device=device),
                               torch.full((), 0.75, device=device))
        keep = (torch.rand((B, T, 1), device=device) > rate).float()
        if self._decoder_full_drop_prob > 0:
            # Per-sample: with prob p, zero the ENTIRE decoder row -> the model must predict
            # the whole score from the encoder alone (the true inference/translation task),
            # which teaches it to escape the all-pad fixed point at generation time.
            full = (torch.rand((B, 1, 1), device=device) < self._decoder_full_drop_prob).float()
            keep = keep * (1.0 - full)
        out = {}
        for k, v in output_streams.items():
            out[k] = v * keep if v.dim() == 3 else v
        return out

    # Rhythm streams whose buckets sit on the 1/24-quarter grid (tuplet-aware reweighting target).
    # duration/offset bucket i -> value*24 = i ; downbeat bucket i -> value*24 = i-1 (min=-1/24).
    _RHYTHM_GRID_OFFSET = {"duration": 0, "offset": 0, "downbeat": 1}

    def _stream_class_weights(self, stream: str, vocab: int, device) -> Optional[torch.Tensor]:
        """Per-class CE weight upweighting tuplet (non-dyadic) buckets by self._tuplet_weight.
        Returns None (=> standard unweighted CE, byte-identical to before) when the weight is 1.0
        or the stream is not a 1/24-grid rhythm stream. Cached per (stream, vocab, device)."""
        if self._tuplet_weight == 1.0 or stream not in self._RHYTHM_GRID_OFFSET:
            return None
        key = (stream, vocab, str(device))
        w = self._tuplet_weight_cache.get(key)
        if w is None:
            idx = torch.arange(vocab)
            val24 = idx - self._RHYTHM_GRID_OFFSET[stream]  # value in 1/24-quarter units
            is_tuplet = (val24 % 3 != 0)                    # dyadic values are multiples of 3
            w = torch.ones(vocab)
            w[is_tuplet] = self._tuplet_weight
            w = w.to(device)
            self._tuplet_weight_cache[key] = w
        return w

    def _compute_loss(self, pred: dict, target: dict) -> tuple[torch.Tensor, dict]:
        """Weighted CE per stream + BCE for pad."""
        losses: Dict[str, torch.Tensor] = {}
        total = torch.tensor(0.0, device=pred[list(pred.keys())[0]].device)

        # Position mask: 1 where the target is a real note
        # target['pad'] shape: (B, T) long -> 1.0 for real, 0.0 for padded
        if target["pad"].dim() == 2:
            real_mask = target["pad"].float()
        else:
            real_mask = target["pad"].float().squeeze(-1)
        # BCE on pad
        if "pad" in pred:
            pad_logits = pred["pad"].squeeze(-1)
            bce = F.binary_cross_entropy_with_logits(pad_logits, real_mask)
            losses["pad"] = bce
            total = total + bce

        for stream, conf in FEATURES.items():
            if stream == "pad" or stream not in pred:
                continue
            if stream not in target:
                continue
            logits = pred[stream]
            tgt_onehot = target[stream]
            if tgt_onehot.dim() < 3:
                continue
            # Convert one-hot to class index. Padded positions also have argmax=0,
            # but ignore_index masks them out via real_mask.
            tgt_idx = tgt_onehot.argmax(-1)  # (B, T)
            # Apply ignore at padded positions
            ignore_idx = conf.get("ignore_index", -100)
            if ignore_idx is None:
                ignore_idx = -100
            tgt_masked = tgt_idx.clone()
            # Tuplet-aware per-class weighting (None => standard CE, byte-identical to before).
            cw = self._stream_class_weights(stream, logits.size(-1), logits.device)
            if ignore_idx >= 0:
                tgt_masked[real_mask == 0] = ignore_idx
                ce = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    tgt_masked.reshape(-1),
                    ignore_index=ignore_idx,
                    weight=cw,
                )
            else:
                # use boolean mask
                B, T, V = logits.shape
                lr = logits.reshape(-1, V)
                tr = tgt_masked.reshape(-1)
                m = real_mask.reshape(-1) > 0
                if m.sum() == 0:
                    ce = torch.tensor(0.0, device=logits.device)
                else:
                    ce = F.cross_entropy(lr[m], tr[m], weight=cw)
            losses[stream] = ce
            # Defensive: when a batch has 100% ignore-class for a given stream
            # (e.g. the entire piece has hand=2 because Part IDs lack staff hints),
            # F.cross_entropy returns NaN. Skip that stream rather than poison total.
            if not torch.isnan(ce):
                total = total + conf["loss_weight"] * ce
        return total, losses

    def training_step(self, batch, batch_idx):  # type: ignore[override]
        input_streams, output_streams = batch
        # SSL: the conditioning token marks unpaired (surrogate) rows; use it to pick the
        # per-branch decoder dropout (75% paired / 50% unpaired). Compute BEFORE dropping input.
        unpaired_mask = None
        if getattr(self, "_ssl_mode", False) and "unconditional" in input_streams:
            unpaired_mask = (input_streams["unconditional"][:, 0, 0] > 0.5)
        input_streams = self._maybe_drop_input(input_streams)
        # Feed the RAW target as decoder input. For is_autoregressive=True, MXLEmbeddings
        # rolls it +1 internally and zeros position 0 (embedding.py:139-145) -> position t
        # predicts target[t] from target[t-1], exactly matching generate()'s roll(-1)+roll(+1).
        # (We must NOT pre-shift here too, or it's a double shift = predict target[t] from
        # target[t-2], a silent train/inference skew.) The 75% input-dropout is the exposure-bias
        # regularizer; for the bidirectional path it doubles as the masked-token mechanism.
        decoder_in = self._maybe_drop_decoder(output_streams, unpaired_mask=unpaired_mask)
        pred = self.forward(input_streams=input_streams, output_streams=decoder_in)
        loss, parts = self._compute_loss(pred, output_streams)
        self.log("train/total", loss, on_step=True, prog_bar=True)
        for k, v in parts.items():
            self.log(f"train/{k}", v, on_step=True)
        if self._teacher is not None and self._distill_weight > 0:
            # Same (post-dropout) inputs to the frozen teacher -> comparable outputs.
            with torch.no_grad():
                teacher_pred = self._teacher.forward(input_streams=input_streams,
                                                     output_streams=decoder_in)
            distill = self._distill_loss(pred, teacher_pred, output_streams)
            self.log("train/distill", distill, on_step=True, prog_bar=True)
            loss = loss + self._distill_weight * distill
        return loss

    def validation_step(self, batch, batch_idx):  # type: ignore[override]
        input_streams, output_streams = batch
        # Raw target; the embedding does the single +1 shift when is_autoregressive (no manual shift).
        decoder_in = output_streams
        pred = self.forward(input_streams=input_streams, output_streams=decoder_in)
        loss, parts = self._compute_loss(pred, output_streams)
        self.log("val/total", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        for k, v in parts.items():
            self.log(f"val/{k}", v, on_epoch=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(),
            lr=self._lr,
            betas=self._betas,
            weight_decay=self._weight_decay,
        )
        # Linear warmup + cosine decay
        warmup = self._warmup_steps
        total = max(self._total_steps, warmup + 1)
        def lr_lambda(step: int) -> float:
            if step < warmup:
                return float(step + 1) / float(max(1, warmup))
            progress = (step - warmup) / float(max(1, total - warmup))
            progress = min(1.0, progress)
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}


def _new_model(learning_rate: float = 3e-4, total_steps: int = 40000,
               warmup_steps: int = 1000, autoregressive: bool = False,
               use_beat_relative: bool = False,
               **kwargs) -> TrainableRoformer:
    """Create a fresh model matching the released checkpoint's architecture.
    If autoregressive=True, the DECODER is causal (standard AR generation, which trains
    reliably from scratch — sidesteps the unpublished bidirectional mask-predict recipe)."""
    enc = MyModelConfig(
        is_decoder=False, is_autoregressive=False,
        num_hidden_layers=4, hidden_size=512, num_attention_heads=8,
        intermediate_size=1536, max_position_embeddings=1536,
        embedding_size=512, hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1, layer_norm_eps=1e-12,
        rotary_value=False, positional_encoding="RoPE",
    )
    dec = MyModelConfig(
        is_decoder=True, is_autoregressive=autoregressive, add_cross_attention=True,
        num_hidden_layers=4, hidden_size=512, num_attention_heads=8,
        intermediate_size=1536, max_position_embeddings=1536,
        embedding_size=512, hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1, layer_norm_eps=1e-12,
        rotary_value=False, positional_encoding="RoPE",
        use_beat_relative=use_beat_relative,  # B2: within-quarter offset + quarter_idx head
    )
    hp = {
        "components": ["encoder", "decoder"],
        "domains": {"in": "midi", "out": "mxl"},
    }
    model = TrainableRoformer(
        enc_configuration=enc, dec_configuration=dec, hyperparameters=hp,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        **kwargs,
    )
    return model


def load_pretrained_init(ckpt_path: Optional[str] = None,
                         use_beat_conditioning: bool = False,
                         use_beat_relative: bool = False):
    """Initialise from a released checkpoint (warm-start). If use_beat_conditioning,
    add a zero-init 'beat' input Linear on the encoder (byte-identical to baseline).
    If use_beat_relative (B2), the decoder offset head becomes within-quarter (25) and a new
    quarter_idx head is added; the offset/quarter_idx params are SHAPE-MISMATCHED vs the base
    ckpt, so we load only matching-shape keys (those heads keep fresh init, body warm-starts)."""
    if ckpt_path is None or not os.path.exists(ckpt_path):
        return None
    base = Roformer.load_from_checkpoint(ckpt_path, map_location="cpu", weights_only=False)
    enc = base.enc_config
    dec = base.dec_config
    hp = base.hyperparameters
    if use_beat_relative:
        dec.use_beat_relative = True
        dec.out_offset_vocab_size = 25  # within-quarter
        if not hasattr(dec, "out_quarter_idx_vocab_size"):
            dec.out_quarter_idx_vocab_size = 25
        if not hasattr(dec, "in_quarter_idx_vocab_size"):
            dec.in_quarter_idx_vocab_size = 25
    if use_beat_conditioning:
        # Enable the beat input on the ENCODER only (the MIDI side). The released
        # checkpoint lacks the 'beat' Linear; build it fresh, load the rest via
        # strict=False, then ZERO-init it so the warm-started model is byte-identical
        # to the released checkpoint until training learns to use the beat signal.
        enc.use_beat_conditioning = True
        if not hasattr(enc, "in_beat_vocab_size"):
            enc.in_beat_vocab_size = 13
    model = TrainableRoformer(enc_configuration=enc, dec_configuration=dec, hyperparameters=hp)
    # Shape-filtered warm-start: strict=False still ERRORS on shape mismatch, so drop keys whose
    # shapes differ (B2's offset head 145->25) or are absent in base (quarter_idx) -> fresh init.
    _bsd = base.state_dict(); _msd = model.state_dict()
    _filtered = {k: v for k, v in _bsd.items() if k in _msd and _msd[k].shape == v.shape}
    _reinit = sorted(set(_msd) - set(_filtered))
    if use_beat_relative:
        logging.getLogger(__name__).info("B2 warm-start: loaded %d/%d params; fresh-init %d (offset/quarter_idx): %s",
                                         len(_filtered), len(_msd), len(_reinit), _reinit[:6])
    model.load_state_dict(_filtered, strict=False)
    if use_beat_conditioning:
        with torch.no_grad():
            emb = model.embeddings_enc.embeddings["beat"]
            emb.weight.zero_()
            if emb.bias is not None:
                emb.bias.zero_()
    return model


def lr_range_test(model: TrainableRoformer, loader: DataLoader,
                  start_lr: float = 1e-6, end_lr: float = 1e-3,
                  num_steps: int = 200, out_path: Path | None = None) -> Path:
    """LR-range test (LRFinder-style). Linearly increases LR over num_steps and
    records training loss; you pick the LR at the steepest descent.
    """
    import csv
    model.train()
    device = torch.device(device_str)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=start_lr, betas=(0.9, 0.999), weight_decay=0.2)
    if out_path is None:
        out_path = Path("checkpoints/lr_range_test.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    step = 0
    for batch in loader:
        if step >= num_steps:
            break
        input_streams, output_streams = batch
        input_streams = {k: v.to(device) for k, v in input_streams.items()}
        output_streams = {k: v.to(device) for k, v in output_streams.items()}
        # Linear LR sweep in log space
        prog = step / max(num_steps - 1, 1)
        lr = math.exp(math.log(start_lr) + prog * (math.log(end_lr) - math.log(start_lr)))
        for pg in opt.param_groups:
            pg["lr"] = lr
        opt.zero_grad()
        pred = model.forward(input_streams=input_streams, output_streams=output_streams)
        loss, _ = model._compute_loss(pred, output_streams)
        loss.backward()
        opt.step()
        rows.append({"step": step, "lr": lr, "loss": float(loss.detach().cpu())})
        log.info("[LR test] step %d lr=%.2e loss=%.4f", step, lr, float(loss.detach().cpu()))
        step += 1

    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["step", "lr", "loss"])
        w.writeheader()
        w.writerows(rows)
    return out_path


def make_pdmx_loaders(manifest_csv: str, seq_length: int = 512,
                      batch_size: int = 8, num_workers: int = 4):
    from pdmx_dataset import PDMXDataset
    train_ds = PDMXDataset(manifest_csv, split="train", seq_length=seq_length,
                           padding="per-beat",
                           augmentations={"transpose": True, "random_crop": True,
                                          "tempo_jitter": (0.8, 1.2),
                                          "onset_jitter": 0.05,
                                          "random_shift": 8,
                                          "velocity_jitter": 5})
    val_ds = PDMXDataset(manifest_csv, split="validation", seq_length=seq_length,
                         padding="per-beat", augmentations={})
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, collate_fn=collate_fn,
                              persistent_workers=num_workers > 0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, collate_fn=collate_fn,
                            persistent_workers=num_workers > 0)
    return train_loader, val_loader


# Augmentations applied to the REAL ASAP stream (mirror the released ckpt's recipe).
_REAL_AUG = {"transpose": True, "random_crop": True, "tempo_jitter": (0.8, 1.2),
             "onset_jitter": 0.05, "random_shift": 8, "velocity_jitter": 5}


def make_asap_loaders(seq_length: int = 512, batch_size: int = 16,
                      num_workers: int = 8, data_dir: str = "./data/",
                      use_beat_conditioning: bool = False):
    """Real ASAP (perf-MIDI, engraved-MusicXML) pairs via the upstream ASAPDataset.

    The released model was trained on ASAP; this lets us continue-train on the full
    ~750-performance train split (held out by TEST_PIECE_IDS) instead of only the
    14-piece eval subset. Requires _chunks.json (run chunker.py first).
    """
    from dataset import ASAPDataset
    train_ds = ASAPDataset(data_dir, "train", seq_length=seq_length,
                           padding="per-beat", augmentations=_REAL_AUG,
                           use_beat_conditioning=use_beat_conditioning)
    val_ds = ASAPDataset(data_dir, "validation", seq_length=seq_length,
                         padding="per-beat", augmentations={},
                         use_beat_conditioning=use_beat_conditioning)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, collate_fn=collate_fn,
                              persistent_workers=num_workers > 0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, collate_fn=collate_fn,
                            persistent_workers=num_workers > 0, pin_memory=True)
    return train_loader, val_loader


def make_mixed_loaders(manifest_csv: str, seq_length: int = 512, batch_size: int = 16,
                       num_workers: int = 8, real_fraction: float = 0.8,
                       data_dir: str = "./data/"):
    """Joint real+synthetic loader with REAL-MAJORITY replay (default 80/20).

    A naive concat of ~750 real ASAP pairs with ~50k synthetic pairs is ~65:1
    synthetic-dominated, which empirically degraded MUSTER 11.18 -> 18.10 (see
    benchmark/GPU_FINETUNE_RESULTS.md). A WeightedRandomSampler over the
    ConcatDataset upweights the real pairs so each batch is ~`real_fraction` real,
    directly countering the domination / catastrophic forgetting. Validation is on
    REAL ASAP only (we care about real-performance accuracy).
    """
    from dataset import ASAPDataset
    from pdmx_dataset import PDMXDataset
    from torch.utils.data import ConcatDataset, WeightedRandomSampler

    real_train = ASAPDataset(data_dir, "train", seq_length=seq_length,
                             padding="per-beat", augmentations=_REAL_AUG)
    synth_train = PDMXDataset(manifest_csv, split="train", seq_length=seq_length,
                              padding="per-beat",
                              augmentations={"transpose": True, "random_crop": True,
                                             "tempo_jitter": (0.8, 1.2),
                                             "onset_jitter": 0.05, "random_shift": 8,
                                             "velocity_jitter": 5})
    n_real, n_synth = len(real_train), len(synth_train)
    concat = ConcatDataset([real_train, synth_train])

    rf = real_fraction
    w_real = rf / max(n_real, 1)
    w_synth = (1.0 - rf) / max(n_synth, 1)
    weights = torch.tensor([w_real] * n_real + [w_synth] * n_synth, dtype=torch.double)
    # Epoch = one effective pass over the WHOLE dataset (real+synth). For from-scratch
    # builds this means meaningful epochs + far fewer (slow) validations than sizing the
    # epoch to a single pass over the small real set.
    num_samples = n_real + n_synth
    sampler = WeightedRandomSampler(weights, num_samples=num_samples, replacement=True)

    train_loader = DataLoader(concat, batch_size=batch_size, sampler=sampler,
                              num_workers=num_workers, collate_fn=collate_fn,
                              persistent_workers=num_workers > 0, pin_memory=True)
    val_ds = ASAPDataset(data_dir, "validation", seq_length=seq_length,
                         padding="per-beat", augmentations={})
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, collate_fn=collate_fn,
                            persistent_workers=num_workers > 0, pin_memory=True)
    print(f"[mixed] real={n_real} synth={n_synth} target_real_frac={rf} "
          f"samples/epoch={num_samples}", flush=True)
    return train_loader, val_loader


class _WithConditioning(torch.utils.data.Dataset):
    """Append the paper's binary conditioning token c_i to each sample's encoder input stream:
    flag=1.0 => unpaired/surrogate (no real performance), flag=0.0 => real paired (token
    embedding is then 0 via the bias-free Linear, matching the paper's 'set to 0 for labeled
    data and inference'). Keeps batch keys uniform so pad_batch (which keys off batch[0]) works."""

    def __init__(self, ds, flag: float):
        self.ds = ds
        self.flag = float(flag)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        sample = self.ds[idx]
        inp, out = sample[0], sample[1]
        inp = dict(inp)
        T = inp["pitch"].shape[0]
        inp["unconditional"] = torch.full((T, 1), self.flag, dtype=torch.float32)
        return inp, out


def make_ssl_loaders(manifest_csv: str, seq_length: int = 512, batch_size: int = 16,
                     num_workers: int = 8, real_fraction: float = 0.5,
                     data_dir: str = "./data/", tuplet_gamma: float = 0.0):
    """Released-model recipe: 50/50 mix of REAL ASAP pairs and UNPAIRED engraved scores via
    masked self-supervision (dataset_weights=[0.5,0.5]). Unpaired scores feed a surrogate input
    (the score's own pitch, with onset/duration/velocity masked) + a conditioning token, and the
    decoder reconstructs the score with prior-token dropout (50% unpaired / 75% paired, set in
    training_step). Validation is REAL ASAP only. Paper: Beyer & Dai 2024, sec 3.1.2."""
    from dataset import ASAPDataset
    from pdmx_dataset import UnpairedScoreDataset
    from torch.utils.data import ConcatDataset, WeightedRandomSampler

    real_train = _WithConditioning(
        ASAPDataset(data_dir, "train", seq_length=seq_length, padding="per-beat",
                    augmentations=_REAL_AUG), flag=0.0)
    # Unpaired branch: only transposition is meaningful (timing/velocity are masked anyway).
    unp_train = _WithConditioning(
        UnpairedScoreDataset(manifest_csv, split="train", seq_length=seq_length,
                             padding="per-beat", augmentations={"transpose": True},
                             tuplet_gamma=tuplet_gamma), flag=1.0)
    if tuplet_gamma > 0:
        print(f"[ssl] tuplet-rate RESHAPE on: gamma={tuplet_gamma} "
              f"(upweight tuplet-rich unpaired scores in resampling)", flush=True)
    n_real, n_unp = len(real_train), len(unp_train)
    concat = ConcatDataset([real_train, unp_train])

    rf = real_fraction  # 0.5 = paper's [0.5, 0.5]
    weights = torch.tensor([rf / max(n_real, 1)] * n_real +
                           [(1.0 - rf) / max(n_unp, 1)] * n_unp, dtype=torch.double)
    num_samples = n_real + n_unp
    sampler = WeightedRandomSampler(weights, num_samples=num_samples, replacement=True)

    train_loader = DataLoader(concat, batch_size=batch_size, sampler=sampler,
                              num_workers=num_workers, collate_fn=collate_fn,
                              persistent_workers=num_workers > 0, pin_memory=True)
    val_ds = ASAPDataset(data_dir, "validation", seq_length=seq_length,
                         padding="per-beat", augmentations={})
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, collate_fn=collate_fn,
                            persistent_workers=num_workers > 0, pin_memory=True)
    print(f"[ssl] real={n_real} unpaired={n_unp} real_frac={rf} "
          f"samples/epoch={num_samples}", flush=True)
    return train_loader, val_loader


def fit(stage: str, manifest_csv: str, learning_rate: float, max_epochs: int,
        out_dir: str, init_ckpt: Optional[str] = None,
        batch_size: int = 8, seq_length: int = 512, num_workers: int = 4,
        precision: str = "16-mixed", dataset_type: str = "pdmx",
        real_fraction: float = 0.8, data_dir: str = "./data/",
        use_beat_conditioning: bool = False,
        distill_ckpt: Optional[str] = None, distill_weight: float = 0.0,
        distill_temp: float = 2.0, distill_free: tuple = (),
        decoder_full_drop: float = 0.0, autoregressive: bool = False,
        warmup_steps: int = -1, tuplet_weight: float = 1.0,
        tuplet_gamma: float = 0.0, use_beat_relative: bool = False) -> str:
    if use_beat_relative:
        import tokenizer as _tk
        _tk.BEAT_RELATIVE = True
        log.info("B2 beat-relative tokenization ENABLED (within-quarter offset + quarter_idx head)")
    if dataset_type == "asap":
        train_loader, val_loader = make_asap_loaders(
            seq_length, batch_size, num_workers, data_dir=data_dir,
            use_beat_conditioning=use_beat_conditioning)
    elif dataset_type == "mixed":
        assert manifest_csv, "--manifest (synthetic) required for --dataset-type mixed"
        train_loader, val_loader = make_mixed_loaders(
            manifest_csv, seq_length, batch_size, num_workers,
            real_fraction=real_fraction, data_dir=data_dir)
    elif dataset_type == "ssl":
        assert manifest_csv, "--manifest (unpaired engraved scores) required for --dataset-type ssl"
        train_loader, val_loader = make_ssl_loaders(
            manifest_csv, seq_length, batch_size, num_workers,
            real_fraction=real_fraction, data_dir=data_dir, tuplet_gamma=tuplet_gamma)
    else:
        assert manifest_csv, "--manifest required for --dataset-type pdmx"
        train_loader, val_loader = make_pdmx_loaders(manifest_csv, seq_length, batch_size, num_workers)

    steps_per_epoch = max(1, len(train_loader.dataset) // batch_size)
    total_steps = steps_per_epoch * max_epochs

    # Warmup: explicit --warmup-steps overrides the auto heuristic. The released recipe uses 4000.
    _warmup = warmup_steps if (warmup_steps and warmup_steps > 0) else max(100, total_steps // 50)
    log.info("LR schedule: warmup=%d total_steps=%d (cosine)", _warmup, total_steps)
    if init_ckpt and os.path.exists(init_ckpt):
        model = load_pretrained_init(init_ckpt, use_beat_conditioning=use_beat_conditioning, use_beat_relative=use_beat_relative)
        if model is None:
            model = _new_model(learning_rate=learning_rate,
                               warmup_steps=_warmup,
                               total_steps=total_steps, autoregressive=autoregressive, use_beat_relative=use_beat_relative)
        else:
            model._lr = learning_rate
            model._warmup_steps = _warmup
            model._total_steps = total_steps
    else:
        model = _new_model(learning_rate=learning_rate,
                           warmup_steps=_warmup,
                           total_steps=total_steps, autoregressive=autoregressive, use_beat_relative=use_beat_relative)
    if autoregressive:
        log.info("AUTOREGRESSIVE (causal decoder) from-scratch training ON")

    model._decoder_full_drop_prob = float(decoder_full_drop)
    if decoder_full_drop > 0:
        log.info("Decoder full-drop ON: %.2f of samples train pure generation", decoder_full_drop)
    model._ssl_mode = (dataset_type == "ssl")
    if model._ssl_mode:
        log.info("MASKED-SSL mode ON: 50/50 real-pair / unpaired-score; encoder dropout disabled "
                 "(dataset builds surrogate), decoder prior-token dropout 75%% paired / 50%% unpaired")
    # Tuplet-aware loss reweighting (applies to both warm-start and new-model paths).
    model._tuplet_weight = float(tuplet_weight)
    model._tuplet_weight_cache = {}
    if model._tuplet_weight != 1.0:
        log.info("TUPLET-AWARE loss ON: non-dyadic (tuplet) buckets of duration/offset/downbeat "
                 "upweighted x%.2f in CE (un-collapse the tuplet head)", model._tuplet_weight)

    # Anti-forgetting distillation: load a FROZEN teacher (default: the same baseline).
    if distill_weight > 0:
        teacher_path = distill_ckpt or init_ckpt
        teacher = load_pretrained_init(teacher_path, use_beat_conditioning=use_beat_conditioning)
        assert teacher is not None, f"distill teacher not found: {teacher_path}"
        model.set_teacher(teacher, weight=distill_weight, temp=distill_temp,
                          free_streams=tuple(distill_free))
        log.info("Distillation ON: teacher=%s weight=%.3f temp=%.1f free=%s",
                 teacher_path, distill_weight, distill_temp, tuple(distill_free))

    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)
    ckpt_cb = ModelCheckpoint(
        dirpath=str(out_dir_p), filename=f"{stage}-{{epoch:02d}}-{{val/total:.4f}}",
        monitor="val/total", mode="min", save_top_k=2, save_last=True,
    )
    lr_cb = LearningRateMonitor(logging_interval="step")

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        callbacks=[ckpt_cb, lr_cb],
        # Never let Lightning pick Apple MPS ("auto" would): it corrupts this model's pad logits.
        accelerator=os.environ.get("MUSICML_ACCELERATOR", "gpu" if torch.cuda.is_available() else "cpu"),
        devices=os.environ.get("MUSICML_DEVICES", "auto"),
        precision=precision if device_str == "cuda" else "32-true",
        log_every_n_steps=10,
        check_val_every_n_epoch=1,
        default_root_dir=str(out_dir_p),
        gradient_clip_val=1.0,
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    final_ckpt = ckpt_cb.best_model_path or os.path.join(str(out_dir_p), "last.ckpt")
    log.info("Stage %s best ckpt: %s", stage, final_ckpt)
    return final_ckpt


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("lr_test", help="LR-range test on a few steps")
    p.add_argument("--manifest", required=True)
    p.add_argument("--num-steps", type=int, default=200)
    p.add_argument("--start-lr", type=float, default=1e-6)
    p.add_argument("--end-lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seq-length", type=int, default=512)
    p.add_argument("--out", type=Path, default=Path("checkpoints/lr_range_test.csv"))

    p = sub.add_parser("fit", help="Train a stage")
    p.add_argument("--stage", required=True)
    p.add_argument("--manifest", default=None,
                   help="Synthetic PDMX manifest (required for pdmx/mixed dataset-type).")
    p.add_argument("--dataset-type", choices=["pdmx", "asap", "mixed", "ssl"], default="pdmx",
                   help="pdmx=synthetic only; asap=real ASAP only; "
                        "mixed=real-majority replay (real ASAP + synthetic).")
    p.add_argument("--real-fraction", type=float, default=0.8,
                   help="For --dataset-type mixed: target fraction of REAL samples "
                        "per batch (0.8 = 80%% real / 20%% synthetic).")
    p.add_argument("--data-dir", default="./data/",
                   help="ASAP/ACPAS data root (for asap/mixed).")
    p.add_argument("--use-beat-conditioning", action="store_true",
                   help="Add per-note beat-phase conditioning input (fixes tuplet/meter "
                        "failure). Warm-start adds a zero-init beat Linear; trains from ASAP GT beats.")
    p.add_argument("--lr", type=float, required=True)
    p.add_argument("--max-epochs", type=int, default=5)
    p.add_argument("--warmup-steps", type=int, default=-1,
                   help="Explicit LR warmup steps (the released recipe uses 4000); -1 = auto total_steps//50.")
    p.add_argument("--tuplet-weight", type=float, default=1.0,
                   help="Upweight non-dyadic (tuplet) buckets of duration/offset/downbeat CE by this "
                        "factor to un-collapse the tuplet head (1.0 = off / byte-identical loss).")
    p.add_argument("--tuplet-gamma", type=float, default=0.0,
                   help="RESHAPE: upweight tuplet-rich unpaired scores in resampling by "
                        "(tuplet_rate+floor)^gamma (needs a tuplet_rate manifest column). 0 = off.")
    p.add_argument("--beat-relative", action="store_true",
                   help="B2: within-quarter offset + a quarter_idx head (triplets become common "
                        "within-quarter buckets). Sets tokenizer.BEAT_RELATIVE + use_beat_relative.")
    p.add_argument("--init-ckpt", default=None)
    p.add_argument("--out-dir", default="checkpoints")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seq-length", type=int, default=512)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--precision", default="16-mixed")
    p.add_argument("--distill-weight", type=float, default=0.0,
                   help="Anti-forgetting distillation weight (lambda). >0 enables a frozen "
                        "teacher (the baseline) anchoring outputs to prevent drift.")
    p.add_argument("--distill-ckpt", default=None,
                   help="Teacher checkpoint for distillation (default: --init-ckpt).")
    p.add_argument("--distill-temp", type=float, default=2.0)
    p.add_argument("--distill-free", nargs="*", default=[],
                   help="Streams EXCLUDED from the distill anchor (relearned freely). "
                        "For tuplet/meter: duration offset downbeat.")
    p.add_argument("--decoder-full-drop", type=float, default=0.0,
                   help="Fraction of samples to FULLY drop the decoder input (train true "
                        "generation-from-encoder). Fixes the from-scratch all-pad collapse.")
    p.add_argument("--autoregressive", action="store_true",
                   help="Train a STANDARD causal-decoder AR model from scratch (shifted teacher "
                        "forcing). Reliable from-scratch generation; sidesteps the unpublished "
                        "bidirectional mask-predict recipe. Cannot warm-start from the released ckpt.")

    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    if args.cmd == "lr_test":
        train_loader, _ = make_pdmx_loaders(args.manifest, args.seq_length, args.batch_size, num_workers=0)
        model = _new_model(learning_rate=1e-4, total_steps=args.num_steps,
                           warmup_steps=max(1, args.num_steps // 50))
        out = lr_range_test(model, train_loader, start_lr=args.start_lr, end_lr=args.end_lr,
                            num_steps=args.num_steps, out_path=args.out)
        print(f"Saved LR sweep: {out}")
    elif args.cmd == "fit":
        ckpt = fit(args.stage, args.manifest, args.lr, args.max_epochs, args.out_dir,
                   init_ckpt=args.init_ckpt, batch_size=args.batch_size,
                   seq_length=args.seq_length, num_workers=args.num_workers,
                   precision=args.precision, dataset_type=args.dataset_type,
                   real_fraction=args.real_fraction, data_dir=args.data_dir,
                   use_beat_conditioning=args.use_beat_conditioning,
                   distill_ckpt=args.distill_ckpt, distill_weight=args.distill_weight,
                   distill_temp=args.distill_temp, distill_free=tuple(args.distill_free),
                   decoder_full_drop=args.decoder_full_drop, autoregressive=args.autoregressive,
                   warmup_steps=args.warmup_steps, tuplet_weight=args.tuplet_weight,
                   tuplet_gamma=args.tuplet_gamma, use_beat_relative=args.beat_relative)
        print(f"Best ckpt: {ckpt}")


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    main()
