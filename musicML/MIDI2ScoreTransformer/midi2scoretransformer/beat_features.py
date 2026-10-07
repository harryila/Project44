"""Per-note beat-conditioning features for the MIDI2ScoreTransformer.

The model currently fails on tuplet-heavy / multi-meter pieces (Mazeppa: MeanER ~14
even in-training) because it must INFER the metric grid from raw note timings and
gets it wrong — it under-produces tuplets, invents wrong meters, mis-bars the piece.

This module computes, for each input note onset, its METRIC POSITION relative to a
beat/downbeat grid — the information the model currently lacks. Validated on Mazeppa:
the phase-in-beat histogram peaks at the downbeat (0.0) AND at the triplet
subdivisions (0.33, 0.67), so the feature cleanly encodes triplet structure.

Beats/downbeats come from:
  - TRAINING: ASAP ground-truth (asap_annotations.json performance_beats/downbeats).
  - INFERENCE: a MIDI beat-tracker (partitura / PM2S / madmom), or pretty_midi's
    get_beats()/get_downbeats() as a crude fallback.
"""
from __future__ import annotations
import numpy as np


def phase_features(onsets, beats, downbeats=None):
    """For each onset time, compute (phase_in_beat, phase_in_bar) in [0, 1).

    phase_in_beat: position within the enclosing beat interval — captures sub-beat
        rhythm. Triplet-8ths land at ~0.33/0.67; straight-16ths at 0.25/0.5/0.75.
    phase_in_bar: position within the enclosing bar (between downbeats) — captures
        metric position, helps meter inference. Returned only if downbeats given.

    Args:
        onsets: array-like of note onset times (seconds), any order.
        beats: sorted array of beat times (seconds).
        downbeats: optional sorted array of downbeat times (seconds).

    Returns:
        (N, 1) or (N, 2) float array aligned to `onsets` order, values in [0, 1).
    """
    onsets = np.asarray(onsets, dtype=float)
    beats = np.asarray(sorted(beats), dtype=float)
    if len(beats) < 2:
        # Degenerate: no usable beat grid — return neutral 0.5 phases.
        cols = 1 if downbeats is None else 2
        return np.full((len(onsets), cols), 0.5, dtype=np.float32)

    def _phase(t, grid):
        j = np.searchsorted(grid, t)
        if j <= 0:
            g0, g1 = grid[0], grid[1]
        elif j >= len(grid):
            g0, g1 = grid[-2], grid[-1]
        else:
            g0, g1 = grid[j - 1], grid[j]
        p = (t - g0) / (g1 - g0) if g1 > g0 else 0.0
        return min(max(p, 0.0), 0.999)

    pb = np.array([_phase(t, beats) for t in onsets], dtype=np.float32)
    if downbeats is None:
        return pb.reshape(-1, 1)
    db = np.asarray(sorted(downbeats), dtype=float)
    if len(db) < 2:
        pbar = np.full(len(onsets), 0.5, dtype=np.float32)
    else:
        pbar = np.array([_phase(t, db) for t in onsets], dtype=np.float32)
    return np.stack([pb, pbar], axis=1)


# Beat-feature vocab: 12 sub-beat ticks (a 32nd-note-triplet grid that resolves BOTH
# straight 16ths AND triplets — per Wachter/Klangio arXiv:2604.22290, which beat Beyer
# on onset error with exactly this grid) + 1 reserved "no-beat" bucket (index 12) for
# notes outside the beat grid and for padded positions. 13 classes total.
N_BEAT_TICKS = 12
BEAT_VOCAB = N_BEAT_TICKS + 1  # = 13 ; this is config.in_beat_vocab_size


def bucket_beat_phase(phase_in_beat, valid=None):
    """One-hot the phase-in-beat onto 12 ticks + a no-beat bucket -> (N, 13).

    phase_in_beat: (N,) in [0,1) (use phase_features(..., downbeats=None)[:,0]).
    valid: optional (N,) bool; False entries map to the reserved no-beat bucket 12
        (e.g. notes before the first / after the last beat). If None, all valid.

    Tick 0 = on the beat; tick 4 = triplet-8th (0.333); tick 6 = straight-8th (0.5);
    tick 8 = triplet (0.667). The triplet positions are exactly representable.
    """
    import torch
    p = np.asarray(phase_in_beat, dtype=float).reshape(-1)
    idx = np.clip((p * N_BEAT_TICKS).astype(int), 0, N_BEAT_TICKS - 1)
    if valid is not None:
        idx = np.where(np.asarray(valid, dtype=bool), idx, N_BEAT_TICKS)
    oh = np.zeros((len(idx), BEAT_VOCAB), dtype=np.float32)
    oh[np.arange(len(idx)), idx] = 1.0
    return torch.from_numpy(oh)


def no_beat_stream(n_notes):
    """All-"no-beat" one-hot (N, 13): every note in reserved bucket 12. Used when
    no beat annotations are available (PDMX synthetic, plain MIDI inference)."""
    import torch
    oh = torch.zeros((int(n_notes), BEAT_VOCAB), dtype=torch.float32)
    oh[:, N_BEAT_TICKS] = 1.0
    return oh


def beats_from_pretty_midi(pm):
    """Fallback inference-time beat/downbeat estimate from a pretty_midi object.

    Crude (over-segments dense rubato ~30%); use a real tracker (PM2S/partitura)
    for production. Provided so the pipeline runs end-to-end without extra deps.
    """
    return np.asarray(pm.get_beats()), np.asarray(pm.get_downbeats())
