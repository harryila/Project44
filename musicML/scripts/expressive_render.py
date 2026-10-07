"""Render a MusicXML score to a perturbed performance MIDI plus an alignment json.

The synthetic MIDI is meant to look like a (loose) human performance: tempo
drifts, velocity wiggles, notes start a bit late or early, durations vary.
The corresponding MusicXML score is left untouched. Together this gives us a
synthetic (perturbed-MIDI, MusicXML) training pair for MIDI2ScoreTransformer.

Crucially, alignment is exact by construction: notes are extracted from the
score using the SAME mxl_to_list canonical order MIDI2ScoreTransformer uses,
and written to MIDI in that order. midi_to_list (which sorts by (start, pitch,
duration)) produces the same ordering because we apply per-chord onset jitter
(notes sharing a score offset share a jittered onset), so chord-internal order
is preserved.

Robustness:
    The whole `mxl_to_list(...)` call (and downstream processing) is wrapped
    in try/except. If parsing fails (PDMX has ~42 corrupted files plus rarer
    parser errors), this returns {ok: False, error: ...}. The caller
    (make_pairs.py) is expected to log + skip cleanly.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pretty_midi

# Set up the import path so we can use MIDI2ScoreTransformer's tokenizer.
SCRIPT_DIR = Path(__file__).resolve().parent.parent
TOKENIZER_DIR = SCRIPT_DIR / "MIDI2ScoreTransformer" / "midi2scoretransformer"
if str(TOKENIZER_DIR) not in sys.path:
    sys.path.insert(0, str(TOKENIZER_DIR))

log = logging.getLogger(__name__)


@dataclass
class Perturbations:
    """Per-piece perturbation knobs. All randomized via the seed."""
    tempo_walk_std: float = 0.10        # ~N(1.0, 0.10) per-bar multiplier (legacy, unused by beat curve)
    tempo_smooth_window: int = 3         # bars of moving average over the walk (legacy)
    velocity_offset_std: float = 10.0    # per-note velocity ~N(0, 10)
    phrase_loudness_amp: int = 15        # phrase-level cosine envelope ±15
    phrase_length_bars: int = 8
    onset_jitter_seconds: float = 0.022  # ±22 ms gaussian (per chord, not per note)
    duration_scale_std: float = 0.05     # multiplicative N(1, 0.05)
    early_release_prob: float = 0.05
    early_release_factor: float = 0.7
    # --- realistic-rubato model (beat-resolution tempo curve) ---
    # The legacy renderer held tempo CONSTANT within each bar, so tuplets landed at
    # exactly 1/N of the beat. Real rubato varies tempo within the bar and slows into
    # cadences, so the inverse (timing -> tuplet notation) the model must learn is
    # different at test time than what deadpan-within-bar rendering teaches. These knobs
    # add structured sub-beat rubato + phrase-final ritardando. All affect only the shared
    # quarter-length->seconds map, so chord-internal note order is still preserved.
    beat_tempo_std: float = 0.08         # per-(quarter)beat tempo multiplier ~N(1, 0.08)
    beat_grid_ql: float = 0.25           # tempo-curve sampling resolution (quarter-lengths)
    beat_smooth_window: int = 5          # grid points of moving average (~1.25 beats)
    ritard_factor: float = 0.72          # tempo multiplier reached at a phrase end (slower)
    ritard_window_ql: float = 2.0        # quarter-lengths before a phrase end over which to slow
    use_beat_rubato: bool = True         # False -> fall back to the legacy per-bar tempo walk


def _measure_starts_ql(score) -> list[float]:
    """Sorted unique measure-start offsets in quarter-lengths."""
    starts = set()
    for part in score.parts:
        for m in part.getElementsByClass("Measure"):
            starts.add(float(m.offset))
    if not starts:
        return [0.0]
    return sorted(starts)


def _base_bpm(score) -> float:
    flat = score.flatten()
    base_bpm = 100.0
    for tempo in flat.getElementsByClass("MetronomeMark"):
        try:
            n = float(tempo.number)
            if n > 0:
                base_bpm = n
                break
        except Exception:
            continue
    return base_bpm


def _build_q_to_s_rubato(score, perturb: Perturbations, rng: random.Random,
                         max_ql: float) -> callable:
    """Beat-resolution tempo curve: structured sub-beat rubato + phrase-final ritardando.

    Tempo is sampled on a fine quarter-length grid (beat_grid_ql), each point a smoothed
    random-walk multiplier on the base tempo, with a ramped slow-down (ritardando) over the
    final ritard_window_ql of every phrase (phrase_length_bars long). Seconds are the running
    integral of 60/bpm over the grid, linearly interpolated inside a cell. The map is shared by
    all notes, so notes at one score offset still get one onset -> chord-internal order preserved.
    """
    base_bpm = _base_bpm(score)

    # phrase length in quarter-lengths, from the typical bar size
    bar_starts = _measure_starts_ql(score)
    if len(bar_starts) >= 2:
        diffs = [b - a for a, b in zip(bar_starts[:-1], bar_starts[1:]) if b > a]
        bar_ql = float(np.median(diffs)) if diffs else 4.0
    else:
        bar_ql = 4.0
    phrase_ql = max(bar_ql, perturb.phrase_length_bars * bar_ql)

    # Tempo is a SMOOTH function of musical time: independent multipliers at per-beat
    # KNOTS (a smoothed random walk), linearly interpolated in between. Sub-beat timing
    # (e.g. a triplet inside one beat) then varies only mildly + consistently, while
    # beat-to-beat rubato is full strength -- matching how real rubato actually behaves.
    n_knots = int(math.ceil(max_ql)) + 2
    # mean-reverting AR(1) walk: smooth + autocorrelated like real rubato, and (unlike a
    # zero-padded moving average) free of slow-start/slow-end edge artifacts. rho sets how
    # smooth; the innovation std is chosen so the stationary std == beat_tempo_std.
    rho = 0.6
    innov = perturb.beat_tempo_std * math.sqrt(1.0 - rho * rho)
    knot = np.empty(n_knots)
    prev = 1.0
    for j in range(n_knots):
        prev = 1.0 + rho * (prev - 1.0) + rng.gauss(0.0, innov)
        knot[j] = prev

    # phrase-final ritardando: ramp the knot multiplier toward ritard_factor over the
    # last ritard_window_ql quarter-lengths before each phrase boundary.
    win = max(1.0, perturb.ritard_window_ql)
    for j in range(n_knots):
        q = float(j)
        nb = math.ceil((q + 1e-9) / phrase_ql) * phrase_ql
        dist = nb - q
        if 0.0 <= dist <= win:
            frac = 1.0 - (dist / win)          # 0 at window start -> 1 at the boundary
            knot[j] *= 1.0 + (perturb.ritard_factor - 1.0) * frac
    knot = np.clip(knot, 0.35, 2.2)

    def mult_at(q: float) -> float:
        j = int(q)
        if j >= n_knots - 1:
            return float(knot[-1])
        f = q - j
        return float(knot[j] * (1.0 - f) + knot[j + 1] * f)

    # Integrate 60/bpm on a fine sub-grid so the piecewise-linear tempo gives smooth seconds.
    grid = max(0.0625, perturb.beat_grid_ql)
    n = int(math.ceil((max_ql + grid) / grid)) + 1
    cum = np.zeros(n + 1)
    for k in range(n):
        local_bpm = base_bpm * mult_at((k + 0.5) * grid)   # midpoint rule
        cum[k + 1] = cum[k] + grid * 60.0 / local_bpm

    def q_to_s(q: float) -> float:
        fk = q / grid
        k = int(fk)
        if k >= n:
            k = n - 1
        frac = fk - k
        local_bpm = base_bpm * mult_at((k + 0.5) * grid)
        return float(cum[k] + frac * grid * 60.0 / local_bpm)

    return q_to_s


def _build_q_to_s(score, perturb: Perturbations, rng: random.Random,
                  max_ql: float) -> callable:
    """Quarter-length-to-seconds map. Default: realistic beat-resolution rubato.
    Set perturb.use_beat_rubato=False for the legacy per-bar (deadpan-within-bar) walk."""
    if perturb.use_beat_rubato:
        return _build_q_to_s_rubato(score, perturb, rng, max_ql)
    # ---- legacy per-bar tempo walk (kept for A/B + reproducibility) ----
    base_bpm = _base_bpm(score)

    bar_starts = _measure_starts_ql(score)
    bar_starts = [b for b in bar_starts if b <= max_ql + 0.001]
    if not bar_starts or bar_starts[0] > 0:
        bar_starts = [0.0] + bar_starts
    bar_starts.append(max_ql + 1.0)

    n_bars = len(bar_starts) - 1
    walk = np.array([rng.gauss(1.0, perturb.tempo_walk_std) for _ in range(n_bars)])
    if perturb.tempo_smooth_window > 1:
        kernel = np.ones(perturb.tempo_smooth_window) / perturb.tempo_smooth_window
        walk = np.convolve(walk, kernel, mode="same")
    walk = np.clip(walk, 0.5, 2.0)

    def q_to_s(q: float) -> float:
        # Find which bar this offset falls in
        idx = 0
        for i in range(n_bars):
            if bar_starts[i] <= q < bar_starts[i + 1]:
                idx = i
                break
        else:
            idx = n_bars - 1
        seconds = 0.0
        for i in range(idx):
            bar_q = bar_starts[i + 1] - bar_starts[i]
            local_bpm = base_bpm * walk[i]
            seconds += bar_q * 60.0 / local_bpm
        local_bpm = base_bpm * walk[idx]
        seconds += (q - bar_starts[idx]) * 60.0 / local_bpm
        return seconds

    return q_to_s


def render(mxl_path: Path, out_midi_path: Path, seed: int = 0,
           perturb: Perturbations | None = None) -> dict:
    """Render a MusicXML score to a perturbed performance MIDI.

    Returns:
        {ok: True, n_notes: int, alignment_path: str, n_measures: int}
        {ok: False, error: str}
    """
    if perturb is None:
        perturb = Perturbations()

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from tokenizer import MultistreamTokenizer
            notes_list, score = MultistreamTokenizer.mxl_to_list(str(mxl_path))
    except Exception as exc:
        return {"ok": False, "error": f"parse: {type(exc).__name__}: {exc}"}

    if not notes_list:
        return {"ok": False, "error": "no_notes_in_score"}

    return render_from_parsed(notes_list, score, out_midi_path, seed, perturb)


def render_from_parsed(notes_list, score, out_midi_path: Path, seed: int,
                       perturb: Perturbations) -> dict:
    """Render from an ALREADY-PARSED (notes_list, score) pair.

    The music21 parse (mxl_to_list) dominates render cost, so callers that want many
    rubato realizations of one score should parse it ONCE and call this K times with
    different seeds — each call only rebuilds the (cheap) tempo curve + writes a MIDI.
    """
    try:
        rng = random.Random(seed)
        np_rng = np.random.default_rng(seed)

        # Source-score offsets and pitches in mxl_to_list canonical order
        score_offsets = [float(n.offset) for n in notes_list]
        score_durations = [float(n.duration.quarterLength) for n in notes_list]
        score_pitches = [int(n.pitch.midi) for n in notes_list]
        # source velocity (already realized by mxl_to_list)
        score_velocities = [int(n.volume.velocity or 80) for n in notes_list]

        max_ql = max(o + d for o, d in zip(score_offsets, score_durations))
        q_to_s = _build_q_to_s(score, perturb, rng, max_ql)

        # Per-chord (per-offset) onset jitter so chord-internal order is preserved
        unique_offsets = sorted(set(score_offsets))
        onset_jitters = {
            o: float(np_rng.normal(0, perturb.onset_jitter_seconds))
            for o in unique_offsets
        }

        # Phrase-level loudness envelope
        def phrase_loudness(t_ql: float) -> int:
            phase = (t_ql / (perturb.phrase_length_bars * 4.0)) * 2 * math.pi
            return int(perturb.phrase_loudness_amp * math.cos(phase))

        # Determine BEAT index for each score note (for chunks.json — matches ASAP
        # chunker.py which chunks per beat via midi_score_beats / performance_beats).
        # We approximate "beats" as quarter-note boundaries from offset 0.0 to max_ql.
        # This gives finer-grained crops in training (more random_crop options).
        n_beats = int(math.ceil(max_ql)) + 1

        def beat_idx(offset_ql: float) -> int:
            return min(n_beats - 1, int(offset_ql))

        # We still keep measure_idx around for diagnostics / future use.
        measure_starts = _measure_starts_ql(score)
        if measure_starts and measure_starts[0] > 0:
            measure_starts = [0.0] + measure_starts

        def measure_idx(offset: float) -> int:
            idx = 0
            for i, m_start in enumerate(measure_starts):
                if m_start <= offset:
                    idx = i
                else:
                    break
            return idx

        # Build MIDI in mxl_to_list order
        pm = pretty_midi.PrettyMIDI()
        inst = pretty_midi.Instrument(program=0, is_drum=False, name="Piano")
        alignment_entries = []
        for i, (o_ql, d_ql, p, v) in enumerate(
                zip(score_offsets, score_durations, score_pitches, score_velocities)):
            onset_s = q_to_s(o_ql) + onset_jitters[o_ql]
            offset_end_s = q_to_s(o_ql + d_ql) + onset_jitters[o_ql]
            dur_s = max(0.02, offset_end_s - onset_s)
            dur_s *= float(np_rng.normal(1.0, perturb.duration_scale_std))
            if rng.random() < perturb.early_release_prob:
                dur_s *= perturb.early_release_factor
            dur_s = max(0.02, dur_s)

            vel_offset = int(np_rng.normal(0, perturb.velocity_offset_std))
            new_vel = v + vel_offset + phrase_loudness(o_ql)
            new_vel = int(max(1, min(127, new_vel)))

            inst.notes.append(pretty_midi.Note(
                velocity=new_vel, pitch=int(p),
                start=float(onset_s), end=float(onset_s + dur_s),
            ))
            alignment_entries.append({
                "midi_idx": i,
                "score_idx": i,
                "pitch": int(p),
                "score_offset_ql": o_ql,
                "score_duration_ql": d_ql,
                "measure_idx": measure_idx(o_ql),
                "beat_idx": beat_idx(o_ql),
            })

        pm.instruments.append(inst)

        out_midi_path.parent.mkdir(parents=True, exist_ok=True)
        pm.write(str(out_midi_path))

        alignment_path = out_midi_path.with_suffix(".alignment.json")
        n_meas = max((a["measure_idx"] for a in alignment_entries), default=0) + 1
        n_beat = max((a["beat_idx"] for a in alignment_entries), default=0) + 1
        with alignment_path.open("w") as f:
            json.dump({
                "alignment": alignment_entries,
                "n_score_notes": len(notes_list),
                "n_midi_notes": len(notes_list),
                "n_measures": n_meas,
                "n_beats": n_beat,
                "max_ql": max_ql,
            }, f)

        return {
            "ok": True,
            "n_notes": len(notes_list),
            "alignment_path": str(alignment_path),
            "n_measures": n_meas,
            "n_beats": n_beat,
        }
    except Exception as exc:
        return {"ok": False, "error": f"render: {type(exc).__name__}: {exc}"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mxl", type=Path, help="Input .mxl/.musicxml")
    ap.add_argument("out", type=Path, help="Output .mid")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    result = render(args.mxl, args.out, seed=args.seed)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
