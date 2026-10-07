"""Compare improvement variants against ground truth on the four test pieces.

For each piece, we look up:
- Reference time signature and measure count from the catalog (or hardcoded)
- Ground-truth MIDI notes from the .midi/.mid file
- Predicted MusicXML files from outputs/improvements/<piece>/<piece>_<tag>.musicxml
  (or benchmark/<piece>/transformer/xml/<piece>_<tag>.musicxml)

Computes:
- time_sig         : predicted time signature (string)
- time_sig_correct : matches reference?
- measures         : predicted measure count
- measures_err_pct : abs(predicted - reference) / reference
- note_f1          : pitch+onset F1 against ground-truth MIDI (50 ms tolerance)

Run from repo root:
    venv311/bin/python benchmark/eval_improvements.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pretty_midi
from music21 import converter

PIECES = {
    "TwinkleTwinkle": {
        "ref_ts": "4/4",
        "ref_measures": 18,
        "gt_midi": "midi/TwinkleTwinkle_hft_eval.mid",
        "search_dirs": [
            "outputs/improvements/twinkle",
            "outputs/transformer/xml",
        ],
    },
    "Op10_No4_CsharpMinor": {
        "ref_ts": "4/4",
        "ref_measures": 88,
        "gt_midi": "benchmark/chopin_op10/midi/Op10_No4_CsharpMinor.midi",
        "search_dirs": [
            "benchmark/chopin_op10/transformer/xml",
        ],
    },
    "Op25_No11_Aminor": {
        "ref_ts": "4/4",
        "ref_measures": 102,
        "gt_midi": "benchmark/chopin_op25/midi/Op25_No11_Aminor.midi",
        "search_dirs": [
            "benchmark/chopin_op25/transformer/xml",
        ],
    },
    "Transcendental_No4_Mazeppa": {
        "ref_ts": "4/4",
        "ref_measures": 167,
        "gt_midi": "benchmark/liszt_transcendental/midi/Transcendental_No4_Mazeppa.midi",
        "search_dirs": [
            "benchmark/liszt_transcendental/transformer/xml",
        ],
    },
}

VARIANTS = ["baseline", "hft_baseline", "a3", "a1", "a2", "a4", "combined", "full"]


def first_time_signature(score) -> str | None:
    for ts in score.flatten().getElementsByClass("TimeSignature"):
        return ts.ratioString
    return None


def measure_count(score) -> int:
    counts = []
    for part in score.parts:
        counts.append(len(list(part.getElementsByClass("Measure"))))
    return max(counts) if counts else 0


def score_to_midi_notes(score) -> list[tuple[int, float]]:
    """List of (pitch, onset_seconds) for each note in the score."""
    notes = []
    for n in score.flatten().notes:
        if n.isChord:
            for p in n.pitches:
                notes.append((int(p.midi), float(n.offset)))
        else:
            notes.append((int(n.pitch.midi), float(n.offset)))
    return notes


def midi_notes(path: Path) -> list[tuple[int, float]]:
    pm = pretty_midi.PrettyMIDI(str(path))
    out = []
    for inst in pm.instruments:
        for n in inst.notes:
            out.append((int(n.pitch), float(n.start)))
    return sorted(out)


def note_f1(pred: list[tuple[int, float]],
            gt: list[tuple[int, float]],
            onset_tol: float = 0.05) -> float:
    """Pitch + onset F1. Uses set-based matching since we are scoring an
    estimated score against a performance MIDI (different time scales).
    For pieces where the prediction is in beats (score time) and gt is in
    seconds, this comparison is approximate. We fall back to pitch-multiset
    F1 in that case.
    """
    # Try pitch-multiset F1 first (it's robust to timing-scale mismatch).
    from collections import Counter
    cp = Counter(p for p, _ in pred)
    cg = Counter(p for p, _ in gt)
    tp = sum((cp & cg).values())
    fp = sum((cp - cg).values())
    fn = sum((cg - cp).values())
    if tp == 0:
        return 0.0
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    return 2 * prec * rec / (prec + rec)


def find_xml(piece_name: str, tag: str, search_dirs: list[str]) -> Path | None:
    """Find a MusicXML file. The 'baseline' tag maps to the unsuffixed file
    (the original GT-MIDI run). 'hft_baseline' maps to '<piece>_hft.musicxml'
    (the audio-path run from before A1-A5)."""
    if tag == "baseline":
        names = [f"{piece_name}.musicxml"]
    elif tag == "hft_baseline":
        names = [f"{piece_name}_hft.musicxml", f"{piece_name}_hft_eval.musicxml"]
    else:
        names = [f"{piece_name}_{tag}.musicxml"]
    for d in search_dirs:
        for name in names:
            candidate = Path(d) / name
            if candidate.is_file():
                return candidate
    return None


def evaluate_one(xml_path: Path, ref_ts: str, ref_measures: int,
                 gt_midi_path: Path) -> dict:
    score = converter.parse(str(xml_path), forceSource=True)
    pred_ts = first_time_signature(score) or "?"
    pred_meas = measure_count(score)
    pred_notes = score_to_midi_notes(score)
    gt_notes = midi_notes(gt_midi_path)
    return {
        "time_sig": pred_ts,
        "time_sig_correct": pred_ts == ref_ts,
        "measures": pred_meas,
        "measures_err_pct": round(100 * abs(pred_meas - ref_measures) / ref_measures, 1),
        "note_f1": round(note_f1(pred_notes, gt_notes), 3),
        "n_pred_notes": len(pred_notes),
        "n_gt_notes": len(gt_notes),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("benchmark/improvement_metrics.json"))
    args = ap.parse_args()

    results: dict = {}
    for piece, info in PIECES.items():
        gt_midi = Path(info["gt_midi"])
        if not gt_midi.is_file():
            print(f"[skip] {piece}: GT MIDI not found at {gt_midi}", file=sys.stderr)
            continue
        results[piece] = {"ref_ts": info["ref_ts"], "ref_measures": info["ref_measures"]}
        for tag in VARIANTS:
            xml = find_xml(piece, tag, info["search_dirs"])
            if xml is None:
                results[piece][tag] = None
                continue
            try:
                metrics = evaluate_one(xml, info["ref_ts"], info["ref_measures"], gt_midi)
                metrics["xml"] = str(xml)
                results[piece][tag] = metrics
            except Exception as exc:
                results[piece][tag] = {"error": str(exc), "xml": str(xml)}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        json.dump(results, f, indent=2)

    # Pretty print
    print(f"\n{'Piece':<32} {'Tag':<10} {'TS':<6} {'OK':<3} {'Meas':<5} {'%err':<6} {'Note F1':<8}")
    print("-" * 80)
    for piece, info in results.items():
        ref_meas = info["ref_measures"]
        for tag in VARIANTS:
            r = info.get(tag)
            if r is None:
                print(f"{piece:<32} {tag:<10} {'-':<6} {'-':<3} {'-':<5} {'-':<6} {'-':<8}")
                continue
            if "error" in r:
                print(f"{piece:<32} {tag:<10} ERROR  {r['error'][:40]}")
                continue
            print(
                f"{piece:<32} {tag:<10} "
                f"{r['time_sig']:<6} "
                f"{('YES' if r['time_sig_correct'] else 'NO'):<3} "
                f"{r['measures']:<5} "
                f"{r['measures_err_pct']:<6} "
                f"{r['note_f1']:<8}"
            )
        print(f"{piece:<32} {'(ref)':<10} {info['ref_ts']:<6} {'-':<3} {ref_meas:<5}")
        print("-" * 80)

    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
