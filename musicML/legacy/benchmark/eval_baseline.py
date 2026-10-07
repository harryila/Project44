"""Baseline metrics on the 3 MAESTRO benchmark pieces (and Twinkle).

Computes the three structural metrics that matter for Phase 2:
    - time_sig_correct : predicted TS matches the published score's TS
    - measures_err_pct : abs(predicted - reference) / reference * 100
    - note_f1          : pitch-multiset F1 against the GT MIDI

These are the same metrics the synthetic pretrain plan will use to compare
{baseline, pretrain, finetune} after step 10. By default this script reads the
existing baseline MusicXML outputs from before the synthetic pretrain (i.e.
the ones produced by the released MIDI2ScoreTF.ckpt with default flags).

Usage:
    venv311/bin/python benchmark/eval_baseline.py
    venv311/bin/python benchmark/eval_baseline.py --tag combined
    venv311/bin/python benchmark/eval_baseline.py --xml-dir /path/with/<piece>.musicxml
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pretty_midi
from music21 import converter

PIECES = {
    "TwinkleTwinkle": {
        "ref_ts": "4/4",
        "ref_measures": 18,
        "gt_midi": "midi/TwinkleTwinkle_hft_eval.mid",
        "default_dirs": [
            "outputs/improvements/twinkle",
            "outputs/transformer/xml",
        ],
    },
    "Op10_No4_CsharpMinor": {
        "ref_ts": "4/4",
        "ref_measures": 88,
        "gt_midi": "benchmark/chopin_op10/midi/Op10_No4_CsharpMinor.midi",
        "default_dirs": ["benchmark/chopin_op10/transformer/xml"],
    },
    "Op25_No11_Aminor": {
        "ref_ts": "4/4",
        "ref_measures": 102,
        "gt_midi": "benchmark/chopin_op25/midi/Op25_No11_Aminor.midi",
        "default_dirs": ["benchmark/chopin_op25/transformer/xml"],
    },
    "Transcendental_No4_Mazeppa": {
        "ref_ts": "4/4",
        "ref_measures": 167,
        "gt_midi": "benchmark/liszt_transcendental/midi/Transcendental_No4_Mazeppa.midi",
        "default_dirs": ["benchmark/liszt_transcendental/transformer/xml"],
    },
}


def first_time_signature(score) -> str | None:
    for ts in score.flatten().getElementsByClass("TimeSignature"):
        return ts.ratioString
    return None


def measure_count(score) -> int:
    counts = []
    for part in score.parts:
        counts.append(len(list(part.getElementsByClass("Measure"))))
    return max(counts) if counts else 0


def score_pitches(score) -> list[int]:
    out = []
    for n in score.flatten().notes:
        if n.isChord:
            out.extend(int(p.midi) for p in n.pitches)
        else:
            out.append(int(n.pitch.midi))
    return out


def midi_pitches(path: Path) -> list[int]:
    pm = pretty_midi.PrettyMIDI(str(path))
    return [int(n.pitch) for inst in pm.instruments for n in inst.notes]


def pitch_multiset_f1(pred_pitches: list[int], gt_pitches: list[int]) -> float:
    cp = Counter(pred_pitches)
    cg = Counter(gt_pitches)
    tp = sum((cp & cg).values())
    fp = sum((cp - cg).values())
    fn = sum((cg - cp).values())
    if tp == 0:
        return 0.0
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    return 2 * prec * rec / (prec + rec)


def find_xml(piece: str, tag: str | None, dirs: list[str]) -> Path | None:
    name = f"{piece}.musicxml" if not tag else f"{piece}_{tag}.musicxml"
    for d in dirs:
        candidate = Path(d) / name
        if candidate.is_file():
            return candidate
    return None


def evaluate_piece(piece: str, xml: Path, info: dict) -> dict:
    score = converter.parse(str(xml), forceSource=True)
    pred_ts = first_time_signature(score) or "?"
    pred_meas = measure_count(score)
    pred_pitches = score_pitches(score)
    gt_pitches = midi_pitches(Path(info["gt_midi"]))
    return {
        "xml": str(xml),
        "ref_ts": info["ref_ts"],
        "ref_measures": info["ref_measures"],
        "time_sig": pred_ts,
        "time_sig_correct": pred_ts == info["ref_ts"],
        "measures": pred_meas,
        "measures_err_pct": round(100 * abs(pred_meas - info["ref_measures"]) / info["ref_measures"], 1),
        "note_f1": round(pitch_multiset_f1(pred_pitches, gt_pitches), 4),
        "n_pred_notes": len(pred_pitches),
        "n_gt_notes": len(gt_pitches),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=None,
                    help="Suffix tag, e.g. 'combined', 'pretrain', 'finetune'. Default: no suffix (= baseline).")
    ap.add_argument("--xml-dir", default=None,
                    help="Override XML search dir. If omitted, uses default per-piece dirs.")
    ap.add_argument("--out", type=Path, default=Path("benchmark/baseline_metrics.json"))
    args = ap.parse_args()

    results: dict = {}
    summary_rows = []

    for piece, info in PIECES.items():
        gt = Path(info["gt_midi"])
        if not gt.is_file():
            print(f"[skip] {piece}: GT MIDI missing at {gt}", file=sys.stderr)
            continue
        dirs = [args.xml_dir] if args.xml_dir else info["default_dirs"]
        xml = find_xml(piece, args.tag, dirs)
        if xml is None:
            print(f"[skip] {piece}: XML not found in {dirs} for tag={args.tag!r}", file=sys.stderr)
            results[piece] = None
            continue
        try:
            r = evaluate_piece(piece, xml, info)
            results[piece] = r
            summary_rows.append((piece, r))
        except Exception as exc:
            results[piece] = {"error": str(exc), "xml": str(xml)}
            print(f"[fail] {piece}: {exc}", file=sys.stderr)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))

    print(f"\n{'Piece':<32} {'TS':<6} {'OK':<3} {'Meas':<5} {'%err':<6} {'F1':<6} {'n_pred':<7} {'n_gt':<6}")
    print("-" * 80)
    for piece, r in summary_rows:
        print(
            f"{piece:<32} {r['time_sig']:<6} {('YES' if r['time_sig_correct'] else 'NO'):<3} "
            f"{r['measures']:<5} {r['measures_err_pct']:<6} {r['note_f1']:<6} "
            f"{r['n_pred_notes']:<7} {r['n_gt_notes']:<6}"
        )
        print(f"{'  (ref)':<32} {r['ref_ts']:<6} {'-':<3} {r['ref_measures']:<5}")
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
