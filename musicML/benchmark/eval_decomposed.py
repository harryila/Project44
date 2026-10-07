"""Decomposed audio→score eval: separate the audio→MIDI error (transcription)
from the MIDI→score error (engraving), so we know WHICH stage fails on each piece.

Stage A (audio→MIDI, transcription):  hFT on the audio, scored vs GT performance
    MIDI with mir_eval note-F1 (onset-only @50ms, onset+offset, +velocity).
Stage B (MIDI→score, engraving):      MIDI2ScoreTransformer on the GT MIDI, scored
    vs GT score with MUSTER. (This is Tier-1 on the benchmark pieces.)

Why: end-to-end MUSTER conflates the two. The Mazeppa investigation (2026-06-01)
proved the failure is Stage B (score model on OOD), not Stage A (hFT recovers 92%) —
this script makes that decomposition reproducible and extends it to all pieces.

Usage: venv311/bin/python benchmark/eval_decomposed.py --out benchmark/decomposed.json
"""
import argparse, json, os, sys, time, warnings
from pathlib import Path
warnings.simplefilter("ignore")

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
from musicml_paths import REPO_ROOT as _ROOT
REPO = Path(_ROOT)
TF = REPO / "MIDI2ScoreTransformer" / "midi2scoretransformer"
sys.path.insert(0, str(TF))
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import pretty_midi  # noqa: E402
import mir_eval  # noqa: E402

ASAP = REPO / "MIDI2ScoreTransformer/data/asap-dataset"

# name, audio, GT performance MIDI, GT engraved score
PIECES = [
    {"name": "Chopin_Op10_No4", "dist": "in-dist",
     "audio": REPO / "benchmark/chopin_op10/audio/Op10_No4_CsharpMinor.wav",
     "gt_midi": REPO / "benchmark/chopin_op10/midi/Op10_No4_CsharpMinor.midi",
     "gt_score": ASAP / "Chopin/Etudes_op_10/4/xml_score.musicxml"},
    {"name": "Chopin_Op25_No11", "dist": "in-dist",
     "audio": REPO / "benchmark/chopin_op25/audio/Op25_No11_Aminor.wav",
     "gt_midi": REPO / "benchmark/chopin_op25/midi/Op25_No11_Aminor.midi",
     "gt_score": ASAP / "Chopin/Etudes_op_25/11/xml_score.musicxml"},
    {"name": "Liszt_Mazeppa", "dist": "dense (in ASAP train)",
     "audio": REPO / "benchmark/liszt_transcendental/audio/Transcendental_No4_Mazeppa.wav",
     "gt_midi": REPO / "benchmark/liszt_transcendental/midi/Transcendental_No4_Mazeppa.midi",
     # Use the ASAP edition (the one the model trained on). The PDMX community score
     # (benchmark/liszt_transcendental/gt_score.musicxml) is a DIFFERENT engraving
     # edition that inflated MUSTER ~2x (34 vs ~14) via edition mismatch, not model error.
     "gt_score": ASAP / "Liszt/Transcendental_Etudes/4/xml_score.musicxml"},
]


def midi_intervals_pitches(midi_path):
    """(N,2) onset/offset intervals + (N,) pitch-Hz, for mir_eval."""
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    notes = [n for ins in pm.instruments for n in ins.notes]
    if not notes:
        return np.zeros((0, 2)), np.zeros(0)
    intervals = np.array([[n.start, n.end] for n in notes])
    pitches = np.array([pretty_midi.note_number_to_hz(n.pitch) for n in notes])
    return intervals, pitches


def transcribe_hft(audio, out_midi):
    """Run the production hFT path; returns the note count and writes a MIDI."""
    import transcribe
    import logging
    logging.getLogger().setLevel(logging.ERROR)
    n, raw = transcribe.audio_to_midi_hft(Path(audio), Path(out_midi))
    return n


def stageA_transcription(audio, gt_midi, work):
    """hFT audio→MIDI scored vs GT MIDI with mir_eval note-F1 at 3 strictnesses."""
    hft_midi = work / "hft.mid"
    n = transcribe_hft(audio, hft_midi)
    ref_i, ref_p = midi_intervals_pitches(gt_midi)
    est_i, est_p = midi_intervals_pitches(hft_midi)
    out = {"hft_notes": int(n), "gt_notes": int(len(ref_p))}
    # onset-only F1 (50ms), the standard MAESTRO metric
    p, r, f, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ref_i, ref_p, est_i, est_p, onset_tolerance=0.05, offset_ratio=None)
    out["onset_F1"] = round(f, 4); out["onset_P"] = round(p, 4); out["onset_R"] = round(r, 4)
    # onset+offset F1
    p2, r2, f2, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ref_i, ref_p, est_i, est_p, onset_tolerance=0.05, offset_ratio=0.2)
    out["onset_offset_F1"] = round(f2, 4)
    return out


def stageB_engraving(gt_midi, gt_score, model):
    """MIDI2ScoreTransformer on GT MIDI scored vs GT score with MUSTER (Tier-1)."""
    from tokenizer import MultistreamTokenizer
    from utils import infer, eval as eval_pair
    x = MultistreamTokenizer.tokenize_midi(str(gt_midi))
    y = infer(x, model, overlap=64, chunk=512, verbose=False, kv_cache=True)
    sim = eval_pair(y, str(gt_score))
    m = sim.get("muster") or {}
    return {k: (round(m[k], 2) if isinstance(m.get(k), (int, float)) else None)
            for k in ("MeanER", "PitchER", "MissRate", "ExtraRate", "OnsetER", "OffsetER")}


def load_model():
    import torch
    from config import MyModelConfig
    if not hasattr(MyModelConfig, "_attn_implementation_internal"):
        MyModelConfig._attn_implementation_internal = None
    torch.serialization.add_safe_globals([MyModelConfig])
    from models.roformer import Roformer
    os.chdir(TF.parent)  # so checkpoints/ resolves
    m = Roformer.load_from_checkpoint("checkpoints/MIDI2ScoreTF.ckpt",
                                      map_location="cpu", weights_only=False)
    m.eval()
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="benchmark/decomposed.json")
    ap.add_argument("--stage", choices=["A", "B", "both"], default="both")
    args = ap.parse_args()
    out_path = REPO / args.out if not Path(args.out).is_absolute() else Path(args.out)
    work = REPO / "benchmark" / "decomp_work"
    work.mkdir(parents=True, exist_ok=True)

    model = load_model() if args.stage in ("B", "both") else None
    results = []
    t0 = time.time()
    for p in PIECES:
        rec = {"name": p["name"], "dist": p["dist"]}
        pw = work / p["name"]; pw.mkdir(exist_ok=True)
        try:
            if args.stage in ("A", "both"):
                rec["stageA_transcription"] = stageA_transcription(p["audio"], p["gt_midi"], pw)
                print(f"[A] {p['name']:18s} onsetF1={rec['stageA_transcription']['onset_F1']} "
                      f"(notes {rec['stageA_transcription']['hft_notes']}/{rec['stageA_transcription']['gt_notes']})", flush=True)
            if args.stage in ("B", "both"):
                rec["stageB_engraving"] = stageB_engraving(p["gt_midi"], p["gt_score"], model)
                print(f"[B] {p['name']:18s} MUSTER MeanER={rec['stageB_engraving']['MeanER']}", flush=True)
        except Exception as e:
            import traceback
            rec["error"] = traceback.format_exc().splitlines()[-1]
            print(f"    {p['name']} ERROR: {rec['error']}", flush=True)
        results.append(rec)
        json.dump({"meta": {"elapsed_s": round(time.time()-t0, 1)}, "pieces": results},
                  open(out_path, "w"), indent=2)

    print("\n" + "=" * 78)
    print("DECOMPOSED: Stage A (transcription, note-F1) vs Stage B (engraving, MUSTER)")
    print("=" * 78)
    print(f"{'piece':18s} {'dist':10s} {'A:onsetF1':>10s} {'A:on+offF1':>11s} {'B:MUSTER':>9s}")
    for r in results:
        a = r.get("stageA_transcription", {}); b = r.get("stageB_engraving", {})
        print(f"{r['name']:18s} {r['dist']:10s} {a.get('onset_F1','-'):>10} "
              f"{a.get('onset_offset_F1','-'):>11} {b.get('MeanER','-'):>9}")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
