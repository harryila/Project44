"""Tier-1 accuracy eval: MIDI -> score, scored with MUSTER + score_similarity
over the ASAP held-out test split (the field-standard, paper-comparable metrics).

This is the headline eval from ACCURACY_ROADMAP.md Track A. It runs any checkpoint
(default: the released MIDI2ScoreTF.ckpt) on the 14-piece / 59-performance ASAP
test split and reports the same numbers the Beyer ISMIR-2024 paper reports
(PitchER / MissRate / ExtraRate / OnsetER / OffsetER / MeanER, plus the notation
error rates), so we can anchor every future change to a real baseline.

Forces CPU: MPS produces degenerate pad logits on this model (documented in
PROJECT_REPORT.md), so the released checkpoint must run on CPU here.

Usage:
    venv311/bin/python benchmark/eval_tier1_asap.py \
        --ckpt MIDI2ScoreTransformer/checkpoints/MIDI2ScoreTF.ckpt \
        --out benchmark/tier1_baseline.json
    # smoke test on the 3 shortest performances:
    ...  --limit 3
"""
import argparse
import json
import os
import sys
import time
import traceback
import warnings
from pathlib import Path

warnings.simplefilter("ignore")

REPO_ROOT = Path(__file__).resolve().parent.parent
TF_ROOT = REPO_ROOT / "MIDI2ScoreTransformer"
sys.path.insert(0, str(TF_ROOT / "midi2scoretransformer"))
# ASAPDataset reads "./data/..." relatively, and {ASAP} paths resolve under cwd.
os.chdir(TF_ROOT)

import torch  # noqa: E402

# Same checkpoint-load incantation the production CLI uses (PyTorch 2.6 safe-globals).
from config import MyModelConfig  # noqa: E402
if not hasattr(MyModelConfig, "_attn_implementation_internal"):
    MyModelConfig._attn_implementation_internal = None
torch.serialization.add_safe_globals([MyModelConfig])

from dataset import ASAPDataset  # noqa: E402
from models.roformer import Roformer  # noqa: E402
from tokenizer import MultistreamTokenizer  # noqa: E402
from utils import infer, eval as eval_pair  # noqa: E402


def load_any_checkpoint(ckpt_path, device):
    """Load either the released base Roformer or a fine-tuned TrainableRoformer
    checkpoint (train.py saves the latter). Tries base first, falls back."""
    # strict=False so distillation checkpoints (which carry frozen `_teacher.*` keys)
    # load the student cleanly, ignoring the extra teacher weights.
    try:
        return Roformer.load_from_checkpoint(ckpt_path, map_location=device,
                                             weights_only=False, strict=False)
    except Exception:
        from train import TrainableRoformer  # noqa: E402
        return TrainableRoformer.load_from_checkpoint(ckpt_path, map_location=device,
                                                      weights_only=False, strict=False)


def _score_worker(task):
    """CPU-only MUSTER/score_similarity scoring of one inferred score. Picklable,
    fork-safe (no CUDA touched here). Returns (results_index, sim, error)."""
    idx, y_hat, score_path = task
    try:
        return idx, eval_pair(y_hat, score_path), None
    except Exception:
        return idx, None, traceback.format_exc().splitlines()[-1]


def _annotation_key(midi_path: str) -> str:
    """Map a './data/asap-dataset/...' perf-MIDI path to its asap_annotations.json key."""
    rel = midi_path
    for pre in ("./data/asap-dataset/", "data/asap-dataset/"):
        if rel.startswith(pre):
            return rel[len(pre):]
    return rel


def tokenize_midi_with_beats(midi_path: str, annotations: dict):
    """Tokenize perf-MIDI WITH ASAP gold beats threaded in (mirrors dataset.py's
    beat-conditioning path): parse -> per-note phase-within-beat -> bucket_midi emits
    the real 'beat' one-hot. Returns (x, used_beats). If the piece has no usable
    beats, falls back to the all-no-beat stream (used_beats=False) so it never crashes.
    """
    from beat_features import phase_features
    streams = MultistreamTokenizer.parse_midi(midi_path)
    onsets = streams["onset"].numpy()
    ann = annotations.get(_annotation_key(midi_path))
    used_beats = False
    if ann and ann.get("performance_beats") and len(ann["performance_beats"]) >= 2:
        beats = ann["performance_beats"]
        ph = phase_features(onsets, beats, downbeats=None)[:, 0]
        streams["beat_phase"] = torch.from_numpy(ph)
        valid = (onsets >= beats[0]) & (onsets <= beats[-1])
        streams["beat_valid"] = torch.from_numpy(valid)
        used_beats = True
    x = MultistreamTokenizer.bucket_midi(streams)
    return x, used_beats

# Metrics we headline (match run_eval.py / the paper's "Ours" row).
MUSTER_KEYS = ["PitchER", "MissRate", "ExtraRate", "OnsetER", "OffsetER", "MeanER"]
NOTATION_KEYS = ["NoteDeletion", "NoteInsertion", "NoteDuration",
                 "StaffAssignment", "StemDirection", "NoteSpelling"]
PAPER_SOTA = {  # Beyer & Dai 2024, MUSTER (lower=better)
    "PitchER": 3.11, "MissRate": 7.56, "ExtraRate": 6.44,
    "OnsetER": 15.55, "OffsetER": 23.84, "MeanER": 11.30,
}


def collect_paths(split: str):
    q = ASAPDataset("./data/", split)
    md = q.metadata
    paths = []
    for i in range(len(md)):
        s = md.iloc[i]
        midi = s["performance_MIDI_external"].replace("{ASAP}", "./data/asap-dataset")
        score = os.path.dirname(midi) + "/xml_score.musicxml"
        rel = os.path.relpath(os.path.dirname(midi), "./data/asap-dataset")
        paths.append({"composer": s["composer"], "piece": rel,
                      "piece_id": int(s["piece_id"]), "midi": midi, "score": score})
    return paths


def aggregate(results):
    """Mirror run_eval.py aggregation: mean over non-None values per sub-metric."""
    done = [r for r in results if r.get("sim")]
    out = {"n_scored": len(done), "n_total": len(results)}
    for group in ("muster", "mxl <-> gt_mxl"):
        vals = {}
        for r in done:
            g = r["sim"].get(group) or {}
            for k, v in g.items():
                if isinstance(v, (int, float)):
                    vals.setdefault(k, []).append(v)
        out[group] = {k: (sum(v) / len(v) if v else None) for k, v in vals.items()}
    return out


def print_summary(agg):
    m = agg.get("muster", {})
    n = agg.get("mxl <-> gt_mxl", {})
    print("\n" + "=" * 72)
    print(f"TIER-1 (MIDI->score) on ASAP test split — {agg['n_scored']}/{agg['n_total']} scored")
    print("=" * 72)
    print("MUSTER (lower=better):       " + "  ".join(f"{k}" for k in MUSTER_KEYS))
    print("  Ours      ", "  ".join(f"{(m.get(k) or float('nan')):6.2f}" for k in MUSTER_KEYS))
    print("  Paper SOTA", "  ".join(f"{PAPER_SOTA[k]:6.2f}" for k in MUSTER_KEYS))
    print("\nNotation error rates (x100, lower=better):")
    for k in NOTATION_KEYS:
        v = n.get(k)
        print(f"  {k:16s} {100*v:6.2f}" if isinstance(v, (int, float)) else f"  {k:16s}   n/a")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/MIDI2ScoreTF.ckpt")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None,
                    help="only the N shortest performances (smoke test)")
    ap.add_argument("--overlap", type=int, default=64)   # match run_eval/paper
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--device", default="cpu",
                    help="cpu (default; required on Mac — MPS breaks this model) "
                         "or cuda (use on a GPU box for ~10x faster eval).")
    ap.add_argument("--use-beat-conditioning", action="store_true",
                    help="Thread ASAP gold beats into the input (per-note phase-within-beat). "
                         "Use ONLY with a beat-conditioned checkpoint; this is the gold-beat "
                         "CEILING eval (a real tracker would be the deployable number).")
    ap.add_argument("--jobs", type=int, default=1,
                    help="Parallel MUSTER scoring workers. Inference stays sequential on the "
                         "GPU (main proc); the slow CPU score_similarity/muster step is fanned "
                         "out across N forked workers (no CUDA in workers). 1 = serial (default).")
    args = ap.parse_args()

    annotations = None
    if args.use_beat_conditioning:
        annotations = json.load(open("./data/asap-dataset/asap_annotations.json"))
        print(f"Beat-conditioning ON: gold beats from asap_annotations.json "
              f"({len(annotations)} pieces).", flush=True)

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    device = args.device
    paths = collect_paths(args.split)

    # tokenize up front so we can sort by length (cheap; lets --limit pick short ones)
    print(f"Tokenizing {len(paths)} performances ...", flush=True)
    n_with_beats = 0
    for p in paths:
        try:
            if args.use_beat_conditioning:
                x, used = tokenize_midi_with_beats(p["midi"], annotations)
                p["used_beats"] = used
                n_with_beats += int(used)
            else:
                x = MultistreamTokenizer.tokenize_midi(p["midi"])
            p["_x"] = x
            p["n_notes"] = int(x["pitch"].shape[0])
        except Exception as e:
            p["_x"] = None
            p["n_notes"] = 1 << 30
            p["tok_error"] = str(e)
    if args.use_beat_conditioning:
        print(f"  gold beats applied to {n_with_beats}/{len(paths)} performances "
              f"(others fell back to no-beat).", flush=True)
    paths.sort(key=lambda p: p["n_notes"])
    if args.limit:
        paths = paths[:args.limit]

    print(f"Loading checkpoint {args.ckpt} on CPU ...", flush=True)
    model = load_any_checkpoint(args.ckpt, device)
    model.eval()
    model.to(device)

    results = []
    meta = {"ckpt": str(args.ckpt), "split": args.split, "overlap": args.overlap,
            "chunk": args.chunk, "device": device, "n_performances": len(paths),
            "use_beat_conditioning": bool(args.use_beat_conditioning),
            "n_with_gold_beats": n_with_beats if args.use_beat_conditioning else 0}
    t0 = time.time()

    def save():
        json.dump({"meta": {**meta, "elapsed_s": round(time.time() - t0, 1)},
                   "aggregate": aggregate(results), "per_performance": results},
                  open(out_path, "w"), indent=2)

    if args.jobs > 1:
        # Phase 1: GPU inference (sequential, main proc) -> y_hat per piece.
        score_tasks = []
        for i, p in enumerate(paths):
            rec = {k: p[k] for k in ("composer", "piece", "piece_id", "midi", "score", "n_notes")}
            results.append(rec)
            if p["_x"] is None:
                rec["error"] = "tokenize: " + p.get("tok_error", "?")
                continue
            try:
                ti = time.time()
                y_hat = infer(p["_x"], model, overlap=args.overlap, chunk=args.chunk,
                              verbose=False, kv_cache=True)
                rec["infer_s"] = round(time.time() - ti, 1)
                y_hat = {k: (v.cpu() if hasattr(v, "cpu") else v) for k, v in y_hat.items()}
                score_tasks.append((len(results) - 1, y_hat, p["score"]))
                print(f"infer [{i+1}/{len(paths)}] {p['composer']:11s} {p['piece'][:30]:30s} "
                      f"notes={p['n_notes']:5d} ({rec['infer_s']}s)", flush=True)
            except Exception:
                rec["error"] = traceback.format_exc().splitlines()[-1]
                print(f"infer [{i+1}/{len(paths)}] ERROR {rec['error']}", flush=True)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
        # Phase 2: parallel CPU MUSTER scoring across --jobs workers. Use 'spawn', NOT
        # 'fork': the main proc has initialized CUDA for inference, and forking after
        # CUDA-init deadlocks the (CPU-only) workers because copied library locks are
        # held. Spawn starts clean interpreters (no copied state) — the standard fix.
        from multiprocessing import get_context
        print(f"Scoring {len(score_tasks)} performances with {args.jobs} workers ...", flush=True)
        done = 0
        with get_context("spawn").Pool(args.jobs) as pool:
            for idx, sim, err in pool.imap_unordered(_score_worker, score_tasks):
                if err:
                    results[idx]["error"] = err
                else:
                    results[idx]["sim"] = sim
                done += 1
                mer = (sim or {}).get("muster", {}).get("MeanER") if sim else None
                print(f"  scored {done}/{len(score_tasks)}  {results[idx]['composer']}/"
                      f"{results[idx]['piece'][:24]}  MeanER="
                      f"{mer if mer is None else round(mer,2)}", flush=True)
                save()
    else:
        for i, p in enumerate(paths):
            rec = {k: p[k] for k in ("composer", "piece", "piece_id", "midi", "score", "n_notes")}
            if p["_x"] is None:
                rec["error"] = "tokenize: " + p.get("tok_error", "?")
                results.append(rec)
                continue
            try:
                ti = time.time()
                y_hat = infer(p["_x"], model, overlap=args.overlap, chunk=args.chunk,
                              verbose=False, kv_cache=True)
                sim = eval_pair(y_hat, p["score"])
                rec["sim"] = sim
                rec["infer_s"] = round(time.time() - ti, 1)
                mer = (sim.get("muster") or {}).get("MeanER")
                print(f"[{i+1}/{len(paths)}] {p['composer']:11s} {p['piece'][:34]:34s} "
                      f"notes={p['n_notes']:5d} MeanER={mer if mer is None else round(mer,2)} "
                      f"({rec['infer_s']}s)", flush=True)
            except Exception:
                rec["error"] = traceback.format_exc().splitlines()[-1]
                print(f"[{i+1}/{len(paths)}] {p['composer']:11s} {p['piece'][:34]:34s} ERROR: {rec['error']}", flush=True)
            results.append(rec)
            save()  # incremental save so partial results survive an interrupted run

    agg = aggregate(results)
    json.dump({"meta": {**meta, "elapsed_s": round(time.time() - t0, 1)},
               "aggregate": agg, "per_performance": results},
              open(out_path, "w"), indent=2)
    print_summary(agg)
    print(f"\nWrote {out_path}  ({round(time.time()-t0,1)}s total)")


if __name__ == "__main__":
    main()
