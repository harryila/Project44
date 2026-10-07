# HISTORICAL: built MAESTRO-derived corpora on the 2026 GPU box (/root/datasets paths). MAESTRO v3 and
# those corpora are not part of this repository (see DATA.md, known missing); defaults are the box paths.
"""Route A: build PAIRED (real MAESTRO timing -> released-model score) distillation data, ALIGNMENT-FREE.

Each MAESTRO performance is split into short note-windows (re-zeroed to t=0), and each window is
pseudo-labeled INDEPENDENTLY by the released model. Because each window is short enough to fit in one
sequence, it trains as a SINGLE CHUNK -> no per-note MIDI<->score alignment is needed (the released
model's output is NOT 1:1, so this sidesteps that entirely). The result is real-performance-timing
paired data carrying the released model's (correct) tuplet placement.

Sharded for two GPUs: --shard i --nshards N.
Out: <out-dir>/{id}.mid (window), {id}.musicxml (pseudo score), {id}_chunks.json (single chunk)
     + a per-shard manifest CSV (merge after).
"""
import argparse, glob, json, os, sys, warnings
from pathlib import Path
warnings.simplefilter("ignore")
REPO = Path(__file__).resolve().parent.parent
TF = REPO / "MIDI2ScoreTransformer"
sys.path.insert(0, str(TF / "midi2scoretransformer")); sys.path.insert(0, str(REPO / "benchmark"))
import csv
import pretty_midi
import torch
from config import MyModelConfig
if not hasattr(MyModelConfig, "_attn_implementation_internal"):
    MyModelConfig._attn_implementation_internal = None
torch.serialization.add_safe_globals([MyModelConfig])
from tokenizer import MultistreamTokenizer
from utils import infer
from score_utils import postprocess_score
from eval_tier1_asap import load_any_checkpoint


def windows_of(midi_path, win):
    pm = pretty_midi.PrettyMIDI(midi_path)
    notes = []
    for inst in pm.instruments:
        if inst.is_drum:
            continue
        notes.extend(inst.notes)
    notes.sort(key=lambda n: (n.start, n.pitch))
    for k in range(0, len(notes), win):
        chunk = notes[k:k + win]
        if len(chunk) < 32:
            continue
        t0 = min(n.start for n in chunk)
        seg = pretty_midi.PrettyMIDI()
        inst = pretty_midi.Instrument(program=0)
        for n in chunk:
            inst.notes.append(pretty_midi.Note(velocity=n.velocity, pitch=n.pitch,
                                               start=max(0.0, n.start - t0), end=max(0.01, n.end - t0)))
        seg.instruments.append(inst)
        yield k // win, seg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--maestro-root", default="/root/datasets/maestro-v3.0.0")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--win", type=int, default=384, help="notes per window")
    ap.add_argument("--seq-length", type=int, default=512)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--out-dir", default="/root/datasets/maestro_segments")
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    manifest = args.manifest or f"{args.out_dir}/manifest_shard{args.shard}.csv"
    midis = sorted(glob.glob(os.path.join(args.maestro_root, "*", "*.midi")), key=os.path.getsize)[:args.n]
    mine = midis[args.shard::args.nshards]
    model = load_any_checkpoint(args.ckpt, args.device); model.eval(); model.to(args.device)
    print(f"shard {args.shard}/{args.nshards}: {len(mine)} pieces, win={args.win}", flush=True)

    rows = []; ok = fail = skip = 0
    for pi, mp in enumerate(mine):
        stem = Path(mp).stem[:40]
        for wk, seg in windows_of(mp, args.win):
            pid = f"s{args.shard}_p{pi}_w{wk}"
            mid_path = os.path.join(args.out_dir, f"{pid}.mid")
            mxl_path = os.path.join(args.out_dir, f"{pid}.musicxml")
            ch_path = os.path.join(args.out_dir, f"{pid}_chunks.json")
            if os.path.exists(mxl_path) and os.path.exists(mid_path) and os.path.exists(ch_path):
                ok += 1
                rows.append({"id": pid, "midi": mid_path, "mxl": mxl_path, "chunks": ch_path})
                continue
            try:
                seg.write(mid_path)
                x = MultistreamTokenizer.tokenize_midi(mid_path)
                n_in = x["pitch"].shape[0]
                if n_in < 32 or n_in > args.seq_length:
                    skip += 1; os.remove(mid_path); continue
                with torch.no_grad():
                    y = infer(x, model, overlap=64, chunk=args.seq_length, verbose=False, kv_cache=True)
                score = postprocess_score(MultistreamTokenizer.detokenize_mxl(y), inPlace=True)
                score.write("musicxml", fp=mxl_path)
                n_out = MultistreamTokenizer.parse_mxl(mxl_path)["offset"].shape[0]
                if n_out < 16 or n_out > args.seq_length:
                    skip += 1; continue
                json.dump({"midi": [list(range(n_in))], "mxl": [list(range(n_out))], "swapped": False},
                          open(ch_path, "w"))
                rows.append({"id": pid, "midi": mid_path, "mxl": mxl_path, "chunks": ch_path})
                ok += 1
            except Exception as e:
                fail += 1
                if fail <= 5:
                    print(f"  fail {pid} {stem}: {repr(e)[:80]}", flush=True)
        if (pi + 1) % 10 == 0:
            print(f"  {pi+1}/{len(mine)} pieces | windows ok={ok} skip={skip} fail={fail}", flush=True)
    with open(manifest, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "midi", "mxl", "chunks"])
        w.writeheader(); w.writerows(rows)
    print(f"SHARD {args.shard} DONE: {ok} windows, skip={skip}, fail={fail} -> {manifest}", flush=True)


if __name__ == "__main__":
    main()
