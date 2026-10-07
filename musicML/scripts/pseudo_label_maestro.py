# HISTORICAL: built MAESTRO-derived corpora on the 2026 GPU box (/root/datasets paths). MAESTRO v3 and
# those corpora are not part of this repository (see DATA.md, known missing); defaults are the box paths.
"""Track ST (self-training / distillation): pseudo-label MAESTRO performance-MIDIs with a TEACHER
checkpoint (default: released MIDI2ScoreTF, the best tuplet-placer) into engraved MusicXML scores.

These tuplet-CORRECT pseudo-scores are then added (upweighted) to the unpaired SSL corpus, injecting
correctly-placed-tuplet examples at a natural repertoire distribution — directly fixing the tuplet-poor
corpus that collapsed our offset prior, WITHOUT the reshape's over-emission.

Sharded so two GPUs can split the work:  --shard i --nshards N  (i in 0..N-1)

Run (box, per GPU):
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=MIDI2ScoreTransformer/midi2scoretransformer \
    venv311/bin/python scripts/pseudo_label_maestro.py --ckpt <released> --n 240 --shard 0 --nshards 2 \
    --out-dir /root/datasets/maestro_pseudo --device cuda
"""
import argparse, glob, os, sys, warnings
from pathlib import Path
warnings.simplefilter("ignore")
REPO = Path(__file__).resolve().parent.parent
TF = REPO / "MIDI2ScoreTransformer"
sys.path.insert(0, str(TF / "midi2scoretransformer")); sys.path.insert(0, str(REPO / "benchmark"))
import torch
from config import MyModelConfig
if not hasattr(MyModelConfig, "_attn_implementation_internal"):
    MyModelConfig._attn_implementation_internal = None
torch.serialization.add_safe_globals([MyModelConfig])
from tokenizer import MultistreamTokenizer
from utils import infer
from score_utils import postprocess_score
from eval_tier1_asap import load_any_checkpoint


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--maestro-root", default="/root/datasets/maestro-v3.0.0")
    ap.add_argument("--n", type=int, default=240, help="total pieces (shortest-first for speed)")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--out-dir", default="/root/datasets/maestro_pseudo")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    midis = sorted(glob.glob(os.path.join(args.maestro_root, "*", "*.midi")), key=os.path.getsize)
    midis = midis[:args.n]                       # shortest n (fast generation)
    mine = midis[args.shard::args.nshards]       # round-robin shard
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"shard {args.shard}/{args.nshards}: {len(mine)} of {len(midis)} MAESTRO pieces", flush=True)

    model = load_any_checkpoint(args.ckpt, args.device); model.eval(); model.to(args.device)
    ok = fail = 0
    for i, mp in enumerate(mine):
        stem = Path(mp).stem[:60]
        out = os.path.join(args.out_dir, f"{args.shard}_{i}_{stem}.musicxml")
        if os.path.exists(out):
            ok += 1; continue
        try:
            x = MultistreamTokenizer.tokenize_midi(mp)
            if x["pitch"].shape[0] < 32 or x["pitch"].shape[0] > 6000:
                continue
            with torch.no_grad():
                y = infer(x, model, overlap=64, chunk=512, verbose=False, kv_cache=True)
            score = postprocess_score(MultistreamTokenizer.detokenize_mxl(y), inPlace=True)
            score.write("musicxml", fp=out)
            ok += 1
        except Exception as e:
            fail += 1
            if fail <= 5:
                print(f"  fail {stem}: {repr(e)[:100]}", flush=True)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(mine)} ok={ok} fail={fail}", flush=True)
    print(f"SHARD {args.shard} DONE ok={ok} fail={fail}", flush=True)


if __name__ == "__main__":
    main()
