"""PAPER-CRITICAL diagnostic: does the offset/placement head retain the correct
non-dyadic target bucket in its local distribution, even when free-running greedy decoding
collapses onto dyadic offsets?

The model decodes each note's within-measure offset on a 1/24-quarter grid (145 buckets,
min=0 max=6 step=1/24). The within-quarter phase of bucket i is (i % 24). The model's
1/8-quarter dyadic lattice is the set of multiples of 3; phases not divisible by 3 require
the finer ternary-capable grid. This is a token-grid definition, not a claim that every such
token is an explicit MusicXML triplet.

We need the full softmax distribution, not only the argmax. For each ground-truth note whose
target offset is off the dyadic lattice, we record: (a) probability on the correct target
bucket, (b) its rank, and (c) total mass on off-dyadic vs dyadic phases.

METHOD (primary): TEACHER-FORCING. We feed the perf-MIDI as encoder input and the GROUND-TRUTH
score tokens as decoder output_streams. The decoder embedding rolls output_streams +1 and zeros
position 0 (embedding.py:141-147, exactly as in train.py:training_step), so pred['offset'][:,t,:]
is the model's logit distribution for GT note t conditioned on the TRUE history of notes 0..t-1.
This is the model's own placement belief at every GT note, including the triplet ones.

The MIDI(encoder) and score(decoder) note streams are LENGTH-MATCHED and beat-aligned by the
project's own ASAPDataset (it uses each piece's *_chunks.json index map). We reuse it directly
(return_paths=True, padding=None, no augmentation) to get the exact aligned (input, output) pair
training used, then teacher-force in non-overlapping 512-note windows (the model's trained context;
the released model is itself evaluated at chunk=512). Co-chunking is valid because encoder/decoder
are co-aligned and equal length.

We also run free-running infer()/generate() and capture its per-note offset argmax. These
free-running emission rates have a different denominator and conditioning history from the
teacher-forced rank statistics, so the artifact and paper report them separately.

CPU ONLY: MPS produces degenerate logits on this model (documented). We force device=cpu and
PYTORCH_ENABLE_MPS_FALLBACK=1, and assert the model lives on CPU.

Run:
    PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONPATH=MIDI2ScoreTransformer/midi2scoretransformer \
    venv311/bin/python scripts/diag_topk_offset.py \
      --ckpt checkpoints/MIDI2ScoreTF.ckpt \
      --out benchmark/topk_offset_diag.json
"""
import argparse, contextlib, hashlib, json, os, sys, time, warnings
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # never touch CUDA
warnings.simplefilter("ignore")

REPO = Path(__file__).resolve().parent.parent
TF = REPO / "MIDI2ScoreTransformer"
sys.path.insert(0, str(TF / "midi2scoretransformer"))
sys.path.insert(0, str(REPO / "benchmark"))
os.chdir(TF)

import numpy as np
import torch
import torch.nn.functional as F
from config import MyModelConfig
if not hasattr(MyModelConfig, "_attn_implementation_internal"):
    MyModelConfig._attn_implementation_internal = None
torch.serialization.add_safe_globals([MyModelConfig])
torch.manual_seed(0)
try:
    torch.use_deterministic_algorithms(True, warn_only=True)
except Exception:
    pass
torch.autocast = lambda *args, **kwargs: contextlib.nullcontext()

from tokenizer import MultistreamTokenizer
from dataset import ASAPDataset
from eval_tier1_asap import collect_paths, load_any_checkpoint

WINDOW = 512  # trained context; teacher-force in non-overlapping windows of this size

N_PHASE = 24
N_BUCKETS = 145  # offset vocab (config.py)
TRIPLET_PHASES = [p for p in range(N_PHASE) if p % 3 != 0]   # legacy name: off-dyadic phases
BINARY_PHASES = [p for p in range(N_PHASE) if p % 3 == 0]    # dyadic lattice: 0,3,...,21
PIECES = ["Liszt", "Ravel", "Scriabin", "Mozart"]
TOPK = 15
FREE_RUNNING_SEED = 1

# phase of every offset bucket 0..144
BUCKET_PHASE = np.array([i % N_PHASE for i in range(N_BUCKETS)])
TRIPLET_BUCKET_MASK = torch.tensor((BUCKET_PHASE % 3 != 0), dtype=torch.bool)   # (145,)
BINARY_BUCKET_MASK = torch.tensor((BUCKET_PHASE % 3 == 0), dtype=torch.bool)


def is_triplet_bucket(idx):
    return int(idx) % N_PHASE % 3 != 0


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select_pieces():
    """Return list of (name, dataset_idx, input_stream, output_stream, midi_path, score_path)
    for the diagnostic pieces, using ASAPDataset's beat-aligned, length-matched pairing.
    Picks the SHORTEST performance per piece (matches tuplet_baseline.json selection)."""
    ds = ASAPDataset("./data/", "test", seq_length=None, cache=True, padding=None,
                     augmentations={}, return_paths=True, use_beat_conditioning=False)
    # gather candidates per piece
    md = ds.metadata
    chosen = []
    for name in PIECES:
        cands = []
        for i in range(len(md)):
            s = md.iloc[i]
            midi = s["performance_MIDI_external"]
            comp = str(s.get("composer", ""))
            if name.lower() in (comp + "/" + midi).lower():
                cands.append(i)
        # pick shortest by note count (cheap: tokenize midi)
        best = None; best_n = 1 << 30
        for i in cands:
            s = md.iloc[i]
            mp = s["performance_MIDI_external"].replace("{ASAP}", "./data/asap-dataset")
            try:
                n = int(MultistreamTokenizer.tokenize_midi(mp)["pitch"].shape[0])
            except Exception:
                n = 1 << 30
            if n < best_n:
                best_n = n; best = i
        if best is None:
            print(f"WARNING: no test piece matched {name}")
            continue
        inp, out, midi_path, score_path = ds[best]
        chosen.append((name, best, inp, out, midi_path, score_path))
    return chosen


def teacher_forced_offset_probs(model, inp, out):
    """Return (probs[N,145], gt_off_bucket[N]) from teacher-forcing the aligned GT score
    tokens. Chunked into non-overlapping WINDOW-note windows to stay within the trained
    context (max_position_embeddings=1536; released eval uses chunk=512)."""
    N = out["offset"].shape[0]
    probs_chunks = []
    for s in range(0, N, WINDOW):
        e = min(s + WINDOW, N)
        xb = {k: v[s:e].unsqueeze(0).to(model.device) for k, v in inp.items()}
        yb = {k: v[s:e].unsqueeze(0).to(model.device) for k, v in out.items()}
        with torch.no_grad():
            pred = model.forward(input_streams=xb, output_streams=yb)
        off_logits = pred["offset"][0].float().cpu()       # (win, 145)
        probs_chunks.append(F.softmax(off_logits, dim=-1))
    probs = torch.cat(probs_chunks, dim=0)                 # (N, 145)
    gt_off = out["offset"].argmax(-1).reshape(-1).cpu().numpy()
    return probs, gt_off


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default=str(REPO / "benchmark/topk_offset_diag.json"))
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    assert args.device == "cpu", "MPS/CUDA forbidden for this diagnostic (degenerate logits)."

    model = load_any_checkpoint(args.ckpt, "cpu")
    model.eval(); model.to("cpu")
    # HARD assert the model is on CPU
    devs = {p.device.type for p in model.parameters()}
    assert devs == {"cpu"}, f"model not fully on CPU: {devs}"
    print(f"model on CPU: {devs}; params={sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    from utils import infer  # free-running, for self-consistency argmax

    pieces = select_pieces()
    per_piece = {}
    started = time.time()
    for name, didx, inp, out, midi_path, score_path in pieces:
        probs, gt_off = teacher_forced_offset_probs(model, inp, out)
        N = probs.shape[0]
        assert N == len(gt_off)
        piece_label = os.path.relpath(os.path.dirname(midi_path), "./data/asap-dataset") \
            if midi_path.startswith("./data") else os.path.dirname(midi_path)

        # ---- teacher-forced argmax placement (sanity) ----
        tf_arg = probs.argmax(-1).numpy()
        tf_arg_phase = tf_arg % N_PHASE
        tf_triplet_rate = float(np.mean(tf_arg_phase % 3 != 0))

        # ---- aggregate mass per note ----
        trip_mass_all = probs[:, TRIPLET_BUCKET_MASK].sum(-1).numpy()   # (N,)
        bin_mass_all = probs[:, BINARY_BUCKET_MASK].sum(-1).numpy()

        # ---- GT triplet notes only ----
        gt_phase = gt_off % N_PHASE
        gt_is_trip = (gt_phase % 3 != 0)
        idx_trip = np.where(gt_is_trip)[0]
        n_gt_trip = int(len(idx_trip))

        rec = {
            "composer": name,
            "piece": piece_label,
            "n_notes": int(N),
            "n_gt_triplet_notes": n_gt_trip,
            "gt_triplet_rate": float(gt_is_trip.mean()),
            "tf_argmax_triplet_rate": tf_triplet_rate,
            # mass averaged over ALL notes
            "mean_triplet_mass_all_notes": float(trip_mass_all.mean()),
            "mean_binary_mass_all_notes": float(bin_mass_all.mean()),
        }

        if n_gt_trip:
            sub = probs[idx_trip]                                   # (M,145)
            correct_bucket = gt_off[idx_trip]                      # (M,)
            # prob on the correct triplet bucket
            p_correct = sub[np.arange(n_gt_trip), correct_bucket].numpy()
            # rank of correct bucket (1 = top); count buckets strictly greater
            order = sub.argsort(dim=-1, descending=True).numpy()    # (M,145)
            rank = np.array([int(np.where(order[i] == correct_bucket[i])[0][0]) + 1
                             for i in range(n_gt_trip)])
            top15_hit = float(np.mean(rank <= TOPK))
            top5_hit = float(np.mean(rank <= 5))
            top1_hit = float(np.mean(rank == 1))
            # mass on ALL triplet phases vs binary phases, for the GT-triplet notes
            trip_mass_t = trip_mass_all[idx_trip]
            bin_mass_t = bin_mass_all[idx_trip]
            rec.update({
                "mean_prob_correct_triplet_bucket": float(p_correct.mean()),
                "median_prob_correct_triplet_bucket": float(np.median(p_correct)),
                "max_prob_correct_triplet_bucket": float(p_correct.max()),
                "mean_rank_correct_triplet_bucket": float(rank.mean()),
                "median_rank_correct_triplet_bucket": float(np.median(rank)),
                "top1_hit_rate": top1_hit,
                "top5_hit_rate": top5_hit,
                "top15_hit_rate": top15_hit,
                "mean_triplet_mass_on_gt_triplet_notes": float(trip_mass_t.mean()),
                "mean_binary_mass_on_gt_triplet_notes": float(bin_mass_t.mean()),
            })
        else:
            rec.update({k: None for k in [
                "mean_prob_correct_triplet_bucket", "median_prob_correct_triplet_bucket",
                "max_prob_correct_triplet_bucket", "mean_rank_correct_triplet_bucket",
                "median_rank_correct_triplet_bucket", "top1_hit_rate", "top5_hit_rate",
                "top15_hit_rate", "mean_triplet_mass_on_gt_triplet_notes",
                "mean_binary_mass_on_gt_triplet_notes"]})

        # ---- FREE-RUNNING self-consistency: argmax tuplet rate from generate() ----
        # A forward hook on the offset output head captures the LIVE free-running offset
        # distribution at each decode step (the last position of each forward_dec call).
        # This tests the REALISTIC reranking question: at decode time, does the offset
        # distribution carry recoverable triplet mass, or is it collapsed at the source?
        fr_dists = []

        def _off_hook(module, inp, output):
            # output: (B, T, 145) logits from the offset Linear head. Take last position.
            fr_dists.append(F.softmax(output[:, -1, :].float(), dim=-1).detach().cpu())

        h = model.unembeddings_dec.embeddings["offset"].register_forward_hook(_off_hook)
        x_mid = MultistreamTokenizer.tokenize_midi(midi_path)
        torch.manual_seed(FREE_RUNNING_SEED)
        with torch.no_grad():
            y = infer(x_mid, model, overlap=64, chunk=512, verbose=False, kv_cache=True)
        h.remove()
        fr_off_all = y["offset"].argmax(-1).reshape(-1).cpu().numpy()
        keep = (
            y.get("pad_prob", y["pad"]).reshape(-1).cpu().numpy() > 0.5
        )
        if len(keep) != len(fr_off_all):
            raise RuntimeError("Free-running offset and keep streams differ in length")
        fr_off = fr_off_all[keep]
        fr_phase = fr_off % N_PHASE
        rec["free_running_n_positions"] = int(len(fr_off_all))
        rec["free_running_n_notes"] = int(len(fr_off))
        rec["free_running_keep_rule"] = "pad probability > 0.5"
        rec["free_running_triplet_argmax_rate"] = float(np.mean(fr_phase % 3 != 0))
        rec["free_running_n_triplet_argmax"] = int(np.sum(fr_phase % 3 != 0))
        # Aggregate live free-running offset distributions (one row per decode step).
        if fr_dists:
            FR = torch.cat(fr_dists, dim=0)                  # (steps, 145)
            fr_trip_mass = FR[:, TRIPLET_BUCKET_MASK].sum(-1).numpy()
            fr_bin_mass = FR[:, BINARY_BUCKET_MASK].sum(-1).numpy()
            # "would top-15 reranking ever see a triplet bucket?" — count steps whose
            # top-15 contains ANY triplet bucket, and steps with >5% / >1% triplet mass.
            order = FR.argsort(dim=-1, descending=True)[:, :TOPK].numpy()  # (steps,15)
            top15_has_trip = np.array([
                any(is_triplet_bucket(b) for b in order[i]) for i in range(order.shape[0])])
            rec["free_running_mean_triplet_mass_per_step"] = float(fr_trip_mass.mean())
            rec["free_running_median_triplet_mass_per_step"] = float(np.median(fr_trip_mass))
            rec["free_running_p95_triplet_mass_per_step"] = float(np.percentile(fr_trip_mass, 95))
            rec["free_running_frac_steps_triplet_in_top15"] = float(top15_has_trip.mean())
            rec["free_running_frac_steps_tripmass_gt_0.05"] = float(np.mean(fr_trip_mass > 0.05))
            rec["free_running_frac_steps_tripmass_gt_0.01"] = float(np.mean(fr_trip_mass > 0.01))
            print(f"  FREE-RUN live offset dist: mean triplet-mass/step={fr_trip_mass.mean():.4f} "
                  f"median={np.median(fr_trip_mass):.4f} p95={np.percentile(fr_trip_mass,95):.4f}")
            print(f"  FREE-RUN frac steps with triplet bucket in top-15={top15_has_trip.mean():.3%} "
                  f"| triplet-mass>5%={np.mean(fr_trip_mass>0.05):.3%} >1%={np.mean(fr_trip_mass>0.01):.3%}")

        per_piece[name] = rec
        print(f"\n=== {name} ({piece_label}) ===")
        print(f"  notes(TF)={N}  GT triplet notes={n_gt_trip} ({rec['gt_triplet_rate']:.3%})")
        print(f"  TF argmax triplet rate={tf_triplet_rate:.3%}")
        print(f"  FREE-RUN argmax triplet rate={rec['free_running_triplet_argmax_rate']:.3%} "
              f"({rec['free_running_n_triplet_argmax']} notes)")
        if n_gt_trip:
            print(f"  P(correct triplet bucket): mean={rec['mean_prob_correct_triplet_bucket']:.4f} "
                  f"median={rec['median_prob_correct_triplet_bucket']:.4f} max={rec['max_prob_correct_triplet_bucket']:.4f}")
            print(f"  rank of correct bucket: mean={rec['mean_rank_correct_triplet_bucket']:.1f} "
                  f"median={rec['median_rank_correct_triplet_bucket']:.0f}")
            print(f"  hit@1={rec['top1_hit_rate']:.3%} hit@5={rec['top5_hit_rate']:.3%} hit@15={rec['top15_hit_rate']:.3%}")
            print(f"  on GT-triplet notes: triplet-mass={rec['mean_triplet_mass_on_gt_triplet_notes']:.4f} "
                  f"binary-mass={rec['mean_binary_mass_on_gt_triplet_notes']:.4f}")

    # ---- aggregate (pool GT-triplet notes across pieces by weighting) ----
    def wmean(key, wkey="n_gt_triplet_notes"):
        num = den = 0.0
        for r in per_piece.values():
            if r.get(key) is not None and r.get(wkey):
                num += r[key] * r[wkey]; den += r[wkey]
        return (num / den) if den else None
    agg = {
        "pieces": PIECES,
        "weighted_mean_prob_correct_triplet_bucket": wmean("mean_prob_correct_triplet_bucket"),
        "weighted_mean_rank_correct_triplet_bucket": wmean("mean_rank_correct_triplet_bucket"),
        "weighted_top1_hit_rate": wmean("top1_hit_rate"),
        "weighted_top5_hit_rate": wmean("top5_hit_rate"),
        "weighted_top15_hit_rate": wmean("top15_hit_rate"),
        "weighted_triplet_mass_on_gt_triplet_notes": wmean("mean_triplet_mass_on_gt_triplet_notes"),
        "weighted_binary_mass_on_gt_triplet_notes": wmean("mean_binary_mass_on_gt_triplet_notes"),
    }

    n_target = sum(r["n_gt_triplet_notes"] for r in per_piece.values())
    paper_summary = {
        "teacher_forced_population": (
            "ground-truth notes whose correct offset bucket is off the model's "
            "1/8-quarter dyadic lattice (bucket phase % 3 != 0)"
        ),
        "teacher_forced_n_target_notes": n_target,
        "teacher_forced_correct_bucket_hit_at_1": agg["weighted_top1_hit_rate"],
        "teacher_forced_correct_bucket_hit_at_5": agg["weighted_top5_hit_rate"],
        "teacher_forced_correct_bucket_hit_at_15": agg["weighted_top15_hit_rate"],
        "free_running_population": (
            "generated notes retained by the model keep stream (pad probability > 0.5), "
            "reported separately by piece"
        ),
        "free_running_greedy_off_dyadic_emission": {
            name: {
                "n_notes": rec["free_running_n_notes"],
                "n_off_dyadic": rec["free_running_n_triplet_argmax"],
                "rate": rec["free_running_triplet_argmax_rate"],
            }
            for name, rec in per_piece.items()
        },
        "comparison_rule": (
            "Use hit@1 versus hit@15 as the same-denominator local-rank comparison. "
            "Free-running greedy emission is a separate sequence-level observation."
        ),
    }

    out = {
        "schema_version": 3,
        "method": "Teacher-force GT score tokens through the decoder and inspect the offset-head "
                  "softmax (145 buckets, 1/24-quarter grid). The target population is correct "
                  "offsets off the 1/8-quarter dyadic lattice (phase % 3 != 0). Free-running "
                  "argmax emissions are measured separately. Device=CPU (MPS forbidden).",
        "definitions": {
            "off_dyadic_target": "correct offset bucket has within-quarter phase % 3 != 0",
            "teacher_forced_hit_at_k": "correct target bucket rank <= k under the GT prefix",
            "free_running_off_dyadic_emission": (
                "retained generated argmax offset has phase % 3 != 0 under the model's "
                "own history; keep means pad probability > 0.5"
            ),
            "scope_caveat": (
                "off-dyadic is a token-grid property and is not identical to explicit "
                "MusicXML <time-modification> membership"
            ),
        },
        "ckpt": args.ckpt,
        "free_running_seed": FREE_RUNNING_SEED,
        "n_phase": N_PHASE, "n_buckets": N_BUCKETS, "topk": TOPK,
        "off_dyadic_phases": TRIPLET_PHASES,
        "triplet_phases": TRIPLET_PHASES,
        "per_piece": per_piece,
        "aggregate": agg,
        "paper_summary": paper_summary,
        "elapsed_s": round(time.time() - started, 1),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "provenance": {
            "diagnostic_sha256": file_hash(Path(__file__).resolve()),
            "checkpoint_sha256": file_hash((TF / args.ckpt).resolve()),
            "precision": "CPU FP32 with autocast disabled",
            "correction": (
                "Free-running prevalence now applies the same keep mask used by score "
                "detokenization. Teacher-forced populations and ranks are unchanged."
            ),
            "superseded_unmasked_artifact_sha256": (
                "37975a30379b4711ae1f96ef168816c71909b2e5634c39cb0512f5e856aa670f"
            ),
        },
    }
    outp = Path(args.out)
    if not outp.is_absolute():
        outp = REPO / outp
    json.dump(out, open(outp, "w"), indent=2)
    print(f"\nWrote {outp}")


if __name__ == "__main__":
    main()
