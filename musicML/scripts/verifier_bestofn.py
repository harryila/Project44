"""PAPER-A experiment: VERIFIER-GUIDED BEST-OF-N DECODING.

THE QUESTION (the constructive step after the diagnosis). diag_topk_offset.py showed the
correct triplet offset is in the model's top-15 ~99% of the time; rerank_offset.py showed a
uniform logit boost cannot exploit that (moves production, not placement). This script tests
the selection-based alternative: SAMPLE N candidate decodes (rhythm streams only), score each
with a hard music-theory VERIFIER (no learning, no logit surgery), and keep the best.

WHY THIS IS DIFFERENT FROM THE BOOST: the boost changes every step's distribution blindly;
best-of-n leaves the model's distribution UNTOUCHED and only chooses among full sequences the
model itself proposed. If the knowledge-is-there diagnosis is right, good sequences exist
among samples; the verifier's job is only to find them.

SAMPLING: model.generate(head_overrides={"offset": (K, tau), "duration": (K, tau)}):
top-K crop + temperature tau + multinomial ON THE RHYTHM STREAMS ONLY; pitch/voice/etc stay
greedy (top_k=1 multinomial == argmax, so a no-override run is byte-identical to baseline).

THE VERIFIER (rhythm algebra on the 1/24-quarter grid, computed on raw decoded streams):
  (V1) LONE-TRIPLET INCOHERENCE: a note at a triplet phase (p % 3 != 0, p = bucket % 24) is
       musically valid only as part of a group. Group = notes in the same (measure, quarter).
       A (measure, quarter) containing exactly ONE triplet-phase note = 1 violation.
  (V2) ONSET/DURATION PARITY MISMATCH: on the 1/24 grid dyadic durations are multiples of 3
       ticks (eighth=12, 16th=6) and triplet durations are not (triplet-8th=8, -16th=4).
       A triplet-phase onset carrying a dyadic duration (or vice versa) = 0.5 violation.
  score(candidate) = (V1 + 0.5 * V2) / n_notes    (lower = more musically coherent)

SELECTION + REPORTING per piece:
  baseline   = pure argmax decode (the released behavior)
  verifier   = candidate with the lowest verifier score (tie -> lowest candidate index)
  oracle     = candidate with the lowest MUSTER MeanER (upper bound of best-of-n; needs the
               full MUSTER run on every candidate, which we do anyway for the paper table)
  mean(cand) = average candidate MeanER (shows sampling alone HURTS; selection is the point)

EVAL: same MUSTER + tuplet decomposition pipeline as rerank_offset.py (MTD), CPU-only, FP32.

Run (smoke):
  venv311/bin/python scripts/verifier_bestofn.py --pieces Scriabin --n 4 \
      --out benchmark/verifier_bestofn_smoke.json
Run (full):
  venv311/bin/python scripts/verifier_bestofn.py --pieces Scriabin Mozart Ravel Liszt \
      --n 32 --out benchmark/verifier_bestofn.json
"""
import argparse, json, os, sys, time, warnings
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
warnings.simplefilter("ignore")

REPO = Path(__file__).resolve().parent.parent
TF = REPO / "MIDI2ScoreTransformer"
sys.path.insert(0, str(TF / "midi2scoretransformer"))
sys.path.insert(0, str(REPO / "benchmark"))
sys.path.insert(0, str(REPO / "scripts"))
os.chdir(TF)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from config import MyModelConfig  # noqa: E402
if not hasattr(MyModelConfig, "_attn_implementation_internal"):
    MyModelConfig._attn_implementation_internal = None
torch.serialization.add_safe_globals([MyModelConfig])

# determinism: FP32 decode (no BF16 autocast jitter), explicit seeds per candidate
import contextlib  # noqa: E402
try:
    torch.use_deterministic_algorithms(True, warn_only=True)
except Exception:
    pass
torch.autocast = lambda *a, **k: contextlib.nullcontext()

from tokenizer import MultistreamTokenizer  # noqa: E402
from utils import infer  # noqa: E402
from score_utils import postprocess_score  # noqa: E402
from eval_tier1_asap import collect_paths, load_any_checkpoint  # noqa: E402
import muster_tuplet_decompose as MTD  # noqa: E402

N_PHASE = 24


# --------------------------------------------------------------------------- the verifier
def verifier_score(y, in_onset=None, w3=2.0):
    """Rhythm-algebra violations on raw decoded streams. Returns (score, detail dict).

    V1/V2: internal coherence (lone triplets, onset/duration parity).
    V3 (the evidence term): the decoded notation must EXPLAIN the input performance timing.
    For every decoded (measure, quarter) group with >=3 distinct onsets, the input onset
    times of those notes, normalized to the group span, must match the decoded grid phases
    normalized the same way. A binary decode over performance timing that is really thirds
    gets penalized exactly as hard as a triplet sprinkle over even timing, so the all-binary
    baseline is no longer trivially "coherent"."""
    off = y["offset"].argmax(-1).reshape(-1).cpu().numpy()
    dur = y["duration"].argmax(-1).reshape(-1).cpu().numpy()
    db = y["downbeat"].argmax(-1).reshape(-1).cpu().numpy()
    pad_src = y.get("pad_prob", y.get("pad"))
    kept = pad_src.reshape(-1).cpu().numpy() > 0.5 if pad_src is not None else np.ones(len(off), bool)
    n = len(off)
    if n == 0:
        return float("inf"), {"n": 0, "v1": 0, "v2": 0, "v3": 0.0}
    measure = np.cumsum(db > 0)          # measure index per note
    quarter = off // N_PHASE             # which quarter within the measure
    phase = off % N_PHASE
    is_trip_on = (phase % 3) != 0
    is_trip_dur = (dur % 3) != 0

    # V1: lone triplet-phase note in its (measure, quarter) group
    v1 = 0
    trip_idx = np.flatnonzero(is_trip_on)
    if len(trip_idx):
        keys = measure[trip_idx] * 1000 + quarter[trip_idx]
        _, counts = np.unique(keys, return_counts=True)
        v1 = int(np.sum(counts == 1))

    # V2: onset/duration parity mismatch
    v2 = int(np.sum(is_trip_on != is_trip_dur))

    # V3b: duple-vs-triple HYPOTHESIS TEST per quarter (pre-registered design, theta=0.02).
    # Anchor each (measure, quarter) group's time span with the first onset of the NEXT
    # adjacent decoded quarter; normalize input onsets to that span; fit against the duple
    # grid {0,1/4,1/2,3/4} vs the triple grid {0,1/3,2/3}; penalize decodes that contradict
    # the grid the evidence favors by more than theta. Penalty = evidence margin.
    THETA = 0.02
    DUPLE = np.array([0.0, 0.25, 0.5, 0.75])
    TRIPLE = np.array([0.0, 1.0 / 3.0, 2.0 / 3.0])
    v3 = 0.0
    n_groups = 0
    n_contra = 0
    if in_onset is not None:
        t = np.asarray(in_onset, dtype=float)[: n]
        gkey = measure * 1000 + quarter
        kept_idx = np.flatnonzero(kept)
        # first-onset time per group, for anchoring
        g_first = {}
        for g in np.unique(gkey[kept]):
            idx = kept_idx[gkey[kept_idx] == g]
            g_first[int(g)] = float(t[idx].min())
        for g in np.unique(gkey[kept]):
            idx = kept_idx[gkey[kept_idx] == g]
            anchor = g_first.get(int(g) + 1)  # next quarter in the same measure
            if anchor is None:
                m_, q_ = int(g) // 1000, int(g) % 1000
                anchor = g_first.get((m_ + 1) * 1000)  # first quarter of next measure
            t0 = float(t[idx].min())
            if anchor is None or anchor <= t0:
                continue
            # one representative (chord onset = earliest input time) per distinct phase
            ph_u, t_u = [], []
            for p_ in np.unique(phase[idx]):
                sel = idx[phase[idx] == p_]
                ph_u.append(float(p_)); t_u.append(float(t[sel].min()))
            if len(ph_u) < 3:
                continue
            order = np.argsort(t_u)
            t_s = np.array(t_u)[order]
            if not np.all(np.diff(t_s) > 0):
                continue
            u = (t_s - t0) / (anchor - t0)
            u = u[(u >= -0.05) & (u < 1.05)]
            if len(u) < 3:
                continue
            err_d = float(np.mean(np.min(np.abs(u[:, None] - DUPLE[None, :]), axis=1)))
            err_t = float(np.mean(np.min(np.abs(u[:, None] - TRIPLE[None, :]), axis=1)))
            margin = err_d - err_t                      # >0: evidence favors TRIPLE
            says_triple = bool(np.any((np.array(ph_u) % 3) != 0))
            n_groups += 1
            if margin > THETA and not says_triple:
                v3 += margin; n_contra += 1
            elif margin < -THETA and says_triple:
                v3 += -margin; n_contra += 1

    n_kept = max(int(kept.sum()), 1)
    score = (v1 + 0.5 * v2) / n + w3 * (v3 / n_kept)
    return float(score), {"n": int(n), "n_kept": n_kept, "v1": v1, "v2": v2,
                          "v3": round(float(v3), 4), "v3_groups": n_groups,
                          "v3_contra": n_contra,
                          "trip_onset_rate": float(np.mean(is_trip_on))}


# --------------------------------------------------------------------------- MUSTER eval
def muster_piece(y, p, workdir):
    mxl = postprocess_score(MultistreamTokenizer.detokenize_mxl(y, pad_threshold=0.5), inPlace=True)
    agg, gt_fmt3x, err_detail, auto_match = MTD.run_muster_keep(mxl, p["score"], workdir)
    gt_info = MTD.parse_gt_fmt3x(gt_fmt3x)
    matched = MTD.parse_matched_gtids(auto_match, gt_info)
    err = MTD.parse_err_detail(err_detail)
    dec = MTD.decompose_piece(gt_info, matched, err, agg)
    return {
        "meaner_overall": dec["meaner_overall"],
        "onset_err_tuplet": dec["onset_err_tuplet"],
        "onset_err_nontuplet": dec["onset_err_nontuplet"],
        "n_tuplet": dec["n_tuplet"], "n_nontuplet": dec["n_nontuplet"],
    }


def decode(model, x, overrides, seed):
    torch.manual_seed(seed)
    with torch.no_grad():
        return infer(x, model, overlap=64, chunk=512, verbose=False, kv_cache=True,
                     head_overrides=overrides)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/MIDI2ScoreTF.ckpt")
    ap.add_argument("--pieces", nargs="*", default=["Scriabin"])
    ap.add_argument("--n", type=int, default=8, help="number of sampled candidates")
    ap.add_argument("--topk", type=int, default=15)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--w3", type=float, default=2.0, help="weight of the timing-explanation term")
    ap.add_argument("--streams", nargs="*", default=["offset", "duration"],
                    help="streams to sample; others stay greedy")
    ap.add_argument("--out", default=str(REPO / "benchmark/verifier_bestofn.json"))
    args = ap.parse_args()
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = REPO / out_path

    # pieces (shortest performance per piece, same convention as rerank_offset.py)
    paths = collect_paths("test")
    for p in paths:
        try:
            p["_x"] = MultistreamTokenizer.tokenize_midi(p["midi"])
            p["_t"] = MultistreamTokenizer.parse_midi(p["midi"])["onset"].numpy()
            p["n_notes"] = int(p["_x"]["pitch"].shape[0])
        except Exception:
            p["_x"] = None; p["n_notes"] = 1 << 30
    paths = [p for p in paths
             if any(s.lower() in (p["composer"] + "/" + p["piece"]).lower() for s in args.pieces)]
    paths.sort(key=lambda p: p["n_notes"])
    seen, kept = set(), []
    for p in paths:
        key = (p["composer"], p["piece"])
        if key not in seen and p["_x"] is not None:
            seen.add(key); kept.append(p)
    paths = kept
    print(f"{len(paths)} pieces: " + ", ".join(f"{p['composer']}/{p['piece']}" for p in paths))

    model = load_any_checkpoint(args.ckpt, "cpu"); model.eval(); model.to("cpu")
    assert {pp.device.type for pp in model.parameters()} == {"cpu"}

    overrides = {s: (args.topk, args.temp) for s in args.streams}
    work_root = REPO / "benchmark" / "bestofn_work"
    results = {"ckpt": args.ckpt, "n": args.n, "topk": args.topk, "temp": args.temp,
               "streams": args.streams, "pieces": {}}
    t0 = time.time()

    for p in paths:
        tag = p["piece"].replace("/", "_")
        print(f"\n===== {p['composer']}/{p['piece']} ({p['n_notes']} notes) =====", flush=True)

        # ---- identity sanity: overrides on NON-sampled streams must not change the argmax path.
        # A top_k=1 multinomial IS the argmax, so a no-override decode at any seed == baseline.
        y_base_a = decode(model, p["_x"], None, seed=1)
        y_base_b = decode(model, p["_x"], None, seed=999)
        same = all(torch.equal(y_base_a[k], y_base_b[k]) for k in ("offset", "duration", "pitch"))
        print(f"  [sanity] argmax decode is seed-invariant: {same}")

        vs_b, vdet_b = verifier_score(y_base_a, p["_t"], w3=args.w3)
        rec_b = muster_piece(y_base_a, p, work_root / tag / "baseline")
        print(f"  baseline: MeanER={rec_b['meaner_overall']:.2f} "
              f"onT={rec_b['onset_err_tuplet']} verifier={vs_b:.4f} (v1={vdet_b['v1']} v2={vdet_b['v2']})",
              flush=True)

        cands = []
        for i in range(args.n):
            y_c = decode(model, p["_x"], overrides, seed=1000 + i)
            vs, vdet = verifier_score(y_c, p["_t"], w3=args.w3)
            rec = muster_piece(y_c, p, work_root / tag / f"cand{i:02d}")
            cands.append({"i": i, "seed": 1000 + i, "verifier": vs, **vdet, **rec})
            print(f"  cand{i:02d}: MeanER={rec['meaner_overall']:.2f} onT={rec['onset_err_tuplet']} "
                  f"verifier={vs:.4f} (v1={vdet['v1']} v2={vdet['v2']} v3={vdet['v3']} trip%={vdet['trip_onset_rate']*100:.1f})",
                  flush=True)

        ok = [c for c in cands if c["meaner_overall"] is not None]
        ver_pick = min(cands, key=lambda c: (c["verifier"], c["i"]))
        oracle = min(ok, key=lambda c: c["meaner_overall"]) if ok else None
        mean_cand = float(np.mean([c["meaner_overall"] for c in ok])) if ok else None
        results["pieces"][f"{p['composer']}/{p['piece']}"] = {
            "baseline": {**rec_b, "verifier": vs_b, **vdet_b},
            "candidates": cands,
            "verifier_pick": ver_pick, "oracle_pick": oracle, "mean_candidate_meaner": mean_cand,
            "argmax_seed_invariant": bool(same),
        }
        print(f"  --> baseline {rec_b['meaner_overall']:.2f} | mean(cand) "
              f"{mean_cand:.2f} | verifier-pick {ver_pick['meaner_overall']:.2f} "
              f"(onT {ver_pick['onset_err_tuplet']}) | oracle {oracle['meaner_overall']:.2f} "
              f"(onT {oracle['onset_err_tuplet']})", flush=True)
        json.dump(results, open(out_path, "w"), indent=2)

    results["elapsed_s"] = round(time.time() - t0, 1)
    json.dump(results, open(out_path, "w"), indent=2)
    print(f"\nWrote {out_path} ({results['elapsed_s']}s)")


if __name__ == "__main__":
    main()
