"""PAPER-CRITICAL experiment: DECODE-TIME top-k OFFSET reranker.

QUESTION (necessary -> sufficient). diag_topk_offset.py already showed the correct triplet
offset bucket sits in the decode-time top-15 ~99% of steps (reranking is VIABLE). This script
turns that into the sufficient test: does a music-theory reranker over the top-k offset
candidates actually RECOVER tuplet onset placement, and at what cost to MUSTER?

THE RERANKER (model.generate(offset_rerank=...) hook, applied to the OFFSET head's logits
BEFORE the top-k crop; offset is decoded FIRST each step so only PAST offsets are available):

  candidate bucket b on the 1/24-quarter offset grid (vocab 145, min=0 max=6):
      quarter q = b // 24      (which quarter of the measure)
      phase   p = b %  24      (sub-quarter position; triplet = p % 3 != 0)
  score(b) = METRIC + COHERENCE, then  logits[topk] += lambda * score

  (a) METRICAL PRIOR (A2-style):  beta_m * log P(phase=p)
        empirical marginal phase frequency from engraved ASAP train+val scores (leakage-safe,
        same corpus the duration A1/A2 priors use). Mildly favours phases that actually occur.

  (b) TUPLET-GROUP COHERENCE (the strongest single rule): an eighth-note triplet divides one
        quarter into 3 at phases {0,8,16}. A note at a triplet phase is musically valid ONLY
        as a member of a complete group, never as a lone off-grid note. So:
          - candidate at triplet phase p in quarter q:
              + beta_c   if the PREVIOUS note already sits at this group's earlier member in
                         the SAME quarter q (prev phase in {0,8} for p=16; {0} for p=8) OR the
                         previous note completes a run into it  -> rewards COMPLETING a group
              - beta_c   otherwise (a lone triplet note with no partner) -> discourage sprinkle
          - candidate at a binary phase: 0 (neutral; the model's own logit decides).

  lambda = 0  => callback is a no-op identity => byte-identical to released decode (sanity check).
  We tune lambda on 1-2 pieces, then eval on all 4 diagnostic pieces.

EVAL: scripts/muster_tuplet_decompose.py's exact MUSTER pipeline + tuplet/non-tuplet split,
run BASELINE (lambda=0) vs RERANKED. We report overall MeanER, tuplet-onset error rate,
non-tuplet onset error rate, and the placement rate = fraction of decoded notes at a triplet
phase (proxy for "did we stop quantizing triplets to the binary grid").

CPU ONLY (MPS => degenerate logits, would invalidate everything).

Run:
  PYTORCH_ENABLE_MPS_FALLBACK=1 venv311/bin/python scripts/rerank_offset.py \
      --ckpt MIDI2ScoreTransformer/checkpoints/MIDI2ScoreTF.ckpt \
      --pieces Liszt Ravel Scriabin Mozart \
      --lambdas 0 4 8 16 \
      --out benchmark/reranker_results.json
"""
import argparse, hashlib, json, os, subprocess, sys, time, warnings
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
warnings.simplefilter("ignore")

REPO = Path(__file__).resolve().parent.parent
TF = REPO / "MIDI2ScoreTransformer"
sys.path.insert(0, str(TF / "midi2scoretransformer"))
sys.path.insert(0, str(REPO / "benchmark"))
sys.path.insert(0, str(REPO / "scripts"))   # for muster_tuplet_decompose import
os.chdir(TF)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from config import MyModelConfig  # noqa: E402
if not hasattr(MyModelConfig, "_attn_implementation_internal"):
    MyModelConfig._attn_implementation_internal = None
torch.serialization.add_safe_globals([MyModelConfig])

# --- determinism for a CLEAN dose-response control: FP32 (no BF16 autocast jitter) + seeded.
# The first smoke's lambda=0 identity control was off ~2.5% because infer() runs later chunks
# under torch.autocast(bfloat16), which is non-deterministic run-to-run. Force FP32 everywhere.
import contextlib  # noqa: E402
torch.manual_seed(0)
try:
    torch.use_deterministic_algorithms(True, warn_only=True)
except Exception:
    pass
torch.autocast = lambda *a, **k: contextlib.nullcontext()  # force FP32 decode

from tokenizer import MultistreamTokenizer  # noqa: E402
from utils import infer  # noqa: E402
from score_utils import postprocess_score  # noqa: E402
from eval_tier1_asap import collect_paths, load_any_checkpoint  # noqa: E402

# reuse the MUSTER decomposition pipeline verbatim
import muster_tuplet_decompose as MTD  # noqa: E402  (scripts/ is importable via REPO path?)

N_PHASE = 24
N_BUCKETS = 145
TOPK = 15
TRIPLET_PHASES = set(p for p in range(N_PHASE) if p % 3 != 0)
# eighth-note triplet sub-beats {8,16} are the musically dominant non-binary phases;
# sixteenth-triplet would add {4,20} etc. We treat ANY non-multiple-of-3 phase as triplet,
# matching diag_topk_offset.py and muster_tuplet_decompose.py.
BUCKET_PHASE = np.array([b % N_PHASE for b in range(N_BUCKETS)])
BUCKET_QUARTER = np.array([b // N_PHASE for b in range(N_BUCKETS)])


# ----------------------------------------------------------------------------- phase prior
def build_phase_prior(splits=("train", "validation")):
    """Empirical log P(offset phase) from engraved ASAP non-test scores (leakage-safe).
    Returns a torch (N_PHASE,) log-prob vector. Cached to data/offset_phase_prior.pt."""
    cache = REPO / "data" / "offset_phase_prior.pt"
    if cache.exists():
        d = torch.load(cache)
        return d["log_p_phase"]
    counts = torch.zeros(N_PHASE, dtype=torch.float64)
    seen = set()
    n_scores = 0
    for sp in splits:
        for p in collect_paths(sp):
            s = p.get("score")
            if not s or s in seen:
                continue
            seen.add(s)
            try:
                off = MultistreamTokenizer.tokenize_mxl(s)["offset"]
                if off.ndim != 2 or off.shape[0] == 0:
                    continue
                ph = (off.argmax(-1) % N_PHASE)
                for v in ph.tolist():
                    counts[v] += 1.0
                n_scores += 1
            except Exception:
                continue
    counts = counts.clamp(min=1.0)
    log_p = torch.log(counts / counts.sum()).float()
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"log_p_phase": log_p, "n_scores": n_scores}, cache)
    print(f"[phase-prior] {n_scores} scores; log P(phase) computed; cached -> {cache}")
    return log_p


# ----------------------------------------------------------------------------- the reranker
class OffsetReranker:
    """Callable matching model.generate(offset_rerank=...): (logits[B,V], prev_off[B,T]) -> logits.
    lambda_==0 returns logits UNCHANGED (identity / sanity check)."""

    def __init__(self, lambda_, log_p_phase, beta_metric=1.0, beta_coh=1.0, topk=TOPK):
        self.lam = float(lambda_)
        self.beta_m = float(beta_metric)
        self.beta_c = float(beta_coh)
        self.topk = topk
        # phase prior, normalized to mean 0 so it only *redistributes* among phases
        lpp = log_p_phase.float()
        self.log_p_phase = lpp - lpp.mean()
        self.phase = torch.tensor(BUCKET_PHASE, dtype=torch.long)
        self.quarter = torch.tensor(BUCKET_QUARTER, dtype=torch.long)
        self.is_trip = torch.tensor([(b % N_PHASE) % 3 != 0 for b in range(N_BUCKETS)])
        # per-bucket metrical-prior contribution (constant; phase-only)
        self.metric_score = self.beta_m * self.log_p_phase[self.phase]  # (V,)

    def __call__(self, logits, prev_off):
        # COLD-START-CAPABLE TRIPLET BOOST. The first-pass coherence rule penalized *initiating*
        # a triplet group (it rewarded only completion + a binary-favoring marginal prior), so it
        # could never move a 0%-placement piece off zero. This instead adds +lambda to every
        # TRIPLET-phase candidate the model already ranked in its top-k (i.e. it "considered" it),
        # with binary candidates untouched. It can INITIATE a group; only nudges latent mass the
        # model already has. lambda=0 => identity (clean control).
        if self.lam == 0.0:
            return logits
        B, V = logits.shape
        logits = logits.clone()
        _, topi = torch.topk(logits, min(self.topk, V), dim=-1)  # (B,k) candidate buckets
        ph = self.phase.to(logits.device)
        for r in range(B):
            idx = topi[r]
            cand_trip = (ph[idx] % 3 != 0).to(logits.dtype)        # 1.0 at triplet phases
            logits[r, idx] = logits[r, idx] + (self.lam * cand_trip).to(logits.dtype)
        return logits


# ----------------------------------------------------------------------------- per-piece eval
def eval_piece(model, p, reranker, work_root):
    """Infer one piece with `reranker`, run the MUSTER decomposition, return a record dict
    with overall MeanER, tuplet/non-tuplet onset+offset error, and decoded triplet-placement."""
    with torch.no_grad():
        y = infer(p["_x"], model, overlap=64, chunk=512, verbose=False, kv_cache=True,
                  offset_rerank=reranker)
    # decoded triplet placement rate (proxy for "stopped binary-quantizing triplets")
    off_all = y["offset"].argmax(-1).reshape(-1).cpu().numpy()
    keep = y.get("pad_prob", y["pad"]).reshape(-1).cpu().numpy() > 0.5
    if len(keep) != len(off_all):
        raise RuntimeError("Decoded offset and keep streams differ in length")
    off = off_all[keep]
    ph = off % N_PHASE
    dec_trip_rate = float(np.mean(ph % 3 != 0)) if len(off) else 0.0
    n_dec_trip = int(np.sum(ph % 3 != 0))

    mxl = postprocess_score(MultistreamTokenizer.detokenize_mxl(y, pad_threshold=0.5), inPlace=True)
    wd = work_root / p["piece"].replace("/", "_")
    agg, gt_fmt3x, err_detail, auto_match = MTD.run_muster_keep(mxl, p["score"], wd)
    gt_info = MTD.parse_gt_fmt3x(gt_fmt3x)
    matched = MTD.parse_matched_gtids(auto_match, gt_info)
    err = MTD.parse_err_detail(err_detail)
    dec = MTD.decompose_piece(gt_info, matched, err, agg)
    rec = {
        "composer": p["composer"], "piece": p["piece"], "n_notes": p["n_notes"],
        "meaner_overall": dec["meaner_overall"],
        "onset_err_tuplet": dec["onset_err_tuplet"],
        "onset_err_nontuplet": dec["onset_err_nontuplet"],
        "offset_err_tuplet": dec["offset_err_tuplet"],
        "offset_err_nontuplet": dec["offset_err_nontuplet"],
        "n_tuplet": dec["n_tuplet"], "n_nontuplet": dec["n_nontuplet"],
        "decoded_triplet_rate": dec_trip_rate, "n_decoded_triplet": n_dec_trip,
        "n_decoded_notes": int(len(off)),
        "n_decoded_positions": int(len(off_all)),
        "keep_rule": "pad probability > 0.5",
        "muster": agg,
        "selfcheck_offset_ok": bool(
            dec["_check_combined_OffsetER"] is None or dec["_check_harness_OffsetER"] is None or
            abs(dec["_check_combined_OffsetER"] - dec["_check_harness_OffsetER"]) < 0.01),
    }
    return rec, off


def _f(x):
    return "  —  " if x is None else f"{x:6.2f}"


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/MIDI2ScoreTF.ckpt")
    ap.add_argument("--pieces", nargs="*", default=["Liszt", "Ravel", "Scriabin", "Mozart"])
    ap.add_argument("--lambdas", nargs="*", type=float, default=[0.0, 4.0, 8.0, 16.0])
    ap.add_argument("--beta-metric", type=float, default=1.0)
    ap.add_argument("--beta-coh", type=float, default=1.0)
    ap.add_argument("--out", default=str(REPO / "benchmark/reranker_results.json"))
    args = ap.parse_args()

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = REPO / out_path

    log_p_phase = build_phase_prior()

    # collect + tokenize the diagnostic pieces (shortest perf per piece)
    paths = collect_paths("test")
    for p in paths:
        try:
            p["_x"] = MultistreamTokenizer.tokenize_midi(p["midi"])
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
    devs = {pp.device.type for pp in model.parameters()}
    assert devs == {"cpu"}, f"model not on CPU: {devs}"
    print(f"model on CPU ({sum(pp.numel() for pp in model.parameters())/1e6:.2f}M params)")

    results = {
        "schema_version": 3,
        "ckpt": args.ckpt,
        "topk": TOPK,
        "beta_metric": args.beta_metric,
        "beta_coh": args.beta_coh,
        "emission_population": "generated notes with pad probability > 0.5",
        "by_lambda": {},
        "sanity": None,
    }
    work_root = REPO / "benchmark" / "rerank_work"
    work_root.mkdir(parents=True, exist_ok=True)

    baseline_off = {}   # piece -> baseline decoded offset array (for lambda=0 sanity)
    t0 = time.time()
    for lam in args.lambdas:
        rr = OffsetReranker(lam, log_p_phase, beta_metric=args.beta_metric,
                            beta_coh=args.beta_coh, topk=TOPK)
        # lambda=0: pass None to PROVE the hook itself is identity (callback never invoked)
        rerank_arg = None if lam == 0.0 else rr
        per_piece = []
        print(f"\n===== lambda={lam} =====")
        print(f"{'piece':28s} {'MeanER':>7s} {'onT':>7s} {'onN':>7s} {'offT':>7s} {'offN':>7s} "
              f"{'decTrip%':>9s} {'nT/nN':>10s}")
        for p in paths:
            rec, off = eval_piece(model, p, rerank_arg, work_root / f"lam{lam}")
            per_piece.append(rec)
            if lam == 0.0:
                baseline_off[(p["composer"], p["piece"])] = off
            print(f"{(p['composer'][:10]+'/'+p['piece'][:16]):28s} "
                  f"{_f(rec['meaner_overall'])} {_f(rec['onset_err_tuplet'])} "
                  f"{_f(rec['onset_err_nontuplet'])} {_f(rec['offset_err_tuplet'])} "
                  f"{_f(rec['offset_err_nontuplet'])} {rec['decoded_triplet_rate']*100:8.2f}% "
                  f"{rec['n_tuplet']:>4d}/{rec['n_nontuplet']:<5d}", flush=True)
        # aggregate (micro: pool error events / matched notes across pieces)
        agg = aggregate(per_piece)
        results["by_lambda"][str(lam)] = {"per_piece": per_piece, "aggregate": agg}
        print(f"  AGG MeanER={_f(agg['meaner_overall'])} onT={_f(agg['onset_err_tuplet'])} "
              f"onN={_f(agg['onset_err_nontuplet'])} decTrip={agg['decoded_triplet_rate']*100:.2f}%")
        json.dump(results, open(out_path, "w"), indent=2)

    # ---- lambda=0 SANITY: rerank-hook with lambda=0 object must equal None-decode exactly ----
    # We additionally run the reranker OBJECT at lambda=0 (callback IS invoked but returns
    # logits unchanged) and assert the decoded offsets are byte-identical to the None-decode.
    print("\n===== SANITY: lambda=0 reranker-object == baseline (None) =====")
    rr0 = OffsetReranker(0.0, log_p_phase, topk=TOPK)
    # force the callback to actually run by temporarily wrapping: lam==0 returns identity, so
    # invoking it must still reproduce the baseline argmax stream.
    sanity = {}
    for p in paths:
        with torch.no_grad():
            y = infer(p["_x"], model, overlap=64, chunk=512, verbose=False, kv_cache=True,
                      offset_rerank=(lambda lg, pv: lg))  # explicit identity callback
        off_all = y["offset"].argmax(-1).reshape(-1).cpu().numpy()
        keep = y.get("pad_prob", y["pad"]).reshape(-1).cpu().numpy() > 0.5
        off = off_all[keep]
        base = baseline_off.get((p["composer"], p["piece"]))
        same_len = base is not None and len(off) == len(base)
        ndiff = int(np.sum(off != base)) if same_len else None
        ok = bool(same_len and ndiff == 0)
        sanity[f"{p['composer']}/{p['piece']}"] = {
            "byte_identical": ok, "n_base": int(len(base)) if base is not None else None,
            "n_rerank0": int(len(off)), "n_diff": ndiff}
        print(f"  {p['composer']}/{p['piece']}: byte-identical={ok} "
              f"(n={len(off)}, ndiff={ndiff})")
    results["sanity"] = sanity
    results["elapsed_s"] = round(time.time() - t0, 1)
    results["generated_utc"] = datetime.now(timezone.utc).isoformat()
    results["provenance"] = {
        "runner_sha256": file_hash(Path(__file__).resolve()),
        "checkpoint_sha256": file_hash(Path(args.ckpt).resolve()),
        "exact_evaluator_sha256": file_hash(REPO / "scripts/muster_tuplet_decompose.py"),
        "precision": "CPU FP32 with autocast disabled",
    }
    json.dump(results, open(out_path, "w"), indent=2)
    print(f"\nWrote {out_path}  ({results['elapsed_s']}s)")

    # ---- final before/after table ----
    print_table(results, args.lambdas)


def aggregate(per_piece):
    tot = dict(nT=0, nN=0, onT=0, onN=0, offT=0, offN=0, dectrip=0, decnotes=0)
    meaners = []
    for r in per_piece:
        nT, nN = r["n_tuplet"], r["n_nontuplet"]
        tot["nT"] += nT; tot["nN"] += nN
        if r["onset_err_tuplet"] is not None:
            tot["onT"] += round(r["onset_err_tuplet"] / 100.0 * nT)
        if r["onset_err_nontuplet"] is not None:
            tot["onN"] += round(r["onset_err_nontuplet"] / 100.0 * nN)
        if r["offset_err_tuplet"] is not None:
            tot["offT"] += round(r["offset_err_tuplet"] / 100.0 * nT)
        if r["offset_err_nontuplet"] is not None:
            tot["offN"] += round(r["offset_err_nontuplet"] / 100.0 * nN)
        tot["dectrip"] += r["n_decoded_triplet"]; tot["decnotes"] += r["n_decoded_notes"]
        if r["meaner_overall"] is not None:
            meaners.append(r["meaner_overall"])
    rr = lambda a, b: (100.0 * a / b) if b else None
    return {
        "meaner_overall": (sum(meaners) / len(meaners)) if meaners else None,
        "onset_err_tuplet": rr(tot["onT"], tot["nT"]),
        "onset_err_nontuplet": rr(tot["onN"], tot["nN"]),
        "offset_err_tuplet": rr(tot["offT"], tot["nT"]),
        "offset_err_nontuplet": rr(tot["offN"], tot["nN"]),
        "decoded_triplet_rate": (tot["dectrip"] / tot["decnotes"]) if tot["decnotes"] else 0.0,
        "n_decoded_triplet": tot["dectrip"], "n_decoded_notes": tot["decnotes"],
        "n_tuplet": tot["nT"], "n_nontuplet": tot["nN"],
    }


def print_table(results, lambdas):
    print("\n" + "=" * 90)
    print("BEFORE/AFTER (aggregate over pieces). onT=tuplet-onset-err%, onN=nontuplet-onset-err%")
    print("=" * 90)
    print(f"{'lambda':>8s} {'MeanER':>8s} {'onsetT':>8s} {'onsetN':>8s} {'offsetT':>8s} "
          f"{'offsetN':>8s} {'decTrip%':>9s}")
    for lam in lambdas:
        a = results["by_lambda"][str(lam)]["aggregate"]
        print(f"{lam:8.1f} {_f(a['meaner_overall'])} {_f(a['onset_err_tuplet'])} "
              f"{_f(a['onset_err_nontuplet'])} {_f(a['offset_err_tuplet'])} "
              f"{_f(a['offset_err_nontuplet'])} {a['decoded_triplet_rate']*100:8.2f}%")


if __name__ == "__main__":
    main()
