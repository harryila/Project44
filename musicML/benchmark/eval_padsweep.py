"""Single-pass pad-threshold sweep: generate ONCE per piece, evaluate every threshold.

The AR generation is identical across pad thresholds (pad_prob + raw_* streams are produced
once); only the detokenize keep-gate changes. So we infer once per piece and loop the cheap
detokenize+MUSTER over all thresholds -- ~8x less GPU than re-running eval_tuplet.py per
threshold. Reports per-piece x per-threshold MeanER + predicted/gt tuplet & note counts, the
best threshold per piece, and the corpus mean per threshold (= the calibration frontier).

Also doubles as the full-14-piece standing measurement (no --pieces filter => all test pieces).

Usage:
  venv311/bin/python benchmark/eval_padsweep.py --ckpt <ckpt> --device cuda \
     --out benchmark/padsweep_sslcc.json [--pieces Scriabin Mozart ...] \
     [--thresholds "0.60 0.55 0.50 0.45 0.40 0.35 0.30 0.25"]
"""
import argparse
import json
import os
import signal
import sys
import warnings
from pathlib import Path


class _Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise _Timeout()


signal.signal(signal.SIGALRM, _alarm)
warnings.simplefilter("ignore")

REPO_ROOT = Path(__file__).resolve().parent.parent
TF_ROOT = REPO_ROOT / "MIDI2ScoreTransformer"
sys.path.insert(0, str(TF_ROOT / "midi2scoretransformer"))
os.chdir(TF_ROOT)

import torch  # noqa: E402
from config import MyModelConfig  # noqa: E402
if not hasattr(MyModelConfig, "_attn_implementation_internal"):
    MyModelConfig._attn_implementation_internal = None
torch.serialization.add_safe_globals([MyModelConfig])

from tokenizer import MultistreamTokenizer  # noqa: E402
from utils import infer, eval as eval_pair  # noqa: E402
from score_utils import postprocess_score  # noqa: E402

sys.path.insert(0, str(REPO_ROOT / "benchmark"))
from eval_tier1_asap import collect_paths, load_any_checkpoint  # noqa: E402
import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))
from musicml_paths import device as _device


def count_tuplets(score):
    if score is None:
        return 0, 0
    nt = nn = 0
    for n in score.recurse().notes:
        nn += 1
        try:
            if n.duration.tuplets:
                nt += 1
        except Exception:
            pass
    return nt, nn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--device", default=_device())
    ap.add_argument("--pieces", nargs="*", default=None)
    ap.add_argument("--limit-per", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--overlap", type=int, default=64)
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--thresholds", default="0.60 0.55 0.50 0.45 0.40 0.35 0.30 0.25")
    # Inference-time DURATION placement levers (A1 logit-adjust, A2 metrical prior)
    ap.add_argument("--prior-path", default=None, help="torch dict from compute_duration_priors.py")
    ap.add_argument("--dur-tau", type=float, default=0.0, help="A1: subtract tau*log(prior) from duration logits")
    ap.add_argument("--dur-metrical-lambda", type=float, default=0.0, help="A2: add lambda*logP(dur|phase)")
    args = ap.parse_args()

    dur_log_pi = dur_metrical = None
    if args.prior_path and (args.dur_tau or args.dur_metrical_lambda):
        _pri = torch.load(args.prior_path, map_location="cpu", weights_only=False)
        if args.dur_tau:
            dur_log_pi = _pri["log_pi_dur"]
        if args.dur_metrical_lambda:
            dur_metrical = _pri["log_p_dur_given_phase"]
        print(f"[priors] {args.prior_path} tau={args.dur_tau} lambda={args.dur_metrical_lambda}", flush=True)

    thresholds = [float(t) for t in args.thresholds.split()]
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path

    paths = collect_paths("test")
    for p in paths:
        try:
            p["_x"] = MultistreamTokenizer.tokenize_midi(p["midi"])
            p["n_notes"] = int(p["_x"]["pitch"].shape[0])
        except Exception:
            p["_x"] = None
            p["n_notes"] = 1 << 30
    if args.pieces:
        paths = [p for p in paths
                 if any(s.lower() in (p["composer"] + "/" + p["piece"]).lower() for s in args.pieces)]
    paths.sort(key=lambda p: p["n_notes"])
    seen, kept = {}, []
    for p in paths:
        key = (p["composer"], p["piece"])
        if seen.get(key, 0) < args.limit_per and p["_x"] is not None:
            seen[key] = seen.get(key, 0) + 1
            kept.append(p)
    paths = kept

    model = load_any_checkpoint(args.ckpt, args.device)
    model.eval(); model.to(args.device)
    from music21 import converter

    results = []
    for i, p in enumerate(paths):
        rec = {k: p[k] for k in ("composer", "piece", "n_notes")}
        try:
            y_hat = infer(p["_x"], model, overlap=args.overlap, chunk=args.chunk,
                          verbose=False, kv_cache=True,             # <-- generate ONCE
                          dur_log_pi=dur_log_pi, dur_tau=args.dur_tau,
                          dur_metrical=dur_metrical, dur_metrical_lambda=args.dur_metrical_lambda)
            try:
                gt = converter.parse(p["score"])
                rec["gt_tuplets"], rec["gt_notes"] = count_tuplets(gt)
            except Exception:
                rec["gt_tuplets"] = rec["gt_notes"] = None
            per = {}
            for thr in thresholds:
                d = {}
                try:
                    signal.alarm(120)
                    pred = postprocess_score(
                        MultistreamTokenizer.detokenize_mxl(y_hat, pad_threshold=thr), inPlace=True)
                    d["pred_tuplets"], d["pred_notes"] = count_tuplets(pred)
                    signal.alarm(0)
                except _Timeout:
                    signal.alarm(0); d["detok_timeout"] = True
                except Exception as e:
                    signal.alarm(0); d["error"] = repr(e)[:120]
                try:
                    signal.alarm(180)
                    sim = eval_pair(y_hat, p["score"], pad_threshold=thr)
                    d["MeanER"] = (sim.get("muster") or {}).get("MeanER")
                    signal.alarm(0)
                except _Timeout:
                    signal.alarm(0); d["muster_timeout"] = True
                except Exception as e:
                    signal.alarm(0); d.setdefault("error", repr(e)[:120])
                per[f"{thr:.2f}"] = d
            rec["per_threshold"] = per
            row = " ".join(
                f"{t}={(per[t].get('MeanER') if isinstance(per[t].get('MeanER'),(int,float)) else '—')}"
                if not isinstance(per[t].get('MeanER'), float) else f"{t}={per[t]['MeanER']:.2f}"
                for t in (f"{x:.2f}" for x in thresholds))
            print(f"[{i+1}/{len(paths)}] {p['composer'][:8]}/{p['piece'][:22]:22s} {row}", flush=True)
        except _Timeout:
            signal.alarm(0); rec["error"] = "infer_timeout"
            print(f"[{i+1}/{len(paths)}] {p['composer']}/{p['piece'][:22]} INFER TIMEOUT", flush=True)
        except Exception:
            signal.alarm(0)
            import traceback
            rec["error"] = traceback.format_exc().splitlines()[-1]
            print(f"[{i+1}/{len(paths)}] {p['composer']}/{p['piece'][:22]} ERROR {rec['error']}", flush=True)
        results.append(rec)
        json.dump({"ckpt": args.ckpt, "thresholds": thresholds, "results": results},
                  open(out_path, "w"), indent=2)

    # ---- summary ----
    print("\n=== MeanER : piece x threshold ===")
    tcols = [f"{t:.2f}" for t in thresholds]
    hdr = "piece".ljust(32) + "".join(c.rjust(8) for c in tcols)
    print(hdr); print("-" * len(hdr))
    corpus = {t: [] for t in tcols}
    best_lines = []
    for r in results:
        per = r.get("per_threshold", {})
        name = f"{r['composer'][:8]}/{r['piece'][:20]}"
        cells = ""
        vals = {}
        for t in tcols:
            v = per.get(t, {}).get("MeanER")
            if isinstance(v, (int, float)):
                cells += f"{v:.2f}".rjust(8); corpus[t].append(v); vals[t] = v
            else:
                cells += "—".rjust(8)
        print(name.ljust(32) + cells)
        if vals:
            bt = min(vals, key=vals.get)
            best_lines.append(f"  {name.ljust(32)} best={bt} ({vals[bt]:.2f})  @0.50={vals.get('0.50','—')}")
    print("\n=== best threshold per piece (min MeanER) ===")
    print("\n".join(best_lines))
    print("\n=== corpus-mean MeanER per threshold ===")
    for t in tcols:
        if corpus[t]:
            print(f"  thr={t}: mean={sum(corpus[t])/len(corpus[t]):.3f}  (n={len(corpus[t])})")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
