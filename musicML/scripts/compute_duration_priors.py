"""Compute duration-placement priors for the inference-time tuplet levers (A1 + A2).

From correctly-notated classical scores (ASAP TRAIN+VALIDATION engraved MusicXML — NEVER test, to
avoid leakage), accumulate over the DURATION output stream on the 1/24-quarter grid:

  A1  log_pi_dur[d]              = log marginal P(duration bucket d)            (Menon logit-adjustment)
  A2  log_p_dur_given_phase[p,d] = log P(duration d | metrical phase p)         (Shibata metrical prior)
      where phase p = offset_bucket % 24  (sub-quarter phase: triplets land at {8,16}, binary at {6,12,18})

Saved as a torch dict consumed by model.generate(dur_log_pi=..., dur_metrical=...).

Run (box):  PYTHONPATH=MIDI2ScoreTransformer/midi2scoretransformer \
              venv311/bin/python scripts/compute_duration_priors.py --out data/duration_priors.pt
"""
import argparse
import os
import sys
import warnings
from pathlib import Path

warnings.simplefilter("ignore")
REPO = Path(__file__).resolve().parent.parent
TF = REPO / "MIDI2ScoreTransformer"
sys.path.insert(0, str(TF / "midi2scoretransformer"))
sys.path.insert(0, str(REPO / "benchmark"))
os.chdir(TF)

import torch  # noqa: E402
from config import MyModelConfig  # noqa: E402
if not hasattr(MyModelConfig, "_attn_implementation_internal"):
    MyModelConfig._attn_implementation_internal = None
from tokenizer import MultistreamTokenizer  # noqa: E402
from eval_tier1_asap import collect_paths  # noqa: E402

EPS = 1e-6
N_PHASE = 24  # sub-quarter phases on the 1/24-quarter grid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/duration_priors.pt")
    ap.add_argument("--splits", nargs="*", default=["train", "validation"])
    args = ap.parse_args()

    # gather engraved-score paths from non-test splits (leakage-safe)
    score_paths, seen = [], set()
    for sp in args.splits:
        for p in collect_paths(sp):
            s = p.get("score")
            if s and s not in seen:
                seen.add(s); score_paths.append(s)
    print(f"{len(score_paths)} unique engraved scores from splits={args.splits}")

    pi = None          # (vocab_dur,)
    joint = None       # (N_PHASE, vocab_dur)
    n_notes = 0
    n_ok = 0
    for i, sp in enumerate(score_paths):
        try:
            toks = MultistreamTokenizer.tokenize_mxl(sp)
            dur = toks["duration"]          # (T, vocab_dur) one-hot float
            off = toks["offset"]            # (T, vocab_off) one-hot float
            if dur.ndim != 2 or off.ndim != 2 or dur.shape[0] == 0:
                continue
            vocab_dur = dur.shape[1]
            if pi is None:
                pi = torch.zeros(vocab_dur, dtype=torch.float64)
                joint = torch.zeros(N_PHASE, vocab_dur, dtype=torch.float64)
            d_idx = dur.argmax(-1)                       # (T,)
            o_idx = off.argmax(-1)                       # (T,)
            phase = (o_idx % N_PHASE)                    # (T,)
            for d, ph in zip(d_idx.tolist(), phase.tolist()):
                pi[d] += 1.0
                joint[ph, d] += 1.0
            n_notes += d_idx.numel(); n_ok += 1
        except Exception:
            continue
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(score_paths)} scores, {n_notes} notes", flush=True)

    assert pi is not None and n_notes > 0, "no notes accumulated"
    # log marginal (A1)
    log_pi = torch.log(pi / pi.sum() + EPS).float()
    # log conditional P(dur|phase) (A2): row-normalize, floor for unseen
    row = joint.sum(1, keepdim=True).clamp(min=1.0)
    log_cond = torch.log(joint / row + EPS).float()      # (N_PHASE, vocab_dur)

    # report: tuplet (non-dyadic on 1/24 grid) mass overall and at triplet vs binary phases
    vocab_dur = pi.shape[0]
    is_tuplet = torch.tensor([(d % 3 != 0) for d in range(vocab_dur)])
    tup_frac = (pi[is_tuplet].sum() / pi.sum()).item()
    out = REPO / args.out if not os.path.isabs(args.out) else Path(args.out)
    torch.save({"log_pi_dur": log_pi, "log_p_dur_given_phase": log_cond,
                "n_phase": N_PHASE, "vocab_dur": vocab_dur, "n_notes": n_notes,
                "tuplet_frac": tup_frac}, out)
    print(f"scores ok={n_ok} notes={n_notes} vocab_dur={vocab_dur} overall tuplet_frac={tup_frac:.4f}")
    # sanity: at a clearly-triplet phase (8) vs binary phase (12), which durations dominate?
    for ph in (0, 6, 8, 12, 16, 18):
        top = log_cond[ph].topk(3).indices.tolist()
        print(f"  phase {ph:2d}: top duration buckets {top} (tuplet={[ (d%3!=0) for d in top]})")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
