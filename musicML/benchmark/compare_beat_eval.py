#!/usr/bin/env python3
"""Decompose the beat-conditioning eval: A (baseline) / C (beat ckpt, no beats) /
B (beat ckpt, gold beats), aligned per-performance, sorted hardest-first.

  C - A  = warm-start DRIFT (did beat-training change the no-beat behavior?)
  B - C  = BEAT SIGNAL effect (same weights, only the beat input differs — the clean test)
  B - A  = NET effect of the whole intervention vs published baseline.

The headline question is whether B beats A/C on the HARD (high-baseline-MeanER,
tuplet/multi-meter) pieces without regressing the easy ones.
"""
import argparse
import json
from pathlib import Path


def load(path):
    d = json.load(open(path))
    per = {}
    for r in d.get("per_performance", []):
        mer = (r.get("sim") or {}).get("muster", {})
        key = (r.get("composer"), r.get("piece"), r.get("midi"))
        per[key] = {
            "MeanER": mer.get("MeanER"),
            "OnsetER": mer.get("OnsetER"),
            "MissRate": mer.get("MissRate"),
            "ExtraRate": mer.get("ExtraRate"),
            "n_notes": r.get("n_notes"),
            "error": r.get("error"),
        }
    agg = (d.get("aggregate") or {}).get("muster", {})
    meta = d.get("meta", {})
    return per, agg, meta


def fmt(v, w=6, p=2):
    return f"{v:{w}.{p}f}" if isinstance(v, (int, float)) else " " * (w - 3) + "n/a"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="benchmark/eval_A_baseline.json", help="baseline")
    ap.add_argument("--c", default="benchmark/eval_C_beat_nobeat.json", help="beat ckpt, no beats")
    ap.add_argument("--b", default="benchmark/eval_B_beat_gold.json", help="beat ckpt, gold beats")
    args = ap.parse_args()

    A, aggA, metaA = load(args.a)
    C, aggC, metaC = load(args.c)
    B, aggB, metaB = load(args.b)
    keys = sorted(set(A) | set(C) | set(B),
                  key=lambda k: -(A.get(k, {}).get("MeanER") or -1))

    print("=" * 108)
    print("BEAT-CONDITIONING 3-WAY  (MeanER, lower=better) — sorted hardest-first by baseline")
    print(f"  A baseline      = {metaA.get('ckpt')}")
    print(f"  C beat,no-beat  = {metaC.get('ckpt')}")
    print(f"  B beat,gold     = {metaB.get('ckpt')}  (gold beats on {metaB.get('n_with_gold_beats','?')} perfs)")
    print("=" * 108)
    hdr = f"{'piece':40s} {'notes':>6s} | {'A':>6s} {'C':>6s} {'B':>6s} | {'C-A':>6s} {'B-C':>6s} {'B-A':>6s}"
    print(hdr)
    print("-" * 108)

    sums = {"A": [], "C": [], "B": []}
    win = lose = flat = 0
    for k in keys:
        a = A.get(k, {}).get("MeanER")
        c = C.get(k, {}).get("MeanER")
        b = B.get(k, {}).get("MeanER")
        n = (A.get(k) or C.get(k) or B.get(k) or {}).get("n_notes")
        name = f"{k[0]}/{k[1]}"[:40]
        ca = (c - a) if isinstance(a, (int, float)) and isinstance(c, (int, float)) else None
        bc = (b - c) if isinstance(b, (int, float)) and isinstance(c, (int, float)) else None
        ba = (b - a) if isinstance(b, (int, float)) and isinstance(a, (int, float)) else None
        print(f"{name:40s} {fmt(n,6,0)} | {fmt(a)} {fmt(c)} {fmt(b)} | "
              f"{fmt(ca)} {fmt(bc)} {fmt(ba)}")
        if isinstance(a, (int, float)):
            sums["A"].append(a)
        if isinstance(c, (int, float)):
            sums["C"].append(c)
        if isinstance(b, (int, float)):
            sums["B"].append(b)
        if isinstance(ba, (int, float)):
            if ba < -0.5:
                win += 1
            elif ba > 0.5:
                lose += 1
            else:
                flat += 1

    print("-" * 108)

    def mean(xs):
        return sum(xs) / len(xs) if xs else float("nan")
    mA, mC, mB = mean(sums["A"]), mean(sums["C"]), mean(sums["B"])
    print(f"{'MEAN over scored perfs':40s} {'':>6s} | {mA:6.2f} {mC:6.2f} {mB:6.2f} | "
          f"{mC-mA:6.2f} {mB-mC:6.2f} {mB-mA:6.2f}")
    print(f"\nPer-piece net (B vs A):  wins(<-0.5)={win}  losses(>+0.5)={lose}  flat={flat}")
    print("\nAggregate MUSTER (from each run's own aggregate block):")
    for tag, agg in [("A baseline", aggA), ("C beat,no-beat", aggC), ("B beat,gold", aggB)]:
        ks = ["PitchER", "MissRate", "ExtraRate", "OnsetER", "OffsetER", "MeanER"]
        print(f"  {tag:16s} " + "  ".join(f"{k}={fmt(agg.get(k),5,2).strip()}" for k in ks))
    print("\nReading: C-A<0 => beat-training improved even WITHOUT beats; B-C<0 => the beat "
          "signal helps;\n  B-A<0 on hard/tuplet pieces with easy pieces flat = the win.")


if __name__ == "__main__":
    main()
