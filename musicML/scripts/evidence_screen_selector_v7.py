"""V7 evidence-screened best-of-N selector.

Chunked teacher-forced scoring is required for long held-out pieces, but it compresses model
loss enough that a z-sum selector can prefer generic high-likelihood donors over the donor
with the strongest timing evidence. V7 makes the intended priority explicit:

  1. rank donors by V3b-style timing contradiction per group;
  2. keep the best evidence screen (default: top 25%, at least 2 donors);
  3. choose the lowest chunked teacher-forced rhythm loss inside that screen.

This is still donor-only best-of-N. The baseline is evaluated and reported, but not selected
unless --baseline-veto is set and no screened donor improves the baseline timing score.
"""

import argparse
import json
import math
import time
from pathlib import Path

import local_repair_v4 as V4
import likelihood_timing_selector_v6 as V6
import verifier_bestofn as VB


def select_screened(records, screen_frac):
    k = max(2, int(math.ceil(len(records) * screen_frac)))
    screened = sorted(records, key=lambda r: (r["timing_per_group"], r["i"]))[:k]
    pick = min(screened, key=lambda r: (r["model_loss"], r["i"]))
    for r in records:
        r["in_timing_screen"] = any(s["i"] == r["i"] for s in screened)
    return pick, screened


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/MIDI2ScoreTF.ckpt")
    ap.add_argument("--pieces", nargs="*", default=["Scriabin"])
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--topk", type=int, default=15)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--screen-frac", type=float, default=0.25)
    ap.add_argument("--baseline-veto", action="store_true",
                    help="select baseline if the screened donor does not improve baseline timing")
    ap.add_argument("--out", default=str(VB.REPO / "benchmark/evidence_screen_selector_v7_dev.json"))
    args = ap.parse_args()
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = VB.REPO / out_path

    paths = V6.piece_paths(args)
    print(f"{len(paths)} pieces: " + ", ".join(f"{p['composer']}/{p['piece']}" for p in paths))
    model = VB.load_any_checkpoint(args.ckpt, "cpu")
    model.eval()
    model.to("cpu")
    overrides = {"offset": (args.topk, args.temp), "duration": (args.topk, args.temp)}
    work_root = VB.REPO / "benchmark" / "v7_work"
    results = {
        "ckpt": args.ckpt,
        "n": args.n,
        "topk": args.topk,
        "temp": args.temp,
        "selector": "top timing-evidence screen, then lowest chunked teacher-forced rhythm loss",
        "screen_frac": args.screen_frac,
        "baseline_veto": bool(args.baseline_veto),
        "pieces": {},
    }
    t0 = time.time()

    for p in paths:
        tag = p["piece"].replace("/", "_")
        print(f"\n===== {p['composer']}/{p['piece']} ({p['n_notes']} notes) =====", flush=True)
        y_base = VB.decode(model, p["_x"], None, seed=1)
        p["_y_base"] = y_base
        gt_total, _ = V4.gt_tuplet_total(p, work_root / tag / "gtcount")
        rec_b = VB.muster_piece(y_base, p, work_root / tag / "baseline")
        tpf_b = V4.tpf(rec_b, gt_total)
        base_loss, base_loss_parts = V6.teacher_forced_rhythm_loss(model, p["_x"], y_base, p["n_notes"])
        base_ev, base_ev_detail = V6.timing_contradiction(y_base, p["_t"], p["n_notes"])
        print(f"  baseline: MeanER={rec_b['meaner_overall']:.2f} TPF={tpf_b[0]} "
              f"model_loss={base_loss:.4f} timing={base_ev:.6f}", flush=True)

        records = []
        donors_y = []
        for i in range(args.n):
            y_c = VB.decode(model, p["_x"], overrides, seed=1000 + i)
            donors_y.append(y_c)
            model_loss, loss_parts = V6.teacher_forced_rhythm_loss(model, p["_x"], y_c, p["n_notes"])
            ev, ev_detail = V6.timing_contradiction(y_c, p["_t"], p["n_notes"])
            records.append({
                "i": i,
                "seed": 1000 + i,
                "model_loss": round(model_loss, 8),
                "loss_parts": loss_parts,
                **ev_detail,
                "timing_per_group": ev,
            })
            print(f"  cand{i:02d}: model_loss={model_loss:.4f} timing={ev:.6f}", flush=True)

        pick, screened = select_screened(records, args.screen_frac)
        selected_baseline = False
        if args.baseline_veto and pick["timing_per_group"] >= base_ev:
            selected_baseline = True
            rec_p = rec_b
            tpf_p = tpf_b
            print("  --> baseline veto fired", flush=True)
        else:
            rec_p = VB.muster_piece(
                donors_y[pick["i"]], p, work_root / tag / f"pick_cand{pick['i']:02d}"
            )
            tpf_p = V4.tpf(rec_p, gt_total)
        gate_ok = tpf_p[0] <= 50 and rec_p["meaner_overall"] <= rec_b["meaner_overall"] + 0.5
        pick_label = "baseline" if selected_baseline else f"cand{pick['i']:02d}"
        print(f"  --> pick {pick_label}: MeanER={rec_p['meaner_overall']:.2f} "
              f"TPF={tpf_p[0]} gate_ok={gate_ok}", flush=True)

        results["pieces"][f"{p['composer']}/{p['piece']}"] = {
            "gt_tuplets": gt_total,
            "baseline": {
                **rec_b,
                "tpf": tpf_b[0],
                "tpf_missed": tpf_b[1],
                "tpf_onset": tpf_b[2],
                "model_loss": round(base_loss, 8),
                "loss_parts": base_loss_parts,
                **base_ev_detail,
            },
            "candidates": records,
            "timing_screen": [r["i"] for r in screened],
            "selector_pick": {
                "selected_baseline": bool(selected_baseline),
                **pick,
                "metrics": {
                    **rec_p,
                    "tpf": tpf_p[0],
                    "tpf_missed": tpf_p[1],
                    "tpf_onset": tpf_p[2],
                },
            },
            "gate_ok": bool(gate_ok),
        }
        json.dump(results, open(out_path, "w"), indent=2)

    results["elapsed_s"] = round(time.time() - t0, 1)
    json.dump(results, open(out_path, "w"), indent=2)
    print(f"\nWrote {out_path} ({results['elapsed_s']}s)")


if __name__ == "__main__":
    main()
