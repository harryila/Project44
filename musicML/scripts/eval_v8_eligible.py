"""Evaluate candidates from a guarded_selector_v8 JSON artifact."""

import argparse
import json
from pathlib import Path

import local_repair_v4 as V4
import likelihood_timing_selector_v6 as V6
import verifier_bestofn as VB


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v8-json", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pieces", nargs="*", default=None)
    ap.add_argument("--all-candidates", action="store_true")
    args = ap.parse_args()
    v8_path = Path(args.v8_json)
    if not v8_path.is_absolute():
        v8_path = VB.REPO / v8_path
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = VB.REPO / out_path

    v8 = json.load(open(v8_path))
    wanted = {}
    for piece_key, rec in v8["pieces"].items():
        if args.pieces and not any(s.lower() in piece_key.lower() for s in args.pieces):
            continue
        if args.all_candidates:
            wanted[piece_key] = [c["i"] for c in rec["candidates"]]
        else:
            wanted[piece_key] = [c["i"] for c in rec["candidates"] if c.get("eligible")]

    class A:
        pieces = []

    pieces = sorted({k.split("/")[0] for k in wanted})
    A.pieces = pieces
    paths = V6.piece_paths(A)
    model = VB.load_any_checkpoint(v8["ckpt"], "cpu")
    model.eval()
    model.to("cpu")
    streams = v8.get("streams", ["offset", "duration"])
    overrides = {s: (v8["topk"], v8["temp"]) for s in streams}
    work_root = VB.REPO / "benchmark" / "v8_eligible_work"
    results = {"source": str(v8_path), "pieces": {}}

    for p in paths:
        key = f"{p['composer']}/{p['piece']}"
        if key not in wanted:
            continue
        tag = p["piece"].replace("/", "_")
        print(f"\n===== {key} =====", flush=True)
        y_base = VB.decode(model, p["_x"], None, seed=1)
        p["_y_base"] = y_base
        gt_total, _ = V4.gt_tuplet_total(p, work_root / tag / "gtcount")
        rec_b = VB.muster_piece(y_base, p, work_root / tag / "baseline")
        tpf_b = V4.tpf(rec_b, gt_total)
        out_piece = {
            "gt_tuplets": gt_total,
            "baseline": {**rec_b, "tpf": tpf_b[0], "tpf_missed": tpf_b[1], "tpf_onset": tpf_b[2]},
            "candidates": [],
        }
        for i in wanted[key]:
            y = VB.decode(model, p["_x"], overrides, seed=1000 + i)
            rec = VB.muster_piece(y, p, work_root / tag / f"cand{i:02d}")
            tpf = V4.tpf(rec, gt_total)
            cand_meta = next(c for c in v8["pieces"][key]["candidates"] if c["i"] == i)
            row = {
                **cand_meta,
                "metrics": {**rec, "tpf": tpf[0], "tpf_missed": tpf[1], "tpf_onset": tpf[2]},
            }
            out_piece["candidates"].append(row)
            print(f"  cand{i:02d}: MeanER={rec['meaner_overall']:.2f} "
                  f"TPF={tpf[0]} timing={cand_meta['timing_per_group']:.6f} "
                  f"model={cand_meta['model_loss']:.4f}", flush=True)
            json.dump(results, open(out_path, "w"), indent=2)
        results["pieces"][key] = out_piece
        json.dump(results, open(out_path, "w"), indent=2)

    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
