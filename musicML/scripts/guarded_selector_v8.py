"""V8 guarded best-of-N selector.

V7 showed that timing evidence alone can pick globally damaged candidates on the hard tail.
V8 keeps V7's useful evidence-first idea, but makes three guards explicit:

  * baseline competes, so donors must improve timing evidence to be selected;
  * donor keep-count must stay close to baseline keep-count;
  * donor teacher-forced rhythm loss must stay within a small absolute delta of baseline.

Among candidates passing those guards, V8 keeps the best timing screen and chooses the
lowest model-loss donor. If none pass, it returns the argmax baseline.
"""

import argparse
import json
import math
import time
from pathlib import Path

import local_repair_v4 as V4
import likelihood_timing_selector_v6 as V6
import verifier_bestofn as VB


def keep_count(y, n_real):
    s = V4.streams_np(y)
    n = min(n_real, len(s["pad"]))
    return int(s["pad"][:n].sum())


def trip_phase_count(y, n_real):
    s = V4.streams_np(y)
    n = min(n_real, len(s["off"]))
    return int(((s["off"][:n] % 24) % 3 != 0).sum())


def choose(records, base_record, screen_frac, max_keep_delta_frac, max_model_delta):
    gated = []
    base_keep = max(base_record["keep_count"], 1)
    for r in records:
        r["timing_improves_baseline"] = r["timing_per_group"] < base_record["timing_per_group"]
        r["keep_delta_frac"] = abs(r["keep_count"] - base_record["keep_count"]) / base_keep
        r["coverage_ok"] = r["keep_delta_frac"] <= max_keep_delta_frac
        r["model_delta"] = r["model_loss"] - base_record["model_loss"]
        r["model_ok"] = r["model_delta"] <= max_model_delta
        r["eligible"] = (
            r["timing_improves_baseline"]
            and r["coverage_ok"]
            and r["model_ok"]
        )
        if r["eligible"]:
            gated.append(r)

    if not gated:
        return None, []

    k = max(1, int(math.ceil(len(gated) * screen_frac)))
    screened = sorted(gated, key=lambda r: (r["timing_per_group"], r["i"]))[:k]
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
    ap.add_argument("--streams", nargs="*", default=["offset", "duration"])
    ap.add_argument("--eval-all", action="store_true")
    ap.add_argument("--screen-frac", type=float, default=0.5,
                    help="fraction of eligible candidates kept after timing sort")
    ap.add_argument("--max-keep-delta-frac", type=float, default=0.03)
    ap.add_argument("--max-model-delta", type=float, default=0.03)
    ap.add_argument("--out", default=str(VB.REPO / "benchmark/guarded_selector_v8_dev.json"))
    args = ap.parse_args()
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = VB.REPO / out_path

    paths = V6.piece_paths(args)
    print(f"{len(paths)} pieces: " + ", ".join(f"{p['composer']}/{p['piece']}" for p in paths))
    model = VB.load_any_checkpoint(args.ckpt, "cpu")
    model.eval()
    model.to("cpu")
    overrides = {s: (args.topk, args.temp) for s in args.streams}
    work_root = VB.REPO / "benchmark" / "v8_work"
    results = {
        "ckpt": args.ckpt,
        "n": args.n,
        "topk": args.topk,
        "temp": args.temp,
        "streams": args.streams,
        "selector": "baseline competes; timing-improving, coverage-safe, model-safe donors only",
        "screen_frac": args.screen_frac,
        "max_keep_delta_frac": args.max_keep_delta_frac,
        "max_model_delta": args.max_model_delta,
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
        base_record = {
            "model_loss": round(base_loss, 8),
            "loss_parts": base_loss_parts,
            **base_ev_detail,
            "timing_per_group": base_ev,
            "keep_count": keep_count(y_base, p["n_notes"]),
            "trip_phase_count": trip_phase_count(y_base, p["n_notes"]),
        }
        print(f"  baseline: MeanER={rec_b['meaner_overall']:.2f} TPF={tpf_b[0]} "
              f"model_loss={base_loss:.4f} timing={base_ev:.6f} "
              f"keep={base_record['keep_count']}", flush=True)

        records = []
        donors_y = []
        for i in range(args.n):
            y_c = VB.decode(model, p["_x"], overrides, seed=1000 + i)
            donors_y.append(y_c)
            model_loss, loss_parts = V6.teacher_forced_rhythm_loss(model, p["_x"], y_c, p["n_notes"])
            ev, ev_detail = V6.timing_contradiction(y_c, p["_t"], p["n_notes"])
            kc = keep_count(y_c, p["n_notes"])
            metrics = None
            if args.eval_all:
                cand_rec = VB.muster_piece(y_c, p, work_root / tag / f"cand{i:02d}")
                cand_tpf = V4.tpf(cand_rec, gt_total)
                metrics = {
                    **cand_rec,
                    "tpf": cand_tpf[0],
                    "tpf_missed": cand_tpf[1],
                    "tpf_onset": cand_tpf[2],
                }
            rec = {
                "i": i,
                "seed": 1000 + i,
                "model_loss": round(model_loss, 8),
                "loss_parts": loss_parts,
                **ev_detail,
                "timing_per_group": ev,
                "keep_count": kc,
                "trip_phase_count": trip_phase_count(y_c, p["n_notes"]),
                "metrics": metrics,
            }
            records.append(rec)
            metric_s = ""
            if metrics is not None:
                metric_s = f" MeanER={metrics['meaner_overall']:.2f} TPF={metrics['tpf']}"
            print(f"  cand{i:02d}: model_loss={model_loss:.4f} timing={ev:.6f} "
                  f"keep={kc}{metric_s}", flush=True)

        pick, screened = choose(
            records,
            base_record,
            args.screen_frac,
            args.max_keep_delta_frac,
            args.max_model_delta,
        )
        if pick is None:
            selected_baseline = True
            rec_p = rec_b
            tpf_p = tpf_b
        else:
            selected_baseline = False
            if pick.get("metrics") is not None:
                rec_p = {k: v for k, v in pick["metrics"].items()
                         if k not in ("tpf", "tpf_missed", "tpf_onset")}
                tpf_p = (pick["metrics"]["tpf"], pick["metrics"]["tpf_missed"],
                         pick["metrics"]["tpf_onset"])
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
                **base_record,
            },
            "candidates": records,
            "timing_screen": [r["i"] for r in screened],
            "selector_pick": {
                "selected_baseline": bool(selected_baseline),
                **({} if pick is None else pick),
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
