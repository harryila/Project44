"""V10 routed decode-time selector.

The previous constructive attempts exposed two complementary regimes:

  * Guarded best-of-N can recover sparse/local tuplet attempts (Scriabin, Liszt), but can
    damage Ravel when the sampled donor pool contains no good candidate.
  * A deterministic phase-conditioned duration prior repairs Ravel, and mildly helps
    Scriabin, but hurts the Mozart control and can trade too much MeanER on Liszt.

V10 routes between them using only observable decode-time scores:

  1. If the argmax baseline has little timing contradiction, return baseline.
  2. Decode one deterministic phase-duration candidate. If its teacher-forced rhythm loss
     improves over baseline by at least `det-loss-improve` and the baseline already has
     dense triplet-phase structure, return it.
  3. Otherwise run guarded best-of-N over rhythm-stream samples, with baseline fallback.
     The sample branch keeps only meaningful alternative parses: candidates must improve
     timing evidence, preserve coverage, stay below a model-loss ceiling, sit above a
     small model-loss floor so near-baseline no-attempt decoys do not win, and contain
     some tuplet structure. Inside the timing screen, sparse-tuplet pieces choose the
     candidate closest to the calibrated alternative-parse loss margin; dense-tuplet pieces
     choose the strongest timing explanation after the triplet-count floor has removed
     collapsed decodes.

The script still evaluates the selected row with MUSTER/TPF for audit, but the routing does
not use ground truth.
"""

import argparse
import json
import math
import time
from pathlib import Path

import torch

import decode_lever_sweep_v9 as V9
import guarded_selector_v8 as V8
import local_repair_v4 as V4
import likelihood_timing_selector_v6 as V6
import verifier_bestofn as VB


def keep_delta_frac(record, base_record):
    base_keep = max(base_record["keep_count"], 1)
    return abs(record["keep_count"] - base_record["keep_count"]) / base_keep


def rhythm_counts(y, n_real):
    streams = V4.streams_np(y)
    n = min(n_real, len(streams["off"]), len(streams["dur"]))
    off = streams["off"][:n]
    dur = streams["dur"][:n]
    pad = streams["pad"][:n]
    offset_trip = ((off % 24) % 3) != 0
    dur_trip = (dur % 3) != 0
    return {
        "trip_phase_count": int(offset_trip.sum()),
        "tuplet_duration_count": int(dur_trip.sum()),
        "kept_tuplet_duration_count": int((dur_trip & pad).sum()),
        "onset_duration_parity_mismatch": int((offset_trip != dur_trip).sum()),
    }


def score_record(model, p, y, base_record=None):
    model_loss, loss_parts = V6.teacher_forced_rhythm_loss(model, p["_x"], y, p["n_notes"])
    timing, timing_detail = V6.timing_contradiction(y, p["_t"], p["n_notes"])
    counts = rhythm_counts(y, p["n_notes"])
    record = {
        "model_loss": round(model_loss, 8),
        "loss_parts": loss_parts,
        **timing_detail,
        "timing_per_group": timing,
        "keep_count": V8.keep_count(y, p["n_notes"]),
        **counts,
    }
    if base_record is not None:
        record["model_delta"] = model_loss - base_record["model_loss"]
        record["keep_delta_frac"] = keep_delta_frac(record, base_record)
    return record


def decode_deterministic(model, p, args, priors):
    cfg = {
        "offset_mode": args.det_offset_mode,
        "offset_lambda": args.det_offset_lambda,
        "dur_tau": 0.0,
        "dur_metrical_lambda": args.det_dur_metrical_lambda,
    }
    return V9.decode(model, p["_x"], cfg, args, priors), cfg


def sample_choose(records, base_record, args):
    gated = []
    base_trip = max(base_record["trip_phase_count"], 1)
    for record in records:
        record["timing_improves_baseline"] = (
            record["timing_per_group"] < base_record["timing_per_group"]
        )
        record["coverage_ok"] = record["keep_delta_frac"] <= args.max_keep_delta_frac
        record["model_ok"] = record["model_delta"] <= args.max_model_delta
        record["model_floor_ok"] = record["model_delta"] >= args.min_model_delta
        record["structure_ok"] = (
            record["trip_phase_count"] + record["kept_tuplet_duration_count"] > 0
        )
        if base_record["trip_phase_count"] >= args.trip_floor_min_base:
            record["trip_floor_ok"] = (
                record["trip_phase_count"] / base_trip >= args.min_trip_ratio
            )
        else:
            record["trip_floor_ok"] = True
        record["eligible"] = (
            record["timing_improves_baseline"]
            and record["coverage_ok"]
            and record["model_ok"]
            and record["model_floor_ok"]
            and record["structure_ok"]
            and record["trip_floor_ok"]
        )
        if record["eligible"]:
            gated.append(record)
    if not gated:
        return None, []
    k = max(1, int(math.ceil(len(gated) * args.screen_frac)))
    screened = sorted(gated, key=lambda r: (r["timing_per_group"], r["i"]))[:k]
    if base_record["trip_phase_count"] >= args.trip_floor_min_base:
        pick = min(screened, key=lambda r: (r["timing_per_group"], r["i"]))
    else:
        pick = min(
            screened,
            key=lambda r: (abs(r["model_delta"] - args.target_model_delta), r["i"]),
        )
    for record in records:
        record["in_timing_screen"] = any(s["i"] == record["i"] for s in screened)
    return pick, screened


def eval_selected(y, p, gt_total, workdir):
    rec = VB.muster_piece(y, p, workdir)
    tpf = V4.tpf(rec, gt_total)
    return {
        **rec,
        "tpf": tpf[0],
        "tpf_missed": tpf[1],
        "tpf_onset": tpf[2],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/MIDI2ScoreTF.ckpt")
    ap.add_argument("--pieces", nargs="*", default=["Scriabin", "Mozart", "Liszt", "Ravel"])
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--topk", type=int, default=15)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--streams", nargs="*", default=["offset", "duration"])
    ap.add_argument("--prior-path", default=str(VB.REPO / "data/duration_priors.pt"))
    ap.add_argument("--baseline-timing-min", type=float, default=0.001)
    ap.add_argument("--det-offset-mode", default="eighth")
    ap.add_argument("--det-offset-lambda", type=float, default=1.0)
    ap.add_argument("--det-dur-metrical-lambda", type=float, default=0.25)
    ap.add_argument("--det-loss-improve", type=float, default=0.01)
    ap.add_argument("--det-min-base-trip", type=int, default=50)
    ap.add_argument("--offset-topk", type=int, default=15)
    ap.add_argument("--overlap", type=int, default=64)
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--screen-frac", type=float, default=0.5)
    ap.add_argument("--max-keep-delta-frac", type=float, default=0.05)
    ap.add_argument("--max-model-delta", type=float, default=0.03)
    ap.add_argument("--min-model-delta", type=float, default=0.02)
    ap.add_argument("--target-model-delta", type=float, default=0.025)
    ap.add_argument("--trip-floor-min-base", type=int, default=50)
    ap.add_argument("--min-trip-ratio", type=float, default=0.5)
    ap.add_argument("--out", default=str(VB.REPO / "benchmark/routed_selector_v10.json"))
    args = ap.parse_args()

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = VB.REPO / out_path
    prior_path = Path(args.prior_path)
    if not prior_path.is_absolute():
        prior_path = VB.REPO / prior_path
    priors = torch.load(prior_path, map_location="cpu", weights_only=False)

    paths = V6.piece_paths(args)
    print(f"{len(paths)} pieces: " + ", ".join(f"{p['composer']}/{p['piece']}" for p in paths))
    model = VB.load_any_checkpoint(args.ckpt, "cpu")
    model.eval()
    model.to("cpu")

    overrides = {s: (args.topk, args.temp) for s in args.streams}
    work_root = VB.REPO / "benchmark" / "v10_work"
    results = {
        "ckpt": args.ckpt,
        "n": args.n,
        "topk": args.topk,
        "temp": args.temp,
        "streams": args.streams,
        "baseline_timing_min": args.baseline_timing_min,
        "deterministic_config": {
            "offset_mode": args.det_offset_mode,
            "offset_lambda": args.det_offset_lambda,
            "dur_metrical_lambda": args.det_dur_metrical_lambda,
            "det_loss_improve": args.det_loss_improve,
            "det_min_base_trip": args.det_min_base_trip,
        },
        "sample_guards": {
            "screen_frac": args.screen_frac,
            "max_keep_delta_frac": args.max_keep_delta_frac,
            "max_model_delta": args.max_model_delta,
            "min_model_delta": args.min_model_delta,
            "target_model_delta": args.target_model_delta,
            "trip_floor_min_base": args.trip_floor_min_base,
            "min_trip_ratio": args.min_trip_ratio,
        },
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
        base_record = score_record(model, p, y_base)
        print(
            f"  baseline: MeanER={rec_b['meaner_overall']:.2f} TPF={tpf_b[0]} "
            f"timing={base_record['timing_per_group']:.6f} "
            f"model={base_record['model_loss']:.6f} keep={base_record['keep_count']} "
            f"trip={base_record['trip_phase_count']}",
            flush=True,
        )

        route = "baseline_easy"
        selected_y = y_base
        selected_record = {**base_record}
        selected_cfg = None
        det_record = None
        sample_records = []
        screened = []

        if base_record["timing_per_group"] >= args.baseline_timing_min:
            y_det, det_cfg = decode_deterministic(model, p, args, priors)
            det_record = score_record(model, p, y_det, base_record)
            det_record["config"] = det_cfg
            det_record["deterministic_accept"] = (
                det_record["model_delta"] <= -args.det_loss_improve
                and det_record["keep_delta_frac"] <= args.max_keep_delta_frac
                and base_record["trip_phase_count"] >= args.det_min_base_trip
            )
            print(
                f"  det: model_delta={det_record['model_delta']:.6f} "
                f"timing={det_record['timing_per_group']:.6f} "
                f"keep_delta={det_record['keep_delta_frac']:.4f} "
                f"accept={det_record['deterministic_accept']}",
                flush=True,
            )
            if det_record["deterministic_accept"]:
                route = "deterministic_phase_duration"
                selected_y = y_det
                selected_record = det_record
                selected_cfg = det_cfg
            else:
                donors_y = []
                for i in range(args.n):
                    y_c = VB.decode(model, p["_x"], overrides, seed=1000 + i)
                    donors_y.append(y_c)
                    record = score_record(model, p, y_c, base_record)
                    record["i"] = i
                    record["seed"] = 1000 + i
                    sample_records.append(record)
                    print(
                        f"  cand{i:02d}: model_delta={record['model_delta']:.6f} "
                        f"timing={record['timing_per_group']:.6f} "
                        f"keep={record['keep_count']} trip={record['trip_phase_count']}",
                        flush=True,
                    )
                pick, screened = sample_choose(sample_records, base_record, args)
                if pick is not None:
                    route = "guarded_bestofn"
                    selected_y = donors_y[pick["i"]]
                    selected_record = pick
                    selected_cfg = {"candidate": pick["i"], "seed": pick["seed"]}
                else:
                    route = "baseline_fallback"

        selected_metrics = eval_selected(
            selected_y, p, gt_total, work_root / tag / f"selected_{route}"
        )
        base_metrics = {
            **rec_b,
            "tpf": tpf_b[0],
            "tpf_missed": tpf_b[1],
            "tpf_onset": tpf_b[2],
        }
        print(
            f"  --> route {route}: MeanER={selected_metrics['meaner_overall']:.2f} "
            f"TPF={selected_metrics['tpf']} selected={selected_cfg}",
            flush=True,
        )

        results["pieces"][f"{p['composer']}/{p['piece']}"] = {
            "gt_tuplets": gt_total,
            "baseline": {**base_metrics, **base_record},
            "deterministic": det_record,
            "sample_candidates": sample_records,
            "timing_screen": [r["i"] for r in screened],
            "route": route,
            "selected": {
                **selected_record,
                "config": selected_cfg,
                "metrics": selected_metrics,
            },
        }
        json.dump(results, open(out_path, "w"), indent=2)

    results["elapsed_s"] = round(time.time() - t0, 1)
    json.dump(results, open(out_path, "w"), indent=2)
    print(f"\nWrote {out_path} ({results['elapsed_s']}s)", flush=True)


if __name__ == "__main__":
    main()
