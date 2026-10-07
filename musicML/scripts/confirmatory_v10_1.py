"""CONFIRMATORY run of frozen V10.1 on the ten untouched pieces (prereg Addendum 2).

Two phases, so selections are LOCKED before any candidate ever meets MUSTER:

  --phase select : run V10.1 routing on all ten pieces with NO MUSTER calls. Writes
                   benchmark/confirmatory_v10_1_selections.json containing per piece the
                   route, selected candidate identity, guard records, and sha256 hashes of
                   the baseline + selected decoded streams, plus a UTC timestamp.
  --phase eval   : re-decodes baseline + selected, VERIFIES stream hashes against the
                   locked selections (abort on mismatch), then runs MUSTER/TPF on baseline
                   and selected only, applies the frozen decision rule, and writes
                   benchmark/confirmatory_v10_1_results.json.

Uses routed_selector_v10 functions verbatim; all parameters are V10.1 defaults (hashes and
values recorded in paper/bestofn_prereg.md Addendum 2). No GT contact in phase select.
"""
import argparse, hashlib, json, time
from datetime import datetime, timezone
from pathlib import Path

import torch

import routed_selector_v10 as V10
import local_repair_v4 as V4
import tuplet_metrics_v2 as TM2
import verifier_bestofn as VB

DEV_KEYS = {
    ("Scriabin", "Scriabin/Etudes_op_8/11"),
    ("Mozart", "Mozart/Piano_Sonatas/12-1"),
    ("Liszt", "Liszt/Annees_de_pelerinage_2/1_Gondoliera"),
    ("Ravel", "Ravel/Gaspard_de_la_Nuit/1_Ondine"),
}
SEL_PATH = VB.REPO / "benchmark/confirmatory_v10_1_selections.json"
RES_PATH = VB.REPO / "benchmark/confirmatory_v10_1_results.json"
EXPECTED_CONFIRMATORY_KEYS = {
    "Bach/Bach/Fugue/bwv_846",
    "Beethoven/Beethoven/Piano_Sonatas/10-1",
    "Brahms/Brahms/Six_Pieces_op_118/2",
    "Chopin/Chopin/Ballades/1",
    "Debussy/Debussy/Images_Book_1/1_Reflets_dans_lEau",
    "Haydn/Haydn/Keyboard_Sonatas/31-1",
    "Prokofiev/Prokofiev/Toccata",
    "Rachmaninoff/Rachmaninoff/Preludes_op_23/4",
    "Schubert/Schubert/Impromptu_op.90_D.899/1",
    "Schumann/Schumann/Arabeske",
}


def v10_args():
    """Reconstruct V10.1 defaults exactly (mirrors routed_selector_v10.main's parser)."""
    ns = argparse.Namespace(
        ckpt="checkpoints/MIDI2ScoreTF.ckpt", n=32, topk=15, temp=0.7,
        streams=["offset", "duration"], prior_path=str(VB.REPO / "data/duration_priors.pt"),
        baseline_timing_min=0.001, det_offset_mode="eighth", det_offset_lambda=1.0,
        det_dur_metrical_lambda=0.25, det_loss_improve=0.01, det_min_base_trip=50,
        offset_topk=15, overlap=64, chunk=512, seed=1234, screen_frac=0.5,
        max_keep_delta_frac=0.05, max_model_delta=0.03, min_model_delta=0.02,
        target_model_delta=0.025, trip_floor_min_base=50, min_trip_ratio=0.5,
    )
    return ns


def stream_hash(y):
    h = hashlib.sha256()
    s = V4.streams_np(y)
    h.update(s["off"].astype("int64").tobytes())
    h.update(s["dur"].astype("int64").tobytes())
    h.update(s["pad"].astype("uint8").tobytes())
    return h.hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def confirmatory_pieces():
    paths, seen = [], set()
    for p in VB.collect_paths("test"):
        key = (p["composer"], p["piece"])
        if key in seen or key in DEV_KEYS:
            continue
        seen.add(key)
        try:
            p["_x"] = VB.MultistreamTokenizer.tokenize_midi(p["midi"])
            p["_t"] = VB.MultistreamTokenizer.parse_midi(p["midi"])["onset"].numpy()
            p["n_notes"] = int(p["_x"]["pitch"].shape[0])
            paths.append(p)
        except Exception as ex:
            print(f"  SKIP {key}: tokenize failed: {ex}")
    paths.sort(key=lambda p: p["n_notes"])
    actual = {f"{p['composer']}/{p['piece']}" for p in paths}
    if actual != EXPECTED_CONFIRMATORY_KEYS:
        missing = sorted(EXPECTED_CONFIRMATORY_KEYS - actual)
        extra = sorted(actual - EXPECTED_CONFIRMATORY_KEYS)
        raise RuntimeError(
            f"Confirmatory population mismatch: missing={missing}, extra={extra}"
        )
    return paths


def route_piece(model, p, args, priors, overrides):
    """V10.1 routing verbatim, minus every MUSTER call."""
    y_base = VB.decode(model, p["_x"], None, seed=1)
    base_record = V10.score_record(model, p, y_base)
    route, selected_y, selected_record, selected_cfg = "baseline_easy", y_base, dict(base_record), None
    det_record, sample_records, screened = None, [], []
    if base_record["timing_per_group"] >= args.baseline_timing_min:
        y_det, det_cfg = V10.decode_deterministic(model, p, args, priors)
        det_record = V10.score_record(model, p, y_det, base_record)
        det_record["config"] = det_cfg
        det_record["deterministic_accept"] = (
            det_record["model_delta"] <= -args.det_loss_improve
            and det_record["keep_delta_frac"] <= args.max_keep_delta_frac
            and base_record["trip_phase_count"] >= args.det_min_base_trip
        )
        if det_record["deterministic_accept"]:
            route, selected_y, selected_record, selected_cfg = (
                "deterministic_phase_duration", y_det, det_record, det_cfg)
        else:
            donors_y = []
            for i in range(args.n):
                y_c = VB.decode(model, p["_x"], overrides, seed=1000 + i)
                donors_y.append(y_c)
                rec = V10.score_record(model, p, y_c, base_record)
                rec["i"] = i; rec["seed"] = 1000 + i
                sample_records.append(rec)
            pick, screened = V10.sample_choose(sample_records, base_record, args)
            if pick is not None:
                route = "guarded_bestofn"
                selected_y = donors_y[pick["i"]]
                selected_record = pick
                selected_cfg = {"candidate": pick["i"], "seed": pick["seed"]}
            else:
                route = "baseline_fallback"
    return {
        "route": route, "selected_cfg": selected_cfg,
        "base_record": base_record, "det_record": det_record,
        "sample_records": sample_records, "screen": [r["i"] for r in screened],
        "base_hash": stream_hash(y_base), "selected_hash": stream_hash(selected_y),
    }, y_base, selected_y


def redecode_selected(model, p, args, priors, overrides, sel):
    if sel["route"] in ("baseline_easy", "baseline_fallback"):
        return VB.decode(model, p["_x"], None, seed=1)
    if sel["route"] == "deterministic_phase_duration":
        y, _ = V10.decode_deterministic(model, p, args, priors)
        return y
    return VB.decode(model, p["_x"], overrides, seed=sel["selected_cfg"]["seed"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["select", "eval"], required=True)
    a = ap.parse_args()
    args = v10_args()
    priors = torch.load(VB.REPO / "data/duration_priors.pt", map_location="cpu", weights_only=False)
    overrides = {s: (args.topk, args.temp) for s in args.streams}
    paths = confirmatory_pieces()
    print(f"{len(paths)} confirmatory pieces: " +
          ", ".join(f"{p['composer']}" for p in paths), flush=True)
    model = VB.load_any_checkpoint(args.ckpt, "cpu"); model.eval(); model.to("cpu")
    t0 = time.time()

    if a.phase == "select":
        out = {"frozen": "V10.1 (prereg Addendum 2)", "phase": "select",
               "utc": datetime.now(timezone.utc).isoformat(), "pieces": {}}
        for p in paths:
            print(f"\n===== {p['composer']}/{p['piece']} ({p['n_notes']} notes) =====", flush=True)
            sel, _, _ = route_piece(model, p, args, priors, overrides)
            print(f"  route={sel['route']} cfg={sel['selected_cfg']} "
                  f"timing={sel['base_record']['timing_per_group']:.6f}", flush=True)
            out["pieces"][f"{p['composer']}/{p['piece']}"] = sel
            json.dump(out, open(SEL_PATH, "w"), indent=2)
        out["elapsed_s"] = round(time.time() - t0, 1)
        json.dump(out, open(SEL_PATH, "w"), indent=2)
        digest = hashlib.sha256(open(SEL_PATH, "rb").read()).hexdigest()
        print(f"\nSELECTIONS LOCKED {out['utc']} sha256={digest}", flush=True)
        return

    # ---- phase eval ----
    locked = json.load(open(SEL_PATH))
    if locked.get("phase") != "select" or set(locked.get("pieces", {})) != EXPECTED_CONFIRMATORY_KEYS:
        raise RuntimeError("Locked selection artifact has the wrong phase or piece population")
    results = {"frozen": "V10.1", "phase": "eval", "selections_utc": locked["utc"],
               "selections_sha256": file_hash(SEL_PATH),
               "metric": {
                   "tuplet_tag": "exact MusicXML <time-modification> joined by Fmt3x GtID",
                   "muster_tuplet_decompose_sha256": file_hash(VB.REPO / "scripts/muster_tuplet_decompose.py"),
                   "tuplet_metrics_v2_sha256": file_hash(VB.REPO / "scripts/tuplet_metrics_v2.py"),
               },
               "utc": datetime.now(timezone.utc).isoformat(), "pieces": {}}
    # Keep the original TPQN=24 evaluation directory intact as an audit artifact.
    work_root = VB.REPO / "benchmark" / "confirmatory_work_metric_v2"
    for p in paths:
        key = f"{p['composer']}/{p['piece']}"
        sel = locked["pieces"][key]
        tag = p["piece"].replace("/", "_")
        print(f"\n===== {key} route={sel['route']} =====", flush=True)
        y_base = VB.decode(model, p["_x"], None, seed=1)
        assert stream_hash(y_base) == sel["base_hash"], f"BASE HASH MISMATCH {key}"
        y_sel = redecode_selected(model, p, args, priors, overrides, sel)
        assert stream_hash(y_sel) == sel["selected_hash"], f"SELECTED HASH MISMATCH {key}"
        rec_b = TM2.muster_piece(y_base, p, work_root / tag / "baseline")
        gt_total = rec_b["n_gt_tuplet"]
        tpf_b = TM2.tpf(rec_b)
        rec_s = TM2.muster_piece(y_sel, p, work_root / tag / "selected")
        if rec_s["n_gt_tuplet"] != gt_total:
            raise RuntimeError(
                f"Candidate-specific GT tuplet population changed for {key}: "
                f"baseline={gt_total}, selected={rec_s['n_gt_tuplet']}"
            )
        tpf_s = TM2.tpf(rec_s)
        sb, ss = V4.streams_np(y_base), V4.streams_np(y_sel)
        import numpy as np
        emit_b = int((((sb["off"] % 24) % 3 != 0) & sb["pad"]).sum())
        emit_s = int((((ss["off"] % 24) % 3 != 0) & ss["pad"]).sum())
        rec = {
            "gt_tuplets": gt_total, "route": sel["route"],
            "baseline": {**rec_b, "tpf": tpf_b[0], "tpf_missed": tpf_b[1], "tpf_onset": tpf_b[2],
                         "kept_trip_emissions": emit_b},
            "selected": {**rec_s, "tpf": tpf_s[0], "tpf_missed": tpf_s[1], "tpf_onset": tpf_s[2],
                         "kept_trip_emissions": emit_s},
        }
        d_meaner = (rec_s["meaner_overall"] or 0) - (rec_b["meaner_overall"] or 0)
        print(f"  gtT={gt_total} MeanER {rec_b['meaner_overall']:.3f}->{rec_s['meaner_overall']:.3f} "
              f"(d={d_meaner:+.3f}) TPF {tpf_b[0]}->{tpf_s[0]} emissions {emit_b}->{emit_s}", flush=True)
        results["pieces"][key] = rec
        json.dump(results, open(RES_PATH, "w"), indent=2)

    # frozen decision rule
    bearing = {k: v for k, v in results["pieces"].items() if v["gt_tuplets"] >= 20}
    controls = {k: v for k, v in results["pieces"].items() if v["gt_tuplets"] < 20}
    rels, improved, p3_ok = [], 0, True
    for k, v in bearing.items():
        b, s = v["baseline"]["tpf"], v["selected"]["tpf"]
        rel = (s - b) / max(b, 1)
        rels.append(rel)
        if rel <= -0.05: improved += 1
        if (s - b) > max(0.05 * b, 3): p3_ok = False
    g1_ok = all(
        v["selected"]["meaner_overall"] is not None
        and v["baseline"]["meaner_overall"] is not None
        and v["selected"]["meaner_overall"] - v["baseline"]["meaner_overall"] <= 0.50
        for v in results["pieces"].values()
    )
    g2_ok = all((v["selected"]["kept_trip_emissions"] - v["baseline"]["kept_trip_emissions"]) <= 5
                and (v["selected"]["tpf"] - v["baseline"]["tpf"]) <= 1 for v in controls.values())
    rule = {
        "P1_pieces_improved_5pct": improved, "P1_ok": improved >= 3,
        "P2_mean_rel_tpf": (sum(rels) / len(rels)) if rels else None,
        "P2_ok": bool(rels) and (sum(rels) / len(rels)) <= -0.05,
        "P3_ok": p3_ok, "G1_ok": g1_ok, "G2_ok": g2_ok,
        "n_bearing": len(bearing), "n_controls": len(controls),
    }
    rule["CONFIRMED"] = all([rule["P1_ok"], rule["P2_ok"], rule["P3_ok"], g1_ok, g2_ok])
    results["decision_rule"] = rule
    results["elapsed_s"] = round(time.time() - t0, 1)
    json.dump(results, open(RES_PATH, "w"), indent=2)
    print(f"\nDECISION RULE: {json.dumps(rule, indent=1)}", flush=True)
    print(f"Wrote {RES_PATH}", flush=True)


if __name__ == "__main__":
    main()
