"""Re-score the four frozen V10 development picks with exact tuplet labels.

The original JSON's routing choices and MeanER values remain the identity reference.  This
script decodes only those already-selected outputs, verifies their MeanER values reproduce,
and replaces the TPQN-dependent TPF calculation with ``tuplet_metrics_v2``.  It does not
search candidates or change V10.
"""

import json
import time
from datetime import datetime, timezone

import torch

import confirmatory_v10_1 as C
import routed_selector_v10 as V10
import tuplet_metrics_v2 as TM2
import verifier_bestofn as VB


SOURCE_PATH = VB.REPO / "benchmark/routed_selector_v10_n32_final.json"
OUT_PATH = VB.REPO / "benchmark/routed_selector_v10_n32_final_metric_v2.json"


def development_pieces():
    paths = []
    for piece in VB.collect_paths("test"):
        if (piece["composer"], piece["piece"]) not in C.DEV_KEYS:
            continue
        try:
            piece["_x"] = VB.MultistreamTokenizer.tokenize_midi(piece["midi"])
            piece["n_notes"] = int(piece["_x"]["pitch"].shape[0])
        except Exception:
            piece["_x"] = None
            piece["n_notes"] = 1 << 30
        paths.append(piece)
    paths.sort(key=lambda p: p["n_notes"])
    seen, kept = set(), []
    for piece in paths:
        key = (piece["composer"], piece["piece"])
        if key not in seen and piece["_x"] is not None:
            seen.add(key)
            kept.append(piece)
    actual = {(p["composer"], p["piece"]) for p in kept}
    if actual != C.DEV_KEYS:
        raise RuntimeError(
            f"Development population mismatch: missing={sorted(C.DEV_KEYS - actual)}, "
            f"extra={sorted(actual - C.DEV_KEYS)}"
        )
    return kept


def selected_decode(model, piece, route, config, args, priors, overrides, baseline):
    if route in ("baseline_easy", "baseline_fallback"):
        return baseline
    if route == "deterministic_phase_duration":
        decoded, actual_config = V10.decode_deterministic(model, piece, args, priors)
        if actual_config != config:
            raise RuntimeError(f"Deterministic config mismatch: {actual_config} != {config}")
        return decoded
    if route == "guarded_bestofn":
        return VB.decode(model, piece["_x"], overrides, seed=config["seed"])
    raise ValueError(f"Unknown V10 route: {route}")


def assert_meaner(label, actual, expected):
    if actual is None or expected is None or abs(actual - expected) > 1e-5:
        raise RuntimeError(f"{label} MeanER identity mismatch: {actual} != {expected}")


def main():
    source = json.load(open(SOURCE_PATH))
    args = C.v10_args()
    priors = torch.load(
        VB.REPO / "data/duration_priors.pt", map_location="cpu", weights_only=False
    )
    overrides = {stream: (args.topk, args.temp) for stream in args.streams}
    model = VB.load_any_checkpoint(args.ckpt, "cpu")
    model.eval()
    model.to("cpu")

    out = {
        "status": "development metrics corrected; selector choices unchanged",
        "utc": datetime.now(timezone.utc).isoformat(),
        "source": str(SOURCE_PATH.relative_to(VB.REPO)),
        "source_sha256": C.file_hash(SOURCE_PATH),
        "metric": {
            "tuplet_tag": "exact MusicXML <time-modification> joined by Fmt3x GtID",
            "muster_tuplet_decompose_sha256": C.file_hash(
                VB.REPO / "scripts/muster_tuplet_decompose.py"
            ),
            "tuplet_metrics_v2_sha256": C.file_hash(VB.REPO / "scripts/tuplet_metrics_v2.py"),
        },
        "pieces": {},
    }
    work_root = VB.REPO / "benchmark/v10_dev_metric_v2_work"
    t0 = time.time()

    for piece in development_pieces():
        key = f"{piece['composer']}/{piece['piece']}"
        frozen = source["pieces"][key]
        route = frozen["route"]
        config = frozen["selected"]["config"]
        print(f"\n===== {key} route={route} config={config} =====", flush=True)

        baseline = VB.decode(model, piece["_x"], None, seed=1)
        selected = selected_decode(
            model, piece, route, config, args, priors, overrides, baseline
        )
        tag = piece["piece"].replace("/", "_")
        base_metrics = TM2.muster_piece(baseline, piece, work_root / tag / "baseline")
        selected_metrics = TM2.muster_piece(selected, piece, work_root / tag / "selected")
        if base_metrics["n_gt_tuplet"] != selected_metrics["n_gt_tuplet"]:
            raise RuntimeError(f"GT tuplet population changed for {key}")

        assert_meaner(
            f"{key} baseline",
            base_metrics["meaner_overall"],
            frozen["baseline"]["meaner_overall"],
        )
        assert_meaner(
            f"{key} selected",
            selected_metrics["meaner_overall"],
            frozen["selected"]["metrics"]["meaner_overall"],
        )
        base_tpf = TM2.tpf(base_metrics)
        selected_tpf = TM2.tpf(selected_metrics)
        out["pieces"][key] = {
            "route": route,
            "config": config,
            "baseline_hash": C.stream_hash(baseline),
            "selected_hash": C.stream_hash(selected),
            "gt_tuplets": base_metrics["n_gt_tuplet"],
            "baseline": {
                **base_metrics,
                "tpf": base_tpf[0],
                "tpf_missed": base_tpf[1],
                "tpf_onset": base_tpf[2],
            },
            "selected": {
                **selected_metrics,
                "tpf": selected_tpf[0],
                "tpf_missed": selected_tpf[1],
                "tpf_onset": selected_tpf[2],
            },
        }
        print(
            f"  gtT={base_metrics['n_gt_tuplet']} "
            f"MeanER {base_metrics['meaner_overall']:.3f}->{selected_metrics['meaner_overall']:.3f} "
            f"TPF {base_tpf[0]}->{selected_tpf[0]}",
            flush=True,
        )
        json.dump(out, open(OUT_PATH, "w"), indent=2)

    rows = list(out["pieces"].values())
    out["aggregate"] = {
        "baseline_tpf": sum(r["baseline"]["tpf"] for r in rows),
        "selected_tpf": sum(r["selected"]["tpf"] for r in rows),
        "macro_meaner_delta": sum(
            r["selected"]["meaner_overall"] - r["baseline"]["meaner_overall"]
            for r in rows
        ) / len(rows),
    }
    out["elapsed_s"] = round(time.time() - t0, 1)
    json.dump(out, open(OUT_PATH, "w"), indent=2)
    print(f"\n{json.dumps(out['aggregate'], indent=2)}", flush=True)
    print(f"Wrote {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
