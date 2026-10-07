"""Read-only integrity and arithmetic audit for the V10.1 confirmation."""

import hashlib
import json
import math
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent
SELECTION_SHA256 = "f1c4a6f184bc05724c4584864440149296f339fa03a1fb931178223c1fd7b9cc"
INVALID_RESULT_SHA256 = "85de60af449c9270b84e0f45c3494044ccce729df7bbf705990ca05efb786f9e"
FROZEN_HASHES = {
    "scripts/routed_selector_v10.py": "cb9556b057abd85906eafcf66be3a6ef3dd84e911c2a9a3d32c67241b61d230d",
    "scripts/decode_lever_sweep_v9.py": "18fd0c62482dbfd4e806277e23155f75e98e3df480aca4e0c6b4dd0b44fc40ac",
    "scripts/guarded_selector_v8.py": "65e183aace68c385e8a1f5cba58a892942afae9de94c77150b07d5636ed4e0b3",
    "scripts/likelihood_timing_selector_v6.py": "2c7fbf435bbe5eca5035298a71cc0bf44956e280adbea050a6257d295d077fb2",
    "scripts/local_repair_v4.py": "370cf01a2e529b0e314c65a4550033e47231cfdd551ed5a771308e966d2776f4",
    "scripts/verifier_bestofn.py": "eace77dca21dd49b14f92e72be3af89f88c3209eeee0b8bf3bcc0600ca8197dd",
    "data/duration_priors.pt": "7e0b8c2543362e75f70f42b46d93832db983c4f41692320a3771b0cee47710e4",
    "MIDI2ScoreTransformer/checkpoints/MIDI2ScoreTF.ckpt": "7b8ec6e3da365b97443fb67a8f0b37d63997e93c152d665d43cb2011245db638",
}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def assert_selection_is_gt_blind(value, path="root"):
    forbidden = ("meaner", "muster", "tpf", "gt_tuplet", "onset_err", "offset_err")
    if isinstance(value, dict):
        for key, child in value.items():
            lowered = key.lower()
            if any(term in lowered for term in forbidden):
                raise AssertionError(f"GT/evaluation field in selections at {path}.{key}")
            assert_selection_is_gt_blind(child, f"{path}.{key}")
    elif isinstance(value, list):
        for i, child in enumerate(value):
            assert_selection_is_gt_blind(child, f"{path}[{i}]")


def recompute_route(piece):
    base = piece["base_record"]
    if base["timing_per_group"] < 0.001:
        return "baseline_easy", None

    det = piece["det_record"]
    det_accept = (
        det["model_delta"] <= -0.01
        and det["keep_delta_frac"] <= 0.05
        and base["trip_phase_count"] >= 50
    )
    assert det_accept == det["deterministic_accept"]
    if det_accept:
        return "deterministic_phase_duration", det["config"]

    gated = []
    base_trip = max(base["trip_phase_count"], 1)
    for candidate in piece["sample_records"]:
        eligible = (
            candidate["timing_per_group"] < base["timing_per_group"]
            and candidate["keep_delta_frac"] <= 0.05
            and 0.02 <= candidate["model_delta"] <= 0.03
            and candidate["trip_phase_count"]
            + candidate["kept_tuplet_duration_count"] > 0
            and (
                base["trip_phase_count"] < 50
                or candidate["trip_phase_count"] / base_trip >= 0.5
            )
        )
        assert eligible == candidate["eligible"]
        if eligible:
            gated.append(candidate)
    if not gated:
        return "baseline_fallback", None

    screen_size = max(1, math.ceil(len(gated) * 0.5))
    screened = sorted(gated, key=lambda r: (r["timing_per_group"], r["i"]))[:screen_size]
    assert [row["i"] for row in screened] == piece["screen"]
    if base["trip_phase_count"] >= 50:
        selected = min(screened, key=lambda r: (r["timing_per_group"], r["i"]))
    else:
        selected = min(screened, key=lambda r: (abs(r["model_delta"] - 0.025), r["i"]))
    return "guarded_bestofn", {"candidate": selected["i"], "seed": selected["seed"]}


def recompute_rule(results):
    pieces = list(results["pieces"].values())
    bearing = [p for p in pieces if p["gt_tuplets"] >= 20]
    controls = [p for p in pieces if p["gt_tuplets"] < 20]
    rels = [
        (p["selected"]["tpf"] - p["baseline"]["tpf"]) / max(p["baseline"]["tpf"], 1)
        for p in bearing
    ]
    p1 = sum(rel <= -0.05 for rel in rels)
    p3 = all(
        p["selected"]["tpf"] - p["baseline"]["tpf"]
        <= max(0.05 * p["baseline"]["tpf"], 3)
        for p in bearing
    )
    g1 = all(
        p["selected"]["meaner_overall"] - p["baseline"]["meaner_overall"] <= 0.50
        for p in pieces
    )
    g2 = all(
        p["selected"]["kept_trip_emissions"] - p["baseline"]["kept_trip_emissions"] <= 5
        and p["selected"]["tpf"] - p["baseline"]["tpf"] <= 1
        for p in controls
    )
    return {
        "P1_pieces_improved_5pct": p1,
        "P1_ok": p1 >= 3,
        "P2_mean_rel_tpf": sum(rels) / len(rels),
        "P2_ok": sum(rels) / len(rels) <= -0.05,
        "P3_ok": p3,
        "G1_ok": g1,
        "G2_ok": g2,
        "n_bearing": len(bearing),
        "n_controls": len(controls),
        "CONFIRMED": p1 >= 3 and sum(rels) / len(rels) <= -0.05 and p3 and g1 and g2,
    }


def main():
    selections_path = REPO / "artifacts/confirmatory_v10_1_selections.json"
    results_path = REPO / "artifacts/confirmatory_v10_1_results.json"
    selections = json.load(open(selections_path))
    results = json.load(open(results_path))
    assert sha256(selections_path) == SELECTION_SHA256
    assert_selection_is_gt_blind(selections)
    for key, piece in selections["pieces"].items():
        route, config = recompute_route(piece)
        assert route == piece["route"], f"Route mismatch for {key}"
        assert config == piece["selected_cfg"], f"Selected config mismatch for {key}"
    assert results["selections_sha256"] == SELECTION_SHA256
    assert len(selections["pieces"]) == len(results["pieces"]) == 10
    assert set(selections["pieces"]) == set(results["pieces"])

    metric = results["metric"]
    assert metric["muster_tuplet_decompose_sha256"] == sha256(
        REPO / "scripts/muster_tuplet_decompose.py"
    )
    assert metric["tuplet_metrics_v2_sha256"] == sha256(
        REPO / "scripts/tuplet_metrics_v2.py"
    )
    recomputed = recompute_rule(results)
    for key, value in recomputed.items():
        stored = results["decision_rule"][key]
        if isinstance(value, float):
            assert math.isclose(value, stored, rel_tol=0, abs_tol=1e-12)
        else:
            assert value == stored

    dev_path = REPO / "artifacts/routed_selector_v10_n32_final_metric_v2.json"
    dev = json.load(open(dev_path))
    assert dev["metric"] == metric
    dev_pieces = list(dev["pieces"].values())
    dev_aggregate = {
        "baseline_tpf": sum(p["baseline"]["tpf"] for p in dev_pieces),
        "selected_tpf": sum(p["selected"]["tpf"] for p in dev_pieces),
        "macro_meaner_delta": sum(
            p["selected"]["meaner_overall"] - p["baseline"]["meaner_overall"]
            for p in dev_pieces
        ) / len(dev_pieces),
    }
    assert dev_aggregate == dev["aggregate"]

    pieces = list(results["pieces"].values())
    summary = {
        "frozen_hashes_recorded": FROZEN_HASHES,
        "selection_sha256": SELECTION_SHA256,
        "selection_gt_blind": True,
        "routes_recomputed": True,
        "n_pieces": len(pieces),
        "pooled_tpf_baseline": sum(p["baseline"]["tpf"] for p in pieces),
        "pooled_tpf_selected": sum(p["selected"]["tpf"] for p in pieces),
        "macro_meaner_delta": sum(
            p["selected"]["meaner_overall"] - p["baseline"]["meaner_overall"]
            for p in pieces
        ) / len(pieces),
        "decision_rule": recomputed,
        "development_aggregate": dev_aggregate,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
