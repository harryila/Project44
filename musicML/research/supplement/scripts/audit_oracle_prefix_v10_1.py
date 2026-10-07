"""Read-only integrity and arithmetic audit for the frozen V10.1 oracle analysis."""

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.dont_write_bytecode = True
import audit_confirmatory_v10_1 as CONFIRM_AUDIT


REPO = Path(__file__).resolve().parent.parent
ORACLE_PATH = REPO / "artifacts" / "oracle_prefix_v10_1.json"
SELECTION_PATH = REPO / "artifacts" / "confirmatory_v10_1_selections.json"
CONFIRM_PATH = REPO / "artifacts" / "confirmatory_v10_1_results.json"
PREFIXES = (1, 2, 4, 8, 16, 32)
SENSITIVITY_GUARDS = (0.0, 0.25, 0.50)


def close(actual, expected, tolerance=1e-12):
    if not math.isclose(actual, expected, rel_tol=0, abs_tol=tolerance):
        raise AssertionError(f"{actual} != {expected}")


def best(rows):
    return min(
        rows,
        key=lambda row: (
            row["metrics"]["tpf"],
            row["metrics"]["meaner_overall"],
            -1 if row.get("candidate") is None else row["candidate"],
        ),
    )


def compact(row, baseline):
    return {
        "source": row["source"],
        "candidate": row.get("candidate"),
        "seed": row.get("seed"),
        "tpf": row["metrics"]["tpf"],
        "meaner": row["metrics"]["meaner_overall"],
        "delta_tpf": row["metrics"]["tpf"] - baseline["tpf"],
        "delta_meaner": row["metrics"]["meaner_overall"] - baseline["meaner_overall"],
    }


def recompute_piece_curve(piece):
    baseline = piece["baseline"]
    baseline_row = {"source": "baseline", "metrics": baseline}
    candidates = sorted(piece["candidates"], key=lambda row: row["candidate"])
    curve = {}
    for n in PREFIXES:
        prefix = candidates[:n]
        guarded = [
            row for row in prefix
            if row["metrics"]["meaner_overall"] <= baseline["meaner_overall"] + 0.50
        ]
        curve[str(n)] = {
            "sample_oracle": compact(best(prefix), baseline),
            "baseline_inclusive_oracle": compact(best([baseline_row, *prefix]), baseline),
            "meaner_guarded_oracle": compact(best([baseline_row, *guarded]), baseline),
            "n_tpf_better_than_baseline": sum(
                row["metrics"]["tpf"] < baseline["tpf"] for row in prefix
            ),
            "n_guarded_tpf_better_than_baseline": sum(
                row["metrics"]["tpf"] < baseline["tpf"]
                and row["metrics"]["meaner_overall"] <= baseline["meaner_overall"] + 0.50
                for row in prefix
            ),
            "n_strict_both_better_than_baseline": sum(
                row["metrics"]["tpf"] < baseline["tpf"]
                and row["metrics"]["meaner_overall"] < baseline["meaner_overall"]
                for row in prefix
            ),
        }
    return curve


def aggregate(pieces, n, field):
    picks = {
        key: piece["prefix_curve"][str(n)][field]
        for key, piece in pieces.items()
    }
    baseline_tpf = sum(piece["baseline"]["tpf"] for piece in pieces.values())
    picked_tpf = sum(pick["tpf"] for pick in picks.values())
    rels = [
        (picks[key]["tpf"] - piece["baseline"]["tpf"])
        / max(piece["baseline"]["tpf"], 1)
        for key, piece in pieces.items()
    ]
    return {
        "baseline_tpf": baseline_tpf,
        "picked_tpf": picked_tpf,
        "delta_tpf": picked_tpf - baseline_tpf,
        "pooled_relative_tpf_delta": (picked_tpf - baseline_tpf) / max(baseline_tpf, 1),
        "mean_per_piece_relative_tpf_delta": sum(rels) / len(rels),
        "macro_meaner_delta": sum(pick["delta_meaner"] for pick in picks.values()) / len(picks),
        "n_pieces_tpf_improved": sum(pick["delta_tpf"] < 0 for pick in picks.values()),
    }


def aggregate_at_guard(pieces, guard):
    picks = {}
    for key, piece in pieces.items():
        baseline = piece["baseline"]
        baseline_row = {"source": "baseline", "metrics": baseline}
        eligible = [
            row for row in piece["candidates"]
            if row["metrics"]["meaner_overall"]
            <= baseline["meaner_overall"] + guard + 1e-12
        ]
        picks[key] = compact(best([baseline_row, *eligible]), baseline)

    baseline_tpf = sum(piece["baseline"]["tpf"] for piece in pieces.values())
    picked_tpf = sum(pick["tpf"] for pick in picks.values())
    return {
        "guard": guard,
        "baseline_tpf": baseline_tpf,
        "picked_tpf": picked_tpf,
        "delta_tpf": picked_tpf - baseline_tpf,
        "pooled_relative_tpf_delta": (picked_tpf - baseline_tpf) / baseline_tpf,
        "macro_meaner_delta": sum(pick["delta_meaner"] for pick in picks.values())
        / len(picks),
        "n_pieces_tpf_improved": sum(pick["delta_tpf"] < 0 for pick in picks.values()),
        "picks": picks,
    }


def subset_min_pmf(piece, n, guard=0.50):
    """Exact oracle-TPF distribution for uniform subsets of one frozen pool."""
    baseline = piece["baseline"]
    base_tpf = baseline["tpf"]
    effective_tpf = []
    for row in piece["candidates"]:
        metrics = row["metrics"]
        if metrics["meaner_overall"] <= baseline["meaner_overall"] + guard + 1e-12:
            effective_tpf.append(min(base_tpf, metrics["tpf"]))
        else:
            effective_tpf.append(base_tpf)

    denominator = math.comb(len(effective_tpf), n)
    pmf = {}
    for value in sorted(set(effective_tpf)):
        n_ge = sum(candidate >= value for candidate in effective_tpf)
        n_gt = sum(candidate > value for candidate in effective_tpf)
        ways_ge = math.comb(n_ge, n) if n_ge >= n else 0
        ways_gt = math.comb(n_gt, n) if n_gt >= n else 0
        if ways_ge > ways_gt:
            pmf[value] = (ways_ge - ways_gt) / denominator
    close(sum(pmf.values()), 1.0)
    return pmf


def convolve_pmfs(pmfs):
    combined = {0: 1.0}
    for pmf in pmfs:
        updated = defaultdict(float)
        for left, left_prob in combined.items():
            for right, right_prob in pmf.items():
                updated[left + right] += left_prob * right_prob
        combined = dict(updated)
    close(sum(combined.values()), 1.0)
    return combined


def quantile(pmf, probability):
    cumulative = 0.0
    for value, mass in sorted(pmf.items()):
        cumulative += mass
        if cumulative >= probability - 1e-15:
            return value
    raise AssertionError("PMF did not reach requested quantile")


def exact_subset_reference(pieces):
    baseline_tpf = sum(piece["baseline"]["tpf"] for piece in pieces.values())
    support_counts = {}
    for key, piece in pieces.items():
        baseline = piece["baseline"]
        support_counts[key] = sum(
            row["metrics"]["tpf"] < baseline["tpf"]
            and row["metrics"]["meaner_overall"]
            <= baseline["meaner_overall"] + 0.50 + 1e-12
            for row in piece["candidates"]
        )

    curve = {}
    for n in PREFIXES:
        pooled_pmf = convolve_pmfs([subset_min_pmf(piece, n) for piece in pieces.values()])
        expected_tpf = sum(value * mass for value, mass in pooled_pmf.items())
        lower_tpf = quantile(pooled_pmf, 0.05)
        upper_tpf = quantile(pooled_pmf, 0.95)
        support_probabilities = {}
        for key, count in support_counts.items():
            misses = (
                math.comb(32 - count, n) / math.comb(32, n)
                if 32 - count >= n
                else 0.0
            )
            support_probabilities[key] = 1.0 - misses
        curve[str(n)] = {
            "expected_pooled_tpf": expected_tpf,
            "expected_pooled_relative_tpf_delta": (expected_tpf - baseline_tpf)
            / baseline_tpf,
            "pooled_tpf_p05": lower_tpf,
            "pooled_tpf_p95": upper_tpf,
            "relative_tpf_delta_p05": (lower_tpf - baseline_tpf) / baseline_tpf,
            "relative_tpf_delta_p95": (upper_tpf - baseline_tpf) / baseline_tpf,
            "expected_pieces_with_support": sum(support_probabilities.values()),
            "piece_support_probability": support_probabilities,
        }
    return {
        "definition": (
            "Exact finite-pool reference for independently chosen uniform size-N subsets "
            "of each frozen 32-candidate pool; descriptive, not population inference."
        ),
        "support_counts": support_counts,
        "curve": curve,
    }


def assert_aggregate(actual, expected):
    for field, value in expected.items():
        if isinstance(value, float):
            close(actual[field], value)
        else:
            assert actual[field] == value, f"Aggregate mismatch for {field}"


def main():
    oracle = json.load(open(ORACLE_PATH))
    selections = json.load(open(SELECTION_PATH))
    confirmation = json.load(open(CONFIRM_PATH))
    provenance = oracle["provenance"]

    assert oracle["status"] == "complete" and oracle["analysis_complete"] is True
    assert provenance["selection_sha256"] == CONFIRM_AUDIT.SELECTION_SHA256
    assert provenance["selection_sha256"] == CONFIRM_AUDIT.sha256(SELECTION_PATH)
    assert provenance["ground_truth_used_for_selection"] is False
    assert provenance["post_selection_secondary_analysis"] is True
    assert provenance["selector_changed"] is False
    for relative, expected in CONFIRM_AUDIT.FROZEN_HASHES.items():
        assert provenance["frozen_hashes"][relative] == expected
    assert provenance["exact_evaluator_sha256"] == CONFIRM_AUDIT.sha256(
        REPO / "scripts" / "muster_tuplet_decompose.py"
    )
    assert provenance["metric_adapter_sha256"] == CONFIRM_AUDIT.sha256(
        REPO / "scripts" / "tuplet_metrics_v2.py"
    )
    assert set(oracle["pieces"]) == set(selections["pieces"]) == set(confirmation["pieces"])

    sampled = {}
    identity_counts = {}
    for key, piece in oracle["pieces"].items():
        locked = selections["pieces"][key]
        confirmed = confirmation["pieces"][key]
        assert piece["route"] == locked["route"] == confirmed["route"]
        assert piece["selected_cfg"] == locked["selected_cfg"]
        assert piece["gt_tuplets"] == piece["baseline"]["n_gt_tuplet"]
        assert piece["baseline"]["tpf"] == confirmed["baseline"]["tpf"]
        close(piece["baseline"]["meaner_overall"], confirmed["baseline"]["meaner_overall"])

        candidates = piece["candidates"]
        if not locked["sample_records"]:
            assert candidates == []
            assert piece["selected"]["tpf"] == confirmed["selected"]["tpf"]
            close(piece["selected"]["meaner_overall"], confirmed["selected"]["meaner_overall"])
            continue

        assert len(candidates) == len(locked["sample_records"]) == 32
        sampled[key] = piece
        selected_index = (
            locked["selected_cfg"].get("candidate") if locked["selected_cfg"] else None
        )
        selected_rows = []
        for index, (candidate, locked_record) in enumerate(zip(candidates, locked["sample_records"])):
            assert candidate["candidate"] == locked_record["i"] == index
            assert candidate["seed"] == locked_record["seed"] == 1000 + index
            assert candidate["metrics"]["n_gt_tuplet"] == piece["gt_tuplets"]
            assert candidate["selected_by_v10"] == (index == selected_index)
            identity = candidate["identity_check"]
            identity_counts[identity] = identity_counts.get(identity, 0) + 1
            if candidate["selected_by_v10"]:
                selected_rows.append(candidate)
        if selected_index is None:
            assert selected_rows == []
            assert confirmed["selected"]["tpf"] == confirmed["baseline"]["tpf"]
        else:
            assert len(selected_rows) == 1
            assert selected_rows[0]["metrics"]["tpf"] == confirmed["selected"]["tpf"]
            close(
                selected_rows[0]["metrics"]["meaner_overall"],
                confirmed["selected"]["meaner_overall"],
            )
        assert recompute_piece_curve(piece) == piece["prefix_curve"]

    assert len(sampled) == 8
    bearing = {key: piece for key, piece in sampled.items() if piece["gt_tuplets"] >= 20}
    controls = {key: piece for key, piece in sampled.items() if piece["gt_tuplets"] < 20}
    assert len(bearing) == 6 and len(controls) == 2
    for n in PREFIXES:
        for population_name, population in (
            ("bearing_sampled_pieces", bearing),
            ("control_sampled_pieces", controls),
        ):
            for field in (
                "sample_oracle",
                "baseline_inclusive_oracle",
                "meaner_guarded_oracle",
            ):
                expected = aggregate(population, n, field)
                actual = oracle["aggregate_prefix_curve"][population_name][str(n)][field]
                assert_aggregate(actual, expected)

    guarded_n32 = aggregate(bearing, 32, "meaner_guarded_oracle")
    assert guarded_n32["baseline_tpf"] == 1248
    assert guarded_n32["picked_tpf"] == 1129
    assert guarded_n32["delta_tpf"] == -119
    close(guarded_n32["pooled_relative_tpf_delta"], -0.0953525641025641)
    close(guarded_n32["macro_meaner_delta"], -0.558135)
    assert guarded_n32["n_pieces_tpf_improved"] == 6
    assert all(
        piece["prefix_curve"]["32"]["n_guarded_tpf_better_than_baseline"] > 0
        for piece in bearing.values()
    )
    selected_tpf = sum(confirmation["pieces"][key]["selected"]["tpf"] for key in bearing)
    assert selected_tpf == 1372
    selection_recovery = (
        guarded_n32["baseline_tpf"] - selected_tpf
    ) / (guarded_n32["baseline_tpf"] - guarded_n32["picked_tpf"])
    close(selection_recovery, -1.0420168067226891)
    assert all(row["metrics"]["tpf"] == 0 for row in controls["Prokofiev/Prokofiev/Toccata"]["candidates"])

    sensitivity = {
        f"{guard:.2f}": aggregate_at_guard(bearing, guard)
        for guard in SENSITIVITY_GUARDS
    }
    assert sensitivity["0.00"]["picked_tpf"] == 1145
    close(sensitivity["0.00"]["pooled_relative_tpf_delta"], -0.08253205128205128)
    close(sensitivity["0.00"]["macro_meaner_delta"], -0.60533)
    assert sensitivity["0.00"]["n_pieces_tpf_improved"] == 4
    for guard in ("0.25", "0.50"):
        assert sensitivity[guard]["picked_tpf"] == 1129
        close(sensitivity[guard]["pooled_relative_tpf_delta"], -0.0953525641025641)
        close(sensitivity[guard]["macro_meaner_delta"], -0.558135)
        assert sensitivity[guard]["n_pieces_tpf_improved"] == 6

    subset_reference = exact_subset_reference(bearing)
    assert list(subset_reference["support_counts"].values()) == [3, 1, 10, 18, 2, 1]
    expected_checks = {
        "1": (1224.9375, 1.09375),
        "8": (1168.4645969405701, 3.504797067967257),
        "16": (1149.4562432738821, 4.645037158839935),
        "32": (1129.0, 6.0),
    }
    for n, (expected_tpf, expected_pieces) in expected_checks.items():
        close(subset_reference["curve"][n]["expected_pooled_tpf"], expected_tpf)
        close(
            subset_reference["curve"][n]["expected_pieces_with_support"],
            expected_pieces,
        )

    summary = {
        "oracle_sha256": CONFIRM_AUDIT.sha256(ORACLE_PATH),
        "selection_sha256": provenance["selection_sha256"],
        "frozen_hashes_recorded": True,
        "candidate_rows": sum(len(piece["candidates"]) for piece in sampled.values()),
        "identity_checks": identity_counts,
        "sampled_pieces": len(sampled),
        "sampled_tuplet_bearing": len(bearing),
        "sampled_controls": len(controls),
        "guarded_n32": guarded_n32,
        "selected_tpf_on_bearing_pools": selected_tpf,
        "selection_recovery": selection_recovery,
        "guard_sensitivity": sensitivity,
        "exact_uniform_subset_reference": subset_reference,
        "all_bearing_pools_have_guarded_improvement": True,
        "prokofiev_zero_gt_implies_candidate_tpf_zero": True,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
