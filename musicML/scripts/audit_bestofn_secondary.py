"""Recompute the post-selection secondary analyses used in the best-of-N writeup."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SELECTIONS = ROOT / "benchmark/confirmatory_v10_1_selections.json"
CONFIRMATION = ROOT / "benchmark/confirmatory_v10_1_results.json"
ORACLE = ROOT / "benchmark/oracle_prefix_v10_1.json"
OUTPUT = ROOT / "benchmark/bestofn_secondary_audit.json"


def load(path: Path):
    with path.open() as stream:
        return json.load(stream)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rankdata(values: list[float]) -> list[float]:
    """Average ranks for ties, matching the usual Spearman definition."""
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + 1 + end) / 2.0
        for index in order[start:end]:
            ranks[index] = rank
        start = end
    return ranks


def pearson(left: list[float], right: list[float]) -> float:
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right))
    left_ss = sum((a - left_mean) ** 2 for a in left)
    right_ss = sum((b - right_mean) ** 2 for b in right)
    return numerator / math.sqrt(left_ss * right_ss)


def spearman(left: list[float], right: list[float]) -> float:
    return pearson(rankdata(left), rankdata(right))


def metric_totals(rows: list[dict]) -> dict:
    return {
        key: sum(row[key] for row in rows)
        for key in ("tpf", "tpf_missed", "tpf_onset")
    }


def guarded_oracle(piece: dict, guard: float) -> dict:
    baseline = piece["baseline"]
    choices = [{"source": "baseline", "candidate": -1, "metrics": baseline}]
    choices.extend(
        candidate
        for candidate in piece["candidates"]
        if candidate["metrics"]["meaner_overall"]
        <= baseline["meaner_overall"] + guard + 1e-12
    )
    return min(
        choices,
        key=lambda row: (
            row["metrics"]["tpf"],
            row["metrics"]["meaner_overall"],
            row.get("candidate", -1),
        ),
    )


def main() -> None:
    selections = load(SELECTIONS)
    confirmation = load(CONFIRMATION)
    oracle = load(ORACLE)

    efficacy = [
        key
        for key, piece in oracle["pieces"].items()
        if piece["gt_tuplets"] >= 20 and len(piece["candidates"]) == 32
    ]
    assert len(efficacy) == 6

    base_rows = []
    sealed_rows = []
    minimum_loss_rows = []
    oracle_rows = {str(guard): [] for guard in (0.0, 0.25, 0.5)}
    per_piece = {}
    gate_totals = {
        stage: {"all_samples": 0, "guarded_improvements": 0}
        for stage in ("saved", "model_loss_band", "all_eligibility", "timing_screen")
    }

    for key in efficacy:
        piece = oracle["pieces"][key]
        baseline = piece["baseline"]
        base_record = selections["pieces"][key]["base_record"]
        sample_records = {
            row["i"]: row for row in selections["pieces"][key]["sample_records"]
        }
        candidates = {row["candidate"]: row for row in piece["candidates"]}
        assert set(sample_records) == set(candidates) == set(range(32))

        minimum_sample_loss = min(
            sample_records.values(), key=lambda row: (row["model_loss"], row["i"])
        )
        if base_record["model_loss"] <= minimum_sample_loss["model_loss"]:
            minimum_loss = {
                "source": "baseline",
                "i": -1,
                "seed": None,
                "model_loss": base_record["model_loss"],
            }
            minimum_metrics = baseline
        else:
            minimum_loss = {"source": "candidate", **minimum_sample_loss}
            minimum_metrics = candidates[minimum_loss["i"]]["metrics"]
        sealed = confirmation["pieces"][key]["selected"]

        indices = sorted(sample_records)
        tpfs = [candidates[index]["metrics"]["tpf"] for index in indices]
        model_losses = [sample_records[index]["model_loss"] for index in indices]
        timing_scores = [sample_records[index]["timing_per_group"] for index in indices]

        guarded_indices = {
            index
            for index in indices
            if candidates[index]["metrics"]["tpf"] < baseline["tpf"]
            and candidates[index]["metrics"]["meaner_overall"]
            <= baseline["meaner_overall"] + 0.5 + 1e-12
        }
        stage_indices = {
            "saved": set(indices),
            "model_loss_band": {
                index
                for index in indices
                if 0.02 <= sample_records[index]["model_delta"] <= 0.03
            },
            "all_eligibility": {
                index for index in indices if sample_records[index]["eligible"]
            },
            "timing_screen": {
                index
                for index in indices
                if sample_records[index].get("in_timing_screen", False)
            },
        }
        for stage, surviving in stage_indices.items():
            gate_totals[stage]["all_samples"] += len(surviving)
            gate_totals[stage]["guarded_improvements"] += len(
                surviving & guarded_indices
            )

        guard_picks = {}
        for guard in (0.0, 0.25, 0.5):
            pick = guarded_oracle(piece, guard)
            oracle_rows[str(guard)].append(pick["metrics"])
            guard_picks[str(guard)] = {
                "source": pick["source"],
                "candidate": pick.get("candidate"),
                "tpf": pick["metrics"]["tpf"],
                "delta_meaner": (
                    pick["metrics"]["meaner_overall"] - baseline["meaner_overall"]
                ),
            }

        base_rows.append(baseline)
        sealed_rows.append(sealed)
        minimum_loss_rows.append(minimum_metrics)
        per_piece[key] = {
            "gt_tuplets": piece["gt_tuplets"],
            "baseline_tpf": baseline["tpf"],
            "sealed_tpf": sealed["tpf"],
            "minimum_model_loss": {
                "choice_set": "baseline_plus_32_samples",
                "source": minimum_loss["source"],
                "candidate": minimum_loss["i"],
                "seed": minimum_loss["seed"],
                "model_loss": minimum_loss["model_loss"],
                "baseline_model_loss": base_record["model_loss"],
                "minimum_sample_model_loss": minimum_sample_loss["model_loss"],
                "tpf": minimum_metrics["tpf"],
                "delta_meaner": (
                    minimum_metrics["meaner_overall"] - baseline["meaner_overall"]
                ),
            },
            "feature_fidelity": {
                "spearman_model_loss_tpf": spearman(model_losses, tpfs),
                "spearman_timing_tpf": spearman(timing_scores, tpfs),
                "n": len(indices),
            },
            "gate_audit": {
                "guarded_improvements": len(guarded_indices),
                "surviving": {
                    stage: len(indices_at_stage)
                    for stage, indices_at_stage in stage_indices.items()
                },
                "guarded_surviving": {
                    stage: len(indices_at_stage & guarded_indices)
                    for stage, indices_at_stage in stage_indices.items()
                },
            },
            "oracle": guard_picks,
        }

    totals = {
        "greedy": metric_totals(base_rows),
        "sealed_v10_1": metric_totals(sealed_rows),
        "minimum_model_loss": metric_totals(minimum_loss_rows),
        "oracle_0.0": metric_totals(oracle_rows["0.0"]),
        "oracle_0.25": metric_totals(oracle_rows["0.25"]),
        "oracle_0.5": metric_totals(oracle_rows["0.5"]),
    }
    baseline_tpf = totals["greedy"]["tpf"]
    oracle_tpf = totals["oracle_0.5"]["tpf"]
    denominator = baseline_tpf - oracle_tpf

    for name, rows in (
        ("sealed_v10_1", sealed_rows),
        ("minimum_model_loss", minimum_loss_rows),
    ):
        totals[name]["rho_32_against_oracle_0.5"] = (
            baseline_tpf - totals[name]["tpf"]
        ) / denominator
        deltas = [
            row["meaner_overall"] - baseline["meaner_overall"]
            for row, baseline in zip(rows, base_rows)
        ]
        totals[name]["macro_delta_meaner"] = sum(deltas) / len(deltas)
        totals[name]["guard_0.5_violations"] = sum(delta > 0.5 + 1e-12 for delta in deltas)

    macro_tpf_per_gt_tuplet_pct = {
        name: 100
        * sum(row["tpf"] / oracle["pieces"][key]["gt_tuplets"] for key, row in zip(efficacy, rows))
        / len(efficacy)
        for name, rows in (
            ("greedy", base_rows),
            ("sealed_v10_1", sealed_rows),
            ("minimum_model_loss", minimum_loss_rows),
            ("oracle_0.5", oracle_rows["0.5"]),
        )
    }

    leave_one_out = {}
    for omitted in efficacy:
        kept = [key for key in efficacy if key != omitted]
        base = sum(per_piece[key]["baseline_tpf"] for key in kept)
        sealed = sum(per_piece[key]["sealed_tpf"] for key in kept)
        guarded = sum(per_piece[key]["oracle"]["0.5"]["tpf"] for key in kept)
        leave_one_out[omitted] = {
            "baseline_tpf": base,
            "sealed_tpf": sealed,
            "oracle_tpf": guarded,
            "oracle_relative_reduction": (base - guarded) / base,
            "rho_32": (base - sealed) / (base - guarded),
        }

    model_correlations = [
        per_piece[key]["feature_fidelity"]["spearman_model_loss_tpf"]
        for key in efficacy
    ]
    timing_correlations = [
        per_piece[key]["feature_fidelity"]["spearman_timing_tpf"]
        for key in efficacy
    ]
    summary = {
        "analysis": "best-of-N post-selection secondary audit",
        "inputs_sha256": {
            "selections": sha256(SELECTIONS),
            "confirmation": sha256(CONFIRMATION),
            "oracle": sha256(ORACLE),
        },
        "efficacy_pieces": efficacy,
        "totals": totals,
        "macro_tpf_per_gt_tuplet_pct": macro_tpf_per_gt_tuplet_pct,
        "per_piece": per_piece,
        "leave_one_piece_out": leave_one_out,
        "gate_audit": {
            "definition": "sample TPF below greedy and MeanER no more than 0.50 points above greedy",
            "model_delta": "candidate model loss minus greedy model loss",
            "stages": gate_totals,
        },
        "feature_fidelity_summary": {
            "model_loss_range": [min(model_correlations), max(model_correlations)],
            "model_loss_median": sorted(model_correlations)[2:4],
            "timing_range": [min(timing_correlations), max(timing_correlations)],
            "timing_median": sorted(timing_correlations)[2:4],
        },
    }
    summary["feature_fidelity_summary"]["model_loss_median"] = sum(
        summary["feature_fidelity_summary"]["model_loss_median"]
    ) / 2
    summary["feature_fidelity_summary"]["timing_median"] = sum(
        summary["feature_fidelity_summary"]["timing_median"]
    ) / 2

    assert totals["greedy"]["tpf"] == 1248
    assert totals["sealed_v10_1"]["tpf"] == 1372
    assert totals["minimum_model_loss"]["tpf"] == 1239
    assert totals["oracle_0.0"]["tpf"] == 1145
    assert totals["oracle_0.25"]["tpf"] == 1129
    assert totals["oracle_0.5"]["tpf"] == 1129
    assert totals["sealed_v10_1"]["guard_0.5_violations"] == 4
    assert totals["minimum_model_loss"]["guard_0.5_violations"] == 4
    assert gate_totals == {
        "saved": {"all_samples": 192, "guarded_improvements": 35},
        "model_loss_band": {"all_samples": 17, "guarded_improvements": 0},
        "all_eligibility": {"all_samples": 7, "guarded_improvements": 0},
        "timing_screen": {"all_samples": 5, "guarded_improvements": 0},
    }
    assert all(
        row["minimum_model_loss"]["source"] == "candidate"
        and row["minimum_model_loss"]["minimum_sample_model_loss"]
        < row["minimum_model_loss"]["baseline_model_loss"]
        for row in per_piece.values()
    )
    assert math.isclose(macro_tpf_per_gt_tuplet_pct["greedy"], 34.503431688108904)
    assert math.isclose(macro_tpf_per_gt_tuplet_pct["sealed_v10_1"], 36.88042551289487)
    assert math.isclose(macro_tpf_per_gt_tuplet_pct["oracle_0.5"], 31.10262571578731)

    with OUTPUT.open("w") as stream:
        json.dump(summary, stream, indent=2)
        stream.write("\n")
    print(f"PASS: wrote {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
