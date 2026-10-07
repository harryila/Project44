"""Read-only audit of the local-rank and hidden-tail headline values."""

import json
import math
import sys
from pathlib import Path

sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"


def load(name):
    with open(ARTIFACTS / name) as handle:
        return json.load(handle)


def close(actual, expected):
    assert math.isclose(actual, expected, rel_tol=0, abs_tol=1e-12), (actual, expected)


def main():
    topk = load("topk_offset_diag.json")
    pieces = topk["per_piece"]
    target_count = sum(row["n_gt_triplet_notes"] for row in pieces.values())
    assert target_count == 1184
    weighted = {}
    for field in ("top1_hit_rate", "top5_hit_rate", "top15_hit_rate"):
        hit_count = sum(
            round(row[field] * row["n_gt_triplet_notes"])
            for row in pieces.values()
        )
        weighted[field] = hit_count / target_count
        close(topk["aggregate"][f"weighted_{field}"], weighted[field])
    assert [round(weighted[field] * target_count) for field in (
        "top1_hit_rate", "top5_hit_rate", "top15_hit_rate"
    )] == [434, 1053, 1166]
    weighted_mean_rank = sum(
        row["mean_rank_correct_triplet_bucket"] * row["n_gt_triplet_notes"]
        for row in pieces.values()
    ) / target_count
    close(topk["aggregate"]["weighted_mean_rank_correct_triplet_bucket"], weighted_mean_rank)
    close(weighted_mean_rank, 3.1875)
    expected_rollout = {
        "Liszt": (494, 2445),
        "Ravel": (266, 4230),
        "Scriabin": (0, 1307),
        "Mozart": (0, 2452),
    }
    for name, (count, total) in expected_rollout.items():
        assert pieces[name]["free_running_n_triplet_argmax"] == count
        assert pieces[name]["free_running_n_notes"] == total
        close(
            pieces[name]["free_running_triplet_argmax_rate"],
            count / total,
        )
        assert pieces[name]["free_running_keep_rule"] == "pad probability > 0.5"
    old_rollout = load("topk_offset_diag_unmasked_positions.json")
    assert old_rollout["paper_summary"]["free_running_greedy_off_dyadic_emission"]["Ravel"] == {
        "n_notes": 4544,
        "n_off_dyadic": 1135,
        "rate": 1135 / 4544,
    }
    assert topk["provenance"]["superseded_unmasked_artifact_sha256"] == (
        "37975a30379b4711ae1f96ef168816c71909b2e5634c39cb0512f5e856aa670f"
    )
    expected_target_rates = {
        "Liszt": 222 / 2674,
        "Ravel": 872 / 4979,
        "Scriabin": 66 / 1382,
        "Mozart": 24 / 2462,
    }
    for name, expected in expected_target_rates.items():
        close(pieces[name]["gt_triplet_rate"], expected)

    tail = load("muster_tuplet_decomposed.json")
    summaries = {}
    for model_name, model in tail["models"].items():
        assert len(model["per_piece"]) == 14
        for row in model["per_piece"]:
            if row["n_tuplet"]:
                close(
                    row["onset_err_tuplet"],
                    100 * row["n_onset_err_tuplet"] / row["n_tuplet"],
                )
                close(
                    row["offset_err_tuplet"],
                    100 * row["n_offset_err_tuplet"] / row["n_tuplet"],
                )
            if row["n_nontuplet"]:
                close(
                    row["onset_err_nontuplet"],
                    100 * row["n_onset_err_nontuplet"] / row["n_nontuplet"],
                )
                close(
                    row["offset_err_nontuplet"],
                    100 * row["n_offset_err_nontuplet"] / row["n_nontuplet"],
                )
            assert row["n_tuplet"] + row["n_nontuplet"] == row["n_matched"]
            assert row["n_miss_tuplet"] + row["n_tuplet"] == row["n_gt_tuplet"]
        rows = [row for row in model["per_piece"] if row["n_gt_tuplet"] >= 20]
        ratios = [row["onset_err_tuplet"] / row["onset_err_nontuplet"] for row in rows]
        elevated = [value for value in ratios if value > 1]
        summaries[model_name] = {
            "pieces": len(rows),
            "elevated": len(elevated),
            "elevated_min": min(elevated),
            "elevated_max": max(elevated),
        }
    assert summaries["released"]["pieces"] == 11
    assert summaries["released"]["elevated"] == 8
    assert summaries["ours_ssl_classical"]["pieces"] == 11
    assert summaries["ours_ssl_classical"]["elevated"] == 9
    close(summaries["released"]["elevated_min"], 1.8388509102081574)
    close(summaries["released"]["elevated_max"], 9.884288781884434)
    close(summaries["ours_ssl_classical"]["elevated_min"], 1.3139274518584863)
    close(summaries["ours_ssl_classical"]["elevated_max"], 7.3180004777830865)

    released = {row["composer"]: row for row in tail["models"]["released"]["per_piece"]}
    expected_ratios = {
        "Chopin": 9.884288781884434,
        "Brahms": 8.547303271441203,
        "Beethoven": 5.037825059101655,
        "Mozart": 4.964583333333333,
        "Scriabin": 4.556075808249721,
    }
    for name, expected in expected_ratios.items():
        row = released[name]
        close(row["onset_err_tuplet"] / row["onset_err_nontuplet"], expected)
    second = {row["composer"]: row for row in tail["models"]["ours_ssl_classical"]["per_piece"]}
    for model in (released, second):
        assert model["Bach"]["n_gt_tuplet"] == model["Bach"]["n_tuplet"] == 0
        assert model["Prokofiev"]["n_gt_tuplet"] == model["Prokofiev"]["n_tuplet"] == 0

    print(
        json.dumps(
            {
                "teacher_forced_targets": target_count,
                "hit_at_1": weighted["top1_hit_rate"],
                "hit_at_5": weighted["top5_hit_rate"],
                "hit_at_15": weighted["top15_hit_rate"],
                "mean_rank": weighted_mean_rank,
                "rollout_counts": expected_rollout,
                "target_off_dyadic_rates": expected_target_rates,
                "tail": summaries,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
