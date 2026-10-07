#!/usr/bin/env python3
"""Reproduce the numerical claims in the paper's intervention record."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path


sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"


def load(name: str) -> dict:
    return json.loads((ARTIFACTS / name).read_text())


def close(actual: float, expected: float, tolerance: float = 1e-9) -> None:
    if not math.isclose(float(actual), float(expected), abs_tol=tolerance, rel_tol=0.0):
        raise AssertionError(f"{actual} != {expected}")


def one_piece(record: dict) -> dict:
    pieces = list(record["pieces"].values())
    if len(pieces) != 1:
        raise AssertionError(f"Expected one piece, found {len(pieces)}")
    return pieces[0]


def audit_boost() -> None:
    record = load("reranker_doseresponse_exact.json")
    expected = {
        "0.0": (0, 1307, 43.58974358974359, 10.3743),
        "1.0": (2, 1307, 45.45454545454545, 10.5259),
        "2.0": (5, 1308, 42.857142857142854, 10.5018),
        "3.0": (281, 1311, 43.58974358974359, 14.0612),
        "4.0": (280, 1313, 50.617283950617285, 15.6855),
        "6.0": (599, 1317, 52.38095238095238, 19.7788),
        "8.0": (644, 1309, 43.75, 19.9992),
    }
    if set(record["by_lambda"]) != set(expected):
        raise AssertionError("Unexpected boost levels")
    for level, (emitted, kept, onset_error, meaner) in expected.items():
        aggregate = record["by_lambda"][level]["aggregate"]
        if aggregate["n_decoded_triplet"] != emitted:
            raise AssertionError("Boost sweep off-dyadic count changed")
        if aggregate["n_decoded_notes"] != kept:
            raise AssertionError("Boost sweep retained-note population changed")
        close(aggregate["decoded_triplet_rate"], emitted / kept)
        close(aggregate["onset_err_tuplet"], onset_error)
        close(aggregate["meaner_overall"], meaner)
    if record["emission_population"] != "generated notes with pad probability > 0.5":
        raise AssertionError("Boost sweep keep rule changed")
    old = load("reranker_doseresponse_unmasked_positions.json")
    close(old["by_lambda"]["8.0"]["aggregate"]["decoded_triplet_rate"], 644 / 1408)
    sanity = one_piece({"pieces": record["sanity"]})
    if not sanity["byte_identical"] or sanity["n_diff"] != 0:
        raise AssertionError("Lambda-zero identity check failed")


def v3b_score(row: dict, evidence_weight: float) -> float:
    return (
        (row["v1"] + 0.5 * row["v2"]) / row["n"]
        + evidence_weight * row["v3"] / row["n_kept"]
    )


def audit_v3b() -> None:
    record = load("verifier_bestofn_dev_v3b.json")
    if (record["n"], record["topk"], record["temp"], record["streams"]) != (
        8,
        15,
        0.7,
        ["offset", "duration"],
    ):
        raise AssertionError("V3b sampling configuration changed")
    piece = one_piece(record)
    candidates = sorted(piece["candidates"], key=lambda row: row["i"])
    if [row["seed"] for row in candidates] != list(range(1000, 1008)):
        raise AssertionError("V3b seeds changed")
    candidate_2, candidate_7 = candidates[2], candidates[7]
    if candidate_2["v1"] + candidate_2["v2"] != 7:
        raise AssertionError("Candidate 2 consistency count changed")
    if candidate_7["v1"] + candidate_7["v2"] != 0:
        raise AssertionError("Candidate 7 should make no recorded triplet attempt")
    close(candidate_2["v3"], 0.2604)
    close(candidate_7["v3"], 0.2562)
    close(candidate_2["meaner_overall"], 10.701)
    close(candidate_7["meaner_overall"], 11.8423)
    close(candidate_2["onset_err_tuplet"], 39.53488372093023)
    close(candidate_7["onset_err_tuplet"], 42.16867469879518)
    for weight in (1.0, 2.0, 4.0, 8.0, 16.0):
        winner = min(candidates, key=lambda row: (v3b_score(row, weight), row["i"]))
        if winner["i"] != 7:
            raise AssertionError(f"V3b winner changed at evidence weight {weight}")

    exact = load("routed_selector_v10_n32_final_metric_v2.json")
    scriabin = next(iter(exact["pieces"].values()))
    if scriabin["baseline"]["tpf"] != 55 or scriabin["selected"]["tpf"] != 47:
        raise AssertionError("Exact Scriabin TPF cross-check failed")


def audit_v4() -> None:
    record = load("local_repair_v4_dev.json")
    if (record["n"], record["topk"], record["temp"]) != (8, 15, 0.7):
        raise AssertionError("V4 sampling configuration changed")
    piece = one_piece(record)
    if piece["baseline"]["tpf"] != 55 or piece["repaired"]["tpf"] != 65:
        raise AssertionError("V4 TPF values changed")
    close(piece["baseline"]["meaner_overall"], 10.3743)
    close(piece["repaired"]["meaner_overall"], 16.6251)
    if piece["windows"] != [{"lo": 0, "hi": 1326}]:
        raise AssertionError("V4 saturation window changed")
    if piece["drift_ok"]:
        raise AssertionError("V4 drift gate should reject the repair")


def find_piece(record: dict, model: str, composer: str) -> dict:
    rows = record["models"][model]["per_piece"]
    return next(row for row in rows if row["composer"] == composer)


def audit_discarded_labeler() -> None:
    invalid = load("muster_tuplet_decomposed_invalid_tpqn24.json")
    corrected = load("muster_tuplet_decomposed.json")
    for composer in ("Bach", "Prokofiev"):
        old = find_piece(invalid, "released", composer)
        new = find_piece(corrected, "released", composer)
        if old["n_tuplet"] <= 0:
            raise AssertionError(f"Discarded labeler did not expose its {composer} error")
        if new["n_gt_tuplet"] != 0 or new["n_tuplet"] != 0:
            raise AssertionError(f"Corrected exact labels should contain no {composer} tuplets")

    invalid_confirmation = load("confirmatory_v10_1_results_invalid_tpqn24.json")
    corrected_confirmation = load("confirmatory_v10_1_results.json")
    if invalid_confirmation["decision_rule"]["CONFIRMED"]:
        raise AssertionError("Discarded confirmation unexpectedly passed")
    if corrected_confirmation["decision_rule"]["CONFIRMED"]:
        raise AssertionError("Corrected confirmation unexpectedly passed")


def main() -> None:
    audit_boost()
    audit_v3b()
    audit_v4()
    audit_discarded_labeler()
    print("PASS: boost, V3b, V4, and discarded-labeler claims reproduce exactly")


if __name__ == "__main__":
    main()
