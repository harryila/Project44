"""Audit the frozen revision-locality record and its paper claims."""
# The original version of this script also cross-checked a manuscript source file that is
# not part of this repository. That step is now optional and is skipped when the file is absent.

import hashlib
import json
import math
import statistics
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent
RESULT_PATH = REPO / "benchmark" / "revision_locality_v10_1.json"
ORACLE_PATH = REPO / "benchmark" / "oracle_prefix_v10_1.json"
SELECTION_PATH = REPO / "benchmark" / "confirmatory_v10_1_selections.json"
# Optional manuscript source for the cross-check at the end of main(); not part of this repository.
TEX_PATH = REPO / "manuscript" / "main.tex"


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def close(actual, expected, tolerance=1e-12):
    if not math.isclose(float(actual), float(expected), abs_tol=tolerance, rel_tol=0):
        raise AssertionError(f"{actual} != {expected}")


def main():
    result = json.load(open(RESULT_PATH))
    oracle = json.load(open(ORACLE_PATH))
    if result["status"] != "complete":
        raise AssertionError("Locality result is incomplete")
    if len(result["pieces"]) != 6:
        raise AssertionError("Expected exactly six efficacy pieces")
    if result["provenance"]["oracle_sha256"] != sha256(ORACLE_PATH):
        raise AssertionError("Oracle provenance hash mismatch")
    if result["provenance"]["selection_sha256"] != sha256(SELECTION_PATH):
        raise AssertionError("Selection provenance hash mismatch")

    total_positions = 0
    total_changed = 0
    changed_fractions = []
    span_fractions = []
    for key, row in result["pieces"].items():
        oracle_piece = oracle["pieces"][key]
        pick = oracle_piece["prefix_curve"]["32"]["meaner_guarded_oracle"]
        if row["candidate"] != pick["candidate"] or row["seed"] != pick["seed"]:
            raise AssertionError(f"Frozen pick mismatch for {key}")
        expected_outcomes = {
            "baseline_tpf": oracle_piece["baseline"]["tpf"],
            "oracle_tpf": pick["tpf"],
            "delta_tpf": pick["delta_tpf"],
            "baseline_meaner": oracle_piece["baseline"]["meaner_overall"],
            "oracle_meaner": pick["meaner"],
            "delta_meaner": pick["delta_meaner"],
        }
        for field, expected in expected_outcomes.items():
            close(row["frozen_outcomes"][field], expected)

        locality = row["locality"]
        runs = locality["change_runs"]["runs"]
        if len(runs) != locality["change_runs"]["count"]:
            raise AssertionError(f"Run-count mismatch for {key}")
        previous_end = -2
        run_total = 0
        for run in runs:
            if run["start"] <= previous_end + 1:
                raise AssertionError(f"Runs overlap or are not maximal for {key}")
            if run["length"] != run["end_inclusive"] - run["start"] + 1:
                raise AssertionError(f"Run-length mismatch for {key}")
            previous_end = run["end_inclusive"]
            run_total += run["length"]
        if run_total != locality["n_changed_positions"]:
            raise AssertionError(f"Runs do not partition changed positions for {key}")
        close(
            locality["changed_position_fraction"],
            locality["n_changed_positions"] / locality["n_positions"],
        )
        if runs:
            span = runs[-1]["end_inclusive"] - runs[0]["start"] + 1
            if locality["change_span"]["positions"] != span:
                raise AssertionError(f"Change-span mismatch for {key}")
            close(locality["change_span"]["fraction_of_sequence"], span / locality["n_positions"])

        total_positions += locality["n_positions"]
        total_changed += locality["n_changed_positions"]
        changed_fractions.append(locality["changed_position_fraction"])
        span_fractions.append(locality["change_span"]["fraction_of_sequence"])

    aggregate = result["aggregate"]
    if total_positions != aggregate["n_positions"] or total_changed != aggregate["n_changed_positions"]:
        raise AssertionError("Aggregate counts do not match piece rows")
    close(aggregate["micro_changed_position_fraction"], total_changed / total_positions)
    close(
        aggregate["median_piece_changed_position_fraction"],
        statistics.median(changed_fractions),
    )
    close(
        aggregate["min_piece_change_span_fraction"],
        min(span_fractions),
    )
    if (total_changed, total_positions) != (13006, 17921):
        raise AssertionError("Headline locality counts changed")
    if round(100 * total_changed / total_positions, 2) != 72.57:
        raise AssertionError("Headline pooled percentage changed")
    if round(100 * statistics.median(changed_fractions), 2) != 66.14:
        raise AssertionError("Headline median percentage changed")
    if round(100 * min(span_fractions), 2) != 98.07:
        raise AssertionError("Headline minimum span changed")

    # Optional cross-check against the manuscript source; skipped when the file is absent.
    tex = TEX_PATH.read_text() if TEX_PATH.exists() else None
    required_claims = (
        "13,006 of 17,921",
        "72.57\\%",
        "66.14\\%",
        "at least 98.07\\%",
        "Pooled & 13006 / 17921 & 72.57",
    )
    missing = [claim for claim in required_claims if tex is not None and claim not in tex]
    if missing:
        raise AssertionError(f"Paper locality claims missing: {missing}")

    print(
        "REVISION_LOCALITY_AUDIT_OK "
        f"pieces=6 changed={total_changed}/{total_positions} "
        f"micro={100 * total_changed / total_positions:.4f}% "
        f"median={100 * statistics.median(changed_fractions):.4f}% "
        f"min_span={100 * min(span_fractions):.4f}%"
    )


if __name__ == "__main__":
    main()
