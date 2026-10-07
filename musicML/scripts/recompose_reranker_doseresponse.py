#!/usr/bin/env python3
"""Reapply the exact MusicXML tuplet labels to the preserved boost sweep outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import muster_tuplet_decompose as MTD
from muster.muster import parse_line_to_dict


REPO = Path(__file__).resolve().parent.parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def exact_record(workdir: Path) -> dict:
    paths = {
        "aggregate": workdir / "out.txt",
        "gt_xml": workdir / "gt.xml",
        "gt_fmt3x": workdir / "gt_fmt3x.txt",
        "err_detail": workdir / "est_err_detail.txt",
        "auto_match": workdir / "est_auto_match.txt",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing preserved boost-sweep files: {missing}")

    with paths["aggregate"].open() as handle:
        aggregate = parse_line_to_dict(handle.readline())
    gt_info = MTD.parse_gt_fmt3x(paths["gt_fmt3x"])
    matched = MTD.parse_matched_gtids(paths["auto_match"], gt_info)
    errors = MTD.parse_err_detail(paths["err_detail"])
    record = MTD.decompose_piece(gt_info, matched, errors, aggregate)
    if record["_onset_cost_residual"] not in (0, 1, 2):
        raise ValueError(f"Unexpected onset residual in {workdir}")
    if abs(record["_check_combined_OffsetER"] - record["_check_harness_OffsetER"]) >= 0.005:
        raise ValueError(f"Offset recombination failed in {workdir}")
    return record, {name: sha256(path) for name, path in paths.items()}


def aggregate(rows: list[dict]) -> dict:
    totals = {
        "n_tuplet": 0,
        "n_nontuplet": 0,
        "n_onset_err_tuplet": 0,
        "n_onset_err_nontuplet": 0,
        "n_offset_err_tuplet": 0,
        "n_offset_err_nontuplet": 0,
        "n_decoded_triplet": 0,
        "n_decoded_notes": 0,
    }
    for row in rows:
        for key in totals:
            totals[key] += int(row[key])

    def rate(events: int, population: int) -> float | None:
        return 100.0 * events / population if population else None

    return {
        "meaner_overall": sum(row["meaner_overall"] for row in rows) / len(rows),
        "onset_err_tuplet": rate(totals["n_onset_err_tuplet"], totals["n_tuplet"]),
        "onset_err_nontuplet": rate(
            totals["n_onset_err_nontuplet"], totals["n_nontuplet"]
        ),
        "offset_err_tuplet": rate(totals["n_offset_err_tuplet"], totals["n_tuplet"]),
        "offset_err_nontuplet": rate(
            totals["n_offset_err_nontuplet"], totals["n_nontuplet"]
        ),
        "decoded_triplet_rate": (
            totals["n_decoded_triplet"] / totals["n_decoded_notes"]
            if totals["n_decoded_notes"]
            else 0.0
        ),
        **totals,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", default="benchmark/reranker_doseresponse.json"
    )
    parser.add_argument(
        "--work-root", default="benchmark/rerank_work"
    )
    parser.add_argument(
        "--out", default="benchmark/reranker_doseresponse_exact.json"
    )
    args = parser.parse_args()

    input_path = (REPO / args.input).resolve()
    work_root = (REPO / args.work_root).resolve()
    output_path = (REPO / args.out).resolve()
    source = json.loads(input_path.read_text())

    output = dict(source)
    output["schema_version"] = 2
    output["generated_utc"] = datetime.now(timezone.utc).isoformat()
    output["provenance"] = {
        "source_artifact": input_path.relative_to(REPO).as_posix(),
        "source_artifact_sha256": sha256(input_path),
        "exact_evaluator": "scripts/muster_tuplet_decompose.py",
        "exact_evaluator_sha256": sha256(REPO / "scripts/muster_tuplet_decompose.py"),
        "method": (
            "Recomposed from preserved MUSTER correspondence and error files using exact "
            "MusicXML time-modification labels. No model decode or selector was rerun."
        ),
    }

    for lambda_key, block in output["by_lambda"].items():
        rows = []
        for source_row in block["per_piece"]:
            tag = source_row["piece"].replace("/", "_")
            workdir = work_root / f"lam{lambda_key}" / tag
            exact, hashes = exact_record(workdir)
            if abs(float(source_row["meaner_overall"]) - float(exact["meaner_overall"])) > 1e-9:
                raise ValueError(f"MeanER changed for lambda={lambda_key}")

            row = dict(source_row)
            for key in (
                "meaner_overall",
                "onset_err_tuplet",
                "onset_err_nontuplet",
                "offset_err_tuplet",
                "offset_err_nontuplet",
                "n_tuplet",
                "n_nontuplet",
                "n_onset_err_tuplet",
                "n_onset_err_nontuplet",
                "n_offset_err_tuplet",
                "n_offset_err_nontuplet",
                "n_gt_tuplet",
                "n_miss_tuplet",
            ):
                row[key] = exact[key]
            row["intermediate_sha256"] = hashes
            rows.append(row)
        block["per_piece"] = rows
        block["aggregate"] = aggregate(rows)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n")
    print(f"Wrote {output_path} sha256={sha256(output_path)}")


if __name__ == "__main__":
    main()
