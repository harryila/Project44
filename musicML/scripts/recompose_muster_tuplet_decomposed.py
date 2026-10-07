"""Rebuild the canonical hidden-tail JSON from preserved exact MUSTER work directories.

The expensive model decode and MUSTER passes are kept under ``benchmark/decomp_work``.
This script reapplies the current exact MusicXML tuplet decomposition to those immutable
intermediates, verifies every row against MUSTER, and emits paper-facing structured summaries.
"""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import muster_tuplet_decompose as MTD
from muster.muster import parse_line_to_dict


REPO = Path(__file__).resolve().parent.parent


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def recompose_row(tag, row):
    workdir = REPO / "benchmark" / "decomp_work" / tag / row["piece"].replace("/", "_")
    required = {
        "aggregate": workdir / "out.txt",
        "gt_fmt3x": workdir / "gt_fmt3x.txt",
        "err_detail": workdir / "est_err_detail.txt",
        "auto_match": workdir / "est_auto_match.txt",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing preserved MUSTER files for {tag}/{row['piece']}: {missing}")

    with open(required["aggregate"]) as handle:
        aggregate = parse_line_to_dict(handle.readline())
    gt_info = MTD.parse_gt_fmt3x(required["gt_fmt3x"])
    matched = MTD.parse_matched_gtids(required["auto_match"], gt_info)
    errors = MTD.parse_err_detail(required["err_detail"])
    decomposed = MTD.decompose_piece(gt_info, matched, errors, aggregate)

    offset_ok = (
        decomposed["_check_combined_OffsetER"] is None
        or decomposed["_check_harness_OffsetER"] is None
        or abs(
            decomposed["_check_combined_OffsetER"]
            - decomposed["_check_harness_OffsetER"]
        ) < 0.005
    )
    residual = decomposed.get("_onset_cost_residual")
    onset_ok = residual is None or 0 <= residual <= 2
    if not (offset_ok and onset_ok):
        raise ValueError(
            f"MUSTER recombination failed for {tag}/{row['piece']}: "
            f"offset={decomposed['_check_combined_OffsetER']} vs "
            f"{decomposed['_check_harness_OffsetER']}, onset residual={residual}"
        )

    return {
        "composer": row["composer"],
        "piece": row["piece"],
        "n_notes": row["n_notes"],
        **decomposed,
        "muster_aggregate": aggregate,
        "selfcheck_ok": True,
    }


def paper_summary(rows):
    per_piece = []
    for row in rows:
        onset_t = row.get("onset_err_tuplet")
        onset_n = row.get("onset_err_nontuplet")
        ratio = onset_t / onset_n if onset_t is not None and onset_n else None
        per_piece.append({
            "composer": row["composer"],
            "piece": row["piece"],
            "n_gt_tuplet": row.get("n_gt_tuplet", 0),
            "n_matched_tuplet": row.get("n_tuplet", 0),
            "n_matched_nontuplet": row.get("n_nontuplet", 0),
            "onset_err_tuplet": onset_t,
            "onset_err_nontuplet": onset_n,
            "onset_tuplet_over_nontuplet": ratio,
        })
    per_piece.sort(
        key=lambda row: (
            row["onset_tuplet_over_nontuplet"] is not None,
            row["onset_tuplet_over_nontuplet"] or -1,
        ),
        reverse=True,
    )
    bearing = [row for row in per_piece if row["n_gt_tuplet"] > 0]
    headline = [row for row in per_piece if row["n_gt_tuplet"] >= 20]
    elevated = [
        row for row in headline
        if row["onset_tuplet_over_nontuplet"] is not None
        and row["onset_tuplet_over_nontuplet"] > 1.0
    ]
    ratios = [row["onset_tuplet_over_nontuplet"] for row in elevated]
    return {
        "definition": (
            "Per-piece ratio of MUSTER onset-error rate on exact explicit non-grace tuplet "
            "notes to the rate on matched non-tuplet notes."
        ),
        "n_pieces": len(per_piece),
        "n_zero_tuplet_controls": len(per_piece) - len(bearing),
        "n_tuplet_bearing": len(bearing),
        "headline_min_gt_tuplets": 20,
        "n_headline_pieces": len(headline),
        "n_headline_ratio_above_one": len(elevated),
        "elevated_ratio_min": min(ratios) if ratios else None,
        "elevated_ratio_max": max(ratios) if ratios else None,
        "per_piece": per_piece,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.out)
    if not input_path.is_absolute():
        input_path = REPO / input_path
    if not output_path.is_absolute():
        output_path = REPO / output_path
    input_path = input_path.resolve()
    output_path = output_path.resolve()
    source = json.load(open(input_path))
    models = {}
    for tag, model in source["models"].items():
        rows = [recompose_row(tag, row) for row in model["per_piece"]]
        models[tag] = {
            "ckpt": model["ckpt"],
            "elapsed_s": model.get("elapsed_s"),
            "per_piece": rows,
            "aggregate": MTD.aggregate(rows),
            "paper_summary": paper_summary(rows),
        }

    output = {
        "schema_version": 2,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "population": "ASAP tier-1 test, shortest performance per score, 14 scores",
        "models": models,
        "_metric_defs": {
            "onset_err": (
                "fraction (%) of matched GT notes in the exact paper bucket flagged "
                "OnsetError(shift|scale) by MUSTER"
            ),
            "offset_err": (
                "fraction (%) of matched GT notes in the exact paper bucket flagged "
                "OffsetError by MUSTER"
            ),
            "tuplet_tag": "exact MusicXML <time-modification> joined by Fmt3x GtID",
            "matched_population": (
                "exact GT IDs in ScoreMatchEvaluation's GT-to-EST correspondence block; "
                "this reproduces MUSTER's nGT-nMiss denominator"
            ),
            "self_consistency": (
                "note-weighted bucket errors reproduce MUSTER; the onset rhythm-correction "
                "residual is recorded explicitly"
            ),
        },
        "provenance": {
            "source_json": str(input_path.relative_to(REPO)),
            "source_json_sha256": file_hash(input_path),
            "exact_evaluator": "scripts/muster_tuplet_decompose.py",
            "exact_evaluator_sha256": file_hash(REPO / "scripts/muster_tuplet_decompose.py"),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as handle:
        json.dump(output, handle, indent=2)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
