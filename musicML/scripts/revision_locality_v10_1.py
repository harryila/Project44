"""Describe baseline-to-oracle revision locality in the frozen V10.1 pools.

The protocol is fixed in paper/REVISION_LOCALITY_PROTOCOL.md. This script compares
aligned raw decoder streams, not serialized MusicXML, and does not select or score any
new candidate.
"""

import argparse
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

import audit_confirmatory_v10_1 as AUDIT
import confirmatory_v10_1 as CONFIRM
import local_repair_v4 as V4
import oracle_prefix_v10_1 as ORACLE
import verifier_bestofn as VB


REPO = Path(__file__).resolve().parent.parent
DEFAULT_ORACLE = REPO / "benchmark" / "oracle_prefix_v10_1.json"
DEFAULT_SELECTION = REPO / "benchmark" / "confirmatory_v10_1_selections.json"
DEFAULT_OUT = REPO / "benchmark" / "revision_locality_v10_1.json"
MIN_GT_TUPLETS = 20
EXPECTED_PIECES = 6


def resolve_repo_path(value):
    path = Path(value)
    return path if path.is_absolute() else REPO / path


def maximal_runs(mask):
    indices = np.flatnonzero(mask)
    if not len(indices):
        return []
    split_points = np.flatnonzero(np.diff(indices) != 1) + 1
    groups = np.split(indices, split_points)
    return [
        {
            "start": int(group[0]),
            "end_inclusive": int(group[-1]),
            "length": int(len(group)),
        }
        for group in groups
    ]


def rhythm_streams(decoded, n_positions):
    streams = V4.streams_np(decoded)
    values = {
        "offset": streams["off"],
        "duration": streams["dur"],
        "downbeat": streams["db"],
        "keep": streams["pad"],
    }
    lengths = {name: len(value) for name, value in values.items()}
    if any(length < n_positions for length in lengths.values()):
        raise RuntimeError(
            f"Decoded stream shorter than input: n={n_positions}, lengths={lengths}"
        )
    return {name: value[:n_positions] for name, value in values.items()}


def compare_streams(baseline, candidate, n_positions):
    base = rhythm_streams(baseline, n_positions)
    cand = rhythm_streams(candidate, n_positions)
    common_kept = base["keep"] & cand["keep"]
    keep_flip = base["keep"] != cand["keep"]
    offset_change = common_kept & (base["offset"] != cand["offset"])
    duration_change = common_kept & (base["duration"] != cand["duration"])
    downbeat_change = common_kept & (base["downbeat"] != cand["downbeat"])
    changed = keep_flip | offset_change | duration_change | downbeat_change
    runs = maximal_runs(changed)
    changed_indices = np.flatnonzero(changed)
    n_changed = int(changed.sum())

    if n_changed:
        first = int(changed_indices[0])
        last = int(changed_indices[-1])
        span = last - first + 1
        run_lengths = [run["length"] for run in runs]
        median_run = float(statistics.median(run_lengths))
        max_run = max(run_lengths)
    else:
        first = last = None
        span = 0
        median_run = 0.0
        max_run = 0

    n_common_kept = int(common_kept.sum())
    return {
        "n_positions": n_positions,
        "baseline_kept": int(base["keep"].sum()),
        "oracle_kept": int(cand["keep"].sum()),
        "common_kept": n_common_kept,
        "n_changed_positions": n_changed,
        "changed_position_fraction": n_changed / n_positions,
        "head_changes": {
            "keep_flips": int(keep_flip.sum()),
            "offset_among_common_kept": int(offset_change.sum()),
            "duration_among_common_kept": int(duration_change.sum()),
            "downbeat_among_common_kept": int(downbeat_change.sum()),
        },
        "head_change_fractions": {
            "keep_flips_over_all_positions": float(keep_flip.mean()),
            "offset_over_common_kept": (
                float(offset_change.sum() / n_common_kept) if n_common_kept else 0.0
            ),
            "duration_over_common_kept": (
                float(duration_change.sum() / n_common_kept) if n_common_kept else 0.0
            ),
            "downbeat_over_common_kept": (
                float(downbeat_change.sum() / n_common_kept) if n_common_kept else 0.0
            ),
        },
        "change_runs": {
            "count": len(runs),
            "median_length": median_run,
            "max_length": max_run,
            "max_run_share_of_changes": max_run / n_changed if n_changed else 0.0,
            "runs": runs,
        },
        "change_span": {
            "first_index": first,
            "last_index": last,
            "positions": span,
            "fraction_of_sequence": span / n_positions,
        },
    }


def aggregate(piece_rows):
    total_positions = sum(row["locality"]["n_positions"] for row in piece_rows.values())
    total_changed = sum(
        row["locality"]["n_changed_positions"] for row in piece_rows.values()
    )
    total_common_kept = sum(row["locality"]["common_kept"] for row in piece_rows.values())
    head_totals = {
        key: sum(row["locality"]["head_changes"][key] for row in piece_rows.values())
        for key in (
            "keep_flips",
            "offset_among_common_kept",
            "duration_among_common_kept",
            "downbeat_among_common_kept",
        )
    }
    fractions = [
        row["locality"]["changed_position_fraction"] for row in piece_rows.values()
    ]
    spans = [
        row["locality"]["change_span"]["fraction_of_sequence"]
        for row in piece_rows.values()
    ]
    run_counts = [row["locality"]["change_runs"]["count"] for row in piece_rows.values()]
    return {
        "n_pieces": len(piece_rows),
        "n_positions": total_positions,
        "n_changed_positions": total_changed,
        "micro_changed_position_fraction": total_changed / total_positions,
        "median_piece_changed_position_fraction": float(statistics.median(fractions)),
        "min_piece_changed_position_fraction": min(fractions),
        "max_piece_changed_position_fraction": max(fractions),
        "median_piece_change_span_fraction": float(statistics.median(spans)),
        "min_piece_change_span_fraction": min(spans),
        "max_piece_change_span_fraction": max(spans),
        "median_piece_change_run_count": float(statistics.median(run_counts)),
        "min_piece_change_run_count": min(run_counts),
        "max_piece_change_run_count": max(run_counts),
        "head_change_totals": head_totals,
        "head_change_micro_fractions": {
            "keep_flips_over_all_positions": head_totals["keep_flips"] / total_positions,
            "offset_over_common_kept": (
                head_totals["offset_among_common_kept"] / total_common_kept
            ),
            "duration_over_common_kept": (
                head_totals["duration_among_common_kept"] / total_common_kept
            ),
            "downbeat_over_common_kept": (
                head_totals["downbeat_among_common_kept"] / total_common_kept
            ),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", default=str(DEFAULT_SELECTION))
    parser.add_argument("--oracle", default=str(DEFAULT_ORACLE))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    selection_path = resolve_repo_path(args.selection)
    oracle_path = resolve_repo_path(args.oracle)
    output_path = resolve_repo_path(args.out)
    locked = ORACLE.check_frozen_inputs(selection_path)
    oracle = json.load(open(oracle_path))
    if not oracle.get("analysis_complete"):
        raise RuntimeError("Oracle analysis is not complete")

    oracle_pieces = {
        key: piece
        for key, piece in oracle["pieces"].items()
        if piece.get("pool_status") == "frozen_n32_sample_pool"
        and piece["gt_tuplets"] >= MIN_GT_TUPLETS
    }
    if len(oracle_pieces) != EXPECTED_PIECES:
        raise RuntimeError(
            f"Expected {EXPECTED_PIECES} tuplet-bearing frozen pools, "
            f"found {len(oracle_pieces)}"
        )

    paths = {
        f"{piece['composer']}/{piece['piece']}": piece
        for piece in CONFIRM.confirmatory_pieces()
    }
    if not set(oracle_pieces).issubset(paths):
        raise RuntimeError("Oracle piece population is not in the confirmatory split")

    v10_args = CONFIRM.v10_args()
    overrides = {
        stream: (v10_args.topk, v10_args.temp) for stream in v10_args.streams
    }
    model = VB.load_any_checkpoint(v10_args.ckpt, "cpu")
    model.eval()
    model.to("cpu")
    if {parameter.device.type for parameter in model.parameters()} != {"cpu"}:
        raise RuntimeError("Locality analysis must run entirely on CPU")

    output = {
        "analysis": "frozen V10.1 baseline-to-guarded-oracle revision locality",
        "status": "in_progress",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "paper/REVISION_LOCALITY_PROTOCOL.md",
        "definitions": {
            "population": (
                "six frozen N=32 pools with at least 20 exact GT tuplet notes"
            ),
            "comparison": (
                "greedy baseline versus the already selected N=32 baseline-inclusive "
                "+0.50 MeanER-guarded oracle candidate"
            ),
            "position": "one aligned performed-note input index",
            "changed_position": (
                "keep decision differs, or both outputs keep the position and offset, "
                "duration, or downbeat argmax differs"
            ),
            "change_run": "maximal run of consecutive changed input indices",
        },
        "provenance": {
            "selection_path": str(selection_path.relative_to(REPO)),
            "selection_sha256": AUDIT.sha256(selection_path),
            "oracle_path": str(oracle_path.relative_to(REPO)),
            "oracle_sha256": AUDIT.sha256(oracle_path),
            "checkpoint": f"MIDI2ScoreTransformer/{v10_args.ckpt}",
            "checkpoint_sha256": AUDIT.sha256(
                REPO / "MIDI2ScoreTransformer" / v10_args.ckpt
            ),
            "candidate_pools_changed": False,
            "candidate_selection_changed": False,
            "new_ground_truth_evaluation": False,
            "locality_uses_ground_truth": False,
            "oracle_candidate_was_previously_ground_truth_selected": True,
        },
        "pieces": {},
    }

    with torch.inference_mode():
        for key in sorted(oracle_pieces):
            piece = paths[key]
            oracle_piece = oracle_pieces[key]
            locked_piece = locked["pieces"][key]
            pick = oracle_piece["prefix_curve"]["32"]["meaner_guarded_oracle"]
            if pick["source"] != "candidate":
                raise RuntimeError(f"Guarded oracle is not a candidate for {key}")
            candidate_index = int(pick["candidate"])
            candidate_seed = int(pick["seed"])
            locked_record = next(
                record
                for record in locked_piece["sample_records"]
                if int(record["i"]) == candidate_index
            )
            if int(locked_record["seed"]) != candidate_seed:
                raise RuntimeError(f"Candidate seed mismatch for {key}")

            baseline = VB.decode(model, piece["_x"], None, seed=1)
            baseline_hash = CONFIRM.stream_hash(baseline)
            if baseline_hash != locked_piece["base_hash"]:
                raise RuntimeError(f"Baseline stream hash mismatch for {key}")

            candidate = VB.decode(
                model, piece["_x"], overrides, seed=candidate_seed
            )
            ORACLE.assert_candidate_identity(key, locked_record, candidate, piece)
            candidate_hash = CONFIRM.stream_hash(candidate)
            oracle_candidate_row = next(
                row
                for row in oracle_piece["candidates"]
                if int(row["candidate"]) == candidate_index
            )
            recorded_hash = oracle_candidate_row.get("stream_sha256")
            if recorded_hash is not None and candidate_hash != recorded_hash:
                raise RuntimeError(f"Oracle candidate stream hash mismatch for {key}")

            locality = compare_streams(baseline, candidate, piece["n_notes"])
            output["pieces"][key] = {
                "gt_tuplets": oracle_piece["gt_tuplets"],
                "candidate": candidate_index,
                "seed": candidate_seed,
                "baseline_stream_sha256": baseline_hash,
                "candidate_stream_sha256": candidate_hash,
                "candidate_identity_check": (
                    "locked keep/rhythm/timing fingerprint"
                    + (" plus recorded stream hash" if recorded_hash is not None else "")
                ),
                "frozen_outcomes": {
                    "baseline_tpf": oracle_piece["baseline"]["tpf"],
                    "oracle_tpf": pick["tpf"],
                    "delta_tpf": pick["delta_tpf"],
                    "baseline_meaner": oracle_piece["baseline"]["meaner_overall"],
                    "oracle_meaner": pick["meaner"],
                    "delta_meaner": pick["delta_meaner"],
                },
                "locality": locality,
            }
            print(
                f"{key}: candidate={candidate_index} "
                f"changed={locality['n_changed_positions']}/{locality['n_positions']} "
                f"runs={locality['change_runs']['count']}",
                flush=True,
            )

    output["aggregate"] = aggregate(output["pieces"])
    output["status"] = "complete"
    output["completed_utc"] = datetime.now(timezone.utc).isoformat()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as handle:
        json.dump(output, handle, indent=2)
    print(f"Wrote {output_path}", flush=True)


if __name__ == "__main__":
    main()
