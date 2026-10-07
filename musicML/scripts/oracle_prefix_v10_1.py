"""Secondary oracle and prefix-scaling analysis for the frozen V10.1 candidate pools.

This analysis was explicitly declared secondary in bestofn_prereg.md Addendum 2. It makes
no selector or parameter changes: candidate order, seeds, sampling configuration, model,
and source hashes are inherited from the locked GT-blind selection artifact. Every one of
the 32 sampled candidates is evaluated with the exact MusicXML tuplet metric.
"""

import argparse
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

import audit_confirmatory_v10_1 as AUDIT
import confirmatory_v10_1 as CONFIRM
import tuplet_metrics_v2 as TM2
import verifier_bestofn as VB


REPO = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO / "benchmark" / "oracle_prefix_v10_1.json"
DEFAULT_WORK = REPO / "benchmark" / "oracle_prefix_v10_1_work"
PREFIXES = (1, 2, 4, 8, 16, 32)
CORE_INTEGER_FIELDS = (
    "keep_count",
    "trip_phase_count",
    "tuplet_duration_count",
    "kept_tuplet_duration_count",
    "onset_duration_parity_mismatch",
)
TIMING_FIELDS = ("timing_penalty", "timing_groups", "timing_contra")


def resolve_repo_path(value):
    path = Path(value)
    return path if path.is_absolute() else REPO / path


def check_frozen_inputs(selection_path):
    for relative, expected in AUDIT.FROZEN_HASHES.items():
        actual = AUDIT.sha256(REPO / relative)
        if actual != expected:
            raise RuntimeError(f"Frozen hash mismatch for {relative}: {actual}")
    selection_hash = AUDIT.sha256(selection_path)
    if selection_hash != AUDIT.SELECTION_SHA256:
        raise RuntimeError(f"Locked selection hash mismatch: {selection_hash}")
    locked = json.load(open(selection_path))
    AUDIT.assert_selection_is_gt_blind(locked)
    for key, piece in locked["pieces"].items():
        route, config = AUDIT.recompute_route(piece)
        if route != piece["route"] or config != piece["selected_cfg"]:
            raise RuntimeError(f"Locked route no longer reconstructs for {key}")
    return locked


def assert_candidate_identity(key, locked, decoded, piece):
    counts = CONFIRM.V10.rhythm_counts(decoded, piece["n_notes"])
    keep_count = CONFIRM.V10.V8.keep_count(decoded, piece["n_notes"])
    timing, timing_detail = CONFIRM.V10.V6.timing_contradiction(
        decoded, piece["_t"], piece["n_notes"]
    )
    actual = {"keep_count": keep_count, **counts, **timing_detail}
    for field in CORE_INTEGER_FIELDS:
        if int(actual[field]) != int(locked[field]):
            raise RuntimeError(
                f"Candidate record mismatch for {key}/cand{locked['i']:02d} "
                f"field {field}: {actual[field]} != {locked[field]}"
            )
    if not math.isclose(float(timing), float(locked["timing_per_group"]), abs_tol=1e-12, rel_tol=0):
        raise RuntimeError(
            f"Candidate record mismatch for {key}/cand{locked['i']:02d} "
            f"field timing_per_group: {timing} != {locked['timing_per_group']}"
        )
    for field in TIMING_FIELDS:
        if not math.isclose(float(actual[field]), float(locked[field]), abs_tol=1e-6, rel_tol=0):
            raise RuntimeError(
                f"Candidate record mismatch for {key}/cand{locked['i']:02d} "
                f"field {field}: {actual[field]} != {locked[field]}"
            )


def evaluate(y, piece, workdir):
    record = TM2.muster_piece(y, piece, workdir)
    total, missed, onset = TM2.tpf(record)
    return {
        **record,
        "tpf": total,
        "tpf_missed": missed,
        "tpf_onset": onset,
    }


def compact_pick(row, baseline):
    return {
        "source": row["source"],
        "candidate": row.get("candidate"),
        "seed": row.get("seed"),
        "tpf": row["metrics"]["tpf"],
        "meaner": row["metrics"]["meaner_overall"],
        "delta_tpf": row["metrics"]["tpf"] - baseline["tpf"],
        "delta_meaner": row["metrics"]["meaner_overall"] - baseline["meaner_overall"],
    }


def pick_best(rows):
    return min(
        rows,
        key=lambda row: (
            row["metrics"]["tpf"],
            row["metrics"]["meaner_overall"],
            -1 if row.get("candidate") is None else row["candidate"],
        ),
    )


def summarize_prefixes(piece):
    baseline = piece["baseline"]
    baseline_row = {"source": "baseline", "metrics": baseline}
    candidates = sorted(piece["candidates"], key=lambda row: row["candidate"])
    curves = {}
    for n in PREFIXES:
        prefix = candidates[:n]
        if len(prefix) != n:
            raise RuntimeError(f"Expected {n} candidates, found {len(prefix)}")
        sample_best = pick_best(prefix)
        inclusive_best = pick_best([baseline_row, *prefix])
        guarded_rows = [
            row for row in prefix
            if row["metrics"]["meaner_overall"] <= baseline["meaner_overall"] + 0.50
        ]
        guarded_best = pick_best([baseline_row, *guarded_rows])
        curves[str(n)] = {
            "sample_oracle": compact_pick(sample_best, baseline),
            "baseline_inclusive_oracle": compact_pick(inclusive_best, baseline),
            "meaner_guarded_oracle": compact_pick(guarded_best, baseline),
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
    return curves


def aggregate_pick(pieces, n, field):
    rows = []
    for key, piece in pieces.items():
        pick = piece["prefix_curve"][str(n)][field]
        baseline = piece["baseline"]
        rows.append((key, baseline, pick))
    baseline_tpf = sum(row[1]["tpf"] for row in rows)
    picked_tpf = sum(row[2]["tpf"] for row in rows)
    rels = [
        (picked["tpf"] - baseline["tpf"]) / max(baseline["tpf"], 1)
        for _, baseline, picked in rows
    ]
    return {
        "n_pieces": len(rows),
        "baseline_tpf": baseline_tpf,
        "picked_tpf": picked_tpf,
        "delta_tpf": picked_tpf - baseline_tpf,
        "pooled_relative_tpf_delta": (
            (picked_tpf - baseline_tpf) / max(baseline_tpf, 1)
        ),
        "mean_per_piece_relative_tpf_delta": sum(rels) / len(rels),
        "macro_meaner_delta": sum(row[2]["delta_meaner"] for row in rows) / len(rows),
        "n_pieces_tpf_improved": sum(row[2]["delta_tpf"] < 0 for row in rows),
        "picks": {key: picked for key, _, picked in rows},
    }


def finalize(output):
    sampled = {
        key: piece for key, piece in output["pieces"].items()
        if len(piece.get("candidates", [])) == 32
    }
    bearing = {key: piece for key, piece in sampled.items() if piece["gt_tuplets"] >= 20}
    controls = {key: piece for key, piece in sampled.items() if piece["gt_tuplets"] < 20}
    for piece in sampled.values():
        piece["prefix_curve"] = summarize_prefixes(piece)

    aggregate = {"bearing_sampled_pieces": {}, "control_sampled_pieces": {}}
    for n in PREFIXES:
        aggregate["bearing_sampled_pieces"][str(n)] = {
            field: aggregate_pick(bearing, n, field)
            for field in (
                "sample_oracle",
                "baseline_inclusive_oracle",
                "meaner_guarded_oracle",
            )
        }
        aggregate["control_sampled_pieces"][str(n)] = {
            field: aggregate_pick(controls, n, field)
            for field in (
                "sample_oracle",
                "baseline_inclusive_oracle",
                "meaner_guarded_oracle",
            )
        }

    n32 = aggregate["bearing_sampled_pieces"]["32"]
    output["aggregate_prefix_curve"] = aggregate
    output["n32_diagnosis"] = {
        "n_sampled_pieces": len(sampled),
        "n_sampled_tuplet_bearing": len(bearing),
        "n_sampled_controls": len(controls),
        "n_bearing_pools_with_any_tpf_improvement": sum(
            piece["prefix_curve"]["32"]["n_tpf_better_than_baseline"] > 0
            for piece in bearing.values()
        ),
        "n_bearing_pools_with_guarded_tpf_improvement": sum(
            piece["prefix_curve"]["32"]["n_guarded_tpf_better_than_baseline"] > 0
            for piece in bearing.values()
        ),
        "unconstrained_baseline_inclusive_oracle": n32["baseline_inclusive_oracle"],
        "meaner_guarded_baseline_inclusive_oracle": n32["meaner_guarded_oracle"],
        "interpretation_rule": (
            "Good oracle candidates with poor V10 picks indicate a selection bottleneck; "
            "absence of good oracle candidates indicates a sampling/generation bottleneck."
        ),
    }
    output["analysis_complete"] = True
    output["completed_utc"] = datetime.now(timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", default="benchmark/confirmatory_v10_1_selections.json")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--work-root", default=str(DEFAULT_WORK))
    parser.add_argument("--pieces", nargs="*", default=None, help="Optional smoke-test filters")
    parser.add_argument("--max-candidates", type=int, default=32)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    if args.max_candidates < 1 or args.max_candidates > 32:
        raise ValueError("--max-candidates must be in [1, 32]")

    selection_path = resolve_repo_path(args.selection)
    output_path = resolve_repo_path(args.out)
    work_root = resolve_repo_path(args.work_root)
    locked = check_frozen_inputs(selection_path)
    v10_args = CONFIRM.v10_args()
    priors = torch.load(
        REPO / "data" / "duration_priors.pt", map_location="cpu", weights_only=False
    )
    overrides = {stream: (v10_args.topk, v10_args.temp) for stream in v10_args.streams}
    paths = CONFIRM.confirmatory_pieces()
    if args.pieces:
        paths = [
            piece for piece in paths
            if any(
                token.lower() in f"{piece['composer']}/{piece['piece']}".lower()
                for token in args.pieces
            )
        ]

    model = VB.load_any_checkpoint(v10_args.ckpt, "cpu")
    model.eval()
    model.to("cpu")
    if {parameter.device.type for parameter in model.parameters()} != {"cpu"}:
        raise RuntimeError("Oracle analysis must run entirely on CPU")

    output = {
        "analysis": "frozen V10.1 secondary oracle and prefix scaling",
        "status": "in_progress",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "definitions": {
            "prefixes": list(PREFIXES),
            "candidate_order": "locked candidate index order, seeds 1000 through 1031",
            "sample_oracle": "lowest TPF sample; ties use MeanER then candidate index",
            "baseline_inclusive_oracle": (
                "lowest TPF among baseline and prefix samples; ties use MeanER then baseline/index"
            ),
            "meaner_guarded_oracle": (
                "baseline-inclusive oracle restricted to samples with MeanER <= baseline + 0.50"
            ),
            "metric": (
                "TPF = missed exact explicit-tuplet GT notes + onset-error events on matched "
                "exact explicit-tuplet GT notes"
            ),
        },
        "provenance": {
            "selection_path": str(selection_path.relative_to(REPO)),
            "selection_sha256": AUDIT.sha256(selection_path),
            "frozen_hashes": AUDIT.FROZEN_HASHES,
            "exact_evaluator_sha256": AUDIT.sha256(
                REPO / "scripts" / "muster_tuplet_decompose.py"
            ),
            "metric_adapter_sha256": AUDIT.sha256(REPO / "scripts" / "tuplet_metrics_v2.py"),
            "selector_changed": False,
            "ground_truth_used_for_selection": False,
            "post_selection_secondary_analysis": True,
            "candidate_identity_check": (
                "seed plus locked keep/rhythm/timing fingerprint for every candidate; "
                "full stream hash additionally verified for every V10-selected candidate"
            ),
        },
        "pieces": {},
    }
    can_resume = not args.no_resume and not args.pieces and args.max_candidates == 32
    if can_resume and output_path.exists():
        previous = json.load(open(output_path))
        previous_provenance = previous.get("provenance", {})
        if (
            previous_provenance.get("selection_sha256")
            == output["provenance"]["selection_sha256"]
            and previous_provenance.get("exact_evaluator_sha256")
            == output["provenance"]["exact_evaluator_sha256"]
            and previous_provenance.get("metric_adapter_sha256")
            == output["provenance"]["metric_adapter_sha256"]
        ):
            if previous.get("analysis_complete"):
                print(f"Complete compatible result already exists at {output_path}")
                return
            output = previous
            for piece in output.get("pieces", {}).values():
                for row in piece.get("candidates", []):
                    row.setdefault("identity_check", "full_locked_record_reconstruction")
            print(f"Resuming compatible partial result from {output_path}", flush=True)
    t0 = time.time()
    for piece in paths:
        key = f"{piece['composer']}/{piece['piece']}"
        locked_piece = locked["pieces"][key]
        tag = piece["piece"].replace("/", "_")
        print(f"\n===== {key} route={locked_piece['route']} =====", flush=True)

        y_base = VB.decode(model, piece["_x"], None, seed=1)
        if CONFIRM.stream_hash(y_base) != locked_piece["base_hash"]:
            raise RuntimeError(f"Baseline stream hash mismatch for {key}")
        piece_output = output["pieces"].get(key)
        if piece_output is None:
            baseline = evaluate(y_base, piece, work_root / tag / "baseline")
            piece_output = {
                "route": locked_piece["route"],
                "selected_cfg": locked_piece["selected_cfg"],
                "gt_tuplets": baseline["n_gt_tuplet"],
                "baseline": baseline,
                "candidates": [],
            }
        else:
            baseline = piece_output["baseline"]

        if not locked_piece["sample_records"]:
            y_selected = CONFIRM.redecode_selected(
                model, piece, v10_args, priors, overrides, locked_piece
            )
            if CONFIRM.stream_hash(y_selected) != locked_piece["selected_hash"]:
                raise RuntimeError(f"Selected stream hash mismatch for {key}")
            if "selected" not in piece_output:
                piece_output["selected"] = evaluate(
                    y_selected, piece, work_root / tag / "selected"
                )
            piece_output["pool_status"] = "not_sampled_by_frozen_route"
            output["pieces"][key] = piece_output
            output_path.parent.mkdir(parents=True, exist_ok=True)
            json.dump(output, open(output_path, "w"), indent=2)
            continue

        piece_output["pool_status"] = "frozen_n32_sample_pool"
        existing_indices = {row["candidate"] for row in piece_output["candidates"]}
        for locked_record in locked_piece["sample_records"][:args.max_candidates]:
            index = locked_record["i"]
            seed = locked_record["seed"]
            if index in existing_indices:
                print(f"  cand{index:02d}: resume skip", flush=True)
                continue
            y_candidate = VB.decode(model, piece["_x"], overrides, seed=seed)
            assert_candidate_identity(key, locked_record, y_candidate, piece)
            if locked_piece["selected_cfg"] and index == locked_piece["selected_cfg"].get("candidate"):
                if CONFIRM.stream_hash(y_candidate) != locked_piece["selected_hash"]:
                    raise RuntimeError(f"Selected candidate hash mismatch for {key}/cand{index:02d}")
            metrics = evaluate(
                y_candidate, piece, work_root / tag / f"candidate_{index:02d}"
            )
            if metrics["n_gt_tuplet"] != baseline["n_gt_tuplet"]:
                raise RuntimeError(f"Candidate-dependent GT tuplet population for {key}/cand{index:02d}")
            row = {
                "source": "candidate",
                "candidate": index,
                "seed": seed,
                "eligible": locked_record["eligible"],
                "in_timing_screen": locked_record.get("in_timing_screen", False),
                "selected_by_v10": bool(
                    locked_piece["selected_cfg"]
                    and index == locked_piece["selected_cfg"].get("candidate")
                ),
                "stream_sha256": CONFIRM.stream_hash(y_candidate),
                "identity_check": "locked_keep_rhythm_timing_fingerprint",
                "metrics": metrics,
            }
            piece_output["candidates"].append(row)
            print(
                f"  cand{index:02d}: TPF={metrics['tpf']} "
                f"MeanER={metrics['meaner_overall']:.3f}",
                flush=True,
            )
            output["pieces"][key] = piece_output
            output["elapsed_s"] = round(time.time() - t0, 1)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            json.dump(output, open(output_path, "w"), indent=2)

        piece_output["candidates"].sort(key=lambda row: row["candidate"])
        if len(piece_output["candidates"]) == 32:
            piece_output["prefix_curve"] = summarize_prefixes(piece_output)

    if args.pieces or args.max_candidates != 32:
        output["status"] = "smoke_complete"
        output["analysis_complete"] = False
    else:
        finalize(output)
        output["status"] = "complete"
    output["elapsed_s"] = round(time.time() - t0, 1)
    json.dump(output, open(output_path, "w"), indent=2)
    print(f"\nWrote {output_path} ({output['elapsed_s']}s)", flush=True)


if __name__ == "__main__":
    main()
