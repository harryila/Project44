#!/usr/bin/env python3
"""Build paper figures from the frozen, audited benchmark artifacts."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FIGURES = HERE / "figures"
if (ROOT / "benchmark").exists():
    BENCHMARK = ROOT / "benchmark"
elif (HERE / "artifacts").exists():
    BENCHMARK = HERE / "artifacts"
elif (HERE / "supplement" / "artifacts").exists():
    BENCHMARK = HERE / "supplement" / "artifacts"
else:
    raise FileNotFoundError("Could not locate the canonical paper artifacts")

TEAL = "#0B6E69"
RUST = "#B24A33"
OCHRE = "#C58B20"
CHARCOAL = "#30343B"
MID_GRAY = "#7A8088"
LIGHT_GRAY = "#D7DADF"


def load(name: str) -> dict:
    with (BENCHMARK / name).open() as handle:
        return json.load(handle)


def configure() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.5,
            "axes.titlesize": 8.5,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.axisbelow": True,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.transparent": False,
            "savefig.facecolor": "white",
        }
    )


def save(fig: plt.Figure, name: str) -> None:
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(FIGURES / f"{name}.png", dpi=220, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def local_support_and_rollout() -> None:
    data = load("topk_offset_diag.json")
    agg = data["aggregate"]

    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.15), gridspec_kw={"wspace": 0.34})

    ranks = ["hit@1", "hit@5", "hit@15"]
    values = [
        100 * agg["weighted_top1_hit_rate"],
        100 * agg["weighted_top5_hit_rate"],
        100 * agg["weighted_top15_hit_rate"],
    ]
    bars = axes[0].bar(ranks, values, color=[CHARCOAL, OCHRE, TEAL], width=0.62)
    axes[0].set_ylim(0, 108)
    axes[0].set_ylabel("Correct offset bucket (%)")
    axes[0].set_title(
        "A  Teacher-forced rank (1,184 targets)", loc="left", fontweight="bold", fontsize=8
    )
    axes[0].grid(axis="y", color=LIGHT_GRAY, linewidth=0.6)
    for bar, value in zip(bars, values):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            value + 2.3,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    composers = ["Liszt", "Ravel", "Scriabin", "Mozart"]
    rollout = [100 * data["per_piece"][name]["free_running_triplet_argmax_rate"] for name in composers]
    target = [100 * data["per_piece"][name]["gt_triplet_rate"] for name in composers]
    x = list(range(len(composers)))
    bars = axes[1].bar(x, rollout, color=[TEAL, OCHRE, RUST, CHARCOAL], width=0.66)
    axes[1].scatter(
        x,
        target,
        marker="D",
        s=27,
        facecolor="white",
        edgecolor=CHARCOAL,
        linewidth=1.0,
        zorder=4,
    )
    axes[1].set_ylim(0, 29)
    axes[1].set_ylabel("Off-dyadic offsets (%)")
    axes[1].set_title("B  Greedy bars; target diamonds", loc="left", fontweight="bold", fontsize=8)
    axes[1].grid(axis="y", color=LIGHT_GRAY, linewidth=0.6)
    axes[1].set_xticks(x, composers)
    axes[1].tick_params(axis="x", rotation=18)
    for bar, value in zip(bars, rollout):
        label_x = bar.get_x() + bar.get_width() / 2 - (0.13 if value == 0 else 0)
        axes[1].text(
            label_x,
            value + (0.25 if value == 0 else 0.7),
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    save(fig, "local_support_rollout")


def hidden_tail() -> None:
    data = load("muster_tuplet_decomposed.json")
    released = {row["composer"]: row for row in data["models"]["released"]["per_piece"]}
    second = {
        row["composer"]: row for row in data["models"]["ours_ssl_classical"]["per_piece"]
    }

    composers = [name for name, row in released.items() if row["n_gt_tuplet"] >= 20]
    ratios = {
        name: (
            released[name]["onset_err_tuplet"] / released[name]["onset_err_nontuplet"],
            second[name]["onset_err_tuplet"] / second[name]["onset_err_nontuplet"],
        )
        for name in composers
    }
    composers.sort(key=lambda name: ratios[name][0])

    fig, ax = plt.subplots(figsize=(5.5, 3.0))
    y = list(range(len(composers)))
    for yi, name in zip(y, composers):
        x1, x2 = ratios[name]
        ax.plot([x1, x2], [yi, yi], color=LIGHT_GRAY, linewidth=1.2, zorder=1)
    ax.scatter([ratios[name][0] for name in composers], y, color=TEAL, s=28, label="Released checkpoint", zorder=3)
    ax.scatter(
        [ratios[name][1] for name in composers],
        y,
        color=RUST,
        marker="s",
        s=22,
        label="Second checkpoint",
        zorder=3,
    )
    ax.axvline(1.0, color=CHARCOAL, linewidth=0.9, linestyle="--")
    ax.set_xscale("log")
    ax.set_xlim(0.42, 11.4)
    ax.set_xticks([0.5, 1, 2, 4, 8])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}x"))
    ax.set_yticks(y, composers)
    ax.set_xlabel("Tuplet-onset error divided by non-tuplet-onset error")
    ax.set_title(
        "Explicit tuplets form a hidden error tail across checkpoints",
        loc="left",
        fontweight="bold",
    )
    ax.grid(axis="x", color=LIGHT_GRAY, linewidth=0.6, which="major")
    ax.legend(loc="lower right", frameon=False, ncol=2, handletextpad=0.45, columnspacing=1.0)

    save(fig, "hidden_tail")


def subset_min_pmf(piece: dict, n: int, guard: float = 0.50) -> dict[int, float]:
    baseline = piece["baseline"]
    base_tpf = baseline["tpf"]
    effective = []
    for row in piece["candidates"]:
        metrics = row["metrics"]
        if metrics["meaner_overall"] <= baseline["meaner_overall"] + guard + 1e-12:
            effective.append(min(base_tpf, metrics["tpf"]))
        else:
            effective.append(base_tpf)

    denominator = math.comb(len(effective), n)
    pmf = {}
    for value in sorted(set(effective)):
        n_ge = sum(candidate >= value for candidate in effective)
        n_gt = sum(candidate > value for candidate in effective)
        ways_ge = math.comb(n_ge, n) if n_ge >= n else 0
        ways_gt = math.comb(n_gt, n) if n_gt >= n else 0
        if ways_ge > ways_gt:
            pmf[value] = (ways_ge - ways_gt) / denominator
    return pmf


def convolve_pmfs(pmfs: list[dict[int, float]]) -> dict[int, float]:
    combined = {0: 1.0}
    for pmf in pmfs:
        updated = defaultdict(float)
        for left, left_prob in combined.items():
            for right, right_prob in pmf.items():
                updated[left + right] += left_prob * right_prob
        combined = dict(updated)
    return combined


def pmf_quantile(pmf: dict[int, float], probability: float) -> int:
    cumulative = 0.0
    for value, mass in sorted(pmf.items()):
        cumulative += mass
        if cumulative >= probability - 1e-15:
            return value
    raise RuntimeError("Invalid finite-pool distribution")


def sequence_support_and_selection() -> None:
    oracle = load("oracle_prefix_v10_1.json")
    confirm = load("confirmatory_v10_1_results.json")
    curve = oracle["aggregate_prefix_curve"]["bearing_sampled_pieces"]

    prefixes = [1, 2, 4, 8, 16, 32]
    reductions = [
        -100 * curve[str(prefix)]["meaner_guarded_oracle"]["pooled_relative_tpf_delta"]
        for prefix in prefixes
    ]
    improved = [
        curve[str(prefix)]["meaner_guarded_oracle"]["n_pieces_tpf_improved"]
        for prefix in prefixes
    ]
    bearing = {
        key: piece
        for key, piece in oracle["pieces"].items()
        if piece.get("pool_status") == "frozen_n32_sample_pool"
        and piece["gt_tuplets"] >= 20
    }
    baseline_tpf = sum(piece["baseline"]["tpf"] for piece in bearing.values())
    expected_reductions = []
    reduction_p05 = []
    reduction_p95 = []
    for prefix in prefixes:
        pooled = convolve_pmfs(
            [subset_min_pmf(piece, prefix) for piece in bearing.values()]
        )
        expected_tpf = sum(value * mass for value, mass in pooled.items())
        expected_reductions.append(100 * (baseline_tpf - expected_tpf) / baseline_tpf)
        reduction_p05.append(
            100 * (baseline_tpf - pmf_quantile(pooled, 0.95)) / baseline_tpf
        )
        reduction_p95.append(
            100 * (baseline_tpf - pmf_quantile(pooled, 0.05)) / baseline_tpf
        )

    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.2), gridspec_kw={"wspace": 0.34})
    x = list(range(len(prefixes)))
    axes[0].fill_between(
        x,
        reduction_p05,
        reduction_p95,
        color=MID_GRAY,
        alpha=0.14,
        linewidth=0,
        label="Uniform subsets, 90%",
    )
    axes[0].plot(
        x,
        expected_reductions,
        color=MID_GRAY,
        linestyle="--",
        linewidth=1.4,
        label="Subset expectation",
    )
    axes[0].plot(
        x,
        reductions,
        color=TEAL,
        marker="o",
        linewidth=1.8,
        markersize=4.5,
        label="Frozen prefix",
    )
    axes[0].set_xticks(x, prefixes)
    axes[0].set_ylim(0, 11.4)
    axes[0].set_xlabel("Frozen sample-prefix size N")
    axes[0].set_ylabel("Guarded oracle TPF reduction (%)")
    axes[0].set_title("A  Prefix oracle with subset reference", loc="left", fontweight="bold", fontsize=8)
    axes[0].grid(axis="y", color=LIGHT_GRAY, linewidth=0.6)
    for xi, value, count in zip(x, reductions, improved):
        axes[0].text(xi, value + 0.45, f"{count}/6", ha="center", va="bottom", color=CHARCOAL)
    axes[0].legend(loc="upper left", frameon=False, handlelength=1.7, fontsize=6.2)

    picks = oracle["n32_diagnosis"]["meaner_guarded_baseline_inclusive_oracle"]["picks"]
    order = ["Haydn", "Brahms", "Debussy", "Beethoven", "Schubert", "Chopin"]
    selected_delta = []
    oracle_delta = []
    qualifying = []
    for composer in order:
        confirm_key = next(key for key in confirm["pieces"] if key.startswith(f"{composer}/"))
        piece = confirm["pieces"][confirm_key]
        selected_delta.append(piece["selected"]["tpf"] - piece["baseline"]["tpf"])
        oracle_key = next(key for key in picks if key.startswith(f"{composer}/"))
        oracle_delta.append(picks[oracle_key]["delta_tpf"])
        piece = oracle["pieces"][oracle_key]
        baseline = piece["baseline"]
        qualifying.append(
            sum(
                row["metrics"]["tpf"] < baseline["tpf"]
                and row["metrics"]["meaner_overall"]
                <= baseline["meaner_overall"] + 0.50 + 1e-12
                for row in piece["candidates"]
            )
        )

    y = list(range(len(order)))
    for yi, left, right in zip(y, selected_delta, oracle_delta):
        axes[1].plot([left, right], [yi, yi], color=LIGHT_GRAY, linewidth=1.2, zorder=1)
    axes[1].scatter(selected_delta, y, marker="x", linewidth=1.8, color=RUST, s=35, label="Frozen selector", zorder=3)
    axes[1].scatter(oracle_delta, y, marker="o", color=TEAL, s=24, label="Guarded oracle", zorder=3)
    for yi, value, count in zip(y, oracle_delta, qualifying):
        axes[1].text(
            value - 3.0,
            yi - 0.16,
            f"{count}/32",
            ha="right",
            va="bottom",
            color=TEAL,
            fontsize=6.2,
            fontweight="bold",
        )
    axes[1].axvline(0, color=CHARCOAL, linewidth=0.9)
    axes[1].set_yticks(y, order)
    axes[1].invert_yaxis()
    axes[1].set_xlim(-68, 58)
    axes[1].set_xlabel("TPF change from baseline")
    axes[1].set_title(
        "B  Frozen selection versus oracle", loc="left", fontweight="bold", fontsize=8
    )
    axes[1].grid(axis="x", color=LIGHT_GRAY, linewidth=0.6)

    save(fig, "sequence_support_selection")


def main() -> None:
    configure()
    local_support_and_rollout()
    hidden_tail()
    sequence_support_and_selection()
    print(f"Wrote figures to {FIGURES}")


if __name__ == "__main__":
    main()
