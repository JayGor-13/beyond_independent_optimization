"""Create publication-ready figures and tables from Experiment 1's summary CSV.

Run from the repository root:
    python tools/plot_compression_results.py

Or provide an alternate CSV:
    python tools/plot_compression_results.py --input path/to/summary.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "results/published/deepseek/compression_routing/summary.csv"
DEFAULT_OUTPUT = ROOT / "artifacts/compression_plots"

METHODS = ["l2", "random"]
LABELS = {"l2": "L2 retention", "random": "Random retention"}
COLORS = {"l2": "#0072B2", "random": "#D55E00"}  # Okabe-Ito palette
MARKERS = {"l2": "o", "random": "s"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def save(fig: plt.Figure, output_dir: Path, name: str, dpi: int) -> None:
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"{name}.{suffix}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def condition_rows(data: pd.DataFrame) -> pd.DataFrame:
    rows = data[data["method"].isin(METHODS)].copy()
    rows["retention_pct"] = rows["actual_retention"] * 100
    rows["accuracy_pct"] = rows["accuracy"] * 100
    rows["accuracy_delta_pp"] = rows["accuracy_delta"] * 100
    rows["accuracy_low_pp"] = rows["accuracy_delta_ci95_low"] * 100
    rows["accuracy_high_pp"] = rows["accuracy_delta_ci95_high"] * 100
    return rows.sort_values(["method", "actual_retention"])


def plot_accuracy(
    rows: pd.DataFrame, baseline: pd.Series, output_dir: Path, dpi: int
) -> None:
    fig, axis = plt.subplots(figsize=(6.5, 4.1))
    for method in METHODS:
        group = rows[rows["method"] == method].sort_values("actual_retention")
        y_error = np.vstack(
            (
                group["accuracy_delta_pp"] - group["accuracy_low_pp"],
                group["accuracy_high_pp"] - group["accuracy_delta_pp"],
            )
        )
        axis.errorbar(
            group["retention_pct"],
            group["accuracy_pct"],
            yerr=y_error,
            color=COLORS[method],
            marker=MARKERS[method],
            linewidth=2,
            capsize=3,
            label=LABELS[method],
        )
    axis.axhline(
        baseline["accuracy"] * 100,
        color="black",
        linestyle="--",
        linewidth=1.2,
        label=f"Unpruned baseline ({baseline['accuracy'] * 100:.1f}%)",
    )
    axis.set(
        xlabel="Visual-token retention (%)",
        ylabel="ScienceQA accuracy (%)",
        xlim=(20, 100),
        ylim=(55, 74),
    )
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    save(fig, output_dir, "fig1_accuracy_vs_retention", dpi)


def plot_routing(rows: pd.DataFrame, output_dir: Path, dpi: int) -> None:
    metrics = [
        ("racr_example", "RACR (%)", "Routing assignment change rate"),
        ("top1_flip", "Top-1 expert flips (%)", "Dominant-expert reassignment"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.9), sharex=True)
    for axis, (metric, ylabel, title) in zip(axes, metrics):
        for method in METHODS:
            group = rows[rows["method"] == method].sort_values("actual_retention")
            axis.plot(
                group["retention_pct"],
                group[metric] * 100,
                color=COLORS[method],
                marker=MARKERS[method],
                linewidth=2,
                label=LABELS[method],
            )
        axis.set_title(title, fontsize=10)
        axis.set(xlabel="Visual-token retention (%)", ylabel=ylabel, xlim=(20, 80))
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False, loc="upper right")
    fig.tight_layout()
    save(fig, output_dir, "fig2_routing_disruption_vs_retention", dpi)


def plot_matched_comparison(rows: pd.DataFrame, output_dir: Path, dpi: int) -> None:
    metrics = [
        ("accuracy_delta_pp", "Accuracy change (pp)"),
        ("racr_example", "RACR (%)"),
        ("top1_flip", "Top-1 expert flips (%)"),
    ]
    retentions = sorted(rows["retention_pct"].unique())
    positions = np.arange(len(retentions))
    width = 0.36
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.8))
    for axis, (metric, ylabel) in zip(axes, metrics):
        for index, method in enumerate(METHODS):
            group = (
                rows[rows["method"] == method]
                .set_index("retention_pct")
                .loc[retentions]
            )
            values = group[metric].to_numpy() * (
                100 if metric != "accuracy_delta_pp" else 1
            )
            axis.bar(
                positions + (index - 0.5) * width,
                values,
                width,
                color=COLORS[method],
                label=LABELS[method],
            )
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set(
            ylabel=ylabel,
            xlabel="Visual-token retention",
            xticks=positions,
            xticklabels=[f"{value:.0f}%" for value in retentions],
        )
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False, loc="lower left")
    fig.tight_layout()
    save(fig, output_dir, "fig3_matched_method_comparison", dpi)


def markdown_tables(rows: pd.DataFrame, baseline: pd.Series) -> str:
    table = rows.copy()
    table["Method"] = table["method"].map(LABELS)
    table["Retention"] = table["retention_pct"].map(lambda value: f"{value:.0f}%")
    table["Accuracy"] = table["accuracy_pct"].map(lambda value: f"{value:.1f}")
    table["Δ accuracy (pp), 95% CI"] = table.apply(
        lambda row: (
            f"{row.accuracy_delta_pp:+.1f} [{row.accuracy_low_pp:+.1f}, {row.accuracy_high_pp:+.1f}]"
        ),
        axis=1,
    )
    table["RACR"] = table["racr_example"].map(lambda value: f"{value * 100:.1f}%")
    table["Top-1 flips"] = table["top1_flip"].map(lambda value: f"{value * 100:.1f}%")
    table["Top-k changed"] = table["exact_topk_change"].map(
        lambda value: f"{value * 100:.1f}%"
    )
    table["Δ load CV"] = table["load_cv_delta"].map(lambda value: f"{value:+.3f}")
    display = table[
        [
            "Method",
            "Retention",
            "Accuracy",
            "Δ accuracy (pp), 95% CI",
            "RACR",
            "Top-1 flips",
            "Top-k changed",
            "Δ load CV",
        ]
    ]

    matched = rows.pivot(
        index="retention_pct",
        columns="method",
        values=["racr_example", "top1_flip", "accuracy_delta_pp"],
    )
    match_rows = []
    for retention, row in matched.iterrows():
        match_rows.append(
            {
                "Retention": f"{retention:.0f}%",
                "L2 − random RACR (pp)": f"{(row['racr_example', 'l2'] - row['racr_example', 'random']) * 100:+.1f}",
                "L2 − random Top-1 flips (pp)": f"{(row['top1_flip', 'l2'] - row['top1_flip', 'random']) * 100:+.1f}",
                "L2 − random accuracy (pp)": f"{row['accuracy_delta_pp', 'l2'] - row['accuracy_delta_pp', 'random']:+.1f}",
            }
        )
    matched_display = pd.DataFrame(match_rows)
    return (
        f"""# Experiment 1: Results, Figures, and Table Guidance

## Experimental scope

DeepSeek-VL2-Tiny was evaluated on {int(baseline.examples):,} paired ScienceQA examples. The unpruned baseline accuracy was **{baseline.accuracy * 100:.1f}%**. Each pruning condition retained the indicated proportion of visual tokens. Random retention has one reported seed (`seed=0`), so it is a matched control rather than a multi-seed estimate.

## Recommended main-paper table

Use the following as the main quantitative table. Accuracy changes are paired with the unpruned prediction for the same examples; intervals are 95% bootstrap confidence intervals.

| {" | ".join(display.columns)} |
|{"|".join(["---"] * len(display.columns))}|
"""
        + "\n".join(
            "| " + " | ".join(map(str, row)) + " |" for row in display.to_numpy()
        )
        + f"""

## Matched L2-versus-random comparison

Negative values in the first two columns mean L2 retention changes routing less than random retention at the same budget.

| {" | ".join(matched_display.columns)} |
|{"|".join(["---"] * len(matched_display.columns))}|
"""
        + "\n".join(
            "| " + " | ".join(map(str, row)) + " |"
            for row in matched_display.to_numpy()
        )
        + """

## Figures to include

1. **Figure 1 — Accuracy versus token retention.** Shows the task-performance cost as progressively fewer visual tokens are retained. Error bars are paired 95% bootstrap intervals for the accuracy change.
2. **Figure 2 — Routing disruption versus token retention.** Shows that the routing identities of retained tokens change substantially after pruning. RACR measures the fraction of baseline Top-k expert assignments not preserved; Top-1 flips isolate changes to the highest-ranked expert.
3. **Figure 3 — Matched L2 versus random pruning.** Shows the same-budget comparison. L2 consistently causes less routing reassignment than random retention, but the current results do not establish that this routing preservation causes the observed accuracy differences.

## Paper-ready interpretation

> Visual-token pruning substantially changes downstream MoE routing even among retained tokens. At matched token budgets, L2-based retention consistently yields lower routing assignment change and fewer Top-1 expert flips than random retention. For example, at 50% retention, L2 retention produces a RACR of 28.5% compared with 33.5% for random retention, while changing the dominant expert for 40.8% and 48.2% of retained token-layer units, respectively. Accuracy declines relative to the 71.6% unpruned baseline for both methods. These observations demonstrate co-occurring pruning, routing, and accuracy changes; they do not establish that routing disruption causes the accuracy loss.

## Definitions for the paper

- **Retention:** fraction of original visual tokens kept after pruning.
- **Accuracy change:** pruned accuracy minus the paired unpruned accuracy, in percentage points.
- **RACR (Routing Assignment Change Rate):** fraction of baseline Top-k expert assignments for retained token-layer units that is not preserved after pruning. Zero means the same Top-k assignments; higher values mean more reassignment.
- **Top-1 flips:** fraction of retained token-layer units whose highest-ranked expert differs from the unpruned run.
- **Top-k changed:** fraction of units whose complete Top-k expert set differs from the unpruned run.
- **Load-CV change:** change in the coefficient of variation of aggregate expert assignment counts. Negative values indicate less uneven aggregate expert load than the baseline.

## Scope note

This is a single-model, static-image QA experiment. Random pruning is currently represented by one seed, so claims about L2 versus random pruning should be described as descriptive results in this evaluated setting, not as a general result across random draws or models.
"""
    )


def main() -> None:
    args = parse_args()
    data = pd.read_csv(args.input)
    required = {"method", "accuracy", "actual_retention", "racr_example", "top1_flip"}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    baseline_rows = data[data["method"] == "baseline"]
    if len(baseline_rows) != 1:
        raise ValueError("Expected exactly one baseline row.")
    baseline = baseline_rows.iloc[0]
    rows = condition_rows(data)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_accuracy(rows, baseline, args.output_dir, args.dpi)
    plot_routing(rows, args.output_dir, args.dpi)
    plot_matched_comparison(rows, args.output_dir, args.dpi)
    report = args.output_dir / "EXPERIMENT_1_FIGURES_AND_TABLES.md"
    report.write_text(markdown_tables(rows, baseline), encoding="utf-8")
    print(f"Saved figures and report to {args.output_dir}")


if __name__ == "__main__":
    main()
