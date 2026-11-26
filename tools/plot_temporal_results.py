"""Create paper-ready plots from the Experiment 2 TRC result tables.

Run from the repository root or from any directory:

    python tools/plot_temporal_results.py

By default, the script reads ``experiment_2_TRC/Result`` and writes PNG and
PDF files to ``experiment_2_TRC/Result/plots``.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CONDITION_ORDER = [
    "duplicate",
    "chronological",
    "shuffled",
    "cross_clip",
    "router_permutation",
]

CONDITION_LABELS = {
    "duplicate": "Duplicate",
    "chronological": "Chronological",
    "shuffled": "Shuffled",
    "cross_clip": "Cross-clip",
    "router_permutation": "Router permutation",
}

# Colorblind-safe palette based on Okabe-Ito colors.
CONDITION_COLORS = {
    "duplicate": "#009E73",
    "chronological": "#0072B2",
    "shuffled": "#E69F00",
    "cross_clip": "#D55E00",
    "router_permutation": "#777777",
}

METRIC_LABELS = {
    "trc4": "TRC@4",
    "trc8": "TRC@8",
    "trc16": "TRC@16",
    "weighted_trc": "Weighted TRC",
}


def parse_args():
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=script_dir.parent / "results/published/deepseek/temporal_routing",
        help="Directory containing the Experiment 2 CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir.parent / "artifacts/temporal_plots",
        help="Plot destination. Defaults to RESULTS_DIR/plots.",
    )
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=("png", "pdf", "svg"),
        default=("png", "pdf"),
    )
    return parser.parse_args()


def configure_style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "grid.color": "#D9D9D9",
            "grid.linewidth": 0.7,
            "grid.alpha": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
        }
    )


def require_columns(table, columns, filename):
    missing = set(columns) - set(table.columns)
    if missing:
        raise ValueError(f"{filename} is missing required columns: {sorted(missing)}")


def read_summary(results_dir):
    candidates = [
        results_dir / "summary.csv",
        *sorted(results_dir.glob("summary*.csv")),
    ]
    seen = set()
    for path in candidates:
        if path in seen or not path.exists():
            continue
        seen.add(path)
        table = pd.read_csv(path)
        required = {"condition", "metric", "mean", "ci95_low", "ci95_high"}
        if required.issubset(table.columns):
            return table, path
    raise FileNotFoundError(f"Could not find a valid summary CSV under {results_dir}")


def save_figure(figure, output_dir, stem, formats, dpi):
    saved = []
    for extension in formats:
        path = output_dir / f"{stem}.{extension}"
        figure.savefig(path, dpi=dpi if extension == "png" else None)
        saved.append(path)
    plt.close(figure)
    return saved


def plot_condition_comparison(summary, output_dir, formats, dpi):
    # Keep this figure's publication style local so other plots are unaffected.
    with plt.rc_context(
        {
            "font.family": "STIXGeneral",
            "font.size": 10,
            "axes.labelsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 9,
            "legend.fontsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
        }
    ):
        figure, axis = plt.subplots(figsize=(7.0, 3.0))
        positions = np.arange(len(CONDITION_ORDER))
        width = 0.30
        metrics = [("trc8", "TRC@8", "0.40"), ("weighted_trc", "Weighted TRC", "0.85")]

        for index, (metric, label, color) in enumerate(metrics):
            metric_rows = summary[summary["metric"] == metric].set_index("condition")
            missing = set(CONDITION_ORDER) - set(metric_rows.index)
            if missing:
                raise ValueError(
                    f"Summary is missing {metric} rows for: {sorted(missing)}"
                )
            metric_rows = metric_rows.reindex(CONDITION_ORDER)
            means = metric_rows["mean"].to_numpy(dtype=float)
            low = metric_rows["ci95_low"].to_numpy(dtype=float)
            high = metric_rows["ci95_high"].to_numpy(dtype=float)
            x = positions + (index - 0.5) * (width + 0.04)
            axis.bar(
                x,
                means,
                width=width,
                label=label,
                color=color,
                edgecolor="0.15",
                linewidth=0.6,
                yerr=np.vstack([means - low, high - means]),
                error_kw={
                    "ecolor": "0.05",
                    "elinewidth": 0.8,
                    "capsize": 2.5,
                    "capthick": 0.8,
                },
                zorder=3,
            )
            for position, mean, upper in zip(x, means, high):
                axis.text(
                    position,
                    upper + 0.018,
                    f"{mean:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8.5,
                )

        labels = [
            CONDITION_LABELS[name].replace("Router permutation", "Router\npermutation")
            for name in CONDITION_ORDER
        ]
        axis.set_xticks(positions, labels)
        axis.set_ylabel("Mean routing consistency")
        axis.set_ylim(0, 1.0)
        axis.set_yticks(np.arange(0, 1.01, 0.2))
        axis.set_xlim(-0.6, len(CONDITION_ORDER) - 0.4)
        axis.grid(axis="y", color="0.90", linewidth=0.5)
        axis.set_axisbelow(True)
        axis.tick_params(axis="x", length=0, pad=7)
        axis.tick_params(axis="y", length=3, width=0.6)
        axis.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, 1.01),
            ncol=2,
            frameon=False,
            handlelength=1.3,
            columnspacing=2.2,
        )
        figure.tight_layout(pad=0.5)
        return save_figure(
            figure, output_dir, "fig1_condition_trc_comparison", formats, dpi
        )


def bootstrap_layer_profiles(per_layer, metric, samples, seed):
    rng = np.random.default_rng(seed)
    profiles = {}
    for condition in CONDITION_ORDER:
        condition_rows = per_layer[per_layer["condition"] == condition]
        pivot = condition_rows.pivot(
            index="id", columns="layer", values=metric
        ).sort_index(axis=1)
        if pivot.empty:
            raise ValueError(f"No layer-wise rows found for condition '{condition}'.")
        if pivot.isna().any().any():
            raise ValueError(f"Incomplete layer profile for condition '{condition}'.")
        values = pivot.to_numpy(dtype=float)
        means = values.mean(axis=0)
        bootstrap_means = np.empty((samples, values.shape[1]), dtype=float)
        for sample_index in range(samples):
            indices = rng.integers(0, values.shape[0], size=values.shape[0])
            bootstrap_means[sample_index] = values[indices].mean(axis=0)
        lower, upper = np.percentile(bootstrap_means, [2.5, 97.5], axis=0)
        profiles[condition] = {
            "layers": pivot.columns.to_numpy(dtype=int),
            "mean": means,
            "low": lower,
            "high": upper,
        }
    return profiles


def plot_layer_profiles(per_layer, output_dir, formats, dpi, bootstrap, seed):
    profiles = bootstrap_layer_profiles(per_layer, "trc8", bootstrap, seed)
    figure, axis = plt.subplots(figsize=(7.2, 4.8))

    for condition in CONDITION_ORDER:
        profile = profiles[condition]
        axis.plot(
            profile["layers"],
            profile["mean"],
            color=CONDITION_COLORS[condition],
            marker="o",
            markersize=3.8,
            linewidth=1.7 if condition == "chronological" else 1.35,
            label=CONDITION_LABELS[condition],
            zorder=3,
        )
        axis.fill_between(
            profile["layers"],
            profile["low"],
            profile["high"],
            color=CONDITION_COLORS[condition],
            alpha=0.09,
            linewidth=0,
        )

    all_layers = profiles["chronological"]["layers"]
    axis.set_xticks(all_layers)
    axis.set_xlim(all_layers.min() - 0.25, all_layers.max() + 0.25)
    axis.set_ylim(0.0, 0.93)
    axis.set_xlabel("Measured routed MoE layer")
    axis.set_ylabel("Mean TRC@8 (95% bootstrap CI)")
    figure.suptitle("Layer-wise temporal routing consistency", y=0.96, fontsize=12)
    axis.grid(axis="y")
    handles, labels = axis.get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        ncol=3,
        frameon=False,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.80))
    return save_figure(figure, output_dir, "fig2_layerwise_trc8", formats, dpi)


def plot_accuracy_effects(accuracy_relation, output_dir, formats, dpi):
    metric_order = ["trc4", "trc8", "trc16", "weighted_trc"]
    aggregate = accuracy_relation[
        accuracy_relation["layer"].astype(str) == "all"
    ].set_index("metric")
    missing = set(metric_order) - set(aggregate.index)
    if missing:
        raise ValueError(
            f"accuracy_relation.csv is missing aggregate rows for: {sorted(missing)}"
        )
    aggregate = aggregate.reindex(metric_order)

    differences = aggregate["correct_minus_wrong"].to_numpy(dtype=float)
    lower = aggregate["difference_ci95_low"].to_numpy(dtype=float)
    upper = aggregate["difference_ci95_high"].to_numpy(dtype=float)
    correlations = aggregate["point_biserial_r"].to_numpy(dtype=float)
    y_positions = np.arange(len(metric_order))

    figure, axis = plt.subplots(figsize=(6.5, 3.6))
    axis.axvline(0, color="#333333", linestyle="--", linewidth=1, zorder=1)
    axis.errorbar(
        differences,
        y_positions,
        xerr=np.vstack([differences - lower, upper - differences]),
        fmt="o",
        color="#0072B2",
        ecolor="#0072B2",
        markersize=6,
        markeredgecolor="white",
        markeredgewidth=0.6,
        elinewidth=1.5,
        capsize=3,
        zorder=3,
    )

    limit = max(0.02, float(np.max(np.abs(np.concatenate([lower, upper])))) * 1.35)
    axis.set_xlim(-limit, limit)
    text_x = limit * 0.96
    for y_position, correlation in zip(y_positions, correlations):
        axis.text(
            text_x,
            y_position,
            f"r = {correlation:+.3f}",
            ha="right",
            va="center",
            fontsize=8.5,
        )

    axis.set_yticks(y_positions, [METRIC_LABELS[name] for name in metric_order])
    axis.invert_yaxis()
    axis.set_xlabel("Mean(correct) − mean(incorrect), with 95% bootstrap CI")
    axis.set_title("TRC–answer correctness association")
    axis.grid(axis="x")
    axis.tick_params(axis="y", length=0)
    figure.tight_layout()
    return save_figure(figure, output_dir, "fig3_trc_accuracy_effects", formats, dpi)


def main():
    args = parse_args()
    if args.bootstrap < 1:
        raise ValueError("--bootstrap must be at least one.")
    results_dir = args.results_dir.resolve()
    output_dir = (args.output_dir or results_dir / "plots").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_style()

    summary, summary_path = read_summary(results_dir)
    per_layer_path = results_dir / "trc_per_clip_layer.csv"
    accuracy_path = results_dir / "accuracy_relation.csv"
    if not per_layer_path.exists() or not accuracy_path.exists():
        raise FileNotFoundError(
            f"Expected trc_per_clip_layer.csv and accuracy_relation.csv under {results_dir}"
        )
    per_layer = pd.read_csv(per_layer_path)
    accuracy_relation = pd.read_csv(accuracy_path)

    require_columns(
        per_layer,
        {"id", "condition", "layer", "trc8"},
        per_layer_path.name,
    )
    require_columns(
        accuracy_relation,
        {
            "layer",
            "metric",
            "correct_minus_wrong",
            "point_biserial_r",
            "difference_ci95_low",
            "difference_ci95_high",
        },
        accuracy_path.name,
    )

    saved = []
    saved.extend(plot_condition_comparison(summary, output_dir, args.formats, args.dpi))
    saved.extend(
        plot_layer_profiles(
            per_layer,
            output_dir,
            args.formats,
            args.dpi,
            args.bootstrap,
            args.seed,
        )
    )
    saved.extend(
        plot_accuracy_effects(accuracy_relation, output_dir, args.formats, args.dpi)
    )

    print(f"Read condition summary: {summary_path}")
    print("Saved figures:")
    for path in saved:
        print(f"  {path}")


if __name__ == "__main__":
    main()
