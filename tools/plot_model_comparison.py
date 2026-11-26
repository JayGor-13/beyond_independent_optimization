"""Plot measured model comparisons; add models with --extra-trc 'Name=summary.csv'."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = [
    "duplicate",
    "chronological",
    "shuffled",
    "cross_clip",
    "router_permutation",
]
LABELS = ["Duplicate", "Chronological", "Shuffled", "Cross-clip", "Router\npermutation"]
EXP1_CONDITIONS = ["baseline", "l2_r50", "random_r50_s0"]
COLORS = ["#3984CB", "#39A66B", "#D55E00", "#9467BD", "#CC79A7", "#666666"]
METRICS = {"trc8": ("TRC@8", "-", "o"), "weighted_trc": ("Weighted TRC", "--", "s")}


def extra_tables(table, specifications):
    for spec in specifications:
        label, separator, path = spec.partition("=")
        if not separator or not label.strip():
            raise ValueError("Extra models must use 'Model name=path/to/summary.csv'.")
        extra = pd.read_csv(path)
        extra["model_label"] = label.strip()
        table = pd.concat([table, extra], ignore_index=True)
    return table


def validate_trc(data):
    required = {
        "model_label",
        "condition",
        "metric",
        "mean",
        "ci95_low",
        "ci95_high",
        "clips",
    }
    if not required.issubset(data):
        raise ValueError(f"Missing TRC columns: {required - set(data.columns)}")
    selected = data[data["metric"].isin(METRICS)].copy()
    if selected.empty:
        raise ValueError("No measured TRC@8/weighted TRC rows.")
    for label, group in selected.groupby("model_label", sort=False):
        if len(group) != 10 or group.duplicated(["condition", "metric"]).any():
            raise ValueError(
                f"{label}: expected five conditions for each of the two metrics."
            )
        for metric in METRICS:
            if set(group[group.metric == metric].condition) != set(CONDITIONS):
                raise ValueError(f"{label}: missing a sequence control.")
        numeric = group[["mean", "ci95_low", "ci95_high"]].to_numpy(float)
        if not np.isfinite(numeric).all() or (numeric < 0).any() or (numeric > 1).any():
            raise ValueError(
                f"{label}: TRC means/intervals must be finite and in [0,1]."
            )
        if (group.ci95_low > group["mean"]).any() or (
            group["mean"] > group.ci95_high
        ).any():
            raise ValueError(f"{label}: confidence interval does not contain its mean.")
        if group.clips.nunique() != 1 or float(group.clips.iloc[0]) < 2:
            raise ValueError(f"{label}: inconsistent or insufficient sample counts.")
    return selected


def save(fig, directory, stem):
    for extension in ("pdf", "svg", "png"):
        fig.savefig(directory / f"{stem}.{extension}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def axes_style(ax):
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#DADDE2", linestyle=(0, (3, 3)), linewidth=0.8)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#DADDE2")
    ax.tick_params(length=0, pad=8)


def plot_trc(data, output, metrics, stem):
    models = list(data.model_label.unique())
    if len(models) > len(COLORS):
        raise ValueError(
            "At most six model colors are supported; extend COLORS for more."
        )
    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    fig.subplots_adjust(bottom=0.29, top=0.9)
    for index, model in enumerate(models):
        for metric in metrics:
            group = (
                data[(data.model_label == model) & (data.metric == metric)]
                .set_index("condition")
                .loc[CONDITIONS]
            )
            mean = group["mean"].to_numpy(float)
            errors = np.vstack(
                [
                    mean - group.ci95_low.to_numpy(float),
                    group.ci95_high.to_numpy(float) - mean,
                ]
            )
            _, style, marker = METRICS[metric]
            ax.errorbar(
                np.arange(5),
                mean,
                yerr=errors,
                color=COLORS[index],
                linestyle=style,
                marker=marker,
                markersize=6,
                linewidth=2,
                capsize=3,
                markeredgecolor="white",
                markeredgewidth=0.8,
            )
    axes_style(ax)
    ax.set_xticks(range(5), LABELS)
    ax.set_ylim(0, 1.02)
    ax.set_yticks(np.linspace(0, 1, 5))
    ax.set_ylabel("Routing consistency")
    ax.set_title(
        "Temporal routing consistency under sequence controls", loc="left", pad=16
    )
    handles = [
        Line2D([], [], color=COLORS[i], linewidth=2.5, label=name)
        for i, name in enumerate(models)
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.105),
        ncol=min(3, len(models)),
        frameon=False,
        fontsize=10,
    )
    if len(metrics) > 1:
        metric_handles = [
            Line2D(
                [],
                [],
                color="#444444",
                linestyle=METRICS[m][1],
                marker=METRICS[m][2],
                label=METRICS[m][0],
            )
            for m in metrics
        ]
        fig.legend(
            handles=metric_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.055),
            ncol=2,
            frameon=False,
            fontsize=10,
        )
    else:
        ax.set_ylabel(METRICS[metrics[0]][0])
    counts = [
        f"{name}: n={int(data[data.model_label == name].clips.iloc[0])}"
        for name in models
    ]
    fig.text(
        0.5,
        0.012,
        "Means and 95% bootstrap CIs. " + "; ".join(counts),
        ha="center",
        fontsize=8,
        color="#555555",
    )
    save(fig, output, stem)


def exp1_outputs(data, output, table_output):
    """Show paired accuracy changes and RACR across the measured retention sweep."""
    required = {
        "model_label",
        "condition",
        "examples",
        "accuracy",
        "racr_example",
        "top1_flip",
        "target_retention",
        "method",
        "accuracy_delta_ci95_low",
        "accuracy_delta_ci95_high",
        "racr_example_ci95_low",
        "racr_example_ci95_high",
    }
    if not required.issubset(data):
        raise ValueError(
            f"Missing Experiment 1 columns: {required - set(data.columns)}"
        )
    models = list(data.model_label.unique())
    if len(models) > len(COLORS):
        raise ValueError("Extend COLORS for additional models.")
    data = data.copy()
    for model in models:
        group = data[data.model_label == model]
        baseline = group[group.condition == "baseline"]
        if (
            len(baseline) != 1
            or group.condition.duplicated().any()
            or group.examples.nunique() != 1
        ):
            raise ValueError(
                f"{model}: require one baseline, unique conditions and matched counts."
            )
        for column in ("accuracy", "racr_example", "top1_flip", "target_retention"):
            values = group[column].to_numpy(float)
            if (
                not np.isfinite(values).all()
                or (values < 0).any()
                or (values > 1).any()
            ):
                raise ValueError(f"{model}: invalid {column} values.")
        reference = float(baseline.accuracy.iloc[0])
        if reference <= 0:
            raise ValueError(f"{model}: baseline accuracy must be positive.")
        data.loc[group.index, "accuracy_change_percent"] = 100 * (
            group.accuracy / reference - 1
        )
        for bound in ("low", "high"):
            data.loc[group.index, f"accuracy_change_ci95_{bound}"] = (
                100 * group[f"accuracy_delta_ci95_{bound}"] / reference
            )

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.1))
    fig.subplots_adjust(bottom=0.34, top=0.87, wspace=0.28)
    plotted = []
    accuracy_lows = [0.0]
    accuracy_highs = [0.0]
    racr_highs = [0.0]
    for i, model in enumerate(models):
        group = data[data.model_label == model]
        for method, style, marker in (("l2", "-", "o"), ("random", "--", "s")):
            series = group[group.method == method]
            if series.empty:
                raise ValueError(f"{model}: missing {method} results.")
            # Use measured identity controls where available; otherwise anchor at baseline.
            if not np.isclose(series.target_retention, 1.0).any():
                series = pd.concat([series, group[group.method == "baseline"]])
            series = series.sort_values("target_retention")
            if series.target_retention.duplicated().any():
                raise ValueError(
                    f"{model}: plot one {method} replicate per retention budget."
                )
            label = f"{model}: {'L2' if method == 'l2' else 'Random'}"
            accuracy = series.accuracy_change_percent.to_numpy(float)
            low = series.accuracy_change_ci95_low.to_numpy(float)
            high = series.accuracy_change_ci95_high.to_numpy(float)
            racr = 100 * series.racr_example.to_numpy(float)
            racr_low = 100 * series.racr_example_ci95_low.to_numpy(float)
            racr_high = 100 * series.racr_example_ci95_high.to_numpy(float)
            intervals = np.column_stack(
                [accuracy, low, high, racr, racr_low, racr_high]
            )
            if (
                not np.isfinite(intervals).all()
                or np.any(low > accuracy)
                or np.any(high < accuracy)
                or np.any(racr_low > racr)
                or np.any(racr_high < racr)
                or np.any(racr_low < 0)
                or np.any(racr_high > 100)
            ):
                raise ValueError(f"{model}: invalid {method} confidence intervals.")
            for ax, mean, lower, upper in (
                (axes[0], accuracy, low, high),
                (axes[1], racr, racr_low, racr_high),
            ):
                ax.errorbar(
                    100 * series.target_retention,
                    mean,
                    yerr=np.vstack([mean - lower, upper - mean]),
                    color=COLORS[i],
                    linestyle=style,
                    marker=marker,
                    markersize=5,
                    linewidth=1.8,
                    elinewidth=0.8,
                    capsize=2,
                    label=label,
                )
            accuracy_lows.extend(low)
            accuracy_highs.extend(high)
            racr_highs.extend(racr_high)
            points = series.copy()
            points["plotted_method"] = method
            plotted.append(points)
    axes[0].set_title("(a) Accuracy change", loc="left", fontsize=12, pad=12)
    axes[1].set_title("(b) Routing assignment change", loc="left", fontsize=12, pad=12)
    axes[0].set_ylabel("Accuracy change from own baseline (%)")
    axes[1].set_ylabel("RACR (%)")
    axes[0].set_ylim(min(accuracy_lows) - 1.5, max(accuracy_highs) + 1.5)
    axes[1].set_ylim(-1, min(100, max(racr_highs) + 3))
    axes[0].axhline(0, color="#444444", linewidth=0.8)
    for ax in axes:
        axes_style(ax)
        ax.set_xlabel("Visual tokens retained (%)")
        ax.set_xticks([25, 50, 70, 75, 100])
        ax.tick_params(axis="x", labelsize=10)
        # Keep adjacent 70% and 75% labels separate at manuscript print width.
        ax.get_xticklabels()[2].set_horizontalalignment("right")
        ax.get_xticklabels()[3].set_horizontalalignment("left")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.105),
        ncol=2,
        frameon=False,
        fontsize=9,
    )
    fig.text(
        0.5,
        0.035,
        "Error bars: 95% paired bootstrap CIs. Lines guide the eye between measured points.\n"
        "Samples, precision and preprocessing differ across models.",
        ha="center",
        fontsize=8,
        color="#555555",
    )
    save(fig, output, "exp1_model_comparison")
    pd.concat(plotted, ignore_index=True).to_csv(
        output / "exp1_models_plotted.csv", index=False
    )

    rows, latex = [], []
    for model in models:
        group = data[data.model_label == model].sort_values(
            ["target_retention", "method"]
        )
        group = pd.concat(
            [group[group.method == "baseline"], group[group.method != "baseline"]]
        )
        for index, (_, row) in enumerate(group.iterrows()):
            baseline = row.method == "baseline"
            label = "Baseline" if baseline else "L2" if row.method == "l2" else "Random"
            racr = "--" if baseline else f"{100 * row.racr_example:.2f}"
            flip = "--" if baseline else f"{100 * row.top1_flip:.2f}"
            values = [
                model if index == 0 else "",
                f"{100 * row.target_retention:.0f}%",
                label,
                f"{100 * row.accuracy:.2f}",
                f"{row.accuracy_change_percent:.2f}",
                racr,
                flip,
            ]
            rows.append(values)
            latex.append(" & ".join(v.replace("%", r"\%") for v in values) + r" \\")
        latex.append(r"\addlinespace")
    table_output.mkdir(parents=True, exist_ok=True)
    (table_output / "exp1_comparison_table.tex").write_text(
        "\n".join(
            [
                r"\begin{tabular}{lclrrrr}",
                r"\toprule",
                r"Model & Retention & Method & Accuracy (\%) & Change (\%) & RACR (\%) & Top-1 change (\%) \\",
                r"\midrule",
                *latex,
                r"\bottomrule",
                r"\end{tabular}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    data.to_csv(table_output / "exp1_paper_values.csv", index=False)
    fig, ax = plt.subplots(figsize=(14, 0.38 * len(rows) + 1.1))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=[
            "Model",
            "Retention",
            "Method",
            "Accuracy (%)",
            "Change (%)",
            "RACR (%)",
            "Top-1 change (%)",
        ],
        colWidths=[0.25, 0.08, 0.1, 0.14, 0.13, 0.12, 0.18],
        loc="center",
        cellLoc="left",
        colLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.7)
    for (row, _), cell in table.get_celld().items():
        cell.visible_edges = "B"
        cell.set_edgecolor("#DFE3E7")
        cell.set_linewidth(0.6)
        if row == 0:
            cell.set_text_props(weight="bold")
    save(fig, output, "exp1_comparison_table")


def exp2_table(data, data_dir):
    deep = pd.read_csv(
        ROOT / "results/published/deepseek/temporal_routing/trc_per_clip.csv"
    )
    intern = pd.read_csv(data_dir / "internvl_condition_accuracy.csv")
    rows, latex = [], []
    for model in data.model_label.unique():
        for i, condition in enumerate(CONDITIONS):
            group = data[
                (data.model_label == model) & (data.condition == condition)
            ].set_index("metric")
            accuracy, status = np.nan, "not_measured"
            if model == "DeepSeek-VL2-Tiny":
                correct = deep[deep.condition == condition].correct
                if correct.notna().all():
                    if (
                        len(correct) != int(group.clips.iloc[0])
                        or not correct.astype(str)
                        .str.lower()
                        .isin(["true", "false"])
                        .all()
                    ):
                        raise ValueError(
                            "Incomplete or invalid DeepSeek correctness records"
                        )
                    accuracy = 100 * correct.astype(str).str.lower().eq("true").mean()
                    status = "measured"
            else:
                measured = intern[
                    (intern.model_label == model) & (intern.condition == condition)
                ]
                if len(measured) == 1:
                    accuracy = 100 * float(measured.accuracy.iloc[0])
                    status = "measured"
            if condition == "router_permutation":
                status = "routing_only"
            shown = (
                f"{accuracy:.2f}"
                if np.isfinite(accuracy)
                else r"\textit{n/a}"
                if status == "routing_only"
                else r"\textit{n/m}"
            )
            name = [
                "Duplicate",
                "Chronological",
                "Shuffled",
                "Cross-clip",
                "Router permutation",
            ][i]
            latex.append(
                f"{model if i == 0 else ''} & {name} & {group.loc['trc8', 'mean']:.4f} & {group.loc['weighted_trc', 'mean']:.4f} & {shown} "
                + r"\\"
            )
            rows.append(
                dict(
                    model_label=model,
                    condition=condition,
                    clips=int(group.clips.iloc[0]),
                    trc8=group.loc["trc8", "mean"],
                    weighted_trc=group.loc["weighted_trc", "mean"],
                    accuracy_percent=accuracy,
                    accuracy_status=status,
                )
            )
        latex.append(r"\addlinespace")
    (data_dir / "exp2_comparison_table.tex").write_text(
        "\n".join(
            [
                r"\begin{tabular}{llrrr}",
                r"\toprule",
                r"Model & Condition & TRC@8 & Weighted TRC & Accuracy (\%) \\",
                r"\midrule",
                *latex,
                r"\bottomrule",
                r"\end{tabular}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    pd.DataFrame(rows).to_csv(data_dir / "exp2_paper_values.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "results/paper_tables")
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "artifacts/reproduction/figures"
    )
    parser.add_argument(
        "--extra-trc",
        action="append",
        default=[],
        help="Measured model summary: 'Name=path.csv'; repeatable.",
    )
    parser.add_argument(
        "--extra-exp1",
        action="append",
        default=[],
        help="Measured pruning summary: 'Name=path.csv'; repeatable.",
    )
    args = parser.parse_args()
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    trc = validate_trc(
        extra_tables(pd.read_csv(args.data_dir / "trc_comparison.csv"), args.extra_trc)
    )
    plot_trc(trc, args.output_dir, list(METRICS), "trc_models_combined")
    plot_trc(trc, args.output_dir, ["trc8"], "trc8_models")
    plot_trc(trc, args.output_dir, ["weighted_trc"], "weighted_trc_models")
    exp1 = extra_tables(
        pd.read_csv(args.data_dir / "exp1_comparison.csv"), args.extra_exp1
    )
    exp1_outputs(exp1, args.output_dir, args.data_dir)
    exp2_table(trc, args.data_dir)
    trc.to_csv(args.output_dir / "trc_models_plotted.csv", index=False)
    print(
        f"Saved PDF, SVG and PNG comparisons for {trc.model_label.nunique()} measured models to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
