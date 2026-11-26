import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from core import layer_metrics


ROUTING_METRICS = ("trc8", "weighted_trc")
ALL_ROUTE_METRICS = (
    "union_trc",
    "trc4",
    "trc8",
    "trc16",
    "weighted_trc",
    "router_entropy",
    "load_cv",
)
PRIMARY_PERTURBATIONS = ("mild", "medium", "severe")
CATEGORY_ORDER = ("all", "temporal", "causal", "descriptive")


def default_output_dir():
    kaggle_working = Path("/kaggle/working")
    if kaggle_working.exists():
        return kaggle_working / "trc_accuracy_outputs"
    return Path(__file__).resolve().parent / "outputs"


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze paired TRC and NExT-QA performance changes.")
    parser.add_argument("--output-dir", type=Path, default=default_output_dir())
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--randomizations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def expected_stems(output_dir):
    manifest_path = output_dir / "manifests" / "nextqa_manifest.json"
    permutation_path = output_dir / "manifests" / "permutations.json"
    if not manifest_path.exists() or not permutation_path.exists():
        raise FileNotFoundError("Run manifest or permutation bank is missing. Run run.py first.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    permutations = json.loads(permutation_path.read_text(encoding="utf-8"))
    stems = {f"{item['id']}__{spec['key']}" for item in manifest for spec in permutations}
    return manifest, permutations, stems


def check_completeness(output_dir, allow_incomplete):
    manifest, permutations, expected = expected_stems(output_dir)
    predictions = {path.stem for path in (output_dir / "predictions").glob("*.json")}
    routes = {path.stem for path in (output_dir / "routes").glob("*.npz")}
    missing_predictions = sorted(expected - predictions)
    missing_routes = sorted(expected - routes)
    if (missing_predictions or missing_routes) and not allow_incomplete:
        preview = (missing_predictions + missing_routes)[:8]
        raise RuntimeError(
            f"Experiment is incomplete: {len(missing_predictions)} predictions and "
            f"{len(missing_routes)} route files are missing. Examples: {preview}. "
            "Rerun run.py to resume, or use --allow-incomplete for a smoke-test analysis."
        )
    return manifest, permutations, predictions & routes


def collect_rows(output_dir, complete_stems):
    per_layer_rows = []
    per_run_rows = []
    for stem in sorted(complete_stems):
        prediction_path = output_dir / "predictions" / f"{stem}.json"
        route_path = output_dir / "routes" / f"{stem}.npz"
        prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
        with np.load(route_path, allow_pickle=False) as route:
            mass = route["mass"]
            counts = route["counts"]
            layer_names = route["layer_names"]
        metrics = layer_metrics(mass, counts)
        base = {
            "id": prediction["id"],
            "video_id": str(prediction["video_id"]),
            "question_type": prediction["type"],
            "category": prediction["category"],
            "condition_key": prediction["condition_key"],
            "condition": prediction["condition"],
            "replicate": prediction["replicate"],
            "severity_rank": prediction["severity_rank"],
            "kendall_distance": prediction["kendall_distance"],
            "adjacency_retention": prediction["adjacency_retention"],
            "adjacency_destruction": prediction["adjacency_destruction"],
            "target": prediction["target"],
            "predicted": prediction["predicted"],
            "correct": bool(prediction["correct"]),
            "correct_option_margin": float(prediction["correct_option_margin"]),
        }
        for letter, value in prediction["choice_logprobs"].items():
            base[f"logprob_{letter}"] = float(value)
        for layer, values in enumerate(metrics):
            per_layer_rows.append(
                {
                    **base,
                    "layer": layer,
                    "layer_name": str(layer_names[layer]),
                    **values,
                }
            )
        aggregate = {
            metric: float(np.mean([values[metric] for values in metrics])) for metric in ALL_ROUTE_METRICS
        }
        per_run_rows.append({**base, **aggregate})
    if not per_run_rows:
        raise RuntimeError("No complete prediction/route pairs were found.")
    return pd.DataFrame(per_layer_rows), pd.DataFrame(per_run_rows)


def add_baseline_deltas(per_run):
    baseline = per_run[per_run["condition"] == "chronological"].copy()
    if baseline["id"].duplicated().any():
        raise RuntimeError("More than one chronological baseline was found for a question.")
    baseline_columns = {
        "trc8": "baseline_trc8",
        "weighted_trc": "baseline_weighted_trc",
        "correct_option_margin": "baseline_margin",
        "correct": "baseline_correct",
    }
    baseline = baseline[["id", *baseline_columns]].rename(columns=baseline_columns)
    merged = per_run.merge(baseline, on="id", how="left", validate="many_to_one")
    if merged[["baseline_trc8", "baseline_margin"]].isna().any().any():
        raise RuntimeError("At least one question is missing its chronological baseline.")
    merged["delta_trc8"] = merged["trc8"] - merged["baseline_trc8"]
    merged["delta_weighted_trc"] = merged["weighted_trc"] - merged["baseline_weighted_trc"]
    merged["delta_margin"] = merged["correct_option_margin"] - merged["baseline_margin"]
    merged["delta_correct"] = merged["correct"].astype(int) - merged["baseline_correct"].astype(int)
    return merged


def cluster_indices(frame, cluster_column="id"):
    values = frame[cluster_column].astype(str).to_numpy()
    identifiers = np.unique(values)
    return [np.flatnonzero(values == identifier) for identifier in identifiers]


def bootstrap_vectors(frame, statistic, samples, rng):
    observed = np.asarray(statistic(frame), dtype=float)
    if samples <= 0:
        return observed, np.full_like(observed, np.nan), np.full_like(observed, np.nan)
    clusters = cluster_indices(frame)
    bootstrapped = np.full((samples, observed.size), np.nan, dtype=float)
    for sample in range(samples):
        selected = rng.integers(0, len(clusters), size=len(clusters))
        row_indices = np.concatenate([clusters[index] for index in selected])
        bootstrapped[sample] = statistic(frame.iloc[row_indices])
    low = np.full(observed.size, np.nan)
    high = np.full(observed.size, np.nan)
    for index in range(observed.size):
        finite = bootstrapped[:, index][np.isfinite(bootstrapped[:, index])]
        if finite.size:
            low[index], high[index] = np.percentile(finite, [2.5, 97.5])
    return observed, low, high


def safe_correlation(left, right):
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if left.size < 3 or np.std(left) == 0 or np.std(right) == 0:
        return np.nan
    return float(np.corrcoef(left, right)[0, 1])


def association_statistics(frame, x_column, y_column):
    x = frame[x_column].to_numpy(dtype=float)
    y = frame[y_column].to_numpy(dtype=float)
    pearson = safe_correlation(x, y)
    spearman = float(spearmanr(x, y).statistic) if x.size >= 3 and np.std(x) and np.std(y) else np.nan
    variance = np.sum((x - x.mean()) ** 2)
    slope = float(np.sum((x - x.mean()) * (y - y.mean())) / variance) if variance else np.nan

    medium = (frame["condition"].to_numpy() == "medium").astype(float)
    severe = (frame["condition"].to_numpy() == "severe").astype(float)
    design = np.column_stack([np.ones(len(frame)), x, medium, severe])
    adjusted_slope = float(np.linalg.lstsq(design, y, rcond=None)[0][1]) if len(frame) >= 5 else np.nan
    return np.asarray([pearson, spearman, slope, adjusted_slope], dtype=float)


def within_question_randomization_p(frame, x_column, y_column, repetitions, rng):
    observed = safe_correlation(frame[x_column], frame[y_column])
    if not np.isfinite(observed) or repetitions <= 0:
        return np.nan
    x = frame[x_column].to_numpy(dtype=float)
    y = frame[y_column].to_numpy(dtype=float)
    clusters = cluster_indices(frame)
    exceedances = 0
    for _ in range(repetitions):
        randomized = y.copy()
        for indices in clusters:
            randomized[indices] = rng.permutation(randomized[indices])
        value = safe_correlation(x, randomized)
        if np.isfinite(value) and value >= observed:
            exceedances += 1
    return float((exceedances + 1) / (repetitions + 1))


def category_groups(frame):
    yield "all", frame
    for category in CATEGORY_ORDER[1:]:
        yield category, frame[frame["category"] == category]


def condition_summary(per_run, samples, rng):
    metric_columns = {
        "trc8": "trc8",
        "weighted_trc": "weighted_trc",
        "margin": "correct_option_margin",
        "accuracy": "correct",
        "router_entropy": "router_entropy",
        "load_cv": "load_cv",
    }
    rows = []
    for category, category_frame in category_groups(per_run):
        for condition, group in category_frame.groupby("condition", sort=False):
            for metric, column in metric_columns.items():
                statistic = lambda sampled, column=column: np.asarray([sampled[column].astype(float).mean()])
                observed, low, high = bootstrap_vectors(group, statistic, samples, rng)
                rows.append(
                    {
                        "category": category,
                        "condition": condition,
                        "questions": group["id"].nunique(),
                        "observations": len(group),
                        "metric": metric,
                        "mean": observed[0],
                        "ci95_low": low[0],
                        "ci95_high": high[0],
                    }
                )
    return pd.DataFrame(rows)


def paired_differences(with_deltas, samples, rng):
    delta_columns = {
        "trc8": "delta_trc8",
        "weighted_trc": "delta_weighted_trc",
        "margin": "delta_margin",
        "accuracy": "delta_correct",
    }
    perturbations = with_deltas[with_deltas["condition"] != "chronological"]
    rows = []
    for category, category_frame in category_groups(perturbations):
        for condition, group in category_frame.groupby("condition", sort=False):
            for metric, column in delta_columns.items():
                statistic = lambda sampled, column=column: np.asarray([sampled[column].astype(float).mean()])
                observed, low, high = bootstrap_vectors(group, statistic, samples, rng)
                rows.append(
                    {
                        "category": category,
                        "condition": condition,
                        "questions": group["id"].nunique(),
                        "observations": len(group),
                        "metric": metric,
                        "perturbed_minus_chronological": observed[0],
                        "ci95_low": low[0],
                        "ci95_high": high[0],
                    }
                )
    return pd.DataFrame(rows)


def association_table(with_deltas, samples, randomizations, rng):
    primary = with_deltas[with_deltas["condition"].isin(PRIMARY_PERTURBATIONS)].copy()
    rows = []
    for category, group in category_groups(primary):
        if group.empty:
            continue
        for routing_metric in ROUTING_METRICS:
            x_column = f"delta_{routing_metric}"
            for outcome, y_column in (("margin", "delta_margin"), ("correctness", "delta_correct")):
                statistic = lambda sampled, x=x_column, y=y_column: association_statistics(sampled, x, y)
                observed, low, high = bootstrap_vectors(group, statistic, samples, rng)
                randomization_p = within_question_randomization_p(
                    group, x_column, y_column, randomizations, rng
                )
                rows.append(
                    {
                        "category": category,
                        "routing_metric": routing_metric,
                        "performance_outcome": outcome,
                        "questions": group["id"].nunique(),
                        "observations": len(group),
                        "pearson_r": observed[0],
                        "pearson_ci95_low": low[0],
                        "pearson_ci95_high": high[0],
                        "pearson_within_question_randomization_p_one_sided": randomization_p,
                        "spearman_rho": observed[1],
                        "spearman_ci95_low": low[1],
                        "spearman_ci95_high": high[1],
                        "slope": observed[2],
                        "slope_ci95_low": low[2],
                        "slope_ci95_high": high[2],
                        "severity_adjusted_slope": observed[3],
                        "severity_adjusted_slope_ci95_low": low[3],
                        "severity_adjusted_slope_ci95_high": high[3],
                    }
                )
    return pd.DataFrame(rows)


def accuracy_flips(with_deltas, samples, rng):
    perturbed = with_deltas[
        (with_deltas["condition"] != "chronological") & with_deltas["baseline_correct"].astype(bool)
    ].copy()
    perturbed["flipped_to_wrong"] = ~perturbed["correct"].astype(bool)
    rows = []
    for category, category_frame in category_groups(perturbed):
        for condition, group in category_frame.groupby("condition", sort=False):
            statistic = lambda sampled: np.asarray([sampled["flipped_to_wrong"].astype(float).mean()])
            observed, low, high = bootstrap_vectors(group, statistic, samples, rng)
            rows.append(
                {
                    "category": category,
                    "condition": condition,
                    "baseline_correct_questions": group["id"].nunique(),
                    "observations": len(group),
                    "correct_to_wrong_rate": observed[0],
                    "ci95_low": low[0],
                    "ci95_high": high[0],
                }
            )
    return pd.DataFrame(rows)


def permutation_summary(per_run):
    columns = [
        "condition_key",
        "condition",
        "replicate",
        "severity_rank",
        "kendall_distance",
        "adjacency_retention",
    ]
    return (
        per_run.groupby(columns, dropna=False, as_index=False)
        .agg(
            questions=("id", "nunique"),
            mean_trc8=("trc8", "mean"),
            mean_weighted_trc=("weighted_trc", "mean"),
            mean_margin=("correct_option_margin", "mean"),
            accuracy=("correct", "mean"),
        )
        .sort_values(["severity_rank", "condition_key"])
    )


def make_figures(with_deltas, summary, figure_dir):
    figure_dir.mkdir(parents=True, exist_ok=True)
    temporal = with_deltas[
        (with_deltas["category"] == "temporal")
        & with_deltas["condition"].isin(PRIMARY_PERTURBATIONS)
    ]
    if len(temporal) >= 3:
        colors = {"mild": "#4C78A8", "medium": "#F58518", "severe": "#E45756"}
        fig, axis = plt.subplots(figsize=(6.5, 4.5))
        for condition in PRIMARY_PERTURBATIONS:
            group = temporal[temporal["condition"] == condition]
            axis.scatter(
                group["delta_trc8"],
                group["delta_margin"],
                s=18,
                alpha=0.55,
                color=colors[condition],
                label=condition.capitalize(),
            )
        x = temporal["delta_trc8"].to_numpy(dtype=float)
        y = temporal["delta_margin"].to_numpy(dtype=float)
        if np.std(x):
            coefficients = np.polyfit(x, y, 1)
            grid = np.linspace(x.min(), x.max(), 100)
            axis.plot(grid, coefficients[0] * grid + coefficients[1], color="black", linewidth=1.5)
        axis.axhline(0, color="0.75", linewidth=0.8)
        axis.axvline(0, color="0.75", linewidth=0.8)
        axis.set_xlabel("Change in TRC@8 from chronological baseline")
        axis.set_ylabel("Change in correct-option margin")
        axis.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(figure_dir / "temporal_delta_trc8_vs_margin.png", dpi=200)
        plt.close(fig)

    all_summary = summary[summary["category"] == "all"]
    order = [condition for condition in ("chronological", "mild", "medium", "severe", "reverse") if condition in set(all_summary["condition"])]
    metrics = (("trc8", "TRC@8"), ("margin", "Correct-option margin"), ("accuracy", "Accuracy"))
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for axis, (metric, label) in zip(axes, metrics):
        table = all_summary[all_summary["metric"] == metric].set_index("condition").reindex(order)
        values = table["mean"].to_numpy(dtype=float)
        errors = np.vstack(
            [values - table["ci95_low"].to_numpy(dtype=float), table["ci95_high"].to_numpy(dtype=float) - values]
        )
        axis.errorbar(range(len(order)), values, yerr=errors, marker="o", capsize=3, color="#4C78A8")
        axis.set_xticks(range(len(order)), [name.capitalize() for name in order], rotation=30, ha="right")
        axis.set_ylabel(label)
    fig.tight_layout()
    fig.savefig(figure_dir / "condition_trc_margin_accuracy.png", dpi=200)
    plt.close(fig)


def format_number(value, digits=4):
    return "NA" if value is None or not np.isfinite(value) else f"{value:.{digits}f}"


def write_results_summary(path, per_run, condition_table, associations):
    primary = associations[
        (associations["category"] == "temporal")
        & (associations["routing_metric"] == "trc8")
        & (associations["performance_outcome"] == "margin")
    ]
    lines = [
        "# Experiment 3 result summary",
        "",
        f"Complete question-condition observations: **{len(per_run)}** across **{per_run['id'].nunique()}** questions.",
        "",
        "## Primary within-question test",
        "",
    ]
    if primary.empty:
        lines.append("The primary temporal-question analysis could not be estimated from the available runs.")
    else:
        row = primary.iloc[0]
        supported = row["pearson_ci95_low"] > 0
        adjusted_supported = row["severity_adjusted_slope_ci95_low"] > 0
        lines.extend(
            [
                (
                    "For temporal questions, the association between change in TRC@8 and change in "
                    f"correct-option margin was r={format_number(row['pearson_r'])}, 95% cluster-bootstrap "
                    f"CI [{format_number(row['pearson_ci95_low'])}, {format_number(row['pearson_ci95_high'])}]."
                ),
                (
                    "The slope after controlling for perturbation severity was "
                    f"{format_number(row['severity_adjusted_slope'])}, 95% CI "
                    f"[{format_number(row['severity_adjusted_slope_ci95_low'])}, "
                    f"{format_number(row['severity_adjusted_slope_ci95_high'])}]."
                ),
                "",
            ]
        )
        if supported and adjusted_supported:
            lines.append(
                "The preregistered direction is supported: within-question TRC reductions track margin reductions, including after accounting for perturbation severity."
            )
        elif supported:
            lines.append(
                "A positive unadjusted within-question association is present, but it is not independently supported after controlling for perturbation severity."
            )
        else:
            lines.append(
                "The primary analysis does not support a reliable positive within-question TRC-margin association."
            )
    lines.extend(
        [
            "",
            "## Condition-level descriptive results",
            "",
            "| Condition | TRC@8 | Margin | Accuracy |",
            "|---|---:|---:|---:|",
        ]
    )
    all_rows = condition_table[condition_table["category"] == "all"]
    for condition in ("chronological", "mild", "medium", "severe", "reverse"):
        subset = all_rows[all_rows["condition"] == condition]
        if subset.empty:
            continue
        values = {row["metric"]: row["mean"] for _, row in subset.iterrows()}
        lines.append(
            f"| {condition.capitalize()} | {format_number(values.get('trc8'))} | "
            f"{format_number(values.get('margin'))} | {format_number(values.get('accuracy'))} |"
        )
    lines.extend(
        [
            "",
            "This experiment measures sampled frames presented as a multi-image sequence; it is not native-video inference. A positive association does not by itself establish that routing changes cause accuracy changes.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    if args.bootstrap < 0 or args.randomizations < 0:
        raise ValueError("Bootstrap and randomization counts must be non-negative.")
    summary_dir = args.output_dir / "summaries"
    figure_dir = args.output_dir / "figures"
    summary_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    manifest, permutations, complete_stems = check_completeness(args.output_dir, args.allow_incomplete)
    per_layer, per_run = collect_rows(args.output_dir, complete_stems)
    with_deltas = add_baseline_deltas(per_run)
    conditions = condition_summary(per_run, args.bootstrap, rng)
    differences = paired_differences(with_deltas, args.bootstrap, rng)
    associations = association_table(with_deltas, args.bootstrap, args.randomizations, rng)
    flips = accuracy_flips(with_deltas, args.bootstrap, rng)
    permutations_table = permutation_summary(per_run)

    per_layer.to_csv(summary_dir / "per_run_layer.csv", index=False)
    with_deltas.to_csv(summary_dir / "per_run.csv", index=False)
    conditions.to_csv(summary_dir / "condition_summary.csv", index=False)
    differences.to_csv(summary_dir / "paired_differences.csv", index=False)
    associations.to_csv(summary_dir / "within_question_association.csv", index=False)
    flips.to_csv(summary_dir / "accuracy_flips.csv", index=False)
    permutations_table.to_csv(summary_dir / "permutation_summary.csv", index=False)
    metadata = {
        "schema_version": 1,
        "manifest_questions": len(manifest),
        "permutations_per_question": len(permutations),
        "complete_observations": len(per_run),
        "complete_questions": int(per_run["id"].nunique()),
        "bootstrap_samples": args.bootstrap,
        "within_question_randomizations": args.randomizations,
        "primary_cohort": "temporal",
        "primary_routing_metric": "trc8",
        "primary_performance_outcome": "correct_option_margin",
        "reverse_in_primary_association": False,
    }
    (summary_dir / "analysis_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    make_figures(with_deltas, conditions, figure_dir)
    write_results_summary(summary_dir / "RESULTS_SUMMARY.md", per_run, conditions, associations)

    print(f"Analyzed {len(per_run)} observations from {per_run['id'].nunique()} questions.")
    primary = associations[
        (associations["category"] == "temporal")
        & (associations["routing_metric"] == "trc8")
        & (associations["performance_outcome"] == "margin")
    ]
    if not primary.empty:
        print("\nPrimary temporal-question association:")
        print(primary.to_string(index=False))
    print(f"\nSaved analysis to {summary_dir}")


if __name__ == "__main__":
    main()
