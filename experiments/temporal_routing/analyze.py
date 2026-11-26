import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def default_output_dir():
    kaggle_working = Path("/kaggle/working")
    if kaggle_working.exists():
        return kaggle_working / "trc_outputs"
    return Path(__file__).resolve().parent / "outputs"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=default_output_dir())
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--permutations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260819)
    return parser.parse_args()


def jaccard(left, right):
    union = np.logical_or(left, right).sum()
    return float(np.logical_and(left, right).sum() / union) if union else 1.0


def top_m_mask(values, m):
    m = min(m, values.size)
    mask = np.zeros(values.size, dtype=bool)
    mask[np.argpartition(values, -m)[-m:]] = True
    return mask


def layer_metrics(mass, counts):
    frames, layers, experts = mass.shape
    rows = []
    for layer in range(layers):
        union_values = []
        trc4_values = []
        trc8_values = []
        trc16_values = []
        weighted_values = []
        for frame in range(frames - 1):
            left_mass = mass[frame, layer]
            right_mass = mass[frame + 1, layer]
            union_values.append(jaccard(counts[frame, layer] > 0, counts[frame + 1, layer] > 0))
            trc4_values.append(jaccard(top_m_mask(left_mass, 4), top_m_mask(right_mass, 4)))
            trc8_values.append(jaccard(top_m_mask(left_mass, 8), top_m_mask(right_mass, 8)))
            trc16_values.append(jaccard(top_m_mask(left_mass, 16), top_m_mask(right_mass, 16)))
            denominator = np.maximum(left_mass, right_mass).sum()
            weighted_values.append(float(np.minimum(left_mass, right_mass).sum() / denominator))

        positive = np.where(mass[:, layer] > 0, mass[:, layer], 1.0)
        entropy = -(mass[:, layer] * np.log(positive)).sum(axis=1) / np.log(experts)
        aggregate = mass[:, layer].sum(axis=0)
        load_cv = aggregate.std() / aggregate.mean()
        rows.append(
            {
                "union_trc": np.mean(union_values),
                "trc4": np.mean(trc4_values),
                "trc8": np.mean(trc8_values),
                "trc16": np.mean(trc16_values),
                "weighted_trc": np.mean(weighted_values),
                "router_entropy": entropy.mean(),
                "load_cv": load_cv,
            }
        )
    return rows


def permute_expert_identity(mass, counts, rng):
    permuted_mass = np.empty_like(mass)
    permuted_counts = np.empty_like(counts)
    for frame in range(mass.shape[0]):
        for layer in range(mass.shape[1]):
            permutation = rng.permutation(mass.shape[2])
            permuted_mass[frame, layer, permutation] = mass[frame, layer]
            permuted_counts[frame, layer, permutation] = counts[frame, layer]
    return permuted_mass, permuted_counts


def bootstrap_interval(values, samples, rng):
    values = np.asarray(values, dtype=float)
    means = np.empty(samples)
    for index in range(samples):
        means[index] = rng.choice(values, size=len(values), replace=True).mean()
    return np.percentile(means, [2.5, 97.5])


def bootstrap_difference(values, correct, samples, rng):
    values = np.asarray(values, dtype=float)
    correct = np.asarray(correct, dtype=bool)
    right = values[correct]
    wrong = values[~correct]
    if len(right) == 0 or len(wrong) == 0:
        return np.nan, np.nan
    differences = np.empty(samples)
    for index in range(samples):
        right_sample = rng.choice(right, size=len(right), replace=True)
        wrong_sample = rng.choice(wrong, size=len(wrong), replace=True)
        differences[index] = right_sample.mean() - wrong_sample.mean()
    return np.percentile(differences, [2.5, 97.5])


def prediction_map(prediction_dir):
    return {
        path.stem: json.loads(path.read_text(encoding="utf-8"))
        for path in prediction_dir.glob("*.json")
    }


def collect_rows(output_dir, seed, permutation_count):
    predictions = prediction_map(output_dir / "predictions")
    rows = []
    for path in sorted((output_dir / "routes").glob("*.npz")):
        item_id, condition = path.stem.split("__", 1)
        route = np.load(path)
        mass = route["mass"]
        counts = route["counts"]
        names = route["layer_names"].astype(str)
        prediction = predictions[item_id]
        for layer, metrics in enumerate(layer_metrics(mass, counts)):
            rows.append(
                {
                    "id": item_id,
                    "video_id": prediction["video_id"],
                    "question_type": prediction["type"],
                    "category": prediction["category"],
                    "condition": condition,
                    "layer": layer,
                    "layer_name": names[layer],
                    "correct": prediction["correct"] if condition == "chronological" else np.nan,
                    **metrics,
                }
            )

        if condition == "chronological":
            rng = np.random.default_rng(seed + prediction["dataset_index"])
            repeated = [[] for _ in range(mass.shape[1])]
            for _ in range(permutation_count):
                permuted_mass, permuted_counts = permute_expert_identity(mass, counts, rng)
                for layer, metrics in enumerate(layer_metrics(permuted_mass, permuted_counts)):
                    repeated[layer].append(metrics)
            for layer, layer_repeats in enumerate(repeated):
                metrics = {
                    key: np.mean([repeat[key] for repeat in layer_repeats])
                    for key in layer_repeats[0]
                }
                rows.append(
                    {
                        "id": item_id,
                        "video_id": prediction["video_id"],
                        "question_type": prediction["type"],
                        "category": prediction["category"],
                        "condition": "router_permutation",
                        "layer": layer,
                        "layer_name": names[layer],
                        "correct": np.nan,
                        **metrics,
                    }
                )
    return pd.DataFrame(rows), predictions


def summarize(clip_rows, samples, rng):
    metric_names = ["union_trc", "trc4", "trc8", "trc16", "weighted_trc", "router_entropy", "load_cv"]
    rows = []
    for condition, group in clip_rows.groupby("condition"):
        for metric in metric_names:
            values = group[metric].to_numpy()
            low, high = bootstrap_interval(values, samples, rng)
            rows.append(
                {
                    "condition": condition,
                    "clips": len(values),
                    "metric": metric,
                    "mean": values.mean(),
                    "median": np.median(values),
                    "q1": np.percentile(values, 25),
                    "q3": np.percentile(values, 75),
                    "ci95_low": low,
                    "ci95_high": high,
                }
            )
    return pd.DataFrame(rows)


def control_differences(clip_rows, samples, rng):
    metrics = ["union_trc", "trc4", "trc8", "trc16", "weighted_trc"]
    rows = []
    for metric in metrics:
        values = clip_rows.pivot(index="id", columns="condition", values=metric)
        for control in ["duplicate", "shuffled", "cross_clip", "router_permutation"]:
            difference = (values["chronological"] - values[control]).dropna().to_numpy()
            low, high = bootstrap_interval(difference, samples, rng)
            rows.append(
                {
                    "comparison": f"chronological_minus_{control}",
                    "metric": metric,
                    "clips": len(difference),
                    "mean_difference": difference.mean(),
                    "median_difference": np.median(difference),
                    "ci95_low": low,
                    "ci95_high": high,
                }
            )
    return pd.DataFrame(rows)


def accuracy_relation(per_layer, clip_rows, samples, rng):
    metrics = ["trc4", "trc8", "trc16", "weighted_trc", "union_trc"]
    rows = []
    groups = [("all", clip_rows[clip_rows["condition"] == "chronological"])]
    chronological = per_layer[per_layer["condition"] == "chronological"]
    groups.extend((str(layer), group) for layer, group in chronological.groupby("layer"))

    for layer, group in groups:
        correct = group["correct"].astype(bool).to_numpy()
        for metric in metrics:
            values = group[metric].to_numpy()
            right = values[correct]
            wrong = values[~correct]
            correlation = np.corrcoef(values, correct.astype(float))[0, 1] if len(right) and len(wrong) else np.nan
            low, high = bootstrap_difference(values, correct, samples, rng)
            rows.append(
                {
                    "layer": layer,
                    "metric": metric,
                    "examples": len(values),
                    "correct_examples": len(right),
                    "wrong_examples": len(wrong),
                    "mean_correct": right.mean() if len(right) else np.nan,
                    "mean_wrong": wrong.mean() if len(wrong) else np.nan,
                    "correct_minus_wrong": right.mean() - wrong.mean() if len(right) and len(wrong) else np.nan,
                    "point_biserial_r": correlation,
                    "difference_ci95_low": low,
                    "difference_ci95_high": high,
                }
            )
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    summary_dir = args.output_dir / "summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    per_layer, predictions = collect_rows(args.output_dir, args.seed, args.permutations)
    if per_layer.empty:
        raise RuntimeError(f"No routes found under {args.output_dir / 'routes'}")

    metric_names = ["union_trc", "trc4", "trc8", "trc16", "weighted_trc", "router_entropy", "load_cv"]
    clip_rows = (
        per_layer.groupby(["id", "video_id", "question_type", "category", "condition"], as_index=False)[metric_names]
        .mean()
    )
    correctness = {item_id: value["correct"] for item_id, value in predictions.items()}
    clip_rows["correct"] = clip_rows["id"].map(correctness).where(
        clip_rows["condition"] == "chronological"
    )

    summary = summarize(clip_rows, args.bootstrap, rng)
    differences = control_differences(clip_rows, args.bootstrap, rng)
    relation = accuracy_relation(per_layer, clip_rows, args.bootstrap, rng)

    per_layer.to_csv(summary_dir / "trc_per_clip_layer.csv", index=False)
    clip_rows.to_csv(summary_dir / "trc_per_clip.csv", index=False)
    summary.to_csv(summary_dir / "summary.csv", index=False)
    differences.to_csv(summary_dir / "control_differences.csv", index=False)
    relation.to_csv(summary_dir / "accuracy_relation.csv", index=False)

    accuracy = np.mean([value["correct"] for value in predictions.values()])
    print(f"Video-QA accuracy: {accuracy:.3f} ({len(predictions)} questions)")
    print("\nMean TRC by condition:")
    display = summary[summary["metric"].isin(["trc8", "weighted_trc"])][["condition", "metric", "mean", "ci95_low", "ci95_high"]]
    print(display.to_string(index=False))
    print("\nPaired chronological-control differences:")
    print(differences[differences["metric"].isin(["trc8", "weighted_trc"])].to_string(index=False))
    print("\nTRC-accuracy relationship, averaged across layers:")
    print(relation[(relation["layer"] == "all") & relation["metric"].isin(["trc8", "weighted_trc"])].to_string(index=False))
    print(f"\nSaved analysis to {summary_dir}")


if __name__ == "__main__":
    main()
