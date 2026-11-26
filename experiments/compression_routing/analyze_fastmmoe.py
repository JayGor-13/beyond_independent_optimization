import argparse
import csv
import json
from pathlib import Path

import numpy as np


METRICS = [
    "accuracy_delta",
    "actual_retention",
    "assignment_retention",
    "load_tv",
    "load_jsd",
    "load_cv_delta",
]


def default_output_dir():
    kaggle_working = Path("/kaggle/working")
    if kaggle_working.exists():
        return kaggle_working / "compression_routing_exp1"
    return Path(__file__).resolve().parent / "outputs_exp1"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=default_output_dir())
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260820)
    parser.add_argument("--expected-examples", type=int)
    parser.add_argument("--require-complete-matrix", action="store_true")
    return parser.parse_args()


def load_predictions(condition_dir):
    return {
        path.stem: json.loads(path.read_text(encoding="utf-8"))
        for path in (condition_dir / "predictions").glob("*.json")
    }


def distribution(counts):
    total = counts.sum()
    return counts / total if total else np.zeros_like(counts, dtype=float)


def total_variation(first, second):
    return float(0.5 * np.abs(distribution(first) - distribution(second)).sum())


def jensen_shannon(first, second):
    first = distribution(first)
    second = distribution(second)
    midpoint = 0.5 * (first + second)

    def kl(left, right):
        mask = left > 0
        return np.sum(left[mask] * np.log(left[mask] / right[mask]))

    return float(0.5 * kl(first, midpoint) + 0.5 * kl(second, midpoint))


def coefficient_of_variation(counts):
    mean = counts.mean()
    return float(counts.std() / mean) if mean else 0.0


def paired_example(baseline_dir, condition_dir, item_id, baseline_prediction, prediction):
    baseline = np.load(baseline_dir / "routes" / f"{item_id}.npz")
    condition = np.load(condition_dir / "routes" / f"{item_id}.npz")
    common_layers = sorted(set(baseline["layer_indices"].tolist()) & set(condition["layer_indices"].tolist()))

    layer_rows = []
    for layer_index in common_layers:
        baseline_position = int(np.where(baseline["layer_indices"] == layer_index)[0][0])
        condition_position = int(np.where(condition["layer_indices"] == layer_index)[0][0])
        baseline_counts = baseline["expert_counts"][baseline_position]
        condition_counts = condition["expert_counts"][condition_position]
        layer_rows.append(
            {
                "layer": layer_index,
                "baseline_visual_tokens": int(baseline["visual_tokens"][baseline_position]),
                "condition_visual_tokens": int(condition["visual_tokens"][condition_position]),
                "baseline_assignments": int(baseline["active_assignments"][baseline_position]),
                "condition_assignments": int(condition["active_assignments"][condition_position]),
                "load_tv": total_variation(baseline_counts, condition_counts),
                "load_jsd": jensen_shannon(baseline_counts, condition_counts),
                "load_cv_delta": coefficient_of_variation(condition_counts) - coefficient_of_variation(baseline_counts),
            }
        )

    if not layer_rows:
        raise RuntimeError(f"No common routed MoE layers found for {item_id}.")

    first_tokens = prediction["visual_tokens_first_layer"]
    last_tokens = prediction["visual_tokens_last_layer"]
    baseline_assignments = sum(row["baseline_assignments"] for row in layer_rows)
    condition_assignments = sum(row["condition_assignments"] for row in layer_rows)
    return {
        "id": item_id,
        "baseline_correct": int(baseline_prediction["correct"]),
        "condition_correct": int(prediction["correct"]),
        "accuracy_delta": int(prediction["correct"]) - int(baseline_prediction["correct"]),
        "actual_retention": last_tokens / first_tokens,
        "assignment_retention": condition_assignments / baseline_assignments,
        "load_tv": float(np.mean([row["load_tv"] for row in layer_rows])),
        "load_jsd": float(np.mean([row["load_jsd"] for row in layer_rows])),
        "load_cv_delta": float(np.mean([row["load_cv_delta"] for row in layer_rows])),
        "layers": layer_rows,
    }


def aggregate(rows):
    return {
        "examples": len(rows),
        "accuracy": float(np.mean([row["condition_correct"] for row in rows])),
        **{metric: float(np.mean([row[metric] for row in rows])) for metric in METRICS},
    }


def bootstrap_intervals(rows, samples, seed):
    rng = np.random.default_rng(seed)
    values = {metric: [] for metric in METRICS}
    for _ in range(samples):
        indices = rng.integers(0, len(rows), len(rows))
        result = aggregate([rows[index] for index in indices])
        for metric in METRICS:
            values[metric].append(result[metric])

    intervals = {}
    for metric, metric_values in values.items():
        low, high = np.percentile(metric_values, [2.5, 97.5])
        intervals[f"{metric}_ci95_low"] = float(low)
        intervals[f"{metric}_ci95_high"] = float(high)
    return intervals


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    root = args.output_dir / "fastmmoe_conditions"
    baseline_dir = root / "baseline"
    baseline_predictions = load_predictions(baseline_dir)
    if not baseline_predictions:
        raise RuntimeError(f"No FastMMoE baseline results found in {baseline_dir}.")
    if args.expected_examples and len(baseline_predictions) != args.expected_examples:
        raise RuntimeError(
            f"FastMMoE baseline has {len(baseline_predictions)} examples; expected {args.expected_examples}."
        )
    if any(prediction.get("schema_version") != 2 for prediction in baseline_predictions.values()):
        raise RuntimeError("FastMMoE baseline contains results from an older output schema.")

    if args.require_complete_matrix:
        required = {
            *(f"fastmmoe_prune_r{retention}" for retention in (75, 50, 25)),
            *(f"fastmmoe_full_r{retention}" for retention in (75, 50, 25)),
        }
        available = {path.name for path in root.iterdir() if path.is_dir()}
        missing = sorted(required - available)
        if missing:
            raise RuntimeError(f"Missing required FastMMoE conditions: {', '.join(missing)}")

    summary_rows = []
    per_example_rows = []
    layer_rows = []
    condition_dirs = sorted(path for path in root.iterdir() if path.is_dir() and path.name != "baseline")

    for condition_index, condition_dir in enumerate(condition_dirs):
        predictions = load_predictions(condition_dir)
        ids = sorted(set(baseline_predictions) & set(predictions))
        if not ids:
            continue
        if any(prediction.get("schema_version") != 2 for prediction in predictions.values()):
            raise RuntimeError(f"{condition_dir.name} contains results from an older output schema.")
        if args.expected_examples and len(ids) != args.expected_examples:
            raise RuntimeError(
                f"{condition_dir.name} has {len(ids)} paired examples; expected {args.expected_examples}."
            )
        examples = [
            paired_example(
                baseline_dir,
                condition_dir,
                item_id,
                baseline_predictions[item_id],
                predictions[item_id],
            )
            for item_id in ids
        ]
        first = predictions[ids[0]]
        summary_rows.append(
            {
                "condition": condition_dir.name,
                "mode": first["mode"],
                "target_retention": first["target_retention"],
                **aggregate(examples),
                **bootstrap_intervals(examples, args.bootstrap_samples, args.bootstrap_seed + condition_index),
            }
        )

        for row in examples:
            per_example_rows.append(
                {
                    "condition": condition_dir.name,
                    "mode": first["mode"],
                    "target_retention": first["target_retention"],
                    **{key: value for key, value in row.items() if key != "layers"},
                }
            )
            for layer in row["layers"]:
                layer_rows.append(
                    {
                        "condition": condition_dir.name,
                        "id": row["id"],
                        **layer,
                    }
                )

    baseline_accuracy = float(np.mean([prediction["correct"] for prediction in baseline_predictions.values()]))
    baseline_row = {
        "condition": "baseline",
        "mode": "baseline",
        "target_retention": 1.0,
        "examples": len(baseline_predictions),
        "accuracy": baseline_accuracy,
        **{metric: 1.0 if metric in {"actual_retention", "assignment_retention"} else 0.0 for metric in METRICS},
        **{
            f"{metric}_{bound}": 1.0 if metric in {"actual_retention", "assignment_retention"} else 0.0
            for metric in METRICS
            for bound in ("ci95_low", "ci95_high")
        },
    }
    summary_rows.insert(0, baseline_row)

    summary_dir = args.output_dir / "summaries" / "fastmmoe"
    write_csv(summary_dir / "summary.csv", summary_rows)
    if per_example_rows:
        write_csv(summary_dir / "per_example.csv", per_example_rows)
        write_csv(summary_dir / "per_example_layer.csv", layer_rows)

    for row in summary_rows:
        print(
            row["condition"],
            f"accuracy={row['accuracy']:.4f}",
            f"delta={row['accuracy_delta']:+.4f}",
            f"retention={row['actual_retention']:.4f}",
            f"load_TV={row['load_tv']:.4f}",
            sep="\t",
        )
    print(f"\nSaved {summary_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
