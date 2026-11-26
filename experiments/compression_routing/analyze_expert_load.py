import argparse
import csv
import json
from pathlib import Path

import numpy as np


def default_output_dir():
    kaggle_working = Path("/kaggle/working")
    if kaggle_working.exists():
        return kaggle_working / "compression_routing_exp1"
    return Path(__file__).resolve().parent / "outputs_exp1"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=default_output_dir())
    return parser.parse_args()


def load_predictions(condition_dir):
    return {
        path.stem: json.loads(path.read_text(encoding="utf-8"))
        for path in (condition_dir / "predictions").glob("*.json")
    }


def layer_counts(topk, n_experts):
    return np.stack(
        [np.bincount(layer.ravel(), minlength=n_experts) for layer in topk]
    )


def shares(counts):
    return counts / counts.sum(axis=-1, keepdims=True)


def total_variation(first, second):
    return 0.5 * np.abs(first - second).sum(axis=-1)


def normalized_entropy(distribution):
    safe = np.where(distribution > 0, distribution, 1.0)
    return -(distribution * np.log(safe)).sum(axis=-1) / np.log(distribution.shape[-1])


def coefficient_of_variation(counts):
    return counts.std(axis=-1) / counts.mean(axis=-1)


def transition_name(baseline_correct, condition_correct):
    if baseline_correct and condition_correct:
        return "still_correct"
    if baseline_correct and not condition_correct:
        return "new_wrong"
    if not baseline_correct and condition_correct:
        return "new_correct"
    return "still_wrong"


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def mean_for_group(rows, transition, key):
    values = [row[key] for row in rows if row["transition"] == transition]
    return float(np.mean(values)) if values else float("nan")


def main():
    args = parse_args()
    conditions_root = args.output_dir / "conditions"
    baseline_dir = conditions_root / "baseline"
    baseline_predictions = load_predictions(baseline_dir)
    if not baseline_predictions:
        raise RuntimeError(f"No baseline results found in {baseline_dir}.")

    condition_rows = []
    layer_rows = []
    expert_rows = []
    example_rows = []
    accuracy_group_rows = []

    condition_dirs = sorted(
        path for path in conditions_root.iterdir()
        if path.is_dir() and path.name != "baseline"
    )

    for condition_dir in condition_dirs:
        predictions = load_predictions(condition_dir)
        ids = sorted(set(baseline_predictions) & set(predictions))
        if not ids:
            continue

        first_baseline = np.load(baseline_dir / "routes" / f"{ids[0]}.npz")
        n_experts = int(first_baseline["n_experts"])
        layer_names = first_baseline["layer_names"].astype(str)
        n_layers = len(layer_names)

        baseline_all_counts = np.zeros((n_layers, n_experts), dtype=np.int64)
        baseline_matched_counts = np.zeros((n_layers, n_experts), dtype=np.int64)
        compressed_counts = np.zeros((n_layers, n_experts), dtype=np.int64)
        condition_example_rows = []

        for item_id in ids:
            baseline = np.load(baseline_dir / "routes" / f"{item_id}.npz")
            condition = np.load(condition_dir / "routes" / f"{item_id}.npz")
            visual_ids = condition["visual_ids"].astype(int)

            baseline_all = baseline["topk"]
            baseline_matched = baseline_all[:, visual_ids]
            compressed = condition["topk"]

            all_counts = layer_counts(baseline_all, n_experts)
            matched_counts = layer_counts(baseline_matched, n_experts)
            current_counts = layer_counts(compressed, n_experts)

            baseline_all_counts += all_counts
            baseline_matched_counts += matched_counts
            compressed_counts += current_counts

            matched_share = shares(matched_counts)
            compressed_share = shares(current_counts)
            layer_tv = total_variation(matched_share, compressed_share)
            transition = transition_name(
                baseline_predictions[item_id]["correct"],
                predictions[item_id]["correct"],
            )

            row = {
                "condition": condition_dir.name,
                "id": item_id,
                "transition": transition,
                "baseline_correct": baseline_predictions[item_id]["correct"],
                "condition_correct": predictions[item_id]["correct"],
                "load_tv_mean": float(layer_tv.mean()),
                "load_tv_max": float(layer_tv.max()),
            }
            example_rows.append(row)
            condition_example_rows.append(row)

        full_share = shares(baseline_all_counts)
        matched_share = shares(baseline_matched_counts)
        compressed_share = shares(compressed_counts)
        layer_tv = total_variation(matched_share, compressed_share)
        full_entropy = normalized_entropy(full_share)
        matched_entropy = normalized_entropy(matched_share)
        compressed_entropy = normalized_entropy(compressed_share)
        matched_cv = coefficient_of_variation(baseline_matched_counts)
        compressed_cv = coefficient_of_variation(compressed_counts)
        assignment_retention = compressed_counts.sum(axis=-1) / baseline_all_counts.sum(axis=-1)

        baseline_accuracy = np.mean([baseline_predictions[item_id]["correct"] for item_id in ids])
        condition_accuracy = np.mean([predictions[item_id]["correct"] for item_id in ids])

        condition_rows.append(
            {
                "condition": condition_dir.name,
                "examples": len(ids),
                "baseline_accuracy": baseline_accuracy,
                "condition_accuracy": condition_accuracy,
                "accuracy_delta": condition_accuracy - baseline_accuracy,
                "assignment_retention": float(assignment_retention.mean()),
                "load_tv_mean": float(layer_tv.mean()),
                "load_tv_max": float(layer_tv.max()),
                "entropy_delta_mean": float((compressed_entropy - matched_entropy).mean()),
                "load_cv_delta_mean": float((compressed_cv - matched_cv).mean()),
                "max_expert_share_delta_mean": float((compressed_share.max(axis=-1) - matched_share.max(axis=-1)).mean()),
                "load_tv_still_correct": mean_for_group(condition_example_rows, "still_correct", "load_tv_mean"),
                "load_tv_new_wrong": mean_for_group(condition_example_rows, "new_wrong", "load_tv_mean"),
            }
        )

        for transition in ["still_correct", "new_wrong", "new_correct", "still_wrong"]:
            group = [row for row in condition_example_rows if row["transition"] == transition]
            accuracy_group_rows.append(
                {
                    "condition": condition_dir.name,
                    "transition": transition,
                    "examples": len(group),
                    "load_tv_mean": float(np.mean([row["load_tv_mean"] for row in group])) if group else float("nan"),
                    "load_tv_max_mean": float(np.mean([row["load_tv_max"] for row in group])) if group else float("nan"),
                }
            )

        for layer_index, layer_name in enumerate(layer_names):
            active_matched = int((baseline_matched_counts[layer_index] > 0).sum())
            active_compressed = int((compressed_counts[layer_index] > 0).sum())
            layer_rows.append(
                {
                    "condition": condition_dir.name,
                    "layer_index": layer_index,
                    "layer_name": layer_name,
                    "assignment_retention": float(assignment_retention[layer_index]),
                    "load_tv": float(layer_tv[layer_index]),
                    "baseline_entropy": float(full_entropy[layer_index]),
                    "baseline_matched_entropy": float(matched_entropy[layer_index]),
                    "compressed_entropy": float(compressed_entropy[layer_index]),
                    "entropy_delta": float(compressed_entropy[layer_index] - matched_entropy[layer_index]),
                    "baseline_matched_cv": float(matched_cv[layer_index]),
                    "compressed_cv": float(compressed_cv[layer_index]),
                    "load_cv_delta": float(compressed_cv[layer_index] - matched_cv[layer_index]),
                    "baseline_matched_max_share": float(matched_share[layer_index].max()),
                    "compressed_max_share": float(compressed_share[layer_index].max()),
                    "active_experts_baseline_matched": active_matched,
                    "active_experts_compressed": active_compressed,
                }
            )

            expected_scaled = baseline_all_counts[layer_index] * assignment_retention[layer_index]
            for expert_id in range(n_experts):
                expert_rows.append(
                    {
                        "condition": condition_dir.name,
                        "layer_index": layer_index,
                        "layer_name": layer_name,
                        "expert_id": expert_id,
                        "baseline_all_tokens": int(baseline_all_counts[layer_index, expert_id]),
                        "baseline_matched_tokens": int(baseline_matched_counts[layer_index, expert_id]),
                        "compressed_tokens": int(compressed_counts[layer_index, expert_id]),
                        "baseline_matched_share": float(matched_share[layer_index, expert_id]),
                        "compressed_share": float(compressed_share[layer_index, expert_id]),
                        "share_delta": float(compressed_share[layer_index, expert_id] - matched_share[layer_index, expert_id]),
                        "redistribution_delta_tokens": int(compressed_counts[layer_index, expert_id] - baseline_matched_counts[layer_index, expert_id]),
                        "expected_tokens_after_scaling": float(expected_scaled[expert_id]),
                        "residual_vs_scaled": float(compressed_counts[layer_index, expert_id] - expected_scaled[expert_id]),
                    }
                )

    summary_dir = args.output_dir / "summaries" / "expert_load"
    summary_dir.mkdir(parents=True, exist_ok=True)
    write_csv(summary_dir / "conditions.csv", condition_rows)
    write_csv(summary_dir / "layers.csv", layer_rows)
    write_csv(summary_dir / "experts.csv", expert_rows)
    write_csv(summary_dir / "examples.csv", example_rows)
    write_csv(summary_dir / "accuracy_groups.csv", accuracy_group_rows)

    for row in condition_rows:
        print(
            row["condition"],
            f"accuracy_delta={row['accuracy_delta']:+.4f}",
            f"load_TV={row['load_tv_mean']:.4f}",
            f"CV_delta={row['load_cv_delta_mean']:+.4f}",
            f"TV_new_wrong={row['load_tv_new_wrong']:.4f}",
            sep="\t",
        )
    print(f"\nSaved expert-load analysis to {summary_dir}")


if __name__ == "__main__":
    main()
