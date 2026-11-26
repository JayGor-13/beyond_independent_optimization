import argparse
import csv
import json
from pathlib import Path

import numpy as np


DEFAULT_COMPRESSION_RATIOS = [0.25, 0.30, 0.50, 0.75]


PRIMARY_METRICS = [
    "accuracy_delta",
    "actual_retention",
    "racr_example",
    "top1_flip",
    "exact_topk_change",
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
    parser.add_argument(
        "--compression-ratios",
        nargs="+",
        type=float,
        default=DEFAULT_COMPRESSION_RATIOS,
        help="Expected fractions of visual tokens removed when checking the complete matrix.",
    )
    return parser.parse_args()


def load_predictions(condition_dir):
    return {
        path.stem: json.loads(path.read_text(encoding="utf-8"))
        for path in (condition_dir / "predictions").glob("*.json")
    }


def coefficient_of_variation(counts):
    mean = counts.mean()
    return float(counts.std() / mean) if mean else 0.0


def paired_example(baseline_dir, condition_dir, item_id, baseline_prediction, prediction):
    baseline = np.load(baseline_dir / "routes" / f"{item_id}.npz")
    condition = np.load(condition_dir / "routes" / f"{item_id}.npz")

    visual_ids = condition["visual_ids"].astype(int)
    baseline_topk = baseline["topk"][:, visual_ids]
    condition_topk = condition["topk"]
    k = condition_topk.shape[-1]

    matches = condition_topk[..., :, None] == baseline_topk[..., None, :]
    intersection = matches.any(axis=-1).sum(axis=-1)
    racr = 1.0 - intersection / k
    top1 = condition_topk[..., 0] != baseline_topk[..., 0]
    exact = np.all(
        np.sort(condition_topk, axis=-1) == np.sort(baseline_topk, axis=-1),
        axis=-1,
    )

    n_experts = int(condition["n_experts"])
    baseline_counts = np.bincount(baseline_topk.ravel(), minlength=n_experts)
    condition_counts = np.bincount(condition_topk.ravel(), minlength=n_experts)
    before = prediction["visual_tokens_before"]
    after = prediction["visual_tokens_after"]

    return {
        "id": item_id,
        "baseline_correct": int(baseline_prediction["correct"]),
        "condition_correct": int(prediction["correct"]),
        "accuracy_delta": int(prediction["correct"]) - int(baseline_prediction["correct"]),
        "actual_retention": after / before,
        "racr_example": float(racr.mean()),
        "top1_flip": float(top1.mean()),
        "exact_topk_change": float((~exact).mean()),
        "units": int(racr.size),
        "racr_sum": float(racr.sum()),
        "baseline_counts": baseline_counts,
        "condition_counts": condition_counts,
    }


def aggregate(example_rows):
    baseline_counts = np.sum([row["baseline_counts"] for row in example_rows], axis=0)
    condition_counts = np.sum([row["condition_counts"] for row in example_rows], axis=0)
    return {
        "examples": len(example_rows),
        "accuracy": float(np.mean([row["condition_correct"] for row in example_rows])),
        "accuracy_delta": float(np.mean([row["accuracy_delta"] for row in example_rows])),
        "actual_retention": float(np.mean([row["actual_retention"] for row in example_rows])),
        "racr_token": float(sum(row["racr_sum"] for row in example_rows) / sum(row["units"] for row in example_rows)),
        "racr_example": float(np.mean([row["racr_example"] for row in example_rows])),
        "top1_flip": float(np.mean([row["top1_flip"] for row in example_rows])),
        "exact_topk_change": float(np.mean([row["exact_topk_change"] for row in example_rows])),
        "load_cv_delta": coefficient_of_variation(condition_counts) - coefficient_of_variation(baseline_counts),
    }


def bootstrap_intervals(example_rows, samples, seed):
    rng = np.random.default_rng(seed)
    values = {metric: [] for metric in PRIMARY_METRICS}
    for _ in range(samples):
        indices = rng.integers(0, len(example_rows), len(example_rows))
        result = aggregate([example_rows[index] for index in indices])
        for metric in PRIMARY_METRICS:
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
    if any(not 0 < ratio < 1 for ratio in args.compression_ratios):
        raise ValueError("Compression ratios must be between 0 and 1.")
    retention_percentages = tuple(
        round((1.0 - ratio) * 100) for ratio in args.compression_ratios
    )
    conditions_root = args.output_dir / "conditions"
    baseline_dir = conditions_root / "baseline"
    baseline_predictions = load_predictions(baseline_dir)
    if not baseline_predictions:
        raise RuntimeError(f"No completed baseline examples found in {baseline_dir}.")
    if args.expected_examples and len(baseline_predictions) != args.expected_examples:
        raise RuntimeError(
            f"Baseline has {len(baseline_predictions)} examples; expected {args.expected_examples}."
        )
    if any(prediction.get("schema_version") != 2 for prediction in baseline_predictions.values()):
        raise RuntimeError("Baseline contains results from an older output schema.")

    if args.require_complete_matrix:
        required = {
            *(f"l2_r{retention}" for retention in retention_percentages),
            *(
                f"random_r{retention}_s{seed}"
                for retention in retention_percentages
                for seed in (0, 1, 2)
            ),
        }
        available = {path.name for path in conditions_root.iterdir() if path.is_dir()}
        missing = sorted(required - available)
        if missing:
            raise RuntimeError(f"Missing required conditions: {', '.join(missing)}")

    summary_rows = []
    per_example_rows = []
    condition_dirs = sorted(
        path for path in conditions_root.iterdir()
        if path.is_dir() and path.name != "baseline"
    )

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
        result = aggregate(examples)
        first = predictions[ids[0]]
        result = {
            "condition": condition_dir.name,
            "method": first["method"],
            "target_retention": first["retention"],
            "target_compression": first.get("compression_ratio", 1.0 - first["retention"]),
            "random_seed": first.get("random_seed"),
            **result,
            **bootstrap_intervals(
                examples,
                args.bootstrap_samples,
                args.bootstrap_seed + condition_index,
            ),
        }
        summary_rows.append(result)

        for row in examples:
            per_example_rows.append(
                {
                    "condition": condition_dir.name,
                    "method": first["method"],
                    "target_retention": first["retention"],
                    "target_compression": first.get("compression_ratio", 1.0 - first["retention"]),
                    "random_seed": first.get("random_seed"),
                    **{
                        key: value for key, value in row.items()
                        if key not in {"baseline_counts", "condition_counts", "racr_sum"}
                    },
                }
            )

    baseline_accuracy = float(np.mean([prediction["correct"] for prediction in baseline_predictions.values()]))
    zero_intervals = {
        f"{metric}_{bound}": 0.0
        for metric in PRIMARY_METRICS
        for bound in ("ci95_low", "ci95_high")
    }
    baseline_row = {
        "condition": "baseline",
        "method": "baseline",
        "target_retention": 1.0,
        "target_compression": 0.0,
        "random_seed": None,
        "examples": len(baseline_predictions),
        "accuracy": baseline_accuracy,
        "accuracy_delta": 0.0,
        "actual_retention": 1.0,
        "racr_token": 0.0,
        "racr_example": 0.0,
        "top1_flip": 0.0,
        "exact_topk_change": 0.0,
        "load_cv_delta": 0.0,
        **zero_intervals,
    }
    summary_rows.insert(0, baseline_row)

    summary_dir = args.output_dir / "summaries"
    write_csv(summary_dir / "summary.csv", summary_rows)
    if per_example_rows:
        write_csv(summary_dir / "per_example.csv", per_example_rows)

    for row in summary_rows:
        print(
            row["condition"],
            f"compression={row['target_compression']:.0%}",
            f"accuracy={row['accuracy']:.4f}",
            f"delta={row['accuracy_delta']:+.4f}",
            f"RACR={row['racr_example']:.4f}",
            f"top1_flip={row['top1_flip']:.4f}",
            sep="\t",
        )
    print(f"\nSaved {summary_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
