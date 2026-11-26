"""Analyze TextVQA score changes and retained-token routing changes."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


PRIMARY_METRICS = [
    "accuracy_delta", "actual_retention", "racr_example", "top1_flip",
    "exact_topk_change", "load_cv_delta",
]


def default_output_dir() -> Path:
    root = Path("/kaggle/working")
    return root / "compression_routing_exp1_textvqa" if root.exists() else Path(__file__).parent / "outputs_textvqa"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=default_output_dir())
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260820)
    parser.add_argument("--expected-examples", type=int)
    return parser.parse_args()


def load_predictions(condition_dir: Path) -> dict[str, dict]:
    return {
        path.stem: json.loads(path.read_text(encoding="utf-8"))
        for path in (condition_dir / "predictions").glob("*.json")
    }


def coefficient_of_variation(counts: np.ndarray) -> float:
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
    exact = np.all(np.sort(condition_topk, axis=-1) == np.sort(baseline_topk, axis=-1), axis=-1)
    n_experts = int(condition["n_experts"])
    baseline_counts = np.bincount(baseline_topk.ravel(), minlength=n_experts)
    condition_counts = np.bincount(condition_topk.ravel(), minlength=n_experts)
    before = prediction["visual_tokens_before"]
    after = prediction["visual_tokens_after"]
    baseline_score = float(baseline_prediction["vqa_score"])
    condition_score = float(prediction["vqa_score"])
    return {
        "id": item_id,
        "baseline_score": baseline_score,
        "condition_score": condition_score,
        "accuracy_delta": condition_score - baseline_score,
        "actual_retention": after / before,
        "racr_example": float(racr.mean()),
        "top1_flip": float(top1.mean()),
        "exact_topk_change": float((~exact).mean()),
        "units": int(racr.size),
        "racr_sum": float(racr.sum()),
        "baseline_counts": baseline_counts,
        "condition_counts": condition_counts,
    }


def aggregate(rows):
    baseline_counts = np.sum([row["baseline_counts"] for row in rows], axis=0)
    condition_counts = np.sum([row["condition_counts"] for row in rows], axis=0)
    return {
        "examples": len(rows),
        "accuracy": float(np.mean([row["condition_score"] for row in rows])),
        "accuracy_delta": float(np.mean([row["accuracy_delta"] for row in rows])),
        "actual_retention": float(np.mean([row["actual_retention"] for row in rows])),
        "racr_token": float(sum(row["racr_sum"] for row in rows) / sum(row["units"] for row in rows)),
        "racr_example": float(np.mean([row["racr_example"] for row in rows])),
        "top1_flip": float(np.mean([row["top1_flip"] for row in rows])),
        "exact_topk_change": float(np.mean([row["exact_topk_change"] for row in rows])),
        "load_cv_delta": coefficient_of_variation(condition_counts) - coefficient_of_variation(baseline_counts),
    }


def bootstrap_intervals(rows, samples, seed):
    rng = np.random.default_rng(seed)
    values = {metric: [] for metric in PRIMARY_METRICS}
    for _ in range(samples):
        indices = rng.integers(0, len(rows), len(rows))
        result = aggregate([rows[index] for index in indices])
        for metric in PRIMARY_METRICS:
            values[metric].append(result[metric])
    intervals = {}
    for metric, metric_values in values.items():
        low, high = np.percentile(metric_values, [2.5, 97.5])
        intervals[f"{metric}_ci95_low"] = float(low)
        intervals[f"{metric}_ci95_high"] = float(high)
    return intervals


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    root = args.output_dir / "conditions"
    baseline_dir = root / "baseline"
    baseline_predictions = load_predictions(baseline_dir)
    if not baseline_predictions:
        raise RuntimeError(f"No completed baseline predictions found in {baseline_dir}.")
    if args.expected_examples and len(baseline_predictions) != args.expected_examples:
        raise RuntimeError(f"Baseline has {len(baseline_predictions)} examples; expected {args.expected_examples}.")

    summary_rows = []
    per_example_rows = []
    condition_dirs = sorted(path for path in root.iterdir() if path.is_dir() and path.name != "baseline")
    for condition_index, condition_dir in enumerate(condition_dirs):
        predictions = load_predictions(condition_dir)
        ids = sorted(set(baseline_predictions) & set(predictions))
        if not ids:
            continue
        if args.expected_examples and len(ids) != args.expected_examples:
            raise RuntimeError(f"{condition_dir.name} has {len(ids)} paired examples; expected {args.expected_examples}.")
        examples = [
            paired_example(baseline_dir, condition_dir, item_id, baseline_predictions[item_id], predictions[item_id])
            for item_id in ids
        ]
        result = aggregate(examples)
        first = predictions[ids[0]]
        summary_rows.append({
            "condition": condition_dir.name,
            "method": first["method"],
            "target_retention": first["retention"],
            "target_compression": first["compression_ratio"],
            "random_seed": first.get("random_seed"),
            "metric": "textvqa_vqa_soft_accuracy",
            **result,
            **bootstrap_intervals(examples, args.bootstrap_samples, args.bootstrap_seed + condition_index),
        })
        for row in examples:
            per_example_rows.append({
                "condition": condition_dir.name,
                "method": first["method"],
                "target_retention": first["retention"],
                "target_compression": first["compression_ratio"],
                "random_seed": first.get("random_seed"),
                **{key: value for key, value in row.items() if key not in {"baseline_counts", "condition_counts", "racr_sum"}},
            })

    baseline_accuracy = float(np.mean([prediction["vqa_score"] for prediction in baseline_predictions.values()]))
    zero_intervals = {f"{metric}_ci95_{bound}": 0.0 for metric in PRIMARY_METRICS for bound in ("low", "high")}
    baseline_row = {
        "condition": "baseline", "method": "baseline", "target_retention": 1.0,
        "target_compression": 0.0, "random_seed": None,
        "metric": "textvqa_vqa_soft_accuracy", "examples": len(baseline_predictions),
        "accuracy": baseline_accuracy, "accuracy_delta": 0.0, "actual_retention": 1.0,
        "racr_token": 0.0, "racr_example": 0.0, "top1_flip": 0.0,
        "exact_topk_change": 0.0, "load_cv_delta": 0.0, **zero_intervals,
    }
    summary_rows.insert(0, baseline_row)
    summary_dir = args.output_dir / "summaries"
    write_csv(summary_dir / "summary.csv", summary_rows)
    if per_example_rows:
        write_csv(summary_dir / "per_example.csv", per_example_rows)
    for row in summary_rows:
        print(row["condition"], f"TextVQA={row['accuracy']:.4f}", f"delta={row['accuracy_delta']:+.4f}",
              f"RACR={row['racr_example']:.4f}", f"top1_flip={row['top1_flip']:.4f}", sep="\t")
    print(f"Saved {summary_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()

