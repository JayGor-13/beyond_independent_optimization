#!/usr/bin/env python3
"""Validate complete paired runs and export summaries, contrasts and layerwise CIs."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from protocol import SCHEMA, write_json
from routing import METRICS, bootstrap_columns, paired_bootstrap, routing_metrics


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def read_complete_run(root, verify_routes=False):
    config = json.loads((root / "run_config.json").read_text())
    manifest = json.loads((root / "manifest.json").read_text())
    if config["schema"] != SCHEMA or config["examples"] != len(manifest["indices"]):
        raise ValueError(f"Invalid run configuration: {root}.")
    ids = [f"{config['benchmark']}_{index:06d}" for index in manifest["indices"]]
    records = {}
    for condition in config["conditions"]:
        directory = root / "conditions" / condition["name"]
        if {path.stem for path in directory.glob("*.json")} != set(ids):
            raise ValueError(f"Incomplete or unexpected examples in {directory}; no partial summary was produced.")
        rows = []
        for item_id in ids:
            row = json.loads((directory / f"{item_id}.json").read_text())
            if row["schema"] != SCHEMA or row["example_id"] != item_id or row["condition"] != condition:
                raise ValueError(f"Mismatched record in {directory}/{item_id}.")
            if config["route_storage"] == "full" and not (directory / f"{item_id}.npz").exists():
                raise ValueError(f"Missing raw routes: {directory}/{item_id}.npz.")
            for metric in METRICS:
                if not np.isfinite(row[metric]) or not np.isfinite(row["layerwise"][metric]).all():
                    raise ValueError("Non-finite metrics in saved record.")
            rows.append(row)
        records[condition["name"]] = rows
    baseline = records["baseline"]
    for name, rows in records.items():
        for reference, row in zip(baseline, rows):
            if (row["baseline_route_sha256"] != reference["baseline_route_sha256"]
                    or row["baseline_correct"] != reference["correct"] or row["problem"] != reference["problem"]):
                raise ValueError("Conditions are not paired to the same problem and baseline.")
            if row["condition"]["retention"] == 1 and (row["raw_answer"] != reference["raw_answer"]
                                                       or any(row[key] != 0 for key in METRICS)):
                raise ValueError("Identity control failed in saved outputs.")
            if verify_routes:
                if config["route_storage"] != "full":
                    raise ValueError("--verify-routes requires --route-storage full outputs.")
                with np.load(root / "conditions" / "baseline" / f"{row['example_id']}.npz", allow_pickle=False) as old:
                    with np.load(root / "conditions" / name / f"{row['example_id']}.npz", allow_pickle=False) as new:
                        info = json.loads((root / "loading_info.json").read_text())
                        metrics = routing_metrics(old["topk"], new["topk"], new["visual_ids"], info["n_experts"])
                for key in METRICS:
                    np.testing.assert_allclose(metrics[key], row[key], atol=1e-12, rtol=1e-12)
    if not list((root / "native_equivalence").glob("*.json")):
        raise ValueError("Native-equivalence validation is missing.")
    for path in (root / "native_equivalence").glob("*.json"):
        check = json.loads(path.read_text())
        if not check["next_token_matches"] or not check["all_visual_routes_match"]:
            raise ValueError("Native-equivalence validation failed.")
    return config, records


def arrays(rows):
    return {**{key: np.asarray([row[key] for row in rows], dtype=float) for key in METRICS},
            "accuracy": np.asarray([row["correct"] for row in rows], dtype=float),
            "valid_answer": np.asarray([row["valid_answer"] for row in rows], dtype=float),
            "actual_retention": np.asarray([row["actual_retention"] for row in rows], dtype=float)}


def summarize_run(root, bootstrap_samples=2000, verify_routes=False):
    config, records = read_complete_run(root, verify_routes)
    baseline = arrays(records["baseline"])["accuracy"]
    common = {"model": config["repo"], "model_revision": config["revision"], "precision": config["precision"],
              "benchmark": config["benchmark"], "phase": config["phase"], "examples": len(baseline)}
    groups = [(condition, arrays(records[condition["name"]]), records[condition["name"]])
              for condition in config["conditions"]]
    for retention in sorted({c["retention"] for c in config["conditions"] if c["retention"] < 1}, reverse=True):
        random_rows = [records[c["name"]] for c in config["conditions"]
                       if c["method"] == "random" and c["retention"] == retention]
        combined = {key: np.stack([arrays(rows)[key] for rows in random_rows]).mean(axis=0)
                    for key in arrays(random_rows[0])}
        layers = [{"layerwise": {key: np.stack([rows[i]["layerwise"][key] for rows in random_rows]).mean(axis=0)
                                  for key in METRICS}} for i in range(len(baseline))]
        groups.append(({"name": f"random_mean_r{round(retention * 10000)}", "method": "random_seed_mean",
                        "retention": retention, "seed": None}, combined, layers))
    summary, layerwise = [], []
    for condition, values, rows in groups:
        row = {**common, "condition": condition["name"], "method": condition["method"],
               "retention": condition["retention"], "random_seed": condition["seed"]}
        measures = {**values, "accuracy_delta_pp": 100 * (values["accuracy"] - baseline)}
        estimates = bootstrap_columns(np.column_stack(list(measures.values())), bootstrap_samples)
        for column, key in enumerate(measures):
            mean, low, high = (float(value[column]) for value in estimates)
            row.update({key: mean, key + "_ci95_low": low, key + "_ci95_high": high})
        summary.append(row)
        all_layers = np.concatenate([np.asarray([record["layerwise"][key] for record in rows]) for key in METRICS], axis=1)
        layer_estimates = bootstrap_columns(all_layers, bootstrap_samples)
        layer_count = all_layers.shape[1] // len(METRICS)
        for metric_index, key in enumerate(METRICS):
            for layer in range(layer_count):
                column = metric_index * layer_count + layer
                mean, low, high = (float(value[column]) for value in layer_estimates)
                layerwise.append({**common, "condition": condition["name"], "layer": layer, "metric": key,
                                  "mean": mean, "ci95_low": low, "ci95_high": high})
    contrasts = []
    for condition, values, _ in groups:
        if condition["method"] != "l2" or condition["retention"] == 1:
            continue
        random = next(v for c, v, _ in groups if c["method"] == "random_seed_mean" and c["retention"] == condition["retention"])
        contrast = {**common, "retention": condition["retention"], "contrast": "l2_minus_random_seed_mean"}
        for key in ("accuracy", *METRICS):
            unit = "accuracy_pp" if key == "accuracy" else key
            mean, low, high = paired_bootstrap((values[key] - random[key]) * (100 if key == "accuracy" else 1), bootstrap_samples)
            contrast.update({unit: mean, unit + "_ci95_low": low, unit + "_ci95_high": high})
        contrasts.append(contrast)
    destination = root / "summaries"
    write_csv(destination / "summary.csv", summary)
    write_csv(destination / "layerwise.csv", layerwise)
    write_csv(destination / "l2_random_contrasts.csv", contrasts)
    write_json(destination / "analysis_metadata.json", {"schema": SCHEMA, "bootstrap_replicates": bootstrap_samples,
               "sampling_unit": "paired question; random seeds averaged within each question for pooled random rows",
               "units": "accuracy and routing metrics are fractions; accuracy_delta_pp is percentage points",
               "verify_routes": verify_routes})
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path(__file__).parent / "outputs")
    parser.add_argument("--include-smoke", action="store_true")
    parser.add_argument("--verify-routes", action="store_true")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    args = parser.parse_args(argv)
    combined = []
    for path in sorted(args.output_root.glob("*/*/*/*/run_config.json")):
        config = json.loads(path.read_text())
        if config["phase"] == "smoke" and not args.include_smoke:
            continue
        combined.extend(summarize_run(path.parent, args.bootstrap_samples, args.verify_routes))
    if not combined:
        raise RuntimeError("No eligible complete runs found. Smoke outputs require --include-smoke.")
    write_csv(args.output_root / "combined_summary.csv", combined)
    lines = ["# Additional Section 4.2 results", "", "Measured outputs only; each row is a separate model/benchmark/precision.", "",
             "| Model | Precision | Benchmark | Phase | N | Retention | Method | Accuracy % | Δ accuracy pp | RACR % | Top-1 change % |",
             "|---|---|---|---|---:|---:|---|---:|---:|---:|---:|"]
    for row in combined:
        if row["method"] == "random":
            continue  # Individual seeds remain in CSV; Markdown displays their within-question average.
        lines.append(f"| {row['model']} | {row['precision']} | {row['benchmark']} | {row['phase']} | {row['examples']} | "
                     f"{row['retention']:.0%} | {row['method']} | {100 * row['accuracy']:.2f} | "
                     f"{row['accuracy_delta_pp']:+.2f} | {100 * row['racr']:.2f} | {100 * row['top1_change']:.2f} |")
    (args.output_root / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Validated results written to {args.output_root / 'RESULTS.md'}")


if __name__ == "__main__":
    main()
