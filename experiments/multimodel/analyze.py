"""Validate complete matrices and run the existing experiment metrics."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from core import CONDITIONS, MODELS


def validate(root):
    config = json.loads((root / "run_config.json").read_text())
    manifest = json.loads((root / "manifests" / "examples.json").read_text())
    ids = {item["id"] for item in manifest}
    if len(ids) != config["limit"]:
        raise ValueError(f"Manifest size does not match config: {root}")
    if not (root / "complete.json").exists() or not (root / "native_equivalence.json").exists():
        raise ValueError(f"Run is incomplete or native equivalence was not checked: {root}")
    if config["experiment"] == 1:
        conditions = ["baseline"]
        for r in config["retentions"]:
            conditions.append(f"l2_r{round(r * 100)}")
            conditions.extend(f"random_r{round(r * 100)}_s{s}" for s in config["random_seeds"])
        if {p.name for p in (root / "conditions").iterdir() if p.is_dir()} != set(conditions):
            raise ValueError("Unexpected or missing pruning conditions.")
        for name in conditions:
            directory = root / "conditions" / name
            if {p.stem for p in (directory / "predictions").glob("*.json")} != ids:
                raise ValueError(f"Prediction matrix incomplete: {directory}")
            if {p.stem for p in (directory / "routes").glob("*.npz")} != ids:
                raise ValueError(f"Routing matrix incomplete: {directory}")
            for item_id in ids:
                with np.load(directory / "routes" / f"{item_id}.npz") as route:
                    if route["topk"].ndim != 3 or route["topk"].shape[1] != len(route["visual_ids"]):
                        raise ValueError(f"Invalid route dimensions: {item_id}")
    else:
        expected = {f"{item_id}__{name}" for item_id in ids for name in CONDITIONS}
        if {p.stem for p in (root / "routes").glob("*.npz")} != expected:
            raise ValueError("TRC routing matrix is incomplete or contains unexpected files.")
        for name in CONDITIONS:
            if {p.stem for p in (root / "condition_predictions" / name).glob("*.json")} != ids:
                raise ValueError(f"TRC prediction matrix incomplete: {name}")
        if {p.stem for p in (root / "predictions").glob("*.json")} != ids:
            raise ValueError("Chronological predictions are incomplete.")
        for path in (root / "routes").glob("*.npz"):
            with np.load(path) as route:
                if route["mass"].shape != route["counts"].shape or route["mass"].shape[0] != config["frames"]:
                    raise ValueError(f"Invalid TRC dimensions in {path}")
                if not np.allclose(route["mass"].sum(-1), 1.0, atol=1e-5):
                    raise ValueError(f"Unnormalized routing mass in {path}")
    return config, ids


def condition_accuracy(root, bootstrap):
    rng = np.random.default_rng(20260819)
    rows = []
    chronological = {p.stem: json.loads(p.read_text())["correct"] for p in (root / "predictions").glob("*.json")}
    for condition in CONDITIONS:
        predictions = {p.stem: json.loads(p.read_text()) for p in (root / "condition_predictions" / condition).glob("*.json")}
        ids = sorted(chronological)
        correct = np.asarray([predictions[i]["correct"] for i in ids], dtype=float)
        delta = correct - np.asarray([chronological[i] for i in ids], dtype=float)
        draws = rng.integers(len(ids), size=(bootstrap, len(ids)))
        low, high = np.percentile(delta[draws].mean(1), [2.5, 97.5])
        rows.append({"condition": condition, "examples": len(ids), "accuracy": correct.mean(),
                     "accuracy_delta_from_chronological": delta.mean(), "delta_ci95_low": low,
                     "delta_ci95_high": high,
                     "invalid_answer_fraction": np.mean([not predictions[i]["predicted"] for i in ids])})
    pd.DataFrame(rows).to_csv(root / "summaries" / "condition_accuracy.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("/kaggle/working/new_model_results"))
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--experiments", type=int, nargs="+", choices=[1, 2], default=[1, 2])
    parser.add_argument("--limit", type=int, default=256)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--permutations", type=int, default=100)
    args = parser.parse_args()
    if args.bootstrap < 1 or args.permutations < 1:
        parser.error("Bootstrap and permutation counts must be positive.")
    here = Path(__file__).resolve().parent
    for experiment in args.experiments:
        root = args.output_root / args.model / f"experiment_{experiment}_{'smoke' if args.smoke else 'n' + str(args.limit)}"
        config, ids = validate(root)
        command = [sys.executable, str(here / f"analyze_exp{experiment}.py"), "--output-dir", str(root)]
        if experiment == 1:
            command += ["--expected-examples", str(len(ids)), "--bootstrap-samples", str(args.bootstrap)]
        else:
            command += ["--bootstrap", str(args.bootstrap), "--permutations", str(args.permutations)]
        subprocess.run(command, check=True)
        if experiment == 2:
            condition_accuracy(root, args.bootstrap)
        for file in (root / "summaries").glob("*.csv"):
            table = pd.read_csv(file)
            table["model"] = config["model"]
            table["model_revision"] = config["model_revision"]
            table["precision"] = config["precision"]
            table["protocol"] = config["protocol"]
            table.to_csv(file, index=False)


if __name__ == "__main__":
    main()
