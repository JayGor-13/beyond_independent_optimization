"""Rebuild empirical paper tables and figures from included evidence; no GPU needed."""

import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/reproduction")
    args = p.parse_args()
    tables = args.output_dir.resolve() / "tables"
    figures = args.output_dir.resolve() / "figures"
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", str(args.output_dir.resolve() / "matplotlib-cache"))
    env.setdefault("MPLBACKEND", "Agg")
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/extract_model_results.py"),
            "--output-dir",
            str(tables),
        ],
        cwd=ROOT,
        env=env,
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/plot_model_comparison.py"),
            "--data-dir",
            str(tables),
            "--output-dir",
            str(figures),
        ],
        cwd=ROOT,
        env=env,
        check=True,
    )
    print(f"Rebuilt empirical tables and figures in {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
