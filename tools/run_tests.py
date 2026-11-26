"""Run each standalone experiment suite in a fresh process."""

import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SUITES = {
    "compression": "compression_routing",
    "textvqa": "textvqa",
    "temporal_accuracy": "temporal_accuracy",
    "control_accuracy": "control_accuracy",
    "multimodel": "multimodel",
    "minor_revision": "minor_revision",
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("suites", nargs="+", choices=list(SUITES) + ["legacy"])
    selected = []
    for name in p.parse_args().suites:
        selected.extend(
            ["compression", "textvqa", "temporal_accuracy", "control_accuracy"]
            if name == "legacy"
            else [name]
        )
    for name in dict.fromkeys(selected):
        directory = ROOT / "experiments" / SUITES[name]
        env = os.environ.copy()
        env.update(
            PYTHONPATH=str(directory),
            PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
            USE_TORCH="1",
            USE_TF="0",
            USE_FLAX="0",
        )
        print(f"Testing {name} with {sys.executable}", flush=True)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "-p",
                "no:cacheprovider",
                str(directory),
            ],
            cwd=directory,
            env=env,
            check=True,
        )


if __name__ == "__main__":
    main()
