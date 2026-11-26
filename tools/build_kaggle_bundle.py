"""Build upload archives with the folder names expected by existing Kaggle notebooks."""

import argparse
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
BUNDLES = {
    "compression": [("compression_routing", "compression_routing_kaggle")],
    "textvqa": [
        ("textvqa", "experiment_1_textvqa"),
        ("compression_routing", "compression_routing_kaggle"),
    ],
    "temporal_routing": [("temporal_routing", "experiment_2_TRC")],
    "temporal_accuracy": [("temporal_accuracy", "experiment_3_TRC_accuracy")],
    "multimodel": [("multimodel", "kaggle_new_models")],
    "minor_revision": [("minor_revision", "minor_revision_experiments")],
    "control_accuracy": [("control_accuracy", "experiment_2_accuracy_only")],
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("suite", choices=list(BUNDLES) + ["all"])
    args = p.parse_args()
    (ROOT / "dist").mkdir(exist_ok=True)
    for name in BUNDLES if args.suite == "all" else [args.suite]:
        target = ROOT / "dist" / f"{name}_kaggle.zip"
        with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
            for folder, prefix in BUNDLES[name]:
                for path in sorted((ROOT / "experiments" / folder).iterdir()):
                    if path.is_file() and path.suffix in {
                        ".py",
                        ".sh",
                        ".txt",
                        ".md",
                        ".ipynb",
                        ".csv",
                    }:
                        archive.write(path, f"{prefix}/{path.name}")
        print(target)


if __name__ == "__main__":
    main()
