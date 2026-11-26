"""Regenerate the two ready-to-import notebooks and upload archive."""
import json
import zipfile
from pathlib import Path


def cell(kind, text):
    result = {"cell_type": kind, "metadata": {}, "source": text.splitlines(keepends=True)}
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


def notebook(alias, title):
    data = {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python"},
                     "kaggle": {"isInternetEnabled": True, "isGpuEnabled": True, "accelerator": "nvidiaTeslaT4"}},
        "cells": [
            cell("markdown", f"# {title}: experiments 1 and 2\n\nAttach the uploaded `kaggle_new_models` dataset, enable Internet, and select **T4 x2** manually. This notebook runs only `{alias}`. Default precision is expert-NF4; TRC uses eight separately encoded frames. See README.md for methodology and limits.\n"),
            cell("code", "from pathlib import Path\nimport subprocess, sys, os, shutil\n"
                 "matches = list(Path('/kaggle/input').rglob('kaggle_new_models/run.py'))\n"
                 "assert len(matches) == 1, f'Expected one uploaded package; found {matches}'\n"
                 "CODE = matches[0].parent\n"
                 f"MODEL = '{alias}'\n"
                 "OUTPUT = Path('/kaggle/working/new_model_results')\n"
                 "os.environ['HF_XET_CHUNK_CACHE_SIZE_BYTES'] = '0'\n"
                 "subprocess.run(['bash', str(CODE / 'setup_kaggle.sh')], check=True)\n"
                 "subprocess.run([sys.executable, '-m', 'pytest', str(CODE), '-q', '-p', 'no:cacheprovider'], check=True)\n"),
            cell("markdown", "## Smoke test\nTwo examples per experiment, in separate folders. Weight downloads are still full size, but only one shard is retained at a time. A failed native-equivalence check or incomplete matrix stops the subprocess.\n"),
            cell("code", "subprocess.run([sys.executable, str(CODE / 'run.py'), '--model', MODEL, '--smoke'], check=True)\n"
                 "subprocess.run([sys.executable, str(CODE / 'analyze.py'), '--model', MODEL, '--smoke'], check=True)\n"),
            cell("code", "import json, pandas as pd\n"
                 "for experiment in (1, 2):\n"
                 "    root = OUTPUT / MODEL / f'experiment_{experiment}_smoke'\n"
                 "    print(f'Experiment {experiment}:', json.loads((root / 'native_equivalence.json').read_text()))\n"
                 "    display(pd.read_csv(root / 'summaries' / 'summary.csv'))\n"),
            cell("markdown", "## Full runs\nAfter inspecting the smoke results, run this cell. Each experiment uses 256 examples. Experiment 1 runs baseline, L2 at 50% retention, and random at 50% retention. Experiment 2 runs all four frame conditions. Rerun the same cell to resume; results must still be present under `/kaggle/working`. Model loading downloads shards again because the smoke subprocess released its model. Full runtime has not been measured; save outputs before the Kaggle session expires.\n"),
            cell("code", "subprocess.run([sys.executable, str(CODE / 'run.py'), '--model', MODEL, '--limit', '256'], check=True)\n"
                 "subprocess.run([sys.executable, str(CODE / 'analyze.py'), '--model', MODEL, '--limit', '256'], check=True)\n"),
            cell("code", "for experiment in (1, 2):\n"
                 "    root = OUTPUT / MODEL / f'experiment_{experiment}_n256'\n"
                 "    display(pd.read_csv(root / 'summaries' / 'summary.csv'))\n"
                 "display(pd.read_csv(OUTPUT / MODEL / 'experiment_2_n256/summaries/condition_accuracy.csv'))\n"),
            cell("markdown", "## Download outputs\nThis cell can also be run after an interrupted experiment to preserve partial progress. Keep raw routes, predictions, manifests and configuration with the summaries.\n"),
            cell("code", "archive = shutil.make_archive(f'/kaggle/working/{MODEL}_experiment_results', 'zip', OUTPUT)\n"
                 "from IPython.display import FileLink, display\n"
                 "display(FileLink(archive))\n"),
        ],
    }
    if alias == "internvl35":
        description = (
            "Experiment 1 runs eleven conditions: the unpruned baseline plus L2 and random "
            "at 100%, 25%, 50%, 75%, and 70% retention. Both 100% conditions retain every "
            "visual token in its original order. Random uses seed 0. "
        )
        for index in (0, 2, 5):
            source = "".join(data["cells"][index]["source"])
            source = source.replace(
                "Experiment 1 runs baseline, L2 at 50% retention, and random at 50% retention. ", ""
            )
            data["cells"][index]["source"] = (source + "\n" + description + "\n").splitlines(keepends=True)
        setup = "".join(data["cells"][1]["source"])
        setup = setup.replace(
            "OUTPUT = Path('/kaggle/working/new_model_results')\n",
            "OUTPUT = Path('/kaggle/working/internvl35_retention_sweep_results')\n"
            "RETENTIONS = [1.0, 0.25, 0.50, 0.75, 0.70]\n"
            "RANDOM_SEEDS = [0]\n"
            "RUN_ARGS = ['--retentions', *map(str, RETENTIONS), "
            "'--random-seeds', *map(str, RANDOM_SEEDS)]\n"
            "OUTPUT_ARGS = ['--output-root', str(OUTPUT)]\n"
            "assert 'INTERNVL_RETENTIONS' in (CODE / 'run.py').read_text(), "
            "'Upload the updated package with the retention sweep runner.'\n"
        )
        data["cells"][1]["source"] = setup.splitlines(keepends=True)
        for index in (3, 6):
            source = "".join(data["cells"][index]["source"])
            lines = []
            for line in source.splitlines(keepends=True):
                extra = " + RUN_ARGS + OUTPUT_ARGS" if "'run.py'" in line else " + OUTPUT_ARGS"
                lines.append(line.replace("], check=True)", "]" + extra + ", check=True)"))
            data["cells"][index]["source"] = lines
    return data


def main():
    root = Path(__file__).resolve().parent
    for alias, title, filename in [("internvl35", "InternVL3.5-30B-A3B", "InternVL35_Kaggle.ipynb"),
                                    ("qwen3vl", "Qwen3-VL-30B-A3B-Instruct", "Qwen3VL_Kaggle.ipynb")]:
        data = notebook(alias, title)
        for index, entry in enumerate(data["cells"]):
            entry["id"] = f"{alias}-{index:02d}"
        (root / filename).write_text(json.dumps(data, indent=2), encoding="utf-8")
    target = root.parents[1] / "dist/kaggle_new_models.zip"
    target.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        for file in sorted(root.iterdir()):
            if file.is_file() and file.suffix in {".py", ".md", ".txt", ".sh", ".ipynb"}:
                archive.write(file, f"kaggle_new_models/{file.name}")
    print(target)


if __name__ == "__main__":
    main()
