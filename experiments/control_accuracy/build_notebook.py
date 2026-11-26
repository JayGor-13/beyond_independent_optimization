"""Build a self-contained Kaggle notebook and optional source upload ZIP."""
import base64
import csv
import io
import json
import shutil
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def cell(kind, content):
    result = {"cell_type": kind, "metadata": {}, "source": content.splitlines(keepends=True)}
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


def main():
    shutil.copy2(ROOT / "experiments/temporal_routing/run.py", HERE / "original_trc_run.py")
    source = ROOT / "results/published/deepseek/temporal_routing/trc_per_clip.csv"
    with source.open(newline="", encoding="utf-8") as handle:
        rows = [r for r in csv.DictReader(handle) if r["condition"] == "chronological"]
    assert len(rows) == 256 and sum(r["correct"] == "True" for r in rows) == 162
    with (HERE / "reference_chronological.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["id", "video_id", "question_type", "category", "correct"]
        writer = csv.DictWriter(handle, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    (ROOT / "dist").mkdir(exist_ok=True)
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in ["run.py", "core.py", "original_trc_run.py", "reference_chronological.csv", "requirements-kaggle.txt", "test_core.py", "setup_runtime.py"]:
            archive.write(HERE/name, name)
    encoded = base64.b64encode(payload.getvalue()).decode()
    bootstrap = f'''from pathlib import Path
import base64, io, json, os, subprocess, sys, zipfile
CODE = Path('/kaggle/working/deepseek_accuracy_code')
CODE.mkdir(parents=True, exist_ok=True)
PAYLOAD = {encoded!r}
with zipfile.ZipFile(io.BytesIO(base64.b64decode(PAYLOAD))) as archive:
    archive.extractall(CODE)
os.environ.update(USE_TORCH='1', USE_TF='0', USE_FLAX='0', TOKENIZERS_PARALLELISM='false')
subprocess.run([sys.executable, '-u', str(CODE/'setup_runtime.py')], check=True)
PYTHON = '/kaggle/working/deepseek_accuracy_py311/bin/python'
'''
    config = '''# Use 2 for a smoke test, 32 for a pilot, 256 for paper results.
SAMPLES = 32
OUTPUT = Path('/kaggle/working/deepseek_control_accuracy')
MODEL = 'deepseek-ai/deepseek-vl2-tiny'  # Or the original uploaded checkpoint path.
REVISION = None  # Set to the original checkpoint revision if available.
ORIGINAL_MANIFEST = None  # Set to the original nextqa_256_seed20260819.json if available.
args = [PYTHON, '-u', str(CODE/'run.py'), '--limit', str(SAMPLES),
        '--model', MODEL, '--output-dir', str(OUTPUT)]
if REVISION:
    args += ['--revision', REVISION]
if ORIGINAL_MANIFEST:
    args += ['--manifest', str(ORIGINAL_MANIFEST)]
'''
    nb = {"nbformat": 4, "nbformat_minor": 5,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"},
                       "kaggle": {"isInternetEnabled": True, "isGpuEnabled": True}},
          "cells": [cell("markdown", "# DeepSeek NExT-QA: missing control accuracies\n\n"
                         "Version: 2026-10-13-py311-arrow17-fix. Enable a GPU and Internet in Kaggle settings. This notebook embeds its code and reference IDs; no code dataset upload is needed. The setup creates a separate Python 3.11 environment because the original Transformers/tokenizers stack does not provide Python 3.13 wheels. NumPy 1.26.4 and PyArrow 17.0.0 are pinned together. The notebook kernel is unchanged. Initial setup downloads CUDA PyTorch wheels and can take several minutes. It generates answers only, without routing capture.\n\n"
                         "Start with `SAMPLES=2` or `32`; use `256` to fill the original paper table. Keep the same output directory to resume and reuse completed answers. The original 256-item donor ring is used even in a pilot.\n\n"
                         "Attach the original Experiment 2 manifest/checkpoint if available. Otherwise the notebook reconstructs the seeded manifest and verifies all question/video IDs against the saved TRC CSV. Reconstruction cannot independently verify the historical cross-clip donor order or checkpoint revision. Review those before combining results with the old TRC table.\n"),
                    cell("code", bootstrap), cell("code", config),
                    cell("code", "subprocess.run(args, check=True)\n"),
                    cell("code", "import pandas as pd\ndisplay(pd.read_csv(OUTPUT/'condition_accuracy.csv'))\nprint((OUTPUT/'checks.json').read_text())\n"),
                    cell("markdown", "## Download outputs\nThe ZIP contains per-condition answers, prompts, frame indices, donor IDs, runtime provenance and summary/check files. A pilot is exploratory; do not insert its accuracies into an n=256 table. Router permutation remains n/a because it changes routing labels only.\n"),
                    cell("code", "import shutil\narchive = shutil.make_archive('/kaggle/working/deepseek_control_accuracy_results', 'zip', OUTPUT)\nprint(archive)\nfrom IPython.display import FileLink\ndisplay(FileLink(archive))\n")]}
    target = HERE / "DeepSeek_Control_Accuracy_Kaggle.ipynb"
    target.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    with zipfile.ZipFile(ROOT / "dist/DeepSeek_Control_Accuracy_Kaggle.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(HERE.glob("*")):
            if path.is_file() and path.suffix in (".py", ".txt", ".md", ".csv", ".ipynb"):
                archive.write(path, "experiment_2_accuracy_only/"+path.name)
    print(f"Built {target}; optional source ZIP in dist/.")


if __name__ == "__main__":
    main()
