#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TMLR_PYTHON="${TMLR_PYTHON:-python3.12}"
TMLR_VENV="${TMLR_VENV:-$SCRIPT_DIR/.venv}"
"$TMLR_PYTHON" -c 'import sys; assert sys.version_info >= (3, 12), "Use Python 3.12 or newer"'
"$TMLR_PYTHON" -m venv "$TMLR_VENV"
"$TMLR_VENV/bin/python" -m pip install --upgrade pip
"$TMLR_VENV/bin/python" -m pip install -r "$SCRIPT_DIR/requirements.txt"
"$TMLR_VENV/bin/python" -m pytest "$SCRIPT_DIR/test_experiments.py" -q
"$TMLR_VENV/bin/python" -c 'import torch, transformers; print("torch", torch.__version__, "transformers", transformers.__version__); print("CUDA available:", torch.cuda.is_available()); print("GPUs:", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])'
printf '\nEnvironment ready: %s\nRead README.md, then run a smoke experiment before a full experiment.\n' "$TMLR_VENV"
