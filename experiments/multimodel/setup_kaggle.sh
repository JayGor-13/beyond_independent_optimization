#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python -m pip install -r "$HERE/requirements.txt"
# Kaggle's base image may contain unrelated package conflicts; import the actual
# experiment dependencies rather than making unrelated pip-check warnings fatal.
python -c 'import transformers, accelerate, bitsandbytes, datasets, torchvision, timm, cv2, pandas'
python -c 'import torch, transformers; print("Transformers:", transformers.__version__); print("CUDA GPUs:", torch.cuda.device_count()); print("Use a fresh kernel after installation.")'
