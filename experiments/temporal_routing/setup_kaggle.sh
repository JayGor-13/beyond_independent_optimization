#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR=/kaggle/working/trc_outputs
if [ -d /kaggle/working ]; then
  DEEPSEEK_DIR=/kaggle/working/DeepSeek-VL2
else
  DEEPSEEK_DIR="${TMLR_EXTERNAL_DIR:-$SCRIPT_DIR/../../external}/DeepSeek-VL2"
  OUTPUT_DIR="$SCRIPT_DIR/outputs"
fi

mkdir -p "$OUTPUT_DIR/manifests" "$OUTPUT_DIR/routes" "$OUTPUT_DIR/predictions" "$OUTPUT_DIR/summaries"

if [ ! -d "$DEEPSEEK_DIR" ]; then
  git clone --depth 1 https://github.com/deepseek-ai/DeepSeek-VL2.git "$DEEPSEEK_DIR"
fi

python -m pip install -q -r "$SCRIPT_DIR/requirements-kaggle.txt"
python -m pip install -q -e "$DEEPSEEK_DIR" --no-deps

echo "Experiment code: $SCRIPT_DIR"
echo "Writable results: $OUTPUT_DIR"
