#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR=/kaggle/working/compression_routing_exp1
if [ -d /kaggle/working ]; then
  FASTM_DIR=/kaggle/working/FastMMoE
else
  FASTM_DIR="${TMLR_EXTERNAL_DIR:-$SCRIPT_DIR/../../external}/FastMMoE"
fi
FASTM_COMMIT=b355b15a46601be24e440e1449df82de08e3684d
DEEPSEEK_DIR="$FASTM_DIR/DeepSeek-VL2"

if [ ! -d /kaggle/working ]; then OUTPUT_DIR="$SCRIPT_DIR/outputs_exp1"; fi
mkdir -p "$OUTPUT_DIR/manifests" "$OUTPUT_DIR/conditions" "$OUTPUT_DIR/summaries"

if [ ! -d "$FASTM_DIR/.git" ]; then
  git clone https://github.com/MindVLA-Team/FastMMoE.git "$FASTM_DIR"
fi
git -C "$FASTM_DIR" fetch --depth 1 origin "$FASTM_COMMIT"
git -C "$FASTM_DIR" checkout --detach "$FASTM_COMMIT"
git -C "$FASTM_DIR" rev-parse HEAD > "$OUTPUT_DIR/fastmmoe_commit.txt"

echo "Code: $SCRIPT_DIR"
python -m pip install -q -r "$SCRIPT_DIR/requirements-kaggle.txt"
python -m pip install -q -e "$DEEPSEEK_DIR" --no-deps

echo "Code: $SCRIPT_DIR"
echo "Outputs: $OUTPUT_DIR"
echo "FastMMoE: $FASTM_COMMIT"
