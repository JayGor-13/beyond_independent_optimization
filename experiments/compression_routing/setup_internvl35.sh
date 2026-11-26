#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR=/kaggle/working/compression_routing_exp1_internvl35

mkdir -p "$OUTPUT_DIR/manifests" "$OUTPUT_DIR/conditions" "$OUTPUT_DIR/summaries"
python -m pip install -q -r "$SCRIPT_DIR/requirements-internvl35.txt"

echo "Code: $SCRIPT_DIR"
echo "InternVL3.5 outputs: $OUTPUT_DIR"
echo "Restart the notebook kernel if an older transformers release was already imported."
