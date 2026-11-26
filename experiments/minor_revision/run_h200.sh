#!/usr/bin/env bash
# Usage: CUDA_VISIBLE_DEVICES=<H200 index> bash run_h200.sh smoke|full|plan [extra run.py arguments]
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TMLR_PYTHON="${TMLR_PYTHON:-$SCRIPT_DIR/.venv/bin/python}"
TMLR_OUTPUT="${TMLR_OUTPUT:-$SCRIPT_DIR/outputs}"
TMLR_SCRATCH="${TMLR_SCRATCH:-$SCRIPT_DIR/model_shards}"
TMLR_MODEL="${TMLR_MODEL:-qwen35_122b}"
TMLR_PRECISION="${TMLR_PRECISION:-expert-nf4}"
PHASE="${1:-smoke}"
if [[ $# -gt 0 ]]; then shift; fi
EXTRA=()
case "$PHASE" in
  smoke) EXTRA+=(--smoke) ;;
  full) ;;
  plan) EXTRA+=(--plan-only) ;;
  *) printf 'Use smoke, full, or plan.\n' >&2; exit 2 ;;
esac
"$TMLR_PYTHON" "$SCRIPT_DIR/run.py" --model "$TMLR_MODEL" --precision "$TMLR_PRECISION" \
  --output-root "$TMLR_OUTPUT" --scratch "$TMLR_SCRATCH" "${EXTRA[@]}" "$@"
