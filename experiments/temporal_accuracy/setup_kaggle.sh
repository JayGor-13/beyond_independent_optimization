#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_ROOT=/kaggle/working/experiment_3_runtime
DEEPSEEK_DIR="$WORK_ROOT/DeepSeek-VL2"

mkdir -p "$WORK_ROOT" /kaggle/working/trc_accuracy_outputs

python -m pip install -q -r "$SCRIPT_DIR/requirements-kaggle.txt"

if [ ! -d "$DEEPSEEK_DIR" ]; then
  UPLOADED_DIR="$(find /kaggle/input -type d -name 'DeepSeek-VL2' -print -quit 2>/dev/null || true)"
  if [ -n "$UPLOADED_DIR" ]; then
    echo "Using uploaded DeepSeek-VL2 source: $UPLOADED_DIR"
    cp -R "$UPLOADED_DIR" "$DEEPSEEK_DIR"
  else
    UPLOADED_ZIP="$(find /kaggle/input -type f \( -iname '*deepseek*vl2*.zip' -o -iname '*deepseek-vl2*.tar.gz' \) -print -quit 2>/dev/null || true)"
    if [ -n "$UPLOADED_ZIP" ]; then
      echo "Extracting uploaded DeepSeek-VL2 source: $UPLOADED_ZIP"
      mkdir -p "$DEEPSEEK_DIR"
      case "$UPLOADED_ZIP" in
        *.zip) unzip -q "$UPLOADED_ZIP" -d "$WORK_ROOT" ;;
        *.tar.gz) tar -xzf "$UPLOADED_ZIP" -C "$WORK_ROOT" ;;
      esac
      FOUND_SETUP="$(find "$WORK_ROOT" -type f -name 'setup.py' -path '*DeepSeek*VL2*' -print -quit)"
      if [ -n "$FOUND_SETUP" ]; then
        FOUND_DIR="$(dirname "$FOUND_SETUP")"
        if [ "$FOUND_DIR" != "$DEEPSEEK_DIR" ]; then
          cp -R "$FOUND_DIR"/. "$DEEPSEEK_DIR"/
        fi
      fi
    else
      echo "No uploaded DeepSeek-VL2 source found; cloning the official repository."
      git clone --depth 1 https://github.com/deepseek-ai/DeepSeek-VL2.git "$DEEPSEEK_DIR"
    fi
  fi
fi

if [ ! -f "$DEEPSEEK_DIR/setup.py" ] && [ ! -f "$DEEPSEEK_DIR/pyproject.toml" ]; then
  echo "DeepSeek-VL2 package files were not found under $DEEPSEEK_DIR" >&2
  exit 1
fi

python -m pip install -q "$DEEPSEEK_DIR" --no-deps
python -m unittest discover -s "$SCRIPT_DIR" -p 'test_*.py'

echo "Experiment code: $SCRIPT_DIR"
echo "Writable results: /kaggle/working/trc_accuracy_outputs"
echo "Setup complete for a single Kaggle T4."
