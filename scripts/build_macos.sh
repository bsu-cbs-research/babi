#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${ROOT_DIR}"

if [[ ! -f "program/static/model.tflite" ]]; then
  echo "Missing program/static/model.tflite"
  echo "Run: python scripts/convert_model_to_tflite.py"
  exit 1
fi

PYTHON_BIN="python"
if [[ -x "./.venv/bin/python" ]]; then
  PYTHON_BIN="./.venv/bin/python"
fi

"${PYTHON_BIN}" -m PyInstaller --noconfirm "BABI Data Analysis.spec"
