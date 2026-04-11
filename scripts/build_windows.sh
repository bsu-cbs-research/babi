#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${ROOT_DIR}"

if [[ "$(uname -s)" != "MINGW"* ]] && [[ "$(uname -s)" != "MSYS"* ]] && [[ "$(uname -s)" != "CYGWIN"* ]]; then
  echo "This script is intended to run on Windows (Git Bash/MSYS2/Cygwin)."
  echo "Use GitHub Actions workflow for cross-platform artifact builds from macOS."
  exit 1
fi

if [[ ! -f "program/static/model.tflite" ]]; then
  echo "Missing program/static/model.tflite"
  echo "Run: python scripts/convert_model_to_tflite.py"
  exit 1
fi

PYTHON_BIN="python"
if [[ -x "./.venv/Scripts/python.exe" ]]; then
  PYTHON_BIN="./.venv/Scripts/python.exe"
fi

"${PYTHON_BIN}" -m PyInstaller --noconfirm "BABI Data Analysis.spec"
