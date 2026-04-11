#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${ROOT_DIR}"

if [[ "$(uname -s)" != "MINGW"* ]] && [[ "$(uname -s)" != "MSYS"* ]] && [[ "$(uname -s)" != "CYGWIN"* ]]; then
  echo "This script is intended to run on Windows (Git Bash/MSYS2/Cygwin)."
  exit 1
fi

if [[ ! -f "program/static/model.tflite" ]]; then
  echo "Missing program/static/model.tflite"
  exit 1
fi

PYTHON_BIN="python"
if [[ -x "./.venv/Scripts/python.exe" ]]; then
  PYTHON_BIN="./.venv/Scripts/python.exe"
fi

"${PYTHON_BIN}" -m PyInstaller \
  --noconfirm \
  --clean \
  --onefile \
  --windowed \
  --name "BABI Data Analysis Onefile" \
  --icon "program/static/program-exe-icon.ico" \
  --add-data "program/static;program/static" \
  --hidden-import "sklearn.utils._typedefs" \
  --hidden-import "sklearn.neighbors._partition_nodes" \
  --hidden-import "ai_edge_litert.interpreter" \
  --exclude-module "ipykernel" \
  --exclude-module "ipywidgets" \
  --exclude-module "jupyterlab_widgets" \
  --exclude-module "IPython" \
  --exclude-module "matplotlib" \
  --exclude-module "tensorflow" \
  --exclude-module "keras" \
  --exclude-module "tensorboard" \
  "program/app.py"
