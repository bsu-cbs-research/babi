#!/bin/bash
./.venv/bin/python -m PyInstaller --noconfirm --onedir --windowed --name "BABI Data Analysis" \
    --add-data "program/static:program/static" \
    --icon "program/static/program-exe-icon.ico" \
    --paths . \
    --hidden-import "sklearn.utils._typedefs" \
    --hidden-import "sklearn.neighbors._partition_nodes" \
    --exclude-module "ipykernel" \
    --exclude-module "ipywidgets" \
    --exclude-module "jupyterlab-widgets" \
    program/app.py