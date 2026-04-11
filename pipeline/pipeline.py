"""End-to-end data pipeline: validate -> parse -> label -> align.

This module is consumed both by the CLI (``python -m pipeline.pipeline``) and
by the GUI (``program/screens/configuration.py``). The GUI passes a
``status_cb`` to surface progress messages and a ``static_dir`` pointing at
the bundled model artifacts so it works inside the PyInstaller app.
"""
from __future__ import annotations

import os
from typing import Callable, Optional

import pandas as pd

import labeling
import alignment
import data

# Column name the labeling step operates on. Lookup by name (not positional
# index) to avoid breakage if the parser's column order changes.
CO2_COLUMN = "CO\u2082 Wave"


StatusCallback = Optional[Callable[[str], None]]


def _noop(_msg: str) -> None:
    pass


def _discover_files(pkg: str) -> tuple[str, set[str]]:
    """Find the single xlsx and the set of tsv files inside ``pkg``."""
    if not pkg or not os.path.isdir(pkg):
        raise ValueError(f"Selected path is not a directory: {pkg}")

    entries = os.listdir(pkg)
    xlsx_files = [f for f in entries if f.lower().endswith(".xlsx")]
    tsv_files = [f for f in entries if f.lower().endswith(".tsv")]

    if len(xlsx_files) == 0:
        raise ValueError(
            f"No capnostream .xlsx file found in {pkg}. "
            "Expected exactly one .xlsx file."
        )
    if len(xlsx_files) > 1:
        raise ValueError(
            f"Expected exactly one .xlsx file in {pkg}, found {len(xlsx_files)}: "
            f"{xlsx_files}"
        )
    if len(tsv_files) == 0:
        raise ValueError(
            f"No motion .tsv files found in {pkg}. Expected at least one."
        )

    capnostream_path = os.path.join(pkg, xlsx_files[0])
    motion_paths = {os.path.join(pkg, f) for f in tsv_files}
    return capnostream_path, motion_paths


def execute(
    pkg: str,
    offset: float = 0,
    *,
    status_cb: StatusCallback = None,
    static_dir: Optional[str] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the full pipeline against the package folder ``pkg``.

    Parameters
    ----------
    pkg: path to a folder containing exactly one .xlsx (capnostream) and one
        or more .tsv files (motion data).
    offset: seconds to shift the capnostream time axis before aligning onto
        the motion DatetimeIndex.
    status_cb: optional callable invoked with human-readable status strings
        as each stage starts. The GUI uses this to update its progress label.
    static_dir: directory containing the labeling model artifacts. Defaults
        to ``program/static`` relative to cwd for CLI use.

    Returns
    -------
    (motion_df, capnostream_df): the aligned motion frame (with a
    ``co2_valid`` column broadcast onto its DatetimeIndex) and the labeled
    capnostream frame.
    """
    emit = status_cb or _noop

    # --- discovery ---
    emit("Discovering files...")
    capnostream_path, motion_paths = _discover_files(pkg)

    # --- validation ---
    emit("Validating capnostream...")
    data.validating.capnostream(capnostream_path)
    emit("Validating motion files...")
    data.validating.motion(motion_paths)

    # --- parsing ---
    emit("Parsing capnostream...")
    capnostream_data = data.parsing.capnostream(capnostream_path)
    if capnostream_data is None or capnostream_data.empty:
        raise ValueError("Parsed capnostream dataframe is empty.")
    if CO2_COLUMN not in capnostream_data.columns:
        raise ValueError(
            f"Parsed capnostream dataframe is missing expected column "
            f"{CO2_COLUMN!r}. Got: {list(capnostream_data.columns)[:6]}..."
        )

    emit("Parsing motion files...")
    motion_data = data.parsing.motion(motion_paths)
    if motion_data is None or motion_data.empty:
        raise ValueError("Parsed motion dataframe is empty.")

    # --- labeling ---
    emit("Labeling CO\u2082 waveform...")
    signal = capnostream_data[CO2_COLUMN].to_numpy()
    labels = labeling.label(signal, static_dir=static_dir)
    if len(labels) != len(capnostream_data):
        raise ValueError(
            f"Labeling returned {len(labels)} labels for a signal of length "
            f"{len(capnostream_data)}."
        )
    capnostream_data["co2_valid"] = labels

    # --- alignment ---
    emit("Aligning signals...")
    motion, capnostream = alignment.static(capnostream_data, motion_data, offset)

    return motion, capnostream


if __name__ == "__main__":
    def _print_status(msg: str) -> None:
        print(f"[pipeline] {msg}")

    execute(
        os.path.join("data", "testing", "p1"),
        offset=360,
        status_cb=_print_status,
    )
