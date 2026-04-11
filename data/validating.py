"""Structural and schema validation for capnostream and motion data files.

Errors raised here are surfaced verbatim in the UI, so messages should be
specific and actionable.
"""
from __future__ import annotations

import os
from datetime import datetime

import pandas as pd

from globals import constants

# --- capnostream (.xlsx) ---------------------------------------------------

# Required header cells in row 4 (0-indexed row 3) of the raw xlsx. These are
# the columns the downstream parser/pipeline actually depends on.
_CAPNOSTREAM_REQUIRED_HEADERS = ("Date", "Time", "CO\u2082 Wave")

# The metadata cell at row 3 (0-indexed [2, 1]) must parse as this format.
_CAPNOSTREAM_START_TIME_FORMAT = "%b %d,%y  %I:%M:%S %p"


def capnostream(path: str) -> bool:
    """Validate a single raw capnostream xlsx file.

    Raises ValueError with a specific message on any failure. Returns True
    when everything looks consistent with what ``data.parsing.capnostream``
    expects.
    """
    if not path:
        raise ValueError("Capnostream path is empty.")
    if not os.path.isfile(path):
        raise ValueError(f"Capnostream file not found: {path}")
    if os.path.splitext(path)[1].lower() != ".xlsx":
        raise ValueError(
            f"Capnostream file must be an .xlsx, got {os.path.basename(path)}"
        )

    # Mirror the parser's read call (default header -> first row becomes the
    # column names) so our row indices line up with data.parsing.capnostream.
    try:
        raw = pd.read_excel(path)
    except Exception as e:
        raise ValueError(f"Could not read capnostream xlsx: {e}") from e

    if raw.shape[0] < 5:
        raise ValueError(
            f"Capnostream xlsx has only {raw.shape[0]} data rows; expected at "
            "least 5 (metadata + header + one data row)."
        )
    if raw.shape[1] < 3:
        raise ValueError(
            f"Capnostream xlsx has only {raw.shape[1]} columns; expected at least 3."
        )

    # Cell at iloc[2, 1] (post default-header read) holds "Report Generation
    # Time", which the parser uses as the start time for the synthetic 50 ms
    # index.
    start_cell = raw.iat[2, 1]
    if pd.isna(start_cell):
        raise ValueError(
            "Capnostream xlsx Report Generation Time cell is empty; "
            "cannot determine start time."
        )
    try:
        datetime.strptime(str(start_cell), _CAPNOSTREAM_START_TIME_FORMAT)
    except ValueError as e:
        raise ValueError(
            f"Capnostream start time cell = {start_cell!r} does not match "
            f"expected format '{_CAPNOSTREAM_START_TIME_FORMAT}': {e}"
        ) from e

    # Row at iloc[3, :] holds the real column headers.
    header_row = [str(v).strip() if not pd.isna(v) else "" for v in raw.iloc[3, :]]
    missing = [h for h in _CAPNOSTREAM_REQUIRED_HEADERS if h not in header_row]
    if missing:
        raise ValueError(
            "Capnostream xlsx header row is missing required columns: "
            f"{missing}. Found: {header_row[:6]}..."
        )

    # Must have at least one data row past the metadata/header block
    # (rows 0-4 are metadata/header/units in the parser's coordinate system).
    if raw.shape[0] <= 5:
        raise ValueError("Capnostream xlsx contains no data rows.")

    return True


# --- motion (.tsv) ---------------------------------------------------------

_MOTION_EXPECTED_KEYS = (
    "FILE_VERSION",
    "NO_OF_FRAMES",
    "NO_OF_CAMERAS",
    "NO_OF_MARKERS",
    "FREQUENCY",
    "DESCRIPTION",
    "TIME_STAMP",
    "DATA_INCLUDED",
    "MARKER_NAMES",
    "TRAJECTORY_TYPES",
)

_MOTION_TIMESTAMP_FORMAT = "%Y-%m-%d, %H:%M:%S.%f"


def _read_motion_header(path: str) -> dict[str, list[str]]:
    """Read only the first 11 lines of a motion tsv and return the header kv map.

    Layout (0-indexed):
      rows 0..9  -> metadata KEY\tvalue[s]
      row 10     -> data column header row ("Frame\tTime\t<marker cols>")

    Does NOT read the full file (which can be tens of thousands of rows).
    """
    header: dict[str, list[str]] = {}
    data_header_cols: list[str] = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for i, raw_line in enumerate(f):
            if i >= 11:
                break
            line = raw_line.rstrip("\n").rstrip("\r")
            parts = line.split("\t")
            if i < 10:
                # metadata rows are KEY\tval[\tval...]
                if not parts or not parts[0]:
                    continue
                header[parts[0]] = parts[1:]
            elif i == 10:
                data_header_cols = parts
    header["__DATA_HEADER__"] = data_header_cols
    return header


def _validate_single_motion_file(path: str) -> tuple[tuple[str, ...], float, datetime]:
    """Validate one motion tsv and return (marker_names, frequency, start_time)."""
    if not os.path.isfile(path):
        raise ValueError(f"Motion file not found: {path}")
    if os.path.splitext(path)[1].lower() != ".tsv":
        raise ValueError(
            f"Motion file must be a .tsv, got {os.path.basename(path)}"
        )
    if os.path.getsize(path) == 0:
        raise ValueError(f"Motion file is empty: {os.path.basename(path)}")

    try:
        header = _read_motion_header(path)
    except Exception as e:
        raise ValueError(
            f"Could not read motion file header ({os.path.basename(path)}): {e}"
        ) from e

    missing = [k for k in _MOTION_EXPECTED_KEYS if k not in header]
    if missing:
        raise ValueError(
            f"Motion file {os.path.basename(path)} is missing header keys: {missing}"
        )

    # FREQUENCY must match our expected motion sampling rate.
    freq_vals = header["FREQUENCY"]
    if not freq_vals or not freq_vals[0]:
        raise ValueError(
            f"Motion file {os.path.basename(path)} has empty FREQUENCY field."
        )
    try:
        frequency = float(freq_vals[0])
    except ValueError as e:
        raise ValueError(
            f"Motion file {os.path.basename(path)} FREQUENCY is not numeric: "
            f"{freq_vals[0]!r}"
        ) from e
    if int(frequency) != int(constants.motion_sr):
        raise ValueError(
            f"Motion file {os.path.basename(path)} FREQUENCY={int(frequency)} Hz "
            f"does not match expected {int(constants.motion_sr)} Hz."
        )

    # TIME_STAMP parseable.
    ts_vals = header["TIME_STAMP"]
    if not ts_vals or not ts_vals[0]:
        raise ValueError(
            f"Motion file {os.path.basename(path)} has empty TIME_STAMP field."
        )
    try:
        start_time = datetime.strptime(ts_vals[0], _MOTION_TIMESTAMP_FORMAT)
    except ValueError as e:
        raise ValueError(
            f"Motion file {os.path.basename(path)} TIME_STAMP "
            f"{ts_vals[0]!r} does not match '{_MOTION_TIMESTAMP_FORMAT}': {e}"
        ) from e

    # MARKER_NAMES non-empty.
    marker_names = tuple(m for m in header["MARKER_NAMES"] if m)
    if not marker_names:
        raise ValueError(
            f"Motion file {os.path.basename(path)} has empty MARKER_NAMES list."
        )

    # Data header row (row 12) should at minimum start with Frame and Time.
    data_cols = header.get("__DATA_HEADER__") or []
    if len(data_cols) < 2 or data_cols[0] != "Frame" or data_cols[1] != "Time":
        raise ValueError(
            f"Motion file {os.path.basename(path)} data header row does not "
            f"start with 'Frame\\tTime'; found: {data_cols[:4]}"
        )

    return marker_names, frequency, start_time


def motion(paths: set[str]) -> bool:
    """Validate a set of motion tsv files.

    Checks each file structurally, then verifies that all files share the same
    MARKER_NAMES (so the composite dataframe columns line up).
    """
    if not paths:
        raise ValueError("No motion .tsv files were provided.")

    per_file: list[tuple[str, tuple[str, ...]]] = []
    for path in sorted(paths):
        marker_names, _freq, _start = _validate_single_motion_file(path)
        per_file.append((path, marker_names))

    # All files must share the same marker set (order matters for composite).
    reference_path, reference_markers = per_file[0]
    for path, markers in per_file[1:]:
        if markers != reference_markers:
            raise ValueError(
                f"Motion file {os.path.basename(path)} has different MARKER_NAMES "
                f"than {os.path.basename(reference_path)}; all motion files in a "
                "folder must share the same marker schema."
            )

    return True
