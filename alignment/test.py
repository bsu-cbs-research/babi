"""
alignment/test.py

Runs ensemble time-alignment over each package in data/testing/ and prints
the difference between the calculated shift and the true shift (shift.txt).

Run from the project root:

    .venv/bin/python3 -m alignment.test
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MOCAP_SR       = 100
CAPNO_SR       = 20
TARGET_SR      = CAPNO_SR
BREATH_LOW_HZ  = 0.1
BREATH_HIGH_HZ = 0.8
FILTER_ORDER   = 4
SEARCH_MIN_S   = -100
SEARCH_MAX_S   = 900
SEARCH_STEP_S  = 0.5
MIN_SEGMENT_S  = 10

TESTING_DIR    = os.path.join("data", "testing")


# ---------------------------------------------------------------------------
# Signal processing helpers
# ---------------------------------------------------------------------------

def bandpass_filter(sig: np.ndarray, sr: float) -> np.ndarray:
    nyq = 0.5 * sr
    lo  = max(BREATH_LOW_HZ / nyq, 1e-6)
    hi  = min(BREATH_HIGH_HZ / nyq, 1.0 - 1e-6)
    b, a = butter(FILTER_ORDER, [lo, hi], btype="band")
    return filtfilt(b, a, sig).astype(np.float64)


def z_score(sig: np.ndarray) -> np.ndarray:
    std = sig.std()
    if std < 1e-12:
        return sig - sig.mean()
    return (sig - sig.mean()) / std


def downsample(sig: np.ndarray, old_sr: int, new_sr: int) -> np.ndarray:
    factor = old_sr // new_sr
    if factor <= 1:
        return sig
    nyq = 0.5 * old_sr
    cutoff = 0.5 * new_sr / nyq
    b, a = butter(4, cutoff, btype="low")
    return filtfilt(b, a, sig)[::factor]


# ---------------------------------------------------------------------------
# Data loading (package-relative paths)
# ---------------------------------------------------------------------------

def load_capnostream(pkg_dir: str) -> tuple[np.ndarray, pd.Timestamp]:
    file_path = os.path.join(pkg_dir, "co2.xlsx")
    raw = pd.read_excel(file_path)

    data_start = 6
    waveform_col = raw.iloc[data_start:, 2].reset_index(drop=True)
    flag_col     = raw.iloc[data_start:, 16].reset_index(drop=True).to_numpy()
    date_col     = raw.iloc[data_start:, 0].reset_index(drop=True)
    time_col     = raw.iloc[data_start:, 1].reset_index(drop=True)

    available = np.where(flag_col == 0)[0]
    if len(available) == 0:
        raise ValueError(f"No available CO2 data in {pkg_dir}")

    first, last = available[0], available[-1]
    start_ts = pd.to_datetime(
        f"{date_col.iloc[first]} {time_col.iloc[first]}",
        format="%b %d,%y %H:%M:%S"
    )

    full_range = waveform_col.iloc[first:last + 1].reset_index(drop=True)
    full_flags = flag_col[first:last + 1]

    waveform = full_range.map(
        lambda x: float(x) if isinstance(x, (int, float)) else np.nan
    ).to_numpy().astype(np.float64)

    waveform[full_flags != 0] = np.nan

    nans = np.isnan(waveform)
    if nans.any():
        idx = np.arange(len(waveform))
        waveform[nans] = np.interp(idx[nans], idx[~nans], waveform[~nans])

    return waveform, start_ts


def load_mocap_segments(pkg_dir: str) -> list[dict]:
    expected_cols = [
        'CMid Z', 'DMid Z',
        'CL1 Z', 'CL2 Z', 'CR1 Z', 'CR2 Z',
        'DL1 Z', 'DL2 Z', 'DR1 Z', 'DR2 Z',
    ]

    segments = []
    for fname in sorted(os.listdir(pkg_dir)):
        if not fname.endswith(".tsv"):
            continue
        fpath = os.path.join(pkg_dir, fname)
        start = pd.to_datetime(
            pd.read_csv(fpath, sep="\t", header=6, nrows=0).columns[1]
        )
        df = pd.read_csv(
            fpath, sep="\t", header=10, usecols=[i + 2 for i in range(69)]
        ).filter(regex=r"^[CD].*Z$", axis=1)

        if list(df.columns) != expected_cols:
            continue

        n = len(df)
        end = start + pd.to_timedelta((n - 1) / MOCAP_SR, unit="s")
        segments.append({
            "data": df.mask(df == 0),
            "start": start,
            "end": end,
        })

    segments.sort(key=lambda s: s["start"])
    return segments


# ---------------------------------------------------------------------------
# Prepare motion column
# ---------------------------------------------------------------------------

def prepare_motion(col: pd.Series) -> np.ndarray | None:
    col = col.copy().interpolate(method="linear", limit=10).bfill().ffill().fillna(0.0)
    raw = col.to_numpy().astype(np.float64)
    if len(raw) < int(MIN_SEGMENT_S * MOCAP_SR):
        return None
    raw -= raw.mean()
    try:
        filtered = bandpass_filter(raw, MOCAP_SR)
    except ValueError:
        return None
    from scipy.signal import windows
    filtered *= windows.tukey(len(raw), alpha=0.1)
    return z_score(downsample(filtered, MOCAP_SR, TARGET_SR))


# ---------------------------------------------------------------------------
# NCC and offset sweep
# ---------------------------------------------------------------------------

def ncc(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2:
        return 0.0
    a_z, b_z = a - a.mean(), b - b.mean()
    denom = np.sqrt(np.sum(a_z ** 2) * np.sum(b_z ** 2))
    if denom < 1e-12:
        return 0.0
    return float(np.sum(a_z * b_z) / denom)


def offset_sweep(capno: np.ndarray, capno_start: pd.Timestamp,
                 segments: list[dict], marker: str,
                 offsets_s: np.ndarray) -> np.ndarray:
    prepared = []
    for seg in segments:
        motion = prepare_motion(seg["data"][marker])
        if motion is None:
            continue
        prepared.append({
            "motion": motion,
            "start_s": (seg["start"] - capno_start).total_seconds(),
            "n_samples": len(motion),
        })

    if not prepared:
        return np.zeros(len(offsets_s))

    capno_len = len(capno)
    scores = np.zeros(len(offsets_s))

    for i, delta in enumerate(offsets_s):
        weighted_sum = 0.0
        total_weight = 0.0
        for seg in prepared:
            c0 = int(round((seg["start_s"] - delta) * TARGET_SR))
            c1 = c0 + seg["n_samples"]
            m0, m1 = 0, seg["n_samples"]

            if c0 < 0:
                m0 -= c0
                c0 = 0
            if c1 > capno_len:
                m1 -= (c1 - capno_len)
                c1 = capno_len

            overlap = m1 - m0
            if overlap < int(MIN_SEGMENT_S * TARGET_SR):
                continue

            cw = capno[c0:c1]
            mw = seg["motion"][m0:m1]
            n = min(len(cw), len(mw))
            r = ncc(cw[:n], mw[:n])
            weighted_sum += r * n
            total_weight += n

        if total_weight > 0:
            scores[i] = weighted_sum / total_weight

    return scores


# ---------------------------------------------------------------------------
# Align a single package — returns (calculated_shift, true_shift, error)
# ---------------------------------------------------------------------------

def align_package(pkg_dir: str) -> tuple[float, float, float]:
    shift_path = os.path.join(pkg_dir, "shift.txt")
    true_shift = float(open(shift_path).read().strip())

    capno_raw, capno_start = load_capnostream(pkg_dir)
    capno = z_score(bandpass_filter(capno_raw, CAPNO_SR))

    segments = load_mocap_segments(pkg_dir)
    if not segments:
        raise ValueError(f"No valid motion segments found in {pkg_dir}")

    markers = segments[0]["data"].columns.tolist()
    offsets_s = np.arange(SEARCH_MIN_S, SEARCH_MAX_S, SEARCH_STEP_S)

    # Per-marker sweep, then average across markers (ensemble)
    all_scores = np.array([
        offset_sweep(capno, capno_start, segments, marker, offsets_s)
        for marker in markers
    ])
    ensemble_scores = all_scores.mean(axis=0)

    # Best offset = strongest absolute correlation
    best_idx = np.argmax(np.abs(ensemble_scores))
    calculated_shift = offsets_s[best_idx]
    error = calculated_shift - true_shift

    return calculated_shift, true_shift, error


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    packages = sorted([
        d for d in os.listdir(TESTING_DIR)
        if os.path.isdir(os.path.join(TESTING_DIR, d))
    ])

    print(f"{'Package':<12}  {'Calculated':>12}  {'True':>8}  {'Error':>8}  {'Status'}")
    print("-" * 56)

    for pkg in packages:
        pkg_dir = os.path.join(TESTING_DIR, pkg)
        try:
            calc, true, error = align_package(pkg_dir)
            status = "OK" if abs(error) < 5 else "MISS"
            print(f"{pkg:<12}  {calc:>11.1f}s  {true:>7.1f}s  {error:>+7.1f}s  {status}")
        except Exception as e:
            print(f"{pkg:<12}  ERROR: {e}")


if __name__ == "__main__":
    main()
