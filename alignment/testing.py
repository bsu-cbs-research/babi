"""
alignment/testing.py

Standalone script to find the time shift between mocap and capnostream signals
via cross-correlation. Run from the project root:

    .venv/bin/python3 -m alignment.testing

True shift (from the trial schedule): Mocap lags Capno by ~360 seconds.

Approach — per-segment offset sweep (based on literature review):
  - Morelli et al. (2016): compute CCF using only valid-data intersections;
    weight by segment duration.
  - Xiao, Ding & Hu (2022): zero-phase bandpass filtering is critical; 30s
    overlap is sufficient; min-max normalization recommended.
  - Wright et al. (2025, TRACC-PHYSIO): restrict search window; peak
    correlation as confidence metric.

Key principle: do NOT concatenate non-contiguous segments with zero-padding —
this introduces spectral artifacts that dominate the correlation. Instead,
sweep candidate time offsets and for each offset, extract the corresponding
window from the continuous capnostream signal, compute normalized cross-
correlation with the motion segment, and aggregate across all segments.
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, correlate, correlation_lags
from scipy import signal as sp_signal
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import constants

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MOCAP_SR        = 100        # Hz – native mocap sample rate
CAPNO_SR        = 20         # Hz – native capnostream sample rate
TARGET_SR       = CAPNO_SR   # Hz – common rate (downsample mocap to this)
TRUE_SHIFT_S    = 360        # seconds – known from trial schedule
BREATH_LOW_HZ   = 0.1        # lower edge of respiratory bandpass
BREATH_HIGH_HZ  = 0.8        # upper edge (tightened per Wright 2025)
FILTER_ORDER    = 4
OUTPUT_DIR      = os.path.join("alignment", "output")

# Search window for offset sweep (seconds)
SEARCH_MIN_S    = -100       # how far before the true shift to search
SEARCH_MAX_S    = 900        # how far after zero to search
SEARCH_STEP_S   = 0.5        # resolution of the offset sweep

# Minimum segment length (seconds) to include in correlation
MIN_SEGMENT_S   = 10


# ---------------------------------------------------------------------------
# Signal processing helpers
# ---------------------------------------------------------------------------

def bandpass_filter(sig: np.ndarray, sr: float,
                    low: float = BREATH_LOW_HZ,
                    high: float = BREATH_HIGH_HZ,
                    order: int = FILTER_ORDER) -> np.ndarray:
    """Zero-phase Butterworth bandpass filter."""
    nyq = 0.5 * sr
    lo  = max(low / nyq, 1e-6)
    hi  = min(high / nyq, 1.0 - 1e-6)
    b, a = butter(order, [lo, hi], btype="band")
    return filtfilt(b, a, sig).astype(np.float64)


def z_score(sig: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-variance normalization."""
    std = sig.std()
    if std < 1e-12:
        return sig - sig.mean()
    return (sig - sig.mean()) / std


def downsample(sig: np.ndarray, old_sr: int, new_sr: int) -> np.ndarray:
    """Resample by an integer factor (old_sr must be a multiple of new_sr)."""
    factor = old_sr // new_sr
    if factor <= 1:
        return sig
    nyq = 0.5 * old_sr
    cutoff = 0.5 * new_sr / nyq
    b, a = butter(4, cutoff, btype="low")
    filtered = filtfilt(b, a, sig)
    return filtered[::factor]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_capnostream_waveform() -> tuple[np.ndarray, pd.Timestamp]:
    """
    Load the CO2 waveform from co2.xlsx.
    Returns (waveform_array_at_CAPNO_SR, start_timestamp).

    IMPORTANT: the available-data range contains internal gaps (rows where
    the availability flag != 0).  We must preserve these gaps as NaN and
    then interpolate, so the time axis stays consistent.  Concatenating
    only the available rows would close the gaps and shift all subsequent
    data earlier, causing a systematic offset in the alignment result.
    """
    file_path = os.path.join(constants.raw_data_dir, "package", "co2.xlsx")
    raw = pd.read_excel(file_path)

    data_start = 6
    waveform_col = raw.iloc[data_start:, 2].reset_index(drop=True)
    flag_col     = raw.iloc[data_start:, 16].reset_index(drop=True).to_numpy()
    date_col     = raw.iloc[data_start:, 0].reset_index(drop=True)
    time_col     = raw.iloc[data_start:, 1].reset_index(drop=True)

    available = np.where(flag_col == 0)[0]
    if len(available) == 0:
        raise ValueError("No available CO2 data found")

    first = available[0]
    last  = available[-1]
    start_str = f"{date_col.iloc[first]} {time_col.iloc[first]}"
    start_ts = pd.to_datetime(start_str, format="%b %d,%y %H:%M:%S")

    # Extract the FULL range from first to last available row (inclusive),
    # keeping unavailable rows as NaN so that the time axis is preserved.
    full_range = waveform_col.iloc[first:last + 1].reset_index(drop=True)
    full_flags = flag_col[first:last + 1]

    waveform_raw = full_range.map(
        lambda x: float(x) if isinstance(x, (int, float)) else np.nan
    ).to_numpy().astype(np.float64)

    # Mark unavailable rows as NaN
    unavailable_mask = full_flags != 0
    waveform_raw[unavailable_mask] = np.nan
    n_gaps = int(unavailable_mask.sum())
    if n_gaps > 0:
        print(f"    Gap rows : {n_gaps} unavailable rows within range, "
              f"interpolating ({n_gaps/CAPNO_SR:.1f}s)")

    # Linearly interpolate the gaps
    nans = np.isnan(waveform_raw)
    if nans.any():
        idx = np.arange(len(waveform_raw))
        waveform_raw[nans] = np.interp(idx[nans], idx[~nans],
                                        waveform_raw[~nans])

    return waveform_raw, start_ts


def load_mocap_segments() -> list[dict]:
    """
    Load each mocap TSV file as a separate segment.
    Returns list of dicts with keys: name, data (DataFrame), start, end.
    """
    pkg_dir = os.path.join(constants.raw_data_dir, "package")
    tsv_files = sorted([f for f in os.listdir(pkg_dir) if f.endswith(".tsv")])

    expected_cols = ['CMid Z', 'DMid Z', 'CL1 Z', 'CL2 Z',
                     'CR1 Z', 'CR2 Z', 'DL1 Z', 'DL2 Z', 'DR1 Z', 'DR2 Z']

    segments = []
    for fname in tsv_files:
        fpath = os.path.join(pkg_dir, fname)
        start = pd.to_datetime(
            pd.read_csv(fpath, sep="\t", header=6, nrows=0).columns[1]
        )
        df = pd.read_csv(
            fpath, sep="\t", header=10, usecols=[i + 2 for i in range(69)]
        ).filter(regex=r"^[CD].*Z$", axis=1)

        if list(df.columns) != expected_cols:
            print(f"  WARNING: skipping {fname}, unexpected columns")
            continue

        n = len(df)
        duration_s = (n - 1) / MOCAP_SR
        end = start + pd.to_timedelta(duration_s, unit="s")

        # Replace exact zeros with NaN (marker dropout in QTM)
        df = df.mask(df == 0)

        segments.append({
            "name": fname,
            "data": df,
            "start": start,
            "end": end,
        })

    segments.sort(key=lambda s: s["start"])
    return segments


# ---------------------------------------------------------------------------
# Prepare individual motion segments (filter, downsample)
# ---------------------------------------------------------------------------

def prepare_motion_segment(col: pd.Series) -> np.ndarray | None:
    """
    Prepare a single marker column from one motion segment:
      1. Interpolate short NaN runs (marker dropout)
      2. Mean-subtract
      3. Bandpass filter at native MOCAP_SR
      4. Downsample to TARGET_SR
      5. z-score normalize
    Returns None if the segment is too short or all NaN.
    """
    col = col.copy()
    col = col.interpolate(method="linear", limit=10)
    col = col.bfill().ffill().fillna(0.0)

    raw = col.to_numpy().astype(np.float64)
    n = len(raw)

    if n < int(MIN_SEGMENT_S * MOCAP_SR):
        return None

    raw = raw - raw.mean()

    try:
        filtered = bandpass_filter(raw, MOCAP_SR)
    except ValueError:
        return None

    # Tukey window to taper edges
    window = sp_signal.windows.tukey(n, alpha=0.1)
    filtered *= window

    ds = downsample(filtered, MOCAP_SR, TARGET_SR)
    return z_score(ds)


# ---------------------------------------------------------------------------
# Normalized cross-correlation coefficient (Pearson-based)
# ---------------------------------------------------------------------------

def ncc(a: np.ndarray, b: np.ndarray) -> float:
    """
    Normalized cross-correlation at zero lag (Pearson r) between two signals
    of the same length. Returns a value in [-1, +1].
    """
    n = len(a)
    if n < 2:
        return 0.0
    a_z = a - a.mean()
    b_z = b - b.mean()
    denom = np.sqrt(np.sum(a_z**2) * np.sum(b_z**2))
    if denom < 1e-12:
        return 0.0
    return float(np.sum(a_z * b_z) / denom)


# ---------------------------------------------------------------------------
# Per-segment offset sweep
# ---------------------------------------------------------------------------

def offset_sweep(capno: np.ndarray, capno_start: pd.Timestamp,
                 segments: list[dict], marker: str,
                 offsets_s: np.ndarray) -> np.ndarray:
    """
    For each candidate offset δ in offsets_s:
      For each motion segment:
        1. Compute where the segment falls in capno time if we subtract δ
           from each segment's timestamps.
           i.e., capno_index = (seg_start - capno_start - δ) * TARGET_SR
        2. Extract the corresponding window from capno.
        3. Compute NCC between the motion segment and the capno window.
        4. Weight the NCC by the segment length (more data = more confidence).
      Sum (or average) the weighted NCCs across all segments.

    Returns an array of aggregate NCC scores, one per candidate offset.

    Convention: a positive offset δ means the motion timestamps are AHEAD of
    the capno timestamps by δ seconds (i.e., an event at capno time T
    corresponds to the same event at motion time T + δ).
    """
    # Pre-process all motion segments for this marker
    prepared = []
    for seg in segments:
        col = seg["data"][marker]
        motion = prepare_motion_segment(col)
        if motion is None:
            continue
        seg_duration_s = len(motion) / TARGET_SR
        seg_start_s = (seg["start"] - capno_start).total_seconds()
        prepared.append({
            "motion": motion,
            "start_s": seg_start_s,
            "duration_s": seg_duration_s,
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
            # Where this segment's data falls in capno time, given offset δ
            capno_time_s = seg["start_s"] - delta
            capno_start_idx = int(round(capno_time_s * TARGET_SR))
            capno_end_idx = capno_start_idx + seg["n_samples"]

            # Clip to valid capno range
            motion_start = 0
            motion_end = seg["n_samples"]

            if capno_start_idx < 0:
                motion_start = -capno_start_idx
                capno_start_idx = 0
            if capno_end_idx > capno_len:
                motion_end -= (capno_end_idx - capno_len)
                capno_end_idx = capno_len

            overlap = motion_end - motion_start
            if overlap < int(MIN_SEGMENT_S * TARGET_SR):
                continue

            capno_window = capno[capno_start_idx:capno_end_idx]
            motion_window = seg["motion"][motion_start:motion_end]

            if len(capno_window) != len(motion_window):
                min_len = min(len(capno_window), len(motion_window))
                capno_window = capno_window[:min_len]
                motion_window = motion_window[:min_len]

            r = ncc(capno_window, motion_window)
            weight = overlap  # weight by number of overlapping samples
            weighted_sum += r * weight
            total_weight += weight

        if total_weight > 0:
            scores[i] = weighted_sum / total_weight

    return scores


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("  ALIGNMENT TESTING — Per-Segment Offset Sweep")
    print("=" * 70)
    print()
    print("  References:")
    print("    - Morelli et al. (2016): CCF with valid-data intersections")
    print("    - Xiao, Ding & Hu (2022): zero-phase bandpass, min-max norm")
    print("    - Wright et al. (2025, TRACC-PHYSIO): restricted search window")
    print()

    # ------------------------------------------------------------------
    # 1. Load capnostream waveform
    # ------------------------------------------------------------------
    print("[1] Loading capnostream waveform ...")
    capno_raw, capno_start = load_capnostream_waveform()
    capno_duration_s = len(capno_raw) / CAPNO_SR
    print(f"    Samples : {len(capno_raw)}")
    print(f"    Duration: {capno_duration_s:.1f}s ({capno_duration_s/60:.1f} min)")
    print(f"    Start   : {capno_start}")

    # Bandpass filter at native rate, then z-score
    capno_filtered = bandpass_filter(capno_raw, CAPNO_SR)
    capno_filtered = z_score(capno_filtered)
    print(f"    Filtered: bandpass {BREATH_LOW_HZ}–{BREATH_HIGH_HZ} Hz, z-scored")
    print()

    # ------------------------------------------------------------------
    # 2. Load mocap segments
    # ------------------------------------------------------------------
    print("[2] Loading mocap segments ...")
    segments = load_mocap_segments()
    for seg in segments:
        offset = (seg["start"] - capno_start).total_seconds()
        dur = seg["data"].shape[0] / MOCAP_SR
        print(f"    {seg['name']:30s}  {seg['data'].shape[0]:>6} frames  "
              f"({dur:>6.1f}s)  start={seg['start'].strftime('%H:%M:%S')}  "
              f"offset_from_capno={offset:>+8.1f}s")
    print()

    markers = segments[0]["data"].columns.tolist()

    # ------------------------------------------------------------------
    # 3. Offset sweep
    # ------------------------------------------------------------------
    offsets_s = np.arange(SEARCH_MIN_S, SEARCH_MAX_S, SEARCH_STEP_S)
    n_offsets = len(offsets_s)

    print(f"[3] Running per-segment offset sweep ...")
    print(f"    Search range : [{SEARCH_MIN_S}, {SEARCH_MAX_S}) seconds")
    print(f"    Step size    : {SEARCH_STEP_S}s  ({n_offsets} candidates)")
    print(f"    Bandpass     : {BREATH_LOW_HZ}–{BREATH_HIGH_HZ} Hz")
    print(f"    Min segment  : {MIN_SEGMENT_S}s")
    print()

    # --- Per-marker results ---
    marker_scores = {}
    marker_shifts = {}

    for marker in markers:
        scores = offset_sweep(capno_filtered, capno_start, segments, marker,
                              offsets_s)
        marker_scores[marker] = scores

        # Use argmin for negative correlation (as requested)
        peak_idx = np.argmin(scores)
        shift = offsets_s[peak_idx]
        peak_val = scores[peak_idx]
        error = shift - TRUE_SHIFT_S
        marker_shifts[marker] = shift
        status = "OK" if abs(error) < 5 else "MISS"
        print(f"    {marker:10s}  shift={shift:>8.1f}s  NCC={peak_val:>+.4f}  "
              f"error={error:>+8.1f}s  [{status}]")

    # --- Ensemble: average scores across markers, then find peak ---
    print()
    all_scores = np.array(list(marker_scores.values()))
    ensemble_scores = all_scores.mean(axis=0)

    ensemble_peak_idx = np.argmin(ensemble_scores)
    ensemble_shift = offsets_s[ensemble_peak_idx]
    ensemble_error = ensemble_shift - TRUE_SHIFT_S
    status = "OK" if abs(ensemble_error) < 5 else "MISS"
    print(f"    {'ENSEMBLE':10s}  shift={ensemble_shift:>8.1f}s  "
          f"NCC={ensemble_scores[ensemble_peak_idx]:>+.4f}  "
          f"error={ensemble_error:>+8.1f}s  [{status}]")

    # --- Median across per-marker shifts ---
    all_shifts = np.array(list(marker_shifts.values()))
    median_shift = np.median(all_shifts)
    median_error = median_shift - TRUE_SHIFT_S
    status = "OK" if abs(median_error) < 5 else "MISS"
    print(f"    {'MEDIAN':10s}  shift={median_shift:>8.1f}s  "
          f"error={median_error:>+8.1f}s  [{status}]")
    print()

    # ------------------------------------------------------------------
    # 4. Also try argmax (positive correlation) to compare
    # ------------------------------------------------------------------
    print("[4] Checking argmax (positive correlation) for comparison ...")
    for marker in markers:
        scores = marker_scores[marker]
        peak_idx_pos = np.argmax(scores)
        shift_pos = offsets_s[peak_idx_pos]
        peak_val_pos = scores[peak_idx_pos]
        error_pos = shift_pos - TRUE_SHIFT_S
        status = "OK" if abs(error_pos) < 5 else "MISS"
        print(f"    {marker:10s}  shift={shift_pos:>8.1f}s  NCC={peak_val_pos:>+.4f}  "
              f"error={error_pos:>+8.1f}s  [{status}]")

    ens_peak_pos = np.argmax(ensemble_scores)
    ens_shift_pos = offsets_s[ens_peak_pos]
    ens_error_pos = ens_shift_pos - TRUE_SHIFT_S
    status = "OK" if abs(ens_error_pos) < 5 else "MISS"
    print(f"    {'ENSEMBLE':10s}  shift={ens_shift_pos:>8.1f}s  "
          f"NCC={ensemble_scores[ens_peak_pos]:>+.4f}  "
          f"error={ens_error_pos:>+8.1f}s  [{status}]")
    print()

    # ------------------------------------------------------------------
    # 5. Also try argmax(abs) — strongest correlation regardless of sign
    # ------------------------------------------------------------------
    print("[5] Checking argmax(|NCC|) (strongest correlation, either sign) ...")
    for marker in markers:
        scores = marker_scores[marker]
        peak_idx_abs = np.argmax(np.abs(scores))
        shift_abs = offsets_s[peak_idx_abs]
        peak_val_abs = scores[peak_idx_abs]
        error_abs = shift_abs - TRUE_SHIFT_S
        status = "OK" if abs(error_abs) < 5 else "MISS"
        print(f"    {marker:10s}  shift={shift_abs:>8.1f}s  NCC={peak_val_abs:>+.4f}  "
              f"error={error_abs:>+8.1f}s  [{status}]")

    ens_peak_abs = np.argmax(np.abs(ensemble_scores))
    ens_shift_abs = offsets_s[ens_peak_abs]
    ens_error_abs = ens_shift_abs - TRUE_SHIFT_S
    status = "OK" if abs(ens_error_abs) < 5 else "MISS"
    print(f"    {'ENSEMBLE':10s}  shift={ens_shift_abs:>8.1f}s  "
          f"NCC={ensemble_scores[ens_peak_abs]:>+.4f}  "
          f"error={ens_error_abs:>+8.1f}s  [{status}]")
    print()

    # ------------------------------------------------------------------
    # 6. Determine which strategy is best
    # ------------------------------------------------------------------
    strategies = {
        "argmin (neg peak)": (ensemble_shift, ensemble_error),
        "argmax (pos peak)": (ens_shift_pos, ens_error_pos),
        "argmax(|NCC|)":     (ens_shift_abs, ens_error_abs),
    }
    best_strategy = min(strategies, key=lambda k: abs(strategies[k][1]))
    best_shift, best_error = strategies[best_strategy]

    # ------------------------------------------------------------------
    # 7. Plots
    # ------------------------------------------------------------------
    print("[6] Generating plots ...")

    # --- Per-marker NCC score curves ---
    n_markers = len(markers)
    fig, axes = plt.subplots(n_markers + 1, 1, figsize=(18, 3 * (n_markers + 1)),
                             sharex=True)

    for i, marker in enumerate(markers):
        ax = axes[i]
        ax.plot(offsets_s, marker_scores[marker], color="teal", linewidth=0.5)
        ax.axvline(x=marker_shifts[marker], color="red", linestyle="-",
                   label=f"argmin={marker_shifts[marker]:.1f}s")
        ax.axvline(x=TRUE_SHIFT_S, color="blue", linestyle="--",
                   label=f"True={TRUE_SHIFT_S}s")
        err = marker_shifts[marker] - TRUE_SHIFT_S
        ax.set_title(f"{marker}  (error={err:+.1f}s)")
        ax.set_ylabel("NCC")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

    ax = axes[-1]
    ax.plot(offsets_s, ensemble_scores, color="purple", linewidth=0.8)
    ax.axvline(x=ensemble_shift, color="red", linestyle="-",
               label=f"argmin={ensemble_shift:.1f}s")
    ax.axvline(x=ens_shift_pos, color="green", linestyle="-.",
               label=f"argmax={ens_shift_pos:.1f}s")
    ax.axvline(x=TRUE_SHIFT_S, color="blue", linestyle="--",
               label=f"True={TRUE_SHIFT_S}s")
    ax.set_title(f"ENSEMBLE  (best strategy: {best_strategy})")
    ax.set_xlabel("Offset (s)")
    ax.set_ylabel("NCC")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    corr_path = os.path.join(OUTPUT_DIR, "offset_sweep.png")
    fig.savefig(corr_path, dpi=100)
    plt.close(fig)
    print(f"    Saved: {corr_path}")

    # --- Summary bar chart ---
    fig, ax = plt.subplots(figsize=(12, 6))
    names = markers + ["ENSEMBLE"]
    shifts_list = [marker_shifts[m] for m in markers] + [best_shift]
    colors = ["green" if abs(s - TRUE_SHIFT_S) < 5 else
              ("orange" if abs(s - TRUE_SHIFT_S) < 30 else "red")
              for s in shifts_list]
    ax.barh(names, shifts_list, color=colors, alpha=0.7)
    ax.axvline(x=TRUE_SHIFT_S, color="blue", linestyle="--", linewidth=2,
               label=f"True shift ({TRUE_SHIFT_S}s)")
    ax.set_xlabel("Calculated Shift (s)")
    ax.set_title(f"Alignment Results (strategy: {best_strategy})")
    ax.legend()
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    summary_path = os.path.join(OUTPUT_DIR, "summary.png")
    fig.savefig(summary_path, dpi=100)
    plt.close(fig)
    print(f"    Saved: {summary_path}")

    # ------------------------------------------------------------------
    # 8. Final summary
    # ------------------------------------------------------------------
    print()
    print("=" * 70)
    print(f"  BEST STRATEGY       : {best_strategy}")
    print(f"  ENSEMBLE SHIFT      : {best_shift:.2f}s  (error={best_error:+.2f}s)")
    print(f"  MEDIAN (argmin)     : {median_shift:.1f}s  (error={median_error:+.1f}s)")
    print(f"  TRUE                : {TRUE_SHIFT_S}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
