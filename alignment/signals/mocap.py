from typing import cast
import os
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
from data import constants

sr = 100
expected_cols = ['CMid Z', 'DMid Z', 'CL1 Z', 'CL2 Z', 'CR1 Z', 'CR2 Z', 'DL1 Z', 'DL2 Z', 'DR1 Z', 'DR2 Z']
n_expected_markers = len(expected_cols)

def _parse_file(name: str):
    file_path = os.path.normpath(os.path.join(constants.raw_data_dir, "package", name))
    start = pd.to_datetime(pd.read_csv(file_path, sep="\t", header=6, nrows=0).columns[1])
    df = pd.read_csv(file_path, sep="\t", header=10, usecols=[i+2 for i in range(69)]).filter(regex='^[CD].*Z$', axis=1)

    if df.shape[1] != n_expected_markers:
        raise ValueError(f"Expected {n_expected_markers} motion markers, but found {df.shape[1]} in file {name})")
    
    if list(df.columns) != expected_cols:
        raise ValueError(f"Expected column order {expected_cols}, but found {list(df.columns)} in file {name})")

    duration_seconds = (len(df) - 1) / sr
    end = start + pd.to_timedelta(duration_seconds, unit='s')
    df.index = pd.date_range(start=start, end=end, periods=len(df))

    df.attrs = {"start": start, "end": end}
    return df.mask(df == 0)

def normalize_signal(col: pd.Series) -> pd.Series:
    """Centers and scales the signal, ignoring NaNs in the stats."""
    return (col - col.mean()) / col.std()

# NOTE: CRZ1-Z is perfect at cutoff=2, order=4
def highpass_filter(col: pd.Series, cutoff: float = 2, order: int = 4) -> pd.Series:
    """Applies a zero-phase high-pass filter while preserving NaN locations."""
    nyquist = 0.5 * sr
    normal_cutoff = cutoff / nyquist
    b, a = cast(tuple[np.ndarray, np.ndarray], butter(order, normal_cutoff, btype='high', analog=False))
    mask = col.isna()
    filled_sig = col.fillna(0).values
    filtered_values = filtfilt(b, a, filled_sig)
    result = pd.Series(filtered_values, index=col.index)
    result[mask] = np.nan
    return result

def from_package(dir: str):
    motion_paths = [n for n in os.listdir(dir) if n.endswith(".tsv")]
    raw_motions = sorted([_parse_file(n) for n in motion_paths], key=lambda x: x.attrs["end"])
    df = pd.concat(raw_motions).sort_index()
    df = df.groupby(level=0).mean() # CRITICAL: Removes duplicate timestamps

    full_index = pd.date_range(
        start=df.index.min(), 
        end=df.index.max(), 
        freq="10ms"
    )

    df = df.reindex(full_index, method='nearest', tolerance=pd.Timedelta(milliseconds=5))

    def process_column(col: pd.Series):
        col = highpass_filter(col - col.mean())
        mu, sd = col.mean(), col.std()
        lower_limit = mu - (sd * 2)
        upper_limit = mu + (sd * 2)
        return col.mask((col < lower_limit) | (col > upper_limit))

    df = df.apply(process_column)
    return df.fillna(0)


def plot(motion: np.ndarray, r: tuple[int, int] | None = None):
    r = r if r else (0, len(motion))
    plt.figure(figsize=(18, 8))
    plt.plot(motion[r[0]:r[1]])
    plt.legend()
    plt.title("Motion Markers")
    plt.xlabel("Frame")
    plt.ylabel("Z Position (mm)")
    plt.show()

def plot_many(motions: np.ndarray, markers: list[str], r: tuple[int, int] | None = None):
    r = r if r else (0, len(motions[0]))
    plt.figure(figsize=(18, 8))
    for i, marker in enumerate(markers):
        current_motion = motions[i, r[0]:r[1]]
        plt.plot(current_motion, label=marker)
    
    plt.legend()
    plt.title("Motion Markers")
    plt.xlabel("Frame")
    plt.ylabel("Z Position (mm)")
    plt.show()