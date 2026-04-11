import os
import pandas as pd
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
    return df