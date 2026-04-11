import pandas as pd
from globals import constants
from datetime import datetime

def capnostream(path: str):
    df = pd.read_excel(path).reset_index(drop=True).replace("--", -1)
    start_time = datetime.strptime(df.iat[2, 1].__str__(), "%b %d,%y  %I:%M:%S %p")
    df.columns = df.iloc[3, :].tolist()
    df = df.drop(df.index[:5]).reset_index(drop=True).replace("--", -1).drop(columns=["Time", "Date"])
    df['time'] = pd.date_range(start=start_time, periods=len(df), freq='50ms')
    return df

# todo: ensure validated?
def _parse_file(path: str):
    start = pd.to_datetime(pd.read_csv(path, sep="\t", header=6, nrows=0).columns[1])
    df = pd.read_csv(path, sep="\t", header=10, usecols=[i+2 for i in range(69)])
    duration_seconds = (len(df) - 1) / constants.motion_sr
    end = start + pd.to_timedelta(duration_seconds, unit='s')
    df.index = pd.date_range(start=start, end=end, periods=len(df))

    df.attrs = {"start": start, "end": end}
    return df.mask(df == 0)

def motion(paths: set[str]):
    """Parses motion data and forms composite"""
    raw_motions = sorted([_parse_file(n) for n in paths], key=lambda x: x.attrs["end"])
    df = pd.concat(raw_motions).sort_index()
    df = df.groupby(level=0).mean() # CRITICAL: Removes duplicate timestamps
    full_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq="10ms")
    df = df.reindex(full_index, method='nearest', tolerance=pd.Timedelta(milliseconds=5))
    return df