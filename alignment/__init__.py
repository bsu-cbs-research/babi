import numpy as np
from scipy import signal
from globals import constants
import pandas as pd

def automatic(capnostream: np.ndarray, motion: np.ndarray):
    correlation = signal.correlate(capnostream, motion, mode='full')
    lags = signal.correlation_lags(len(capnostream), len(motion), mode='full')

    peak_idx = np.argmin(correlation)
    calculated_shift = lags[peak_idx]
    calculated_shift_s = calculated_shift / constants.capnostream_sr
    
    print(f"Calculated Shift via Correlation ({correlation[peak_idx]}): {calculated_shift_s}")


def static(capnostream: pd.DataFrame, motion: pd.DataFrame, offset: int):
    capnostream["time"] = capnostream["time"] + pd.to_timedelta(offset, unit="s")
    labels_by_time = pd.Series(capnostream["co2_valid"].to_numpy(), index=pd.DatetimeIndex(capnostream["time"]))
    motion_labels = labels_by_time.reindex(motion.index,method="nearest",tolerance=pd.Timedelta("25ms")).fillna(0).astype(int)
    motion["co2_valid"] = motion_labels.to_numpy()
    return motion, capnostream