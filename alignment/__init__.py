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
    capnostream["Time"] = pd.to_datetime(capnostream["Time"])
    capnostream["Time"] = capnostream["Time"] + pd.Timedelta(seconds=offset)
    # 3. Perform the "As-Of" Join
    # For every row in df_100, find the closest row in df_20
    # result = pd.merge_asof(
    #     motion, 
    #     capnostream, 
    #     on='ts', 
    #     direction='nearest'  # Matches the absolute closest timestamp
    # )

    # Your new labels are now in result['label']
    return motion, capnostream