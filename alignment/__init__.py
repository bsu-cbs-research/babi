import numpy as np
from scipy import signal
from globals import constants

def automatic(capnostream: np.ndarray, motion: np.ndarray):
    correlation = signal.correlate(capnostream, motion, mode='full')
    lags = signal.correlation_lags(len(capnostream), len(motion), mode='full')

    peak_idx = np.argmin(correlation)
    calculated_shift = lags[peak_idx]
    calculated_shift_s = calculated_shift / constants.capnostream_sr
    
    print(f"Calculated Shift via Correlation ({correlation[peak_idx]}): {calculated_shift_s}")


def static():
    return True