from typing import cast
import numpy as np
from scipy import signal

def upsample(sig: np.ndarray, old: int, new: int) -> np.ndarray:
    new_length = len(sig) * (new // old)
    return cast(np.ndarray, signal.resample(sig, new_length))