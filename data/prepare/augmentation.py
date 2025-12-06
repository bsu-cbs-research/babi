import numpy as np
from data.prepare import constants

def scale(signal: np.ndarray, scale_factor: float) -> np.ndarray:
    """Scale the amplitude of the signal by scale_factor"""
    return signal * scale_factor

def roll(signal: np.ndarray, roll_amount: int) -> np.ndarray:
    """Time shift the signal by roll_amount samples"""
    return np.roll(signal, roll_amount)

def noise(signal: np.ndarray, noise_level: float) -> np.ndarray:
    """Add random noise to the signal"""
    noise = np.random.normal(0, noise_level, size=signal.shape)
    return signal + noise

def stretch(signal: np.ndarray, stretch_factor: float) -> np.ndarray:
    """Stretch or compress the signal in time by stretch_factor using simple indexing"""
    from scipy.interpolate import interp1d
    new_length = int(len(signal) * stretch_factor)
    new_time = np.linspace(0, len(signal) - 1, new_length)

    interpolator = interp1d(np.arange(len(signal)), signal, kind='linear')
    return interpolator(new_time)

def stretch_to_unit_length(signal: np.ndarray, unit_length: int = constants.unit_length) -> tuple[np.ndarray, float]:
    """Stretch or compress the signal to a fixed unit_length using simple indexing"""
    stretch_factor = unit_length / len(signal)
    return stretch(signal, stretch_factor=stretch_factor), stretch_factor