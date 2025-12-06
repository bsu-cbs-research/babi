import numpy as np
from data.prepare import load, augmentation

def _corrupt(sample: np.ndarray, noise: float = 0.015) -> np.ndarray:
    """Applies corruption to a given sample. Currently applies noise addition."""
    return augmentation.noise(sample, noise_level=noise)

def reverse_sample_scaling(scaled_data: np.ndarray, g_min: float, g_max: float):
    return scaled_data * (g_max - g_min) + g_min

def scale_sample(sample: np.ndarray, g_min: float, g_max: float):
    return (sample - g_min) / (g_max - g_min)

def get_training_dataset(
        train_percentage: float = 0.8,
        validation_percentage: float = 0.1,
    ) -> tuple[tuple[np.ndarray, np.ndarray], np.ndarray, np.ndarray, np.ndarray, tuple[float, float]]:
    """Get the training dataset with samples stretched to unit_length (in samples)"""
    normative_signals = load.normative()
    abnormal_signals = load.abnormal()

    train, val = np.split(normative_signals, [int(train_percentage * len(normative_signals))])
    val, test = np.split(val, [int((validation_percentage / (1 - train_percentage)) * len(val))])
    g_max, g_min = np.max(train), np.min(train)

    train_scaled = (train - g_min) / (g_max - g_min)
    val_scaled   = (val   - g_min) / (g_max - g_min)
    normative_test_scaled = (test - g_min) / (g_max - g_min)
    abnormal_test_scaled = (abnormal_signals - g_min) / (g_max - g_min)
    return (_corrupt(train_scaled), train_scaled), val_scaled, normative_test_scaled, abnormal_test_scaled, (g_min, g_max)

def get_abnormal_testing_dataset(g_min: float, g_max: float) -> np.ndarray:
    """Get the abnormal testing dataset with samples stretched to unit_length (in samples)"""
    signals = load.abnormal()
    signals_scaled = (signals - g_min) / (g_max - g_min)
    return signals_scaled