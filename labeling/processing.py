import numpy as np
from data.prepare import augmentation, constants
from numpy.lib.stride_tricks import sliding_window_view

def _corrupt(sample: np.ndarray, noise: float = 0.015) -> np.ndarray:
    """Applies corruption to a given sample. Currently applies noise addition."""
    return augmentation.noise(sample, noise_level=noise)

def stack_signals(signals: np.ndarray, unit_length = constants.unit_length ) -> np.ndarray:
    """Convert signals into overlapping windows."""
    length = (signals.shape[0] - 1) + unit_length
    averages = np.zeros(length)
    counts = np.zeros(length)

    for i in range(signals.shape[0]):
        start = i
        end = i + unit_length
        averages[start:end] += signals[i]
        counts[start:end] += 1

    return averages / counts

def reverse_sample_scaling(scaled_data: np.ndarray, g_min: float, g_max: float):
    return scaled_data * (g_max - g_min) + g_min

def _scale_sample(sample: np.ndarray, g_min: float, g_max: float):
    return (sample - g_min) / (g_max - g_min)

def get_training_dataset(
        train_percentage: float = 0.8,
        validation_percentage: float = 0.1,
        export_min_max: bool = False
    ) -> tuple[tuple[np.ndarray, np.ndarray], np.ndarray, np.ndarray, np.ndarray, tuple[float, float]]:
    """Get the training dataset with samples stretched to unit_length (in samples)"""
    from data.prepare import load

    normative_signals = load.normative()
    abnormal_signals = load.abnormal()

    train, val = np.split(normative_signals, [int(train_percentage * len(normative_signals))])
    val, test = np.split(val, [int((validation_percentage / (1 - train_percentage)) * len(val))])
    g_max, g_min = np.max(train), np.min(train)

    train_scaled = (train - g_min) / (g_max - g_min)
    val_scaled   = (val   - g_min) / (g_max - g_min)
    normative_test_scaled = (test - g_min) / (g_max - g_min)
    abnormal_test_scaled = (abnormal_signals - g_min) / (g_max - g_min)

    if export_min_max:
        with open("program/static/scaling.txt", "w") as f: f.write(f"{g_min},{g_max}\n")

    return (_corrupt(train_scaled), train_scaled), val_scaled, normative_test_scaled, abnormal_test_scaled, (g_min, g_max)

def get_abnormal_testing_dataset(g_min: float, g_max: float) -> np.ndarray:
    """Get the abnormal testing dataset with samples stretched to unit_length (in samples)"""
    from data.prepare import load

    signals = load.abnormal()
    signals_scaled = (signals - g_min) / (g_max - g_min)
    return signals_scaled

def signal_to_units(signal: np.ndarray, g_min: float, g_max: float) -> np.ndarray:
    """Prepare a single signal for the model."""
    signal_scaled = _scale_sample(signal, g_min, g_max)
    units = sliding_window_view(signal_scaled, (constants.unit_length, ))
    units = np.array([augmentation.stretch_to_unit_length(unit)[0] for unit in units])
    return units
