import numpy as np
from scipy.signal import spectrogram
from numpy.lib.stride_tricks import sliding_window_view
import data.constants
from data import helpers
from models import constants
from data.prepare import extraction, augmentation, pruning

def create_windows(signal: np.ndarray, window_size: int, step: int) -> np.ndarray:
    """Create overlapping windows from the 1D data array."""
    assert len(signal) >= window_size, f"Signal length {len(signal)} is shorter than window size {window_size}."
    windows = sliding_window_view(signal, window_shape=window_size)[::step]
    if len(signal) % window_size != 0:
        last_window = signal[-window_size:]
        windows = np.vstack([windows, last_window])

    return windows

def process_signal(signal: np.ndarray, window_size: int = constants.capnostream_window_size) -> np.ndarray:
    """Process capnostream signal to create windows and convert to spectrograms"""
    sr = data.constants.capnostream_sampling_rate
    window_size = window_size * sr  # convert to number of samples
    windows = create_windows(signal, window_size=window_size, step=window_size)
    spectrograms = np.array([_signal_to_stft_spectrogram(x, nperseg=30, noverlap=6) for x in windows])
    return spectrograms.reshape(spectrograms.shape[0], -1)

# todo: turn into generator (with augmentation)
def get_training_dataset(
        unit_length: int = constants.unit_length, 
        constraints = extraction.ExtractionConstraints(),
        with_pruning: bool = False
    ) -> tuple[np.ndarray, tuple[float, float]]:
    """Get the training dataset with samples stretched to unit_length (in samples)"""
    df = helpers.load_processed_capnostream()
    signal = df["co2_wave"].to_numpy()
    unit_markers = extraction.extract_unit_markers(signal, constraints) # get all unit start/end indices
    mean, std = signal.mean(axis=0), signal.std(axis=0)
    # signal = (signal - mean) / std # normalize after extracting markers
    unit_signals = [signal[start:end] for start, end in unit_markers] # extract unit signals
    stretched_signals = [augmentation.stretch_to_unit_length(us, unit_length)[0] for us in unit_signals]
    print(f"Extracted {len(stretched_signals)} units for training dataset.")

    signals = np.expand_dims(np.vstack(stretched_signals), axis=-1)
    if with_pruning:
        signals = pruning.apply_pruning_filter(signals, constraints_hash=constraints.hash())
        print(f"After pruning, {len(signals)} units remain for training dataset.")

    return signals, (mean, std)

def get_training_generator():
    """Generates unit signals for training with augmentation applied on-the-fly."""
    pass


def _signal_to_stft_spectrogram(signal: np.ndarray, nperseg: int, noverlap: int) -> np.ndarray:
    _, _, Sxx = spectrogram(signal, data.constants.capnostream_sampling_rate, nperseg=nperseg, noverlap=noverlap, scaling='density', mode='magnitude')
    Sxx = np.log1p(Sxx)
    Sxx = (Sxx - Sxx.min()) / (Sxx.max() - Sxx.min())
    return Sxx