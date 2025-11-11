from typing import Literal
import numpy as np
from scipy.signal import spectrogram
from numpy.lib.stride_tricks import sliding_window_view
import data.constants
from data import helpers
from models import constants
from data.prepare import extraction, augmentation

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

# def get_training_dataset(window_size: int = constants.capnostream_window_size) -> np.ndarray:
#     """Load capnostream annotated data, extract valid samples, and convert to spectrograms"""
#     sr = data.constants.capnostream_sampling_rate
#     window_size = window_size * sr  # convert to number of samples
#     valid_samples = helpers.load_capnostream_annotated_data()
#     sample_windows: np.ndarray = np.vstack([
#         create_windows(np.array(x), window_size=window_size, step=(window_size // 2)) for x in valid_samples if len(x) >= window_size
#     ])

#     spectrograms = np.array([_signal_to_stft_spectrogram(x, nperseg=30, noverlap=6) for x in sample_windows])
#     return spectrograms.reshape(spectrograms.shape[0], -1)

def get_training_dataset(unit_length: int = 20, grouping: Literal["baby", "group", "both"] | None = None) -> dict[str, np.ndarray]:
    # todo: turn into generator (with augmentation)
    """Get the training dataset with samples stretched to unit_length (in samples) and converted to spectrograms"""
    df = helpers.load_processed_capnostream()

    signals: dict[str, np.ndarray] = {}

    if not grouping: signals = {"x": df['co2_wave'].to_numpy()}
    elif grouping == "baby":
        signals = df.groupby('baby_index')['co2_wave'].apply(lambda x: np.array(x)).to_dict()
    elif grouping == "group":
        signals = df.groupby('group_index')['co2_wave'].apply(lambda x: np.array(x)).to_dict()
    elif grouping == "both":
        signals = df.groupby(['baby_index', 'group_index'])['co2_wave'].apply(lambda x: np.array(x)).to_dict()

    processed_signals: dict[str, np.ndarray] = {}

    for key, signal in signals.items():
        signal = (signal - signal.mean(axis=0)) / signal.std(axis=0) # normalize
        unit_markers = extraction.extract_unit_markers(signal) # get all unit start/end indices
        unit_signals = [signal[start:end] for start, end in unit_markers] # extract unit signals
        stretched_signals = [augmentation.stretch(us, stretch_factor=unit_length / len(us)) for us in unit_signals]
        processed_signals[key] = np.vstack(stretched_signals)

    return processed_signals


    # if grouping == "baby":
    #     df[df['baby_index']].reset_index(drop=True) # filter by baby

    # df[(df['baby_index'] == bi) & (df['group_index'] == gi)].reset_index(drop=True) # filter by baby and group as needed
    # df[df['group_index'] == gi].reset_index(drop=True) # filter by group

def _signal_to_stft_spectrogram(signal: np.ndarray, nperseg: int, noverlap: int) -> np.ndarray:
    _, _, Sxx = spectrogram(signal, data.constants.capnostream_sampling_rate, nperseg=nperseg, noverlap=noverlap, scaling='density', mode='magnitude')
    Sxx = np.log1p(Sxx)
    Sxx = (Sxx - Sxx.min()) / (Sxx.max() - Sxx.min())
    return Sxx