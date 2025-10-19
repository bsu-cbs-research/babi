import numpy as np
import pywt
from scipy.signal import spectrogram
from sklearn.metrics import pairwise_distances
from numpy.lib.stride_tricks import sliding_window_view
import data.constants
from data import helpers
from models import constants

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

def get_training_dataset(window_size: int = constants.capnostream_window_size) -> np.ndarray:
    """Load capnostream annotated data, extract valid samples, and convert to spectrograms"""
    sr = data.constants.capnostream_sampling_rate
    window_size = window_size * sr  # convert to number of samples
    valid_samples = helpers.load_capnostream_annotated_data()
    sample_windows: np.ndarray = np.vstack([
        create_windows(np.array(x), window_size=window_size, step=(window_size // 2)) for x in valid_samples if len(x) >= window_size
    ])

    spectrograms = np.array([_signal_to_stft_spectrogram(x, nperseg=30, noverlap=6) for x in sample_windows])
    return spectrograms.reshape(spectrograms.shape[0], -1)


def _signal_to_stft_spectrogram(signal: np.ndarray, nperseg: int, noverlap: int) -> np.ndarray:
    _, _, Sxx = spectrogram(signal, data.constants.capnostream_sampling_rate, nperseg=nperseg, noverlap=noverlap, scaling='density', mode='magnitude')
    Sxx = np.log1p(Sxx)
    Sxx = (Sxx - Sxx.min()) / (Sxx.max() - Sxx.min())
    return Sxx

def _signal_to_scalogram(signal: np.ndarray, wavelet='morl', widths=np.arange(1,128)):
    cwtmatr, freqs = pywt.cwt(signal, widths, wavelet, sampling_period=1/data.constants.capnostream_sampling_rate)
    S = np.abs(cwtmatr)
    S = np.log1p(S)
    S = (S - S.min()) / (S.max() - S.min())
    return S

def _signal_to_recurrence_plot(signal: np.ndarray, eps=None):
    x = np.column_stack([signal[:-1], signal[1:]])
    D = pairwise_distances(x, metric='euclidean')
    if eps is None:
        eps = 0.1 * np.std(D)   # heuristic
    RP = (D <= eps).astype(float)
    return RP