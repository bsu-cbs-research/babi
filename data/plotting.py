import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from data import constants

def plot_capnostream_waveform(
        data: pd.Series | np.ndarray, 
        view_range: tuple[int, int] | None = None,
        anomalies: tuple[int, np.ndarray] | None = None, 
        export: str | None = None
    ):
    """Plot the Capnostream CO2 waveform data over time with optional anomaly markers."""
    data = data.to_numpy() if isinstance(data, pd.Series) else data
    start, end = view_range if view_range is not None else (0, len(data))
    assert 0 <= start < end <= len(data), f"view_range must be within the data length (0 to {len(data)}"

    signal_start, signal_end = int(start * constants.capnostream_sampling_rate), int(end * constants.capnostream_sampling_rate)
    
    time = (np.arange(len(data)) / constants.capnostream_sampling_rate)[signal_start:signal_end]
    data = data[signal_start:signal_end]

    plt.figure(figsize=(24, 6))
    plt.plot(time, data, color='blue')
    plt.title('Capnostream CO₂ Waveform')
    plt.xlabel('Time (s)')
    plt.ylabel('CO₂ Level')
    plt.grid(True)

    if anomalies is not None and len(anomalies) > 0:
        window_size, values = anomalies
        values = values * window_size
        for x in values:
            if start <= x < end:
                plt.axvspan(x, x+window_size, color='red', alpha=0.3)
                plt.axvline(x, color='red', alpha=0.7)
                
    if export is not None:
        plt.savefig(export)

    plt.show()

def dual(s1: np.ndarray, s2: np.ndarray, title: str = 'Original vs Reconstructed', l1: str = 'Original', l2: str = 'Reconstructed'):
    plt.figure(figsize=(18, 4))
    plt.plot(s1, label=l1)
    plt.plot(s2, label=l2)
    plt.title(title)
    plt.legend()
    plt.show()