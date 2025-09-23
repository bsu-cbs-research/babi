import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from data import constants
import scipy as sp

def plot_capnostream_waveform(data: pd.Series | np.ndarray):
    data = data.to_numpy() if isinstance(data, pd.Series) else data
    time = np.arange(len(data)) / constants.capnostream_sampling_rate

    plt.figure(figsize=(24, 6))
    plt.plot(time, data, color='blue')
    plt.title('Capnostream CO₂ Waveform')
    plt.xlabel('Time (s)')
    plt.ylabel('CO₂ Level')
    plt.grid(True)
    plt.show()

def plot_capnostream_fourier_transform(data: pd.Series | np.ndarray):
    data = data.to_numpy() if isinstance(data, pd.Series) else data
    n = len(data)
    dft = constants.capnostream_sampling_rate / n * sp.fft.fft(data)
    freq = sp.fft.fftfreq(n, d=1/constants.capnostream_sampling_rate)
    magnitude = np.abs(dft)
    half_n = n // 2
    freq = freq[:half_n]
    magnitude = magnitude[:half_n]
    plt.figure(figsize=(24, 6))
    plt.plot(freq, magnitude, color='red')
    plt.title('Fourier Transform of Capnostream CO₂ Waveform')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.grid(True)
    plt.show()