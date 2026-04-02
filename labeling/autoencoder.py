from typing import cast
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Input, Dense, Conv1D, MaxPooling1D, UpSampling1D, Flatten, Reshape
from data.prepare import constants

def build(unit_length: int = constants.unit_length) -> Sequential:
    """Builds and returns the compiled autoencoder model."""
    autoencoder = Sequential([
        Input(shape=(unit_length, 1)),
        Conv1D(filters=16, kernel_size=3, activation='relu', padding='same'), # compression 1: (20, 1) -> (20, 16)
        MaxPooling1D(pool_size=2, padding='same'), # downsample 1: (20, 1) -> (10, 16)
        Conv1D(filters=8, kernel_size=3, activation='relu', padding='same'), # compression 2: (10, 16) -> (10, 8)
        MaxPooling1D(pool_size=2, padding='same'), # downsample 2: (10, 8) -> (5, 8)
        Flatten(), # flatten to vector: (5, 8) -> (40,)
        Dense(8, activation='relu', name='bottleneck'), # latent space: (40,) -> (8,)
        Dense(5 * 8, activation='relu'), # expand bottleneck: (8,) -> (40,)
        Reshape((5, 8)), # reshape: (40,) -> (5, 8)
        UpSampling1D(size=2), # upsample back to (10, 8)
        Conv1D(filters=8, kernel_size=3, activation='relu', padding='same'), # expansion conv: (10, 8) -> (10, 8)
        UpSampling1D(size=2), # upsample back to (20, 8)
        Conv1D(filters=16, kernel_size=3, activation='relu', padding='same'), # expansion conv: (20, 8) -> (20, 16)
        Conv1D(filters=1, kernel_size=3, activation='linear', padding='same') # reconstruction: (20, 16) -> (20, 1)
    ])

    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder

def from_file(file_path: str) -> Sequential:
    """Loads the autoencoder model from a file."""
    return cast(Sequential, load_model(file_path))

def calculate_threshold(ae: Sequential, val: np.ndarray, percentile: float = 95, export: bool = False) -> float:
    """Calculates the 95th percentile MSE threshold on the validation set."""
    reconstructions = ae.predict(val, verbose='silent')
    mse = np.mean(np.power(val - reconstructions, 2), axis=1)

    threshold = np.percentile(mse, percentile)
    if export:
        with open("models/out/threshold.txt", "w") as f:
            f.write(f"{threshold}\n")

    return threshold

def predictor(ae: Sequential, threshold: float):
    """Predicts the reconstruction of a given sample."""
    def predict(sample: np.ndarray) -> tuple[np.ndarray, np.floating, np.bool]:
        recon = np.array(ae.predict(sample.reshape(1, -1, 1), verbose='silent'),).reshape(-1)
        error = np.mean(np.power(sample - recon, 2))
        return recon, error, error < threshold
    
    return predict

def batch_predictor(ae: Sequential, threshold: float):
    """Predicts the reconstruction of a given sample."""
    def batch_predict(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        samples = np.expand_dims(samples, axis=-1)
        recon = np.array(ae.predict(samples, verbose='silent'),)
        errors = np.array(np.mean(np.power(samples - recon, 2), axis=1)).reshape(-1)
        return np.squeeze(recon), np.squeeze(errors), errors < threshold
    
    return batch_predict