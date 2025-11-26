import numpy as np
from keras.models import Sequential
from keras.layers import Input, Dense
from models import constants

def build(unit_length: int = constants.unit_length, latent: int = 2) -> Sequential:
    """Builds and returns the compiled autoencoder model."""
    autoencoder = Sequential([
        Input(shape=(unit_length,)),
        Dense(12, activation="relu"),
        Dense(latent, activation='relu', name='latent'), # latent space (bottle neck)
        Dense(12, activation="relu"),
        Dense(unit_length, activation="linear")
    ])

    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder

def calculate_threshold(ae: Sequential, val: np.ndarray, percentile: float = 95) -> float:
    """Calculates the 95th percentile MSE threshold on the validation set."""
    reconstructions = ae.predict(val, verbose='silent')
    mse = np.mean(np.power(np.squeeze(val) - reconstructions, 2), axis=1)
    return np.percentile(mse, percentile) 

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
        recon = np.array(ae.predict(np.expand_dims(samples, axis=-1), verbose='silent'),)
        errors = np.mean(np.power(samples - recon, 2), axis=1)
        return recon, errors, errors < threshold
    
    return batch_predict