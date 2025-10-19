from typing import Any
import numpy as np
from keras.models import Sequential
from keras.layers import Input, Dense, Conv2D, Conv2DTranspose, Flatten, Reshape
import data.constants

def build_cae(input_shape: tuple[int, int, int] = (128,128,1), latent_dim: int = 64):
    h, w, _ = input_shape
    ds_steps = 2  # Use 2 downsampling steps to ensure output shape matches input
    h_ds, w_ds = h // (2 ** ds_steps), w // (2 ** ds_steps)
    flattened_dim = h_ds * w_ds * 64

    cae = Sequential(name='ConvolutionalAutoencoder', layers=[
        Input(shape=input_shape),
        # ---------------------------------- encoder --------------------------------- #
        Conv2D(32, 3, strides=2, padding='same', activation='relu'),
        Conv2D(64, 3, strides=2, padding='same', activation='relu'),
        Flatten(),
        Dense(latent_dim, name='latent'),
        # ---------------------------------- decoder --------------------------------- #
        Dense(flattened_dim),
        Reshape((h_ds, w_ds, 64)),
        Conv2DTranspose(64, 3, strides=2, padding='same', activation='relu'),
        Conv2DTranspose(32, 3, strides=2, padding='same', activation='relu'),
        Conv2D(1, 3, padding='same', activation='sigmoid')
    ])

    cae.compile(optimizer='adam', loss='mse')
    return cae

def build_dae(window_size: int = data.constants.capnostream_sampling_rate * 10, latent_dim: int = 16):
    dae = Sequential([
        Input(shape=(window_size, 1)),
        # ---------------------------------- encoder --------------------------------- #
        Dense(128, activation="relu"),
        Dense(64, activation="relu"),
        Dense(latent_dim, activation="relu"), # latent space
        # ---------------------------------- decoder --------------------------------- #
        Dense(64, activation="relu"),
        Dense(128, activation="relu"),
        Dense(window_size, activation="sigmoid"),
        Reshape((window_size, 1))
    ])

    dae.compile(optimizer="adam", loss="mse")
    return dae

def fit(ae: Sequential, X: Any, epochs: int = 50, batch_size: int = 32, validation_split: float = 0.2):
    history = ae.fit(X, X, epochs=epochs, batch_size=batch_size, validation_split=validation_split)
    return history

def predict(ae: Sequential, X: Any):
    # reconstruct windows and compute errors
    recon = ae.predict(X)
    errors = np.mean((X - recon)**2, axis=(1,2))

    # flag anomalies above 95th percentile
    threshold = np.percentile(errors, 95)
    anomalies = errors > threshold
    return recon, errors, anomalies, threshold