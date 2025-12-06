from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, Conv1D, BatchNormalization, GlobalAveragePooling1D
from data.prepare import constants

def build_mlp(unit_length: int = constants.unit_length) -> Sequential:
    """Builds and returns the compiled autoencoder model."""
    mlp = Sequential([
        Input(shape=(unit_length,)),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(2, activation='softmax')
    ])

    mlp.compile(optimizer='adam', loss='mse')
    return mlp

def build_cnn(unit_length: int = constants.unit_length) -> Sequential:
    """Builds and returns the compiled autoencoder model."""
    cnn = Sequential([
        Input(shape=(unit_length, 1)), 
        Conv1D(filters=32, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(),
        Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(),
        GlobalAveragePooling1D(),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(2, activation='softmax')
    ])

    cnn.compile(optimizer='adam', loss='mse')
    return cnn