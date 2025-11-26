import numpy as np
from data import helpers
from models import constants
from data.prepare import extraction, augmentation, pruning

def _corrupt(sample: np.ndarray, noise: float = 0.1) -> np.ndarray:
    """Applies corruption to a given sample. Currently applies noise addition."""
    return augmentation.noise(sample, noise_level=noise)

# todo: turn into generator (with augmentation)
def get_training_dataset(
        unit_length: int = constants.unit_length,
        train_percentage: float = 0.8,
        constraints = extraction.ExtractionConstraints(),
        with_pruning: bool = False
    ) -> tuple[tuple[np.ndarray, np.ndarray], np.ndarray]:
    """Get the training dataset with samples stretched to unit_length (in samples)"""
    df = helpers.load_processed_capnostream()
    signal = df["co2_wave"].to_numpy()
    unit_markers = extraction.extract_unit_markers(signal, constraints) # get all unit start/end indices
    unit_signals = [signal[start:end] for start, end in unit_markers] # extract unit signals
    stretched_signals = [augmentation.stretch_to_unit_length(us, unit_length)[0] for us in unit_signals]
    print(f"Extracted {len(stretched_signals)} units for training dataset.")

    signals = np.expand_dims(np.vstack(stretched_signals), axis=-1)
    if with_pruning:
        signals = pruning.apply_pruning_filter(signals, constraints_hash=constraints.hash())
        print(f"After pruning, {len(signals)} units remain for training dataset.")

    train, val = np.split(signals, [int(train_percentage * len(signals))])
    return (_corrupt(train), train), val

def get_training_generator():
    """Generates unit signals for training with augmentation applied on-the-fly."""
    raise NotImplementedError("Training generator not yet implemented.")