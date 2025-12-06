import numpy as np
from data import helpers
from data.prepare import constants, extraction, augmentation, pruning

def normative(unit_length: int = constants.unit_length, constraints = extraction.ExtractionConstraints(), with_pruning: bool = True) -> np.ndarray:
    signal = helpers.load_processed_capnostream()["co2_wave"].to_numpy()
    normative_unit_markers = extraction.extract_unit_markers(signal, constraints)
    unit_signals = [signal[start:end] for start, end in normative_unit_markers]
    stretched_signals = [augmentation.stretch_to_unit_length(us, unit_length)[0] for us in unit_signals]

    signals = np.expand_dims(np.vstack(stretched_signals), axis=-1)
    if with_pruning: signals = pruning.apply_mask(signals, key=pruning.hash_array(signals))

    return signals

def abnormal(unit_length: int = constants.unit_length, constraints = extraction.ExtractionConstraints(), with_pruning: bool = True) -> np.ndarray:
    signal = helpers.load_processed_capnostream()["co2_wave"].to_numpy()
    mask = np.ones(len(signal), dtype=bool)
    for start, end in extraction.extract_unit_markers(signal, constraints): mask[start:end] = False
    abnormal_signal = signal[mask]
    num_chunks = len(abnormal_signal) // unit_length
    limit = num_chunks * unit_length
    truncated_data = abnormal_signal[:limit]
    chunked_signals = truncated_data.reshape(-1, unit_length)
    signals = np.expand_dims(chunked_signals, axis=-1)
    if with_pruning: signals = pruning.apply_mask(signals, key=pruning.hash_array(signals))

    return signals