from labeling import autoencoder, processing
import numpy as np
from data.prepare import constants

def label(capnostream_signal: np.ndarray):
    # model = autoencoder.from_file("program/static/model.keras")
    # threshold = float(open("program/static/threshold.txt").read().strip())
    # min_val, max_val = [float(v) for v in open("program/static/scaling.txt").read().strip().split(',')]
    # batch_predict = autoencoder.batch_predictor(model, threshold)

    # original_length = len(capnostream_signal)

    # units = processing.signal_to_units(capnostream_signal, min_val, max_val)

    # _, _, labels = batch_predict(units)

    # # create analysis blocks 
    # labels = np.concatenate([labels.astype(int), [0] * (original_length - len(labels) + constants.unit_length)]) # pad labels
    # normative_label_indices = np.where(labels == 1)[0] # gather indices of normative labels
    # insertion_block = np.full(constants.unit_length, 1) # create block of ones
    # for idx in normative_label_indices: labels[idx : idx + constants.unit_length] = insertion_block
    # labels = labels[:original_length] # trim back to original length if overflow

    # return labels

    return True