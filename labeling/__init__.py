import os
from typing import Optional

import numpy as np

from labeling import lite_runtime, processing
from data.prepare import constants

# Default location of the bundled model artifacts, relative to cwd.
_DEFAULT_STATIC_DIR = os.path.join("program", "static")

# Lazily-loaded, cached model artifacts keyed by resolved static_dir so repeated
# calls (e.g. from the UI) don't reload the interpreter on every invocation.
_ARTIFACT_CACHE: dict[str, tuple[lite_runtime.LiteModel, float, float, float]] = {}


def _load_artifacts(static_dir: str):
    """Load (or fetch cached) tflite model, threshold, and (min, max) scaling."""
    key = os.path.abspath(static_dir)
    cached = _ARTIFACT_CACHE.get(key)
    if cached is not None:
        return cached

    model_path = os.path.join(static_dir, "model.tflite")
    threshold_path = os.path.join(static_dir, "threshold.txt")
    scaling_path = os.path.join(static_dir, "scaling.txt")

    for p in (model_path, threshold_path, scaling_path):
        if not os.path.isfile(p):
            raise FileNotFoundError(
                f"Labeling artifact not found: {p}. "
                f"Pass static_dir=... pointing at the directory containing "
                "model.tflite, threshold.txt, and scaling.txt."
            )

    model = lite_runtime.load_model(model_path)
    with open(threshold_path) as f:
        threshold = float(f.read().strip())
    with open(scaling_path) as f:
        min_val, max_val = (float(v) for v in f.read().strip().split(","))

    artifacts = (model, threshold, min_val, max_val)
    _ARTIFACT_CACHE[key] = artifacts
    return artifacts


def label(signal: np.ndarray, static_dir: Optional[str] = None) -> np.ndarray:
    """Label a 1-D CO2 waveform signal as normative (1) / abnormal (0).

    Parameters
    ----------
    signal: 1-D array of raw CO2 wave samples.
    static_dir: directory containing model.tflite, threshold.txt, scaling.txt.
        Defaults to ``program/static`` (relative to cwd) for CLI usage. The
        UI should pass ``resource_path("program/static")`` so the bundled
        PyInstaller app can find the files via ``_MEIPASS``.
    """
    resolved_static_dir = static_dir or _DEFAULT_STATIC_DIR
    model, threshold, min_val, max_val = _load_artifacts(resolved_static_dir)

    batch_predict = lite_runtime.batch_predictor(model, threshold)
    original_length = len(signal)
    units = processing.signal_to_units(signal, min_val, max_val)
    _, _, labels = batch_predict(units)

    # create analysis blocks
    labels = np.concatenate(
        [labels.astype(int), [0] * (original_length - len(labels) + constants.unit_length)]
    )  # pad labels
    normative_label_indices = np.where(labels == 1)[0]
    insertion_block = np.full(constants.unit_length, 1)
    for idx in normative_label_indices:
        labels[idx : idx + constants.unit_length] = insertion_block
    labels = labels[:original_length]  # trim back to original length if overflow

    return labels
