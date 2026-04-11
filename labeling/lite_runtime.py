from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable

import numpy as np


class LiteRuntimeImportError(ImportError):
    pass


def _load_interpreter_class():
    try:
        from ai_edge_litert.interpreter import Interpreter

        return Interpreter, "ai-edge-litert"
    except ImportError:
        pass

    try:
        from tflite_runtime.interpreter import Interpreter

        return Interpreter, "tflite-runtime"
    except ImportError:
        pass

    raise LiteRuntimeImportError(
        "No TensorFlow Lite runtime is installed. Install ai-edge-litert "
        "(preferred) or tflite-runtime."
    )


@dataclass
class LiteModel:
    interpreter: object
    input_index: int
    output_index: int
    output_dtype: np.dtype
    source: str


def load_model(model_path: str) -> LiteModel:
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"TFLite model artifact not found: {model_path}")

    interpreter_cls, source = _load_interpreter_class()
    interpreter = interpreter_cls(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    if not input_details or not output_details:
        raise ValueError("Invalid TFLite model: missing input/output tensor details.")

    input_index = int(input_details[0]["index"])
    output_index = int(output_details[0]["index"])
    output_dtype = np.dtype(output_details[0]["dtype"])

    return LiteModel(
        interpreter=interpreter,
        input_index=input_index,
        output_index=output_index,
        output_dtype=output_dtype,
        source=source,
    )


def batch_predictor(model: LiteModel, threshold: float) -> Callable[[np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    def batch_predict(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # input shape: (n, unit_length) -> (n, unit_length, 1)
        samples = np.expand_dims(samples, axis=-1)
        recon = np.empty(samples.shape, dtype=model.output_dtype)

        for i in range(samples.shape[0]):
            sample = samples[i : i + 1].astype(np.float32, copy=False)
            model.interpreter.set_tensor(model.input_index, sample)
            model.interpreter.invoke()
            recon[i] = model.interpreter.get_tensor(model.output_index)[0]

        errors = np.array(np.mean(np.power(samples - recon, 2), axis=1)).reshape(-1)
        return np.squeeze(recon), np.squeeze(errors), errors < threshold

    return batch_predict
