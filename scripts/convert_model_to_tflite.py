from __future__ import annotations

import argparse
from pathlib import Path


def convert(keras_model_path: Path, tflite_model_path: Path) -> None:
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError(
            "TensorFlow is required for conversion. Install training extras first, "
            "for example: pip install .[training]"
        ) from exc

    model = tf.keras.models.load_model(keras_model_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    tflite_model_path.parent.mkdir(parents=True, exist_ok=True)
    tflite_model_path.write_bytes(tflite_model)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a Keras model into a TensorFlow Lite model."
    )
    parser.add_argument(
        "--input",
        default="program/static/model.keras",
        help="Path to input Keras model file.",
    )
    parser.add_argument(
        "--output",
        default="program/static/model.tflite",
        help="Path for output TFLite model file.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.is_file():
        raise FileNotFoundError(f"Input Keras model not found: {input_path}")

    convert(input_path, output_path)
    print(f"Wrote TFLite model: {output_path}")


if __name__ == "__main__":
    main()
