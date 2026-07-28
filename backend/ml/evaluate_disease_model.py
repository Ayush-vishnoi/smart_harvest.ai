#!/usr/bin/env python3
"""Evaluate the deployed disease TFLite model on the seeded validation split."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = ROOT / "PlantVillage" / "PlantVillage"
MODEL_PATH = ROOT / "models" / "disease_model.tflite"
LABELS_PATH = ROOT / "models" / "class_labels.json"
METRICS_PATH = ROOT / "models" / "disease_metrics.json"
CONFUSION_PATH = ROOT / "models" / "disease_confusion_matrix.png"
IMAGE_SIZE = (192, 192)
BATCH_SIZE = 16
SEED = 123


def main() -> int:
    labels_map = json.loads(LABELS_PATH.read_text())
    class_names = [labels_map[str(index)] for index in range(len(labels_map))]
    dataset_classes = sorted(path.name for path in DATASET_DIR.iterdir() if path.is_dir())
    if dataset_classes != class_names:
        raise ValueError(
            "Dataset class order does not match the deployed model labels.\n"
            f"Dataset: {dataset_classes}\nModel: {class_names}"
        )

    validation = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        validation_split=0.2,
        subset="validation",
        seed=SEED,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=True,
    )

    interpreter = tf.lite.Interpreter(model_path=str(MODEL_PATH))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    expected_shape = tuple(int(value) for value in input_details["shape"])

    true_labels = []
    predicted_labels = []
    processed = 0
    total = int(tf.data.experimental.cardinality(validation).numpy())

    for batch_number, (images, labels) in enumerate(validation, start=1):
        images = images.numpy().astype(np.float32) / 255.0
        for image, label in zip(images, labels.numpy()):
            array = np.expand_dims(image, axis=0)
            if tuple(array.shape) != expected_shape:
                interpreter.resize_tensor_input(input_details["index"], array.shape, strict=True)
                interpreter.allocate_tensors()
                input_details = interpreter.get_input_details()[0]
                output_details = interpreter.get_output_details()[0]
            if input_details["dtype"] != np.float32:
                scale, zero_point = input_details["quantization"]
                array = np.round(array / scale + zero_point).astype(input_details["dtype"])
            interpreter.set_tensor(input_details["index"], array)
            interpreter.invoke()
            output = interpreter.get_tensor(output_details["index"])[0]
            predicted_labels.append(int(np.argmax(output)))
            true_labels.append(int(label))
            processed += 1
        print(f"Evaluated batch {batch_number}/{total} ({processed} images)", end="\r", flush=True)
    print()

    accuracy = accuracy_score(true_labels, predicted_labels)
    weighted = precision_recall_fscore_support(
        true_labels, predicted_labels, average="weighted", zero_division=0
    )
    macro = precision_recall_fscore_support(
        true_labels, predicted_labels, average="macro", zero_division=0
    )
    per_class = precision_recall_fscore_support(
        true_labels,
        predicted_labels,
        labels=list(range(len(class_names))),
        zero_division=0,
    )
    matrix = confusion_matrix(
        true_labels, predicted_labels, labels=list(range(len(class_names)))
    )

    metrics = {
        "model": "MobileNetV3Small transfer learning (TensorFlow Lite)",
        "dataset": "PlantVillage",
        "validation_split": 0.2,
        "split_seed": SEED,
        "validation_images": len(true_labels),
        "classes": len(class_names),
        "accuracy": round(float(accuracy), 6),
        "weighted_precision": round(float(weighted[0]), 6),
        "weighted_recall": round(float(weighted[1]), 6),
        "weighted_f1": round(float(weighted[2]), 6),
        "macro_precision": round(float(macro[0]), 6),
        "macro_recall": round(float(macro[1]), 6),
        "macro_f1": round(float(macro[2]), 6),
        "per_class": {
            name: {
                "precision": round(float(per_class[0][index]), 6),
                "recall": round(float(per_class[1][index]), 6),
                "f1": round(float(per_class[2][index]), 6),
                "support": int(per_class[3][index]),
            }
            for index, name in enumerate(class_names)
        },
        "confusion_matrix": matrix.tolist(),
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2) + "\n")

    figure, axis = plt.subplots(figsize=(16, 14))
    display = ConfusionMatrixDisplay(matrix, display_labels=class_names)
    display.plot(ax=axis, xticks_rotation=45, colorbar=False, cmap="Greens", values_format="d")
    axis.set_title("Disease Model Confusion Matrix — PlantVillage Validation Split")
    figure.tight_layout()
    figure.savefig(CONFUSION_PATH, dpi=150)
    plt.close(figure)

    print(json.dumps({key: value for key, value in metrics.items() if key not in {"per_class", "confusion_matrix"}}, indent=2))
    print(f"Metrics saved to {METRICS_PATH}")
    print(f"Confusion matrix saved to {CONFUSION_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
