#!/usr/bin/env python3
"""Validate the local Smart Harvest model artifacts and prepare a runtime environment."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
REQUIRED = [
    ROOT / "models/yield_model/yield_xgb_pipeline.pkl",
    ROOT / "models/yield_model/feature_columns.pkl",
    ROOT / "models/yield_model/unique_values.pkl",
    ROOT / "models/yield_model/metrics.json",
    ROOT / "models/disease_model.tflite",
    ROOT / "models/class_labels.json",
]


def main():
    missing = [str(path.relative_to(ROOT)) for path in REQUIRED if not path.exists()]
    if missing:
        print("Missing required model artifacts:")
        print("\n".join(f"- {path}" for path in missing))
        return 1

    print("All pipeline-based yield and disease model artifacts are present.")
    print("Install dependencies with: python3 -m pip install -r requirements.txt")
    print("Start locally with: gunicorn --chdir backend app_v2:app --bind 0.0.0.0:5001 --workers 1")
    return 0


if __name__ == "__main__":
    sys.exit(main())
