#!/usr/bin/env python3
"""Retrain and promote the crop_yield.csv XGBoost pipeline via the canonical notebook."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / "notebooks" / "crop_yield_eda_feature_engineering_xgboost.ipynb"
CANDIDATE_DIR = ROOT / "models" / "yield_model_crop_yield_candidate"
PRODUCTION_DIR = ROOT / "models" / "yield_model"
ARTIFACTS = (
    "yield_xgb_pipeline.pkl",
    "feature_columns.pkl",
    "unique_values.pkl",
    "metrics.json",
)


def main() -> int:
    if not (ROOT / "crop_yield.csv").exists():
        print("Missing crop_yield.csv", file=sys.stderr)
        return 1
    if not NOTEBOOK.exists():
        print(f"Missing notebook: {NOTEBOOK.relative_to(ROOT)}", file=sys.stderr)
        return 1

    command = [
        sys.executable,
        "-m",
        "jupyter",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        str(NOTEBOOK),
        "--output",
        NOTEBOOK.name,
        "--output-dir",
        str(NOTEBOOK.parent),
        "--ExecutePreprocessor.timeout=1200",
    ]
    print("Executing canonical yield notebook...")
    subprocess.run(command, cwd=ROOT, check=True)

    missing = [name for name in ARTIFACTS if not (CANDIDATE_DIR / name).exists()]
    if missing:
        print(f"Notebook did not produce: {', '.join(missing)}", file=sys.stderr)
        return 1

    metrics = json.loads((CANDIDATE_DIR / "metrics.json").read_text())
    if metrics.get("r2", 0) <= 0 or metrics.get("train_r2", 1) >= 0.95:
        print(f"Candidate metrics failed promotion guard: {metrics}", file=sys.stderr)
        return 1

    PRODUCTION_DIR.mkdir(parents=True, exist_ok=True)
    for artifact in ARTIFACTS:
        shutil.copy2(CANDIDATE_DIR / artifact, PRODUCTION_DIR / artifact)

    for obsolete in ("yield_xgb_model.pkl", "yield_scaler.pkl", "yield_label_encoders.pkl"):
        (PRODUCTION_DIR / obsolete).unlink(missing_ok=True)

    print("Promoted pipeline artifacts to models/yield_model/")
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
