"""Shared inference utilities for the crop_yield.csv XGBoost pipeline."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


RAW_FEATURES = [
    "state",
    "crop",
    "crop_year",
    "season",
    "area",
    "annual_rainfall",
    "fertilizer",
    "pesticide",
]
CATEGORICAL_FEATURES = {"state", "crop", "season"}
REQUIRED_NUMERIC_FEATURES = {
    "crop_year",
    "area",
    "annual_rainfall",
    "fertilizer",
    "pesticide",
}


class YieldInference:
    """Load and serve the candidate pipeline with one consistent input contract."""

    def __init__(self, model_dir: Path):
        self.model_dir = Path(model_dir)
        self.pipeline = joblib.load(self.model_dir / "yield_xgb_pipeline.pkl")
        self.feature_columns = list(joblib.load(self.model_dir / "feature_columns.pkl"))
        self.unique_values = joblib.load(self.model_dir / "unique_values.pkl")
        metrics_path = self.model_dir / "metrics.json"
        self.metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
        self.start_year = int(self.metrics.get("train_years", "1997-2015").split("-")[0])

    def _category(self, value: Any, field: str) -> str:
        normalized = str(value).strip().lower()
        if not normalized:
            raise ValueError(f"Missing categorical field: {field}")
        allowed = [str(item).strip().lower() for item in self.unique_values.get(field, [])]
        if normalized in allowed:
            return normalized
        matches = [item for item in allowed if normalized in item or item in normalized]
        if len(matches) == 1:
            return matches[0]
        raise ValueError(f"Unknown {field}: {value}")

    @staticmethod
    def _number(value: Any, field: str) -> float:
        if value in (None, ""):
            raise ValueError(f"Missing numeric field: {field}")
        try:
            parsed = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid numeric field: {field}") from exc
        if not np.isfinite(parsed):
            raise ValueError(f"Invalid numeric field: {field}")
        return parsed

    def prepare(self, data: dict[str, Any]) -> pd.DataFrame:
        row: dict[str, Any] = {}
        for field in RAW_FEATURES:
            if field in CATEGORICAL_FEATURES:
                row[field] = self._category(data.get(field), field)
            else:
                row[field] = self._number(data.get(field), field)
        if row["area"] <= 0:
            raise ValueError("Area must be greater than zero")
        if row["crop_year"] < 1997 or row["crop_year"] > 2030:
            raise ValueError("Crop year must be between 1997 and 2030")
        if row["annual_rainfall"] < 0 or row["fertilizer"] < 0 or row["pesticide"] < 0:
            raise ValueError("Rainfall, fertilizer, and pesticide values cannot be negative")

        frame = pd.DataFrame([row])
        for field in ("area", "annual_rainfall", "fertilizer", "pesticide"):
            frame[f"log1p_{field}"] = np.log1p(frame[field].clip(lower=0))
        frame["year_from_start"] = frame["crop_year"] - self.start_year
        frame["fertilizer_per_area"] = frame["fertilizer"] / frame["area"].clip(lower=1e-6)
        frame["pesticide_per_area"] = frame["pesticide"] / frame["area"].clip(lower=1e-6)
        frame["rainfall_per_area"] = frame["annual_rainfall"] / frame["area"].clip(lower=1e-6)
        return frame[self.feature_columns]

    def predict(self, data: dict[str, Any]) -> tuple[float, float]:
        prediction = float(self.pipeline.predict(self.prepare(data))[0])
        return max(0.0, prediction), prediction

    def options(self) -> dict[str, list[str]]:
        return {
            "states": sorted(self.unique_values.get("state", [])),
            "crops": sorted(self.unique_values.get("crop", [])),
            "seasons": sorted(self.unique_values.get("season", [])),
        }
