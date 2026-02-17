#!/usr/bin/env python3
"""
Smart Harvest AI - Backend API (app_v2.py)
Flask REST API for yield prediction and irrigation recommendation
"""

import json
import logging
import traceback
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from flask import Response

# ─── Setup ────────────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "backend" / "models"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

app = Flask(__name__)

# ─── CORS (manual, no flask-cors needed) ─────────────────────────────────────
@app.after_request
def add_cors(response):
    response.headers["Access-Control-Allow-Origin"]  = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return response

@app.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        return Response(status=200)

# ─── Load Artifacts ───────────────────────────────────────────────────────────
YIELD_DIR   = MODEL_DIR / "yield_model"
IRR_DIR     = MODEL_DIR / "irrigation_model"

def load_yield_models():
    rf      = joblib.load(YIELD_DIR / "yield_rf_model.pkl")
    gb      = joblib.load(YIELD_DIR / "yield_gb_model.pkl")
    scaler  = joblib.load(YIELD_DIR / "yield_scaler.pkl")
    le_dict = joblib.load(YIELD_DIR / "yield_label_encoders.pkl")
    features= joblib.load(YIELD_DIR / "feature_columns.pkl")
    unique  = joblib.load(YIELD_DIR / "unique_values.pkl")
    with open(YIELD_DIR / "metrics.json") as f:
        metrics = json.load(f)
    return rf, gb, scaler, le_dict, features, unique, metrics

def load_irrigation_models():
    clf        = joblib.load(IRR_DIR / "irrigation_rf_model.pkl")
    scaler     = joblib.load(IRR_DIR / "irrigation_scaler.pkl")
    le_irr     = joblib.load(IRR_DIR / "irrigation_label_encoders.pkl")
    le_target  = joblib.load(IRR_DIR / "irrigation_target_encoder.pkl")
    features   = joblib.load(IRR_DIR / "irrigation_features.pkl")
    return clf, scaler, le_irr, le_target, features

try:
    rf_model, gb_model, y_scaler, y_le, y_features, unique_vals, y_metrics = load_yield_models()
    irr_clf, irr_scaler, irr_le, irr_target_le, irr_features = load_irrigation_models()
    MODELS_LOADED = True
    log.info("✅ All models loaded successfully")
except Exception as e:
    MODELS_LOADED = False
    log.error(f"❌ Failed to load models: {e}")
    traceback.print_exc()

# ─── Helpers ──────────────────────────────────────────────────────────────────
def safe_encode(le, value):
    """Encode a value; fall back to closest or index 0 if unseen."""
    val = str(value).strip().lower()
    if val in le.classes_:
        return le.transform([val])[0]
    # fuzzy: find partial match
    for cls in le.classes_:
        if val in cls or cls in val:
            return le.transform([cls])[0]
    return 0  # default

def irrigation_advice(label, rainfall, temp, humidity):
    """Return human-readable irrigation advice."""
    advice = {
        "Low": {
            "frequency": "Once every 10–14 days",
            "amount_mm": "25–40 mm per session",
            "method": "Drip irrigation recommended",
            "notes": "Natural rainfall likely sufficient. Monitor soil moisture."
        },
        "Moderate": {
            "frequency": "Once every 7 days",
            "amount_mm": "40–60 mm per session",
            "method": "Sprinkler or furrow irrigation",
            "notes": "Supplement rainfall. Adjust based on soil moisture."
        },
        "High": {
            "frequency": "Every 4–5 days",
            "amount_mm": "60–80 mm per session",
            "method": "Flood or sprinkler irrigation",
            "notes": "Regular irrigation required. Check drainage to avoid waterlogging."
        },
        "Very High": {
            "frequency": "Every 2–3 days",
            "amount_mm": "80–100 mm per session",
            "method": "Continuous drip or flood irrigation",
            "notes": "High demand crop. Mulching recommended to retain moisture."
        },
    }
    return advice.get(label, advice["Moderate"])

# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "models_loaded": MODELS_LOADED,
        "version": "2.0.0",
        "endpoints": [
            "/api/health",
            "/api/predict/yield",
            "/api/predict/irrigation",
            "/api/options",
            "/api/recent"
        ]
    })

@app.route("/api/options", methods=["GET"])
def options():
    if not MODELS_LOADED:
        return jsonify({"error": "Models not loaded"}), 500
    return jsonify({
        "states":   sorted(unique_vals.get("state",   [])),
        "districts": sorted(unique_vals.get("district",[])),
        "crops":    sorted(unique_vals.get("crop",    [])),
        "seasons":  sorted(unique_vals.get("season",  [])),
    })

@app.route("/api/predict/yield", methods=["POST"])
def predict_yield():
    if not MODELS_LOADED:
        return jsonify({"error": "Models not loaded"}), 500

    try:
        data = request.get_json(force=True)

        # Build feature vector
        row = {}
        for feat in y_features:
            if feat in ["state", "district", "crop", "season"]:
                row[feat] = safe_encode(y_le[feat], data.get(feat, ""))
            else:
                row[feat] = float(data.get(feat, 0) or 0)

        X = pd.DataFrame([row])[y_features]
        X_sc = y_scaler.transform(X)

        rf_pred = float(rf_model.predict(X_sc)[0])
        gb_pred = float(gb_model.predict(X_sc)[0])
        hybrid  = 0.55 * rf_pred + 0.45 * gb_pred

        # FIX: clamp to valid range — GB can extrapolate below zero
        hybrid = round(max(0.0, min(hybrid, 500.0)), 4)

        # Confidence band ±10%
        low  = round(max(0.0, hybrid * 0.90), 4)
        high = round(hybrid * 1.10, 4)

        return jsonify({
            "yield_prediction": hybrid,
            "unit": "tons/hectare",
            "confidence_interval": {"low": low, "high": high},
            "model_r2": y_metrics.get("r2", 0),
            "input_received": data,
        })

    except Exception as e:
        log.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route("/api/predict/irrigation", methods=["POST"])
def predict_irrigation():
    if not MODELS_LOADED:
        return jsonify({"error": "Models not loaded"}), 500

    try:
        data = request.get_json(force=True)

        row = {}
        for feat in irr_features:
            if feat in irr_le:
                row[feat] = safe_encode(irr_le[feat], data.get(feat, ""))
            else:
                row[feat] = float(data.get(feat, 0) or 0)

        X = pd.DataFrame([row])[irr_features]
        X_sc = irr_scaler.transform(X)

        pred_idx   = irr_clf.predict(X_sc)[0]
        pred_label = irr_target_le.inverse_transform([pred_idx])[0]
        proba      = irr_clf.predict_proba(X_sc)[0]
        confidence = round(float(proba.max()) * 100, 1)

        rainfall = float(data.get("total_rainfall_mm", 0) or 0)
        temp     = float(data.get("AvgTemp_C", data.get("avgtemp_c", 25)) or 25)
        humidity = float(data.get("Humidity_perc", data.get("humidity_perc", 60)) or 60)

        advice = irrigation_advice(pred_label, rainfall, temp, humidity)

        return jsonify({
            "irrigation_need": pred_label,
            "confidence_pct": confidence,
            "advice": advice,
            "input_received": data,
        })

    except Exception as e:
        log.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route("/api/recent", methods=["GET"])
def recent_predictions():
    """Returns mock recent predictions for the dashboard."""
    sample = [
        {"state": "karnataka", "crop": "Rice",    "yield": 4.21, "irrigation": "Moderate"},
        {"state": "punjab",    "crop": "Wheat",   "yield": 5.83, "irrigation": "High"},
        {"state": "kerala",    "crop": "Coconut", "yield": 3.12, "irrigation": "Low"},
        {"state": "gujarat",   "crop": "Cotton",  "yield": 2.97, "irrigation": "Very High"},
        {"state": "assam",     "crop": "Tea",     "yield": 1.54, "irrigation": "Moderate"},
    ]
    return jsonify({"recent": sample})

# ─── Entry Point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    log.info("🌾 Starting Smart Harvest AI backend on http://localhost:5001")
    app.run(host="0.0.0.0", port=5001, debug=False)
