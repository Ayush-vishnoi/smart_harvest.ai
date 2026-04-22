#!/usr/bin/env python3
import sys
import os
import json
import logging
import traceback
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, Response
from frontend import FrontendController, register_frontend_routes

BASE_DIR  = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "backend" / "models"

app = Flask(__name__,
    template_folder=str(BASE_DIR / 'frontend'),
    static_folder=str(BASE_DIR / 'static')
)
app.secret_key = os.environ.get('SECRET_KEY', 'smart_harvest_ai_secret_key_2024')

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

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

YIELD_DIR = MODEL_DIR / "yield_model"
IRR_DIR   = MODEL_DIR / "irrigation_model"

try:
    rf_model    = joblib.load(YIELD_DIR / "yield_rf_model.pkl")
    gb_model    = joblib.load(YIELD_DIR / "yield_gb_model.pkl")
    y_scaler    = joblib.load(YIELD_DIR / "yield_scaler.pkl")
    y_le        = joblib.load(YIELD_DIR / "yield_label_encoders.pkl")
    y_features  = list(joblib.load(YIELD_DIR / "feature_columns.pkl"))
    unique_vals = joblib.load(YIELD_DIR / "unique_values.pkl")
    with open(YIELD_DIR / "metrics.json") as f:
        y_metrics = json.load(f)

    irr_clf      = joblib.load(IRR_DIR / "irrigation_rf_model.pkl")
    irr_scaler   = joblib.load(IRR_DIR / "irrigation_scaler.pkl")
    irr_le       = joblib.load(IRR_DIR / "irrigation_label_encoders.pkl")
    irr_target_le= joblib.load(IRR_DIR / "irrigation_target_encoder.pkl")
    irr_features = list(joblib.load(IRR_DIR / "irrigation_features.pkl"))

    MODELS_LOADED = True
    log.info("✅ All models loaded successfully")
    log.info(f"   y_le type: {type(y_le)}, irr_le type: {type(irr_le)}")

    models_dict = {
        'rf_model': rf_model, 'gb_model': gb_model, 'y_scaler': y_scaler,
        'y_le': y_le, 'y_features': y_features, 'unique_vals': unique_vals,
        'y_metrics': y_metrics, 'irr_clf': irr_clf, 'irr_scaler': irr_scaler,
        'irr_le': irr_le, 'irr_target_le': irr_target_le,
        'irr_features': irr_features, 'models_loaded': True
    }
    frontend = FrontendController(models_dict)
    register_frontend_routes(app, frontend)

except Exception as e:
    MODELS_LOADED = False
    log.error(f"⚠️ Failed to load models: {e}")
    log.error(traceback.format_exc())
    models_dict = {
        'rf_model': None, 'gb_model': None, 'y_scaler': None,
        'y_le': {}, 'y_features': [], 'unique_vals': {'state': [], 'district': [], 'crop': [], 'season': []},
        'y_metrics': {'r2': 0, 'mae': 0}, 'irr_clf': None, 'irr_scaler': None,
        'irr_le': {}, 'irr_target_le': None,
        'irr_features': [], 'models_loaded': False
    }
    frontend = FrontendController(models_dict)
    register_frontend_routes(app, frontend)

def safe_encode(le, value):
    val = str(value).strip().lower()
    if val in le.classes_:
        return int(le.transform([val])[0])
    for cls in le.classes_:
        if val in cls or cls in val:
            return int(le.transform([cls])[0])
    return 0

def irrigation_advice(label, rainfall, temp, humidity):
    advice = {
        "Low":      {"frequency": "Once every 10–14 days", "amount_mm": "25–40 mm",  "method": "Drip irrigation",           "notes": "Natural rainfall likely sufficient."},
        "Moderate": {"frequency": "Once every 7 days",     "amount_mm": "40–60 mm",  "method": "Sprinkler or furrow",        "notes": "Supplement rainfall as needed."},
        "High":     {"frequency": "Every 4–5 days",        "amount_mm": "60–80 mm",  "method": "Flood or sprinkler",         "notes": "Regular irrigation required."},
        "Very High":{"frequency": "Every 2–3 days",        "amount_mm": "80–100 mm", "method": "Continuous drip or flood",   "notes": "High demand. Mulching recommended."},
    }
    return advice.get(label, advice["Moderate"])

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "models_loaded": MODELS_LOADED, "version": "2.0.0"})

@app.route("/api/options", methods=["GET"])
def options():
    if not MODELS_LOADED:
        return jsonify({"error": "Models not loaded"}), 500
    return jsonify({
        "states":    sorted(unique_vals.get("state",    [])),
        "districts": sorted(unique_vals.get("district", [])),
        "crops":     sorted(unique_vals.get("crop",     [])),
        "seasons":   sorted(unique_vals.get("season",   [])),
    })

@app.route("/api/predict/yield", methods=["POST"])
def predict_yield():
    if not MODELS_LOADED:
        return jsonify({"error": "Models not loaded"}), 500
    try:
        data = request.get_json(force=True)
        row = {}
        for feat in y_features:
            if feat in ["state", "district", "crop", "season"]:
                row[feat] = safe_encode(y_le[feat], data.get(feat, ""))
            else:
                row[feat] = float(data.get(feat, 0) or 0)
        X = np.array([[row[f] for f in y_features]], dtype=float)
        X_sc = y_scaler.transform(X)
        hybrid = max(0.0, min(0.55 * float(rf_model.predict(X_sc)[0]) + 0.45 * float(gb_model.predict(X_sc)[0]), 500.0))
        return jsonify({"yield_prediction": round(hybrid, 4), "unit": "tons/hectare",
                        "confidence_interval": {"low": round(hybrid*0.9, 4), "high": round(hybrid*1.1, 4)},
                        "model_r2": y_metrics.get("r2", 0)})
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
        X = np.array([[row[f] for f in irr_features]], dtype=float)
        X_sc = irr_scaler.transform(X)
        pred_label = irr_target_le.inverse_transform([irr_clf.predict(X_sc)[0]])[0]
        confidence = round(float(irr_clf.predict_proba(X_sc)[0].max()) * 100, 1)
        return jsonify({"irrigation_need": pred_label, "confidence_pct": confidence,
                        "advice": irrigation_advice(pred_label, 0, 0, 0)})
    except Exception as e:
        log.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route("/api/recent", methods=["GET"])
def recent_predictions():
    return jsonify({"recent": [
        {"state": "karnataka", "crop": "Rice",   "yield": 4.21, "irrigation": "Moderate"},
        {"state": "punjab",    "crop": "Wheat",  "yield": 5.83, "irrigation": "High"},
        {"state": "gujarat",   "crop": "Cotton", "yield": 2.97, "irrigation": "Very High"},
    ]})

@app.route("/api/location-data", methods=["GET"])
def location_data():
    try:
        lat = float(request.args.get('lat', 20.5))
        lon = float(request.args.get('lon', 78.9))
        if lat > 30:
            state = 'punjab' if lon < 77 else 'haryana'
        elif lat > 28:
            state = 'rajasthan' if lon < 76 else ('haryana' if lon < 77.5 else 'uttar pradesh')
        elif lat > 26:
            state = 'rajasthan' if lon < 76 else ('uttar pradesh' if lon < 82 else 'bihar')
        elif lat > 23:
            state = 'gujarat' if lon < 74 else ('madhya pradesh' if lon < 78 else ('chhattisgarh' if lon < 82 else 'west bengal'))
        elif lat > 20:
            state = 'maharashtra' if lon < 74 else ('telangana' if lon < 78 else 'odisha')
        elif lat > 17:
            state = 'karnataka' if lon < 78 else 'andhra pradesh'
        elif lat > 13:
            state = 'karnataka' if lon < 76 else 'tamil nadu'
        elif lat > 10:
            state = 'kerala' if lon < 76 else 'tamil nadu'
        else:
            state = 'kerala'

        if lat < 15:
            d = {'rainfall':1200,'temperature':27,'humidity':75,'ph':6.2,'organic_carbon':22,'clay':280,'sand':380,'region':'South India'}
        elif lat > 28:
            d = {'rainfall':650,'temperature':22,'humidity':60,'ph':7.2,'organic_carbon':18,'clay':220,'sand':450,'region':'North India'}
        elif lon < 75:
            d = {'rainfall':550,'temperature':26,'humidity':55,'ph':7.5,'organic_carbon':15,'clay':200,'sand':500,'region':'West India'}
        elif lon > 85:
            d = {'rainfall':1800,'temperature':26,'humidity':80,'ph':5.8,'organic_carbon':25,'clay':300,'sand':350,'region':'East India'}
        else:
            d = {'rainfall':900,'temperature':25,'humidity':65,'ph':6.8,'organic_carbon':20,'clay':250,'sand':400,'region':'Central India'}

        d.update({'success': True, 'latitude': lat, 'longitude': lon, 'state': state})
        return jsonify(d)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=False)
