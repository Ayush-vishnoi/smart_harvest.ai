#!/usr/bin/env python3
"""
Smart Harvest AI - Backend API + Frontend Controller
Serves both JSON API (for JS) and HTML pages (for Python templates)
"""

# FIRST: Fix numpy compatibility issue before ANY numpy.random usage
# This must happen BEFORE joblib/sklearn imports
import sys
import os

# Remove any sklearn/numpy from cache to force reimport with our patch
mods_to_remove = [k for k in sys.modules.keys() if 'numpy' in k or 'sklearn' in k or 'joblib' in k]
for mod in mods_to_remove:
    if mod in sys.modules:
        del sys.modules[mod]

# Now fix numpy FIRST before any other imports
import numpy as np
from numpy.random import MT19937

# Create proper _mt19937 module for pickle BEFORE anything else
# The issue is the pickled models have old BitGenerator references
# We need to provide a module that returns the current MT19937

# Patch numpy.random._generator to accept old BitGenerator classes
def _patch_bitgenerator_validation():
    """Patch numpy's internal BitGenerator validation"""
    try:
        import numpy.random._generator as gen
        
        # Store original function
        orig_check = getattr(gen, '_check_bit_generator', None)
        
        if orig_check:
            def patched_check(bg):
                # If it's the old class string reference, return default
                if isinstance(bg, type) and 'MT19937' in str(bg):
                    return np.random.default_bit_generator
                try:
                    return orig_check(bg)
                except (TypeError, AttributeError):
                    return np.random.default_bit_generator
            
            gen._check_bit_generator = patched_check
    except Exception:
        pass

_patch_bitgenerator_validation()

# Create proper _mt19937 module for pickle
class _mt19937_module:
    MT19937 = np.random.MT19937
    
sys.modules['numpy.random._mt19937'] = _mt19937_module()

# Also patch mtrand
class _mtrand_module:
    MT19937 = np.random.MT19937
    
sys.modules['numpy.random.mtrand'] = _mtrand_module()

# Patch numpy.random
import numpy.random
if not hasattr(numpy.random, 'MT19937'):
    numpy.random.MT19937 = MT19937

# PATCH PICKLE TO HANDLE OLD NUMPY BITGENERATORS
import pickle
import copyreg

# Simple pickle fix - just ensure MT19937 can be found
# We'll just ensure MT19937 is in the right place for the unpickler to find

# PATCH THE NUMPY INTERNAL FUNCTIONS DIRECTLY
# This must happen before any numpy.random usage
def _patch_numpy_bitgenerator():
    """Patch numpy to accept old BitGenerator references - legacy function"""
    pass  # Already patched above

_patch_numpy_bitgenerator()

# Create patched Unpickler
class PatchedUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # Handle old numpy.random._mt19937.MT19937 references
        if module == 'numpy.random._mt19937' and name == 'MT19937':
            return MT19937
        if module == 'numpy.random.mtrand' and name == 'MT19937':
            return MT19937
        # Handle old BitGenerator references
        if 'BitGenerator' in name:
            return MT19937
        return super().find_class(module, name)

# Patch pickle to use our custom unpickler globally
_original_unpickler = pickle.Unpickler

# Also create a proper _mt19937 module for pickle
class _mt19937_module:
    MT19937 = MT19937

# Register the module
sys.modules['numpy.random._mt19937'] = _mt19937_module()

# Now patch joblib to use our custom unpickler
import joblib

# Save original load function
_original_joblib_load = joblib.load

def patched_joblib_load(filename, mmap_mode=None):
    """Load a joblib file with numpy compatibility patches"""
    import io
    
    filename = joblib.os.path.abspath(filename)
    
    with open(filename, "rb") as f:
        # Try to load using the patched unpickler
        try:
            unpickler = PatchedUnpickler(f)
            obj = unpickler.load()
            return obj
        except Exception as e:
            # If our patch fails, try original
            pass
    
    # Fallback to original
    return _original_joblib_load(filename, mmap_mode)

# Replace joblib.load with patched version
joblib.load = patched_joblib_load

# Now import the rest
import json
import logging
import traceback
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request, Response

# Import our separate frontend controller (same folder)
from frontend import FrontendController, register_frontend_routes

# ─── Setup ────────────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "backend" / "models"

app = Flask(__name__, 
    template_folder=str(BASE_DIR / 'frontend'),
    static_folder=str(BASE_DIR / 'static')
)
app.secret_key = os.environ.get('SECRET_KEY', 'smart_harvest_ai_secret_key_2024')

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ─── CORS ─────────────────────────────────────────────────────────────────────
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

# ─── Load Models ──────────────────────────────────────────────────────────────
YIELD_DIR = MODEL_DIR / "yield_model"
IRR_DIR   = MODEL_DIR / "irrigation_model"

try:
    # Yield models
    rf_model  = joblib.load(YIELD_DIR / "yield_rf_model.pkl")
    gb_model  = joblib.load(YIELD_DIR / "yield_gb_model.pkl")
    y_scaler  = joblib.load(YIELD_DIR / "yield_scaler.pkl")
    y_le      = joblib.load(YIELD_DIR / "yield_label_encoders.pkl")
    y_features= joblib.load(YIELD_DIR / "feature_columns.pkl")
    unique_vals = joblib.load(YIELD_DIR / "unique_values.pkl")
    with open(YIELD_DIR / "metrics.json") as f:
        y_metrics = json.load(f)
    
    # Irrigation models
    irr_clf   = joblib.load(IRR_DIR / "irrigation_rf_model.pkl")
    irr_scaler= joblib.load(IRR_DIR / "irrigation_scaler.pkl")
    irr_le    = joblib.load(IRR_DIR / "irrigation_label_encoders.pkl")
    irr_target_le = joblib.load(IRR_DIR / "irrigation_target_encoder.pkl")
    irr_features  = joblib.load(IRR_DIR / "irrigation_features.pkl")
    
    MODELS_LOADED = True
    log.info("✅ All models loaded successfully")
    
    # Initialize frontend controller
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
    log.info("📄 Running in demo mode without ML models")
    # Initialize frontend controller with empty models for demo mode
    models_dict = {
        'rf_model': None, 'gb_model': None, 'y_scaler': None,
        'y_le': {}, 'y_features': [], 'unique_vals': {'state': [], 'district': [], 'crop': [], 'season': []},
        'y_metrics': {'r2': 0, 'mae': 0}, 'irr_clf': None, 'irr_scaler': None,
        'irr_le': {}, 'irr_target_le': None,
        'irr_features': [], 'models_loaded': False
    }
    frontend = FrontendController(models_dict)
    register_frontend_routes(app, frontend)

# ─── Helpers (for API) ────────────────────────────────────────────────────────
def safe_encode(le, value):
    val = str(value).strip().lower()
    if val in le.classes_:
        return le.transform([val])[0]
    for cls in le.classes_:
        if val in cls or cls in val:
            return le.transform([cls])[0]
    return 0

def irrigation_advice(label, rainfall, temp, humidity):
    advice = {
        "Low": {
            "frequency": "Once every 10–14 days", "amount_mm": "25–40 mm per session",
            "method": "Drip irrigation recommended",
            "notes": "Natural rainfall likely sufficient. Monitor soil moisture."
        },
        "Moderate": {
            "frequency": "Once every 7 days", "amount_mm": "40–60 mm per session",
            "method": "Sprinkler or furrow irrigation",
            "notes": "Supplement rainfall. Adjust based on soil moisture."
        },
        "High": {
            "frequency": "Every 4–5 days", "amount_mm": "60–80 mm per session",
            "method": "Flood or sprinkler irrigation",
            "notes": "Regular irrigation required. Check drainage to avoid waterlogging."
        },
        "Very High": {
            "frequency": "Every 2–3 days", "amount_mm": "80–100 mm per session",
            "method": "Continuous drip or flood irrigation",
            "notes": "High demand crop. Mulching recommended to retain moisture."
        },
    }
    return advice.get(label, advice["Moderate"])

# ══════════════════════════════════════════════════════════════════════════════
# JSON API ROUTES (For JavaScript Frontend)
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "models_loaded": MODELS_LOADED,
        "version": "2.0.0",
        "endpoints": ["/api/health", "/api/predict/yield", "/api/predict/irrigation", "/api/options", "/api/recent"]
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
        hybrid  = max(0.0, min(hybrid, 500.0))
        hybrid  = round(hybrid, 4)

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
    sample = [
        {"state": "karnataka", "crop": "Rice",    "yield": 4.21, "irrigation": "Moderate"},
        {"state": "punjab",    "crop": "Wheat",   "yield": 5.83, "irrigation": "High"},
        {"state": "kerala",    "crop": "Coconut", "yield": 3.12, "irrigation": "Low"},
        {"state": "gujarat",   "crop": "Cotton",  "yield": 2.97, "irrigation": "Very High"},
        {"state": "assam",     "crop": "Tea",     "yield": 1.54, "irrigation": "Moderate"},
    ]
    return jsonify({"recent": sample})

@app.route("/api/location-data", methods=["GET"])
def location_data():
    """Fetch climate and soil data based on GPS coordinates"""
    try:
        lat = float(request.args.get('lat', 20.5))
        lon = float(request.args.get('lon', 78.9))
        
        # Detect state based on coordinates
        state = 'karnataka'  # default
        
        # North India states
        if lat > 30:
            if lon < 75: state = 'jammu and kashmir'
            elif lon < 77: state = 'punjab'
            elif lon < 78.5: state = 'haryana'
            else: state = 'himachal pradesh'
        elif lat > 28:
            if lon < 76: state = 'rajasthan'
            elif lon < 77.5: state = 'haryana'
            elif lon < 80: state = 'uttar pradesh'
            else: state = 'uttarakhand'
        # Central-North
        elif lat > 26:
            if lon < 76: state = 'rajasthan'
            elif lon < 82: state = 'uttar pradesh'
            elif lon < 85: state = 'madhya pradesh'
            else: state = 'bihar'
        # Central India
        elif lat > 23:
            if lon < 74: state = 'gujarat'
            elif lon < 78: state = 'madhya pradesh'
            elif lon < 82: state = 'chhattisgarh'
            elif lon < 86: state = 'jharkhand'
            else: state = 'west bengal'
        # Central-South
        elif lat > 20:
            if lon < 74: state = 'maharashtra'
            elif lon < 78: state = 'telangana'
            elif lon < 82: state = 'odisha'
            else: state = 'west bengal'
        # South India
        elif lat > 17:
            if lon < 75: state = 'goa'
            elif lon < 78: state = 'karnataka'
            elif lon < 80: state = 'andhra pradesh'
            else: state = 'andhra pradesh'
        elif lat > 13:
            if lon < 76: state = 'karnataka'
            elif lon < 78: state = 'tamil nadu'
            else: state = 'tamil nadu'
        elif lat > 10:
            if lon < 76: state = 'kerala'
            else: state = 'tamil nadu'
        else:
            state = 'kerala'
        
        # Regional climate/soil based on location
        if lat < 15:  # South
            data = {'rainfall': 1200, 'temperature': 27, 'humidity': 75,
                    'ph': 6.2, 'organic_carbon': 22, 'clay': 280, 'sand': 380, 'region': 'South India'}
        elif lat > 28:  # North
            data = {'rainfall': 650, 'temperature': 22, 'humidity': 60,
                    'ph': 7.2, 'organic_carbon': 18, 'clay': 220, 'sand': 450, 'region': 'North India'}
        elif lon < 75:  # West
            data = {'rainfall': 550, 'temperature': 26, 'humidity': 55,
                    'ph': 7.5, 'organic_carbon': 15, 'clay': 200, 'sand': 500, 'region': 'West India'}
        elif lon > 85:  # East
            data = {'rainfall': 1800, 'temperature': 26, 'humidity': 80,
                    'ph': 5.8, 'organic_carbon': 25, 'clay': 300, 'sand': 350, 'region': 'East India'}
        else:  # Central
            data = {'rainfall': 900, 'temperature': 25, 'humidity': 65,
                    'ph': 6.8, 'organic_carbon': 20, 'clay': 250, 'sand': 400, 'region': 'Central India'}
        
        data['success'] = True
        data['latitude'] = lat
        data['longitude'] = lon
        data['state'] = state
        return jsonify(data)
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

# ─── Entry Point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    log.info("🌾 Starting Smart Harvest AI")
    log.info(f"   API (JSON): http://0.0.0.0:{port}/api/health")
    log.info(f"   Web (HTML): http://0.0.0.0:{port}/web")
    log.info("   Frontend logic separated into: backend/frontend.py")
    app.run(host="0.0.0.0", port=port, debug=False)
