#!/usr/bin/env python3
import sys
import os
import json
import logging
import traceback
import threading
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

try:
    from dotenv import load_dotenv

    load_dotenv(BASE_DIR / ".env")
except ImportError:
    pass

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, Response
from frontend import FrontendController, register_frontend_routes
from routes.chatbot import chatbot_bp
from utils.database import database
from yield_inference import YieldInference

try:
    from PIL import Image
except ImportError:  # Disease inference remains optional until Pillow is installed.
    Image = None

try:
    import tensorflow as tf
except ImportError:  # The yield service must still be able to start without TensorFlow.
    tf = None

MODEL_DIR = BASE_DIR / "models"
DISEASE_MODEL_DIR = MODEL_DIR
DISEASE_IMAGE_SIZE = (192, 192)

app = Flask(__name__,
    template_folder=str(BASE_DIR / 'frontend'),
    static_folder=str(BASE_DIR / 'static')
)
app.secret_key = os.environ.get('SECRET_KEY', 'smart_harvest_ai_secret_key_2024')
database.init_app(app)
app.register_blueprint(chatbot_bp)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

@app.after_request
def add_cors(response):
    response.headers["Access-Control-Allow-Origin"]  = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    # Prevent the browser from showing an older cached yield form after model/UI updates.
    if response.mimetype == "text/html":
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response

@app.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        return Response(status=200)

YIELD_DIR = MODEL_DIR / "yield_model"
DISEASE_TFLITE_PATH = DISEASE_MODEL_DIR / "disease_model.tflite"
DISEASE_LABELS_PATH = DISEASE_MODEL_DIR / "class_labels.json"

DISEASE_LOADED = False
_disease_interpreter = None
_disease_input = None
_disease_output = None
_disease_labels = {}
_disease_lock = threading.Lock()
try:
    if tf is None or Image is None:
        raise RuntimeError("TensorFlow and Pillow are required for disease inference")
    with open(DISEASE_LABELS_PATH, encoding="utf-8") as f:
        _disease_labels = json.load(f)
    _disease_interpreter = tf.lite.Interpreter(model_path=str(DISEASE_TFLITE_PATH))
    _disease_interpreter.allocate_tensors()
    _disease_input = _disease_interpreter.get_input_details()[0]
    _disease_output = _disease_interpreter.get_output_details()[0]
    DISEASE_LOADED = True
    log.info("✅ Disease model loaded successfully")
except Exception as e:
    log.warning("Disease model unavailable: %s", e)

try:
    yield_service = YieldInference(YIELD_DIR)
    unique_vals = yield_service.unique_values
    y_metrics = yield_service.metrics

    MODELS_LOADED = True
    log.info("✅ All models loaded successfully")

    models_dict = {
        'yield_service': yield_service,
        'unique_vals': unique_vals,
        'y_metrics': y_metrics,
        'models_loaded': True,
    }
    frontend = FrontendController(models_dict)
    register_frontend_routes(app, frontend)

except Exception as e:
    MODELS_LOADED = False
    log.error(f"⚠️ Failed to load models: {e}")
    log.error(traceback.format_exc())
    yield_service = None
    unique_vals = {'state': [], 'crop': [], 'season': []}
    y_metrics = {'r2': 0, 'mae': 0}
    models_dict = {
        'yield_service': None,
        'unique_vals': unique_vals,
        'y_metrics': y_metrics,
        'models_loaded': False,
    }
    frontend = FrontendController(models_dict)
    register_frontend_routes(app, frontend)

def _predict_disease(image):
    if not DISEASE_LOADED:
        raise RuntimeError("Disease model is not available")
    image = image.convert("RGB").resize(DISEASE_IMAGE_SIZE)
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = np.expand_dims(array, axis=0)
    input_dtype = _disease_input["dtype"]
    if input_dtype != np.float32:
        scale, zero_point = _disease_input["quantization"]
        array = np.round(array / scale + zero_point).astype(input_dtype)
    with _disease_lock:
        _disease_interpreter.set_tensor(_disease_input["index"], array)
        _disease_interpreter.invoke()
        output = _disease_interpreter.get_tensor(_disease_output["index"])[0]
    if _disease_output["dtype"] != np.float32:
        scale, zero_point = _disease_output["quantization"]
        output = (output.astype(np.float32) - zero_point) * scale
    probabilities = np.asarray(output, dtype=np.float32)
    if probabilities.ndim != 1:
        probabilities = probabilities.reshape(-1)
    if probabilities.min() < 0 or probabilities.max() > 1.001 or not np.isclose(probabilities.sum(), 1, atol=0.02):
        probabilities = np.exp(probabilities - probabilities.max())
        probabilities /= probabilities.sum()
    indices = np.argsort(probabilities)[::-1][:3]
    predictions = [{"label": _disease_labels.get(str(int(i)), f"class_{i}"), "confidence": round(float(probabilities[i]) * 100, 2)} for i in indices]
    return predictions


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "models_loaded": MODELS_LOADED,
        "yield_model_loaded": MODELS_LOADED,
        "disease_model_loaded": DISEASE_LOADED,
        "database_connected": database.is_healthy(),
        "database_backend": database.backend_name,
        "version": "2.1.0"
    })

@app.route("/api/options", methods=["GET"])
def options():
    if not MODELS_LOADED:
        return jsonify({"error": "Models not loaded"}), 500
    return jsonify(yield_service.options())

@app.route("/api/predict/yield", methods=["POST"])
def predict_yield():
    if not MODELS_LOADED:
        return jsonify({"error": "Models not loaded"}), 500
    try:
        data = request.get_json(silent=True)
        if data is None:
            raise ValueError("Content-Type must be application/json with a valid JSON object")
        if not isinstance(data, dict):
            raise ValueError("JSON object required")
        pred, raw_pred = yield_service.predict(data)
        response = {"yield_prediction": round(pred, 4), "unit": "tons/hectare",
                    "raw_prediction": round(raw_pred, 4),
                    "confidence_interval": {"low": round(max(0, pred-y_metrics.get("mae", 0)), 4), "high": round(pred+y_metrics.get("mae", 0), 4)},
                    "model_r2": y_metrics.get("r2", 0), "model_mae": y_metrics.get("mae", 0)}
        database.save_prediction("yield", data, response, result_value=pred)
        return jsonify(response)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        log.error(traceback.format_exc())
        return jsonify({"error": "Prediction failed"}), 500

@app.route("/api/predict/disease", methods=["POST"])
def predict_disease():
    if not DISEASE_LOADED:
        return jsonify({"error": "Disease model is not available"}), 503
    upload = request.files.get("image")
    if upload is None or not upload.filename:
        return jsonify({"error": "Upload an image using the 'image' field"}), 400
    try:
        image = Image.open(upload.stream)
        predictions = _predict_disease(image)
        response = {"success": True, "prediction": predictions[0], "top_predictions": predictions}
        database.save_prediction(
            "disease", {"filename": upload.filename}, response,
            result_label=predictions[0]["label"], confidence=predictions[0]["confidence"],
        )
        return jsonify(response)
    except Exception as e:
        log.warning("Disease prediction failed: %s", e)
        return jsonify({"error": "Invalid or unsupported image"}), 400

@app.route("/api/predict/irrigation", methods=["POST"])
def predict_irrigation():
    return jsonify({"error": "Irrigation prediction removed"}), 410

@app.route("/api/recent", methods=["GET"])
def recent_predictions():
    return jsonify({"recent": database.recent_predictions(request.args.get("limit", 10, type=int))})

STATE_CENTROIDS = {
    "andhra pradesh": (15.9129, 79.7400), "assam": (26.2006, 92.9376),
    "bihar": (25.0961, 85.3131), "chhattisgarh": (21.2787, 81.8661),
    "goa": (15.2993, 74.1240), "gujarat": (22.2587, 71.1924),
    "haryana": (29.0588, 76.0856), "himachal pradesh": (31.1048, 77.1734),
    "jammu and kashmir": (33.7782, 76.5762), "jharkhand": (23.6102, 85.2799),
    "karnataka": (15.3173, 75.7139), "kerala": (10.8505, 76.2711),
    "madhya pradesh": (22.9734, 78.6569), "maharashtra": (19.7515, 75.7139),
    "manipur": (24.6637, 93.9063), "meghalaya": (25.4670, 91.3662),
    "mizoram": (23.1645, 92.9376), "nagaland": (26.1584, 94.5624),
    "odisha": (20.9517, 85.0985), "orissa": (20.9517, 85.0985),
    "punjab": (31.1471, 75.3412), "rajasthan": (27.0238, 74.2179),
    "sikkim": (27.5330, 88.5122), "tamil nadu": (11.1271, 78.6569),
    "telangana": (18.1124, 79.0193), "tripura": (23.9408, 91.9882),
    "uttar pradesh": (26.8467, 80.9462), "uttarakhand": (30.0668, 79.0193),
    "west bengal": (22.9868, 87.8550),
}

@app.route("/api/state-rainfall", methods=["GET"])
def state_rainfall():
    """Fetch the latest available 30-day rainfall total for a state's centroid."""
    try:
        from datetime import date, timedelta
        import urllib.parse
        import urllib.request

        state = request.args.get("state", "").strip().lower()
        if state not in STATE_CENTROIDS:
            return jsonify({"success": False, "error": "Rainfall data is unavailable for this state"}), 400
        latitude, longitude = STATE_CENTROIDS[state]
        end_date = date.today() - timedelta(days=3)
        start_date = end_date - timedelta(days=30)
        params = urllib.parse.urlencode({
            "latitude": latitude, "longitude": longitude,
            "start_date": start_date.isoformat(), "end_date": end_date.isoformat(),
            "daily": "precipitation_sum", "timezone": "auto",
        })
        with urllib.request.urlopen(
            f"https://archive-api.open-meteo.com/v1/archive?{params}", timeout=8
        ) as response:
            data = json.loads(response.read())
        values = data.get("daily", {}).get("precipitation_sum", [])
        rainfall = round(sum(value for value in values if value is not None), 1)
        if rainfall <= 0:
            raise ValueError("No rainfall observations returned")
        # The model was trained with annual rainfall, so convert the observed
        # 30-day amount to an annualized rate while showing the monthly amount.
        annualized_rainfall = round(rainfall * 365 / 30, 1)
        return jsonify({
            "success": True, "state": state, "rainfall_mm": rainfall,
            "annualized_rainfall_mm": annualized_rainfall,
            "period": f"{start_date.isoformat()} to {end_date.isoformat()}",
            "source": "Open-Meteo historical weather data",
        })
    except Exception as e:
        log.warning("State rainfall lookup failed: %s", e)
        return jsonify({"success": False, "error": "Live rainfall is temporarily unavailable"}), 503

@app.route("/api/weather", methods=["GET"])
def weather_forecast():
    try:
        import urllib.request
        lat = float(request.args.get('lat', 20.5))
        lon = float(request.args.get('lon', 78.9))
        url = (f"https://api.open-meteo.com/v1/forecast"
               f"?latitude={lat}&longitude={lon}"
               f"&daily=precipitation_sum"
               f"&forecast_days=7&timezone=auto")
        with urllib.request.urlopen(url, timeout=5) as resp:
            data = json.loads(resp.read())
        daily_rain = data["daily"]["precipitation_sum"]
        total_7d   = round(sum(v for v in daily_rain if v is not None), 1)
        return jsonify({"success": True, "forecast_7d_mm": total_7d, "daily_mm": daily_rain})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

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

app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=False)
