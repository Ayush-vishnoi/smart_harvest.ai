# 🌾 Smart Harvest AI

Link-https://ayushvishnoi.pythonanywhere.com

AI-powered crop yield prediction and irrigation recommendation system for Indian agriculture.
Trained on 235,000+ real records across 32 states, 55 crops, and 6 growing seasons.

---

## 📊 Model Performance

| Model | Metric | Score |
|-------|--------|-------|
| Yield Prediction (RF + GB Hybrid) | R² | **80.2%** |
| Irrigation Classifier (Rule based) | Accuracy | **95.97%** |

---

## 🗂 Project Structure

```
smart_harvest/
├── FINAL_DATASET_cleaned.csv        ← Training data (235K rows)
├── setup.py                         ← Automated setup script
├── start.sh / start.bat             ← Launch scripts
├── backend/
│   ├── app_v2.py                    ← Flask REST API
│   ├── requirements.txt
│   ├── ml/
│   │   ├── train_yield_model_v2.py  ← Yield model training
│   │   └── train_irrigation_model.py
│   └── models/
│       ├── yield_model/             ← Trained yield artifacts
│       └── irrigation_model/        ← Trained irrigation artifacts
└── frontend/
    └── index.html                   ← Single-page UI
```

---

## 🚀 Quick Start

### Option A — Automated (recommended)
```bash
python3 setup.py
```

### Option B — Manual

**1. Install dependencies**
```bash
pip install -r backend/requirements.txt
```

**2. Place dataset**  
Put `FINAL_DATASET_cleaned.csv` in the project root.

**3. Train models** (already done if models/ folder is populated)
```bash
python3 backend/ml/train_yield_model_v2.py
python3 backend/ml/train_irrigation_model.py
```

**4. Start backend**
```bash
cd backend && python3 app_v2.py
# Runs on http://localhost:5001
```

**5. Start frontend**
```bash
cd frontend && python3 -m http.server 8000
# Open http://localhost:8000
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Health check + model status |
| GET | `/api/options` | Dropdown values (states, crops, seasons) |
| POST | `/api/predict/yield` | Yield prediction |
| POST | `/api/predict/irrigation` | Irrigation recommendation |
| GET | `/api/recent` | Recent prediction history |

### Example — Yield Prediction
```bash
curl -X POST http://localhost:5001/api/predict/yield \
  -H "Content-Type: application/json" \
  -d '{
    "state": "karnataka",
    "district": "bangalore",
    "crop": "Rice",
    "crop_year": 2024,
    "season": "Kharif",
    "area": 1209,
    "latitude": 12.97,
    "longitude": 77.59,
    "total_rainfall_mm": 1200,
    "AvgTemp_C": 22.5,
    "Humidity_perc": 78,
    "ph": 6.5,
    "organic_carbon_g_kg": 25,
    "clay_g_kg": 250,
    "sand_g_kg": 400
  }'
```

Response:
```json
{
  "yield_prediction": 4.213,
  "unit": "tons/hectare",
  "confidence_interval": { "low": 3.791, "high": 4.634 },
  "model_r2": 0.9716
}
```

---

## 🧠 Model Details

### Yield Model
- **Algorithm**: Random Forest (200 trees) + Gradient Boosting (150 trees) Hybrid (0.6 RF + 0.4 GB)
- **Features**: state, district, crop, year, season, area, lat/lon, rainfall, temperature, humidity, pH, organic carbon, clay, sand
- **Target**: Yield (tons/hectare)
- **R²**: 0.7916

### Irrigation Model  
- **Algorithm**: Random Forest Classifier (150 trees)
- **Classes**: Low / Moderate / High / Very High
- **Accuracy**: 95.97%
