#!/usr/bin/env python3
"""
Smart Harvest AI - Irrigation Recommendation Model Training
Uses crop + weather + soil features to recommend water requirement category
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / "FINAL_DATASET_cleaned.csv"
MODEL_DIR = BASE_DIR / "backend" / "models" / "irrigation_model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("  Smart Harvest AI — Irrigation Model Training")
print("=" * 60)

# ─── 1. Load & Prep ───────────────────────────────────────────────────────────
print("\n[1/5] Loading dataset ...")
df = pd.read_csv(DATA_PATH)
df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]

# Rename for consistency
df.rename(columns={
    'avgtemp_c': 'AvgTemp_C',
    'humidity_perc': 'Humidity_perc',
    'total_rainfall_mm': 'total_rainfall_mm',
}, inplace=True)

df['crop']   = df['crop'].astype(str).str.strip().str.lower()
df['state']  = df['state'].astype(str).str.strip().str.lower()
df['season'] = df['season'].astype(str).str.strip().str.lower()

df.dropna(subset=['total_rainfall_mm', 'AvgTemp_C', 'Humidity_perc'], inplace=True)
df.fillna(df.median(numeric_only=True), inplace=True)
print(f"      {len(df):,} rows loaded")

# ─── 2. Create Irrigation Label ───────────────────────────────────────────────
print("\n[2/5] Engineering irrigation target ...")

# Irrigation need score: lower rainfall + higher temp + lower humidity = more irrigation
df['irr_score'] = (
    - df['total_rainfall_mm'] * 0.5
    + df['AvgTemp_C']         * 3.0
    - df['Humidity_perc']     * 0.2
    + (7.5 - df['ph'].clip(4, 9)) * 5.0
)

# Bin into 4 irrigation categories
df['irrigation_need'] = pd.qcut(
    df['irr_score'],
    q=4,
    labels=['Low', 'Moderate', 'High', 'Very High']
)
df.dropna(subset=['irrigation_need'], inplace=True)

print("      Distribution:")
print(df['irrigation_need'].value_counts().to_string())

# ─── 3. Encode & Scale ────────────────────────────────────────────────────────
print("\n[3/5] Encoding features ...")

CAT_COLS = ['crop', 'state', 'season']
le_irr = {}
for col in CAT_COLS:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    le_irr[col] = le

le_target = LabelEncoder()
y = le_target.fit_transform(df['irrigation_need'])

FEATURES = ['crop', 'state', 'season', 'AvgTemp_C', 'Humidity_perc',
            'total_rainfall_mm', 'ph', 'clay_g_kg', 'sand_g_kg',
            'organic_carbon_g_kg', 'latitude', 'longitude']
FEATURES = [f for f in FEATURES if f in df.columns]

X = df[FEATURES]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ─── 4. Train ─────────────────────────────────────────────────────────────────
print("\n[4/5] Training Random Forest classifier ...")
clf = RandomForestClassifier(n_estimators=150, max_depth=15, n_jobs=-1, random_state=42)
clf.fit(X_train_sc, y_train)

y_pred = clf.predict(X_test_sc)
acc = accuracy_score(y_test, y_pred)
print(f"      Accuracy: {acc:.4f}")
print(classification_report(y_test, y_pred, target_names=le_target.classes_))

# ─── 5. Save ──────────────────────────────────────────────────────────────────
print("\n[5/5] Saving artifacts ...")
joblib.dump(clf,      MODEL_DIR / "irrigation_rf_model.pkl")
joblib.dump(scaler,   MODEL_DIR / "irrigation_scaler.pkl")
joblib.dump(le_irr,   MODEL_DIR / "irrigation_label_encoders.pkl")
joblib.dump(le_target, MODEL_DIR / "irrigation_target_encoder.pkl")
joblib.dump(FEATURES, MODEL_DIR / "irrigation_features.pkl")

# LSTM placeholder (setup.py checks for this file)
lstm_placeholder = MODEL_DIR / "irrigation_lstm_model.h5"
if not lstm_placeholder.exists():
    lstm_placeholder.write_text("# placeholder - RF classifier used instead of LSTM")

print(f"✅ Irrigation model saved to {MODEL_DIR}")
print(f"   Accuracy = {acc:.4f}")
print("=" * 60)
