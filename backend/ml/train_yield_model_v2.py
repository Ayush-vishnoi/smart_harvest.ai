#!/usr/bin/env python3
"""
Smart Harvest AI - Yield Prediction Model Training (v3 - Overfitting Fixed)

Fixes applied:
  1. LEAKAGE REMOVED  — 'production' column dropped (yield = production/area, so
                         including it let the model trivially reconstruct the target)
  2. UNIT OUTLIER     — Coconut excluded (measured in nuts/tree, not tons/ha)
  3. TIME-BASED SPLIT — train on ≤2015, test on >2015 (realistic future prediction)
  4. REGULARIZATION   — max_depth + min_samples_leaf constraints on RF
  5. TIGHTER OUTLIER  — 2%/98% percentile clip instead of 1%/99%
"""

import json, joblib, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / "FINAL_DATASET_cleaned.csv"
MODEL_DIR = BASE_DIR / "backend" / "models" / "yield_model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("  Smart Harvest AI — Yield Model (Overfitting Fixed)")
print("=" * 60)

# ── 1. Load ────────────────────────────────────────────────────────────────────
print(f"\n[1/8] Loading dataset ...")
df = pd.read_csv(DATA_PATH)
df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
df.rename(columns={'avgtemp_c':'AvgTemp_C','humidity_perc':'Humidity_perc'}, inplace=True)
print(f"      {len(df):,} rows loaded")

# ── 2. Fix: Remove leakage column ─────────────────────────────────────────────
print("\n[2/8] Removing data leakage ...")
if 'production' in df.columns:
    df.drop(columns=['production'], inplace=True)
    print("      Dropped 'production' (yield = production/area — direct leakage)")

# ── 3. Fix: Remove unit outlier ───────────────────────────────────────────────
print("\n[3/8] Removing unit outliers ...")
df['crop'] = df['crop'].astype(str).str.strip().str.lower()
coconut_rows = df['crop'].str.contains('coconut', na=False)
print(f"      Removed {coconut_rows.sum():,} Coconut rows (nuts/tree != tons/ha)")
df = df[~coconut_rows]

# ── 4. Clean ───────────────────────────────────────────────────────────────────
print("\n[4/8] Cleaning data ...")
CAT_COLS = ['state', 'district', 'crop', 'season']
for col in CAT_COLS:
    df[col] = df[col].astype(str).str.strip().str.lower()

df.dropna(subset=['yield'], inplace=True)
df.fillna(df.median(numeric_only=True), inplace=True)

q_low  = df['yield'].quantile(0.02)
q_high = df['yield'].quantile(0.98)
before = len(df)
df = df[(df['yield'] >= q_low) & (df['yield'] <= q_high)]
print(f"      After cleaning: {len(df):,} rows (removed {before - len(df):,} outliers)")
print(f"      Yield range: {df['yield'].min():.3f} to {df['yield'].max():.3f} t/ha")

# FIX: Remove corrupt NASA weather data (temp below 0°C is impossible for Indian farmland)
if 'AvgTemp_C' in df.columns:
    bad_temp = (df['AvgTemp_C'] < 0) | (df['AvgTemp_C'] > 45)
    print(f"      Removed {bad_temp.sum():,} rows with corrupt temperature values")
    df = df[~bad_temp]

# ── 5. Encode ──────────────────────────────────────────────────────────────────
print("\n[5/8] Encoding categorical features ...")
le_dict = {}
for col in CAT_COLS:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    le_dict[col] = le
    print(f"      {col}: {len(le.classes_)} unique values")

# ── 6. Fix: Time-based split ───────────────────────────────────────────────────
print("\n[6/8] Splitting by time (train <=2015, test >2015) ...")
FEATURES = ['state', 'district', 'crop', 'crop_year', 'season',
            'area', 'latitude', 'longitude', 'total_rainfall_mm',
            'AvgTemp_C', 'Humidity_perc', 'ph',
            'organic_carbon_g_kg', 'clay_g_kg', 'sand_g_kg']
FEATURES = [f for f in FEATURES if f in df.columns]

X_train = df[df['crop_year'] <= 2015][FEATURES]
X_test  = df[df['crop_year'] >  2015][FEATURES]
y_train = df[df['crop_year'] <= 2015]['yield']
y_test  = df[df['crop_year'] >  2015]['yield']

print(f"      Train: {len(X_train):,} rows (years 1997-2015)")
print(f"      Test : {len(X_test):,} rows (years 2016-2020)")

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ── 7. Fix: Regularized models ────────────────────────────────────────────────
print("\n[7/8] Training regularized models ...")

print("      Training Random Forest ...")
rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=12,
    min_samples_leaf=20,
    max_features=0.7,
    n_jobs=-1,
    random_state=42
)
rf.fit(X_train_sc, y_train)

print("      Training Gradient Boosting ...")
gb = GradientBoostingRegressor(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    min_samples_leaf=20,
    subsample=0.8,
    random_state=42
)
gb.fit(X_train_sc, y_train)

rf_pred     = rf.predict(X_test_sc)
gb_pred     = gb.predict(X_test_sc)
hybrid_pred = 0.55 * rf_pred + 0.45 * gb_pred

# ── Evaluate ──────────────────────────────────────────────────────────────────
print("\n      Results:")
rf_train_r2 = r2_score(y_train, rf.predict(X_train_sc))
rf_test_r2  = r2_score(y_test,  rf_pred)
gb_test_r2  = r2_score(y_test,  gb_pred)
mae  = mean_absolute_error(y_test, hybrid_pred)
rmse = np.sqrt(mean_squared_error(y_test, hybrid_pred))
r2   = r2_score(y_test, hybrid_pred)
gap  = rf_train_r2 - rf_test_r2

print(f"      RF  — Train R2={rf_train_r2:.4f}  Test R2={rf_test_r2:.4f}  Gap={gap:.4f}")
print(f"      GB  — Test R2={gb_test_r2:.4f}")
print(f"      Hybrid — Test R2={r2:.4f}  MAE={mae:.4f}  RMSE={rmse:.4f}")

if gap < 0.05:
    print(f"      Overfitting check: HEALTHY (gap={gap:.4f} < 0.05)")
elif gap < 0.10:
    print(f"      Overfitting check: MILD (gap={gap:.4f})")
else:
    print(f"      Overfitting check: WARNING (gap={gap:.4f} > 0.10)")

# ── 8. Save ────────────────────────────────────────────────────────────────────
print("\n[8/8] Saving model artifacts ...")

joblib.dump(rf,       MODEL_DIR / "yield_rf_model.pkl")
joblib.dump(gb,       MODEL_DIR / "yield_gb_model.pkl")
joblib.dump(scaler,   MODEL_DIR / "yield_scaler.pkl")
joblib.dump(le_dict,  MODEL_DIR / "yield_label_encoders.pkl")
joblib.dump(FEATURES, MODEL_DIR / "feature_columns.pkl")

unique_vals = {col: sorted(le_dict[col].classes_.tolist()) for col in CAT_COLS}
joblib.dump(unique_vals, MODEL_DIR / "unique_values.pkl")

with open(MODEL_DIR / "yield_hybrid_weights.txt", "w") as f:
    f.write("rf=0.55\ngb=0.45\n")

metrics = {
    "mae": round(mae, 4), "rmse": round(rmse, 4), "r2": round(r2, 4),
    "overfitting_gap": round(gap, 4),
    "train_years": "1997-2015", "test_years": "2016-2020",
    "leakage_removed": True, "coconut_excluded": True
}
with open(MODEL_DIR / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

keras_ph = MODEL_DIR / "yield_nn_model.keras"
if not keras_ph.exists():
    keras_ph.write_text("# placeholder")

print(f"\nAll artifacts saved to {MODEL_DIR}")
print(f"Hybrid Test R2={r2:.4f}  |  Train-Test Gap={gap:.4f}")
print("=" * 60)
