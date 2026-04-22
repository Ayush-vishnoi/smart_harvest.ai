#!/usr/bin/env python3
from flask import render_template, request, redirect, url_for, session
import numpy as np


class FrontendController:

    def __init__(self, models_dict):
        self.rf_model     = models_dict['rf_model']
        self.gb_model     = models_dict['gb_model']
        self.y_scaler     = models_dict['y_scaler']
        self.y_le         = models_dict['y_le']
        self.y_features   = list(models_dict['y_features'])
        self.unique_vals  = models_dict['unique_vals']
        self.y_metrics    = models_dict['y_metrics']
        self.irr_clf      = models_dict['irr_clf']
        self.irr_scaler   = models_dict['irr_scaler']
        self.irr_le       = models_dict['irr_le']
        self.irr_target_le= models_dict['irr_target_le']
        self.irr_features = list(models_dict['irr_features'])
        self.models_loaded= models_dict['models_loaded']

    def safe_encode(self, le, value):
        val = str(value).strip().lower()
        if val in le.classes_:
            return int(le.transform([val])[0])
        for cls in le.classes_:
            if val in cls or cls in val:
                return int(le.transform([cls])[0])
        return 0

    def get_irrigation_advice(self, need_level):
        advice = {
            "Low":       {"frequency": "Once every 10–14 days", "amount": "25–40 mm per session",  "method": "Drip irrigation",                "notes": "Natural rainfall likely sufficient. Monitor soil moisture."},
            "Moderate":  {"frequency": "Once every 7 days",     "amount": "40–60 mm per session",  "method": "Sprinkler or furrow irrigation",  "notes": "Supplement rainfall as needed. Adjust based on soil moisture."},
            "High":      {"frequency": "Every 4–5 days",        "amount": "60–80 mm per session",  "method": "Flood or sprinkler irrigation",   "notes": "Regular irrigation required. Monitor drainage."},
            "Very High": {"frequency": "Every 2–3 days",        "amount": "80–100 mm per session", "method": "Continuous drip or flood irrigation", "notes": "High water demand. Mulching recommended to retain moisture."},
        }
        return advice.get(need_level, advice["Moderate"])

    def convert_acres_to_hectares(self, acres):
        return float(acres) * 0.404686

    def home_page(self):
        return render_template("home.html", models_loaded=self.models_loaded)

    def yield_page(self):
        if not self.models_loaded:
            return render_template("error.html", message="Models not loaded. Please train models first.")

        mode      = request.args.get("mode", "simple")
        result    = session.pop('yield_result', None)
        error     = session.pop('yield_error', None)
        form_data = session.pop('yield_form_data', {})

        if request.method == "POST":
            try:
                mode      = request.form.get("mode", "simple")
                form_data = dict(request.form)

                cat_feats = ['state', 'district', 'crop', 'season']
                row = {}
                for feat in self.y_features:
                    if feat in cat_feats:
                        val = request.form.get(feat, '').strip().lower()
                        row[feat] = self.safe_encode(self.y_le[feat], val)
                    else:
                        row[feat] = float(request.form.get(feat, 0) or 0)

                if mode == "simple" and "area_acres" in request.form:
                    row["area"] = self.convert_acres_to_hectares(float(request.form.get("area_acres", 0) or 0))

                # Build numpy array in exact feature order — avoids ALL pandas/indexing issues
                X = np.array([[row[f] for f in self.y_features]], dtype=float)
                X_scaled = self.y_scaler.transform(X)

                rf_pred = float(self.rf_model.predict(X_scaled)[0])
                gb_pred = float(self.gb_model.predict(X_scaled)[0])
                hybrid  = max(0.0, min(0.55 * rf_pred + 0.45 * gb_pred, 500.0))

                area       = row.get("area", 0)
                total_prod = round(hybrid * area, 1) if area > 0 else None

                result = {
                    "yield": round(hybrid, 2),
                    "low":   round(hybrid * 0.90, 2),
                    "high":  round(hybrid * 1.10, 2),
                    "total_prod": total_prod,
                    "crop":   request.form.get("crop", ""),
                    "season": request.form.get("season", ""),
                    "area":   area,
                    "mode":   mode
                }
                session['yield_result']    = result
                session['yield_form_data'] = form_data
                return redirect(url_for('web_yield', mode=mode))

            except Exception as e:
                import traceback
                session['yield_error']     = f"Prediction failed: {str(e)}"
                session['yield_form_data'] = dict(request.form)
                return redirect(url_for('web_yield', mode=mode))

        options = {
            "states":    sorted(self.unique_vals.get("state", [])),
            "districts": sorted(self.unique_vals.get("district", [])),
            "crops":     sorted(self.unique_vals.get("crop", [])),
            "seasons":   ["kharif", "rabi", "whole year", "summer", "winter", "autumn"]
        }
        return render_template("yield.html", mode=mode, options=options,
                               result=result, error=error, form_data=form_data)

    def irrigation_page(self):
        if not self.models_loaded:
            return render_template("error.html", message="Models not loaded. Please train models first.")

        mode      = request.args.get("mode", "simple")
        result    = session.pop('irrigation_result', None)
        error     = session.pop('irrigation_error', None)
        form_data = session.pop('irrigation_form_data', {})

        if request.method == "POST":
            try:
                mode      = request.form.get("mode", "simple")
                form_data = dict(request.form)

                # irr_le is always a dict: {'crop': le, 'state': le, 'season': le}
                cat_feats = list(self.irr_le.keys())
                row = {}
                for feat in self.irr_features:
                    if feat in cat_feats:
                        val = request.form.get(feat, '').strip().lower()
                        row[feat] = self.safe_encode(self.irr_le[feat], val)
                    else:
                        row[feat] = float(request.form.get(feat, 0) or 0)

                # Build numpy array in exact feature order
                X = np.array([[row[f] for f in self.irr_features]], dtype=float)
                X_scaled = self.irr_scaler.transform(X)

                pred_idx   = self.irr_clf.predict(X_scaled)[0]
                pred_label = self.irr_target_le.inverse_transform([pred_idx])[0]
                confidence = round(float(self.irr_clf.predict_proba(X_scaled)[0].max()) * 100, 1)
                advice     = self.get_irrigation_advice(pred_label)

                result = {
                    "need":       pred_label,
                    "confidence": confidence,
                    "advice":     advice,
                    "crop":       request.form.get("crop", ""),
                    "mode":       mode
                }
                session['irrigation_result']    = result
                session['irrigation_form_data'] = form_data
                return redirect(url_for('web_irrigation', mode=mode))

            except Exception as e:
                session['irrigation_error']     = f"Prediction failed: {str(e)}"
                session['irrigation_form_data'] = dict(request.form)
                return redirect(url_for('web_irrigation', mode=mode))

        options = {
            "states":  sorted(self.unique_vals.get("state", [])),
            "crops":   sorted(self.unique_vals.get("crop", [])),
            "seasons": ["kharif", "rabi", "whole year", "summer", "winter", "autumn"]
        }
        return render_template("irrigation.html", mode=mode, options=options,
                               result=result, error=error, form_data=form_data)

    def dashboard_page(self):
        stats = {
            "test_r2": self.y_metrics.get("r2", 0) * 100,
            "mae":     self.y_metrics.get("mae", 0),
            "records": "226K",
            "crops":   "54"
        }
        recent = [
            {"crop": "Rice",      "state": "Karnataka",    "yield": 4.21,  "irrigation": "Moderate"},
            {"crop": "Wheat",     "state": "Punjab",       "yield": 5.83,  "irrigation": "High"},
            {"crop": "Cotton",    "state": "Gujarat",      "yield": 2.97,  "irrigation": "Very High"},
            {"crop": "Maize",     "state": "Maharashtra",  "yield": 3.54,  "irrigation": "Moderate"},
            {"crop": "Sugarcane", "state": "Uttar Pradesh","yield": 72.10, "irrigation": "Very High"},
        ]
        return render_template("dashboard.html", stats=stats, recent=recent)

    def error_page(self, message):
        return render_template("error.html", message=message)


def register_frontend_routes(app, frontend_controller):

    @app.route("/")
    @app.route("/web")
    def web_home():
        return frontend_controller.home_page()

    app.add_url_rule("/home", view_func=lambda: frontend_controller.home_page(), endpoint="home")

    @app.route("/web/yield", methods=["GET", "POST"])
    def web_yield():
        return frontend_controller.yield_page()

    app.add_url_rule("/yield",         view_func=lambda: frontend_controller.yield_page(), endpoint="yield")
    app.add_url_rule("/yield_predict", view_func=lambda: frontend_controller.yield_page(), endpoint="yield_predict")

    @app.route("/web/irrigation", methods=["GET", "POST"])
    def web_irrigation():
        return frontend_controller.irrigation_page()

    app.add_url_rule("/irrigation",         view_func=lambda: frontend_controller.irrigation_page(), endpoint="irrigation")
    app.add_url_rule("/irrigation_predict", view_func=lambda: frontend_controller.irrigation_page(), endpoint="irrigation_predict")

    @app.route("/web/dashboard")
    def web_dashboard():
        return frontend_controller.dashboard_page()

    app.add_url_rule("/dashboard", view_func=lambda: frontend_controller.dashboard_page(), endpoint="dashboard")
