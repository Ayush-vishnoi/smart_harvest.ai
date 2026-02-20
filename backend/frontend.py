#!/usr/bin/env python3
"""
Smart Harvest AI - Frontend Logic (Pure Python)
All web page routes and form handling - completely separate from API
"""

from flask import render_template, request, redirect, url_for, session
import pandas as pd


class FrontendController:
    """Handles all frontend web routes - no API, pure server-side rendering"""
    
    def __init__(self, models_dict):
        """
        Initialize with loaded models
        
        Args:
            models_dict: Dictionary containing all loaded ML models and encoders
        """
        self.rf_model = models_dict['rf_model']
        self.gb_model = models_dict['gb_model']
        self.y_scaler = models_dict['y_scaler']
        self.y_le = models_dict['y_le']
        self.y_features = models_dict['y_features']
        self.unique_vals = models_dict['unique_vals']
        self.y_metrics = models_dict['y_metrics']
        
        self.irr_clf = models_dict['irr_clf']
        self.irr_scaler = models_dict['irr_scaler']
        self.irr_le = models_dict['irr_le']
        self.irr_target_le = models_dict['irr_target_le']
        self.irr_features = models_dict['irr_features']
        
        self.models_loaded = models_dict['models_loaded']
    
    # ═══════════════════════════════════════════════════════════════════════
    # HELPER METHODS
    # ═══════════════════════════════════════════════════════════════════════
    
    def safe_encode(self, le, value):
        """Encode categorical value with fallback for unknown values"""
        val = str(value).strip().lower()
        if val in le.classes_:
            return le.transform([val])[0]
        # Fuzzy match
        for cls in le.classes_:
            if val in cls or cls in val:
                return le.transform([cls])[0]
        return 0  # default fallback
    
    def get_irrigation_advice(self, need_level):
        """Return human-readable irrigation recommendations"""
        advice = {
            "Low": {
                "frequency": "Once every 10–14 days",
                "amount": "25–40 mm per session",
                "method": "Drip irrigation",
                "notes": "Natural rainfall likely sufficient. Monitor soil moisture."
            },
            "Moderate": {
                "frequency": "Once every 7 days",
                "amount": "40–60 mm per session",
                "method": "Sprinkler or furrow irrigation",
                "notes": "Supplement rainfall as needed. Adjust based on soil moisture."
            },
            "High": {
                "frequency": "Every 4–5 days",
                "amount": "60–80 mm per session",
                "method": "Flood or sprinkler irrigation",
                "notes": "Regular irrigation required. Monitor drainage."
            },
            "Very High": {
                "frequency": "Every 2–3 days",
                "amount": "80–100 mm per session",
                "method": "Continuous drip or flood irrigation",
                "notes": "High water demand. Mulching recommended to retain moisture."
            },
        }
        return advice.get(need_level, advice["Moderate"])
    
    def convert_acres_to_hectares(self, acres):
        """Convert acres to hectares"""
        return float(acres) * 0.404686
    
    # ═══════════════════════════════════════════════════════════════════════
    # PAGE ROUTES
    # ═══════════════════════════════════════════════════════════════════════
    
    def home_page(self):
        """Render home page"""
        return render_template("home.html", models_loaded=self.models_loaded)
    
    def yield_page(self):
        """Handle yield prediction page - both GET and POST"""
        if not self.models_loaded:
            return render_template("error.html", 
                message="Models not loaded. Please train models first.")
        
        mode = request.args.get("mode", "simple")
        
        # Check if we have result in session (from POST redirect)
        result = session.pop('yield_result', None)
        error = session.pop('yield_error', None)
        form_data = session.pop('yield_form_data', {})
        
        # Handle form submission (POST)
        if request.method == "POST":
            try:
                mode = request.form.get("mode", "simple")
                form_data = dict(request.form)
                
                # Build feature dictionary
                row = {}
                for feat in self.y_features:
                    if feat in ["state", "district", "crop", "season"]:
                        val = request.form.get(feat, "").strip().lower()
                        row[feat] = self.safe_encode(self.y_le[feat], val)
                    else:
                        row[feat] = float(request.form.get(feat, 0) or 0)
                
                if mode == "simple" and "area_acres" in request.form:
                    acres = float(request.form.get("area_acres", 0) or 0)
                    row["area"] = self.convert_acres_to_hectares(acres)
                
                # Make prediction
                X = pd.DataFrame([row])[self.y_features]
                X_scaled = self.y_scaler.transform(X)
                
                rf_pred = float(self.rf_model.predict(X_scaled)[0])
                gb_pred = float(self.gb_model.predict(X_scaled)[0])
                hybrid = 0.55 * rf_pred + 0.45 * gb_pred
                hybrid = max(0.0, min(hybrid, 500.0))
                
                low = max(0.0, hybrid * 0.90)
                high = hybrid * 1.10
                area = row.get("area", 0)
                total_prod = (hybrid * area) if area > 0 else None
                
                result = {
                    "yield": round(hybrid, 2),
                    "low": round(low, 2),
                    "high": round(high, 2),
                    "total_prod": round(total_prod, 1) if total_prod else None,
                    "crop": request.form.get("crop", ""),
                    "season": request.form.get("season", ""),
                    "area": area,
                    "mode": mode
                }
                
                # Store in session and redirect (PRG pattern)
                session['yield_result'] = result
                session['yield_form_data'] = form_data
                return redirect(url_for('web_yield', mode=mode))
                
            except Exception as e:
                error = f"Prediction failed: {str(e)}"
                session['yield_error'] = error
                session['yield_form_data'] = dict(request.form)
                return redirect(url_for('web_yield', mode=mode))
        
        options = {
            "states": sorted(self.unique_vals.get("state", [])),
            "districts": sorted(self.unique_vals.get("district", [])),
            "crops": sorted(self.unique_vals.get("crop", [])),
            "seasons": ["kharif", "rabi", "whole year", "summer", "winter", "autumn"]
        }
        
        return render_template("yield.html", 
            mode=mode, 
            options=options, 
            result=result, 
            error=error,
            form_data=form_data
        )
    
    def irrigation_page(self):
        """Handle irrigation recommendation page - both GET and POST"""
        if not self.models_loaded:
            return render_template("error.html", 
                message="Models not loaded. Please train models first.")
        
        mode = request.args.get("mode", "simple")
        
        # Check if we have result in session (from POST redirect)
        result = session.pop('irrigation_result', None)
        error = session.pop('irrigation_error', None)
        form_data = session.pop('irrigation_form_data', {})
        
        # Handle form submission (POST)
        if request.method == "POST":
            try:
                mode = request.form.get("mode", "simple")
                form_data = dict(request.form)
                
                # Build feature dictionary
                row = {}
                for feat in self.irr_features:
                    if feat in self.irr_le:
                        val = request.form.get(feat, "").strip().lower()
                        row[feat] = self.safe_encode(self.irr_le[feat], val)
                    else:
                        row[feat] = float(request.form.get(feat, 0) or 0)
                
                # Make prediction
                X = pd.DataFrame([row])[self.irr_features]
                X_scaled = self.irr_scaler.transform(X)
                
                pred_idx = self.irr_clf.predict(X_scaled)[0]
                pred_label = self.irr_target_le.inverse_transform([pred_idx])[0]
                proba = self.irr_clf.predict_proba(X_scaled)[0]
                confidence = round(float(proba.max()) * 100, 1)
                
                advice = self.get_irrigation_advice(pred_label)
                
                result = {
                    "need": pred_label,
                    "confidence": confidence,
                    "advice": advice,
                    "crop": request.form.get("crop", ""),
                    "mode": mode
                }
                
                # Store in session and redirect (PRG pattern)
                session['irrigation_result'] = result
                session['irrigation_form_data'] = form_data
                return redirect(url_for('web_irrigation', mode=mode))
                
            except Exception as e:
                error = f"Prediction failed: {str(e)}"
                session['irrigation_error'] = error
                session['irrigation_form_data'] = dict(request.form)
                return redirect(url_for('web_irrigation', mode=mode))
        
        options = {
            "states": sorted(self.unique_vals.get("state", [])),
            "crops": sorted(self.unique_vals.get("crop", [])),
            "seasons": ["kharif", "rabi", "whole year", "summer", "winter", "autumn"]
        }
        
        return render_template("irrigation.html", 
            mode=mode, 
            options=options, 
            result=result, 
            error=error,
            form_data=form_data
        )
    
    def dashboard_page(self):
        """Render dashboard with statistics"""
        stats = {
            "test_r2": self.y_metrics.get("r2", 0) * 100,
            "mae": self.y_metrics.get("mae", 0),
            "records": "226K",
            "crops": "54"
        }
        
        # Mock recent predictions (in production, would fetch from database)
        recent = [
            {"crop": "Rice", "state": "Karnataka", "yield": 4.21, "irrigation": "Moderate"},
            {"crop": "Wheat", "state": "Punjab", "yield": 5.83, "irrigation": "High"},
            {"crop": "Cotton", "state": "Gujarat", "yield": 2.97, "irrigation": "Very High"},
            {"crop": "Maize", "state": "Maharashtra", "yield": 3.54, "irrigation": "Moderate"},
            {"crop": "Sugarcane", "state": "Uttar Pradesh", "yield": 72.10, "irrigation": "Very High"},
        ]
        
        return render_template("dashboard.html", 
            stats=stats, 
            recent=recent
        )
    
    def error_page(self, message):
        """Render error page with custom message"""
        return render_template("error.html", message=message)


def register_frontend_routes(app, frontend_controller):
    """
    Register all frontend routes with the Flask app
    
    Args:
        app: Flask application instance
        frontend_controller: Instance of FrontendController
    """
    
    @app.route("/")
    @app.route("/web")
    def web_home():
        return frontend_controller.home_page()
    
    app.add_url_rule("/home", view_func=lambda: frontend_controller.home_page(), endpoint="home")
    
    @app.route("/web/yield", methods=["GET", "POST"])
    def web_yield():
        return frontend_controller.yield_page()
    
    app.add_url_rule("/yield", view_func=lambda: frontend_controller.yield_page(), endpoint="yield")
    app.add_url_rule("/yield_predict", view_func=lambda: frontend_controller.yield_page(), endpoint="yield_predict")
    
    @app.route("/web/irrigation", methods=["GET", "POST"])
    def web_irrigation():
        return frontend_controller.irrigation_page()
    
    app.add_url_rule("/irrigation", view_func=lambda: frontend_controller.irrigation_page(), endpoint="irrigation")
    app.add_url_rule("/irrigation_predict", view_func=lambda: frontend_controller.irrigation_page(), endpoint="irrigation_predict")
    
    @app.route("/web/dashboard")
    def web_dashboard():
        return frontend_controller.dashboard_page()
    
    app.add_url_rule("/dashboard", view_func=lambda: frontend_controller.dashboard_page(), endpoint="dashboard")
