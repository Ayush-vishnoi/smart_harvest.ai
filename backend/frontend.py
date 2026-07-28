#!/usr/bin/env python3
from pathlib import Path

import pandas as pd
from flask import render_template, request, redirect, url_for, session


class FrontendController:

    def __init__(self, models_dict):
        self.yield_service = models_dict['yield_service']
        self.unique_vals = models_dict['unique_vals']
        self.y_metrics = models_dict['y_metrics']
        self.models_loaded = models_dict['models_loaded']
        self._dashboard_analytics = None

    def get_irrigation_advice(self, need_level):
        return "Irrigation recommendations are not available in this release."

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

                payload = dict(request.form)
                if mode == 'simple':
                    area_hectares = self.convert_acres_to_hectares(request.form.get('area_acres', ''))
                    payload.update({
                        'area': area_hectares,
                        'crop_year': 2020,
                        'annual_rainfall': float(request.form.get('annual_rainfall') or 1247.6),
                        'fertilizer': area_hectares * 144.49,
                        'pesticide': area_hectares * 0.27,
                    })
                pred, raw_pred = self.yield_service.predict(payload)
                mae = float(self.y_metrics.get('mae', 0))

                area = float(payload["area"])
                total_prod = round(pred * area, 1) if area > 0 else None

                result = {
                    "yield": round(pred, 2),
                    "raw_prediction": round(raw_pred, 2),
                    "low":   round(max(0, pred - mae), 2),
                    "high":  round(pred + mae, 2),
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

        options = self.yield_service.options()
        return render_template("yield.html", mode=mode, options=options,
                               result=result, error=error, form_data=form_data)

    def irrigation_page(self):
        return render_template("error.html", message="Irrigation feature has been removed.")

    def _build_dashboard_analytics(self):
        """Build a JSON-safe analytics snapshot once per application worker."""
        data_path = Path(__file__).resolve().parent.parent / "crop_yield.csv"
        data = pd.read_csv(data_path)
        data.columns = data.columns.str.strip().str.lower()
        for column in ("crop", "state", "season"):
            data[column] = data[column].astype(str).str.strip()

        def ranked_counts(column, limit=8):
            counts = data[column].value_counts().head(limit)
            return [{"label": str(label), "value": int(value)} for label, value in counts.items()]

        annual = data.groupby("crop_year", as_index=False).agg(
            mean_yield=("yield", "mean"),
            rainfall=("annual_rainfall", "mean"),
        )
        crop_yield = (
            data.groupby("crop", as_index=False)
            .agg(mean_yield=("yield", "mean"), records=("yield", "size"))
            .query("records >= 100")
            .nlargest(8, "mean_yield")
        )
        test_r2 = float(self.y_metrics.get("r2", 0))
        train_r2 = float(self.y_metrics.get("train_r2", 0))
        gap = float(self.y_metrics.get("overfitting_gap", max(0, train_r2 - test_r2)))

        return {
            "summary": {
                "records": int(len(data)),
                "crops": int(data["crop"].nunique()),
                "regions": int(data["state"].nunique()),
                "seasons": int(data["season"].nunique()),
                "year_start": int(data["crop_year"].min()),
                "year_end": int(data["crop_year"].max()),
                "mean_yield": round(float(data["yield"].mean()), 2),
                "median_rainfall": round(float(data["annual_rainfall"].median()), 1),
            },
            "model": {
                "train_r2": round(train_r2 * 100, 2),
                "test_r2": round(test_r2 * 100, 2),
                "gap": round(gap * 100, 2),
                "mae": round(float(self.y_metrics.get("mae", 0)), 3),
                "rmse": round(float(self.y_metrics.get("rmse", 0)), 3),
                "train_years": self.y_metrics.get("train_years", "1997-2015"),
                "test_years": self.y_metrics.get("test_years", "2016-2020"),
            },
            "crop_distribution": ranked_counts("crop"),
            "state_distribution": ranked_counts("state"),
            "season_distribution": ranked_counts("season", 6),
            "annual_trend": [
                {
                    "year": int(row.crop_year),
                    "yield": round(float(row.mean_yield), 3),
                    "rainfall": round(float(row.rainfall), 1),
                }
                for row in annual.itertuples(index=False)
            ],
            "top_crop_yields": [
                {
                    "label": str(row.crop),
                    "value": round(float(row.mean_yield), 2),
                    "records": int(row.records),
                }
                for row in crop_yield.itertuples(index=False)
            ],
            "health": {
                "status": "Healthy" if test_r2 >= 0.8 and gap <= 0.1 else "Review advised",
                "generalization": "Good" if gap <= 0.1 else "Potential overfitting",
                "leakage_control": "Production feature removed",
                "validation": "Time-based holdout",
            },
        }

    def dashboard_page(self):
        if self._dashboard_analytics is None:
            try:
                self._dashboard_analytics = self._build_dashboard_analytics()
            except Exception:
                self._dashboard_analytics = {
                    "summary": {"records": 0, "crops": 0, "regions": 0, "seasons": 0,
                                "year_start": 0, "year_end": 0, "mean_yield": 0,
                                "median_rainfall": 0},
                    "model": {"train_r2": 0, "test_r2": 0, "gap": 0, "mae": 0,
                              "rmse": 0, "train_years": "N/A", "test_years": "N/A"},
                    "crop_distribution": [], "state_distribution": [],
                    "season_distribution": [], "annual_trend": [],
                    "top_crop_yields": [],
                    "health": {"status": "Data unavailable", "generalization": "Unknown",
                               "leakage_control": "Production feature removed",
                               "validation": "Time-based holdout"},
                }
        return render_template("dashboard.html", analytics=self._dashboard_analytics)

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

    @app.route("/web/disease")
    @app.route("/disease")
    def web_disease():
        return render_template("disease.html")
