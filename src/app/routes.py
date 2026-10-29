"""Web and JSON routes for predictions at the paper-tested conditions."""

from __future__ import annotations

from flask import Blueprint, jsonify, render_template, request

from src.inference.domain import service_info
from src.inference.predict import (
    model_status,
    predict_hardness,
    predict_oxidation,
)

app_bp = Blueprint("app_bp", __name__)

OXIDATION_DEFINITION = (
    "Cumulative mass gain per exposed area; g/mm^2 as printed; not time-normalized."
)


@app_bp.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app_bp.route("/healthz", methods=["GET"])
def healthz():
    status = model_status()
    healthy = all(entry["available"] for entry in status.values())
    payload = {"status": "ok" if healthy else "degraded", "models": status}
    return jsonify(payload), (200 if healthy else 503)


@app_bp.route("/api/v1/model-info", methods=["GET"])
def api_model_info():
    return jsonify(service_info()), 200


def _prediction_results(payload):
    hardness = predict_hardness(payload)
    oxidation = predict_oxidation(payload)
    return hardness, oxidation


def _warnings(*results):
    seen: list[str] = []
    for result in results:
        for warning in result.get("warnings", []):
            if warning not in seen:
                seen.append(warning)
    return seen


def _error_status(*results):
    for result in results:
        error = result.get("error", "")
        if "unavailable" in error or "does not exist" in error:
            return 503
    return 400


@app_bp.route("/predict", methods=["POST"])
def predict():
    payload = {
        "Material": request.form.get("Material"),
        "Current": request.form.get("Current"),
        "Exposure_Time_h": request.form.get("Exposure_Time_h"),
    }
    hardness, oxidation = _prediction_results(payload)
    return render_template(
        "index.html",
        hardness=hardness.get("prediction"),
        oxidation=oxidation.get("prediction"),
        hardness_error=hardness.get("error"),
        oxidation_error=oxidation.get("error"),
        hardness_meta={
            "applicability": hardness.get("applicability"),
            "interval": hardness.get("interval"),
            "model_version": hardness.get("model_version"),
        },
        oxidation_meta={
            "applicability": oxidation.get("applicability"),
            "interval": oxidation.get("interval"),
            "model_version": oxidation.get("model_version"),
        },
        warnings=_warnings(hardness, oxidation),
        selected_material=payload["Material"],
        selected_current=payload["Current"],
        selected_exposure=payload["Exposure_Time_h"],
    )


@app_bp.route("/api/v1/predict", methods=["POST"])
def api_predict():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Request body must be a JSON object."}), 400

    hardness, oxidation = _prediction_results(payload)
    response = {
        "ok": not (hardness.get("error") or oxidation.get("error")),
        "hardness": hardness.get("prediction"),
        "hardness_error": hardness.get("error"),
        "hardness_interval_90": hardness.get("interval"),
        "hardness_applicability": hardness.get("applicability"),
        "oxidation_areal_mass_gain_g_mm2": oxidation.get("prediction"),
        "oxidation_error": oxidation.get("error"),
        "oxidation_interval_90": oxidation.get("interval"),
        "oxidation_applicability": oxidation.get("applicability"),
        "oxidation_target_definition": OXIDATION_DEFINITION,
        "model_versions": {
            "hardness": hardness.get("model_version"),
            "oxidation": oxidation.get("model_version"),
        },
        "warnings": _warnings(hardness, oxidation),
    }
    if hardness.get("error") or oxidation.get("error"):
        return jsonify(response), _error_status(hardness, oxidation)
    return jsonify(response), 200
