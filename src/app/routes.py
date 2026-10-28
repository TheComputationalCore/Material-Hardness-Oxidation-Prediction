"""Web and JSON routes for predictions at the paper-tested conditions."""

from flask import Blueprint, jsonify, render_template, request

from src.inference.predict import predict_hardness, predict_oxidation

app_bp = Blueprint("app_bp", __name__)


@app_bp.route("/", methods=["GET"])
def index():
    return render_template("index.html")


def _prediction_results(payload):
    hardness = predict_hardness(payload)
    oxidation = predict_oxidation(payload)
    return hardness, oxidation


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
        "hardness": hardness.get("prediction"),
        "hardness_error": hardness.get("error"),
        "oxidation_areal_mass_gain_g_mm2": oxidation.get("prediction"),
        "oxidation_error": oxidation.get("error"),
        "oxidation_target_definition": "cumulative mass gain per exposed area; g/mm^2; not time-normalized",
    }
    if hardness.get("error") or oxidation.get("error"):
        return jsonify(response), 400
    return jsonify(response), 200
