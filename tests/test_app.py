import json

from src.app.app import create_app


def _client():
    return create_app().test_client()


def test_index_route():
    res = _client().get("/")
    assert res.status_code == 200
    assert b"Model details and limitations" in res.data


def test_health_route():
    res = _client().get("/healthz")
    assert res.status_code == 200
    payload = res.get_json()
    assert payload["status"] == "ok"
    assert payload["models"]["hardness"]["available"] is True


def test_api_predict_json():
    payload = {"Material": "EN-8", "Current": 140, "Exposure_Time_h": 10}
    res = _client().post(
        "/api/v1/predict", data=json.dumps(payload), content_type="application/json"
    )
    assert res.status_code == 200
    data = res.get_json()
    assert data["ok"] is True
    assert "hardness" in data and "oxidation_areal_mass_gain_g_mm2" in data
    assert data["hardness_applicability"]["status"] == "reported_condition"
    assert data["hardness_applicability"]["measured_value"] == 365.0
    assert (
        data["hardness_interval_90"]["lower"]
        < data["hardness"]
        < data["hardness_interval_90"]["upper"]
    )
    assert data["model_versions"] == {"hardness": "0.2.0", "oxidation": "0.3.0"}
    assert any("unit" in warning for warning in data["warnings"])


def test_api_rejects_untested_conditions():
    payload = {"Material": "EN-8", "Current": 130, "Exposure_Time_h": 10}
    res = _client().post(
        "/api/v1/predict", data=json.dumps(payload), content_type="application/json"
    )
    assert res.status_code == 400
    data = res.get_json()
    assert data["ok"] is False
    assert data["hardness_error"]


def test_api_rejects_unknown_material():
    payload = {"Material": "Aluminium", "Current": 140, "Exposure_Time_h": 10}
    res = _client().post(
        "/api/v1/predict", data=json.dumps(payload), content_type="application/json"
    )
    assert res.status_code == 400
    assert "Material must be one of" in res.get_json()["hardness_error"]


def test_api_rejects_non_object_body():
    res = _client().post(
        "/api/v1/predict", data=json.dumps([1, 2]), content_type="application/json"
    )
    assert res.status_code == 400


def test_model_info_route():
    res = _client().get("/api/v1/model-info")
    assert res.status_code == 200
    info = res.get_json()
    assert info["models"]["oxidation"]["model_version"] == "0.3.0"
    assert info["models"]["hardness"]["metrics"]["MAE"] == 26.25
    assert info["calibration"]["targets"]["hardness"]["n_oof_predictions"] == 6
    assert info["calibration"]["targets"]["hardness"]["interval_90_offsets"] == {
        "lower": -17.5,
        "upper": 36.5,
    }
    assert info["audit_findings"]["sign_status"] == "unresolved"
