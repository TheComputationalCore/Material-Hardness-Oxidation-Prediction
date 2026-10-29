from src.inference.domain import (
    applicability,
    interval,
    prediction_context,
    service_info,
)


def test_reported_condition_exposes_the_paper_value():
    result = applicability("hardness", "EN-8", 140.0)
    assert result["status"] == "reported_condition"
    assert result["measured_value"] == 365.0
    assert result["measured_source_table"] == "Table 6"


def test_outside_conditions_are_flagged():
    result = applicability("oxidation", "EN-8", 200.0, 10.0)
    assert result["status"] == "outside_tested_conditions"
    assert result["measured_value"] is None


def test_interval_is_ordered_and_brackets_the_prediction():
    result = interval("oxidation", 0.0047)
    assert result["lower"] < 0.0047 < result["upper"]
    assert result["coverage"] == 0.90
    assert result["effective_independent_observations"] == 6


def test_prediction_context_includes_warnings_and_interval():
    context = prediction_context(
        "oxidation",
        {"Material": "EN-8", "Current": 140.0, "Exposure_Time_h": 10.0},
        0.0047,
    )
    assert context["model_version"]
    assert context["interval"] is not None
    assert any("unit" in warning for warning in context["warnings"])
    assert any("sign" in warning for warning in context["warnings"])


def test_service_info_lists_both_models():
    info = service_info()
    assert set(info["models"]) == {"hardness", "oxidation"}
    assert info["models"]["oxidation"]["model_version"] == "0.3.0"
    assert info["models"]["hardness"]["model_version"] == "0.2.0"
    assert info["audit_findings"]["unit_status"] == "unresolved"
