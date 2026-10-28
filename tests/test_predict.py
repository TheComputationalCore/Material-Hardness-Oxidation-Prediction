# tests/test_predict.py
from src.inference.predict import predict_hardness, predict_oxidation

SAMPLE_PAYLOAD = {
    "Material": "EN-8",
    "Current": 140,
    "Exposure_Time_h": 10,
}


def test_predict_hardness_returns_number():
    r = predict_hardness(SAMPLE_PAYLOAD)
    assert "error" not in r
    assert isinstance(r.get("prediction"), float)


def test_predict_oxidation_returns_number():
    r = predict_oxidation(SAMPLE_PAYLOAD)
    assert "error" not in r
    assert isinstance(r.get("prediction"), float)
