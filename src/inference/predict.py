"""
Prediction module for hardness and oxidation models.
Provides:
- lazy-loading models
- consistent response structure
- robust error handling
- input validation via validator.py
"""

from __future__ import annotations

import os
import traceback
from typing import Any

import joblib

from src.inference.domain import prediction_context
from src.inference.validator import (
    ValidationError,
    to_dataframe,
    validate_hardness_input,
    validate_oxidation_input,
)
from src.models.pipelines import HARDNESS_FEATURES, OXIDATION_FEATURES

# ---------------------------------------------------------
# Model path resolution (robust across CI, local, Render)
# ---------------------------------------------------------

THIS_DIR = os.path.dirname(os.path.abspath(__file__))  # /src/inference
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))  # repo root
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

HARDNESS_MODEL_PATH = os.path.join(MODEL_DIR, "hardness_model.joblib")
OXIDATION_MODEL_PATH = os.path.join(MODEL_DIR, "oxidation_model.joblib")

# ---------------------------------------------------------
# Lazy-loaded models (None until first prediction call)
# ---------------------------------------------------------

_hardness_model = None
_hardness_error = None

_oxidation_model = None
_oxidation_error = None


def _load_model(path: str) -> tuple[Any | None, str | None]:
    """General safe model loader returning (model, error_message)."""
    if not os.path.exists(path):
        return None, f"Model file does not exist: {path}"

    try:
        model = joblib.load(path)
        return model, None
    except Exception as e:  # noqa: BLE001 — model deserialization errors vary by backend.
        print(f"[ERROR] Failed to load model at {path}: {e}")
        traceback.print_exc()
        return None, str(e)


def _ensure_models_loaded():
    """Load models lazily only when needed (prevents import-time failures)."""
    global _hardness_model, _oxidation_model
    global _hardness_error, _oxidation_error

    if _hardness_model is None and _hardness_error is None:
        _hardness_model, _hardness_error = _load_model(HARDNESS_MODEL_PATH)

    if _oxidation_model is None and _oxidation_error is None:
        _oxidation_model, _oxidation_error = _load_model(OXIDATION_MODEL_PATH)


# ---------------------------------------------------------
# Generic prediction helper
# ---------------------------------------------------------


def _predict(model, validator_fn, features, payload, kind) -> dict[str, Any]:
    """
    Generic prediction wrapper.
    Validates input, builds DataFrame, runs model.
    Returns a standardized response dict with model context.
    """
    try:
        validated = validator_fn(payload)
        df = to_dataframe(validated, features)
        pred = float(model.predict(df)[0])
        return {
            "ok": True,
            "prediction": pred,
            **prediction_context(kind, validated, pred),
        }

    except ValidationError as ve:
        return {"ok": False, "error": str(ve)}

    except Exception as e:  # noqa: BLE001 — convert unexpected model failures to the API error shape.
        print("[ERROR] Unexpected prediction failure:", e)
        traceback.print_exc()
        return {"ok": False, "error": "Internal error during prediction. Check logs."}


# ---------------------------------------------------------
# Public prediction APIs
# ---------------------------------------------------------


def predict_hardness(payload: dict) -> dict[str, Any]:
    _ensure_models_loaded()

    if _hardness_model is None:
        return {"ok": False, "error": f"Hardness model unavailable: {_hardness_error}"}

    return _predict(
        _hardness_model,
        validate_hardness_input,
        HARDNESS_FEATURES,
        payload,
        "hardness",
    )


def predict_oxidation(payload: dict) -> dict[str, Any]:
    _ensure_models_loaded()

    if _oxidation_model is None:
        return {"ok": False, "error": f"Oxidation model unavailable: {_oxidation_error}"}

    return _predict(
        _oxidation_model,
        validate_oxidation_input,
        OXIDATION_FEATURES,
        payload,
        "oxidation",
    )


def model_status() -> dict[str, Any]:
    """Report whether each deployable artifact could be loaded."""
    _ensure_models_loaded()
    return {
        "hardness": {"available": _hardness_model is not None, "error": _hardness_error},
        "oxidation": {"available": _oxidation_model is not None, "error": _oxidation_error},
    }


# ---------------------------------------------------------
# Manual reload (useful after retraining)
# ---------------------------------------------------------


def reload_models():
    """Call to force reloading models without restarting the server."""
    global _hardness_model, _oxidation_model
    global _hardness_error, _oxidation_error

    print("[INFO] Reloading models from disk...")

    _hardness_model, _hardness_error = _load_model(HARDNESS_MODEL_PATH)
    _oxidation_model, _oxidation_error = _load_model(OXIDATION_MODEL_PATH)

    print("[INFO] Hardness model:", "OK" if _hardness_error is None else _hardness_error)
    print("[INFO] Oxidation model:", "OK" if _oxidation_error is None else _oxidation_error)
