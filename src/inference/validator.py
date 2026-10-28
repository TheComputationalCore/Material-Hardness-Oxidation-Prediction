"""Validation for the exact material and process conditions reported in the paper."""

from __future__ import annotations

import math
from typing import Any, Dict, List

import pandas as pd

VALID_MATERIALS = {"EN-8", "Mild Steel"}
VALID_CURRENTS_A = {120.0, 140.0, 160.0}
VALID_EXPOSURE_TIMES_H = {5.0, 10.0, 15.0}


class ValidationError(Exception):
    """Raised when an input is not represented by the paper's experiment."""


def validate_material(value: Any) -> str:
    if value is None or str(value).strip() == "":
        raise ValidationError("Material is required.")
    value = str(value).strip()
    for material in VALID_MATERIALS:
        if value.lower() == material.lower():
            return material
    raise ValidationError(f"Material must be one of: {sorted(VALID_MATERIALS)}")


def validate_numeric(name: str, value: Any) -> float:
    if value is None or (isinstance(value, str) and value.strip() == ""):
        raise ValidationError(f"Missing required value for '{name}'.")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValidationError(f"Field '{name}' must be numeric.")
    if not math.isfinite(number):
        raise ValidationError(f"Field '{name}' must be finite.")
    return number


def validate_current(value: Any) -> float:
    current = validate_numeric("Current", value)
    if current not in VALID_CURRENTS_A:
        raise ValidationError("Current must be 120, 140, or 160 A (paper-tested conditions).")
    return current


def validate_hardness_input(data: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "Material": validate_material(data.get("Material")),
        "Current": validate_current(data.get("Current")),
    }


def validate_oxidation_input(data: Dict[str, Any]) -> Dict[str, Any]:
    exposure = validate_numeric("Exposure_Time_h", data.get("Exposure_Time_h"))
    if exposure not in VALID_EXPOSURE_TIMES_H:
        raise ValidationError("Exposure time must be 5, 10, or 15 hours (paper-tested conditions).")
    return {
        "Material": validate_material(data.get("Material")),
        "Current": validate_current(data.get("Current")),
        "Exposure_Time_h": exposure,
    }


def to_dataframe(data: Dict[str, Any], feature_order: List[str]) -> pd.DataFrame:
    try:
        row = [data[feature] for feature in feature_order]
    except KeyError as error:
        raise ValidationError(f"Missing required field: {error}")
    return pd.DataFrame([row], columns=feature_order)
