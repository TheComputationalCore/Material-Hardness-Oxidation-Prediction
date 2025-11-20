"""
Input validation for inference.
Ensures safe, typed, and ordered data for model prediction.
"""

from __future__ import annotations

import pandas as pd
from typing import Any, Dict, List

# Supported materials (categorical)
VALID_MATERIALS = {"EN-8", "Mild Steel"}

# Feature order (imported by pipelines, duplicated for safety)
HARDNESS_FEATURES = ["Material", "Current", "Heat_Input", "Carbon", "Manganese"]
OXIDATION_FEATURES = ["Material", "Current", "Heat_Input", "Soaking_Time", "Carbon", "Manganese"]


class ValidationError(Exception):
    """Custom exception for input validation."""
    pass


# ------------------------------------------------------------
# Helper validation functions
# ------------------------------------------------------------

def validate_material(value: Any) -> str:
    """
    Material must be one of the allowed categories.
    Case-insensitive, returns normalized value.
    """
    if value is None or str(value).strip() == "":
        raise ValidationError("Material is required.")

    value = str(value).strip()

    # Case-insensitive match
    for mat in VALID_MATERIALS:
        if value.lower() == mat.lower():
            return mat  # Return canonical form used in model

    raise ValidationError(
        f"Invalid material '{value}'. "
        f"Must be one of: {sorted(VALID_MATERIALS)}"
    )


def validate_numeric(name: str, value: Any) -> float:
    """
    Validate numeric fields.
    Accepts values like:
        10, "10", "10.5", 0, "0"
    Rejects:
        "", None, "abc"
    """
    if value is None or (isinstance(value, str) and value.strip() == ""):
        raise ValidationError(f"Missing required value for '{name}'.")

    try:
        return float(value)
    except Exception:
        raise ValidationError(f"Field '{name}' must be numeric. Got '{value}'.")


# ------------------------------------------------------------
# Prediction-specific validation
# ------------------------------------------------------------

def validate_hardness_input(data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate inputs for hardness model."""
    return {
        "Material": validate_material(data.get("Material")),
        "Current": validate_numeric("Current", data.get("Current")),
        "Heat_Input": validate_numeric("Heat_Input", data.get("Heat_Input")),
        "Carbon": validate_numeric("Carbon", data.get("Carbon")),
        "Manganese": validate_numeric("Manganese", data.get("Manganese")),
    }


def validate_oxidation_input(data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate inputs for oxidation model."""
    return {
        "Material": validate_material(data.get("Material")),
        "Current": validate_numeric("Current", data.get("Current")),
        "Heat_Input": validate_numeric("Heat_Input", data.get("Heat_Input")),
        "Soaking_Time": validate_numeric("Soaking_Time", data.get("Soaking_Time")),
        "Carbon": validate_numeric("Carbon", data.get("Carbon")),
        "Manganese": validate_numeric("Manganese", data.get("Manganese")),
    }


# ------------------------------------------------------------
# Conversion utility for inference
# ------------------------------------------------------------

def to_dataframe(data: Dict[str, Any], feature_order: List[str]) -> pd.DataFrame:
    """
    Convert validated dict → single-row DataFrame with strict column ordering.
    """
    try:
        row = [data[feature] for feature in feature_order]
    except KeyError as e:
        raise ValidationError(f"Missing required field: {e}")

    return pd.DataFrame([row], columns=feature_order)
