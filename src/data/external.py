"""Validation for new measurements collected outside the published paper.

External files are deliberately separate from ``data/hardness.csv`` and
``data/oxidation.csv``. They are never used for training unless a deliberate
schema and baseline change is made. This module computes physical areal gain from
raw weights and leaves the paper's printed-unit value as an optional comparison
column.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd

ALLOWED_MATERIALS = {"EN-8", "Mild Steel"}
ALLOWED_STATUSES = {"new_measurement", "external_validation"}

HARDNESS_COLUMNS = [
    "Row_ID",
    "Material",
    "Current",
    "Voltage_V",
    "Travel_Speed_mm_s",
    "Heat_Input",
    "Specimen_ID",
    "Heat_Lot",
    "Replicate_ID",
    "Deposition_Run",
    "Operator",
    "Measurement_Date",
    "Test_Load_kgf",
    "Indent_Count",
    "Surface_Prep",
    "Hardness",
    "Source",
    "Data_Status",
]

OXIDATION_COLUMNS = [
    "Row_ID",
    "Material",
    "Current",
    "Voltage_V",
    "Travel_Speed_mm_s",
    "Heat_Input",
    "Exposure_Time_h",
    "Specimen_ID",
    "Heat_Lot",
    "Replicate_ID",
    "Deposition_Run",
    "Operator",
    "Measurement_Date",
    "Exposure_Temperature_K",
    "Furnace_Atmosphere",
    "Heating_Rate_K_min",
    "Cooling_Mode",
    "Balance_ID",
    "Balance_Resolution_mg",
    "Exposed_Area_mm2",
    "w1_g",
    "w2_g",
    "Paper_Unit_Value",
    "Source",
    "Data_Status",
]

SCHEMAS: dict[str, dict[str, Any]] = {
    "hardness": {
        "columns": HARDNESS_COLUMNS,
        "optional": [],
        "numeric": [
            "Current",
            "Voltage_V",
            "Travel_Speed_mm_s",
            "Heat_Input",
            "Test_Load_kgf",
            "Indent_Count",
            "Hardness",
        ],
    },
    "oxidation": {
        "columns": OXIDATION_COLUMNS,
        "optional": ["Paper_Unit_Value"],
        "numeric": [
            "Current",
            "Voltage_V",
            "Travel_Speed_mm_s",
            "Heat_Input",
            "Exposure_Time_h",
            "Exposure_Temperature_K",
            "Heating_Rate_K_min",
            "Balance_Resolution_mg",
            "Exposed_Area_mm2",
            "w1_g",
            "w2_g",
            "Paper_Unit_Value",
        ],
    },
}


class ExternalDataError(ValueError):
    """Raised when an external measurement file violates the schema."""


def validate_external(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    """Validate external rows and derive mass change and areal gain for oxidation.

    An empty file with the correct header is valid and represents a not-yet-filled
    template.
    """
    if kind not in SCHEMAS:
        raise ExternalDataError(f"Unknown external dataset kind: {kind!r}")

    schema = SCHEMAS[kind]
    missing = [
        column
        for column in schema["columns"]
        if column not in df.columns and column not in schema["optional"]
    ]
    if missing:
        raise ExternalDataError(f"{kind}: missing required columns: {missing}")

    result = df.copy()
    for column in schema["optional"]:
        if column not in result.columns:
            result[column] = pd.NA
    if kind == "oxidation":
        result["Mass_Change_g"] = pd.NA
        result["Areal_Gain_mg_cm2"] = pd.NA
        result["Mass_Change_Sign"] = pd.NA

    if len(result) == 0:
        return result

    errors: list[str] = []

    if result["Row_ID"].isna().any() or result["Row_ID"].astype(str).str.strip().eq("").any():
        errors.append("Row_ID contains missing or empty values")
    elif result["Row_ID"].duplicated().any():
        duplicates = sorted(result.loc[result["Row_ID"].duplicated(), "Row_ID"].tolist())
        errors.append(f"duplicate Row_ID values: {duplicates}")

    required_text = [column for column in schema["columns"] if column not in schema["optional"]]
    for column in required_text:
        if result[column].isna().any() or result[column].astype(str).str.strip().eq("").any():
            errors.append(f"column {column!r} contains missing or blank values")

    unknown_materials = sorted(set(result["Material"]) - ALLOWED_MATERIALS)
    if unknown_materials:
        errors.append(f"unsupported materials: {unknown_materials}")

    for column in schema["numeric"]:
        values = pd.to_numeric(result[column], errors="coerce")
        invalid = result[column].notna() & (values.isna() | ~values.map(math.isfinite))
        if invalid.any():
            errors.append(f"column {column!r} contains non-numeric or non-finite values")
        result[column] = values

    if (result["Current"] <= 0).any():
        errors.append("Current must be positive")
    if (result["Voltage_V"] <= 0).any():
        errors.append("Voltage_V must be positive")
    if (result["Travel_Speed_mm_s"] <= 0).any():
        errors.append("Travel_Speed_mm_s must be positive")

    unknown_statuses = sorted(set(result["Data_Status"]) - ALLOWED_STATUSES)
    if unknown_statuses:
        errors.append(
            "Data_Status must be one of "
            f"{sorted(ALLOWED_STATUSES)}; got {unknown_statuses}. "
            "Paper rows belong in data/hardness.csv or data/oxidation.csv."
        )

    if kind == "hardness":
        if (result["Hardness"] <= 0).any():
            errors.append("Hardness must be positive")
        if (result["Test_Load_kgf"] <= 0).any():
            errors.append("Test_Load_kgf must be positive")
        if (result["Indent_Count"] < 1).any():
            errors.append("Indent_Count must be at least 1")
        duplicate_keys = result.duplicated(
            subset=["Material", "Current", "Specimen_ID", "Replicate_ID"]
        )
        if duplicate_keys.any():
            errors.append("duplicate (Material, Current, Specimen_ID, Replicate_ID) rows")

    if kind == "oxidation":
        if (result["Exposure_Time_h"] <= 0).any():
            errors.append("Exposure_Time_h must be positive")
        if (result["Exposure_Temperature_K"] <= 0).any():
            errors.append("Exposure_Temperature_K must be positive")
        if (result["Heating_Rate_K_min"] <= 0).any():
            errors.append("Heating_Rate_K_min must be positive")
        if (result["Balance_Resolution_mg"] <= 0).any():
            errors.append("Balance_Resolution_mg must be positive")
        if (result["Exposed_Area_mm2"] <= 0).any():
            errors.append("Exposed_Area_mm2 must be positive")
        if (result["w1_g"] <= 0).any() or (result["w2_g"] <= 0).any():
            errors.append("w1_g and w2_g must be positive")
        duplicate_keys = result.duplicated(
            subset=["Material", "Current", "Exposure_Time_h", "Specimen_ID", "Replicate_ID"]
        )
        if duplicate_keys.any():
            errors.append(
                "duplicate (Material, Current, Exposure_Time_h, Specimen_ID, Replicate_ID) rows"
            )

        mass_change = result["w2_g"] - result["w1_g"]
        areal_gain = mass_change * 100_000.0 / result["Exposed_Area_mm2"]
        result["Mass_Change_g"] = mass_change
        result["Areal_Gain_mg_cm2"] = areal_gain
        result["Mass_Change_Sign"] = mass_change.map(
            lambda value: "gain" if value > 0 else ("loss" if value < 0 else "zero")
        )

    if errors:
        raise ExternalDataError(
            f"{kind} external dataset failed validation:\n- " + "\n- ".join(errors)
        )
    return result


def load_external(path: str | Path, kind: str) -> pd.DataFrame:
    """Load and validate an external measurement file."""
    return validate_external(pd.read_csv(path), kind)
