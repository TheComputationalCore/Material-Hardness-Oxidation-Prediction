"""Canonical dataset schema, validation, and integrity helpers.

The training, evaluation, audit, and inference layers all consume datasets through
this module so that units, provenance labels, and row counts are checked in one place.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pandas as pd

ALLOWED_MATERIALS = {"EN-8", "Mild Steel"}
ALLOWED_CURRENTS = {120.0, 140.0, 160.0}
ALLOWED_EXPOSURES = {5.0, 10.0, 15.0}
ALLOWED_STATUSES = {"reported_paper_measurement"}
ALLOWED_DERIVATIONS = {
    "reported_table_value",
    "reported_table_value_exact_multiple_of_5h",
    "reported_table_value_near_multiple_of_5h",
}
ALLOWED_REPLICATE_STATUSES = {
    "single_reported_value_no_replicates",
}
ALLOWED_CHEMISTRY_STATUSES = {
    "midpoint_of_reported_range",
    "reported_single_value",
}
# Heat input printed in Table 1; Q = 15 V * I / v reproduces these exactly.
HEAT_INPUT_BY_CURRENT = {120.0: 0.800, 140.0: 0.864, 160.0: 0.912}

COMMON_COLUMNS = [
    "Row_ID",
    "Material",
    "Current",
    "Heat_Input",
    "Carbon",
    "Manganese",
    "Source",
    "Source_Table",
    "Data_Status",
    "Observation_Group",
    "Specimen_ID",
    "Derivation_Status",
    "Replicate_Status",
    "Chemistry_Status",
]

SCHEMAS: dict[str, dict[str, Any]] = {
    "hardness": {
        "target": "Hardness",
        "unit_column": "Hardness_Unit",
        "expected_rows": 6,
        "extra_columns": ["Hardness", "Hardness_Unit"],
        "target_bounds": (100.0, 1000.0),
        "time_column": None,
    },
    "oxidation": {
        "target": "Areal_Mass_Gain_g_mm2",
        "unit_column": "Reported_Unit",
        "expected_rows": 18,
        "extra_columns": [
            "Exposure_Time_h",
            "Areal_Mass_Gain_g_mm2",
            "Reported_Unit",
            "Unit_Status",
        ],
        "target_bounds": (0.0, None),
        "time_column": "Exposure_Time_h",
    },
}


class DataValidationError(ValueError):
    """Raised when a dataset violates the canonical schema."""


def _is_finite_number(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.notna() & (numeric.abs() != float("inf"))


def dataset_sha256(path: str | Path) -> str:
    """Return the SHA-256 digest of a dataset file."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_dataset(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    """Validate a dataset against the canonical schema.

    Collects every violation before raising so that a failing dataset produces one
    actionable report rather than a stream of partial errors.
    """
    if kind not in SCHEMAS:
        raise DataValidationError(f"Unknown dataset kind: {kind!r}")

    schema = SCHEMAS[kind]
    required = [*COMMON_COLUMNS, *schema["extra_columns"]]
    errors: list[str] = []

    missing = [column for column in required if column not in df.columns]
    if missing:
        raise DataValidationError(f"{kind}: missing required columns: {missing}")

    if len(df) != schema["expected_rows"]:
        errors.append(f"expected {schema['expected_rows']} rows for {kind}, found {len(df)}")

    if df["Row_ID"].isna().any() or df["Row_ID"].astype(str).str.strip().eq("").any():
        errors.append("Row_ID contains missing or empty values")
    elif df["Row_ID"].duplicated().any():
        duplicates = sorted(df.loc[df["Row_ID"].duplicated(), "Row_ID"].tolist())
        errors.append(f"duplicate Row_ID values: {duplicates}")

    for column in required:
        if df[column].isna().any():
            errors.append(f"column {column!r} contains missing values")

    unknown_materials = sorted(set(df["Material"]) - ALLOWED_MATERIALS)
    if unknown_materials:
        errors.append(f"unsupported materials: {unknown_materials}")

    for column in ("Current", "Heat_Input", "Carbon", "Manganese", schema["target"]):
        if not _is_finite_number(df[column]).all():
            errors.append(f"column {column!r} contains non-finite or non-numeric values")

    if _is_finite_number(df["Current"]).all():
        unknown_currents = sorted(set(pd.to_numeric(df["Current"])) - ALLOWED_CURRENTS)
        if unknown_currents:
            errors.append(f"current values outside the paper schedule: {unknown_currents}")

    if _is_finite_number(df["Current"]).all() and _is_finite_number(df["Heat_Input"]).all():
        for current, heat_input in zip(df["Current"], df["Heat_Input"], strict=False):
            expected = HEAT_INPUT_BY_CURRENT.get(float(current))
            if expected is not None and abs(float(heat_input) - expected) > 0.01:
                errors.append(
                    f"heat input {heat_input} does not match Table 1 for {current} A "
                    f"(expected {expected})"
                )

    time_column = schema["time_column"]
    if time_column is not None:
        if not _is_finite_number(df[time_column]).all():
            errors.append(f"column {time_column!r} contains non-finite values")
        else:
            unknown_times = sorted(set(pd.to_numeric(df[time_column])) - ALLOWED_EXPOSURES)
            if unknown_times:
                errors.append(f"exposure times outside the paper schedule: {unknown_times}")
        duplicate_keys = df.duplicated(subset=["Material", "Current", time_column])
        if duplicate_keys.any():
            errors.append("duplicate (Material, Current, Exposure_Time_h) rows")

    target = schema["target"]
    if _is_finite_number(df[target]).all():
        values = pd.to_numeric(df[target])
        lower, upper = schema["target_bounds"]
        if (values <= lower).any():
            errors.append(f"{target} contains values <= {lower}")
        if upper is not None and (values >= upper).any():
            errors.append(f"{target} contains values >= {upper}")

    unknown_statuses = sorted(set(df["Data_Status"]) - ALLOWED_STATUSES)
    if unknown_statuses:
        errors.append(f"unsupported Data_Status values: {unknown_statuses}")

    unknown_derivations = sorted(set(df["Derivation_Status"]) - ALLOWED_DERIVATIONS)
    if unknown_derivations:
        errors.append(f"unsupported Derivation_Status values: {unknown_derivations}")

    unknown_replicates = sorted(set(df["Replicate_Status"]) - ALLOWED_REPLICATE_STATUSES)
    if unknown_replicates:
        errors.append(f"unsupported Replicate_Status values: {unknown_replicates}")

    unknown_chemistry = sorted(set(df["Chemistry_Status"]) - ALLOWED_CHEMISTRY_STATUSES)
    if unknown_chemistry:
        errors.append(f"unsupported Chemistry_Status values: {unknown_chemistry}")

    for column in ("Source", "Source_Table", schema["unit_column"]):
        if df[column].astype(str).str.strip().eq("").any():
            errors.append(f"column {column!r} contains blank provenance values")

    expected_groups = df.apply(
        lambda row: f"{row['Material']}@{int(float(row['Current']))}", axis=1
    )
    mismatched_groups = df.loc[df["Observation_Group"] != expected_groups, "Row_ID"].tolist()
    if mismatched_groups:
        errors.append(f"Observation_Group inconsistent with Material/Current: {mismatched_groups}")

    if errors:
        raise DataValidationError(f"{kind} dataset failed validation:\n- " + "\n- ".join(errors))
    return df


def load_dataset(path: str | Path, kind: str) -> pd.DataFrame:
    """Load and validate a canonical dataset."""
    df = pd.read_csv(path)
    return validate_dataset(df, kind)


def measured_lookup(df: pd.DataFrame, kind: str) -> dict[tuple, dict[str, Any]]:
    """Index reported measurements by their exact experimental condition."""
    time_column = SCHEMAS[kind]["time_column"]
    target = SCHEMAS[kind]["target"]
    lookup: dict[tuple, dict[str, Any]] = {}
    for _, row in df.iterrows():
        key = (
            row["Material"],
            float(row["Current"]),
            float(row[time_column]) if time_column else None,
        )
        lookup[key] = {
            "value": float(row[target]),
            "row_id": row["Row_ID"],
            "source_table": row["Source_Table"],
            "data_status": row["Data_Status"],
        }
    return lookup
