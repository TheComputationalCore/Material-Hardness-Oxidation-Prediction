"""Applicability context and empirical intervals attached to every prediction.

Predictions are only offered at the paper-tested conditions. This module makes
that explicit in the response: which reported value the condition corresponds to,
whether the input is inside the tested grid, and an approximate interval derived
from grouped out-of-fold residuals.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from src.data.schema import SCHEMAS, load_dataset, measured_lookup
from src.models.manifest import load_manifest

DATA_PATHS = {
    "hardness": "data/hardness.csv",
    "oxidation": "data/oxidation.csv",
}
MANIFEST_PATHS = {
    "hardness": "models/hardness_manifest.json",
    "oxidation": "models/oxidation_manifest.json",
}
CALIBRATION_PATH = "models/calibration.json"

COMMON_WARNING = (
    "Supported only at the paper-tested conditions; no replicate specimens or "
    "repeated measurements are published."
)
WARNINGS = {
    "hardness": [
        "The Vickers hardness test load is not reported, so the hardness scale is "
        "identified only by the printed table values.",
    ],
    "oxidation": [
        "The printed g/mm^2 unit implies a specimen mass change that is not physically "
        "plausible; the intended scale is unresolved and values are shown as printed.",
        "The printed sign convention implies mass loss while every tabulated value is "
        "positive; no sign correction is applied.",
        "Exposure-time rows are exact multiples of the 5 h value, so they are not "
        "independent kinetic measurements.",
    ],
}


@lru_cache(maxsize=1)
def _datasets() -> dict[str, Any]:
    return {kind: load_dataset(path, kind) for kind, path in DATA_PATHS.items()}


@lru_cache(maxsize=1)
def _lookups() -> dict[str, dict[tuple, dict[str, Any]]]:
    return {kind: measured_lookup(frame, kind) for kind, frame in _datasets().items()}


@lru_cache(maxsize=1)
def _calibration() -> dict[str, Any]:
    path = Path(CALIBRATION_PATH)
    if not path.exists():
        return {}
    with open(path) as handle:
        return json.load(handle)


@lru_cache(maxsize=1)
def _manifests() -> dict[str, dict[str, Any]]:
    return {
        kind: load_manifest(path) if Path(path).exists() else {}
        for kind, path in MANIFEST_PATHS.items()
    }


def _tested_values(kind: str) -> dict[str, Any]:
    frame = _datasets()[kind]
    time_column = SCHEMAS[kind]["time_column"]
    values = {
        "materials": sorted(frame["Material"].unique().tolist()),
        "currents_a": sorted(float(value) for value in frame["Current"].unique()),
        "exposures_h": (
            sorted(float(value) for value in frame[time_column].unique()) if time_column else None
        ),
    }
    return values


def applicability(
    kind: str,
    material: str,
    current: float,
    exposure_h: float | None = None,
) -> dict[str, Any]:
    """Describe how a requested condition relates to the reported experiment."""
    key = (material, float(current), float(exposure_h) if exposure_h is not None else None)
    measured = _lookups()[kind].get(key)
    tested = _tested_values(kind)

    if measured is not None:
        status = "reported_condition"
        explanation = (
            "This exact condition is reported in the source paper; the model estimate "
            "is compared with the reported value."
        )
    else:
        current_outside = not (
            tested["currents_a"][0] <= float(current) <= tested["currents_a"][-1]
        )
        exposure_outside = (
            tested["exposures_h"] is not None
            and exposure_h is not None
            and not (tested["exposures_h"][0] <= float(exposure_h) <= tested["exposures_h"][-1])
        )
        if current_outside or exposure_outside:
            status = "outside_tested_conditions"
            explanation = (
                "The requested condition lies outside the paper's tested ranges; the "
                "prediction is an extrapolation and is not supported by this project."
            )
        else:
            status = "unmeasured_combination_within_tested_levels"
            explanation = (
                "The individual factor levels were tested, but this combination was not "
                "reported; the prediction interpolates within the tested grid."
            )

    return {
        "status": status,
        "explanation": explanation,
        "tested_materials": tested["materials"],
        "tested_currents_a": tested["currents_a"],
        "tested_exposures_h": tested["exposures_h"],
        "measured_value": measured["value"] if measured else None,
        "measured_source_table": measured["source_table"] if measured else None,
        "measured_row_id": measured["row_id"] if measured else None,
    }


def interval(kind: str, prediction: float) -> dict[str, Any] | None:
    """Approximate 90% interval from grouped out-of-fold residual quantiles."""
    calibration = _calibration()
    target = calibration.get("targets", {}).get(kind)
    if not target or "interval_90_offsets" not in target:
        return None
    offsets = target["interval_90_offsets"]
    return {
        "coverage": 0.90,
        "lower": float(prediction + offsets["lower"]),
        "upper": float(prediction + offsets["upper"]),
        "method": "empirical_out_of_fold_residual_quantiles",
        "n_oof_predictions": target["n_oof_predictions"],
        "effective_independent_observations": target["effective_independent_observations"],
        "note": (
            "Approximate interval from leave-one-current-out residuals; not a "
            "guaranteed coverage interval."
        ),
    }


def prediction_context(
    kind: str,
    payload: dict[str, Any],
    prediction: float,
) -> dict[str, Any]:
    """Bundle model version, applicability, interval, and warnings for a response."""
    manifest = _manifests().get(kind, {})
    exposure = payload.get("Exposure_Time_h")
    return {
        "model_version": manifest.get("model_version"),
        "applicability": applicability(
            kind,
            str(payload["Material"]),
            float(payload["Current"]),
            float(exposure) if exposure is not None else None,
        ),
        "interval": interval(kind, prediction),
        "warnings": [COMMON_WARNING, *WARNINGS.get(kind, [])],
    }


def service_info() -> dict[str, Any]:
    """Model card data exposed by the API."""
    manifests = _manifests()
    calibration = _calibration()
    return {
        "service": "Stellite-6 hardfacing property predictor",
        "models": {
            kind: {
                "model_name": manifest.get("model_name"),
                "model_version": manifest.get("model_version"),
                "features": manifest.get("features"),
                "target": manifest.get("target"),
                "metrics": manifest.get("metrics"),
                "validation": manifest.get("validation"),
                "dataset": manifest.get("dataset"),
                "artifact_sha256": manifest.get("artifact", {}).get("sha256"),
                "known_limitations": manifest.get("known_limitations"),
                "warnings": [COMMON_WARNING, *WARNINGS.get(kind, [])],
            }
            for kind, manifest in manifests.items()
        },
        "calibration": calibration,
        "source": {
            "authors": "C. Dinesh Chandra, B. Rushikesh, Mohammed Numan, B. Venkatesh",
            "title": (
                "Experimental Studies of Stellite-6 Hardfaced Layer on Ferrous "
                "Materials by TIG Surfacing Process"
            ),
            "journal": "IOP Conference Series: Materials Science and Engineering 998 (2020) 012061",
            "doi": "10.1088/1757-899X/998/1/012061",
        },
        "audit_findings": {
            "unit_status": "unresolved",
            "sign_status": "unresolved",
            "effective_independent_observations": {
                "hardness": 6,
                "oxidation": 6,
            },
        },
    }
