"""Calibrate prediction intervals from grouped out-of-fold residuals.

With six independent conditions, interval estimates are deliberately labelled as
empirical and approximate. They come from leave-one-current-out residuals of the
shipped pipelines, not from a replicate-rich calibration experiment.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict

from src.data.schema import load_dataset
from src.models.pipelines import (
    HARDNESS_FEATURES,
    OXIDATION_FEATURES,
    build_hardness_pipeline,
    build_oxidation_pipeline,
)

CALIBRATION_PATH = "models/calibration.json"

TARGETS = {
    "hardness": {
        "data_path": "data/hardness.csv",
        "features": HARDNESS_FEATURES,
        "target": "Hardness",
        "builder": build_hardness_pipeline,
        "effective_n": 6,
    },
    "oxidation": {
        "data_path": "data/oxidation.csv",
        "features": OXIDATION_FEATURES,
        "target": "Areal_Mass_Gain_g_mm2",
        "builder": build_oxidation_pipeline,
        "effective_n": 6,
    },
}


def _calibrate_target(kind: str, spec: dict[str, Any]) -> dict[str, Any]:
    df = load_dataset(spec["data_path"], kind)
    predictions = cross_val_predict(
        spec["builder"](),
        df[spec["features"]],
        df[spec["target"]],
        groups=df["Current"],
        cv=LeaveOneGroupOut(),
    )
    observed = df[spec["target"]].to_numpy(dtype=float)
    residuals = np.asarray(predictions, dtype=float) - observed
    absolute = np.abs(residuals)
    lower_offset = float(np.quantile(residuals, 0.05, method="lower"))
    upper_offset = float(np.quantile(residuals, 0.95, method="higher"))
    return {
        "target": spec["target"],
        "n_oof_predictions": int(len(residuals)),
        "effective_independent_observations": spec["effective_n"],
        "protocol": "LeaveOneGroupOut grouped by Current",
        "residual_bias": float(np.mean(residuals)),
        "residual_std": float(np.std(residuals, ddof=1)),
        "MAE": float(np.mean(absolute)),
        "q05_residual": lower_offset,
        "q95_residual": upper_offset,
        "q50_abs_residual": float(np.quantile(absolute, 0.50, method="higher")),
        "q90_abs_residual": float(np.quantile(absolute, 0.90, method="higher")),
        "q95_abs_residual": float(np.quantile(absolute, 0.95, method="higher")),
        "max_abs_residual": float(np.max(absolute)),
        "interval_90_offsets": {"lower": lower_offset, "upper": upper_offset},
        "interval_90_half_width": float(np.quantile(absolute, 0.90, method="higher")),
        "interval_95_half_width": float(np.quantile(absolute, 0.95, method="higher")),
    }


def build_calibration(write: bool = True) -> dict[str, Any]:
    record = {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "method": (
            "Empirical out-of-fold signed-residual quantiles. With six independent "
            "conditions these are approximate intervals, not guaranteed coverage."
        ),
        "targets": {kind: _calibrate_target(kind, spec) for kind, spec in TARGETS.items()},
    }
    if write:
        Path(CALIBRATION_PATH).parent.mkdir(parents=True, exist_ok=True)
        with open(CALIBRATION_PATH, "w") as handle:
            json.dump(record, handle, indent=4)
            handle.write("\n")
        print(f"Calibration saved to: {CALIBRATION_PATH}")
    return record


if __name__ == "__main__":
    build_calibration()
