"""Evaluate and train the areal oxidation mass-gain model from paper measurements."""

from __future__ import annotations

from typing import Any

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict

from src.data.schema import load_dataset
from src.models.manifest import write_manifest
from src.models.pipelines import OXIDATION_FEATURES, build_oxidation_pipeline
from src.models.utils import save_metadata, save_model

DATA_PATH = "data/oxidation.csv"
MODEL_PATH = "models/oxidation_model.joblib"
META_PATH = "models/oxidation_metadata.json"
MANIFEST_PATH = "models/oxidation_manifest.json"
TARGET = "Areal_Mass_Gain_g_mm2"
MODEL_NAME = "Areal Oxidation Mass Gain Model"
MODEL_VERSION = "0.3.0"
TARGET_DEFINITION = (
    "Cumulative areal mass gain in g/mm^2 as printed by the source paper; "
    "not time-normalized. The printed unit and sign convention are unresolved "
    "(see docs/SOURCE_AUDIT.md)."
)
LIMITATIONS = [
    "The production model is a linear rate model adopted in v0.3.0: it predicts "
    "mass gain per hour from substrate and current, then scales by exposure time. "
    "The benchmark recorded a 69% reduction in leave-one-current-out MAE versus the "
    "previous random forest.",
    "Eighteen reported rows, but eleven of twelve non-base rows equal an exact multiple "
    "of the 5 h value, so the effective number of independent measurements is six.",
    "The sampled exposure times do not add independent information about oxidation "
    "kinetics beyond the near-linear time trend assumed by the paper's table.",
    "The printed g/mm^2 unit implies an implausible specimen mass change; the paper's "
    "intended scale is unresolved.",
    "Current, travel speed, and heat input vary together and are not separable.",
    "Validation leaves out an entire welding current; the linear rate model extrapolates "
    "in current, which is a structural assumption, not an observed result.",
    "See docs/MODEL_CARD.md and docs/SOURCE_AUDIT.md for the full limitations.",
]


def compute_metrics(y_true, y_pred) -> dict[str, float]:
    """Compute regression metrics for out-of-fold predictions."""
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(mean_squared_error(y_true, y_pred, squared=False)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def train_oxidation_model() -> dict[str, Any]:
    """Group-validation by current, then fit the deployable pipeline on all rows."""
    df = load_dataset(DATA_PATH, "oxidation")

    X = df[OXIDATION_FEATURES]
    y = df[TARGET]
    groups = df["Current"]
    predictions = cross_val_predict(
        build_oxidation_pipeline(),
        X,
        y,
        groups=groups,
        cv=LeaveOneGroupOut(),
    )
    metrics = compute_metrics(y, predictions)

    pipeline = build_oxidation_pipeline()
    pipeline.fit(X, y)
    save_model(pipeline, MODEL_PATH)
    validation = {
        "method": "LeaveOneGroupOut",
        "group": "Current",
        "groups": sorted(float(value) for value in groups.unique()),
        "n_samples": len(df),
        "n_validation_predictions": len(predictions),
        "final_fit_rows": len(df),
        "effective_independent_observations": 6,
        "observation_independence_note": (
            "Exposure-time rows within a Material/Current cell are exact multiples of the "
            "5 h value, so they are not independent observations."
        ),
        "leakage_control": (
            "Every row at the held-out current is excluded from that fold's fit; "
            "no interpolated or derived rows are present."
        ),
        "target": TARGET,
        "target_unit": "g/mm^2; cumulative mass gain per exposed area, not time-normalized",
    }
    save_metadata(
        META_PATH,
        MODEL_NAME,
        OXIDATION_FEATURES,
        metrics,
        validation=validation,
        model_version=MODEL_VERSION,
    )
    write_manifest(
        MANIFEST_PATH,
        model_name=MODEL_NAME,
        model_version=MODEL_VERSION,
        model_path=MODEL_PATH,
        estimator=pipeline,
        dataset_path=DATA_PATH,
        dataset_kind="oxidation",
        dataset_rows=len(df),
        features=OXIDATION_FEATURES,
        target=TARGET,
        target_definition=TARGET_DEFINITION,
        metrics=metrics,
        validation=validation,
        seed=None,
        limitations=LIMITATIONS,
    )
    print("Oxidation grouped-validation metrics:", metrics)
    return {"model_path": MODEL_PATH, "meta_path": META_PATH, "metrics": metrics}


if __name__ == "__main__":
    train_oxidation_model()
