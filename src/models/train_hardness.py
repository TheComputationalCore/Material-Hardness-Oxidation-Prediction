"""Evaluate and train the hardness model using only reported paper measurements."""

from __future__ import annotations

from typing import Any

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict

from src.data.schema import load_dataset
from src.models.manifest import write_manifest
from src.models.pipelines import HARDNESS_FEATURES, build_hardness_pipeline
from src.models.utils import save_metadata, save_model

DATA_PATH = "data/hardness.csv"
MODEL_PATH = "models/hardness_model.joblib"
META_PATH = "models/hardness_metadata.json"
MANIFEST_PATH = "models/hardness_manifest.json"
TARGET = "Hardness"
MODEL_NAME = "Hardness Model"
MODEL_VERSION = "0.2.0"
TARGET_DEFINITION = "Vickers hardness number as tabulated; the paper does not report the test load."
LIMITATIONS = [
    "Six reported table values from one study; no replicate hardness measurements.",
    "Current, travel speed, and heat input vary together, so their effects are not separable.",
    "Validation leaves out an entire welding current; it is a stress test, not evidence for "
    "new currents or new substrates.",
    "Predictions are only defined at the paper-tested materials and currents.",
    "See docs/MODEL_CARD.md and docs/SOURCE_AUDIT.md for the full limitations.",
]


def compute_metrics(y_true, y_pred) -> dict[str, float]:
    """Compute regression metrics for out-of-fold predictions."""
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(mean_squared_error(y_true, y_pred, squared=False)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def train_hardness_model() -> dict[str, Any]:
    """Group-validation by welding current, then fit the deployable pipeline on all rows."""
    df = load_dataset(DATA_PATH, "hardness")

    X = df[HARDNESS_FEATURES]
    y = df[TARGET]
    groups = df["Current"]
    predictions = cross_val_predict(
        build_hardness_pipeline(),
        X,
        y,
        groups=groups,
        cv=LeaveOneGroupOut(),
    )
    metrics = compute_metrics(y, predictions)

    # Refit the deployable model on every reported measurement after grouped evaluation.
    pipeline = build_hardness_pipeline()
    pipeline.fit(X, y)
    save_model(pipeline, MODEL_PATH)
    validation = {
        "method": "LeaveOneGroupOut",
        "group": "Current",
        "groups": sorted(float(value) for value in groups.unique()),
        "n_samples": len(df),
        "n_validation_predictions": len(predictions),
        "final_fit_rows": len(df),
        "effective_independent_observations": len(df),
        "leakage_control": (
            "Every row at the held-out current is excluded from that fold's fit; "
            "no interpolated or derived rows are present."
        ),
        "target": TARGET,
        "target_unit": "Vickers hardness number; test load not reported in source paper",
    }
    save_metadata(
        META_PATH,
        MODEL_NAME,
        HARDNESS_FEATURES,
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
        dataset_kind="hardness",
        dataset_rows=len(df),
        features=HARDNESS_FEATURES,
        target=TARGET,
        target_definition=TARGET_DEFINITION,
        metrics=metrics,
        validation=validation,
        seed=None,
        limitations=LIMITATIONS,
    )
    print("Hardness grouped-validation metrics:", metrics)
    return {"model_path": MODEL_PATH, "meta_path": META_PATH, "metrics": metrics}


if __name__ == "__main__":
    train_hardness_model()
