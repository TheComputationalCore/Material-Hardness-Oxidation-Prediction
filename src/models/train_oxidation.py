"""Evaluate and train the areal oxidation mass-gain model from paper measurements."""

from __future__ import annotations

from typing import Any

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict

from src.models.pipelines import OXIDATION_FEATURES, build_oxidation_pipeline
from src.models.utils import load_csv, save_metadata, save_model

DATA_PATH = "data/oxidation.csv"
MODEL_PATH = "models/oxidation_model.joblib"
META_PATH = "models/oxidation_metadata.json"
TARGET = "Areal_Mass_Gain_g_mm2"


def compute_metrics(y_true, y_pred) -> dict[str, float]:
    """Compute regression metrics for out-of-fold predictions."""
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(mean_squared_error(y_true, y_pred, squared=False)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def train_oxidation_model() -> dict[str, Any]:
    """Group-validation by current, then fit the deployable pipeline on all rows."""
    df = load_csv(DATA_PATH)
    required = [*OXIDATION_FEATURES, TARGET, "Data_Status"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise KeyError(f"Dataset missing required columns: {missing}")
    if not df["Data_Status"].eq("reported_paper_measurement").all():
        raise ValueError("Training data must contain only reported paper measurements.")

    X = df[OXIDATION_FEATURES]
    y = df[TARGET]
    groups = df["Current"]
    cv = LeaveOneGroupOut()
    predictions = cross_val_predict(
        build_oxidation_pipeline(),
        X,
        y,
        groups=groups,
        cv=cv,
    )
    metrics = compute_metrics(y, predictions)

    pipeline = build_oxidation_pipeline()
    pipeline.fit(X, y)
    save_model(pipeline, MODEL_PATH)
    save_metadata(
        META_PATH,
        "Areal Oxidation Mass Gain Model",
        OXIDATION_FEATURES,
        metrics,
        validation={
            "method": "LeaveOneGroupOut",
            "group": "Current",
            "groups": sorted(float(value) for value in groups.unique()),
            "n_samples": len(df),
            "n_validation_predictions": len(predictions),
            "final_fit_rows": len(df),
            "target": TARGET,
            "target_unit": "g/mm^2; cumulative mass gain per exposed area, not time-normalized",
        },
    )
    print("Oxidation grouped-validation metrics:", metrics)
    return {"model_path": MODEL_PATH, "meta_path": META_PATH, "metrics": metrics}


if __name__ == "__main__":
    train_oxidation_model()
