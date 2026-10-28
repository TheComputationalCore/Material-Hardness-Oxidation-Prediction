"""Evaluate and train the hardness model using only reported paper measurements."""

from __future__ import annotations

from typing import Any, Dict

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict

from src.models.pipelines import HARDNESS_FEATURES, build_hardness_pipeline
from src.models.utils import load_csv, save_metadata, save_model


DATA_PATH = "data/hardness.csv"
MODEL_PATH = "models/hardness_model.joblib"
META_PATH = "models/hardness_metadata.json"
TARGET = "Hardness"


def compute_metrics(y_true, y_pred) -> Dict[str, float]:
    """Compute regression metrics for out-of-fold predictions."""
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(mean_squared_error(y_true, y_pred, squared=False)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def train_hardness_model() -> Dict[str, Any]:
    """Group-validation by welding current, then fit the deployable pipeline on all rows."""
    df = load_csv(DATA_PATH)
    required = [*HARDNESS_FEATURES, TARGET, "Data_Status"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise KeyError(f"Dataset missing required columns: {missing}")
    if not df["Data_Status"].eq("reported_paper_measurement").all():
        raise ValueError("Training data must contain only reported paper measurements.")

    X = df[HARDNESS_FEATURES]
    y = df[TARGET]
    groups = df["Current"]
    cv = LeaveOneGroupOut()
    predictions = cross_val_predict(
        build_hardness_pipeline(),
        X,
        y,
        groups=groups,
        cv=cv,
    )
    metrics = compute_metrics(y, predictions)

    # Refit the deployable model on every reported measurement after grouped evaluation.
    pipeline = build_hardness_pipeline()
    pipeline.fit(X, y)
    save_model(pipeline, MODEL_PATH)
    save_metadata(
        META_PATH,
        "Hardness Model",
        HARDNESS_FEATURES,
        metrics,
        validation={
            "method": "LeaveOneGroupOut",
            "group": "Current",
            "groups": sorted(float(value) for value in groups.unique()),
            "n_samples": int(len(df)),
            "n_validation_predictions": int(len(predictions)),
            "final_fit_rows": int(len(df)),
            "target": TARGET,
            "target_unit": "Vickers hardness number; test load not reported in source paper",
        },
    )
    print("Hardness grouped-validation metrics:", metrics)
    return {"model_path": MODEL_PATH, "meta_path": META_PATH, "metrics": metrics}


if __name__ == "__main__":
    train_hardness_model()
