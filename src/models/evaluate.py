"""Report grouped out-of-fold metrics without modifying saved model artifacts."""

from __future__ import annotations

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict

from src.models.pipelines import (
    HARDNESS_FEATURES,
    OXIDATION_FEATURES,
    build_hardness_pipeline,
    build_oxidation_pipeline,
)
from src.models.utils import load_csv


def evaluate(data_path, features, target, pipeline_builder):
    df = load_csv(data_path)
    if not df["Data_Status"].eq("reported_paper_measurement").all():
        raise ValueError(f"{data_path} contains rows not marked as paper measurements.")
    X = df[features]
    y = df[target]
    predictions = cross_val_predict(
        pipeline_builder(),
        X,
        y,
        groups=df["Current"],
        cv=LeaveOneGroupOut(),
    )
    return {
        "MAE": float(mean_absolute_error(y, predictions)),
        "RMSE": float(mean_squared_error(y, predictions, squared=False)),
        "R2": float(r2_score(y, predictions)),
        "validation": "LeaveOneGroupOut grouped by Current",
        "n_samples": len(df),
        "n_groups": int(df["Current"].nunique()),
    }


if __name__ == "__main__":
    print(
        "Hardness:",
        evaluate(
            "data/hardness.csv",
            HARDNESS_FEATURES,
            "Hardness",
            build_hardness_pipeline,
        ),
    )
    print(
        "Oxidation areal mass gain:",
        evaluate(
            "data/oxidation.csv",
            OXIDATION_FEATURES,
            "Areal_Mass_Gain_g_mm2",
            build_oxidation_pipeline,
        ),
    )
