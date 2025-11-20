"""
Train the oxidation rate prediction model.

Responsibilities:
- Load training data
- Validate required features
- Train sklearn pipeline
- Evaluate metrics
- Save model + metadata
"""

from __future__ import annotations

from typing import Dict, Any

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from src.models.pipelines import (
    build_oxidation_pipeline,
    OXIDATION_FEATURES,
)
from src.models.utils import (
    load_csv,
    save_model,
    save_metadata,
)


DATA_PATH = "data/oxidation.csv"
MODEL_PATH = "models/oxidation_model.joblib"
META_PATH = "models/oxidation_metadata.json"


def compute_metrics(y_true, y_pred) -> Dict[str, float]:
    """Compute regression metrics for evaluation."""
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": mean_squared_error(y_true, y_pred, squared=False),
        "R2": r2_score(y_true, y_pred),
    }


def train_oxidation_model() -> Dict[str, Any]:
    """
    Train the oxidation prediction model and return metadata.

    Raises:
        FileNotFoundError: When CSV is missing.
        KeyError: If required columns are absent.
    """

    # -----------------------------
    # Load Dataset
    # -----------------------------
    df = load_csv(DATA_PATH)

    missing = [f for f in OXIDATION_FEATURES if f not in df.columns]
    if missing:
        raise KeyError(
            f"Dataset missing required columns: {missing}. "
            f"Dataset columns: {list(df.columns)}"
        )

    if "Oxidation_Rate" not in df.columns:
        raise KeyError(
            "Dataset must contain column: 'Oxidation_Rate'. "
            f"Found: {list(df.columns)}"
        )

    # -----------------------------
    # Train-test split
    # -----------------------------
    X = df[OXIDATION_FEATURES]
    y = df["Oxidation_Rate"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    # -----------------------------
    # Fit Model
    # -----------------------------
    pipeline = build_oxidation_pipeline()
    pipeline.fit(X_train, y_train)

    # -----------------------------
    # Evaluate
    # -----------------------------
    y_pred = pipeline.predict(X_test)
    metrics = compute_metrics(y_test, y_pred)

    print("\n===============================")
    print(" Oxidation Model Evaluation")
    print("===============================")
    for k, v in metrics.items():
        print(f"{k}: {v:.6f}")

    # -----------------------------
    # Save Artifacts
    # -----------------------------
    save_model(pipeline, MODEL_PATH)

    save_metadata(
        META_PATH,
        "Oxidation Model",
        OXIDATION_FEATURES,
        metrics,
    )

    print(f"\nModel saved to {MODEL_PATH}")
    print(f"Metadata saved to {META_PATH}\n")

    return {
        "model_path": MODEL_PATH,
        "meta_path": META_PATH,
        "metrics": metrics,
    }


if __name__ == "__main__":
    train_oxidation_model()
