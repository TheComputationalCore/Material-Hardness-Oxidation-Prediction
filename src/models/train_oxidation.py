"""
Train the oxidation rate prediction model.
"""

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.models.pipelines import (
    build_oxidation_pipeline,
    OXIDATION_FEATURES
)
from src.models.utils import load_csv, save_model, save_metadata


DATA_PATH = "data/oxidation.csv"
MODEL_PATH = "models/oxidation_model.joblib"
META_PATH = "models/oxidation_metadata.json"


def train_oxidation_model():
    df = load_csv(DATA_PATH)

    X = df[OXIDATION_FEATURES]
    y = df["Oxidation_Rate"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline = build_oxidation_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    metrics = {
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": mean_squared_error(y_test, y_pred, squared=False),
        "R2": r2_score(y_test, y_pred)
    }

    print("\nOxidation Model Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.6f}")

    save_model(pipeline, MODEL_PATH)
    save_metadata(META_PATH, "Oxidation Model", OXIDATION_FEATURES, metrics)


if __name__ == "__main__":
    train_oxidation_model()
