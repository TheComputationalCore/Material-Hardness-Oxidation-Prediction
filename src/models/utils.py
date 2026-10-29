"""
Utility functions for training pipelines.
"""

import json
from datetime import datetime, timezone

import joblib
import pandas as pd


def load_csv(path):
    return pd.read_csv(path)


def save_model(model, path):
    joblib.dump(model, path)
    print(f"Model saved to: {path}")


def save_metadata(path, model_name, features, metrics, validation=None, model_version=None):
    metadata = {
        "model_name": model_name,
        "model_version": model_version,
        "trained_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),  # noqa: UP017 — Python 3.10 compatibility.
        "features": features,
        "metrics": metrics,
        "validation": validation or {},
    }
    with open(path, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"Metadata saved to: {path}")
