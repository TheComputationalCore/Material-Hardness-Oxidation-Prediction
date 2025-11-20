"""
Utility functions for training pipelines.
"""

import json
from datetime import datetime
import pandas as pd
import joblib


def load_csv(path):
    return pd.read_csv(path)


def save_model(model, path):
    joblib.dump(model, path)
    print(f"Model saved to: {path}")


def save_metadata(path, model_name, features, metrics):
    metadata = {
        "model_name": model_name,
        "trained_on": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "features": features,
        "metrics": metrics
    }
    with open(path, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"Metadata saved to: {path}")
