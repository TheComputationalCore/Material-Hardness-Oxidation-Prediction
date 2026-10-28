"""
Utility functions for loading data, saving models, metadata,
and safe numeric conversion used throughout the project.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import joblib
import pandas as pd

# -------------------------------------------------------
# Numeric conversion
# -------------------------------------------------------

def to_float(value: Any) -> float | None:
    """
    Safely convert a value into float.
    Returns None when:
    - value is empty
    - value cannot be parsed
    - value is None

    This is used by both the Flask UI and the API layer.
    """
    if value is None:
        return None
    if isinstance(value, (float, int)):
        return float(value)
    if isinstance(value, str) and value.strip() == "":
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# -------------------------------------------------------
# CSV Utilities
# -------------------------------------------------------

def load_csv(path: str) -> pd.DataFrame:
    """
    Load CSV files for training models.
    Raises an informative error when the file does not exist.
    """
    try:
        return pd.read_csv(path)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"CSV file not found: {path}. Ensure it exists in the repository."
        ) from e


# -------------------------------------------------------
# Model saving & loading
# -------------------------------------------------------

def save_model(model: Any, path: str) -> None:
    """
    Save a scikit-learn model or pipeline using joblib.
    """
    try:
        joblib.dump(model, path)
        print(f"[INFO] Model saved to: {path}")
    except Exception as e:
        raise RuntimeError(f"Failed to save model to {path}: {e}") from e


def load_model(path: str) -> Any:
    """
    Load a trained ML model from disk.
    Provides a clear error message for inference layer.
    """
    try:
        return joblib.load(path)
    except FileNotFoundError:
        return None
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {path}: {e}") from e


# -------------------------------------------------------
# Metadata handling
# -------------------------------------------------------

def save_metadata(
    path: str,
    model_name: str,
    features: list,
    metrics: dict[str, float],
) -> None:
    """
    Save metadata about the trained model:
    - model name
    - training date
    - training features
    - evaluation metrics

    Stored as a JSON file for reproducibility.
    """

    metadata = {
        "model_name": model_name,
        "trained_at": datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "features": features,
        "metrics": metrics,
    }

    try:
        with open(path, "w") as f:
            json.dump(metadata, f, indent=4)
        print(f"[INFO] Metadata saved to: {path}")
    except Exception as e:
        raise RuntimeError(f"Failed to save metadata to {path}: {e}") from e


def load_metadata(path: str) -> dict[str, Any]:
    """
    Load metadata associated with a trained model.
    """
    try:
        with open(path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except Exception as e:
        raise RuntimeError(f"Failed to load metadata from {path}: {e}") from e
