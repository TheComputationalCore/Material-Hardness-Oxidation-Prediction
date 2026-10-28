"""
pipelines.py
Defines sklearn preprocessing and model pipelines for:
- Hardness prediction
- Oxidation rate prediction

Contains:
- Canonical feature lists
- ColumnTransformer preprocessing
- Fully reproducible pipelines
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# -------------------------------------------------------------------
# Canonical feature order (used across training + inference)
# -------------------------------------------------------------------

HARDNESS_FEATURES = ["Material", "Current"]

OXIDATION_FEATURES = [
    "Material",
    "Current",
    "Exposure_Time_h",
]

# -------------------------------------------------------------------
# Column groups
# -------------------------------------------------------------------

categorical_cols = ["Material"]

numeric_cols_hardness = ["Current"]
numeric_cols_oxidation = ["Current", "Exposure_Time_h"]

# -------------------------------------------------------------------
# Preprocessors
# -------------------------------------------------------------------

def make_preprocessor(numeric_features):
    """
    Create a ColumnTransformer handling:
    - OneHotEncoder for categorical data
    - StandardScaler for numeric features
    """

    return ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                    dtype=float,
                ),
                categorical_cols,
            ),
            ("num", StandardScaler(), numeric_features),
        ],
        remainder="drop",
    )


preprocessor_hardness = make_preprocessor(numeric_cols_hardness)
preprocessor_oxidation = make_preprocessor(numeric_cols_oxidation)

# -------------------------------------------------------------------
# Pipeline Builders
# -------------------------------------------------------------------

def build_hardness_pipeline() -> Pipeline:
    """
    Build the full sklearn pipeline for hardness prediction.
    Linear baseline over the paper's measured current levels.
    """
    return Pipeline(
        steps=[
            ("preprocess", preprocessor_hardness),
            ("model", LinearRegression()),
        ]
    )


def build_oxidation_pipeline() -> Pipeline:
    """
    Build the full sklearn pipeline for oxidation rate prediction.
    Uses a small random forest over the paper's measured current and time levels.
    """
    return Pipeline(
        steps=[
            ("preprocess", preprocessor_oxidation),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=300,
                    max_depth=None,
                    min_samples_leaf=1,
                    random_state=42,
                ),
            ),
        ]
    )
