"""Transparent baseline estimators used by the benchmark and, where adopted, production.

These estimators are deliberately small. They implement the hypotheses stated in
the benchmark: substrate or current means, and a mass-gain rate that is linear in
process conditions and scaled by exposure time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class GroupMeanRegressor(BaseEstimator, RegressorMixin):
    """Predict the training mean of the group a row belongs to."""

    def __init__(self, group_column: str):
        self.group_column = group_column

    def fit(self, X, y):
        frame = pd.DataFrame(X)
        frame["__y"] = np.asarray(y, dtype=float)
        self.global_mean_ = float(frame["__y"].mean())
        self.means_ = frame.groupby(self.group_column)["__y"].mean().to_dict()
        return self

    def predict(self, X):
        groups = pd.DataFrame(X)[self.group_column]
        values = groups.map(self.means_).fillna(self.global_mean_)
        return values.to_numpy(dtype=float)


class RateMeanRegressor(BaseEstimator, RegressorMixin):
    """Predict mass gain as a substrate-mean rate multiplied by exposure time."""

    def fit(self, X, y):
        frame = pd.DataFrame(X)
        rates = np.asarray(y, dtype=float) / frame["Exposure_Time_h"].to_numpy(dtype=float)
        frame["__rate"] = rates
        self.global_rate_ = float(np.mean(rates))
        self.rates_ = frame.groupby("Material")["__rate"].mean().to_dict()
        return self

    def predict(self, X):
        frame = pd.DataFrame(X)
        rates = frame["Material"].map(self.rates_).fillna(self.global_rate_)
        return (rates * frame["Exposure_Time_h"]).to_numpy(dtype=float)


class RateLinearRegressor(BaseEstimator, RegressorMixin):
    """Mass-gain rate linear in substrate and current, scaled by exposure time.

    The source tables report cumulative mass gain at 5, 10, and 15 h. A rate model
    states the physical hypothesis directly: gain per unit time depends on the
    process, while the reported target scales with exposure time. It can
    interpolate or extrapolate a rate across welding current, unlike a tree model.
    """

    def fit(self, X, y):
        frame = pd.DataFrame(X)
        rate = np.asarray(y, dtype=float) / frame["Exposure_Time_h"].to_numpy(dtype=float)
        preprocess = ColumnTransformer(
            transformers=[
                (
                    "cat",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=float),
                    ["Material"],
                ),
                ("num", StandardScaler(), ["Current"]),
            ]
        )
        self.pipeline_ = Pipeline(steps=[("preprocess", preprocess), ("model", LinearRegression())])
        self.pipeline_.fit(frame[["Material", "Current"]], rate)
        return self

    def predict(self, X):
        frame = pd.DataFrame(X)
        rate = self.pipeline_.predict(frame[["Material", "Current"]])
        return rate * frame["Exposure_Time_h"].to_numpy(dtype=float)
