"""Fair comparison of simple models for hardness and oxidation.

Every model is evaluated under the same grouped protocols, with the same rows and
fold definitions, and every out-of-fold prediction is written out for inspection.
The shipped pipelines are included under their production names; challengers are
reported as evidence only.

Protocols
---------
LOCO_current (primary)
    Leave-one-current-out over the three welding currents. This is the honest
    stress test for the question the paper can answer: how well does a model
    transfer to a process setting it has not seen?
LOGO_condition (secondary)
    Leave-one-condition-out over the six Material x Current cells. This is closer
    to interpolation, but the other substrate at the same current remains in the
    training fold, so process information leaks across materials.

Outputs (reports/benchmark/):
- oof_predictions.csv: one row per model, protocol, fold, and held-out observation
- fold_metrics.csv: fold-level MAE, RMSE, and R^2
- summary.md: rankings, fold wins, sensitivity notes, and the selection decision
- selection.json: machine-readable ranking and decision
- sensitivity.csv: change in primary-protocol MAE when one observation is dropped
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data.schema import load_dataset
from src.models.estimators import GroupMeanRegressor, RateLinearRegressor, RateMeanRegressor
from src.models.pipelines import (
    build_hardness_pipeline,
    build_oxidation_random_forest_pipeline,
)

OUTPUT_DIR = "reports/benchmark"

# Adoption rule fixed before looking at the results.
IMPROVEMENT_THRESHOLD = 0.20
SECONDARY_TOLERANCE = 0.10
MIN_FOLDS_IMPROVED = 2

TARGETS: dict[str, dict[str, Any]] = {
    "hardness": {
        "data_path": "data/hardness.csv",
        "features": ["Material", "Current"],
        "target": "Hardness",
        "shipped": "linear",
        "time_column": None,
    },
    "oxidation": {
        "data_path": "data/oxidation.csv",
        "features": ["Material", "Current", "Exposure_Time_h"],
        "target": "Areal_Mass_Gain_g_mm2",
        "shipped": "rate_linear",
        "time_column": "Exposure_Time_h",
    },
}

PROTOCOLS = {
    "LOCO_current": {
        "group_column": "Current",
        "description": (
            "Leave-one-current-out: all rows at the held-out current are excluded "
            "from training, so the model must transfer to an unseen process setting."
        ),
        "role": "primary",
    },
    "LOGO_condition": {
        "group_column": "Observation_Group",
        "description": (
            "Leave-one-condition-out: all rows for one Material x Current cell are "
            "held out. The other substrate at the same current stays in training, "
            "so this measures interpolation only."
        ),
        "role": "secondary",
    },
}


# ---------------------------------------------------------------------------
# Small, transparent baselines
# ---------------------------------------------------------------------------


def _make_estimator(kind: str, name: str) -> BaseEstimator:
    if name == "global_mean":
        return DummyRegressor(strategy="mean")
    if name == "global_median":
        return DummyRegressor(strategy="median")
    if name == "substrate_mean":
        return GroupMeanRegressor(group_column="Material")
    if name == "current_mean":
        return GroupMeanRegressor(group_column="Current")
    if name == "time_mean":
        return GroupMeanRegressor(group_column="Exposure_Time_h")
    if name == "rate_mean":
        return RateMeanRegressor()
    if name == "rate_linear":
        return RateLinearRegressor()
    if kind == "hardness":
        if name == "linear":
            return build_hardness_pipeline()
        if name == "ridge":
            return Pipeline(
                steps=[
                    ("preprocess", _preprocessor(["Current"])),
                    ("model", Ridge(alpha=1.0)),
                ]
            )
        if name == "random_forest":
            return Pipeline(
                steps=[
                    ("preprocess", _preprocessor(["Current"])),
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
    if kind == "oxidation":
        if name == "random_forest":
            return build_oxidation_random_forest_pipeline()
        if name == "linear_full":
            return Pipeline(
                steps=[
                    ("preprocess", _preprocessor(["Current", "Exposure_Time_h"])),
                    ("model", LinearRegression()),
                ]
            )
        if name == "ridge_full":
            return Pipeline(
                steps=[
                    ("preprocess", _preprocessor(["Current", "Exposure_Time_h"])),
                    ("model", Ridge(alpha=1.0)),
                ]
            )
        if name == "linear_time":
            return Pipeline(
                steps=[
                    (
                        "preprocess",
                        ColumnTransformer(
                            [("num", StandardScaler(), ["Current", "Exposure_Time_h"])]
                        ),
                    ),
                    ("model", LinearRegression()),
                ]
            )
    raise ValueError(f"Unknown model {name!r} for target {kind!r}")


def _preprocessor(numeric_features: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=float),
                ["Material"],
            ),
            ("num", StandardScaler(), numeric_features),
        ]
    )


HARDNESS_MODELS = [
    "global_mean",
    "global_median",
    "substrate_mean",
    "current_mean",
    "linear",
    "ridge",
    "random_forest",
]

OXIDATION_MODELS = [
    "global_mean",
    "global_median",
    "substrate_mean",
    "time_mean",
    "rate_mean",
    "rate_linear",
    "linear_time",
    "linear_full",
    "ridge_full",
    "random_forest",
]


# ---------------------------------------------------------------------------
# Cross-validation with full provenance for every prediction
# ---------------------------------------------------------------------------


def _run_cv(
    estimator: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    rows: pd.DataFrame,
    *,
    kind: str,
    model_name: str,
    protocol: str,
    time_column: str | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    splitter = LeaveOneGroupOut()
    predictions: list[dict[str, Any]] = []
    fold_metrics: list[dict[str, Any]] = []

    for fold, (train_idx, test_idx) in enumerate(splitter.split(X, y, groups=groups), start=1):
        model = clone(estimator)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        predicted = np.asarray(model.predict(X.iloc[test_idx]), dtype=float)

        observed = y.iloc[test_idx].to_numpy(dtype=float)
        fold_metrics.append(
            {
                "target": kind,
                "model": model_name,
                "protocol": protocol,
                "fold": fold,
                "test_group": str(sorted(set(groups.iloc[test_idx]))),
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "MAE": float(mean_absolute_error(observed, predicted)),
                "RMSE": float(mean_squared_error(observed, predicted, squared=False)),
                "R2": (float(r2_score(observed, predicted)) if len(test_idx) > 1 else float("nan")),
            }
        )

        train_currents = X.iloc[train_idx]["Current"].to_numpy(dtype=float)
        train_times = X.iloc[train_idx][time_column].to_numpy(dtype=float) if time_column else None
        for offset, row_index in enumerate(test_idx):
            row = rows.iloc[row_index]
            current = float(row["Current"])
            exposure = float(row[time_column]) if time_column else None
            predictions.append(
                {
                    "target": kind,
                    "model": model_name,
                    "protocol": protocol,
                    "fold": fold,
                    "test_group": str(groups.iloc[row_index]),
                    "row_id": row["Row_ID"],
                    "material": row["Material"],
                    "current": current,
                    "exposure_time_h": exposure,
                    "observed": float(observed[offset]),
                    "predicted": float(predicted[offset]),
                    "residual": float(predicted[offset] - observed[offset]),
                    "abs_error": float(abs(predicted[offset] - observed[offset])),
                    "n_train": int(len(train_idx)),
                    "train_row_ids": ";".join(rows.iloc[train_idx]["Row_ID"]),
                    "train_current_min": float(train_currents.min()),
                    "train_current_max": float(train_currents.max()),
                    "current_extrapolated": bool(
                        current < train_currents.min() or current > train_currents.max()
                    ),
                    "train_time_min": (
                        float(train_times.min()) if train_times is not None else None
                    ),
                    "train_time_max": (
                        float(train_times.max()) if train_times is not None else None
                    ),
                    "time_extrapolated": (
                        bool(exposure < train_times.min() or exposure > train_times.max())
                        if train_times is not None
                        else None
                    ),
                }
            )

    return predictions, fold_metrics


def _pooled_metrics(frame: pd.DataFrame) -> dict[str, float]:
    observed = frame["observed"].to_numpy(dtype=float)
    predicted = frame["predicted"].to_numpy(dtype=float)
    return {
        "MAE": float(mean_absolute_error(observed, predicted)),
        "RMSE": float(mean_squared_error(observed, predicted, squared=False)),
        "R2": float(r2_score(observed, predicted)),
        "nMAE": float(mean_absolute_error(observed, predicted) / np.mean(np.abs(observed))),
    }


# ---------------------------------------------------------------------------
# Benchmark orchestration
# ---------------------------------------------------------------------------


def _summarize(
    predictions: pd.DataFrame, fold_metrics: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pooled_rows = []
    for (kind, protocol, model), group in predictions.groupby(
        ["target", "protocol", "model"], sort=False
    ):
        metrics = _pooled_metrics(group)
        folds = fold_metrics[
            (fold_metrics["target"] == kind)
            & (fold_metrics["protocol"] == protocol)
            & (fold_metrics["model"] == model)
        ]
        pooled_rows.append(
            {
                "target": kind,
                "protocol": protocol,
                "model": model,
                "MAE": metrics["MAE"],
                "RMSE": metrics["RMSE"],
                "R2": metrics["R2"],
                "nMAE": metrics["nMAE"],
                "fold_MAE_mean": float(folds["MAE"].mean()),
                "fold_MAE_std": float(folds["MAE"].std(ddof=0)),
                "n_folds": int(len(folds)),
            }
        )
    pooled = pd.DataFrame(pooled_rows)

    # Fold wins: a model wins a fold when its fold MAE is the smallest for that target/protocol/fold.
    fold_wins = []
    for (kind, protocol, fold), group in fold_metrics.groupby(
        ["target", "protocol", "fold"], sort=False
    ):
        best = group["MAE"].min()
        ranks = group["MAE"].rank(method="min").astype(int)
        for model, rank in zip(group["model"], ranks, strict=False):
            fold_wins.append(
                {
                    "target": kind,
                    "protocol": protocol,
                    "fold": fold,
                    "model": model,
                    "is_best": bool(
                        np.isclose(group.loc[group["model"] == model, "MAE"].iloc[0], best)
                    ),
                    "rank": int(rank),
                }
            )
    wins = pd.DataFrame(fold_wins)
    pooled["folds_best"] = [
        int(
            wins[
                (wins["target"] == row.target)
                & (wins["protocol"] == row.protocol)
                & (wins["model"] == row.model)
            ]["is_best"].sum()
        )
        for row in pooled.itertuples()
    ]
    pooled = pooled.sort_values(["target", "protocol", "MAE"]).reset_index(drop=True)
    return pooled, wins


def _decision(
    pooled: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    wins: pd.DataFrame,
    kind: str,
    shipped: str,
) -> dict[str, Any]:
    primary = pooled[(pooled["target"] == kind) & (pooled["protocol"] == "LOCO_current")]
    secondary = pooled[(pooled["target"] == kind) & (pooled["protocol"] == "LOGO_condition")]
    ranking = primary.sort_values("MAE")["model"].tolist()
    best = ranking[0]
    shipped_primary = primary.loc[primary["model"] == shipped].iloc[0]
    shipped_secondary = secondary.loc[secondary["model"] == shipped].iloc[0]
    best_row = primary.loc[primary["model"] == best].iloc[0]
    relative_improvement = float(
        (shipped_primary["MAE"] - best_row["MAE"]) / shipped_primary["MAE"]
    )

    primary_folds = sorted(
        fold_metrics[
            (fold_metrics["target"] == kind) & (fold_metrics["protocol"] == "LOCO_current")
        ]["fold"].unique()
    )
    folds_improved = 0
    for fold in primary_folds:
        shipped_fold = fold_metrics[
            (fold_metrics["target"] == kind)
            & (fold_metrics["protocol"] == "LOCO_current")
            & (fold_metrics["model"] == shipped)
            & (fold_metrics["fold"] == fold)
        ]["MAE"].iloc[0]
        best_fold = fold_metrics[
            (fold_metrics["target"] == kind)
            & (fold_metrics["protocol"] == "LOCO_current")
            & (fold_metrics["model"] == best)
            & (fold_metrics["fold"] == fold)
        ]["MAE"].iloc[0]
        if best_fold < shipped_fold:
            folds_improved += 1

    best_secondary = secondary.loc[secondary["model"] == best].iloc[0]
    secondary_relative_change = float(
        (best_secondary["MAE"] - shipped_secondary["MAE"]) / shipped_secondary["MAE"]
    )
    meets = (
        best != shipped
        and relative_improvement >= IMPROVEMENT_THRESHOLD
        and folds_improved >= MIN_FOLDS_IMPROVED
        and secondary_relative_change <= SECONDARY_TOLERANCE
    )
    if best == shipped:
        decision = "keep_shipped"
        rationale = "The shipped model remains the best under the primary protocol."
    elif meets:
        decision = "challenger_meets_criteria"
        rationale = (
            f"{best} reduces primary LOCO MAE by {relative_improvement:.1%} "
            f"(threshold {IMPROVEMENT_THRESHOLD:.0%}), wins {folds_improved} of "
            f"{len(primary_folds)} fold comparisons, and changes secondary "
            f"LOGO MAE by {secondary_relative_change:+.1%} (limit "
            f"{SECONDARY_TOLERANCE:.0%}). Recommend review for adoption in a separate change."
        )
    else:
        decision = "keep_shipped"
        reasons = []
        if relative_improvement < IMPROVEMENT_THRESHOLD:
            reasons.append(
                f"improvement {relative_improvement:.1%} is below the "
                f"{IMPROVEMENT_THRESHOLD:.0%} threshold"
            )
        if folds_improved < MIN_FOLDS_IMPROVED:
            reasons.append(f"only {folds_improved} of {len(primary_folds)} folds improve")
        if secondary_relative_change > SECONDARY_TOLERANCE:
            reasons.append(f"secondary protocol degrades by {secondary_relative_change:.1%}")
        rationale = f"Best challenger {best}: " + "; ".join(reasons) + ". Keep the shipped model."

    return {
        "target": kind,
        "shipped_model": shipped,
        "best_model": best,
        "ranking": ranking,
        "shipped_primary_MAE": float(shipped_primary["MAE"]),
        "best_primary_MAE": float(best_row["MAE"]),
        "relative_improvement": relative_improvement,
        "folds_improved": folds_improved,
        "n_folds": int(len(primary_folds)),
        "secondary_relative_change": secondary_relative_change,
        "criteria": {
            "improvement_threshold": IMPROVEMENT_THRESHOLD,
            "min_folds_improved": MIN_FOLDS_IMPROVED,
            "secondary_tolerance": SECONDARY_TOLERANCE,
        },
        "folds_best": {
            model: int(
                wins[
                    (wins["target"] == kind)
                    & (wins["protocol"] == "LOCO_current")
                    & (wins["model"] == model)
                ]["is_best"].sum()
            )
            for model in ranking
        },
        "decision": decision,
        "rationale": rationale,
    }


def _run_sensitivity(
    kind: str,
    spec: dict[str, Any],
    df: pd.DataFrame,
    model_names: list[str],
    baseline: dict[str, float],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model_name in model_names:
        for _, dropped in df.iterrows():
            subset = df[df["Row_ID"] != dropped["Row_ID"]].reset_index(drop=True)
            X = subset[spec["features"]]
            y = subset[spec["target"]]
            predictions, _ = _run_cv(
                _make_estimator(kind, model_name),
                X,
                y,
                subset["Current"],
                subset,
                kind=kind,
                model_name=model_name,
                protocol="LOCO_current",
                time_column=spec["time_column"],
            )
            observed = np.array([row["observed"] for row in predictions])
            predicted = np.array([row["predicted"] for row in predictions])
            mae = float(mean_absolute_error(observed, predicted))
            base = baseline[model_name]
            rows.append(
                {
                    "target": kind,
                    "model": model_name,
                    "dropped_row_id": dropped["Row_ID"],
                    "baseline_MAE": base,
                    "MAE_without_row": mae,
                    "delta_MAE": mae - base,
                    "relative_delta": (mae - base) / base,
                }
            )
    return rows


def _fmt(value: float, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    if value != 0 and abs(value) < 1e-3:
        return f"{value:.3e}"
    return f"{value:.{digits}f}"


def _write_summary(
    pooled: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    selections: list[dict[str, Any]],
    sensitivity: pd.DataFrame,
    output_dir: Path,
) -> None:
    lines = [
        "# Benchmark: simple models under grouped validation",
        "",
        "Generated by `python -m src.models.benchmark`. Every number comes from",
        "out-of-fold predictions in `oof_predictions.csv`; fold detail is in",
        "`fold_metrics.csv` and the decision record is in `selection.json`.",
        "",
        "The shipped pipelines are marked **(shipped)**. Challengers use the same",
        "rows and folds, so differences are not caused by different validation.",
        "",
        "## Protocols",
        "",
    ]
    for name, protocol in PROTOCOLS.items():
        lines.append(f"- **{name}** ({protocol['role']}): {protocol['description']}")
    lines += [
        "",
        "The oxidation dataset has 18 rows but only six independent Material x Current",
        "cells: the 10 h and 15 h values are exact multiples of the 5 h value. Treat",
        "oxidation metrics as six-condition evidence, not eighteen-sample evidence.",
        "",
    ]

    for kind in TARGETS:
        for protocol in PROTOCOLS:
            subset = pooled[(pooled["target"] == kind) & (pooled["protocol"] == protocol)]
            if subset.empty:
                continue
            lines += [
                f"## {kind.title()} — {protocol}",
                "",
                "| Model | MAE | RMSE | R² | nMAE | Fold MAE (mean ± sd) | Folds best |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
            for row in subset.itertuples():
                marker = " (shipped)" if row.model == TARGETS[kind]["shipped"] else ""
                lines.append(
                    f"| {row.model}{marker} | {_fmt(row.MAE)} | {_fmt(row.RMSE)} | "
                    f"{_fmt(row.R2)} | {row.nMAE:.3f} | "
                    f"{_fmt(row.fold_MAE_mean)} ± {_fmt(row.fold_MAE_std)} | "
                    f"{row.folds_best}/{row.n_folds} |"
                )
            lines.append("")

    lines += [
        "## Adoption history",
        "",
        "- **v0.3.0**: oxidation adopted `rate_linear` after the archived pre-adoption",
        "  benchmark in `pre_adoption_random_forest/` showed 69.4% lower LOCO MAE,",
        "  improvement in 3 of 3 folds, and a 47.2% lower secondary LOGO MAE than the",
        "  previous random forest. The hardness model is unchanged.",
        "",
        "## Selection",
        "",
        "Adoption rule, fixed before running: adopt a challenger only if primary LOCO MAE",
        f"improves by at least {IMPROVEMENT_THRESHOLD:.0%}, the improvement appears in at",
        f"least {MIN_FOLDS_IMPROVED} folds, and secondary LOGO MAE does not worsen by more",
        f"than {SECONDARY_TOLERANCE:.0%}.",
        "",
    ]
    for selection in selections:
        lines += [
            f"### {selection['target'].title()}",
            "",
            f"- Decision: **{selection['decision']}**",
            f"- Shipped model: `{selection['shipped_model']}` "
            f"(primary MAE {_fmt(selection['shipped_primary_MAE'])})",
            f"- Best challenger: `{selection['best_model']}` "
            f"(primary MAE {_fmt(selection['best_primary_MAE'])})",
            f"- Ranking: {', '.join(selection['ranking'])}",
            f"- Rationale: {selection['rationale']}",
            "",
        ]

    lines += [
        "## Sensitivity to single observations",
        "",
        "Primary-protocol MAE after dropping one row and rerunning every fold. Large",
        "swings mean the ranking is not robust to a single reported value.",
        "",
    ]
    if not sensitivity.empty:
        lines += [
            "| Target | Model | Dropped row | ΔMAE | Relative Δ |",
            "| --- | --- | --- | ---: | ---: |",
        ]
        shocks = sensitivity.reindex(
            sensitivity.groupby(["target", "model"])["relative_delta"]
            .apply(lambda values: values.abs().idxmax())
            .values
        )
        for row in shocks.itertuples():
            lines.append(
                f"| {row.target} | {row.model} | {row.dropped_row_id} | "
                f"{_fmt(row.delta_MAE)} | {row.relative_delta:+.1%} |"
            )
        lines += [
            "",
            "Full per-row results are in `sensitivity.csv`.",
            "",
        ]

    lines += [
        "## Notes",
        "",
        "- Random forests cannot extrapolate beyond the training current range; folds",
        "  that hold out 120 A or 160 A therefore test a known weakness rather than a",
        "  fair interpolation scenario.",
        "- The application only accepts the paper-tested conditions, so these numbers",
        "  describe transfer across the paper's process schedule, not new processes.",
        "- See `docs/MODEL_CARD.md` for intended use and `reports/audit/` for the",
        "  unresolved unit, sign, and independence findings.",
        "",
    ]
    (output_dir / "summary.md").write_text("\n".join(lines))


def run_benchmark(output_dir: str | Path = OUTPUT_DIR) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    all_predictions: list[dict[str, Any]] = []
    all_fold_metrics: list[dict[str, Any]] = []

    datasets: dict[str, pd.DataFrame] = {}
    for kind, spec in TARGETS.items():
        df = load_dataset(spec["data_path"], kind).reset_index(drop=True)
        datasets[kind] = df
        for protocol, protocol_spec in PROTOCOLS.items():
            model_names = HARDNESS_MODELS if kind == "hardness" else OXIDATION_MODELS
            for model_name in model_names:
                predictions, fold_metrics = _run_cv(
                    _make_estimator(kind, model_name),
                    df[spec["features"]],
                    df[spec["target"]],
                    df[protocol_spec["group_column"]],
                    df,
                    kind=kind,
                    model_name=model_name,
                    protocol=protocol,
                    time_column=spec["time_column"],
                )
                all_predictions.extend(predictions)
                all_fold_metrics.extend(fold_metrics)

    predictions_frame = pd.DataFrame(all_predictions)
    fold_frame = pd.DataFrame(all_fold_metrics)
    predictions_frame.to_csv(output / "oof_predictions.csv", index=False)
    fold_frame.to_csv(output / "fold_metrics.csv", index=False)

    pooled, wins = _summarize(predictions_frame, fold_frame)
    selections = [
        _decision(pooled, fold_frame, wins, kind, spec["shipped"]) for kind, spec in TARGETS.items()
    ]

    sensitivity_rows: list[dict[str, Any]] = []
    for kind, spec in TARGETS.items():
        primary = pooled[
            (pooled["target"] == kind) & (pooled["protocol"] == "LOCO_current")
        ].sort_values("MAE")
        focus = [spec["shipped"]]
        for model in primary["model"].tolist():
            if model not in focus:
                focus.append(model)
            if len(focus) == 3:
                break
        baseline = primary.set_index("model")["MAE"].to_dict()
        sensitivity_rows.extend(_run_sensitivity(kind, spec, datasets[kind], focus, baseline))
    sensitivity = pd.DataFrame(sensitivity_rows)
    sensitivity.to_csv(output / "sensitivity.csv", index=False)

    _write_summary(pooled, fold_frame, selections, sensitivity, output)
    selection_record = {
        "rule": {
            "primary_protocol": "LOCO_current",
            "secondary_protocol": "LOGO_condition",
            "improvement_threshold": IMPROVEMENT_THRESHOLD,
            "min_folds_improved": MIN_FOLDS_IMPROVED,
            "secondary_tolerance": SECONDARY_TOLERANCE,
        },
        "targets": selections,
    }
    with open(output / "selection.json", "w") as handle:
        json.dump(selection_record, handle, indent=4)
        handle.write("\n")

    print(pooled.to_string(index=False))
    for selection in selections:
        print(f"{selection['target']}: {selection['decision']} — {selection['rationale']}")
    return {
        "pooled": pooled,
        "fold_metrics": fold_frame,
        "predictions": predictions_frame,
        "sensitivity": sensitivity,
        "selection": selection_record,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    run_benchmark(parser.parse_args().output_dir)
