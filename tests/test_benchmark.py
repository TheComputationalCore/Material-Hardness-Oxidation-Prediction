import numpy as np

from src.data.schema import load_dataset
from src.models.benchmark import PROTOCOLS, TARGETS, _make_estimator, _run_cv


def _run(kind, protocol, model_name):
    spec = TARGETS[kind]
    df = load_dataset(spec["data_path"], kind).reset_index(drop=True)
    predictions, fold_metrics = _run_cv(
        _make_estimator(kind, model_name),
        df[spec["features"]],
        df[spec["target"]],
        df[PROTOCOLS[protocol]["group_column"]],
        df,
        kind=kind,
        model_name=model_name,
        protocol=protocol,
        time_column=spec["time_column"],
    )
    return df, predictions, fold_metrics


def test_loco_never_trains_on_the_held_out_current():
    for kind, model in (("hardness", "linear"), ("oxidation", "rate_linear")):
        df, predictions, _ = _run(kind, "LOCO_current", model)
        currents = dict(zip(df["Row_ID"], df["Current"], strict=False))
        assert len(predictions) == len(df)
        assert {row["row_id"] for row in predictions} == set(df["Row_ID"])
        for row in predictions:
            train_ids = row["train_row_ids"].split(";")
            assert row["row_id"] not in train_ids
            assert all(currents[row_id] != row["current"] for row_id in train_ids)
            assert np.isfinite(row["predicted"])


def test_logo_never_trains_on_the_held_out_condition():
    for kind, model in (("hardness", "linear"), ("oxidation", "rate_linear")):
        df, predictions, fold_metrics = _run(kind, "LOGO_condition", model)
        groups = dict(zip(df["Row_ID"], df["Observation_Group"], strict=False))
        assert len(predictions) == len(df)
        assert len(fold_metrics) == 6
        for row in predictions:
            train_ids = row["train_row_ids"].split(";")
            assert all(groups[row_id] != row["test_group"] for row_id in train_ids)


def test_fold_metrics_cover_every_prediction_once():
    df, predictions, fold_metrics = _run("oxidation", "LOCO_current", "random_forest")
    assert len(fold_metrics) == 3
    assert sum(fold["n_test"] for fold in fold_metrics) == len(df)
    assert {row["fold"] for row in predictions} == {1, 2, 3}
