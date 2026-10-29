import pandas as pd
import pytest

from src.data.schema import (
    DataValidationError,
    dataset_sha256,
    load_dataset,
    measured_lookup,
    validate_dataset,
)


def _mutated(kind: str, mutate) -> pd.DataFrame:
    path = "data/hardness.csv" if kind == "hardness" else "data/oxidation.csv"
    frame = pd.read_csv(path)
    result = mutate(frame)
    return frame if result is None else result


def test_canonical_datasets_load_and_validate():
    hardness = load_dataset("data/hardness.csv", "hardness")
    oxidation = load_dataset("data/oxidation.csv", "oxidation")
    assert len(hardness) == 6
    assert len(oxidation) == 18
    assert set(hardness["Material"]) == {"EN-8", "Mild Steel"}
    assert set(oxidation["Exposure_Time_h"]) == {5.0, 10.0, 15.0}


def test_measured_lookup_keys_are_exact_conditions():
    hardness = measured_lookup(load_dataset("data/hardness.csv", "hardness"), "hardness")
    oxidation = measured_lookup(load_dataset("data/oxidation.csv", "oxidation"), "oxidation")
    assert ("EN-8", 140.0, None) in hardness
    assert ("EN-8", 140.0, 10.0) in oxidation
    assert ("EN-8", 130.0, 10.0) not in oxidation


@pytest.mark.parametrize(
    ("kind", "mutate", "message"),
    [
        ("hardness", lambda frame: frame.drop(index=0), "expected 6 rows"),
        (
            "hardness",
            lambda frame: frame.__setitem__("Current", 200.0),
            "outside the paper schedule",
        ),
        (
            "hardness",
            lambda frame: frame.__setitem__("Heat_Input", 0.5),
            "does not match Table 1",
        ),
        (
            "hardness",
            lambda frame: frame.__setitem__("Row_ID", "H-EN8-120"),
            "duplicate Row_ID",
        ),
        (
            "hardness",
            lambda frame: frame.__setitem__("Material", "Aluminium"),
            "unsupported materials",
        ),
        (
            "oxidation",
            lambda frame: frame.loc.__setitem__((6, "Exposure_Time_h"), 5.0),
            "duplicate .* rows",
        ),
        (
            "oxidation",
            lambda frame: frame.__setitem__("Areal_Mass_Gain_g_mm2", -0.001),
            "contains values <= 0.0",
        ),
    ],
)
def test_invalid_datasets_are_rejected(kind, mutate, message):
    frame = _mutated(kind, mutate)
    with pytest.raises(DataValidationError, match=message):
        validate_dataset(frame, kind)


def test_unknown_dataset_kind_is_rejected():
    with pytest.raises(DataValidationError, match="Unknown dataset kind"):
        validate_dataset(pd.DataFrame(), "unknown")


def test_missing_required_column_is_rejected():
    frame = pd.read_csv("data/hardness.csv").drop(columns=["Source"])
    with pytest.raises(DataValidationError, match="missing required columns"):
        validate_dataset(frame, "hardness")


def test_dataset_hash_is_stable():
    first = dataset_sha256("data/oxidation.csv")
    second = dataset_sha256("data/oxidation.csv")
    assert first == second
    assert len(first) == 64
