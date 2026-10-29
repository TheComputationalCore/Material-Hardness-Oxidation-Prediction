import pandas as pd
import pytest

from src.data.external import (
    HARDNESS_COLUMNS,
    OXIDATION_COLUMNS,
    ExternalDataError,
    load_external,
    validate_external,
)
from src.models.external_validation import evaluate_frame


def _hardness_row(**overrides):
    data = {
        "Row_ID": "EXT-H-001",
        "Material": "EN-8",
        "Current": 140,
        "Voltage_V": 15,
        "Travel_Speed_mm_s": 2.43,
        "Heat_Input": 0.864,
        "Specimen_ID": "EN8-H2-140-01",
        "Heat_Lot": "H-2291",
        "Replicate_ID": "r1",
        "Deposition_Run": "D-2027-03-04-01",
        "Operator": "OP-B",
        "Measurement_Date": "2027-03-04",
        "Test_Load_kgf": 1.0,
        "Indent_Count": 5,
        "Surface_Prep": "ground_1200_polished",
        "Hardness": 360,
        "Source": "LAB-2027-03-04",
        "Data_Status": "new_measurement",
    }
    data.update(overrides)
    return data


def _oxidation_row(**overrides):
    data = {
        "Row_ID": "EXT-O-001",
        "Material": "EN-8",
        "Current": 140,
        "Voltage_V": 15,
        "Travel_Speed_mm_s": 2.43,
        "Heat_Input": 0.864,
        "Exposure_Time_h": 10,
        "Specimen_ID": "EN8-O2-140-01",
        "Heat_Lot": "H-2291",
        "Replicate_ID": "r1",
        "Deposition_Run": "D-2027-03-04-01",
        "Operator": "OP-B",
        "Measurement_Date": "2027-03-04",
        "Exposure_Temperature_K": 973,
        "Furnace_Atmosphere": "air",
        "Heating_Rate_K_min": 10,
        "Cooling_Mode": "furnace_cool",
        "Balance_ID": "BAL-02",
        "Balance_Resolution_mg": 0.1,
        "Exposed_Area_mm2": 225,
        "w1_g": 10.0000,
        "w2_g": 10.0020,
        "Paper_Unit_Value": None,
        "Source": "LAB-2027-03-04",
        "Data_Status": "new_measurement",
    }
    data.update(overrides)
    return data


def _frame(rows, columns):
    return pd.DataFrame(rows, columns=columns)


def test_external_templates_are_empty_and_valid():
    hardness = load_external("data/external/hardness_measurements.csv", "hardness")
    oxidation = load_external("data/external/oxidation_measurements.csv", "oxidation")
    assert len(hardness) == 0
    assert len(oxidation) == 0
    assert "Mass_Change_g" in oxidation.columns


def test_oxidation_derives_mass_change_and_areal_gain():
    validated = validate_external(_frame([_oxidation_row()], OXIDATION_COLUMNS), "oxidation")
    row = validated.iloc[0]
    assert row["Mass_Change_g"] == pytest.approx(0.0020)
    assert row["Areal_Gain_mg_cm2"] == pytest.approx(0.0020 * 100_000 / 225)
    assert row["Mass_Change_Sign"] == "gain"


def test_oxidation_without_paper_value_is_prediction_only():
    validated = validate_external(_frame([_oxidation_row()], OXIDATION_COLUMNS), "oxidation")
    report = evaluate_frame(validated, "oxidation")
    assert report["n_rows"] == 1
    assert report["metrics"]["n_scored"] == 0
    assert report["rows"][0]["status"] == "prediction_only"
    assert report["rows"][0]["prediction"] is not None


def test_oxidation_paper_value_column_is_optional():
    columns = [column for column in OXIDATION_COLUMNS if column != "Paper_Unit_Value"]
    row = {key: value for key, value in _oxidation_row().items() if key != "Paper_Unit_Value"}
    validated = validate_external(_frame([row], columns), "oxidation")
    assert "Paper_Unit_Value" in validated.columns
    assert evaluate_frame(validated, "oxidation")["rows"][0]["status"] == "prediction_only"


def test_oxidation_with_paper_value_is_scored():
    validated = validate_external(
        _frame([_oxidation_row(Paper_Unit_Value=0.0047)], OXIDATION_COLUMNS), "oxidation"
    )
    report = evaluate_frame(validated, "oxidation")
    assert report["metrics"]["n_scored"] == 1
    assert report["rows"][0]["residual"] is not None


def test_hardness_is_scored():
    validated = validate_external(_frame([_hardness_row()], HARDNESS_COLUMNS), "hardness")
    report = evaluate_frame(validated, "hardness")
    assert report["metrics"]["n_scored"] == 1
    assert report["rows"][0]["prediction"] == pytest.approx(355.6666666666667)
    assert report["rows"][0]["residual"] == pytest.approx(355.6666666666667 - 360)


def test_external_rejects_paper_rows():
    with pytest.raises(ExternalDataError, match="Data_Status"):
        validate_external(
            _frame([_hardness_row(Data_Status="reported_paper_measurement")], HARDNESS_COLUMNS),
            "hardness",
        )


def test_external_rejects_bad_area():
    with pytest.raises(ExternalDataError, match="Exposed_Area_mm2"):
        validate_external(
            _frame([_oxidation_row(Exposed_Area_mm2=0)], OXIDATION_COLUMNS), "oxidation"
        )


def test_external_rejects_duplicate_measurements():
    rows = [_hardness_row(), _hardness_row(Row_ID="EXT-H-002")]
    with pytest.raises(ExternalDataError, match="duplicate"):
        validate_external(_frame(rows, HARDNESS_COLUMNS), "hardness")
