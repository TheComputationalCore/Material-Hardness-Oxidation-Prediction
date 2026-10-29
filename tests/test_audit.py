from src.data.audit import run_audit
from src.data.schema import dataset_sha256


def test_audit_reproduces_the_published_tables():
    report = run_audit(write=False)
    assert report["comparison"]["status"] == "pass"
    assert report["comparison"]["rows_checked"] == 24
    assert report["comparison"]["mismatches"] == []
    assert report["hashes"]["data/hardness.csv"] == dataset_sha256("data/hardness.csv")
    assert report["hashes"]["data/oxidation.csv"] == dataset_sha256("data/oxidation.csv")


def test_audit_keeps_oxidation_issues_visible():
    report = run_audit(write=False)
    assert report["time_multiples"]["exact_multiples"] == 11
    assert report["time_multiples"]["near_multiples"] == 1
    assert report["time_multiples"]["total_non_base_rows"] == 12
    assert report["unit_analysis"]["status"] == "warn"
    assert report["unit_analysis"]["alternate_area_mm2"] == 900.0
    assert "raw mass in grams" in report["unit_analysis"]["provisional_reading"]
    assert report["specimen_size"]["status"] == "warn"
    assert report["specimen_size"]["area_ratio"] == 4.0
    assert report["sign_analysis"]["status"] == "warn"
    assert report["independence"]["oxidation"]["effective_independent_observations"] == 6
    assert report["independence"]["hardness"]["effective_independent_observations"] == 6


def test_heat_input_schedule_is_exact():
    report = run_audit(write=False)
    assert report["arithmetic"]["status"] == "pass"
    assert report["arithmetic"]["max_heat_input_difference_kj_mm"] < 1e-3
