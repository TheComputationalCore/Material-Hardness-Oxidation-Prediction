from src.models.reproducibility import check_reproducibility


def test_current_pipelines_reproduce_the_recorded_metrics():
    report = check_reproducibility()
    assert report["ok"] is True
    assert report["expected_baseline"] == "0.3.0"
    for kind, result in report["targets"].items():
        assert result["ok"] is True, (kind, result)
        assert result["dataset_matches_expected"] is True
        assert all(result["metric_matches_expected"].values())
        assert result["manifest_problems"] == []
