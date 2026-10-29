import json
from pathlib import Path

from src.data.schema import dataset_sha256
from src.models.manifest import load_manifest, verify_manifest

MANIFESTS = {
    "hardness": "models/hardness_manifest.json",
    "oxidation": "models/oxidation_manifest.json",
}
DATASETS = {
    "hardness": "data/hardness.csv",
    "oxidation": "data/oxidation.csv",
}


def test_manifests_verify_against_checked_in_artifacts():
    for kind, path in MANIFESTS.items():
        manifest = load_manifest(path)
        assert verify_manifest(path) == []
        assert manifest["dataset"]["sha256"] == dataset_sha256(DATASETS[kind])
        assert manifest["model_version"]
        assert manifest["target"]["definition"]
        assert manifest["known_limitations"]
        assert manifest["metrics"]


def test_manifest_detects_a_changed_dataset(tmp_path):
    manifest = load_manifest(MANIFESTS["hardness"])
    manifest["dataset"]["sha256"] = "0" * 64
    tampered = tmp_path / "manifest.json"
    tampered.write_text(json.dumps(manifest))
    problems = verify_manifest(tampered)
    assert any("dataset sha256" in problem for problem in problems)


def test_manifest_detects_missing_fields(tmp_path):
    manifest = load_manifest(MANIFESTS["oxidation"])
    manifest["validation"] = {}
    manifest["features"] = []
    tampered = tmp_path / "manifest.json"
    tampered.write_text(json.dumps(manifest))
    problems = verify_manifest(tampered)
    assert any("validation" in problem for problem in problems)
    assert any("features" in problem for problem in problems)


def test_manifest_artifact_hash_is_recorded():
    for path in MANIFESTS.values():
        manifest = load_manifest(path)
        assert len(manifest["artifact"]["sha256"]) == 64
        assert Path(manifest["artifact"]["path"]).exists()
