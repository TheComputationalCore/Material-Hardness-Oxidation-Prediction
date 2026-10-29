"""Experiment manifests for trained models.

A manifest records everything needed to tell whether a saved artifact belongs to a
particular dataset, configuration, and evaluation result: dataset digest, artifact
digest, git state, environment, feature list, target definition, metrics, and the
validation protocol. Training writes one manifest per model; CI and the test suite
verify it.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

import sklearn

MANIFEST_SCHEMA_VERSION = "1.0"

TRACKED_PACKAGES = [
    "Flask",
    "gunicorn",
    "numpy",
    "pandas",
    "scikit-learn",
    "joblib",
]


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip()


def git_state() -> dict[str, Any]:
    """Describe the commit that produced an artifact, if the worktree is a git repo."""
    commit = _git("rev-parse", "HEAD")
    if commit is None:
        return {"commit": None, "branch": None, "describe": None, "dirty": None}
    status = _git("status", "--porcelain")
    return {
        "commit": commit,
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "describe": _git("describe", "--tags", "--always", "--dirty"),
        "dirty": bool(status),
    }


def environment() -> dict[str, Any]:
    """Capture the runtime that produced an artifact."""
    packages: dict[str, str] = {}
    for name in TRACKED_PACKAGES:
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = "not-installed"
    return {
        "python": sys.version.split()[0],
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "packages": packages,
        "scikit_learn_version": sklearn.__version__,
    }


def pipeline_config(estimator: Any) -> dict[str, Any]:
    """Serialize a pipeline's steps and parameters independent of sklearn reprs."""
    config: dict[str, Any] = {"class": type(estimator).__name__}
    if hasattr(estimator, "get_params"):
        config["params"] = _jsonable(estimator.get_params(deep=False))
    steps = getattr(estimator, "steps", None)
    if steps is not None:
        config["steps"] = [
            {
                "name": name,
                "class": type(step).__name__,
                "params": _jsonable(step.get_params(deep=False)),
            }
            for name, step in steps
        ]
    return config


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    return repr(value)


def write_manifest(
    path: str | Path,
    *,
    model_name: str,
    model_version: str,
    model_path: str | Path,
    estimator: Any,
    dataset_path: str | Path,
    dataset_kind: str,
    dataset_rows: int,
    features: list[str],
    target: str,
    target_definition: str,
    metrics: dict[str, float],
    validation: dict[str, Any],
    seed: int | None,
    limitations: list[str],
) -> dict[str, Any]:
    """Write a manifest describing one trained artifact and return it."""
    manifest = {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "model_name": model_name,
        "model_version": model_version,
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "artifact": {
            "path": str(model_path),
            "sha256": _sha256_file(model_path),
            "bytes": Path(model_path).stat().st_size,
        },
        "dataset": {
            "path": str(dataset_path),
            "kind": dataset_kind,
            "sha256": _sha256_file(dataset_path),
            "rows": dataset_rows,
        },
        "config": pipeline_config(estimator),
        "seed": seed,
        "features": list(features),
        "target": {
            "column": target,
            "definition": target_definition,
        },
        "metrics": {key: float(value) for key, value in metrics.items()},
        "validation": validation,
        "environment": environment(),
        "git": git_state(),
        "known_limitations": limitations,
    }
    with open(path, "w") as handle:
        json.dump(manifest, handle, indent=4, sort_keys=False)
        handle.write("\n")
    print(f"Manifest saved to: {path}")
    return manifest


def load_manifest(path: str | Path) -> dict[str, Any]:
    with open(path) as handle:
        return json.load(handle)


def verify_manifest(path: str | Path) -> list[str]:
    """Return a list of problems that would make the manifest untrustworthy."""
    manifest = load_manifest(path)
    problems: list[str] = []

    if manifest.get("manifest_schema_version") != MANIFEST_SCHEMA_VERSION:
        problems.append("unexpected manifest_schema_version")

    artifact = manifest.get("artifact", {})
    artifact_path = artifact.get("path")
    if not artifact_path or not Path(artifact_path).exists():
        problems.append(f"artifact missing: {artifact_path}")
    elif _sha256_file(artifact_path) != artifact.get("sha256"):
        problems.append("artifact sha256 does not match the manifest")

    dataset = manifest.get("dataset", {})
    dataset_path = dataset.get("path")
    if not dataset_path or not Path(dataset_path).exists():
        problems.append(f"dataset missing: {dataset_path}")
    elif _sha256_file(dataset_path) != dataset.get("sha256"):
        problems.append("dataset sha256 does not match the manifest")

    for key in ("model_name", "model_version", "features", "metrics", "validation"):
        if not manifest.get(key):
            problems.append(f"manifest field {key!r} is empty")

    return problems
