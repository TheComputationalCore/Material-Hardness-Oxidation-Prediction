"""Check that the checked-in data and pipelines still reproduce the recorded baseline.

The check recomputes out-of-fold metrics from the canonical datasets using the
shipped pipelines. It does not read model artifacts to produce predictions, so a
corrupt artifact cannot silently hide behind stale metadata.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict

from src.data.schema import dataset_sha256, load_dataset
from src.models.manifest import verify_manifest
from src.models.pipelines import (
    HARDNESS_FEATURES,
    OXIDATION_FEATURES,
    build_hardness_pipeline,
    build_oxidation_pipeline,
)

EXPECTED_PATH = "reports/reproducibility/expected_metrics.json"
CURRENT_PATH = "reports/reproducibility/current.json"

TARGETS = {
    "hardness": {
        "data_path": "data/hardness.csv",
        "manifest_path": "models/hardness_manifest.json",
        "features": HARDNESS_FEATURES,
        "target": "Hardness",
        "builder": build_hardness_pipeline,
    },
    "oxidation": {
        "data_path": "data/oxidation.csv",
        "manifest_path": "models/oxidation_manifest.json",
        "features": OXIDATION_FEATURES,
        "target": "Areal_Mass_Gain_g_mm2",
        "builder": build_oxidation_pipeline,
    },
}


def _metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(mean_squared_error(y_true, y_pred, squared=False)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def _close(current: float, expected: float, tolerance: dict[str, float]) -> bool:
    return abs(current - expected) <= max(
        tolerance["absolute"], tolerance["relative"] * abs(expected)
    )


def check_reproducibility() -> dict[str, Any]:
    with open(EXPECTED_PATH) as handle:
        expected = json.load(handle)

    tolerance = expected["tolerance"]
    report: dict[str, Any] = {
        "expected_baseline": expected["baseline_id"],
        "protocol": expected["protocol"],
        "targets": {},
        "ok": True,
    }

    for kind, spec in TARGETS.items():
        df = load_dataset(spec["data_path"], kind)
        X = df[spec["features"]]
        y = df[spec["target"]]
        predictions = cross_val_predict(
            spec["builder"](),
            X,
            y,
            groups=df["Current"],
            cv=LeaveOneGroupOut(),
        )
        metrics = _metrics(y, predictions)
        expected_metrics = expected["metrics"][kind]
        metric_checks = {
            name: _close(metrics[name], expected_metrics[name], tolerance)
            for name in expected_metrics
        }

        dataset_sha = dataset_sha256(spec["data_path"])
        dataset_ok = dataset_sha == expected["datasets"][spec["data_path"]]

        manifest_problems: list[str] = []
        manifest_path = Path(spec["manifest_path"])
        if manifest_path.exists():
            manifest_problems = verify_manifest(manifest_path)
            if not manifest_problems:
                manifest = json.loads(manifest_path.read_text())
                if manifest["dataset"]["sha256"] != dataset_sha:
                    manifest_problems.append("manifest dataset hash differs from current data")
                for name, value in expected_metrics.items():
                    if not _close(manifest["metrics"][name], value, tolerance):
                        manifest_problems.append(
                            f"manifest metric {name} differs from the recorded baseline"
                        )
        else:
            manifest_problems.append(f"manifest missing: {manifest_path}")

        target_ok = all(metric_checks.values()) and dataset_ok and not manifest_problems
        report["targets"][kind] = {
            "ok": target_ok,
            "dataset_sha256": dataset_sha,
            "dataset_matches_expected": dataset_ok,
            "metrics": metrics,
            "expected": expected_metrics,
            "metric_matches_expected": metric_checks,
            "manifest_problems": manifest_problems,
        }
        report["ok"] = report["ok"] and target_ok

    return report


def _print_report(report: dict[str, Any]) -> None:
    status = "PASS" if report["ok"] else "FAIL"
    print(f"Reproducibility check: {status} (baseline {report['expected_baseline']})")
    for kind, result in report["targets"].items():
        print(f"- {kind}: {'ok' if result['ok'] else 'FAILED'}")
        print(f"  dataset sha256: {result['dataset_sha256']}")
        for name, value in result["metrics"].items():
            expected = result["expected"][name]
            print(f"  {name}: {value!r} (expected {expected!r})")
        for problem in result["manifest_problems"]:
            print(f"  manifest: {problem}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write",
        action="store_true",
        help=f"Write the observed check to {CURRENT_PATH}",
    )
    args = parser.parse_args(argv)

    report = check_reproducibility()
    _print_report(report)
    if args.write:
        Path(CURRENT_PATH).parent.mkdir(parents=True, exist_ok=True)
        with open(CURRENT_PATH, "w") as handle:
            json.dump(report, handle, indent=4)
            handle.write("\n")
        print(f"Wrote {CURRENT_PATH}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
