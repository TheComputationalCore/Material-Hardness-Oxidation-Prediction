"""Score new measurements against the shipped models without retraining.

The report keeps two oxidation quantities separate:

* ``Areal_Gain_mg_cm2`` is computed from the external raw weights and area.
* The model prediction is on the paper's printed scale.

A residual is therefore reported only when the external file supplies
``Paper_Unit_Value`` on the same printed scale. This makes the unresolved source
unit impossible to hide inside an apparently precise external metric.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from src.data.external import ExternalDataError, load_external
from src.inference.predict import predict_hardness, predict_oxidation

DEFAULT_PATHS = {
    "hardness": "data/external/hardness_measurements.csv",
    "oxidation": "data/external/oxidation_measurements.csv",
}
REPORT_DIR = Path("reports/external_validation")


def _blank(value: Any) -> bool:
    return (
        value is None
        or (isinstance(value, float) and math.isnan(value))
        or str(value).strip() == ""
    )


def _predict_row(kind: str, row: Any) -> tuple[float | None, dict | None, str | None, str | None]:
    payload = {
        "Material": str(row["Material"]),
        "Current": float(row["Current"]),
    }
    if kind == "oxidation":
        payload["Exposure_Time_h"] = float(row["Exposure_Time_h"])

    result = predict_hardness(payload) if kind == "hardness" else predict_oxidation(payload)
    if not result.get("ok"):
        return None, None, result.get("error", "prediction failed"), None
    return (
        float(result["prediction"]),
        result.get("interval"),
        None,
        result.get("applicability", {}).get("status"),
    )


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scored = [row for row in rows if row["residual"] is not None]
    if not scored:
        return {
            "n_scored": 0,
            "MAE": None,
            "RMSE": None,
            "R2": None,
            "bias": None,
            "interval_coverage": None,
        }

    observed = np.asarray([row["observed"] for row in scored], dtype=float)
    residual = np.asarray([row["residual"] for row in scored], dtype=float)
    mae = float(np.mean(np.abs(residual)))
    rmse = float(np.sqrt(np.mean(residual**2)))
    if len(scored) >= 2 and np.var(observed) > 0:
        r2 = float(1.0 - np.sum(residual**2) / np.sum((observed - np.mean(observed)) ** 2))
    else:
        r2 = None
    covered = [
        row
        for row in scored
        if row["interval_lower"] is not None and row["interval_upper"] is not None
    ]
    coverage = (
        float(
            np.mean(
                [
                    row["interval_lower"] <= row["observed"] <= row["interval_upper"]
                    for row in covered
                ]
            )
        )
        if covered
        else None
    )
    return {
        "n_scored": len(scored),
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "bias": float(np.mean(residual)),
        "interval_coverage": coverage,
        "n_interval_checked": len(covered),
    }


def evaluate_frame(df, kind: str) -> dict[str, Any]:
    """Evaluate a validated external frame and return a report dictionary."""
    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        prediction, interval, error, applicability = _predict_row(kind, row)
        observed = row["Hardness"] if kind == "hardness" else row.get("Paper_Unit_Value")
        if error is None:
            observed = None if _blank(observed) else float(observed)
            residual = None if observed is None else float(prediction) - observed
        else:
            observed = None if _blank(observed) else float(observed)
            residual = None

        record = {
            "row_id": str(row["Row_ID"]),
            "material": str(row["Material"]),
            "current_a": float(row["Current"]),
            "prediction": prediction,
            "observed": observed,
            "residual": residual,
            "abs_error": None if residual is None else abs(residual),
            "interval_lower": interval["lower"] if interval else None,
            "interval_upper": interval["upper"] if interval else None,
            "interval_covered": (
                None
                if residual is None or not interval
                else bool(interval["lower"] <= observed <= interval["upper"])
            ),
            "applicability": applicability,
            "status": "scored"
            if residual is not None
            else ("prediction_only" if error is None else "unsupported"),
            "error": error,
        }
        if kind == "oxidation":
            record.update(
                {
                    "exposure_h": float(row["Exposure_Time_h"]),
                    "measured_areal_gain_mg_cm2": float(row["Areal_Gain_mg_cm2"]),
                    "measured_mass_change_g": float(row["Mass_Change_g"]),
                    "mass_change_sign": str(row["Mass_Change_Sign"]),
                }
            )
        rows.append(record)

    metrics = _metrics(rows)
    by_condition: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = f"{row['material']}@{row['current_a']:.0f}A"
        if kind == "oxidation":
            key += f"/{row['exposure_h']:.0f}h"
        entry = by_condition.setdefault(key, {"n": 0, "n_scored": 0, "abs_errors": []})
        entry["n"] += 1
        if row["abs_error"] is not None:
            entry["n_scored"] += 1
            entry["abs_errors"].append(row["abs_error"])
    for entry in by_condition.values():
        entry["MAE"] = float(np.mean(entry.pop("abs_errors"))) if entry["abs_errors"] else None
        entry.pop("abs_errors", None)

    return {
        "kind": kind,
        "n_rows": len(rows),
        "n_unsupported": sum(1 for row in rows if row["status"] == "unsupported"),
        "metrics": metrics,
        "by_condition": by_condition,
        "rows": rows,
    }


def evaluate_file(kind: str, path: str | Path) -> dict[str, Any]:
    df = load_external(path, kind)
    if len(df) == 0:
        return {"kind": kind, "n_rows": 0, "note": "Template is empty; no external rows to score."}
    return evaluate_frame(df, kind)


def _write_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# External validation",
        "",
        f"Generated: {report['generated_at']}",
        "",
        "External rows are never used for training. Model predictions are on the",
        "paper's printed scale; see the oxidation note below.",
        "",
    ]
    for kind, result in report["results"].items():
        lines += [f"## {kind.capitalize()}", ""]
        if result["n_rows"] == 0:
            lines += [result.get("note", "No rows."), ""]
            continue
        metrics = result["metrics"]
        lines += [
            f"- Rows: {result['n_rows']}",
            f"- Unsupported by the paper-condition validator: {result['n_unsupported']}",
            f"- Scored: {metrics['n_scored']}",
            f"- MAE: {metrics['MAE']}",
            f"- RMSE: {metrics['RMSE']}",
            f"- R²: {metrics['R2']}",
            f"- Bias: {metrics['bias']}",
            f"- 90% interval coverage: {metrics['interval_coverage']}",
            "",
        ]
        if kind == "oxidation":
            lines += [
                "Physical areal gain is computed from `w2_g - w1_g` and the measured area.",
                "Residuals are shown only where `Paper_Unit_Value` is supplied on the same",
                "scale as the printed paper values. Without that column, the unresolved",
                "unit prevents a meaningful oxidation residual.",
                "",
            ]
        if metrics["n_scored"] == 0:
            lines += ["No rows were scored; see per-row statuses.", ""]
    lines += [
        "## Interpretation",
        "",
        "- These metrics describe the supplied specimens only; they are not a new",
        "  population estimate.",
        "- Hardness and oxidation intervals remain approximate empirical intervals.",
        "- Do not merge external rows into training data without updating the schema,",
        "  baseline, benchmark, and model card.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hardness", default=DEFAULT_PATHS["hardness"])
    parser.add_argument("--oxidation", default=DEFAULT_PATHS["oxidation"])
    parser.add_argument(
        "--write", action="store_true", help="Write reports under reports/external_validation"
    )
    args = parser.parse_args(argv)

    results: dict[str, Any] = {}
    try:
        for kind, path in (
            ("hardness", args.hardness),
            ("oxidation", args.oxidation),
        ):
            if not Path(path).exists():
                results[kind] = {"kind": kind, "n_rows": 0, "note": f"File not found: {path}"}
                continue
            results[kind] = evaluate_file(kind, path)
    except ExternalDataError as error:
        print(error)
        return 1

    report = {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "source": "external measurements; not paper training rows",
        "results": results,
    }
    if args.write:
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        (REPORT_DIR / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        (REPORT_DIR / "summary.md").write_text(_write_markdown(report) + "\n")
        print(f"External validation report written to {REPORT_DIR}")

    for kind, result in results.items():
        if result["n_rows"] == 0:
            print(f"{kind}: {result.get('note', 'no rows')}")
        else:
            print(f"{kind}: {result['n_rows']} rows, {result['metrics']['n_scored']} scored")
    return 0


if __name__ == "__main__":
    sys.exit(main())
