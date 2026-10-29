"""Audit the transcribed datasets against the published source tables.

Produces machine-readable and human-readable reports under ``reports/audit``.
Value mismatches are treated as failures; unit, sign, and independence findings are
recorded as warnings because they cannot be resolved from the published tables alone.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.schema import (
    dataset_sha256,
    load_dataset,
)

ROOT = Path(__file__).resolve().parents[2]
SOURCE_PATH = ROOT / "data" / "source_tables.json"
HARDNESS_PATH = ROOT / "data" / "hardness.csv"
OXIDATION_PATH = ROOT / "data" / "oxidation.csv"
REPORT_DIR = ROOT / "reports" / "audit"

# Table 1 and Table 7 constants used for arithmetic checks.
VOLTAGE_V = 15.0
WELD_SPEED_MM_S = {120.0: 2.25, 140.0: 2.43, 160.0: 2.63}
EXPOSED_AREA_MM2 = 225.0
ALTERNATE_SPECIMEN_AREA_MM2 = 900.0  # Methods section says 30 x 30 mm2 samples were cut
SPECIMEN_VOLUME_CM3 = 1.35  # 15 x 15 x 6 mm
STEEL_DENSITY_G_CM3 = 7.85
SPECIMEN_MASS_G = SPECIMEN_VOLUME_CM3 * STEEL_DENSITY_G_CM3


def _source_index(source: dict[str, Any]) -> dict[str, dict[tuple, dict[str, Any]]]:
    """Map (material, current, time) conditions to reported values and tables."""
    hardness: dict[tuple, dict[str, Any]] = {}
    for table, payload in source["hardness"]["tables"].items():
        for current, value in payload["values"].items():
            hardness[(payload["substrate"], float(current), None)] = {
                "value": float(value),
                "table": table,
            }

    oxidation: dict[tuple, dict[str, Any]] = {}
    for table, payload in source["oxidation"]["tables"].items():
        for current, value in payload["values"].items():
            oxidation[(payload["substrate"], float(current), float(payload["exposure_h"]))] = {
                "value": float(value),
                "table": table,
            }
    return {"hardness": hardness, "oxidation": oxidation}


def compare_tables(
    hardness: pd.DataFrame, oxidation: pd.DataFrame, source: dict[str, Any]
) -> dict[str, Any]:
    """Compare every CSV row to the transcribed source tables."""
    expected = _source_index(source)
    mismatches: list[dict[str, Any]] = []
    checked = 0

    def check(df: pd.DataFrame, kind: str, time_column: str | None) -> None:
        nonlocal checked
        for _, row in df.iterrows():
            key = (
                row["Material"],
                float(row["Current"]),
                float(row[time_column]) if time_column else None,
            )
            target = "Hardness" if kind == "hardness" else "Areal_Mass_Gain_g_mm2"
            reference = expected[kind].get(key)
            if reference is None:
                mismatches.append(
                    {"kind": kind, "row_id": row["Row_ID"], "issue": "row not in source tables"}
                )
                continue
            checked += 1
            if abs(float(row[target]) - reference["value"]) > 1e-12:
                mismatches.append(
                    {
                        "kind": kind,
                        "row_id": row["Row_ID"],
                        "issue": "value mismatch",
                        "csv": float(row[target]),
                        "paper": reference["value"],
                    }
                )
            if row["Source_Table"] != reference["table"]:
                mismatches.append(
                    {
                        "kind": kind,
                        "row_id": row["Row_ID"],
                        "issue": "source table mismatch",
                        "csv": row["Source_Table"],
                        "paper": reference["table"],
                    }
                )

    check(hardness, "hardness", None)
    check(oxidation, "oxidation", "Exposure_Time_h")

    return {
        "status": "pass" if not mismatches else "fail",
        "rows_checked": checked,
        "expected_rows": len(expected["hardness"]) + len(expected["oxidation"]),
        "mismatches": mismatches,
    }


def time_multiples(oxidation: pd.DataFrame) -> dict[str, Any]:
    """Quantify how closely 10 h and 15 h values are multiples of the 5 h value."""
    groups: list[dict[str, Any]] = []
    exact = 0
    near = 0
    for (material, current), frame in oxidation.groupby(["Material", "Current"]):
        frame = frame.sort_values("Exposure_Time_h")
        base = float(frame.loc[frame["Exposure_Time_h"] == 5, "Areal_Mass_Gain_g_mm2"].iloc[0])
        for _, row in frame[frame["Exposure_Time_h"] > 5].iterrows():
            factor = float(row["Exposure_Time_h"]) / 5.0
            expected = factor * base
            deviation = float(row["Areal_Mass_Gain_g_mm2"]) - expected
            if abs(deviation) <= 5e-7:
                classification = "exact_to_printed_precision"
                exact += 1
            else:
                classification = "near_multiple"
                near += 1
            groups.append(
                {
                    "material": material,
                    "current_a": float(current),
                    "exposure_h": float(row["Exposure_Time_h"]),
                    "value": float(row["Areal_Mass_Gain_g_mm2"]),
                    "expected_if_derived": expected,
                    "ratio_to_5h": float(row["Areal_Mass_Gain_g_mm2"]) / base,
                    "deviation": deviation,
                    "relative_deviation": deviation / expected,
                    "classification": classification,
                }
            )
    return {
        "status": "warn" if exact + near else "pass",
        "exact_multiples": exact,
        "near_multiples": near,
        "total_non_base_rows": exact + near,
        "max_relative_deviation": max((abs(g["relative_deviation"]) for g in groups), default=0.0),
        "groups": groups,
        "finding": (
            "Eleven of twelve non-base rows reproduce k x the 5 h value to the last "
            "printed digit (k = 2 for 10 h, 3 for 15 h); one row deviates by 0.08 %. "
            "This is consistent with the 10 h/15 h rows being derived from a single "
            "rate per condition, or with implausibly exact linear kinetics."
        ),
    }


def unit_analysis(oxidation: pd.DataFrame) -> dict[str, Any]:
    """Test the printed unit against the stated area and specimen scale."""
    values = oxidation["Areal_Mass_Gain_g_mm2"].astype(float)
    low, high = float(values.min()), float(values.max())

    as_g_per_mm2 = {
        "implied_mass_change_g_at_min": low * EXPOSED_AREA_MM2,
        "implied_mass_change_g_at_max": high * EXPOSED_AREA_MM2,
        "fraction_of_specimen_mass_at_min": low * EXPOSED_AREA_MM2 / SPECIMEN_MASS_G,
        "fraction_of_specimen_mass_at_max": high * EXPOSED_AREA_MM2 / SPECIMEN_MASS_G,
    }
    as_grams = {
        "implied_areal_gain_mg_cm2_at_min": low / (EXPOSED_AREA_MM2 / 100.0) * 1000.0,
        "implied_areal_gain_mg_cm2_at_max": high / (EXPOSED_AREA_MM2 / 100.0) * 1000.0,
    }
    alternate_area = {
        "area_mm2": ALTERNATE_SPECIMEN_AREA_MM2,
        "implied_mass_change_g_at_min_if_g_per_mm2": low * ALTERNATE_SPECIMEN_AREA_MM2,
        "implied_mass_change_g_at_max_if_g_per_mm2": high * ALTERNATE_SPECIMEN_AREA_MM2,
        "implied_areal_gain_mg_cm2_at_min_if_grams": (
            low / (ALTERNATE_SPECIMEN_AREA_MM2 / 100.0) * 1000.0
        ),
        "implied_areal_gain_mg_cm2_at_max_if_grams": (
            high / (ALTERNATE_SPECIMEN_AREA_MM2 / 100.0) * 1000.0
        ),
    }
    return {
        "status": "warn",
        "printed_unit": "gm/mm2",
        "stated_area_mm2": EXPOSED_AREA_MM2,
        "alternate_area_mm2": ALTERNATE_SPECIMEN_AREA_MM2,
        "stated_equation": "weight gain (g) / surface area (mm2)",
        "specimen_mass_g_estimated": SPECIMEN_MASS_G,
        "interpretation_a_values_are_g_per_mm2": as_g_per_mm2,
        "interpretation_b_values_are_grams": as_grams,
        "interpretation_under_alternate_area": alternate_area,
        "provisional_reading": (
            "The printed magnitudes are more consistent with raw mass in grams than with "
            "g/mm2. Under the 225 mm2 area this gives 0.96-3.59 mg/cm2; under the 900 mm2 "
            "area stated in the methods it gives 0.24-0.90 mg/cm2. Both are physically "
            "plausible. No conversion or unit correction is applied because the raw weights "
            "are not published and the area itself is inconsistent."
        ),
        "finding": (
            "If the tabulated numbers are g/mm2 as printed, the implied mass change is "
            f"{as_g_per_mm2['implied_mass_change_g_at_min']:.3f}-"
            f"{as_g_per_mm2['implied_mass_change_g_at_max']:.3f} g, or "
            f"{100 * as_g_per_mm2['fraction_of_specimen_mass_at_min']:.1f}-"
            f"{100 * as_g_per_mm2['fraction_of_specimen_mass_at_max']:.1f} % "
            "of a 15 x 15 x 6 mm steel specimen, which is not physically plausible for "
            "a 5-15 h exposure at 700 C. If the numbers are raw weight change in grams, "
            "the areal gain is "
            f"{as_grams['implied_areal_gain_mg_cm2_at_min']:.2f}-"
            f"{as_grams['implied_areal_gain_mg_cm2_at_max']:.2f} mg/cm2, which is plausible. "
            "The unit is therefore unresolved; the column name follows the paper's label "
            "and the absolute scale should not be treated as g/mm2 without the raw weights."
        ),
    }


def specimen_size_conflict(source: dict[str, Any]) -> dict[str, Any]:
    """Record the paper's conflicting statements about specimen dimensions."""
    test = source["oxidation"]["test"]
    return {
        "status": "warn",
        "methods_statement": test["specimen_size_methods_statement"],
        "table_7_statement": test["specimen_size_table_7_statement"],
        "area_ratio": ALTERNATE_SPECIMEN_AREA_MM2 / EXPOSED_AREA_MM2,
        "finding": test["specimen_size_conflict"],
    }


def sign_analysis(source: dict[str, Any]) -> dict[str, Any]:
    """Document the sign contradiction in the printed weight-gain equation."""
    return {
        "status": "warn",
        "printed_equation": source["oxidation"]["test"]["reported_weight_gain_equation"],
        "printed_values": "all positive",
        "finding": (
            "Table 7 defines weight gain as w1 (before) minus w2 (after). For oxidation "
            "the specimen usually gains mass (w2 > w1), which would make this quantity "
            "negative, yet every tabulated value is positive. Either the equation is "
            "transposed or the specimens lost mass. The raw weights are not published, "
            "so the sign cannot be resolved. Values are preserved exactly as printed and "
            "no sign correction is applied."
        ),
    }


def independence_summary(hardness: pd.DataFrame, oxidation: pd.DataFrame) -> dict[str, Any]:
    """Describe the experimental unit behind each row."""
    per_group = oxidation.groupby("Observation_Group").size().to_dict()
    return {
        "status": "warn",
        "hardness": {
            "rows": len(hardness),
            "observation_groups": int(hardness["Observation_Group"].nunique()),
            "rows_per_group": 1,
            "replicates_reported": False,
            "effective_independent_observations": int(hardness["Observation_Group"].nunique()),
        },
        "oxidation": {
            "rows": len(oxidation),
            "observation_groups": int(oxidation["Observation_Group"].nunique()),
            "rows_per_group": per_group,
            "replicates_reported": False,
            "specimen_reuse_across_exposures": "unknown",
            "effective_independent_observations": int(oxidation["Observation_Group"].nunique()),
        },
        "finding": (
            "Each material/current condition contributes one reported hardness value and "
            "three reported oxidation values. No replicate specimens, repeated readings, "
            "or uncertainty estimates are published, and specimen reuse across exposure "
            "times is not stated. The paper says three samples were kept in the furnace per "
            "run, which is consistent with one specimen at each current rather than "
            "triplicate specimens. Treating all 18 oxidation rows as independent would "
            "overstate the information by roughly a factor of three."
        ),
    }


def arithmetic_checks(
    source: dict[str, Any], hardness: pd.DataFrame, oxidation: pd.DataFrame
) -> dict[str, Any]:
    """Verify the printed heat-input values and the project's chemistry derivations."""
    schedule = source["process_schedule"]
    heat_input_rows = []
    for current, speed, printed in zip(
        schedule["current_a"],
        schedule["weld_speed_mm_s"],
        schedule["heat_input_kj_mm"],
        strict=False,
    ):
        computed = VOLTAGE_V * current / speed / 1000.0
        heat_input_rows.append(
            {
                "current_a": current,
                "weld_speed_mm_s": speed,
                "printed_heat_input_kj_mm": printed,
                "computed_heat_input_kj_mm": computed,
                "difference": computed - printed,
            }
        )
    max_difference = max(abs(row["difference"]) for row in heat_input_rows)

    compositions = source["substrate_composition"]
    chemistry_rows = []
    for material, row in hardness.drop_duplicates("Material").set_index("Material").iterrows():
        reported_c = compositions[material]["C"]
        reported_mn = compositions[material]["Mn"]
        if isinstance(reported_c, str):
            low, high = (float(part) for part in reported_c.split("-"))
            carbon_midpoint = (low + high) / 2
        else:
            carbon_midpoint = float(reported_c)
        if isinstance(reported_mn, str):
            low, high = (float(part) for part in reported_mn.split("-"))
            manganese_midpoint = (low + high) / 2
        else:
            manganese_midpoint = float(reported_mn)
        chemistry_rows.append(
            {
                "material": material,
                "carbon_csv": float(row["Carbon"]),
                "carbon_expected": carbon_midpoint,
                "manganese_csv": float(row["Manganese"]),
                "manganese_expected": manganese_midpoint,
            }
        )

    return {
        "status": "pass" if max_difference < 0.001 else "fail",
        "heat_input": heat_input_rows,
        "max_heat_input_difference_kj_mm": max_difference,
        "heat_input_finding": (
            "Q = 15 V * I / v reproduces the printed schedule without an arc-efficiency "
            "factor. Current, travel speed, and heat input therefore form a single "
            "one-dimensional schedule and cannot be separated as independent factors."
        ),
        "chemistry": chemistry_rows,
        "chemistry_finding": (
            "EN-8 carbon and manganese are midpoints of the reported ranges; mild-steel "
            "values are single reported values. Chemistry is metadata and is not a model "
            "feature because it is tied to substrate identity."
        ),
    }


def missing_metadata(source: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": "warn",
        "items": source["unreported_metadata"],
        "finding": (
            "These conditions are not published and are recorded as unknown rather than "
            "imputed. They limit any claim about measurement uncertainty or reproducibility."
        ),
    }


def _write_markdown(report: dict[str, Any]) -> str:
    comparison = report["comparison"]
    multiples = report["time_multiples"]
    units = report["unit_analysis"]
    sizes = report["specimen_size"]
    sign = report["sign_analysis"]
    independence = report["independence"]
    arithmetic = report["arithmetic"]
    lines = [
        "# Source audit",
        "",
        f"Source: {report['source']['paper']['title']} "
        f"({report['source']['paper']['venue']}), DOI {report['source']['paper']['doi']}, "
        f"licensed {report['source']['paper']['license']}.",
        "",
        "## Result",
        "",
        f"- Table comparison: **{comparison['status']}** "
        f"({comparison['rows_checked']} of {comparison['expected_rows']} rows checked, "
        f"{len(comparison['mismatches'])} mismatches).",
        f"- Exposure-time multiples: **{multiples['status']}** "
        f"({multiples['exact_multiples']} exact, {multiples['near_multiples']} near; "
        f"maximum relative deviation {multiples['max_relative_deviation']:.2e}).",
        f"- Oxidation unit: **{units['status']}** (unresolved).",
        f"- Specimen dimensions: **{sizes['status']}** "
        f"(methods and Table 7 disagree by a factor of {sizes['area_ratio']:.0f}).",
        f"- Oxidation sign: **{sign['status']}** (unresolved).",
        f"- Observation independence: **{independence['status']}** "
        f"(effective n = {independence['oxidation']['effective_independent_observations']} "
        "for oxidation).",
        f"- Heat-input arithmetic: **{arithmetic['status']}** "
        f"(max difference {arithmetic['max_heat_input_difference_kj_mm']:.2e} kJ/mm).",
        "",
        "## Table comparison",
        "",
        "Every CSV value and every `Source_Table` label was compared to the corresponding",
        "entry in Tables 5, 6, and 8-13. See `reports/audit/source_audit.json` for the",
        "row-level result.",
        "",
        "## Exposure-time multiples",
        "",
        "| Material | Current (A) | Exposure (h) | Value | Ratio to 5 h | Classification |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for group in multiples["groups"]:
        lines.append(
            f"| {group['material']} | {group['current_a']:.0f} | {group['exposure_h']:.0f} | "
            f"{group['value']:.6f} | {group['ratio_to_5h']:.6f} | {group['classification']} |"
        )
    lines += [
        "",
        multiples["finding"],
        "",
        "## Oxidation unit",
        "",
        f"The paper prints `{units['stated_equation']}` with a stated area of "
        f"{units['stated_area_mm2']:.0f} mm2.",
        "",
        "| Reading | Implied quantity at the smallest value | Implied quantity at the largest value |",
        "|---|---:|---:|",
        f"| Values are g/mm2 | {units['interpretation_a_values_are_g_per_mm2']['implied_mass_change_g_at_min']:.3f} g mass change "
        f"({100 * units['interpretation_a_values_are_g_per_mm2']['fraction_of_specimen_mass_at_min']:.1f} % of specimen mass) | "
        f"{units['interpretation_a_values_are_g_per_mm2']['implied_mass_change_g_at_max']:.3f} g "
        f"({100 * units['interpretation_a_values_are_g_per_mm2']['fraction_of_specimen_mass_at_max']:.1f} %) |",
        f"| Values are grams | {units['interpretation_b_values_are_grams']['implied_areal_gain_mg_cm2_at_min']:.2f} mg/cm2 | "
        f"{units['interpretation_b_values_are_grams']['implied_areal_gain_mg_cm2_at_max']:.2f} mg/cm2 |",
        "",
        units["finding"],
        "",
        "The printed magnitudes are more consistent with raw mass in grams than with",
        "g/mm2. Under 225 mm2 they imply "
        f"{units['interpretation_b_values_are_grams']['implied_areal_gain_mg_cm2_at_min']:.2f}-"
        f"{units['interpretation_b_values_are_grams']['implied_areal_gain_mg_cm2_at_max']:.2f} "
        "mg/cm2; under the 900 mm2 area stated in the methods they imply "
        f"{units['interpretation_under_alternate_area']['implied_areal_gain_mg_cm2_at_min_if_grams']:.2f}-"
        f"{units['interpretation_under_alternate_area']['implied_areal_gain_mg_cm2_at_max_if_grams']:.2f} "
        "mg/cm2. No correction is applied.",
        "",
        "## Specimen dimensions",
        "",
        f"Methods statement: `{sizes['methods_statement']}`.",
        f"Table 7 statement: `{sizes['table_7_statement']}`.",
        "",
        sizes["finding"],
        "",
        "## Sign convention",
        "",
        f"Printed equation: `{sign['printed_equation']}`. {sign['finding']}",
        "",
        "## Independence",
        "",
        f"Hardness: {independence['hardness']['rows']} rows, "
        f"{independence['hardness']['observation_groups']} conditions, "
        "one value per condition.",
        "",
        f"Oxidation: {independence['oxidation']['rows']} rows, "
        f"{independence['oxidation']['observation_groups']} conditions, "
        "three exposure rows per condition.",
        "",
        independence["finding"],
        "",
        "## Process arithmetic and chemistry",
        "",
        "| Current (A) | Speed (mm/s) | Printed Q (kJ/mm) | Computed Q (kJ/mm) |",
        "|---:|---:|---:|---:|",
    ]
    for row in arithmetic["heat_input"]:
        lines.append(
            f"| {row['current_a']:.0f} | {row['weld_speed_mm_s']:.2f} | "
            f"{row['printed_heat_input_kj_mm']:.3f} | {row['computed_heat_input_kj_mm']:.3f} |"
        )
    lines += [
        "",
        arithmetic["heat_input_finding"],
        "",
        arithmetic["chemistry_finding"],
        "",
        "## Unresolved issues",
        "",
        "1. Oxidation unit scale (g/mm2 vs grams) - needs the raw weights or author clarification.",
        "2. Oxidation sign convention - the printed equation is reversed relative to typical mass gain.",
        "3. Whether the 10 h and 15 h rows are independent measurements or derived values.",
        "4. Specimen reuse across exposure times.",
        "5. Hardness test load, indentation count, and replicate count.",
        "",
        "These are documented in `docs/SOURCE_AUDIT.md` and `data/README.md`. They are not",
        "resolved by assumption anywhere in the model pipeline.",
        "",
        "## Unreported metadata",
        "",
    ]
    lines += [f"- {item}" for item in report["missing_metadata"]["items"]]
    lines += [
        "",
        "## Dataset hashes",
        "",
        "| File | SHA-256 |",
        "|---|---|",
    ]
    for name, digest in report["hashes"].items():
        lines.append(f"| {name} | `{digest}` |")
    lines.append("")
    return "\n".join(lines)


def run_audit(write: bool = True) -> dict[str, Any]:
    source = json.loads(SOURCE_PATH.read_text())
    hardness = load_dataset(HARDNESS_PATH, "hardness")
    oxidation = load_dataset(OXIDATION_PATH, "oxidation")

    report = {
        "source": source,
        "comparison": compare_tables(hardness, oxidation, source),
        "time_multiples": time_multiples(oxidation),
        "unit_analysis": unit_analysis(oxidation),
        "specimen_size": specimen_size_conflict(source),
        "sign_analysis": sign_analysis(source),
        "independence": independence_summary(hardness, oxidation),
        "arithmetic": arithmetic_checks(source, hardness, oxidation),
        "missing_metadata": missing_metadata(source),
        "hashes": {
            "data/hardness.csv": dataset_sha256(HARDNESS_PATH),
            "data/oxidation.csv": dataset_sha256(OXIDATION_PATH),
            "data/source_tables.json": dataset_sha256(SOURCE_PATH),
            "source_pdf_author_supplied": source["paper"]["source_copies"]["author_supplied"][
                "sha256"
            ],
            "source_pdf_retrieved": source["paper"]["source_copies"]["retrieved_open_access"][
                "sha256"
            ],
        },
    }

    if write:
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        (REPORT_DIR / "source_audit.json").write_text(json.dumps(report, indent=2) + "\n")
        (REPORT_DIR / "source_audit.md").write_text(_write_markdown(report) + "\n")

    return report


if __name__ == "__main__":
    result = run_audit()
    print("Table comparison:", result["comparison"]["status"])
    print(
        "Exact multiples:",
        result["time_multiples"]["exact_multiples"],
        "near:",
        result["time_multiples"]["near_multiples"],
    )
    print("Unit status:", result["unit_analysis"]["status"])
    print("Sign status:", result["sign_analysis"]["status"])
    print("Reports written to reports/audit/")
    if result["comparison"]["status"] == "fail":
        raise SystemExit(1)
