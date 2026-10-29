# External measurements

This directory holds new measurements that did **not** come from the source paper.
They are validated separately, scored against the shipped models, and never merged
into `data/hardness.csv` or `data/oxidation.csv` automatically. Training data stays
canonical and hashed.

The two files are header-only templates until real measurements exist:

| File | Purpose |
|---|---|
| `hardness_measurements.csv` | New Vickers hardness readings with specimen and run identifiers |
| `oxidation_measurements.csv` | New coupon weights with raw `w1_g`/`w2_g`, area, and furnace conditions |

## Why raw weights are required

The paper publishes only derived oxidation values, on a unit scale that the audit
could not resolve (g/mm² vs grams) and with a printed sign equation that appears
transposed. See [`../../docs/SOURCE_AUDIT.md`](../../docs/SOURCE_AUDIT.md) §4–§5.

New measurements must therefore record the raw before/after weights and the
measured exposed area. The validator computes:

```text
Mass_Change_g      = w2_g - w1_g
Areal_Gain_mg_cm2  = Mass_Change_g * 100,000 / Exposed_Area_mm2
Mass_Change_Sign   = gain | loss | zero
```

The physical areal gain is reported as-is. It is **not** compared to a prediction
unless the file also supplies `Paper_Unit_Value`, which is the weight change on the
same printed scale as Tables 8–13. Without that column the models still produce a
prediction, but the row is reported as `prediction_only`: no residual is computed
across an unresolved unit.

## Columns

Both templates share provenance columns:

| Column | Required | Meaning |
|---|---|---|
| `Row_ID` | yes | Unique identifier in this file |
| `Material` | yes | `EN-8` or `Mild Steel` |
| `Current` | yes | Amperes |
| `Voltage_V` | yes | Arc voltage |
| `Travel_Speed_mm_s` | yes | Travel speed |
| `Heat_Input` | yes | kJ/mm; metadata, not a model feature |
| `Specimen_ID` | yes | Coupon identifier, unique per specimen |
| `Heat_Lot` | yes | Substrate heat or batch |
| `Replicate_ID` | yes | Replicate label within a specimen/condition |
| `Deposition_Run` | yes | Hardfacing run identifier |
| `Operator` | yes | Person who ran the measurement |
| `Measurement_Date` | yes | ISO date |
| `Source` | yes | Lab, project, or report that produced the row |
| `Data_Status` | yes | `new_measurement` or `external_validation` |

Hardness adds `Test_Load_kgf`, `Indent_Count`, `Surface_Prep`, and `Hardness`.

Oxidation adds `Exposure_Time_h`, `Exposure_Temperature_K`,
`Furnace_Atmosphere`, `Heating_Rate_K_min`, `Cooling_Mode`, `Balance_ID`,
`Balance_Resolution_mg`, `Exposed_Area_mm2`, `w1_g`, `w2_g`, and the optional
`Paper_Unit_Value`.

Validation rejects negative or non-finite numerics, unknown materials, paper rows
(`Data_Status = reported_paper_measurement`), and duplicate measurement keys:

- hardness: `(Material, Current, Specimen_ID, Replicate_ID)`;
- oxidation: `(Material, Current, Exposure_Time_h, Specimen_ID, Replicate_ID)`.

An empty header-only file is valid, so the template can be committed before any
data exists.

## Scoring new measurements

```bash
python -m src.models.external_validation --write
# or point at other files
python -m src.models.external_validation \
  --hardness path/to/hardness.csv \
  --oxidation path/to/oxidation.csv \
  --write
```

Outputs:

- `reports/external_validation/summary.json` — per-row predictions, residuals,
  intervals, applicability status, and by-condition summaries;
- `reports/external_validation/summary.md` — human-readable summary.

The command exits 0 with an "empty template" note when no rows are present, so it
is safe to run in CI.

## Interpreting results

- These metrics describe the supplied specimens only. They are not a new
  population estimate and do not replace the grouped benchmark.
- Predictions are limited to the tested conditions (EN-8 or Mild Steel, 120–160 A,
  5–15 h). Rows outside those conditions are reported as unsupported.
- Oxidation residuals require `Paper_Unit_Value`; physical areal gains in mg/cm²
  are not on the same scale as the paper-printed target.
- Do not add external rows to training without a deliberate change to the schema,
  audit, dataset hashes, reproducibility baseline, benchmark, and model card.
