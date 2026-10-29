# Dataset provenance and units

The checked-in training rows are transcribed from the experimental tables in:

> C. Dinesh Chandra, B. Rushikesh, Mohammed Numan, and B. Venkatesh,
> “Experimental Studies of Stellite-6 Hardfaced Layer on Ferrous Materials by TIG
> Surfacing Process,” *IOP Conference Series: Materials Science and Engineering*
> 998 (2020), 012061.
> DOI: [10.1088/1757-899X/998/1/012061](https://doi.org/10.1088/1757-899X/998/1/012061).
> Licensed CC BY 3.0.

The transcribed data are distributed under CC BY 3.0; see [`LICENSE`](LICENSE).
The project code is MIT-licensed.

## Files

| File | Rows | Source tables | Content |
|---|---:|---|---|
| `hardness.csv` | 6 | Tables 5 and 6 | One Vickers hardness number per Material × Current condition |
| `oxidation.csv` | 18 | Tables 8–13 | Cumulative areal mass gain per Material × Current × Exposure |
| `source_tables.json` | — | Tables 1–13 as transcribed | Source values, equations, units, and PDF digests used by the audit |

The audit compares every CSV value and source-table label with
`source_tables.json`. See [`../docs/SOURCE_AUDIT.md`](../docs/SOURCE_AUDIT.md) and
[`../reports/audit/`](../reports/audit/).

## Column meanings

Both files share these provenance columns:

| Column | Meaning |
|---|---|
| `Row_ID` | Project-assigned stable identifier |
| `Material` | `EN-8` or `Mild Steel` |
| `Current` | Welding current in amperes; 120, 140, or 160 |
| `Heat_Input` | Printed schedule value in kJ/mm; metadata, not a model feature |
| `Carbon`, `Manganese` | Representative substrate chemistry in wt %; metadata only |
| `Source` | Citation and DOI |
| `Source_Table` | Table number in the paper |
| `Data_Status` | `reported_paper_measurement` for every training row |
| `Observation_Group` | `Material@Current`, used for grouping |
| `Specimen_ID` | Project-assigned; the paper does not publish specimen IDs |
| `Derivation_Status` | Whether the row is a table value, an exact time multiple, or a near multiple |
| `Replicate_Status` | `single_reported_value_no_replicates` |
| `Chemistry_Status` | `midpoint_of_reported_range` or `reported_single_value` |

Target columns:

| File | Column | Unit and status |
|---|---|---|
| `hardness.csv` | `Hardness` | Vickers hardness number; `Hardness_Unit` records that the test load is not reported |
| `hardness.csv` | `Hardness_Unit` | Free-text unit/provenance note |
| `oxidation.csv` | `Exposure_Time_h` | Hours at 973 K (700 °C); 5, 10, or 15 |
| `oxidation.csv` | `Areal_Mass_Gain_g_mm2` | Printed value; `Reported_Unit` is `gm/mm2 as printed` |
| `oxidation.csv` | `Reported_Unit`, `Unit_Status` | Record the printed unit and `unresolved_inconsistent_with_stated_area` |

## Experimental scope

- Substrates: EN-8 and mild steel.
- Currents: 120, 140, and 160 A.
- Printed heat inputs: 0.800, 0.864, and 0.912 kJ/mm.
- Oxidation: 5, 10, and 15 h in air at 973 K; the paper states an exposed area
  of 225 mm².

Current, travel speed, and heat input vary together in the paper's schedule:
`Q = 15 V · I / v` reproduces the printed heat inputs within rounding. They are
not independent experimental factors, and no model in this project treats them
as such. See [`../docs/SOURCE_AUDIT.md`](../docs/SOURCE_AUDIT.md) §2.

## Unresolved issues

These are preserved, not corrected:

1. **Exposure-time multiples.** Eleven of the twelve non-base oxidation rows are
   exact multiples of the 5 h value to the last printed digit; the twelfth
   deviates by 0.08 %. The 18 rows therefore contain six effective
   Material × Current conditions, not eighteen independent kinetic observations.
2. **Area and unit.** The methods state 30 × 30 mm² specimens while Table 7 states
   15 × 15 × 6 mm with a 225 mm² exposed face, a fourfold disagreement. If the
   printed numbers are read literally as g/mm² with a 225 mm² area, the implied
   specimen mass change is 0.487–1.817 g (4.6–17.1 % of a ~10.6 g specimen),
   which is implausible for 5–15 h at 700 °C. If they are raw grams, the areal
   gain is 0.96–3.59 mg/cm² at 225 mm² or 0.24–0.90 mg/cm² at 900 mm², both
   physically plausible. The raw weights and the measured area are not published,
   so the scale is unresolved.
3. **Sign.** Table 7 prints `weight gain = w1 (before) − w2 (after)`. For an
   oxidation test that would normally be negative, yet every value is positive.
   Either the equation is transposed or the specimens lost mass. No sign
   correction is applied.

The API, UI, model metadata, and model card all surface these findings. They are
not resolved by assumption anywhere in the pipeline.

## Independence and replication

The paper reports no replicate specimens, repeated indentations, repeated
weighings, or uncertainty estimates, and does not state whether coupons were
reused across exposure times. Every row is a single reported table value. This is
why every manifest records `effective_independent_observations = 6`.

## Adding data

- Do not add interpolated, smoothed, or derived rows as training measurements.
- New measurements belong in `data/external/`, which is validated separately and is
  never merged into the training files automatically. See
  [`external/README.md`](external/README.md).
- External rows must use a `Data_Status` of `new_measurement` or
  `external_validation`; the training loader intentionally rejects anything other
  than `reported_paper_measurement`.
- Keep raw instrument readings (indentations, w1/w2, balance resolution,
  measured area) and complete process metadata. See
  [`../docs/EXPERIMENTAL_PLAN.md`](../docs/EXPERIMENTAL_PLAN.md).
- Promoting external rows to training requires a deliberate change to the audit,
  dataset hash, reproducibility baseline, benchmark, and model card in one commit.

## Verification

```bash
python -m src.data.audit
python -m src.models.reproducibility
pytest tests/test_audit.py tests/test_schema.py
```

The dataset hashes recorded by the current baseline are:

| File | SHA-256 |
|---|---|
| `data/hardness.csv` | `1d83be25aa4b59453bc24b816c7f6267be0e0b7480bc8fc2a394fba9d6745a60` |
| `data/oxidation.csv` | `be5edafcebde4ada00ba3578df623b84953f79ed030f109f20cfdf9ade6f820b` |
| `data/source_tables.json` | `a444175f2ed578396b6a1d1a25de6cfd6553e4b45fcb799a9e0d1aeda8953bb0` |
