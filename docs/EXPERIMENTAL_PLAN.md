# Experimental plan: from six conditions to defensible evidence

This plan describes the experiments needed before the models can claim predictive
skill beyond the reported table. It is deliberately separate from model tuning:
no amount of resampling can create independent specimens, decouple the process
schedule, or resolve the oxidation unit.

The current evidence base is:

| Target | Reported rows | Material × Current conditions | Replicates | Effective independent n |
|---|---:|---:|---|---:|
| Hardness | 6 | 6 | none reported | 6 |
| Oxidation | 18 | 6 | none reported | 6 |

The oxidation rows at 10 h and 15 h are exact multiples of the 5 h row in eleven
of twelve cases (the twelfth deviates by 0.08 %). The welding current, travel
speed, and heat input form one schedule, so their effects cannot be separated.
See [`SOURCE_AUDIT.md`](SOURCE_AUDIT.md) for the arithmetic and unit findings.

## 1. Questions the next experiments must answer

1. **Measurement reproducibility.** How much do hardness and mass gain vary
   between nominally identical specimens and between repeated readings on one
   specimen?
2. **Process-factor separation.** After controlling current and travel speed
   independently, which factor changes hardness or oxidation, and by how much?
3. **Time response.** Are the 10 h and 15 h oxidation values measured, or derived
   from a 5 h rate? Independent coupons at each exposure test this directly.
4. **Unit and sign.** What is the actual mass change in grams, and is the
   specimen gaining or losing mass? Raw weights answer both.
5. **Transfer.** Does a model fitted to one heat, operator, or furnace run predict
   an independent batch?

## 2. Phase 0 — resolve the source record before new deposition

These tasks do not require new specimens and should be completed first.

| Task | Output | Owner |
|---|---|---|
| Obtain raw weights w1 and w2, or clarify the printed subtraction and unit | Signed mass change in grams; confirmed areal unit | Original authors / lab records |
| Confirm whether 10 h and 15 h rows are measured or calculated | Measurement provenance for each row | Original authors |
| Obtain Vickers load, indentation count, and location | Hardness test metadata | Original authors |
| Obtain furnace atmosphere, heating/cooling rates, and balance resolution | Oxidation test metadata | Original authors |

If the original records cannot be recovered, the dataset remains a transcription
of printed values. That is acceptable for a reproduction, but it is not a
measurement archive.

## 3. Phase 1 — replication with independent specimens

**Purpose:** estimate repeatability under the original schedule before changing
any factor.

### 3.1 Specimens

- Two substrate heats per material (EN-8 and mild steel), so heat-to-heat
  variation is visible. Do not reuse one bar for every condition.
- Hardness: 3 specimens per Material × Current cell, 18 specimens total.
- Oxidation: separate coupons for every exposure. Do not reuse a coupon across
  5, 10, and 15 h. Use 3 replicate coupons per Material × Current × Exposure
  cell, 54 coupons total, plus 6 control coupons.
- Record specimen ID, heat/lot, dimensions, mass, surface preparation, and
  deposition run.

### 3.2 Process schedule

Reproduce the original schedule (120/140/160 A; 2.25/2.43/2.63 mm/s) with the
same filler and procedure. Randomize deposition order within blocks (day or
furnace run) so that drift does not align with current.

### 3.3 Measurements

**Hardness**

- State the load (for example HV1 or HV0.5) and dwell time.
- Polish and etch using a documented procedure.
- Take at least five indentations per specimen: a traverse across the deposit,
  away from the fusion boundary, with locations recorded.
- Report every indentation, not only the mean. Report mean, standard deviation,
  and the number of valid indents.
- Calibrate the tester against a certified reference block on each measurement
  day.

**Oxidation**

- Measure coupon dimensions with a micrometer; calculate exposed area from the
  measured dimensions, not nominal size.
- Weigh on a balance with recorded resolution (target 0.1 mg or better),
  including balance ID and calibration check.
- Record w1 before exposure and w2 after exposure, both in grams. Report
  `w2 − w1` explicitly and state the sign convention.
- Record furnace ID, atmosphere, heating rate, hold temperature, hold time,
  cooling rate, and whether the coupon was cooled in the furnace or air.
- Photograph coupons and note spallation or oxide loss.
- Report areal change in **mg/cm²**, with the g/mm² conversion shown, so the unit
  is unambiguous.

## 4. Phase 2 — decoupled process design

**Purpose:** separate current from travel speed and heat input.

Use a full factorial design per substrate:

| Factor | Levels | Notes |
|---|---|---|
| Current | 120, 140, 160 A | Same inverter and settings as Phase 1 |
| Travel speed | 1.8, 2.25, 2.7 mm/s | Independent of current; measured with a linear stage or encoder |
| Substrate | EN-8, mild steel | Blocked separately |

That is 3 × 3 × 2 = 18 deposition conditions. Heat input is then a derived
response, `Q = 15 V · I / v`, not a controlled factor. At the nine current-speed
combinations, Q spans roughly 0.67–1.33 kJ/mm, which brackets the original
0.80–0.91 kJ/mm schedule.

- Repeat the center condition (140 A, 2.25 mm/s) three times per substrate.
- Randomize run order within each substrate block.
- Record actual current, voltage, and speed traces; do not rely on setpoints.
- If the equipment cannot hold 1.8 or 2.7 mm/s, choose the widest feasible range
  and record the achieved values.

### 4.1 Analysis

- Fit a mixed-effects model with fixed effects for material, current, speed, and
  their interactions, and random intercepts for specimen and deposition run.
- Report effect sizes with confidence intervals, not only p-values.
- Treat hardness and oxidation as separate responses.
- Use the model to decide whether a simpler schedule index is adequate; do not
  assume that current alone is the causal factor.

## 5. Phase 3 — held-out external validation

**Purpose:** test transfer, not interpolation.

- Freeze the model, preprocessing, and decision thresholds before opening the
  external set.
- Collect a new batch from a different heat/lot, a different operator, and a
  different day. Where possible, use a different furnace or welding machine.
- Include conditions inside the tested grid and at least one modest extrapolation
  (for example 130 A or 150 A) that is labeled as extrapolation.
- Do not use external results to select features or tune hyperparameters. Report
  all external predictions, including failures.
- Report MAE, RMSE, R², and calibration of the 90 % intervals on the external
  set. With a small external set, report per-condition errors as well as pooled
  metrics.

Suggested minimum: 12 external hardness specimens and 12 external oxidation
coupons spanning at least four conditions.

The repository already provides the ingestion path: copy
[`../data/external/hardness_measurements.csv`](../data/external/hardness_measurements.csv)
and
[`../data/external/oxidation_measurements.csv`](../data/external/oxidation_measurements.csv),
or write new files with the same columns, then run

```bash
python -m src.models.external_validation --write
```

External rows are validated, scored, and reported; they are never merged into the
training CSVs automatically. Oxidation rows must include raw `w1_g`/`w2_g` and the
measured area. Because the paper's unit is unresolved, an oxidation residual is
computed only when `Paper_Unit_Value` is supplied; the physical areal gain in
mg/cm² is reported separately. See
[`../data/external/README.md`](../data/external/README.md).

## 6. Phase 4 — active learning (only after Phases 1–3)

Active learning is useful only when a model exposes uncertainty and when new
experiments are cheap relative to the information they add. The current model
does not meet that bar.

When it does:

1. Fit an ensemble or Gaussian-process model that provides predictive variance.
2. Define a candidate pool over feasible current, speed, and exposure settings.
3. Select a batch of 6–12 conditions by a stated acquisition rule (for example
   maximum predictive variance, or expected improvement in MAE).
4. Require at least two replicates per selected condition so that measurement
   noise is not mistaken for model uncertainty.
5. Retrain and re-evaluate with the same grouped protocol. Keep the previous
   baseline and report the change.
6. Stop when external MAE and interval coverage stop improving, or when the
   budget is exhausted.

Do not use active learning to select points that only improve interpolation
inside an already sampled cell.

## 7. Metadata to record for every new specimen

| Field | Example | Why |
|---|---|---|
| `Specimen_ID` | `EN8-H2-140-02` | Traceability |
| `Heat_Lot` | `H-2291` | Heat-to-heat variation |
| `Replicate_ID` | `r2` | Replicate structure |
| `Deposition_Run` | `D-2027-03-04-07` | Blocking and drift |
| `Operator` | `OP-B` | Operator effect |
| `Date` | `2027-03-04` | Time drift |
| `Test_Load_kgf` | `1.0` | Hardness scale |
| `Indent_Count` | `5` | Sampling |
| `Indent_Locations` | `traverse_1..5` | Spatial variation |
| `Surface_Prep` | `ground_1200_then_polished` | Measurement comparability |
| `Balance_ID` | `BAL-02` | Instrument traceability |
| `Balance_Resolution_mg` | `0.1` | Measurement uncertainty |
| `Exposure_Temperature_K` | `973` | Test condition |
| `Furnace_Atmosphere` | `air` | Oxidation mechanism |
| `Heating_Rate_K_min` | `10` | Thermal history |
| `Cooling_Mode` | `furnace_cool` | Oxide spallation |
| `Exposed_Area_mm2` | `225.4` | Areal normalization |
| `w1_g`, `w2_g` | `10.6241`, `10.6272` | Sign and unit resolution |
| `Data_Status` | `new_measurement` | Distinguish from paper values |

All new measurements should enter through `data/external/`, whose files carry a
`Data_Status` that is not `reported_paper_measurement`. The training loader rejects
any other status by design, so external rows cannot leak into training without a
deliberate schema, audit, baseline, and model-card change.

## 8. Decision gates

| Gate | Requirement before proceeding |
|---|---|
| G0 | Unit, sign, and measurement provenance resolved or explicitly bounded |
| G1 | Replicate standard deviations reported for hardness and mass gain |
| G2 | Phase 2 design executed with current and speed varied independently |
| G3 | External set frozen and predicted once |
| G4 | External MAE and interval coverage reported; model card updated |
| G5 | Only then consider active learning or a broader alloy/process scope |

## 9. What will not change

- The service will continue to label predictions as estimates, not measurements.
- Unresolved issues will remain visible in the API, UI, and documentation.
- Reported, derived, and new measurements will remain distinguishable by
  `Data_Status` and source fields.
- A more complex model will not replace the current one without the benchmark
  evidence described in [`MODEL_CARD.md`](MODEL_CARD.md).

## 10. Related documents

- [`SOURCE_AUDIT.md`](SOURCE_AUDIT.md) — what the paper reports and what is missing.
- [`MODEL_CARD.md`](MODEL_CARD.md) — current models, metrics, and acceptance criteria.
- [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) — how to reproduce and update results.
- [`../data/README.md`](../data/README.md) — dataset columns and provenance.
- [`../data/external/README.md`](../data/external/README.md) — new-measurement templates and scoring.
