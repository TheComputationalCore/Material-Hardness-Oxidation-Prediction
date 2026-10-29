# API reference

Base URL for local use: <http://localhost:5001>

The service accepts only the conditions reported in the source paper. Inputs
outside those levels are rejected; the API does not extrapolate.

| Input | Allowed values |
|---|---|
| `Material` | `EN-8`, `Mild Steel` |
| `Current` | `120`, `140`, `160` (amperes) |
| `Exposure_Time_h` | `5`, `10`, `15` (hours) |

## `POST /api/v1/predict`

Request body: JSON object with all three fields. Numeric fields may be numbers or
numeric strings; `Current` and `Exposure_Time_h` must match an allowed level
exactly.

```bash
curl -s http://localhost:5001/api/v1/predict \
  -H 'Content-Type: application/json' \
  -d '{"Material":"EN-8","Current":140,"Exposure_Time_h":10}'
```

### Successful response (HTTP 200)

Values below are from the shipped artifacts (hardness 0.2.0, oxidation 0.3.0):

```json
{
  "ok": true,
  "hardness": 355.6666666666667,
  "hardness_error": null,
  "hardness_interval_90": {
    "coverage": 0.9,
    "lower": 338.1666666666667,
    "upper": 392.1666666666667,
    "method": "empirical_out_of_fold_residual_quantiles",
    "n_oof_predictions": 6,
    "effective_independent_observations": 6,
    "note": "Approximate interval from leave-one-current-out residuals; not a guaranteed coverage interval."
  },
  "hardness_applicability": {
    "status": "reported_condition",
    "explanation": "This exact condition is reported in the source paper; the model estimate is compared with the reported value.",
    "tested_materials": ["EN-8", "Mild Steel"],
    "tested_currents_a": [120.0, 140.0, 160.0],
    "tested_exposures_h": null,
    "measured_value": 365.0,
    "measured_source_table": "Table 6",
    "measured_row_id": "H-EN8-140"
  },
  "oxidation_areal_mass_gain_g_mm2": 0.004708,
  "oxidation_error": null,
  "oxidation_interval_90": {
    "coverage": 0.9,
    "lower": 0.004361499999999998,
    "upper": 0.0049689999999999995,
    "method": "empirical_out_of_fold_residual_quantiles",
    "n_oof_predictions": 18,
    "effective_independent_observations": 6,
    "note": "Approximate interval from leave-one-current-out residuals; not a guaranteed coverage interval."
  },
  "oxidation_applicability": {
    "status": "reported_condition",
    "explanation": "This exact condition is reported in the source paper; the model estimate is compared with the reported value.",
    "tested_materials": ["EN-8", "Mild Steel"],
    "tested_currents_a": [120.0, 140.0, 160.0],
    "tested_exposures_h": [5.0, 10.0, 15.0],
    "measured_value": 0.004592,
    "measured_source_table": "Table 11",
    "measured_row_id": "O-EN8-140-10"
  },
  "oxidation_target_definition": "Cumulative mass gain per exposed area; g/mm^2 as printed; not time-normalized.",
  "model_versions": {
    "hardness": "0.2.0",
    "oxidation": "0.3.0"
  },
  "warnings": [
    "Supported only at the paper-tested conditions; no replicate specimens or repeated measurements are published.",
    "The Vickers hardness test load is not reported, so the hardness scale is identified only by the printed table values.",
    "The printed g/mm^2 unit implies a specimen mass change that is not physically plausible; the intended scale is unresolved and values are shown as printed.",
    "The printed sign convention implies mass loss while every tabulated value is positive; no sign correction is applied.",
    "Exposure-time rows are exact multiples of the 5 h value, so they are not independent kinetic measurements."
  ]
}
```

### Fields

| Field | Type | Meaning |
|---|---|---|
| `ok` | boolean | True only when both predictions succeeded |
| `hardness` | number or null | Model estimate of the reported Vickers hardness number |
| `hardness_interval_90` | object or null | Approximate 90 % interval from LOCO residuals |
| `hardness_applicability` | object or null | Tested levels, measured value, and source table for the hardness condition |
| `oxidation_areal_mass_gain_g_mm2` | number or null | Model estimate, in the printed unit, of cumulative areal mass gain |
| `oxidation_interval_90` | object or null | Approximate 90 % interval for the oxidation estimate |
| `oxidation_applicability` | object or null | Tested levels, measured value, and source table for the oxidation condition |
| `oxidation_target_definition` | string | Restates the target definition and unit caveat |
| `model_versions` | object | Manifest versions used for this response |
| `warnings` | array of strings | Duplicate-free warnings from both prediction paths |

`measured_value` is the number printed in the paper for that exact condition. It
is returned for comparison only; it is not used to produce the estimate.

### Applicability statuses

| Status | Meaning |
|---|---|
| `reported_condition` | The exact Material × Current (× Exposure) cell is in the paper; `measured_value` is present |
| `unmeasured_combination_within_tested_levels` | Every factor level was tested, but not this combination |
| `outside_tested_conditions` | At least one factor lies outside the reported range |

The public API rejects inputs that would produce the last two statuses before
prediction. The statuses exist for the shared inference layer and its tests, and
would describe extrapolation if the validation policy were ever loosened.

## Errors

### HTTP 400 — validation failure

Returned when the body is not a JSON object or any field is missing, blank,
non-numeric, non-finite, or outside the allowed levels.

```json
{
  "ok": false,
  "hardness": null,
  "hardness_error": "Current must be 120, 140, or 160 A (paper-tested conditions).",
  "hardness_interval_90": null,
  "hardness_applicability": null,
  "oxidation_areal_mass_gain_g_mm2": null,
  "oxidation_error": "Current must be 120, 140, or 160 A (paper-tested conditions).",
  "oxidation_interval_90": null,
  "oxidation_applicability": null,
  "oxidation_target_definition": "Cumulative mass gain per exposed area; g/mm^2 as printed; not time-normalized.",
  "model_versions": {"hardness": null, "oxidation": null},
  "warnings": []
}
```

A non-object body returns `{"error": "Request body must be a JSON object."}`.

### HTTP 503 — model unavailable

Returned when a `.joblib` artifact is missing or cannot be loaded. The error text
contains `unavailable` or `does not exist`; check `/healthz` and the server log.
Unexpected prediction failures are reported as a generic internal error; the log
contains the traceback.

## `GET /api/v1/model-info`

Returns the model-card data used by the UI: manifest versions, features, target
definitions, metrics, validation descriptions, dataset hashes, artifact hashes,
known limitations, calibration statistics, source citation, and audit findings.

Use it to confirm which artifacts are deployed before interpreting predictions.

## `GET /healthz`

Returns HTTP 200 when both artifacts load, otherwise HTTP 503:

```json
{
  "status": "ok",
  "models": {
    "hardness": {"available": true, "error": null},
    "oxidation": {"available": true, "error": null}
  }
}
```

Render uses this endpoint as its health check.

## Browser routes

| Route | Method | Purpose |
|---|---|---|
| `/` | GET | Form and current service information |
| `/predict` | POST | Standard form submission; renders results server-side |
| `/api/v1/predict` | POST | JSON prediction |
| `/api/v1/model-info` | GET | Model card data |
| `/healthz` | GET | Liveness and artifact availability |

The form submits asynchronously when JavaScript is available and falls back to a
normal POST without it.

## Interpretation limits

- Estimates are model outputs, not measurements or guarantees.
- The hardness model has negative LOCO R²; withholding a current produces worse
  predictions than the mean.
- Oxidation metrics rest on six effective conditions, not eighteen independent
  samples.
- Intervals are empirical quantiles from a handful of out-of-fold residuals.
- The oxidation unit and printed sign are unresolved; values are shown as printed.
- Do not use these outputs for safety, qualification, or production control.
