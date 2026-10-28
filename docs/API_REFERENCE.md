# API Reference

Base URL for local use: http://localhost:5001

## POST /api/v1/predict

Accepts a JSON object with conditions represented in the paper:

    {
      "Material": "EN-8",
      "Current": 140,
      "Exposure_Time_h": 10
    }

Material must be EN-8 or Mild Steel. Current must be 120, 140, or 160 A. Exposure_Time_h must be 5, 10, or 15 hours.

Example successful response:

    {
      "hardness": 365.0,
      "hardness_error": null,
      "oxidation_areal_mass_gain_g_mm2": 0.004592,
      "oxidation_error": null,
      "oxidation_target_definition": "cumulative mass gain per exposed area; g/mm^2; not time-normalized"
    }

HTTP 400 is returned when the body is not a JSON object or a prediction input is invalid. Prediction values are estimates from models trained on the paper's small table values; they are not raw lookup values.

## Browser

- GET / renders the form.
- The form submits asynchronously to POST /api/v1/predict.
- POST /predict remains available for standard form submissions.
