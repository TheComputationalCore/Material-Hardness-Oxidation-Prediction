# Stellite-6 Hardfacing Property Predictor

An exploratory Flask application and scikit-learn models based on the experimental results in:

> C. Dinesh Chandra, B. Rushikesh, Mohammed Numan, and B. Venkatesh, “Experimental Studies of Stellite-6 Hardfaced Layer on Ferrous Materials by TIG Surfacing Process,” IOP Conference Series: Materials Science and Engineering 998 (2020), 012061. [DOI: 10.1088/1757-899X/998/1/012061](https://doi.org/10.1088/1757-899X/998/1/012061).

The paper studies Stellite-6 TIG hardfacing on EN-8 and mild steel. This application models the paper's reported hardness and cumulative oxidation mass gain per exposed area. Its purpose is to demonstrate a small, traceable ML workflow; the available measurements are not enough to establish broad predictive generalization.

## Experimental scope and data

Training data contain only the values reported in the paper's tables:

- Hardness: six values across two substrates and three welding currents (Tables 5–6).
- Oxidation: eighteen values across two substrates, three currents, and 5, 10, and 15 hour exposures (Tables 8–13).
- Current settings are 120, 140, and 160 A. Their corresponding paper conditions include heat inputs of 0.800, 0.864, and 0.912 kJ/mm and changing travel speeds. These variables are coupled in the experiment.
- Oxidation is tested at 973 K (700 °C). The target is cumulative mass gain divided by exposed area (g/mm²), not a rate divided by time.

See data/README.md for row provenance, target definitions, and the paper's sign-convention note. Interpolated values are not included as training measurements.

## Models and validation

| Prediction | Features | Estimator |
|---|---|---|
| Hardness number | Substrate, welding current | Linear regression |
| Cumulative areal mass gain | Substrate, welding current, exposure hours | Random forest regression |

Categorical substrate is one-hot encoded and numeric inputs are standardized. Validation uses Leave-One-Group-Out with current as the group, so both substrates and all oxidation times at one current are held out together. The saved deployment model is then refit on all reported paper rows. There are only three current groups; two folds require predicting beyond the remaining current range.

Grouped validation gives hardness MAE 26.25, RMSE 27.51, and R² -0.656; oxidation MAE 0.000440 g/mm², RMSE 0.000512 g/mm², and R² 0.937. The negative hardness R² indicates that the current linear model does not generalize well when a whole current level is withheld. These are limited diagnostics, not evidence of performance on new specimens.

The application accepts only the materials, currents, and exposure times represented in the paper. This does not make a prediction experimentally validated; use the output for exploration and verify it against new measurements before operational use.

## Run locally

    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    python -m src.app.app

Open http://localhost:5001.

To reproduce grouped validation and refresh the saved models:

    python -m src.models.train_hardness
    python -m src.models.train_oxidation

To calculate grouped validation metrics without saving models:

    python -m src.models.evaluate

## API

POST /api/v1/predict accepts a JSON object:

    {
      "Material": "EN-8",
      "Current": 140,
      "Exposure_Time_h": 10
    }

The response contains hardness, oxidation_areal_mass_gain_g_mm2, and validation errors. Oxidation is cumulative mass gain per exposed area in g/mm². Invalid input returns HTTP 400. See docs/API_REFERENCE.md for details.

## Project layout

- data/: paper-transcribed training rows and provenance
- docs/: model card, architecture, and API reference
- models/: fitted sklearn pipelines and metadata
- src/models/: preprocessing, training, grouped evaluation
- src/inference/: model loading and condition validation
- src/app/: Flask routes and interface
- tests/: lightweight application and inference checks

## Deployment

The repository includes Render configuration in render.yaml and a Gunicorn Procfile. Install dependencies from requirements.txt and run the factory with Gunicorn.

## License

MIT. See LICENSE.
