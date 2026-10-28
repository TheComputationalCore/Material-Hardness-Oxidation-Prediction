# Model Card — Stellite-6 Hardfacing Property Predictor

## Model purpose

These models demonstrate regression over the published measurements in Chandra et al. (2020), “Experimental Studies of Stellite-6 Hardfaced Layer on Ferrous Materials by TIG Surfacing Process,” DOI 10.1088/1757-899X/998/1/012061.

The app predicts a reported Vickers hardness number and cumulative oxidation mass gain per exposed area for the tested Stellite-6 hardfaced EN-8 and mild-steel substrates.

## Data and scope

- Hardness: six values transcribed from Tables 5 and 6.
- Oxidation: eighteen values transcribed from Tables 8–13.
- Supported substrates: EN-8 and Mild Steel.
- Supported currents: 120, 140, and 160 A.
- Supported exposure times for oxidation: 5, 10, and 15 hours at 973 K.
- Heat input, travel speed, and current are coupled in the paper's process schedule. Current is used as the model feature; heat input and chemistry are retained as data provenance, not as independent features.
- The paper gives no replicate-level data or uncertainty estimates. The hardness table does not state the test load.
- The oxidation target is reported in g/mm² and is not divided by time. The paper's printed mass-change subtraction order appears reversed; see data/README.md.

## Models

- Hardness: one-hot encoded substrate and standardized current followed by linear regression.
- Oxidation: one-hot encoded substrate and standardized current and exposure hours followed by a 300-tree random forest.

The saved pipelines are fit on all reported rows after grouped cross-validation. Metadata records the validation protocol and sample counts.

## Evaluation

Leave-One-Group-Out validation holds out all records at one current setting per fold. This prevents measurements at the same current from appearing in both training and validation for a fold. Two of the three folds require predicting outside the current range represented in that fold's training data.

Latest grouped-validation metrics:

| Target | MAE | RMSE | R² |
|---|---:|---:|---:|
| Hardness number | 26.25 | 27.51 | -0.656 |
| Areal mass gain (g/mm²) | 0.000440 | 0.000512 | 0.937 |

The negative hardness R² indicates poor current model generalization under this grouped protocol. All metrics are uncertain because validation uses only three current groups and no independent specimens. They are diagnostics, not evidence of performance on new materials or processes.

## Intended use

- Demonstrations and exploratory discussion of the reported experiment.
- Predictions only for the two reported substrates and the paper-tested current and exposure settings.

## Not intended use

- Safety-critical or production process control.
- Claims of causal effects for current independent of travel speed or heat input.
- Prediction for other alloys, deposition procedures, temperatures, currents, exposure times, or specimen preparation.
- Treating an estimate as a substitute for hardness or oxidation testing.

## Limitations and risks

The datasets are small and represent a single published study. No independent validation specimens are available. Inputs such as chemistry are not varied independently of material in this dataset. The paper reports no replicate variation, and the oxidation formula sign convention needs confirmation against source lab records. Predictions should be confirmed experimentally.

## Maintenance

Keep reported measurements, derived rows, and newly measured observations distinguishable using the Data_Status and source fields. Add independent experiments before making broader performance claims. Re-run grouped validation and retrain after changes to the dataset or pipeline.
