# Architecture

## Components

1. data/: paper-transcribed measurements and row-level provenance.
2. src/models/: sklearn preprocessing, grouped evaluation, training, and artifact metadata.
3. models/: saved pipelines and metadata from the latest training run.
4. src/inference/: supported-condition validation, lazy artifact loading, and prediction wrappers.
5. src/app/: Flask routes and browser UI.

## Data and model flow

1. Training loads the six hardness or eighteen oxidation rows and rejects rows not marked as reported paper measurements.
2. Leave-One-Group-Out validation groups by welding current, holding both materials and all exposure times at that current out together.
3. After evaluation, the deployable pipeline is fit on all reported rows and saved with metadata.
4. The app validates substrate, current, and exposure time against conditions in the paper before prediction.
5. The API returns both outputs and defines oxidation as cumulative areal mass gain.

## Experimental limits encoded in the app

Current accepts only 120, 140, or 160 A. Oxidation exposure accepts only 5, 10, or 15 hours. The paper's current, travel speed, and heat input vary together, so the app does not accept heat input as an independent user control. Carbon and manganese are also not independent model inputs because the small dataset ties their values to substrate identity.

## Deployment

Render settings are in render.yaml; Procfile provides a Gunicorn entry point. The runtime is Python 3.10.13.

## Future work

Collect independent replicate measurements, publish raw measurement provenance and uncertainty, confirm the oxidation mass-change sign from lab records, validate on new specimens, and only then expand supported materials or process ranges.
