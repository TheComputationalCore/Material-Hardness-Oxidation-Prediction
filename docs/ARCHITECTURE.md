# Architecture — Material Hardness & Oxidation Prediction

## High-level components
1. **Data** (`data/`)
   - CSV files (`hardness.csv`, `oxidation.csv`)
   - Small, curated dataset used for training and demonstration

2. **Training pipelines** (`src/models/`)
   - `pipelines.py`: builds sklearn pipelines with preprocessing + model
   - `train_hardness.py`, `train_oxidation.py`: train & save models to `models/`
   - `evaluate.py`: evaluate saved models on full datasets

3. **Inference** (`src/inference/`)
   - `validator.py`: input validation and column ordering enforcement
   - `predict.py`: loads models and exposes safe prediction functions used by the web app

4. **Serving** (`src/app/`)
   - Flask app (factory pattern) and HTML UI
   - `templates/index.html` and `static/style.css` for the user interface

5. **Artifacts** (`models/`)
   - `*.joblib` model artifacts and `*_metadata.json` files with metrics and training info

6. **Dev & Ops**
   - `requirements.txt`, `runtime.txt`, `Procfile`, `Makefile`
   - Deploy via Render (or other WSGI hosts)

## Dataflow (simple)
1. CSVs in `data/` → training scripts load data  
2. Pipelines preprocess → train → save pipeline artifact  
3. Flask app loads saved pipelines from `models/`  
4. User inputs → validator → pipeline → prediction → UI/JSON response  
5. Predictions can be logged for monitoring (not enabled by default)

## Deployment notes (Render)
- Use the `Procfile` with gunicorn: `web: gunicorn src.app.app:create_app()`
- Set Python runtime via `runtime.txt`
- Set environment variables in Render (if any)
- CI can be configured to run tests and only deploy when tests pass

## Next improvements
- Add CI to validate model artifact presence and tests  
- Add monitoring and drift detection (e.g., logging predictions)  
- Add dataset versioning (DVC or Git LFS for large data)  
- Add reproducible experiment tracking (MLflow)
