# Contributing

This project treats data provenance and evaluation integrity as part of the
contribution, not as an afterthought.

## Setup

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt -r requirements-dev.txt
```

Run the checks before opening a pull request:

```bash
python -m src.data.audit
python -m src.models.reproducibility
ruff check .
ruff format --check .
pytest -q
```

## Rules

### Data

- Every training row must remain traceable to a source table or a documented
  experiment.
- Do not add interpolated, smoothed, or derived values as measurements.
- Do not silently correct units, signs, or chemistry values. Record the issue and
  its status instead.
- New measurements need a `Data_Status` other than
  `reported_paper_measurement` and a full metadata record. See
  [`docs/EXPERIMENTAL_PLAN.md`](docs/EXPERIMENTAL_PLAN.md).
- Changing a checked-in value requires updating the audit, dataset hash,
  reproducibility baseline, benchmark, and model card in the same change.

### Models

- Use the grouped protocols in `src/models/benchmark.py`. Do not substitute a
  random row split.
- Keep preprocessing inside the scikit-learn pipeline.
- A model may replace the production model only under the pre-registered
  adoption rule in `docs/MODEL_CARD.md`.
- Retrain, recalibrate, and regenerate manifests when the data or pipeline
  changes.
- Do not ship post-hoc explanations that imply causal feature effects from this
  collinear six-condition dataset.

### Code

- Keep changes small and typed where practical.
- Add tests for bug fixes and behavior changes.
- Update `docs/API_REFERENCE.md` when a response field changes.
- Keep the browser UI usable without JavaScript; the form must remain a normal
  POST fallback.
- Run Ruff before committing. Pre-commit is configured in
  `.pre-commit-config.yaml`.

### Documentation

- Keep unresolved scientific issues visible. Do not remove a warning because it
  makes the demo look less polished.
- Distinguish reported measurements, derived rows, and new experiments.
- Do not describe archived screenshots or plots as current results.

## Licensing

- Code contributions are accepted under the MIT License.
- Transcribed paper data remain under CC BY 3.0 with attribution to the original
  authors. See [`data/LICENSE`](data/LICENSE).
- Do not contribute data or figures whose license is incompatible with these
  terms.
