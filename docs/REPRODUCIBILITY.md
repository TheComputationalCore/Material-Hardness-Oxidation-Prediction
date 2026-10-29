# Reproducibility and artifact provenance

The project records a numerical baseline and verifies it in CI. A change to a
source value, a pipeline, or a model choice must fail the check until the
baseline, benchmark, model card, and metadata are updated together.

## 1. Quick check

```bash
python -m src.models.reproducibility
```

The command recomputes out-of-fold predictions directly from the canonical CSVs.
It does **not** load the saved `.joblib` artifacts for prediction, so a stale or
corrupt artifact cannot hide behind matching metadata.

Exit status is 0 on success and 1 on failure. Add `--write` to save the observed
report to [`reports/reproducibility/current.json`](../reports/reproducibility/current.json).

## 2. What is recorded

[`reports/reproducibility/expected_metrics.json`](../reports/reproducibility/expected_metrics.json)
is the reference baseline (currently baseline `0.3.0`). It records:

| Field | Meaning |
|---|---|
| `baseline_id` | Version of the recorded baseline |
| `supersedes` | Previous baseline this one replaces |
| `captured_at` | Date the baseline was recorded |
| `captured_from_commit` | Commit used when the baseline was captured |
| `protocol` | Validation protocol for each target |
| `datasets` | SHA-256 of each training CSV |
| `metrics` | MAE, RMSE, and R² under the recorded protocol |
| `adoption` | Model change that created this baseline, when applicable |
| `tolerance` | Absolute and relative comparison tolerances |

The hardness metrics are unchanged from baseline `0.1.0`. Oxidation changed in
`0.3.0` when the linear rate model replaced the random forest. The pre-adoption
numbers remain in
[`reports/reproducibility/baseline_0.1.0.json`](../reports/reproducibility/baseline_0.1.0.json)
and [`reports/benchmark/pre_adoption_random_forest/`](../reports/benchmark/pre_adoption_random_forest/).

## 3. Validation protocols

Both targets use `LeaveOneGroupOut` with `Current` as the group:

- **Hardness:** three folds. Each fold trains on four rows (two substrates at two
  currents) and predicts the two rows at the held-out current.
- **Oxidation:** three folds. Each fold trains on twelve rows (two substrates at
  two currents × three exposure times) and predicts the six rows at the held-out
  current.

The reported metrics are pooled over out-of-fold predictions. The deployment
pipeline is then refit on all reported rows. Deployment predictions are therefore
**not** the same as the validation predictions; the validation numbers describe
transfer to an unseen current, not accuracy on the training table.

There are only three current groups, so two folds require predicting outside the
training current range. This is a stress test, not a realistic interpolation
benchmark.

## 4. Artifact manifests

Each training run writes a manifest next to the model:

| Model | Manifest | Version |
|---|---|---|
| Hardness | [`models/hardness_manifest.json`](../models/hardness_manifest.json) | 0.2.0 |
| Oxidation | [`models/oxidation_manifest.json`](../models/oxidation_manifest.json) | 0.3.0 |

Manifests follow schema version `1.0` and record:

- model name and version;
- artifact path, SHA-256, and size;
- dataset path, kind, SHA-256, and row count;
- estimator class, parameters, and preprocessing steps;
- feature order, target column, and target definition;
- seed (if any);
- LOCO metrics and validation description, including effective independent n;
- known limitations;
- Python and package versions, git commit, and dirty state.

`verify_manifest` checks the schema version, artifact hash, dataset hash, and
required fields. The reproducibility command also checks that manifest metrics
match the recorded baseline.

Timestamps and git state are expected to change between runs; hashes and metrics
are the invariants. The hardness artifact is deterministic. The oxidation rate
model is also deterministic (no random seed).

## 5. Calibration

Prediction intervals are generated from LOCO residuals:

```bash
python -m src.models.calibration
```

This writes [`models/calibration.json`](../models/calibration.json) with residual
bias, standard deviation, signed 5th/95th percentiles, and absolute-residual
quantiles. The API uses the signed quantiles as additive offsets, so intervals
can be asymmetric:

| Target | n OOF | Effective n | Lower offset | Upper offset |
|---|---:|---:|---:|---:|
| Hardness | 6 | 6 | −17.50 | +36.50 |
| Oxidation | 18 | 6 | −3.465 × 10⁻⁴ | +2.61 × 10⁻⁴ |

These are approximate empirical intervals, not guaranteed-coverage intervals.
With six independent conditions, they cannot represent specimen-to-specimen
variability that was never measured.

## 6. Benchmark

```bash
python -m src.models.benchmark
```

This recomputes every challenger under the same folds and rewrites:

- [`reports/benchmark/oof_predictions.csv`](../reports/benchmark/oof_predictions.csv) — one row per observation, model, protocol, and fold;
- `fold_metrics.csv` — per-fold MAE/RMSE/R²;
- `summary.md` — pooled metrics and the selection decision;
- `selection.json` — machine-readable decision record;
- `sensitivity.csv` — MAE after dropping each single row.

The adoption rule is fixed in code:

1. primary LOCO MAE improves by at least 20 %;
2. at least two of three folds improve;
3. secondary LOGO MAE does not worsen by more than 10 %.

A challenger that passes all three may replace the production model. The change
must also update the manifest, calibration, baseline, model card, and README in
the same commit.

## 6a. External measurements

```bash
python -m src.models.external_validation --write
```

This validates `data/external/hardness_measurements.csv` and
`data/external/oxidation_measurements.csv`, scores every row with the shipped
models, and writes [`reports/external_validation/`](../reports/external_validation/).
Empty header-only templates are valid and produce an "empty template" note, so the
command is safe to run before real data exist. It exits non-zero only when a file
violates the external schema.

External rows are not training data. They are never merged into `data/hardness.csv`
or `data/oxidation.csv` by this command, and the training loader rejects their
`Data_Status`. Oxidation residuals are computed only when the file supplies
`Paper_Unit_Value` on the paper's printed scale; raw weights are otherwise
converted to physical mg/cm² and reported separately. See
[`../data/external/README.md`](../data/external/README.md).

## 7. Full local reproduction

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt -r requirements-dev.txt

python -m src.data.audit
python -m src.models.train_hardness
python -m src.models.train_oxidation
python -m src.models.calibration
python -m src.models.reproducibility --write
python -m src.models.benchmark
python -m src.models.external_validation --write
ruff check .
ruff format --check .
pytest -q
```

Or:

```bash
make audit
make train
make calibrate
make reproduce
make benchmark
make external
make check
```

`make check` runs the audit, reproducibility check, tests, and lint. It does not
retrain; CI retrains before checking.

## 8. CI gates

[`.github/workflows/ci.yml`](../.github/workflows/ci.yml) runs on pushes and pull
requests to `main`:

1. install dependencies;
2. audit source tables (`python -m src.data.audit`);
3. train both models;
4. regenerate calibration;
5. verify the reproducibility baseline;
6. run the benchmark;
7. `ruff check`;
8. `ruff format --check`;
9. `pytest -q`.

A pull request that changes a data value without updating `expected_metrics.json`
will fail at step 5. A pull request that changes the model without updating the
baseline and benchmark will also fail.

## 9. Updating the baseline deliberately

When a change is intentional:

1. Run the audit and confirm every value still matches the paper.
2. Retrain, calibrate, and run the benchmark.
3. Inspect `selection.json` against the adoption rule.
4. Update `expected_metrics.json`:
   - increment `baseline_id`;
   - set `supersedes`;
   - update `captured_at` and `captured_from_commit`;
   - update dataset hashes and metrics;
   - record any adoption evidence.
5. Update the model card, README metrics table, and UI/API metadata.
6. Run `python -m src.models.reproducibility` and the test suite.

Never update the baseline merely to make a failing check pass. A baseline change
is a claim that the underlying result changed intentionally.

## 10. Environment

- Python 3.10 (CI and Render use 3.10; Render pins 3.10.13).
- Flask 3.0, pandas 2.2, NumPy 1.26, scikit-learn 1.3.2, joblib 1.3.2.
- The manifests record the exact package versions used for each artifact.
- `requirements.txt` is the deployment set; `requirements-dev.txt` adds pytest,
  coverage, ruff, and pre-commit.

Cross-version floating-point drift can change the last digits of a metric. The
baseline tolerance is absolute `1e-12` and relative `1e-9`, which is tighter than
typical cross-platform drift; if a dependency upgrade changes a metric, record the
new baseline rather than loosening the tolerance silently.
