# Model card — Stellite-6 hardfacing property predictor

## Model purpose

The service reports two exploratory regression estimates for the published TIG
surfacing experiment in Chandra et al. (2020):

1. a Vickers hardness number for the hardfaced substrate, and
2. cumulative areal oxidation mass gain after a stated exposure.

Both are fitted only to the reported table values for EN-8 and Mild Steel at
120, 140, and 160 A. The service is a reproducibility and communication
demonstration, not an engineering predictor.

## Versions and artifacts

| Target | Production model | Version | Artifact | Manifest |
|---|---|---|---|---|
| Hardness | Linear regression on substrate + current | 0.2.0 | `models/hardness_model.joblib` | `models/hardness_manifest.json` |
| Oxidation | Linear mass-gain **rate** model, scaled by exposure time | 0.3.0 | `models/oxidation_model.joblib` | `models/oxidation_manifest.json` |

Each manifest records the dataset SHA-256, artifact SHA-256, git commit and dirty
state, Python and package versions, pipeline configuration, feature list, target
definition, metrics, validation protocol, seed, and known limitations. The
reproducibility check verifies the data and metrics; see
[`REPRODUCIBILITY.md`](REPRODUCIBILITY.md).

## Data and scope

- Hardness: 6 reported values from Tables 5 and 6 (2 substrates × 3 currents).
- Oxidation: 18 reported values from Tables 8–13 (2 substrates × 3 currents ×
  3 exposure times at 973 K).
- Supported inputs: `EN-8` or `Mild Steel`; `120`, `140`, or `160` A; and `5`,
  `10`, or `15` h for oxidation.
- Current, travel speed, and heat input vary together. Current is the single
  process index in the model; its coefficient is **not** a causal current effect.
- The oxidation exposure rows are exact multiples of the 5 h value, so the
  effective number of independent oxidation observations is **6**, not 18.
- The paper reports no replicate specimens, repeated readings, or uncertainty.
- The Vickers test load is not stated. The oxidation unit and printed sign
  convention are unresolved, and the methods (30 × 30 mm²) and Table 7
  (15 × 15 mm, 225 mm²) disagree on specimen area by a factor of four. See
  [`SOURCE_AUDIT.md`](SOURCE_AUDIT.md).

## Model form

**Hardness.** One-hot substrate and standardized current, followed by ordinary
least squares. With three current levels and six rows, a linear model is the
simplest estimable form.

**Oxidation.** The model predicts mass gain per hour with a linear function of
substrate and current, then multiplies by exposure time:

```text
rate = a + b·current + c·substrate
areal mass gain = rate × exposure_hours
```

This states the table's near-exact linear-time structure directly, works with six
effective conditions, and can extrapolate a rate across current. A random forest
cannot extrapolate and was outperformed under the same validation.

## Evaluation

The primary protocol is **Leave-One-Current-Out (LOCO)** over three current groups:
every row at the held-out current is excluded from training. This is a transfer
stress test, not an interpolation test. A secondary **Leave-One-Condition-Out
(LOGO)** protocol holds out each Material × Current cell; it is closer to
interpolation but leaks the process setting across substrates.

| Target | Protocol | MAE | RMSE | R² | Fold MAE (mean ± sd) |
|---|---|---:|---:|---:|---:|
| Hardness | LOCO | 26.25 | 27.51 | −0.656 | 26.25 ± 7.42 |
| Oxidation | LOCO | 1.346 × 10⁻⁴ | 1.631 × 10⁻⁴ | 0.9936 | 1.35 × 10⁻⁴ ± 0.48 × 10⁻⁴ |

The hardness R² is negative: withholding an entire current produces worse
predictions than the mean. There are only three folds, and all metrics rest on six
independent conditions.

### Fair comparison and the v0.3.0 change

[`reports/benchmark/`](../reports/benchmark/) contains every out-of-fold
prediction, fold-level metric, and single-row sensitivity result. The adoption
rule was fixed before running:

- primary LOCO MAE improves by at least 20 %;
- the improvement appears in at least 2 of 3 folds;
- secondary LOGO MAE does not worsen by more than 10 %.

| Target | Shipped | Best challenger | Δ LOCO MAE | Folds improved | Decision |
|---|---|---|---:|---:|---|
| Hardness | `linear` | `ridge` | −12.0 % | 1 / 3 | Keep `linear` |
| Oxidation | `random_forest` (v0.2.0) | `rate_linear` | −69.4 % | 3 / 3 | Adopt `rate_linear` |

The oxidation change is recorded in
[`reports/benchmark/pre_adoption_random_forest/`](../reports/benchmark/pre_adoption_random_forest/)
and in the reproducibility baseline. The hardness ridge challenger did not meet
the pre-registered evidence bar, so the simpler linear model remains.

### Uncertainty

The API and UI attach an approximate 90 % interval derived from LOCO residuals
(`models/calibration.json`). The interval uses signed residual quantiles, so it
can be asymmetric:

| Target | n OOF | Effective n | 90 % interval offsets (lower, upper) |
|---|---:|---:|---:|
| Hardness | 6 | 6 | −17.50, +36.50 |
| Oxidation | 18 | 6 | −3.465 × 10⁻⁴, +2.61 × 10⁻⁴ |

These are **not** guaranteed-coverage intervals. They summarize observed OOF
errors in six conditions and cannot represent specimen-to-specimen variability
that was never reported.

### Applicability domain

The service accepts only the exact paper-tested conditions. Every response states
whether the input is a `reported_condition`, an
`unmeasured_combination_within_tested_levels`, or `outside_tested_conditions`, and
returns the paper-reported value at that condition when one exists. The UI shows
the model estimate, the interval, and the paper value side by side.

### Explanations

No post-hoc explanation is shipped. With collinear process factors and six rows,
feature attributions would look authoritative without being identified. The
hardness coefficients describe the paper's schedule, not independent physical
effects. The oxidation rate coefficients describe a fitted rate surface within the
three tested currents. The benchmark provides model-level evidence instead of
per-prediction attributions.

## Intended use

- Reproducing the paper's tables and demonstrating a traceable small-data workflow.
- Exploring model behavior at the reported conditions.
- Teaching the difference between grouped validation, interpolation, and
  extrapolation.

## Not intended use

- Safety-critical, production, or qualification decisions.
- Claims about current effects independent of travel speed or heat input.
- Predictions for other alloys, processes, temperatures, currents, exposure times,
  specimen geometries, or surface preparations.
- Treating an estimate as a replacement for hardness or oxidation testing.
- Interpreting the printed oxidation magnitude as a physical mass change without
  resolving the unit and sign issues.

## Limitations and risks

- One study, two substrates, three currents, and no independent test set.
- No replicates; reported values only. The effective oxidation sample is six.
- Hardness generalization across currents is poor (negative LOCO R²).
- Oxidation time rows are multiples of the 5 h value, so time does not provide
  independent kinetic evidence.
- Unit scale and sign of the oxidation target are unresolved, and the exposed
  area is ambiguous by a factor of four.
- Process factors are collinear; causal attribution is impossible.
- No external measurements have been collected yet. The
  [`data/external/`](../data/external/README.md) path exists to validate them
  without retraining, but it does not substitute for an independent test set.
- Intervals are approximate and based on a handful of residuals.
- The API/UI are limited to tested conditions by design.

## Acceptance criteria for future model changes

A challenger may replace a production model only if, under the same LOCO protocol:

1. primary MAE improves by at least 20 %;
2. at least two of three folds improve;
3. secondary LOGO MAE does not worsen by more than 10 %;
4. the new model is no more complex without a stated scientific reason; and
5. the benchmark, manifest, reproducibility baseline, model card, and API/UI
   metadata are updated in the same change.

## Maintenance

- Retrain after any data or pipeline change; regenerate manifests and calibration.
- Run `python -m src.data.audit`, `python -m src.models.reproducibility`,
  `python -m src.models.benchmark`, and `pytest`.
- Keep reported, derived, and newly measured rows distinguishable with
  `Data_Status`, `Derivation_Status`, and source fields.
- Add independent experiments before making broader performance claims; see
  [`EXPERIMENTAL_PLAN.md`](EXPERIMENTAL_PLAN.md).
