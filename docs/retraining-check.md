# Retraining and Model Comparison

## Purpose

Check reproducibility and compare a newly trained model with registered model v2.

## Recorded Results

We retrained using the same data, parameters and random seed.
The new model file had the same SHA256 as the registered model.

Both models were compared on the same 143 validation records:

| Metric | Reference | Candidate |
|---|---:|---:|
| MAE | 11.4020 | 11.4020 |
| RMSE | 15.9389 | 15.9389 |
| R² | 0.8146 | 0.8146 |

The checks allow numerical noise of 1e-9.
This demonstrates reproducibility, not improved accuracy.

## Evidence

- reports/retraining/params.yaml
- reports/retraining/train_metrics.json
- reports/retraining/comparison.json
- src/compare_models.py
- tests/test_model_comparison.py
- tests/test_registration.py

## Failure Handling

A worse candidate returns exit code 1 and saves a failed comparison report.
Tests also reject incorrect model records and invalid validation targets.

## Limitations

The validation data has already been used to evaluate the reference model.
This experiment does not establish real-world performance.
Test-set results must not be used repeatedly to choose model settings.

Real model registration now requires a passing comparison report.
Registration checks the candidate, reference model, validation data,
training run and reference version against that report.

Candidate evaluation and registration preview were verified locally.
All 97 automated tests passed in the recorded final check.

GitHub Actions now reproduces data checks, training and evaluation,
compares the retrained candidate with the registered reference,
and verifies registration checks without uploading a new model.

Deployment proceeds through staging checks before production.
Production continues to use registered model v2;
retraining does not automatically register or promote a new model.
