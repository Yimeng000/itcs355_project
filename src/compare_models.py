"""Compare a candidate with the registered model on validation data."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.features import ALL_FEATURES, TARGET


def fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(actual, predicted):
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--training-record", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    reference_hash = fingerprint(args.reference)
    candidate_hash = fingerprint(args.candidate)
    receipt = json.loads(args.receipt.read_text())
    training = json.loads(args.training_record.read_text())

    if receipt["lineage"]["model_sha256"] != reference_hash:
        raise RuntimeError("Reference does not match the registered model")
    if training["model_sha256"] != candidate_hash:
        raise RuntimeError("Candidate does not match its training record")

    frame = pd.read_csv(args.validation)
    if len(frame) < 2:
        raise ValueError("Validation data must contain at least two rows")
    actual = frame[TARGET].to_numpy(dtype=float)
    if not np.isfinite(actual).all():
        raise ValueError("Validation targets must be finite")

    reference = joblib.load(args.reference)
    candidate = joblib.load(args.candidate)
    inputs = frame[ALL_FEATURES]
    reference_predictions = np.asarray(reference.predict(inputs), dtype=float)
    candidate_predictions = np.asarray(candidate.predict(inputs), dtype=float)

    for values in (reference_predictions, candidate_predictions):
        if values.shape != actual.shape or not np.isfinite(values).all():
            raise ValueError("Invalid model predictions")

    old = metrics(actual, reference_predictions)
    new = metrics(actual, candidate_predictions)

    # Compare unrounded values; allow only numerical noise.
    epsilon = 1e-9
    checks = {
        "candidate_mae_not_worse": new["mae"] <= old["mae"] + epsilon,
        "candidate_rmse_not_worse": new["rmse"] <= old["rmse"] + epsilon,
        "candidate_r2_not_worse": new["r2"] >= old["r2"] - epsilon,
        "candidate_fares_nonnegative": bool((candidate_predictions >= 0).all()),
    }
    passed = all(checks.values())

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "purpose": "Validation comparison; does not register or deploy a model",
        "registered_version": str(receipt["version_id"]),
        "candidate_training_run_id": training["training_run_id"],
        "candidate_training_git_commit": training["training_git_commit"],
        "reference_sha256": reference_hash,
        "candidate_sha256": candidate_hash,
        "validation_sha256": fingerprint(args.validation),
        "validation_rows": len(frame),
        "reference_metrics": old,
        "candidate_metrics": new,
        "identical_model_files": reference_hash == candidate_hash,
        "predictions_match_within_tolerance": bool(
            np.allclose(
                reference_predictions,
                candidate_predictions,
                rtol=0,
                atol=epsilon,
            )
        ),
        "max_prediction_difference": float(
            np.max(np.abs(reference_predictions - candidate_predictions))
        ),
        "checks": checks,
        "comparison_passed": passed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
