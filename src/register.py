"""Validate a trained model and register it through the cloud layer."""
from __future__ import annotations

import argparse
from pathlib import Path
import hashlib
import json

from src import config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--image")
    parser.add_argument("--parent-model")
    parser.add_argument("--params", type=Path, default=config.REPO_ROOT / "params.yaml")
    parser.add_argument("--comparison-report", type=Path)
    parser.add_argument("--reference-model", type=Path, default=config.REPO_ROOT / "models/model.joblib")
    parser.add_argument("--reference-receipt", type=Path, default=config.REPO_ROOT / "reports/registration.json")
    args = parser.parse_args()

    root = config.REPO_ROOT
    import yaml

    params = yaml.safe_load(args.params.read_text()) or {}
    paths = params.get("paths", {})
    model_dir = root / paths.get("model_dir", "models")
    reports_dir = root / params.get("evaluate", {}).get("reports_dir", "reports")

    files = {
        "model.joblib": model_dir / "model.joblib",
        "input_schema.json": model_dir / "input_schema.json",
        "train_metrics.json": reports_dir / "train_metrics.json",
        "metrics.json": reports_dir / params.get("evaluate", {}).get("metrics_file", "metrics.json"),
        "params.yaml": args.params,
        "dvc.lock": root / "dvc.lock",
    }
    for path in files.values():
        if not path.is_file():
            raise RuntimeError(f"Missing required file: {path}")

    training = json.loads(files["train_metrics.json"].read_text())
    evaluation = json.loads(files["metrics.json"].read_text())

    gate = evaluation.get("quality_gate", {})
    checks = gate.get("checks", {})
    if (
        gate.get("passed") is not True
        or not checks
        or any(value is not True for value in checks.values())
    ):
        raise RuntimeError("Evaluation failed; registration stopped")

    model_hash = hashlib.sha256(files["model.joblib"].read_bytes()).hexdigest()
    if training.get("model_sha256") != model_hash:
        raise RuntimeError("Model changed after training; registration stopped")

    # Reject reports produced for different files.
    import yaml
    params = yaml.safe_load(files["params.yaml"].read_text()) or {}
    paths = params.get("paths", {})

    train_path = root / paths.get("train_csv", "data/processed/train.csv")
    test_path = root / paths.get("test_csv", "data/processed/test.csv")

    fingerprints = evaluation.get("artifact_fingerprints", {})
    current_files = {
        "model_sha256": files["model.joblib"],
        "train_sha256": train_path,
        "test_sha256": test_path,
        "params_sha256": files["params.yaml"],
    }
    for name, path in current_files.items():
        if not path.is_file():
            raise RuntimeError(f"Missing evaluation input: {path}")
        current_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if fingerprints.get(name) != current_hash:
            raise RuntimeError(
                f"Evaluation report is missing or outdated: {name}; "
                "run evaluation again before registration"
            )

    train_hash = fingerprints["train_sha256"]
    if training.get("data_fingerprint") != train_hash[:16]:
        raise RuntimeError("Training data changed; registration stopped")

    for key in ("training_run_id", "training_git_commit", "data_fingerprint"):
        if not training.get(key) or training[key] == "unknown":
            raise RuntimeError(f"Missing training record: {key}")

    for metric in ("mae", "rmse", "r2"):
        if (
            evaluation["candidate_metrics"][metric]
            != training["metrics"][f"test_{metric}"]
        ):
            raise RuntimeError("Training and evaluation reports do not match")

    if not args.dry_run and args.comparison_report is None:
        raise RuntimeError("A model comparison report is required for registration")

    if args.comparison_report is not None:
        comparison = json.loads(args.comparison_report.read_text())
        required_checks = {
            "candidate_mae_not_worse",
            "candidate_rmse_not_worse",
            "candidate_r2_not_worse",
            "candidate_fares_nonnegative",
        }
        comparison_checks = comparison.get("checks", {})
        if (
            comparison.get("comparison_passed") is not True
            or not required_checks.issubset(comparison_checks)
            or any(value is not True for value in comparison_checks.values())
        ):
            raise RuntimeError("Model comparison failed; registration stopped")

        validation_path = root / paths.get("val_csv", "data/processed/val.csv")
        reference_receipt = json.loads(args.reference_receipt.read_text())
        reference_hash = hashlib.sha256(args.reference_model.read_bytes()).hexdigest()
        validation_hash = hashlib.sha256(validation_path.read_bytes()).hexdigest()

        expected = {
            "candidate_sha256": model_hash,
            "reference_sha256": reference_hash,
            "validation_sha256": validation_hash,
            "candidate_training_run_id": training["training_run_id"],
            "candidate_training_git_commit": training["training_git_commit"],
            "registered_version": str(reference_receipt["version_id"]),
        }
        if reference_receipt["lineage"]["model_sha256"] != reference_hash:
            raise RuntimeError("Reference does not match its registration receipt")
        for key, value in expected.items():
            if comparison.get(key) != value:
                raise RuntimeError(f"Comparison report is outdated or mismatched: {key}")

        files["comparison.json"] = args.comparison_report

    manifest = {
        "training_run_id": training["training_run_id"],
        "training_git_commit": training["training_git_commit"],
        "data_fingerprint": training["data_fingerprint"],
        "model_sha256": model_hash,
        "model_type": training["model_type"],
        "seed": training["seed"],
        "test_metrics": evaluation["candidate_metrics"],
        "quality_gate_passed": True,
    }
    manifest_path = reports_dir / "model_lineage.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    files["model_lineage.json"] = manifest_path

    if args.dry_run:
        print("Checks passed. Preview only; nothing uploaded.")
        print(json.dumps(manifest, indent=2))
        return

    if not args.image or "@sha256:" not in args.image:
        raise RuntimeError("Specify --image with an immutable image digest")

    from cloudlayer.registry import register_model

    result = register_model(
        config.load(strict=False),
        files,
        manifest,
        args.image,
        args.parent_model,
    )
    receipt = root / "reports/registration.json"
    receipt.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
