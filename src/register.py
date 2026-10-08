"""Validate a trained model and register it through the cloud layer."""
from __future__ import annotations

import argparse
import hashlib
import json

from src import config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--image")
    parser.add_argument("--parent-model")
    args = parser.parse_args()

    root = config.REPO_ROOT
    files = {
        "model.joblib": root / "models/model.joblib",
        "input_schema.json": root / "models/input_schema.json",
        "train_metrics.json": root / "reports/train_metrics.json",
        "metrics.json": root / "reports/metrics.json",
        "params.yaml": root / "params.yaml",
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

    for key in ("training_run_id", "training_git_commit", "data_fingerprint"):
        if not training.get(key) or training[key] == "unknown":
            raise RuntimeError(f"Missing training record: {key}")

    for metric in ("mae", "rmse", "r2"):
        if (
            evaluation["candidate_metrics"][metric]
            != training["metrics"][f"test_{metric}"]
        ):
            raise RuntimeError("Training and evaluation reports do not match")

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
    manifest_path = root / "reports/model_lineage.json"
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
