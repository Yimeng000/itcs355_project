"""Reproducible model training and evaluation entry point for Taxi Fare Prediction."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
from pathlib import Path
from typing import Any, Dict

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from src import config, seeds
from src.features import (
    ALL_FEATURES,
    TARGET,
    build_preprocessor,
    extract_input_schema,
    save_input_schema,
)


def git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=config.REPO_ROOT)
        return out.stdout.strip()
    except Exception:
        return "unknown"


def file_fingerprint(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()[:16]


def load_params(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        import yaml

        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except ImportError:
        return {}


def evaluate_predictions(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))
    return {"mae": round(mae, 4), "rmse": round(rmse, 4), "r2": round(r2, 4)}


def create_baseline_model(baseline_type: str, baseline_cfg: dict[str, Any]) -> Any:
    if baseline_type == "linear_regression":
        return LinearRegression()
    strategy = baseline_cfg.get("strategy", "mean")
    return DummyRegressor(strategy=strategy)


def create_candidate_model(model_type: str, seed: int, params: dict[str, Any]) -> Any:
    if model_type == "gradient_boosting":
        gb_cfg = params.get("gradient_boosting", {})
        return GradientBoostingRegressor(
            random_state=seed,
            n_estimators=int(gb_cfg.get("n_estimators", 150)),
            max_depth=int(gb_cfg.get("max_depth", 5)),
            learning_rate=float(gb_cfg.get("learning_rate", 0.1)),
            min_samples_split=int(gb_cfg.get("min_samples_split", 5)),
            min_samples_leaf=int(gb_cfg.get("min_samples_leaf", 2)),
        )

    rf_cfg = params.get("random_forest", {})
    return RandomForestRegressor(
        random_state=seed,
        n_estimators=int(rf_cfg.get("n_estimators", 150)),
        max_depth=int(rf_cfg.get("max_depth", 10)),
        min_samples_split=int(rf_cfg.get("min_samples_split", 5)),
        min_samples_leaf=int(rf_cfg.get("min_samples_leaf", 2)),
        n_jobs=-1,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train baseline and regression models for taxi fare prediction.")
    p.add_argument("--params", type=Path, default=config.REPO_ROOT / "params.yaml", help="Path to params.yaml")
    p.add_argument("--experiment", default="taxi-fare-prediction", help="MLflow experiment name")
    p.add_argument("--run-name", default=None, help="MLflow run name")
    p.add_argument("--metrics-out", type=Path, default=config.REPO_ROOT / "reports" / "metrics.json", help="Path to export evaluation metrics JSON")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    params_all = load_params(args.params)
    train_cfg = params_all.get("train", {})
    paths_cfg = params_all.get("paths", {})

    seed = int(train_cfg.get("seed", params_all.get("seed", seeds.DEFAULT_SEED)))
    seeds.set_all(seed)

    train_path = config.REPO_ROOT / Path(paths_cfg.get("train_csv", "data/processed/train.csv"))
    val_path = config.REPO_ROOT / Path(paths_cfg.get("val_csv", "data/processed/val.csv"))
    test_path = config.REPO_ROOT / Path(paths_cfg.get("test_csv", "data/processed/test.csv"))

    for p in (train_path, val_path, test_path):
        if not p.exists():
            raise FileNotFoundError(f"Dataset split not found at {p}. Run clean.py and split.py first.")

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    X_train, y_train = train_df[ALL_FEATURES], train_df[TARGET]
    X_val, y_val = val_df[ALL_FEATURES], val_df[TARGET]
    X_test, y_test = test_df[ALL_FEATURES], test_df[TARGET]

    train_fingerprint = file_fingerprint(train_path)
    with_scaler = bool(train_cfg.get("with_scaler", True))
    baseline_type = str(train_cfg.get("baseline_type", "dummy_mean"))
    model_type = str(train_cfg.get("model_type", "random_forest"))

    # 1. Fit baseline model pipeline
    baseline_estimator = create_baseline_model(baseline_type, train_cfg.get("baseline", {}))
    baseline_pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(with_scaler=with_scaler)),
            ("regressor", baseline_estimator),
        ]
    )
    baseline_pipeline.fit(X_train, y_train)
    baseline_val_preds = baseline_pipeline.predict(X_val)
    baseline_val_metrics = evaluate_predictions(y_val, baseline_val_preds)

    # 2. Fit candidate model pipeline
    candidate_estimator = create_candidate_model(model_type, seed, train_cfg)
    model_pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(with_scaler=with_scaler)),
            ("regressor", candidate_estimator),
        ]
    )
    model_pipeline.fit(X_train, y_train)
    val_preds = model_pipeline.predict(X_val)
    model_val_metrics = evaluate_predictions(y_val, val_preds)

    # 3. Final evaluation on test partition (test set is untouched until this point)
    test_preds = model_pipeline.predict(X_test)
    model_test_metrics = evaluate_predictions(y_test, test_preds)

    # 4. Serialize fitted pipeline and schema contract
    model_dir = config.REPO_ROOT / Path(paths_cfg.get("model_dir", "models"))
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "model.joblib"
    joblib.dump(model_pipeline, model_path)

    schema = extract_input_schema(train_df)
    schema_path = model_dir / "input_schema.json"
    save_input_schema(schema, schema_path)

    # 5. MLflow Tracking and Metrics Logging
    cfg_app = config.load(strict=False)
    mlflow.set_tracking_uri(cfg_app.mlflow_tracking_uri)
    mlflow.set_experiment(args.experiment)

    all_metrics = {
        "baseline_val_mae": baseline_val_metrics["mae"],
        "baseline_val_rmse": baseline_val_metrics["rmse"],
        "baseline_val_r2": baseline_val_metrics["r2"],
        "val_mae": model_val_metrics["mae"],
        "val_rmse": model_val_metrics["rmse"],
        "val_r2": model_val_metrics["r2"],
        "test_mae": model_test_metrics["mae"],
        "test_rmse": model_test_metrics["rmse"],
        "test_r2": model_test_metrics["r2"],
    }

    with mlflow.start_run(run_name=args.run_name) as run:
        mlflow.log_params(
            {
                "seed": seed,
                "model_type": model_type,
                "baseline_type": baseline_type,
                "with_scaler": with_scaler,
                **train_cfg.get(model_type, {}),
            }
        )
        mlflow.log_metrics(all_metrics)
        mlflow.set_tags(
            {
                "git_commit": git_commit(),
                "data_fingerprint": train_fingerprint,
                "n_train_rows": len(train_df),
                "n_val_rows": len(val_df),
                "n_test_rows": len(test_df),
            }
        )
        mlflow.sklearn.log_model(
            model_pipeline,
            name="model",
            skops_trusted_types=["numpy.dtype", "sklearn.tree._tree.Tree"],
        )

    summary = {
        "training_run_id": run.info.run_id,
        "training_git_commit": git_commit(),
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "seed": seed,
        "data_fingerprint": train_fingerprint,
        "baseline_type": baseline_type,
        "model_type": model_type,
        "metrics": all_metrics,
        "artifacts": {
            "model_path": str(model_path),
            "schema_path": str(schema_path),
        },
    }
    print(json.dumps(summary, indent=2))

    if args.metrics_out:
        args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
