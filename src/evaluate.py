#!/usr/bin/env python3
"""
Evaluation and Quality Gate Script for Taxi Fare Prediction.

Evaluates the saved model pipeline against baseline performance on the isolated test set,
benchmarks single-row prediction latency to verify sub-second response times,
saves the predicted-vs-actual diagnostic plot, and enforces a strict quality gate.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Ensure repository root is in sys.path for direct script execution
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src import config
from src.features import ALL_FEATURES, TARGET, build_preprocessor
from src.train import create_baseline_model, evaluate_predictions, load_params

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def benchmark_single_row_latency(
    pipeline: Any,
    X_test: pd.DataFrame,
    n_samples: int = 100,
) -> Dict[str, float | bool]:
    """
    Time individual single-row predictions to simulate real-time API inference.
    Collects statistical evidence for the under-one-second SLA requirement.
    """
    sample_count = min(n_samples, len(X_test))
    latencies_ms: List[float] = []

    # Warmup
    _ = pipeline.predict(X_test.iloc[[0]])

    for i in range(sample_count):
        single_row = X_test.iloc[[i]]
        t0 = time.perf_counter()
        _ = pipeline.predict(single_row)
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)

    p50 = float(np.percentile(latencies_ms, 50))
    p95 = float(np.percentile(latencies_ms, 95))
    p99 = float(np.percentile(latencies_ms, 99))
    mean_lat = float(np.mean(latencies_ms))
    max_lat = float(np.max(latencies_ms))

    return {
        "n_samples": sample_count,
        "mean_ms": round(mean_lat, 2),
        "p50_ms": round(p50, 2),
        "p95_ms": round(p95, 2),
        "p99_ms": round(p99, 2),
        "max_ms": round(max_lat, 2),
        "under_one_second": bool(p95 < 1000.0),
    }


def save_predicted_vs_actual_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    plot_path: Path,
    r2_score_val: float,
) -> None:
    """Generate and save a scatter plot of predicted vs. actual values for the model card."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(7, 6))

        ax.scatter(y_true, y_pred, alpha=0.35, edgecolors="none", s=20, label="Predictions")
        min_val = min(float(y_true.min()), float(y_pred.min()))
        max_val = max(float(y_true.max()), float(y_pred.max()))
        ax.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Ideal (y = x)")

        ax.set_xlabel("Actual Fare ($)")
        ax.set_ylabel("Predicted Fare ($)")
        ax.set_title(f"Predicted vs Actual Fare (Test Set — R²: {r2_score_val:.3f})")
        ax.legend(loc="upper left")
        ax.grid(True, linestyle=":", alpha=0.6)

        fig.tight_layout()
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        logger.info("Saved predicted-vs-actual plot to: %s", plot_path)
    except ImportError:
        logger.warning("matplotlib is not installed; skipping plot generation.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate taxi fare prediction model and apply quality gate.")
    p.add_argument("--params", type=Path, default=config.REPO_ROOT / "params.yaml", help="Path to params.yaml")
    p.add_argument("--model-path", type=Path, default=None, help="Override path to model.joblib")
    p.add_argument("--test-csv", type=Path, default=None, help="Override path to test.csv")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    params = load_params(args.params)
    paths_cfg = params.get("paths", {})
    train_cfg = params.get("train", {})
    eval_cfg = params.get("evaluate", {})
    thresholds = eval_cfg.get("thresholds", {})
    latency_cfg = eval_cfg.get("latency", {})

    model_path = args.model_path or (
        config.REPO_ROOT / Path(paths_cfg.get("model_dir", "models")) / "model.joblib"
    )
    train_path = config.REPO_ROOT / Path(paths_cfg.get("train_csv", "data/processed/train.csv"))
    test_path = args.test_csv or (
        config.REPO_ROOT / Path(paths_cfg.get("test_csv", "data/processed/test.csv"))
    )

    reports_dir = config.REPO_ROOT / Path(eval_cfg.get("reports_dir", "reports"))
    reports_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = reports_dir / eval_cfg.get("metrics_file", "metrics.json")
    plot_path = reports_dir / eval_cfg.get("plot_file", "predicted_vs_actual.png")

    if not model_path.exists():
        logger.error("Trained model not found at '%s'. Run src/train.py first.", model_path)
        sys.exit(1)
    if not test_path.exists():
        logger.error("Test dataset not found at '%s'. Run src/split.py first.", test_path)
        sys.exit(1)

    logger.info("Loading model from %s", model_path)
    model_pipeline = joblib.load(model_path)

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train, y_train = train_df[ALL_FEATURES], train_df[TARGET]
    X_test, y_test = test_df[ALL_FEATURES], test_df[TARGET]

    # 1. Evaluate candidate model on isolated test set
    logger.info("Evaluating candidate model on test set (%d rows)...", len(test_df))
    model_test_preds = model_pipeline.predict(X_test)
    model_metrics = evaluate_predictions(y_test, model_test_preds)

    # 2. Fit and evaluate baseline model on test set for direct comparison
    baseline_type = str(train_cfg.get("baseline_type", "dummy_mean"))
    with_scaler = bool(train_cfg.get("with_scaler", True))
    baseline_estimator = create_baseline_model(baseline_type, train_cfg.get("baseline", {}))
    baseline_pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(with_scaler=with_scaler)),
            ("regressor", baseline_estimator),
        ]
    )
    baseline_pipeline.fit(X_train, y_train)
    baseline_test_preds = baseline_pipeline.predict(X_test)
    baseline_metrics = evaluate_predictions(y_test, baseline_test_preds)

    # 3. Single-row latency benchmarking (evidence for under 1s SLA)
    n_latency_samples = int(latency_cfg.get("n_samples", 100))
    logger.info("Benchmarking single-row latency across %d iterations...", n_latency_samples)
    latency_metrics = benchmark_single_row_latency(model_pipeline, X_test, n_samples=n_latency_samples)

    # 4. Generate diagnostic plot
    save_predicted_vs_actual_plot(y_test.values, model_test_preds, plot_path, model_metrics["r2"])

    # 5. Quality Gate Evaluation
    min_r2 = float(thresholds.get("min_r2", 0.70))
    max_mae = float(thresholds.get("max_mae", 5.00))
    max_rmse = float(thresholds.get("max_rmse", 8.00))
    max_p95_ms = float(latency_cfg.get("max_p95_ms", 1000.0))

    gate_checks = {
        "candidate_beats_baseline_mae": bool(model_metrics["mae"] < baseline_metrics["mae"]),
        "candidate_beats_baseline_r2": bool(model_metrics["r2"] > baseline_metrics["r2"]),
        f"r2_meets_threshold (>= {min_r2})": bool(model_metrics["r2"] >= min_r2),
        f"mae_meets_threshold (<= {max_mae})": bool(model_metrics["mae"] <= max_mae),
        f"rmse_meets_threshold (<= {max_rmse})": bool(model_metrics["rmse"] <= max_rmse),
        f"latency_p95_meets_sla (<= {max_p95_ms}ms)": bool(latency_metrics["p95_ms"] <= max_p95_ms),
    }
    quality_gate_passed = all(gate_checks.values())

    # 6. Save audit report
    report_payload = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_path": str(model_path),
        "test_dataset": str(test_path),
        "test_rows": len(test_df),
        "candidate_metrics": model_metrics,
        "baseline_metrics": baseline_metrics,
        "latency_benchmark": latency_metrics,
        "quality_gate": {
            "passed": quality_gate_passed,
            "checks": gate_checks,
        },
        "artifacts": {
            "plot_path": str(plot_path) if plot_path.exists() else None,
        },
    }

    metrics_path.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")
    logger.info("Saved evaluation metrics and quality gate report to %s", metrics_path)

    print("\n================ QUALITY GATE SUMMARY ================")
    print(f"Model Test MAE:     {model_metrics['mae']:.4f}  (Baseline: {baseline_metrics['mae']:.4f})")
    print(f"Model Test RMSE:    {model_metrics['rmse']:.4f}  (Baseline: {baseline_metrics['rmse']:.4f})")
    print(f"Model Test R²:      {model_metrics['r2']:.4f}  (Baseline: {baseline_metrics['r2']:.4f})")
    print(f"Inference Latency:  p50={latency_metrics['p50_ms']}ms, p95={latency_metrics['p95_ms']}ms (<1000ms SLA: {latency_metrics['under_one_second']})")
    print("\nGate Verification Rules:")
    for rule_name, passed in gate_checks.items():
        status = "PASSED" if passed else "FAILED"
        print(f"  [{status}] {rule_name}")

    if not quality_gate_passed:
        print("\n[ALERT] Quality gate failed. Model will NOT be registered or promoted.")
        print("======================================================\n")
        sys.exit(1)

    print("\n[SUCCESS] Quality gate passed. Model is approved for registration.")
    print("======================================================\n")


if __name__ == "__main__":
    main()