#!/usr/bin/env python3
"""
Reproducible train/validation/test data splitter for Taxi Fare Prediction.

Reads cleaned dataset and configuration from params.yaml, performs a deterministic
split using train_test_split, and exports isolated partitions alongside an audit report.
"""

import argparse
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_SEED = 42
DEFAULT_VAL_SIZE = 0.15
DEFAULT_TEST_SIZE = 0.15


def compute_file_sha256(path: Path) -> str:
    """Compute content hash to trace split outputs back to input data."""
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()[:16]


def load_params(params_path: Path) -> Dict[str, Any]:
    """Load split configuration from params.yaml, with safe fallback parsing."""
    if not params_path.exists():
        logger.warning("Params file '%s' not found. Using defaults.", params_path)
        return {}

    try:
        import yaml

        with open(params_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except ImportError:
        # Fallback simple parser if pyyaml is not installed
        params: Dict[str, Any] = {}
        for line in params_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or ":" not in line:
                continue
            k, _, v = line.partition(":")
            val_str = v.strip()
            try:
                val: Any = float(val_str) if "." in val_str else int(val_str)
            except ValueError:
                val = val_str
            params[k.strip()] = val
        return params


def split_data(
    df: pd.DataFrame,
    seed: int = DEFAULT_SEED,
    val_size: float = DEFAULT_VAL_SIZE,
    test_size: float = DEFAULT_TEST_SIZE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Reproducibly split data into train, validation, and test partitions.

    The test partition must remain isolated until final evaluation.
    """
    if not (0 < val_size < 1.0 and 0 < test_size < 1.0 and (val_size + test_size) < 1.0):
        raise ValueError("Invalid split fractions: val_size and test_size must sum to < 1.0")

    # 1. First split: isolate test partition
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        shuffle=True,
    )

    # 2. Second split: partition remaining data into train and validation
    val_relative_size = val_size / (1.0 - test_size)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_relative_size,
        random_state=seed,
        shuffle=True,
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(
        description="Reproducible train/val/test data splitting with isolated test set."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=repo_root / "data" / "processed" / "clean.csv",
        help="Path to cleaned input CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "data" / "processed",
        help="Directory where split CSVs and report will be written.",
    )
    parser.add_argument(
        "--params",
        type=Path,
        default=repo_root / "params.yaml",
        help="Path to params.yaml containing seed and split ratios.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Override random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    params = load_params(args.params)

    # Precedence: CLI args > params.yaml > hardcoded defaults
    split_cfg = params.get("split", {}) if isinstance(params.get("split"), dict) else {}
    seed = (
        args.seed
        if args.seed is not None
        else split_cfg.get("seed", params.get("seed", DEFAULT_SEED))
    )
    val_size = float(split_cfg.get("val_size", DEFAULT_VAL_SIZE))
    test_size = float(split_cfg.get("test_size", DEFAULT_TEST_SIZE))

    input_path: Path = args.input
    if not input_path.exists():
        raise FileNotFoundError(f"Cleaned dataset not found at '{input_path}'. Run src/clean.py first.")

    logger.info("Loading cleaned data from: %s", input_path)
    df = pd.read_csv(input_path)
    input_fingerprint = compute_file_sha256(input_path)

    logger.info("Splitting %d rows with seed=%d (train=%.2f, val=%.2f, test=%.2f)",
                len(df), seed, 1.0 - val_size - test_size, val_size, test_size)
    train_df, val_df, test_df = split_data(df, seed=seed, val_size=val_size, test_size=test_size)

    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = out_dir / "train.csv"
    val_path = out_dir / "val.csv"
    test_path = out_dir / "test.csv"
    report_path = out_dir / "split_report.json"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    report = {
        "input_file": str(input_path),
        "input_sha256": input_fingerprint,
        "seed": seed,
        "total_rows": len(df),
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "test_rows": len(test_df),
        "train_fraction": round(len(train_df) / len(df), 4),
        "val_fraction": round(len(val_df) / len(df), 4),
        "test_fraction": round(len(test_df) / len(df), 4),
    }

    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Splits saved to %s (train: %d, val: %d, test: %d)",
                out_dir, len(train_df), len(val_df), len(test_df))


if __name__ == "__main__":
    main()