"""Check raw taxi data before cleaning and training."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.clean import NUMERIC_COLS, CATEGORICAL_COLS


CATEGORIES = {
    "Time_of_Day": {"morning", "afternoon", "evening", "night"},
    "Day_of_Week": {"weekday", "weekend"},
    "Traffic_Conditions": {"low", "medium", "high"},
    "Weather": {"clear", "rain", "snow"},
}


def check_data(frame, min_rows=100, max_missing_fraction=0.20):
    if min_rows < 1 or not 0 <= max_missing_fraction <= 1:
        raise ValueError("Invalid data quality thresholds")

    required = NUMERIC_COLS + CATEGORICAL_COLS
    missing_columns = sorted(set(required) - set(frame.columns))
    checks = {
        "minimum_rows": len(frame) >= min_rows,
        "required_columns_present": not missing_columns,
        "unique_column_names": bool(frame.columns.is_unique),
    }
    details = {}
    if not missing_columns and frame.columns.is_unique:
        for column in required:
            values = frame[column].replace(r"^\s*$", np.nan, regex=True)
            missing_fraction = float(values.isna().mean()) if len(frame) else 1.0
            checks[f"{column}:missing_fraction"] = (
                missing_fraction <= max_missing_fraction
            )
            info = {"missing_fraction": missing_fraction}

            if column in NUMERIC_COLS:
                numeric = pd.to_numeric(values, errors="coerce")
                invalid = values.notna() & (
                    numeric.isna() | ~np.isfinite(numeric)
                )
                if column in ("Trip_Distance_km", "Per_Km_Rate", "Trip_Price"):
                    invalid |= numeric.notna() & (numeric <= 0)
                elif column in ("Base_Fare", "Per_Minute_Rate"):
                    invalid |= numeric.notna() & (numeric < 0)
                elif column == "Passenger_Count":
                    finite = numeric.notna() & np.isfinite(numeric)
                    invalid |= finite & (
                        (numeric < 1) | (numeric > 6)
                        | (numeric != np.floor(numeric))
                    )
                info["invalid_values"] = int(invalid.sum())
                checks[f"{column}:valid_values"] = not bool(invalid.any())
            else:
                normalized = values.astype("string").str.strip().str.lower()
                invalid = values.notna() & ~normalized.isin(CATEGORIES[column])
                info["unknown_values"] = sorted(
                    normalized[invalid].dropna().unique().tolist()
                )
                checks[f"{column}:known_categories"] = not bool(invalid.any())

            details[column] = info

    return {
        "rows": len(frame),
        "missing_columns": missing_columns,
        "thresholds": {
            "minimum_rows": min_rows,
            "maximum_missing_fraction": max_missing_fraction,
            "training_passenger_range": [1, 6],
        },
        "columns": details,
        "checks": checks,
        "passed": all(checks.values()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--min-rows", type=int, default=100)
    parser.add_argument("--max-missing-fraction", type=float, default=0.20)
    args = parser.parse_args()

    report = check_data(
        pd.read_csv(args.input),
        args.min_rows,
        args.max_missing_fraction,
    )
    report["raw_sha256"] = hashlib.sha256(args.input.read_bytes()).hexdigest()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print("Data quality:", "PASSED" if report["passed"] else "FAILED")
    for name, passed in report["checks"].items():
        if not passed:
            print("FAILED:", name)
    print("Report:", args.report)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
