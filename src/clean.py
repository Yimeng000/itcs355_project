#!/usr/bin/env python3
"""
Data cleaning script for Taxi Fare Prediction.

Reads raw trip pricing data, filters invalid records and missing targets,
drops post-ride leakage columns, standardizes categorical representations,
and outputs cleaned data alongside a cleaning report.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

TARGET_COL = "Trip_Price"
LEAKAGE_COLS = ["Trip_Duration_Minutes"]

CATEGORICAL_COLS = [
    "Time_of_Day",
    "Day_of_Week",
    "Traffic_Conditions",
    "Weather",
]

NUMERIC_COLS = [
    "Trip_Distance_km",
    "Passenger_Count",
    "Base_Fare",
    "Per_Km_Rate",
    "Per_Minute_Rate",
    TARGET_COL,
]


def standardize_categories(df: pd.DataFrame, cat_cols: List[str]) -> pd.DataFrame:
    """Strip whitespace and lowercase string category values while preserving NaNs."""
    for col in cat_cols:
        if col in df.columns:
            cleaned = df[col].astype(str).str.strip().str.lower()
            # Restore genuine NaNs from pandas conversion of float NaN/empty strings
            df[col] = df[col].where(df[col].notna() & (df[col].astype(str).str.strip() != ""), np.nan)
            df[col] = cleaned.where(df[col].notna(), np.nan)
    return df


def enforce_types(df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
    """Coerce numeric columns to float dtypes, turning unparseable values to NaN."""
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def clean_data(raw_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Deterministically clean taxi trip pricing records.

    Returns the cleaned DataFrame and an audit report dictionary.
    """
    rows_in = int(len(raw_df))
    logger.info("Loaded %d raw records.", rows_in)

    df = raw_df.copy()
    rows_dropped_per_rule: Dict[str, int] = {}

    # 1. Standardize types before validation checks
    df = enforce_types(df, NUMERIC_COLS)
    df = standardize_categories(df, CATEGORICAL_COLS)

    # 2. Drop leakage columns (actual duration / post-ride fields)
    dropped_leakage: List[str] = []
    for col in LEAKAGE_COLS:
        if col in df.columns:
            df = df.drop(columns=[col])
            dropped_leakage.append(col)
            logger.info("Dropped leakage column: '%s'", col)

    # 3. Rule: Remove missing target
    missing_target_mask = df[TARGET_COL].isna()
    dropped_missing_target = int(missing_target_mask.sum())
    rows_dropped_per_rule["missing_target"] = dropped_missing_target
    df = df[~missing_target_mask]
    logger.info("Rule 'missing_target' removed: %d rows", dropped_missing_target)

    # 4. Rule: Reject non-positive target fare (Trip_Price <= 0)
    invalid_fare_mask = df[TARGET_COL] <= 0
    dropped_invalid_fare = int(invalid_fare_mask.sum())
    rows_dropped_per_rule["invalid_fare"] = dropped_invalid_fare
    df = df[~invalid_fare_mask]
    logger.info("Rule 'invalid_fare' removed: %d rows", dropped_invalid_fare)

    # 5. Rule: Reject invalid trip distance (distance <= 0, when present)
    # Missing distance values are kept as NaN for pipeline imputation
    invalid_dist_mask = df["Trip_Distance_km"].notna() & (df["Trip_Distance_km"] <= 0)
    dropped_invalid_dist = int(invalid_dist_mask.sum())
    rows_dropped_per_rule["invalid_distance"] = dropped_invalid_dist
    df = df[~invalid_dist_mask]
    logger.info("Rule 'invalid_distance' removed: %d rows", dropped_invalid_dist)

    # 6. Rule: Reject passenger count outside 1–6 (when present)
    # Missing passenger counts are kept as NaN for pipeline imputation
    invalid_pax_mask = df["Passenger_Count"].notna() & (
        (df["Passenger_Count"] < 1) | (df["Passenger_Count"] > 6)
    )
    dropped_invalid_pax = int(invalid_pax_mask.sum())
    rows_dropped_per_rule["invalid_passenger_count"] = dropped_invalid_pax
    df = df[~invalid_pax_mask]
    logger.info("Rule 'invalid_passenger_count' removed: %d rows", dropped_invalid_pax)

    df = df.reset_index(drop=True)
    rows_out = int(len(df))
    total_dropped = rows_in - rows_out

    report = {
        "rows_in": rows_in,
        "rows_out": rows_out,
        "rows_dropped_total": total_dropped,
        "rows_dropped_per_rule": rows_dropped_per_rule,
        "dropped_leakage_columns": dropped_leakage,
        "remaining_features": [col for col in df.columns if col != TARGET_COL],
        "target_column": TARGET_COL,
    }

    logger.info(
        "Cleaning complete: %d in -> %d out (%d total dropped)",
        rows_in,
        rows_out,
        total_dropped,
    )
    return df, report


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(
        description="Deterministic data cleaning and leakage removal for taxi fare prediction."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=repo_root / "data" / "taxi_trip_pricing.csv",
        help="Path to the raw taxi trip pricing CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root / "data" / "processed" / "clean.csv",
        help="Path where the cleaned CSV will be written.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=repo_root / "data" / "processed" / "clean_report.json",
        help="Path where the JSON cleaning report will be written.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    input_path: Path = args.input
    output_path: Path = args.output
    report_path: Path = args.report

    # Fallback to current working directory if path does not exist
    if not input_path.exists():
        cwd_candidate = Path("data/taxi_trip_pricing.csv")
        if cwd_candidate.exists():
            input_path = cwd_candidate
        else:
            raise FileNotFoundError(f"Input file not found at {input_path}")

    logger.info("Reading raw dataset from: %s", input_path)
    raw_df = pd.read_csv(input_path)

    clean_df, report = clean_data(raw_df)

    # Ensure destination directories exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Saving cleaned data to: %s", output_path)
    clean_df.to_csv(output_path, index=False)

    logger.info("Saving cleaning report to: %s", report_path)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n--- Cleaning Report Summary ---")
    print(f"Rows in:              {report['rows_in']}")
    print(f"Rows out:             {report['rows_out']}")
    print(f"Rows dropped:         {report['rows_dropped_total']}")
    for rule, count in report["rows_dropped_per_rule"].items():
        print(f"  - {rule}: {count}")
    print(f"Dropped leakage:      {report['dropped_leakage_columns']}")


if __name__ == "__main__":
    main()