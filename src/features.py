"""Feature definitions, pipeline preprocessing builder, and schema export."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET: str = "Trip_Price"

NUMERIC_FEATURES: list[str] = [
    "Trip_Distance_km",
    "Passenger_Count",
    "Base_Fare",
    "Per_Km_Rate",
    "Per_Minute_Rate",
]

CATEGORICAL_FEATURES: list[str] = [
    "Time_of_Day",
    "Day_of_Week",
    "Traffic_Conditions",
    "Weather",
]

ALL_FEATURES: list[str] = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def build_preprocessor(with_scaler: bool = True) -> ColumnTransformer:
    """Build a ColumnTransformer for numeric and categorical feature preprocessing.

    - Numeric: SimpleImputer(strategy='median') + optional StandardScaler()
    - Categorical: SimpleImputer(strategy='most_frequent') + OneHotEncoder(handle_unknown='ignore')
    """
    num_steps: list[tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy="median")),
    ]
    if with_scaler:
        num_steps.append(("scaler", StandardScaler()))

    numeric_transformer = Pipeline(steps=num_steps)

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def extract_input_schema(df: pd.DataFrame) -> dict[str, Any]:
    """Extract input feature names and dtypes for downstream API payload validation."""
    features_schema: dict[str, str] = {}
    for col in ALL_FEATURES:
        if col in df.columns:
            features_schema[col] = str(df[col].dtype)
        else:
            features_schema[col] = "float64" if col in NUMERIC_FEATURES else "object"

    return {
        "target": TARGET,
        "features": features_schema,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
    }


def save_input_schema(schema: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(schema, indent=2), encoding="utf-8")