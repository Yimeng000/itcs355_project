"""Tests for taxi data cleaning and splitting."""
import numpy as np
import pandas as pd
import pytest

from src.clean import clean_data
from src.split import split_data


@pytest.fixture
def raw_data():
    return pd.DataFrame([
        {
            "Trip_Distance_km": float(i + 1),
            "Passenger_Count": 2.0,
            "Base_Fare": 3.0,
            "Per_Km_Rate": 1.5,
            "Per_Minute_Rate": 0.3,
            "Time_of_Day": " Morning ",
            "Day_of_Week": "Weekday",
            "Traffic_Conditions": "Low",
            "Weather": "Clear",
            "Trip_Duration_Minutes": 10.0,
            "Trip_Price": float(10 + i * 2),
        }
        for i in range(40)
    ])


@pytest.mark.parametrize("field,value", [
    ("Trip_Price", np.nan),
    ("Trip_Price", 0),
    ("Trip_Price", -5),
    ("Trip_Distance_km", 0),
    ("Trip_Distance_km", -5),
    ("Passenger_Count", 0),
    ("Passenger_Count", 7),
])
def test_invalid_record_is_removed(raw_data, field, value):
    raw_data.loc[0, field] = value
    cleaned, report = clean_data(raw_data)

    assert len(cleaned) == len(raw_data) - 1
    assert report["rows_dropped_total"] == 1
    assert report["rows_in"] == len(raw_data)
    assert report["rows_out"] == len(cleaned)


def test_post_ride_duration_is_removed(raw_data):
    cleaned, report = clean_data(raw_data)

    assert "Trip_Duration_Minutes" not in cleaned.columns
    assert "Trip_Duration_Minutes" in report["dropped_leakage_columns"]


def test_categories_match_model_training_format(raw_data):
    cleaned, _ = clean_data(raw_data)

    assert set(cleaned["Time_of_Day"]) == {"morning"}
    assert set(cleaned["Day_of_Week"]) == {"weekday"}
    assert set(cleaned["Traffic_Conditions"]) == {"low"}
    assert set(cleaned["Weather"]) == {"clear"}


def test_cleaning_does_not_change_original_data(raw_data):
    original = raw_data.copy(deep=True)
    clean_data(raw_data)

    pd.testing.assert_frame_equal(raw_data, original)


def test_missing_input_is_kept_for_model_imputation(raw_data):
    raw_data.loc[0, "Base_Fare"] = np.nan
    cleaned, _ = clean_data(raw_data)

    assert len(cleaned) == len(raw_data)
    assert pd.isna(cleaned.loc[0, "Base_Fare"])


def test_split_has_no_overlap_and_loses_no_rows(raw_data):
    cleaned, _ = clean_data(raw_data)
    parts = split_data(cleaned, seed=42)
    identifiers = [set(part["Trip_Distance_km"]) for part in parts]

    assert not identifiers[0] & identifiers[1]
    assert not identifiers[0] & identifiers[2]
    assert not identifiers[1] & identifiers[2]
    assert set.union(*identifiers) == set(cleaned["Trip_Distance_km"])
    assert sum(len(part) for part in parts) == len(cleaned)


def test_same_seed_produces_same_split(raw_data):
    cleaned, _ = clean_data(raw_data)
    first = split_data(cleaned, seed=42)
    second = split_data(cleaned, seed=42)

    for a, b in zip(first, second):
        pd.testing.assert_frame_equal(a, b)


@pytest.mark.parametrize("val_size,test_size", [
    (0, 0.15),
    (0.15, -0.1),
    (0.6, 0.5),
])
def test_invalid_split_settings_are_rejected(raw_data, val_size, test_size):
    with pytest.raises(ValueError):
        split_data(raw_data, val_size=val_size, test_size=test_size)
