import json
import sys

import numpy as np
import pandas as pd
import pytest

from src import data_quality


@pytest.fixture
def quality_data():
    row = {
        "Trip_Distance_km": 10.0,
        "Passenger_Count": 2.0,
        "Base_Fare": 3.0,
        "Per_Km_Rate": 1.5,
        "Per_Minute_Rate": 0.3,
        "Trip_Price": 30.0,
        "Time_of_Day": "Morning",
        "Day_of_Week": "Weekday",
        "Traffic_Conditions": "Low",
        "Weather": "Clear",
    }
    return pd.DataFrame([row.copy() for _ in range(100)])


def test_valid_data_passes(quality_data):
    assert data_quality.check_data(quality_data)["passed"] is True


def test_missing_column_fails(quality_data):
    report = data_quality.check_data(quality_data.drop(columns=["Weather"]))
    assert report["passed"] is False
    assert report["missing_columns"] == ["Weather"]


@pytest.mark.parametrize("column,value", [
    ("Trip_Distance_km", -5),
    ("Trip_Distance_km", np.inf),
    ("Base_Fare", -1),
    ("Per_Km_Rate", 0),
    ("Per_Minute_Rate", -1),
    ("Passenger_Count", 1.5),
    ("Passenger_Count", 7),
    ("Trip_Price", np.inf),
    ("Trip_Price", 0),
    ("Weather", "Unknown"),
])
def test_invalid_values_fail(quality_data, column, value):
    quality_data.loc[0, column] = value
    assert data_quality.check_data(quality_data)["passed"] is False


def test_small_missing_fraction_passes(quality_data):
    quality_data.loc[:4, "Base_Fare"] = np.nan
    report = data_quality.check_data(quality_data)
    assert report["passed"] is True
    assert report["columns"]["Base_Fare"]["missing_fraction"] == 0.05


def test_excessive_missing_values_fail(quality_data):
    quality_data.loc[:29, "Weather"] = np.nan
    assert data_quality.check_data(quality_data)["passed"] is False


def test_too_few_rows_fail(quality_data):
    assert data_quality.check_data(quality_data.iloc[:10])["passed"] is False


def test_duplicate_columns_fail(quality_data):
    frame = pd.concat([quality_data, quality_data[["Weather"]]], axis=1)
    assert data_quality.check_data(frame)["passed"] is False


def test_cli_failure_saves_report(quality_data, tmp_path, monkeypatch):
    quality_data.loc[0, "Weather"] = "Unknown"
    source = tmp_path / "bad.csv"
    report_path = tmp_path / "report.json"
    quality_data.to_csv(source, index=False)
    monkeypatch.setattr(sys, "argv", [
        "data_quality", "--input", str(source),
        "--report", str(report_path),
    ])

    with pytest.raises(SystemExit) as error:
        data_quality.main()

    assert error.value.code == 1
    assert json.loads(report_path.read_text())["passed"] is False
