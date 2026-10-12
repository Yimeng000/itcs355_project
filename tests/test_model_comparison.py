import json
import shutil
import sys

import joblib
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from src import compare_models


@pytest.fixture
def comparison_files(tmp_path, monkeypatch):
    frame = pd.DataFrame({
        "Trip_Distance_km": [1.0, 2.0, 3.0, 4.0],
        "Passenger_Count": [1] * 4,
        "Base_Fare": [2.0] * 4,
        "Per_Km_Rate": [3.0] * 4,
        "Per_Minute_Rate": [0.3] * 4,
        "Time_of_Day": ["morning"] * 4,
        "Day_of_Week": ["weekday"] * 4,
        "Traffic_Conditions": ["low"] * 4,
        "Weather": ["clear"] * 4,
    })
    target = pd.Series([5.0, 8.0, 11.0, 14.0])

    reference = Pipeline([
        ("features", ColumnTransformer([
            ("distance", "passthrough", ["Trip_Distance_km"]),
        ])),
        ("model", LinearRegression()),
    ])
    reference.fit(frame, target)

    paths = {
        name: tmp_path / filename
        for name, filename in {
            "reference": "reference.joblib",
            "candidate": "candidate.joblib",
            "validation": "val.csv",
            "receipt": "registration.json",
            "training-record": "training.json",
            "output": "comparison.json",
        }.items()
    }
    joblib.dump(reference, paths["reference"])
    shutil.copy2(paths["reference"], paths["candidate"])
    frame.assign(Trip_Price=target).to_csv(paths["validation"], index=False)

    paths["receipt"].write_text(json.dumps({
        "version_id": "2",
        "lineage": {
            "model_sha256": compare_models.fingerprint(paths["reference"]),
        },
    }))
    paths["training-record"].write_text(json.dumps({
        "model_sha256": compare_models.fingerprint(paths["candidate"]),
        "training_run_id": "test-run",
        "training_git_commit": "test-commit",
    }))

    arguments = ["compare_models"]
    for name, path in paths.items():
        arguments.extend([f"--{name}", str(path)])
    monkeypatch.setattr(sys, "argv", arguments)
    return paths, frame, target


def test_matching_models_pass(comparison_files):
    paths, _, _ = comparison_files
    compare_models.main()
    report = json.loads(paths["output"].read_text())
    assert report["comparison_passed"] is True
    assert report["predictions_match_within_tolerance"] is True


def test_worse_model_fails_and_saves_report(comparison_files):
    paths, frame, target = comparison_files
    bad_model = DummyRegressor(strategy="constant", constant=1000.0)
    bad_model.fit(frame, target)
    joblib.dump(bad_model, paths["candidate"])

    training = json.loads(paths["training-record"].read_text())
    training["model_sha256"] = compare_models.fingerprint(paths["candidate"])
    paths["training-record"].write_text(json.dumps(training))

    with pytest.raises(SystemExit) as error:
        compare_models.main()

    assert error.value.code == 1
    report = json.loads(paths["output"].read_text())
    assert report["comparison_passed"] is False
    assert report["checks"]["candidate_mae_not_worse"] is False


@pytest.mark.parametrize("record_name", ["receipt", "training-record"])
def test_wrong_model_record_is_rejected(comparison_files, record_name):
    paths, _, _ = comparison_files
    record = json.loads(paths[record_name].read_text())
    if record_name == "receipt":
        record["lineage"]["model_sha256"] = "wrong"
    else:
        record["model_sha256"] = "wrong"
    paths[record_name].write_text(json.dumps(record))

    with pytest.raises(RuntimeError, match="does not match"):
        compare_models.main()


def test_invalid_validation_target_is_rejected(comparison_files):
    paths, _, _ = comparison_files
    frame = pd.read_csv(paths["validation"])
    frame.loc[0, "Trip_Price"] = float("inf")
    frame.to_csv(paths["validation"], index=False)

    with pytest.raises(ValueError, match="targets must be finite"):
        compare_models.main()
