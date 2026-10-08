"""Small model tests using the project's real preprocessing code."""
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

from src.features import ALL_FEATURES, build_preprocessor


@pytest.fixture(scope="module")
def fitted_model():
    rows = pd.DataFrame([
        {
            "Trip_Distance_km": float(i + 1),
            "Passenger_Count": float(1 + i % 4),
            "Base_Fare": 3.0,
            "Per_Km_Rate": 1.5,
            "Per_Minute_Rate": 0.3,
            "Time_of_Day": "morning" if i % 2 else "night",
            "Day_of_Week": "weekday" if i % 3 else "weekend",
            "Traffic_Conditions": "low" if i % 2 else "high",
            "Weather": "clear" if i % 3 else "rain",
        }
        for i in range(40)
    ])[ALL_FEATURES]

    fares = np.array([10.0 + i * 2 for i in range(40)])
    model = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("regressor", RandomForestRegressor(
            n_estimators=20,
            random_state=42,
            n_jobs=1,
        )),
    ])
    model.fit(rows, fares)
    return model, rows


def test_predictions_are_finite_nonnegative_fares(fitted_model):
    model, rows = fitted_model
    predictions = model.predict(rows)

    assert predictions.shape == (len(rows),)
    assert np.isfinite(predictions).all()
    assert (predictions >= 0).all()


def test_same_input_produces_same_prediction(fitted_model):
    model, rows = fitted_model

    np.testing.assert_allclose(
        model.predict(rows.head(3)),
        model.predict(rows.head(3)),
        rtol=0,
        atol=0,
    )


def test_saved_model_preserves_predictions(fitted_model, tmp_path):
    model, rows = fitted_model
    path = tmp_path / "model.joblib"
    expected = model.predict(rows.head(5))

    joblib.dump(model, path)
    restored = joblib.load(path)

    np.testing.assert_allclose(
        restored.predict(rows.head(5)),
        expected,
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("field", ["Base_Fare", "Weather"])
def test_pipeline_handles_missing_input_values(fitted_model, field):
    model, rows = fitted_model
    sample = rows.head(1).copy()
    sample[field] = np.nan

    prediction = model.predict(sample)

    assert np.isfinite(prediction).all()
    assert (prediction >= 0).all()
