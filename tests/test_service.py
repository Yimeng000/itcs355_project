import pytest
from fastapi.testclient import TestClient

from service.app import app


VALID_INPUT = {
    "Trip_Distance_km": 8.5,
    "Passenger_Count": 2,
    "Base_Fare": 3.5,
    "Per_Km_Rate": 1.2,
    "Per_Minute_Rate": 0.3,
    "Time_of_Day": "Morning",
    "Day_of_Week": "Weekday",
    "Traffic_Conditions": "Medium",
    "Weather": "Clear",
}


@pytest.fixture
def client(monkeypatch, tmp_path):
    # Deliberately use a missing model for these tests.
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "missing.joblib"))
    with TestClient(app) as test_client:
        yield test_client


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "alive"}


def test_ready_without_model(client):
    assert client.get("/ready").status_code == 503


def test_predict_without_model(client):
    response = client.post("/predict", json=VALID_INPUT)
    assert response.status_code == 503
    assert response.json()["detail"] == "Model is not loaded yet"


def test_negative_distance_is_rejected(client):
    data = {**VALID_INPUT, "Trip_Distance_km": -5}
    response = client.post("/predict", json=data)
    assert response.status_code == 422
    assert any(
        error["loc"] == ["body", "Trip_Distance_km"]
        for error in response.json()["detail"]
    )


class ExampleModel:
    feature_names_in_ = list(VALID_INPUT)

    def predict(self, frame):
        assert list(frame.columns) == self.feature_names_in_
        expected = {
            **VALID_INPUT,
            "Time_of_Day": "morning",
            "Day_of_Week": "weekday",
            "Traffic_Conditions": "medium",
            "Weather": "clear",
        }
        assert frame.iloc[0].to_dict() == expected
        return [34.234]


def test_predict_with_model(client):
    app.state.model = ExampleModel()
    app.state.model_version = "test-v1"

    response = client.post("/predict", json=VALID_INPUT)

    assert response.status_code == 200
    assert response.json() == {
        "estimated_fare": 34.23,
        "model_version": "test-v1",
    }


def test_ready_with_model(client):
    app.state.model = ExampleModel()
    app.state.model_version = "test-v1"

    response = client.get("/ready")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "model_version": "test-v1",
    }

def test_invalid_input_burst_does_not_break_predictions(client):
    from unittest.mock import Mock

    model = ExampleModel()
    model.predict = Mock(wraps=model.predict)
    app.state.model = model
    app.state.model_version = "test-v1"

    invalid_input = {**VALID_INPUT, "Trip_Distance_km": -5}

    # Reproduce the deliberate failure: 20 invalid requests.
    for _ in range(20):
        response = client.post("/predict", json=invalid_input)
        assert response.status_code == 422
        assert any(
            error["loc"] == ["body", "Trip_Distance_km"]
            for error in response.json()["detail"]
        )

    # Invalid requests must be rejected before reaching the model.
    model.predict.assert_not_called()

    # The service must still be ready and accept valid requests.
    assert client.get("/health").status_code == 200
    assert client.get("/ready").status_code == 200

    response = client.post("/predict", json=VALID_INPUT)
    assert response.status_code == 200
    assert response.json() == {
        "estimated_fare": 34.23,
        "model_version": "test-v1",
    }
    model.predict.assert_called_once()
