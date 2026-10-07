from fastapi.testclient import TestClient

from service.app import app


client = TestClient(app)

VALID_INPUT = {
    "Trip_Distance_km": 8.5,
    "Time_of_Day": "Morning",
    "Day_of_Week": "Weekday",
    "Passenger_Count": 2,
    "Traffic_Conditions": "Medium",
    "Weather": "Clear",
    "Base_Fare": 3.5,
    "Per_Km_Rate": 1.2,
    "Per_Minute_Rate": 0.3,
}


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "alive"}


def test_ready_without_model():
    response = client.get("/ready")
    assert response.status_code == 503


def test_predict_without_model():
    response = client.post("/predict", json=VALID_INPUT)
    assert response.status_code == 503
    assert response.json()["detail"] == "Model is not loaded yet"


def test_negative_distance_is_rejected():
    data = {**VALID_INPUT, "Trip_Distance_km": -5}
    response = client.post("/predict", json=data)
    assert response.status_code == 422
    assert any(
        error["loc"] == ["body", "Trip_Distance_km"]
        for error in response.json()["detail"]
    )
