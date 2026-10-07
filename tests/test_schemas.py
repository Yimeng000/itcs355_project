import pytest
from pydantic import ValidationError

from service.schemas import PredictRequest


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


def test_valid_input():
    request = PredictRequest(**VALID_INPUT)
    assert request.Trip_Distance_km == 8.5
    assert request.Passenger_Count == 2


@pytest.mark.parametrize(
    "field,value",
    [
        ("Trip_Distance_km", -5),
        ("Trip_Distance_km", 0),
        ("Passenger_Count", 0),
        ("Passenger_Count", 5),
        ("Passenger_Count", 1.5),
        ("Weather", "Unknown"),
        ("Time_of_Day", "Unknown"),
        ("Day_of_Week", "Unknown"),
        ("Traffic_Conditions", "Unknown"),
	("Base_Fare", -1),
        ("Per_Km_Rate", 0),
        ("Per_Km_Rate", -1),
        ("Per_Minute_Rate", -1),
    ],
)
def test_invalid_input_is_rejected(field, value):
    data = {**VALID_INPUT, field: value}
    with pytest.raises(ValidationError):
        PredictRequest(**data)


def test_missing_field_is_rejected():
    data = VALID_INPUT.copy()
    del data["Weather"]
    with pytest.raises(ValidationError):
        PredictRequest(**data)


def test_extra_field_is_rejected():
    data = {**VALID_INPUT, "Trip_Price": 20}
    with pytest.raises(ValidationError):
        PredictRequest(**data)
