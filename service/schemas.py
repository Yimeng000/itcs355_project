from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    Trip_Distance_km: float = Field(
        gt=0, allow_inf_nan=False
    )
    Time_of_Day: Literal[
        "Morning", "Afternoon", "Evening", "Night"
    ]
    Day_of_Week: Literal["Weekday", "Weekend"]
    Passenger_Count: int = Field(
        ge=1, le=4, strict=True
    )
    Traffic_Conditions: Literal["Low", "Medium", "High"]
    Weather: Literal["Clear", "Rain", "Snow"]


class PredictResponse(BaseModel):
    estimated_fare: float = Field(
        ge=0, allow_inf_nan=False
    )
    model_version: str
