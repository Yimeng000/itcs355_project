import logging
import math
import os
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from service.schemas import PredictRequest, PredictResponse


log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = None
    app.state.model_version = os.getenv("MODEL_VERSION", "local")

    path = Path(os.getenv("MODEL_PATH", "models/model.joblib"))
    try:
        app.state.model = joblib.load(path)
        log.info("Model loaded from %s", path)
    except Exception:
        log.exception("Model could not be loaded from %s", path)

    yield
    app.state.model = None


app = FastAPI(
    title="Taxi Fare Prediction",
    version="0.2.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {"status": "alive"}


@app.get("/ready")
def ready():
    if getattr(app.state, "model", None) is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded yet",
        )
    return {
        "status": "ready",
        "model_version": app.state.model_version,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    model = getattr(app.state, "model", None)
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded yet",
        )

    try:
        row = payload.model_dump()
        for field in (
            "Time_of_Day",
            "Day_of_Week",
            "Traffic_Conditions",
            "Weather",
        ):
            row[field] = row[field].lower()

        frame = pd.DataFrame([row])
        frame = frame[list(model.feature_names_in_)]
        fare = float(model.predict(frame)[0])

        if not math.isfinite(fare) or fare < 0:
            raise ValueError("Model returned an invalid fare")

    except Exception:
        log.exception("Fare prediction failed")
        raise HTTPException(
            status_code=500,
            detail="Unable to estimate fare",
        )

    return PredictResponse(
        estimated_fare=round(fare, 2),
        model_version=app.state.model_version,
    )