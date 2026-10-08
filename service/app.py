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

    from tempfile import TemporaryDirectory

    with TemporaryDirectory(prefix="taxi-model-") as directory:
        path = Path(os.getenv("MODEL_PATH", "models/model.joblib"))
        try:
            artifact_uri = os.getenv("MODEL_ARTIFACT_URI")
            if artifact_uri:
                from cloudlayer.artifacts import download_model

                path = Path(directory) / "model.joblib"
                download_model(artifact_uri, path)

            app.state.model = joblib.load(path)
            log.info("Model loaded, version=%s", app.state.model_version)
        except Exception:
            log.exception("Model could not be loaded")

        try:
            yield
        finally:
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