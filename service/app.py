from fastapi import FastAPI, HTTPException

from service.schemas import PredictRequest, PredictResponse


app = FastAPI(title="Taxi Fare Prediction", version="0.1.0")


@app.get("/health")
def health():
    return {"status": "alive"}


@app.get("/ready")
def ready():
    raise HTTPException(
        status_code=503,
        detail="Model is not loaded yet",
    )


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    raise HTTPException(
        status_code=503,
        detail="Model is not loaded yet",
    )
