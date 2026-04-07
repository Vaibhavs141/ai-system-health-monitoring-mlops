from fastapi import FastAPI

from api.schemas import PredictionRequest, PredictionResponse
from api.predictor import predict_system_health

app = FastAPI(
    title="AI-Based System Health Monitoring API",
    description="Predicts system health state using trained ML model",
    version="1.0.0",
)


@app.get("/")
def home():
    return {
        "message": "System Health Monitoring API is running"
    }


@app.get("/health")
def health_check():
    return {
        "status": "ok"
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    result = predict_system_health(request.model_dump())
    return PredictionResponse(**result)