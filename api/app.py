import time

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST,
)

from api.schemas import PredictionRequest, PredictionResponse
from api.predictor import predict_system_health

app = FastAPI(
    title="AI-Based System Health Monitoring API",
    description="Predicts system health state using trained ML model",
    version="1.0.0",
)

# Prometheus metrics
REQUEST_COUNT = Counter(
    "api_requests_total",
    "Total number of prediction API requests"
)

PREDICTION_COUNT = Counter(
    "api_prediction_total",
    "Total number of predictions by class",
    ["label"]
)

PREDICTION_FAILURES = Counter(
    "api_prediction_failures_total",
    "Total number of failed prediction requests"
)

REQUEST_LATENCY = Histogram(
    "api_request_latency_seconds",
    "Latency of prediction requests in seconds"
)

FAILURE_PROBABILITY_GAUGE = Gauge(
    "model_failure_probability",
    "Latest predicted failure probability"
)

MODEL_CONFIDENCE_GAUGE = Gauge(
    "model_confidence",
    "Latest model confidence score"
)


@app.get("/")
def home():
    return {"message": "System Health Monitoring API is running"}


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    REQUEST_COUNT.inc()
    start_time = time.time()

    try:
        result = predict_system_health(request.model_dump())

        REQUEST_LATENCY.observe(time.time() - start_time)
        PREDICTION_COUNT.labels(label=result["prediction_label"]).inc()
        FAILURE_PROBABILITY_GAUGE.set(result["failure_probability"])
        MODEL_CONFIDENCE_GAUGE.set(result["confidence"])

        return PredictionResponse(**result)

    except Exception as e:
        PREDICTION_FAILURES.inc()
        raise HTTPException(status_code=500, detail=str(e))
