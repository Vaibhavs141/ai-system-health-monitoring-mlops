from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    cpu_usage: float = Field(..., ge=0, le=100)
    memory_usage: float = Field(..., ge=0, le=100)
    temperature: float = Field(..., ge=0, le=150)
    voltage: float = Field(..., ge=0, le=20)
    disk_usage: float = Field(..., ge=0, le=100)
    fan_speed: float = Field(..., ge=0)
    network_traffic: float = Field(..., ge=0)
    error_count: int = Field(..., ge=0)
    response_time: float = Field(..., ge=0)


class PredictionResponse(BaseModel):
    prediction_label: str
    failure_probability: float
    confidence: float
    class_probabilities: dict
