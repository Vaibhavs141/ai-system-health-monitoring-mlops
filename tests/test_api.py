from fastapi.testclient import TestClient
from unittest.mock import patch

from api.app import app

client = TestClient(app)


def test_home():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


@patch("api.app.predict_system_health")  # keep this (assuming correct import in app.py)
def test_predict(mock_predict):
    mock_predict.return_value = {
        "prediction_label": "warning",
        "failure_probability": 0.42,
        "confidence": 0.85,  # ✅ REQUIRED FIX
        "class_probabilities": {
            "healthy": 0.20,
            "warning": 0.42,
            "critical": 0.38,
        },
    }

    payload = {
        "cpu_usage": 78,
        "memory_usage": 72,
        "temperature": 68,
        "voltage": 12.1,
        "disk_usage": 65,
        "fan_speed": 2400,
        "network_traffic": 85,
        "error_count": 3,
        "response_time": 260,
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    result = response.json()
    assert result["prediction_label"] == "warning"
    assert "failure_probability" in result
    assert "class_probabilities" in result
    assert "confidence" in result  # ✅ good practice
