from fastapi.testclient import TestClient

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


def test_predict():
    payload = {
        "cpu_usage": 78,
        "memory_usage": 72,
        "temperature": 68,
        "voltage": 12.1,
        "disk_usage": 65,
        "fan_speed": 2400,
        "network_traffic": 85,
        "error_count": 3,
        "response_time": 260
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200

    result = response.json()
    assert "prediction_label" in result
    assert "failure_probability" in result
    assert "class_probabilities" in result