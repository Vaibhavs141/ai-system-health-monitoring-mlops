from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT_DIR / "models" / "trained"
ARTIFACTS_DIR = ROOT_DIR / "models" / "artifacts"

MODEL_PATH = MODELS_DIR / "best_model.pkl"
LABEL_ENCODER_PATH = ARTIFACTS_DIR / "label_encoder.pkl"

model = joblib.load(MODEL_PATH)
label_encoder = joblib.load(LABEL_ENCODER_PATH)


def build_features(payload: dict) -> pd.DataFrame:
    """
    Build model input features exactly as required by training pipeline.
    """

    cpu_usage = payload["cpu_usage"]
    memory_usage = payload["memory_usage"]
    temperature = payload["temperature"]
    voltage = payload["voltage"]
    disk_usage = payload["disk_usage"]
    fan_speed = payload["fan_speed"]
    network_traffic = payload["network_traffic"]
    error_count = payload["error_count"]
    response_time = payload["response_time"]

    thermal_stress = (cpu_usage * temperature) / 100.0
    memory_pressure = (memory_usage * cpu_usage) / 100.0
    power_instability = abs(voltage - 12.0)
    cooling_efficiency = fan_speed / (temperature + 1)

    df = pd.DataFrame(
        [
            {
                "cpu_usage": cpu_usage,
                "memory_usage": memory_usage,
                "temperature": temperature,
                "voltage": voltage,
                "disk_usage": disk_usage,
                "fan_speed": fan_speed,
                "network_traffic": network_traffic,
                "error_count": error_count,
                "response_time": response_time,
                "thermal_stress": thermal_stress,
                "memory_pressure": memory_pressure,
                "power_instability": power_instability,
                "cooling_efficiency": cooling_efficiency,
            }
        ]
    )

    return df


def predict_system_health(payload: dict) -> dict:
    features = build_features(payload)

    pred_encoded = model.predict(features)[0]
    pred_label = label_encoder.inverse_transform([pred_encoded])[0]

    probas = model.predict_proba(features)[0]
    class_names = label_encoder.inverse_transform(np.arange(len(probas)))

    class_probabilities = {
        class_name: float(round(prob, 6))
        for class_name, prob in zip(class_names, probas)
    }

    failure_probability = class_probabilities.get("critical", 0.0)

    return {
        "prediction_label": pred_label,
        "failure_probability": round(float(failure_probability), 6),
        "class_probabilities": class_probabilities,
    }
