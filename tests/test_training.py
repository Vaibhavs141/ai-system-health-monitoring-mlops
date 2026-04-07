import pandas as pd
from src.training.train import load_data
from src.utils.config import TRAIN_DATA_PATH, VAL_DATA_PATH


def test_load_data():
    train_df = pd.DataFrame({
        "cpu_usage": [10, 20, 30],
        "memory_usage": [40, 50, 60],
        "temperature": [45, 55, 65],
        "voltage": [12, 12, 12],
        "disk_usage": [30, 40, 50],
        "fan_speed": [1000, 1500, 2000],
        "network_traffic": [10, 20, 30],
        "error_count": [0, 1, 2],
        "response_time": [100, 150, 200],
        "thermal_stress": [4.5, 11.0, 19.5],
        "memory_pressure": [4.0, 10.0, 18.0],
        "power_instability": [0.0, 0.0, 0.0],
        "cooling_efficiency": [20.0, 26.7, 30.3],
        "system_status": ["healthy", "warning", "critical"]
    })

    val_df = train_df.copy()

    TRAIN_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(TRAIN_DATA_PATH, index=False)
    val_df.to_csv(VAL_DATA_PATH, index=False)

    X_train, y_train, X_val, y_val = load_data()

    assert len(X_train) == 3
    assert len(y_train) == 3
    assert len(X_val) == 3
    assert len(y_val) == 3
