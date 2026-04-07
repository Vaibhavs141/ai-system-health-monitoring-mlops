import pandas as pd
from pathlib import Path

from src.preprocessing.preprocess_data import preprocess_data
from src.utils.config import VALIDATED_DATA_PATH, PREPROCESSED_DATA_PATH


def test_preprocess_data_creates_output():
    sample_df = pd.DataFrame({
        "CPUUsage": [50],
        "RAMUsage": [60],
        "Temperature": [70],
        "Voltage": [12],
        "DiskUsage": [40],
        "FanSpeed": [2000],
        "ProblemDetected": ["No Problem"]
    })

    VALIDATED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    sample_df.to_csv(VALIDATED_DATA_PATH, index=False)

    preprocess_data()

    assert Path(PREPROCESSED_DATA_PATH).exists()

    output_df = pd.read_csv(PREPROCESSED_DATA_PATH)
    assert "cpu_usage" in output_df.columns
    assert "memory_usage" in output_df.columns