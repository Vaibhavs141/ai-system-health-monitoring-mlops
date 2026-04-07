import pandas as pd

from src.utils.config import VALIDATED_DATA_PATH, PREPROCESSED_DATA_PATH


def preprocess_data() -> None:
    df = pd.read_csv(VALIDATED_DATA_PATH)

    print("Starting preprocessing...")

    # Drop noisy identity column if present
    if "ModelName" in df.columns:
        df = df.drop(columns=["ModelName"])

    # Standardize column names
    rename_map = {
        "CPUUsage": "cpu_usage",
        "RAMUsage": "memory_usage",
        "Temperature": "temperature",
        "Voltage": "voltage",
        "DiskUsage": "disk_usage",
        "FanSpeed": "fan_speed",
        "ProblemDetected": "problem_detected",
    }

    df = df.rename(columns=rename_map)

    # Basic cleanup
    numeric_cols = [
        "cpu_usage",
        "memory_usage",
        "temperature",
        "voltage",
        "disk_usage",
        "fan_speed",
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    # Clip values to realistic ranges
    df["cpu_usage"] = df["cpu_usage"].clip(0, 100)
    df["memory_usage"] = df["memory_usage"].clip(0, 100)
    df["disk_usage"] = df["disk_usage"].clip(0, 100)
    df["temperature"] = df["temperature"].clip(0, 150)
    df["voltage"] = df["voltage"].clip(0, 20)
    df["fan_speed"] = df["fan_speed"].clip(lower=0)

    # Normalize string field
    if "problem_detected" in df.columns:
        df["problem_detected"] = df["problem_detected"].astype(str).str.strip()

    df.to_csv(PREPROCESSED_DATA_PATH, index=False)
    print(f"Preprocessed data saved to: {PREPROCESSED_DATA_PATH}")


if __name__ == "__main__":
    preprocess_data()