import numpy as np
import pandas as pd

from src.utils.config import PREPROCESSED_DATA_PATH, FEATURED_DATA_PATH


def feature_engineering() -> None:
    df = pd.read_csv(PREPROCESSED_DATA_PATH)

    print("Starting feature engineering...")
    np.random.seed(42)

    # Simulated operational metrics
    df["network_traffic"] = np.random.exponential(scale=50, size=len(df))
    df["error_count"] = np.random.poisson(lam=2, size=len(df))
    df["response_time"] = np.clip(
        np.random.normal(loc=220, scale=60, size=len(df)),
        50,
        1000,
    )

    # Derived features
    df["thermal_stress"] = (df["cpu_usage"] * df["temperature"]) / 100.0
    df["memory_pressure"] = (df["memory_usage"] * df["cpu_usage"]) / 100.0
    df["power_instability"] = abs(df["voltage"] - df["voltage"].median())
    df["cooling_efficiency"] = df["fan_speed"] / (df["temperature"] + 1)

    # Normalized engineered risk inputs
    network_norm = np.clip(df["network_traffic"] / 200, 0, 1) * 100
    error_scaled = np.clip(df["error_count"] / 10, 0, 1) * 100
    response_scaled = np.clip(df["response_time"] / 500, 0, 1) * 100
    thermal_scaled = np.clip(df["thermal_stress"] / 100, 0, 1) * 100
    power_scaled = np.clip(df["power_instability"] / 5, 0, 1) * 100

    # Risk score
    df["risk_score"] = (
        0.18 * df["cpu_usage"]
        + 0.16 * df["memory_usage"]
        + 0.12 * df["disk_usage"]
        + 0.16 * df["temperature"]
        + 0.10 * network_norm
        + 0.10 * error_scaled
        + 0.10 * response_scaled
        + 0.05 * thermal_scaled
        + 0.03 * power_scaled
    ).clip(0, 100)

    df["failure_probability"] = (df["risk_score"] / 100).clip(0, 1)

    # Quantile-based class creation for balanced learning
    q1 = df["risk_score"].quantile(0.33)
    q2 = df["risk_score"].quantile(0.66)

    df["system_status"] = pd.cut(
        df["risk_score"],
        bins=[-np.inf, q1, q2, np.inf],
        labels=["healthy", "warning", "critical"],
        include_lowest=True,
    )

    df = df.dropna(subset=["system_status"]).copy()
    df["system_status"] = df["system_status"].astype(str)

    print(f"Quantile thresholds used: q1={q1:.2f}, q2={q2:.2f}")
    print("Class distribution after target engineering:")
    print(df["system_status"].value_counts(normalize=True))

    df.to_csv(FEATURED_DATA_PATH, index=False)
    print(f"Featured data saved to: {FEATURED_DATA_PATH}")


if __name__ == "__main__":
    feature_engineering()
