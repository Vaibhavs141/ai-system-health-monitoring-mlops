import pandas as pd
from pydantic import ValidationError

from src.utils.config import RAW_DATA_PATH, VALIDATED_DATA_PATH
from src.utils.schema import RawSystemHealthRecord


def validate_row(row: dict, row_index: int) -> None:
    """
    Validate a single row using Pydantic schema.
    """
    try:
        RawSystemHealthRecord(**row)
    except ValidationError as e:
        raise ValueError(f"Validation failed at row {row_index}: {e}")


def validate_data() -> None:
    df = pd.read_csv(RAW_DATA_PATH)

    print("Starting validation...")

    required_columns = [
        "CPUUsage",
        "RAMUsage",
        "Temperature",
        "Voltage",
        "DiskUsage",
        "FanSpeed",
        "ProblemDetected",
    ]

    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Remove duplicates
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    print(f"Removed duplicates: {before - after}")

    # Fill numeric missing values with median
    numeric_cols = df.select_dtypes(include=["number"]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    # Fill categorical missing values with mode
    categorical_cols = df.select_dtypes(exclude=["number"]).columns
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    # Pydantic validation row by row
    for idx, row in df.iterrows():
        validate_row(row.to_dict(), idx)

    df.to_csv(VALIDATED_DATA_PATH, index=False)
    print(f"Validated data saved to: {VALIDATED_DATA_PATH}")


if __name__ == "__main__":
    validate_data()