import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.config import (
    FEATURED_DATA_PATH,
    TRAIN_DATA_PATH,
    VAL_DATA_PATH,
    TEST_DATA_PATH,
)


def split_data() -> None:
    df = pd.read_csv(FEATURED_DATA_PATH)

    X = df.drop(columns=["system_status"])
    y = df["system_status"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y,
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=42,
        stratify=y_temp,
    )

    train_df = X_train.copy()
    train_df["system_status"] = y_train.values

    val_df = X_val.copy()
    val_df["system_status"] = y_val.values

    test_df = X_test.copy()
    test_df["system_status"] = y_test.values

    train_df.to_csv(TRAIN_DATA_PATH, index=False)
    val_df.to_csv(VAL_DATA_PATH, index=False)
    test_df.to_csv(TEST_DATA_PATH, index=False)

    print("Data split completed.")
    print(f"Train shape: {train_df.shape}")
    print(f"Validation shape: {val_df.shape}")
    print(f"Test shape: {test_df.shape}")


if __name__ == "__main__":
    split_data()