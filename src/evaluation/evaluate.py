import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from src.utils.config import TEST_DATA_PATH

ROOT_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT_DIR / "models" / "trained"
ARTIFACTS_DIR = ROOT_DIR / "models" / "artifacts"

LEAKAGE_COLUMNS = ["problem_detected", "risk_score", "failure_probability"]


def evaluate_saved_model(model_name="best_model.pkl"):
    test_df = pd.read_csv(TEST_DATA_PATH)

    X_test = test_df.drop(columns=["system_status"])
    X_test = X_test.drop(columns=[col for col in LEAKAGE_COLUMNS if col in X_test.columns])

    y_test = test_df["system_status"]

    model = joblib.load(MODELS_DIR / model_name)
    label_encoder = joblib.load(ARTIFACTS_DIR / "label_encoder.pkl")

    y_test_encoded = label_encoder.transform(y_test)
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test_encoded, y_pred),
        "weighted_f1": f1_score(y_test_encoded, y_pred, average="weighted"),
        "classification_report": classification_report(y_test_encoded, y_pred, output_dict=True),
        "confusion_matrix": confusion_matrix(y_test_encoded, y_pred).tolist(),
    }

    with open(ARTIFACTS_DIR / "test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print("Test Accuracy:", metrics["accuracy"])
    print("Test Weighted F1:", metrics["weighted_f1"])
    print("Confusion Matrix:")
    print(metrics["confusion_matrix"])

    return metrics


if __name__ == "__main__":
    evaluate_saved_model()
