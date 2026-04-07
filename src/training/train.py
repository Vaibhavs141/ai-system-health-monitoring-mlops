import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.utils.config import TRAIN_DATA_PATH, VAL_DATA_PATH

ROOT_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT_DIR / "models" / "trained"
ARTIFACTS_DIR = ROOT_DIR / "models" / "artifacts"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

LEAKAGE_COLUMNS = ["problem_detected", "risk_score", "failure_probability"]
EXPERIMENT_NAME = "system_health_monitoring"


def load_data():
    train_df = pd.read_csv(TRAIN_DATA_PATH)
    val_df = pd.read_csv(VAL_DATA_PATH)

    X_train = train_df.drop(columns=["system_status"])
    X_val = val_df.drop(columns=["system_status"])

    X_train = X_train.drop(columns=[col for col in LEAKAGE_COLUMNS if col in X_train.columns])
    X_val = X_val.drop(columns=[col for col in LEAKAGE_COLUMNS if col in X_val.columns])

    y_train = train_df["system_status"]
    y_val = val_df["system_status"]

    return X_train, y_train, X_val, y_val


def build_preprocessor(X_train: pd.DataFrame):
    numeric_features = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
        ],
        remainder="drop",
    )
    return preprocessor


def evaluate_model(model, X_val, y_val):
    y_pred = model.predict(X_val)

    accuracy = accuracy_score(y_val, y_pred)
    weighted_f1 = f1_score(y_val, y_pred, average="weighted")
    report = classification_report(y_val, y_pred, output_dict=True)

    return {
        "accuracy": accuracy,
        "weighted_f1": weighted_f1,
        "classification_report": report,
    }


def log_to_mlflow(model_name, model, metrics, params, X_train):
    with mlflow.start_run(run_name=model_name):
        mlflow.log_params(params)
        mlflow.log_metric("accuracy", metrics["accuracy"])
        mlflow.log_metric("weighted_f1", metrics["weighted_f1"])
        mlflow.log_param("num_features", X_train.shape[1])

        # Save classification report as artifact
        report_path = ARTIFACTS_DIR / f"{model_name}_classification_report.json"
        with open(report_path, "w") as f:
            json.dump(metrics["classification_report"], f, indent=4)

        mlflow.log_artifact(str(report_path))

        # Log sklearn model
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path=model_name,
            registered_model_name=f"{model_name}_registry"
        )


def train_models():
    mlflow.set_experiment(EXPERIMENT_NAME)

    X_train, y_train, X_val, y_val = load_data()

    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_val_encoded = label_encoder.transform(y_val)

    preprocessor = build_preprocessor(X_train)

    # Logistic Regression
    logistic_params = {
        "model_type": "LogisticRegression",
        "max_iter": 1000,
        "multi_class": "multinomial",
        "class_weight": "balanced",
        "random_state": 42,
    }

    logistic_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    logistic_pipeline.fit(X_train, y_train_encoded)
    logistic_metrics = evaluate_model(logistic_pipeline, X_val, y_val_encoded)

    log_to_mlflow("logistic_regression", logistic_pipeline, logistic_metrics, logistic_params, X_train)

    # Random Forest
    rf_params = {
        "model_type": "RandomForestClassifier",
        "n_estimators": 300,
        "max_depth": 12,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "class_weight": "balanced_subsample",
        "random_state": 42,
    }

    rf_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=12,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    class_weight="balanced_subsample",
                    random_state=42,
                ),
            ),
        ]
    )

    rf_pipeline.fit(X_train, y_train_encoded)
    rf_metrics = evaluate_model(rf_pipeline, X_val, y_val_encoded)

    log_to_mlflow("random_forest", rf_pipeline, rf_metrics, rf_params, X_train)

    # Save artifacts locally too
    joblib.dump(logistic_pipeline, MODELS_DIR / "logistic_regression.pkl")
    joblib.dump(rf_pipeline, MODELS_DIR / "random_forest.pkl")
    joblib.dump(label_encoder, ARTIFACTS_DIR / "label_encoder.pkl")

    metrics_summary = {
        "logistic_regression": logistic_metrics,
        "random_forest": rf_metrics,
    }

    with open(ARTIFACTS_DIR / "metrics.json", "w") as f:
        json.dump(metrics_summary, f, indent=4)

    # Choose best model
    if logistic_metrics["weighted_f1"] >= rf_metrics["weighted_f1"]:
        best_model_name = "logistic_regression"
        best_model = logistic_pipeline
        best_metrics = logistic_metrics
    else:
        best_model_name = "random_forest"
        best_model = rf_pipeline
        best_metrics = rf_metrics

    joblib.dump(best_model, MODELS_DIR / "best_model.pkl")

    print("Training completed with MLflow logging.")
    print(f"Best model: {best_model_name}")
    print("Best Validation Accuracy:", best_metrics["accuracy"])
    print("Best Validation Weighted F1:", best_metrics["weighted_f1"])

    return metrics_summary


if __name__ == "__main__":
    train_models()
