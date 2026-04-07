from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.utils.config import TRAIN_DATA_PATH

ROOT_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT_DIR / "models" / "trained"
ARTIFACTS_DIR = ROOT_DIR / "models" / "artifacts"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

LEAKAGE_COLUMNS = ["problem_detected", "risk_score", "failure_probability"]
EXPERIMENT_NAME = "system_health_monitoring"


def tune_random_forest():
    mlflow.set_experiment(EXPERIMENT_NAME)

    train_df = pd.read_csv(TRAIN_DATA_PATH)

    X_train = train_df.drop(columns=["system_status"])
    X_train = X_train.drop(columns=[col for col in LEAKAGE_COLUMNS if col in X_train.columns])

    y_train = train_df["system_status"]

    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)

    numeric_features = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
        ],
        remainder="drop",
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    class_weight="balanced_subsample",
                    random_state=42,
                ),
            ),
        ]
    )

    param_dist = {
        "classifier__n_estimators": [100, 200, 300, 400],
        "classifier__max_depth": [5, 8, 10, 12, 15, None],
        "classifier__min_samples_split": [2, 5, 10],
        "classifier__min_samples_leaf": [1, 2, 4],
        "classifier__max_features": ["sqrt", "log2", None],
    }

    with mlflow.start_run(run_name="random_forest_tuning"):
        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_dist,
            n_iter=15,
            scoring="f1_weighted",
            cv=3,
            verbose=1,
            random_state=42,
            n_jobs=-1,
        )

        search.fit(X_train, y_train_encoded)

        mlflow.log_params(search.best_params_)
        mlflow.log_metric("best_cv_score", search.best_score_)

        mlflow.sklearn.log_model(
            sk_model=search.best_estimator_,
            artifact_path="random_forest_tuned",
            registered_model_name="random_forest_tuned_registry"
        )

        joblib.dump(search.best_estimator_, MODELS_DIR / "random_forest_tuned.pkl")
        joblib.dump(label_encoder, ARTIFACTS_DIR / "label_encoder.pkl")

        print("Best Parameters:", search.best_params_)
        print("Best CV Score:", search.best_score_)

        return search.best_estimator_, search.best_params_, search.best_score_


if __name__ == "__main__":
    tune_random_forest()