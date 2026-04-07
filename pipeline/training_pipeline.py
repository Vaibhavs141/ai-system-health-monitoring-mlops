from prefect import flow, task, get_run_logger

from src.validation.validate_data import validate_data
from src.preprocessing.preprocess_data import preprocess_data
from src.feature_engineering.feature_engineering import feature_engineering
from src.utils.common import split_data
from src.training.train import train_models
from src.training.tune_model import tune_random_forest
from src.evaluation.evaluate import evaluate_saved_model


@task(name="Validate Data", retries=2, retry_delay_seconds=5)
def validate_data_task():
    logger = get_run_logger()
    logger.info("Starting data validation task...")
    validate_data()
    logger.info("Data validation completed.")


@task(name="Preprocess Data", retries=2, retry_delay_seconds=5)
def preprocess_data_task():
    logger = get_run_logger()
    logger.info("Starting preprocessing task...")
    preprocess_data()
    logger.info("Preprocessing completed.")


@task(name="Feature Engineering", retries=2, retry_delay_seconds=5)
def feature_engineering_task():
    logger = get_run_logger()
    logger.info("Starting feature engineering task...")
    feature_engineering()
    logger.info("Feature engineering completed.")


@task(name="Split Data", retries=2, retry_delay_seconds=5)
def split_data_task():
    logger = get_run_logger()
    logger.info("Starting data split task...")
    split_data()
    logger.info("Data split completed.")


@task(name="Train Models", retries=2, retry_delay_seconds=5)
def train_models_task():
    logger = get_run_logger()
    logger.info("Starting model training task...")
    metrics = train_models()
    logger.info("Model training completed.")
    return metrics


@task(name="Tune Random Forest", retries=2, retry_delay_seconds=5)
def tune_model_task():
    logger = get_run_logger()
    logger.info("Starting random forest tuning task...")
    best_estimator, best_params, best_score = tune_random_forest()
    logger.info(f"Random forest tuning completed. Best CV score: {best_score}")
    return {
        "best_estimator": best_estimator,
        "best_params": best_params,
        "best_score": best_score,
    }


@task(name="Evaluate Best Model", retries=2, retry_delay_seconds=5)
def evaluate_model_task():
    logger = get_run_logger()
    logger.info("Starting evaluation task...")
    metrics = evaluate_saved_model()
    logger.info("Evaluation completed.")
    return metrics


@flow(name="System Health Monitoring Training Pipeline")
def training_pipeline():
    logger = get_run_logger()
    logger.info("Starting full Prefect pipeline for system health monitoring...")

    validate_data_task()
    preprocess_data_task()
    feature_engineering_task()
    split_data_task()

    training_metrics = train_models_task()
    tuning_metrics = tune_model_task()
    evaluation_metrics = evaluate_model_task()

    logger.info("Pipeline completed successfully.")

    return {
        "training_metrics": training_metrics,
        "tuning_metrics": tuning_metrics,
        "evaluation_metrics": evaluation_metrics,
    }


if __name__ == "__main__":
    training_pipeline()
