import shutil
from pathlib import Path

from src.utils.config import RAW_DATA_PATH


def ingest_data(source_path: str) -> None:
    """
    Copy source dataset into project raw data directory.
    """
    source = Path(source_path)

    if not source.exists():
        raise FileNotFoundError(f"Source file not found: {source}")

    shutil.copy(source, RAW_DATA_PATH)
    print(f"Raw data copied to: {RAW_DATA_PATH}")


if __name__ == "__main__":
    source_file = "C:\\Users\\hp\\Downloads\\ai-system-health-monitoring\\src\\ingestion\\Laptop_Motherboard_Health_Monitoring_Dataset.csv"
    ingest_data(source_file)