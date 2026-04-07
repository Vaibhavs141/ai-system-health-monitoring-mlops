from pathlib import Path

# Root directory
ROOT_DIR = Path(__file__).resolve().parents[2]

# Data paths
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

RAW_DATA_PATH = RAW_DIR / "system_health.csv"
VALIDATED_DATA_PATH = INTERIM_DIR / "validated_system_health.csv"
PREPROCESSED_DATA_PATH = INTERIM_DIR / "preprocessed_system_health.csv"
FEATURED_DATA_PATH = INTERIM_DIR / "featured_system_health.csv"

TRAIN_DATA_PATH = PROCESSED_DIR / "train.csv"
VAL_DATA_PATH = PROCESSED_DIR / "val.csv"
TEST_DATA_PATH = PROCESSED_DIR / "test.csv"

# Create folders if they do not exist
for path in [RAW_DIR, INTERIM_DIR, PROCESSED_DIR]:
    path.mkdir(parents=True, exist_ok=True)