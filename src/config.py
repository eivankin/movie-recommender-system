from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

DATA_PATH = PROJECT_ROOT / "data"
DATASET_PATH = DATA_PATH / "raw" / "ml-100k" / "ml-100k"

SEED = 42
