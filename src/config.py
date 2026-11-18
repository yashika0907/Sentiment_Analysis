"""Central place for paths and constants so no other file hardcodes them."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_PATH = PROJECT_ROOT / "dataset" / "reviews.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "sentiment_model.pkl"
VECTORIZER_PATH = PROJECT_ROOT / "models" / "tfidf_vectorizer.pkl"
RESULTS_DIR = PROJECT_ROOT / "results"
METRICS_PATH = RESULTS_DIR / "metrics.json"

TEXT_COLUMN = "text"
LABEL_COLUMN = "label"
LABELS = ["Negative", "Neutral", "Positive"]

RANDOM_STATE = 42
MAX_INPUT_CHARS = 5000
