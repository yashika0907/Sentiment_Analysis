"""Loading and validating the dataset CSV (columns: text, label)."""
from pathlib import Path

import pandas as pd

from src.config import DATASET_PATH, LABEL_COLUMN, LABELS, TEXT_COLUMN


def load_dataset(path=DATASET_PATH):
    """Read the CSV, check columns, and remove empty / duplicate / invalid rows."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Follow the dataset steps in README.md "
            "to create dataset/reviews.csv."
        )

    df = pd.read_csv(path)
    missing = [c for c in (TEXT_COLUMN, LABEL_COLUMN) if c not in df.columns]
    if missing:
        raise ValueError(f"Missing column(s) {missing}. Found columns: {list(df.columns)}")

    df = df[[TEXT_COLUMN, LABEL_COLUMN]].dropna()
    df[TEXT_COLUMN] = df[TEXT_COLUMN].astype(str).str.strip()
    df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(str).str.strip().str.capitalize()

    df = df[df[TEXT_COLUMN] != ""]
    df = df[df[LABEL_COLUMN].isin(LABELS)]
    df = df.drop_duplicates(subset=TEXT_COLUMN).reset_index(drop=True)

    if df.empty:
        raise ValueError("No valid rows left after cleaning. Check labels: " + ", ".join(LABELS))
    return df
