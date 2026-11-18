"""
Turn a star-rating review CSV into dataset/reviews.csv (text, label).

Default settings fit the Kaggle "Amazon Fine Food Reviews" file (Reviews.csv,
columns Text and Score). Mapping: 1-2 stars = Negative, 3 = Neutral, 4-5 = Positive.

Example:
    python -m src.prepare_dataset --input Reviews.csv --per-class 15000
"""
import argparse

import pandas as pd

from src.config import DATASET_PATH, LABEL_COLUMN, RANDOM_STATE, TEXT_COLUMN


def score_to_label(score):
    if score <= 2:
        return "Negative"
    if score == 3:
        return "Neutral"
    return "Positive"


def main():
    parser = argparse.ArgumentParser(description="Create dataset/reviews.csv")
    parser.add_argument("--input", required=True, help="Path to the raw CSV")
    parser.add_argument("--text-col", default="Text")
    parser.add_argument("--score-col", default="Score")
    parser.add_argument("--per-class", type=int, default=15000,
                        help="Max reviews kept per class (balanced sample)")
    args = parser.parse_args()

    try:
        raw = pd.read_csv(args.input)
    except FileNotFoundError:
        raise SystemExit(f"File not found: {args.input}")

    for col in (args.text_col, args.score_col):
        if col not in raw.columns:
            raise SystemExit(f"Column '{col}' not in file. Columns: {list(raw.columns)}")

    df = raw[[args.text_col, args.score_col]].dropna()
    df.columns = [TEXT_COLUMN, "score"]
    df = df.drop_duplicates(subset=TEXT_COLUMN)
    df[LABEL_COLUMN] = df["score"].apply(score_to_label)

    # Balanced sample: same number of reviews per class (or fewer if not enough)
    parts = []
    for label, group in df.groupby(LABEL_COLUMN):
        parts.append(group.sample(n=min(args.per_class, len(group)), random_state=RANDOM_STATE))
    final = pd.concat(parts).sample(frac=1, random_state=RANDOM_STATE)

    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    final[[TEXT_COLUMN, LABEL_COLUMN]].to_csv(DATASET_PATH, index=False)
    print(f"Saved {len(final)} reviews to {DATASET_PATH}")
    print(final[LABEL_COLUMN].value_counts())


if __name__ == "__main__":
    main()
