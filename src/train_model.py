"""
Train, compare and save the sentiment models.

Run from the project root:
    python -m src.train_model

Data-leakage safeguards:
  * the dataset is split BEFORE the TF-IDF vectorizer is fitted
  * the vectorizer is fitted on the training set only
  * model selection uses the validation set; the test set is used only for the final report
"""
import json

import joblib
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from src.config import (LABEL_COLUMN, LABELS, METRICS_PATH, MODEL_PATH, RANDOM_STATE,
                        RESULTS_DIR, TEXT_COLUMN, VECTORIZER_PATH)
from src.data_loader import load_dataset
from src.evaluation import evaluate_predictions, save_confusion_matrix_png
from src.preprocessing import clean_text


def build_models():
    """The three algorithms. class_weight='balanced' helps if classes are uneven."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Multinomial Naive Bayes": MultinomialNB(),
        # LinearSVC has no probabilities, so we wrap it with calibration.
        "Support Vector Machine": CalibratedClassifierCV(
            LinearSVC(class_weight="balanced"), cv=3),
    }


def main():
    df = load_dataset()
    print(f"Loaded {len(df)} reviews")
    print(df[LABEL_COLUMN].value_counts(), "\n")

    print("Cleaning text (this can take a few minutes)...")
    df["clean_text"] = df[TEXT_COLUMN].apply(clean_text)
    df = df[df["clean_text"].str.len() > 0]

    # 70% train / 15% validation / 15% test, all stratified by label
    x_rest, x_test, y_rest, y_test = train_test_split(
        df["clean_text"], df[LABEL_COLUMN], test_size=0.15,
        stratify=df[LABEL_COLUMN], random_state=RANDOM_STATE)
    x_train, x_val, y_train, y_val = train_test_split(
        x_rest, y_rest, test_size=0.15 / 0.85,
        stratify=y_rest, random_state=RANDOM_STATE)

    # TF-IDF: fit on TRAIN only, then transform the other splits
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=50000,
                                 min_df=3, sublinear_tf=True)
    x_train_vec = vectorizer.fit_transform(x_train)
    x_val_vec = vectorizer.transform(x_val)
    x_test_vec = vectorizer.transform(x_test)

    models = build_models()
    val_results, test_results = {}, {}
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(x_train_vec, y_train)
        val_results[name] = evaluate_predictions(y_val, model.predict(x_val_vec), LABELS)
        test_results[name] = evaluate_predictions(y_test, model.predict(x_test_vec), LABELS)
        print(f"  validation macro-F1 = {val_results[name]['f1_macro']:.4f}")

    # Choose the best model by validation macro-F1 (not by test score)
    best_name = max(val_results, key=lambda n: val_results[n]["f1_macro"])
    print(f"\nBest model (validation macro-F1): {best_name}")

    RESULTS_DIR.mkdir(exist_ok=True)
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(models[best_name], MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)

    summary = {
        "labels": LABELS,
        "split_sizes": {"train": len(y_train), "validation": len(y_val), "test": len(y_test)},
        "class_distribution": df[LABEL_COLUMN].value_counts().to_dict(),
        "validation": val_results,
        "test": test_results,
        "best_model": best_name,
        "selection_rule": "highest macro F1 on the validation set",
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    save_confusion_matrix_png(test_results[best_name]["confusion_matrix"], LABELS,
                              f"{best_name} (test set)", RESULTS_DIR / "confusion_matrix.png")

    print("\nTest-set results")
    print(f"{'Model':28s} {'Acc':>7s} {'Prec':>7s} {'Rec':>7s} {'F1':>7s}")
    for name, m in test_results.items():
        print(f"{name:28s} {m['accuracy']:7.4f} {m['precision_macro']:7.4f} "
              f"{m['recall_macro']:7.4f} {m['f1_macro']:7.4f}")
    print(f"\nSaved: {MODEL_PATH.name}, {VECTORIZER_PATH.name}, {METRICS_PATH.name}")


if __name__ == "__main__":
    main()
