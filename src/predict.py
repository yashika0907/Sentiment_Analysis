"""Load the saved model and predict sentiment for new text."""
import joblib
import pandas as pd

from src.config import MAX_INPUT_CHARS, MODEL_PATH, VECTORIZER_PATH
from src.preprocessing import clean_text


class SentimentPredictor:
    def __init__(self, model_path=MODEL_PATH, vectorizer_path=VECTORIZER_PATH):
        for path in (model_path, vectorizer_path):
            if not path.exists():
                raise FileNotFoundError(
                    f"{path.name} not found. Run 'python -m src.train_model' first.")
        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)

    def predict(self, text):
        """Return {'label', 'confidence', 'probabilities'} for one review."""
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Please enter some text.")
        cleaned = clean_text(text[:MAX_INPUT_CHARS])
        if not cleaned:
            raise ValueError("The text has no usable words after cleaning.")

        features = self.vectorizer.transform([cleaned])
        probs = self.model.predict_proba(features)[0]
        probabilities = dict(zip(self.model.classes_, map(float, probs)))
        label = max(probabilities, key=probabilities.get)
        return {"label": label, "confidence": probabilities[label],
                "probabilities": probabilities}

    def predict_batch(self, texts):
        """Predict for many texts. Rows with unusable text get label 'Unknown'."""
        cleaned = [clean_text(str(t)[:MAX_INPUT_CHARS]) if pd.notna(t) else "" for t in texts]
        result = pd.DataFrame({"predicted_sentiment": ["Unknown"] * len(cleaned),
                               "confidence": [None] * len(cleaned)})
        valid_idx = [i for i, c in enumerate(cleaned) if c]
        if valid_idx:
            features = self.vectorizer.transform([cleaned[i] for i in valid_idx])
            probs = self.model.predict_proba(features)
            best = probs.argmax(axis=1)
            result.loc[valid_idx, "predicted_sentiment"] = list(self.model.classes_[best])
            result.loc[valid_idx, "confidence"] = list((probs.max(axis=1) * 100).round(2))
        return result
