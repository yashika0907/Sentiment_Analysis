import pytest

from src.config import MODEL_PATH, VECTORIZER_PATH
from src.preprocessing import clean_text


def test_html_and_url_removed():
    out = clean_text("Great <br /> product http://example.com")
    assert "br" not in out.split() and "example" not in out


def test_negation_preserved():
    assert "not" in clean_text("I do not like this movie").split()


def test_contraction_expanded():
    assert "not" in clean_text("It wasn't good").split()


def test_empty_and_non_string_input():
    assert clean_text("") == ""
    assert clean_text(None) == ""
    assert clean_text("!!! ???") == ""


models_ready = MODEL_PATH.exists() and VECTORIZER_PATH.exists()


@pytest.mark.skipif(not models_ready, reason="Train the model first")
def test_model_loads_and_predicts():
    from src.predict import SentimentPredictor
    predictor = SentimentPredictor()
    result = predictor.predict("The product quality is terrible and disappointing.")
    assert result["label"] in ["Negative", "Neutral", "Positive"]
    assert abs(sum(result["probabilities"].values()) - 1) < 1e-6


@pytest.mark.skipif(not models_ready, reason="Train the model first")
def test_empty_input_raises():
    from src.predict import SentimentPredictor
    with pytest.raises(ValueError):
        SentimentPredictor().predict("   ")
