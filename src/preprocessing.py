"""
Text preprocessing for SentiScope.

Key design choice: negation words (not, never, no ...) are KEPT, because
"not good" and "good" have opposite sentiment.
"""
import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Words that flip or weaken sentiment - never removed.
NEGATION_WORDS = {"no", "not", "nor", "never", "none", "nothing", "neither",
                  "cannot", "without", "hardly", "barely"}
# Other small words that carry sentiment strength / contrast.
KEEP_WORDS = {"but", "very", "too", "only", "more", "most"}

# Order matters: irregular forms first, then the general "n't" rule.
CONTRACTIONS = [
    (r"won't", "will not"),
    (r"can't", "can not"),
    (r"n't", " not"),
    (r"'re", " are"),
    (r"'ve", " have"),
    (r"'ll", " will"),
    (r"'m", " am"),
    (r"'d", " would"),
]

HTML_PATTERN = re.compile(r"<[^>]+>")
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
NON_LETTER_PATTERN = re.compile(r"[^a-z\s]")
SPACE_PATTERN = re.compile(r"\s+")

_lemmatizer = None
_stop_words = None


def ensure_nltk_resources():
    """Download the NLTK data we need (only the first time)."""
    for resource, path in [("stopwords", "corpora/stopwords"),
                           ("wordnet", "corpora/wordnet")]:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(resource, quiet=True)


def _get_tools():
    """Load stopwords and lemmatizer once and reuse them (faster)."""
    global _lemmatizer, _stop_words
    if _lemmatizer is None:
        ensure_nltk_resources()
        _lemmatizer = WordNetLemmatizer()
        _stop_words = (set(stopwords.words("english")) - NEGATION_WORDS) - KEEP_WORDS
    return _lemmatizer, _stop_words


def clean_text(text):
    """Convert a raw review into a cleaned string ready for TF-IDF."""
    if not isinstance(text, str):
        return ""
    lemmatizer, stop_words = _get_tools()

    text = text.lower()
    text = HTML_PATTERN.sub(" ", text)          # remove <br /> etc.
    text = URL_PATTERN.sub(" ", text)           # remove links
    for pattern, replacement in CONTRACTIONS:   # don't -> do not
        text = re.sub(pattern, replacement, text)
    text = NON_LETTER_PATTERN.sub(" ", text)    # drop digits and punctuation
    text = SPACE_PATTERN.sub(" ", text).strip() # collapse extra spaces

    tokens = []
    for token in text.split():                  # simple whitespace tokenization
        if len(token) < 2 or token in stop_words:
            continue
        token = lemmatizer.lemmatize(token)            # movies -> movie
        token = lemmatizer.lemmatize(token, pos="v")   # loved -> love
        tokens.append(token)
    return " ".join(tokens)


if __name__ == "__main__":
    examples = [
        "I do NOT like this movie!!! <br /> Visit www.example.com",
        "The product wasn't bad, but delivery was very late.",
        "Wonderful, another two hours wasted.",
    ]
    for ex in examples:
        print("Before:", ex)
        print("After :", clean_text(ex))
        print()
