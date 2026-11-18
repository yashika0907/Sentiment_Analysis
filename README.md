# SentiScope - Intelligent Sentiment Analysis System

Three-class (Positive / Negative / Neutral) sentiment analysis using NLTK, TF-IDF,
Scikit-learn and Streamlit.

## 1. Setup (VS Code terminal, from the project folder)

    python -m venv venv
    venv\Scripts\activate          # Windows
    source venv/bin/activate       # macOS / Linux
    pip install -r requirements.txt

## 2. Dataset

Recommended: **Amazon Fine Food Reviews** (Kaggle, `snap/amazon-fine-food-reviews`).
It has a 1-5 star `Score` and a `Text` column. Check the licence shown on the Kaggle
page before use and credit it in your report.

1. Download and unzip it so you have `Reviews.csv` (put it anywhere, e.g. the project folder).
2. Convert stars to labels (1-2 = Negative, 3 = Neutral, 4-5 = Positive) and make a balanced sample:

       python -m src.prepare_dataset --input Reviews.csv --per-class 15000

   This creates `dataset/reviews.csv` with columns `text,label`.

Using another dataset? Make a CSV with `text` and `label` columns (labels: Positive, Negative, Neutral)
and save it as `dataset/reviews.csv`.

Note: star-rating labels are an approximation of sentiment. Mention this in your limitations.

## 3. Run

    python -m src.preprocessing      # see before/after text cleaning
    python -m src.train_model        # trains 3 models, saves best one
    pytest                           # basic tests
    streamlit run app.py             # opens the web app

The first run downloads small NLTK files (stopwords, wordnet) automatically.

## 4. Folder guide

| Path | Purpose |
|---|---|
| `src/config.py` | paths and constants |
| `src/data_loader.py` | load, validate, de-duplicate the CSV |
| `src/prepare_dataset.py` | star ratings to sentiment labels |
| `src/preprocessing.py` | text cleaning (keeps negations) |
| `src/train_model.py` | split, TF-IDF, train, compare, save |
| `src/evaluation.py` | metrics and charts |
| `src/predict.py` | load model and predict |
| `app.py` | Streamlit interface |
| `tests/` | pytest tests |

## 5. Common errors

- **`ModuleNotFoundError: src`** - run commands from the project root, using `python -m src.train_model`.
- **`Dataset not found`** - complete step 2.
- **`LookupError` from NLTK** - run `python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet')"`.
- **Model files not found in the app** - run `python -m src.train_model` first.
- **Training is slow** - lower `--per-class` (e.g. 5000) when preparing the dataset.

## 6. Results

Results are produced by your own training run and saved in `results/metrics.json`.
Copy numbers from there into your report; do not use any numbers that you did not generate.
