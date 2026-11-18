"""SentiScope - Intelligent Sentiment Analysis System (Streamlit front end).

Run with:  streamlit run app.py
"""
import json

import pandas as pd
import plotly.express as px
import streamlit as st

from src.config import LABELS, MAX_INPUT_CHARS, METRICS_PATH
from src.data_loader import load_dataset
from src.evaluation import comparison_figure, confusion_matrix_figure
from src.predict import SentimentPredictor

st.set_page_config(page_title="SentiScope", page_icon="📝", layout="wide")

# One colour per sentiment (green / red / neutral grey-blue)
COLORS = {"Positive": "#2e7d32", "Negative": "#c62828", "Neutral": "#546e7a"}
ICONS = {"Positive": "😊", "Negative": "😞", "Neutral": "😐"}

st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');

        html, body, [class*="css"] { font-family: 'Manrope', sans-serif; }
        .stApp {
            background:
                radial-gradient(circle at 90% 5%, rgba(99, 102, 241, 0.10), transparent 24rem),
                linear-gradient(135deg, #f8fafc 0%, #f3f6fb 100%);
        }
        [data-testid="stHeader"] { background: transparent; }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #111827 0%, #1e293b 100%);
            border-right: 1px solid rgba(255,255,255,0.06);
        }
        [data-testid="stSidebar"] * { color: #e5e7eb; }
        [data-testid="stSidebar"] [role="radiogroup"] label {
            border-radius: 10px;
            padding: 8px 10px;
            margin: 3px 0;
        }
        [data-testid="stSidebar"] [role="radiogroup"] label:hover {
            background: rgba(255,255,255,0.07);
        }
        [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
            background: linear-gradient(135deg, #6366f1, #818cf8);
            box-shadow: 0 8px 22px rgba(99,102,241,.24);
        }
        [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) * {
            color: white;
        }
        .block-container {
            max-width: 1180px;
            padding-top: 2.2rem;
            padding-bottom: 3rem;
        }
        h1, h2, h3 { color: #172033; letter-spacing: -0.025em; }
        h1 { font-weight: 800; }
        p, li, label { color: #475569; }
        .hero {
            padding: 38px 42px;
            border-radius: 24px;
            color: white;
            background: linear-gradient(135deg, #312e81 0%, #6366f1 58%, #8b5cf6 100%);
            box-shadow: 0 18px 45px rgba(79,70,229,.20);
            margin-bottom: 22px;
        }
        .hero h1 { color: white; margin: 0 0 8px 0; }
        .hero p { color: #e0e7ff; max-width: 760px; font-size: 1.02rem; }
        .page-header {
            display: flex;
            align-items: flex-end;
            justify-content: space-between;
            margin: 4px 0 22px 0;
        }
        .page-header h1 { margin: 0; }
        .page-header p { margin: 6px 0 0 0; }
        .surface {
            background: rgba(255,255,255,.92);
            border: 1px solid #e8edf5;
            border-radius: 18px;
            padding: 22px;
            box-shadow: 0 8px 28px rgba(15,23,42,.055);
        }
        [data-testid="stMetric"] {
            background: white;
            border: 1px solid #e8edf5;
            border-radius: 16px;
            padding: 16px 18px;
            box-shadow: 0 8px 24px rgba(15,23,42,.045);
        }
        [data-testid="stMetricLabel"] { color: #64748b; }
        [data-testid="stMetricValue"] { color: #172033; font-weight: 800; }
        .stTextArea textarea, .stTextInput input, [data-baseweb="select"] {
            border-radius: 12px !important;
            border-color: #dfe6f1 !important;
        }
        .stButton > button, .stDownloadButton > button {
            border-radius: 10px;
            border: 1px solid #dbe3ef;
            padding: 0.55rem 1rem;
            font-weight: 700;
            transition: all .18s ease;
        }
        .stButton > button[kind="primary"], .stDownloadButton > button {
            border: none;
            background: linear-gradient(135deg, #4f46e5, #7c3aed);
            color: white;
            box-shadow: 0 8px 18px rgba(79,70,229,.18);
        }
        .stButton > button:hover, .stDownloadButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 10px 22px rgba(15,23,42,.12);
        }
        [data-testid="stFileUploader"] section {
            background: white;
            border: 1px dashed #cbd5e1;
            border-radius: 16px;
        }
        [data-testid="stDataFrame"] {
            border: 1px solid #e8edf5;
            border-radius: 14px;
            overflow: hidden;
        }
        .info-card {
            background: white;
            border: 1px solid #e8edf5;
            border-radius: 16px;
            padding: 18px 20px;
            height: 100%;
        }
        .info-card b { color: #4f46e5; }
        hr { border-color: #e8edf5; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_predictor():
    return SentimentPredictor()


def load_predictor_or_stop():
    """Show a friendly message instead of a crash when the model is missing."""
    try:
        return get_predictor()
    except FileNotFoundError as error:
        st.error(str(error))
        st.stop()


def clear_text():
    st.session_state["review_text"] = ""


def page_header(title, subtitle):
    st.markdown(
        f'<div class="page-header"><div><h1>{title}</h1>'
        f'<p>{subtitle}</p></div></div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------- Home
def home_page():
    st.markdown(
        '<div class="hero"><h1>SentiScope</h1>'
        '<p>Turn customer language into clear, actionable sentiment insights with '
        'a fast, machine-learning-powered review classifier.</p></div>',
        unsafe_allow_html=True,
    )
    predictor = load_predictor_or_stop()

    left, right = st.columns([1.55, 0.45], gap="large")
    with left:
        st.markdown('<div class="surface">', unsafe_allow_html=True)
        st.subheader("Analyze a review")
        st.text_area("Enter your review", key="review_text", height=170,
                     max_chars=MAX_INPUT_CHARS,
                     placeholder="Example: The delivery was quick, but the quality was disappointing.")
        col_predict, col_clear, _ = st.columns([1.15, 0.85, 4])
        predict_clicked = col_predict.button("Analyze sentiment", type="primary",
                                             use_container_width=True)
        col_clear.button("Clear", on_click=clear_text, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown(
            '<div class="surface"><h3>Sentiment classes</h3>'
            '<p><b>😊 Positive</b><br>Favorable and encouraging language.</p>'
            '<p><b>😐 Neutral</b><br>Balanced, factual or mixed language.</p>'
            '<p><b>😞 Negative</b><br>Critical or unfavorable language.</p>'
            '<hr><p style="font-size:.86rem">Your text is processed securely in this app.</p></div>',
            unsafe_allow_html=True,
        )

    if not predict_clicked:
        return
    try:
        result = predictor.predict(st.session_state["review_text"])
    except ValueError as error:
        st.warning(str(error))
        return

    label = result["label"]
    st.markdown(
        f"<div style='padding:16px;border-left:6px solid {COLORS[label]};"
        f"background:{COLORS[label]}14;border-radius:4px'>"
        f"<span style='font-size:1.6rem;color:{COLORS[label]}'><b>{ICONS[label]} {label}</b></span><br>"
        f"Confidence: <b>{result['confidence'] * 100:.1f}%</b></div>",
        unsafe_allow_html=True)

    probs = pd.DataFrame({"Sentiment": list(result["probabilities"].keys()),
                          "Probability (%)": [p * 100 for p in result["probabilities"].values()]})
    fig = px.bar(probs, x="Probability (%)", y="Sentiment", orientation="h",
                 color="Sentiment", color_discrete_map=COLORS, range_x=[0, 100])
    fig.update_layout(showlegend=False, height=220, margin=dict(t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Confidence is the model's own probability estimate. It is not a "
               "guarantee that the prediction is correct.")


# ---------------------------------------------------------------- Dataset
def dataset_page():
    page_header("Dataset Analysis", "Explore the size, balance and structure of your training data.")
    try:
        df = load_dataset()
    except (FileNotFoundError, ValueError) as error:
        st.error(str(error))
        return

    counts = df["label"].value_counts().reindex(LABELS).fillna(0).astype(int)
    st.metric("Total reviews (after removing empty and duplicate rows)", f"{len(df):,}")
    cols = st.columns(3)
    for col, label in zip(cols, LABELS):
        col.metric(label, f"{counts[label]:,}")

    fig = px.bar(x=counts.index, y=counts.values, color=counts.index,
                 color_discrete_map=COLORS, labels={"x": "Sentiment", "y": "Reviews"})
    fig.update_layout(showlegend=False, margin=dict(t=10))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Review length (words)")
    lengths = df["text"].str.split().str.len()
    st.write(lengths.describe().round(1).to_frame("words").T)
    st.subheader("Sample rows")
    st.dataframe(df.sample(min(10, len(df)), random_state=1), use_container_width=True)


# ---------------------------------------------------------------- Model performance
def performance_page():
    page_header("Model Performance", "Compare model quality and inspect test-set predictions.")
    if not METRICS_PATH.exists():
        st.error("metrics.json not found. Run 'python -m src.train_model' first.")
        return
    with open(METRICS_PATH) as f:
        metrics = json.load(f)

    best = metrics["best_model"]
    st.success(f"Selected model: **{best}** ({metrics['selection_rule']})")
    sizes = metrics["split_sizes"]
    st.caption(f"Train: {sizes['train']:,} | Validation: {sizes['validation']:,} | "
               f"Test: {sizes['test']:,}. The table below shows TEST-set results.")

    test = metrics["test"]
    table = pd.DataFrame({
        name: {"Accuracy": m["accuracy"], "Precision (macro)": m["precision_macro"],
               "Recall (macro)": m["recall_macro"], "F1 (macro)": m["f1_macro"]}
        for name, m in test.items()}).T.round(4)
    st.dataframe(table, use_container_width=True)
    st.plotly_chart(comparison_figure(test), use_container_width=True)

    st.subheader("Confusion matrix")
    chosen = st.selectbox("Model", list(test.keys()), index=list(test.keys()).index(best))
    st.plotly_chart(confusion_matrix_figure(test[chosen]["confusion_matrix"], metrics["labels"]),
                    use_container_width=True)


# ---------------------------------------------------------------- Batch
def batch_page():
    page_header("Batch Analysis", "Upload a CSV and classify thousands of reviews in one run.")
    predictor = load_predictor_or_stop()

    uploaded = st.file_uploader("CSV file", type=["csv"])
    if uploaded is None:
        return
    try:
        data = pd.read_csv(uploaded)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError):
        st.error("Could not read this file as a CSV.")
        return
    if data.empty:
        st.error("The file has no rows.")
        return

    guess = next((c for c in data.columns if c.lower() in ("text", "review", "reviews")), data.columns[0])
    column = st.selectbox("Column that contains the review text", data.columns,
                          index=list(data.columns).index(guess))
    if len(data) > 50000:
        st.warning("Large file: only the first 50,000 rows will be used.")
        data = data.head(50000)

    if st.button("Run batch prediction", type="primary"):
        with st.spinner("Predicting..."):
            predictions = predictor.predict_batch(data[column].tolist())
        output = pd.concat([data.reset_index(drop=True), predictions], axis=1)
        st.dataframe(output.head(100), use_container_width=True)
        summary = output["predicted_sentiment"].value_counts()
        st.plotly_chart(px.bar(x=summary.index, y=summary.values, color=summary.index,
                               color_discrete_map={**COLORS, "Unknown": "#9e9e9e"},
                               labels={"x": "Sentiment", "y": "Reviews"}),
                        use_container_width=True)
        st.download_button("Download results as CSV", output.to_csv(index=False).encode("utf-8"),
                           file_name="sentiment_results.csv", mime="text/csv")


# ---------------------------------------------------------------- About
def about_page():
    page_header("About SentiScope", "Project objective, system design and responsible-use notes.")
    st.subheader("Objective")
    st.write("To build a system that classifies review text as Positive, Negative or Neutral "
             "using NLP and supervised machine learning.")
    st.subheader("Technologies")
    st.write("Python, Pandas, NumPy, NLTK, Scikit-learn (TF-IDF, Logistic Regression, "
             "Naive Bayes, SVM), Joblib, Matplotlib, Plotly, Streamlit.")
    st.subheader("Methodology")
    st.write("1. Load and clean the dataset  \n2. Preprocess text (negations kept)  \n"
             "3. Stratified train / validation / test split  \n4. TF-IDF fitted on training data only  \n"
             "5. Train three models and compare macro F1  \n6. Save the best model and serve it with Streamlit")
    st.subheader("Limitations")
    st.write("- Labels come from star ratings, which are only an approximation of sentiment.  \n"
             "- Sarcasm and mixed opinions are hard for bag-of-words models.  \n"
             "- The model reflects the domain of its training data (e.g. food reviews).  \n"
             "- English only.")
    st.subheader("Future scope")
    st.write("Transformer models (BERT), multilingual support, aspect-based sentiment analysis, "
             "and training on more domains.")


PAGES = {"Home": home_page, "Dataset Analysis": dataset_page,
         "Model Performance": performance_page, "Batch Analysis": batch_page,
         "About Project": about_page}

st.sidebar.markdown(
    '<div style="padding:12px 4px 18px">'
    '<div style="font-size:1.45rem;font-weight:800;color:white">SentiScope</div>'
    '<div style="font-size:.78rem;color:#a5b4fc;margin-top:3px">Sentiment Intelligence</div>'
    '</div>',
    unsafe_allow_html=True,
)
choice = st.sidebar.radio("Navigation", list(PAGES.keys()), label_visibility="collapsed")
PAGES[choice]()
