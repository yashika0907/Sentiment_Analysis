"""Metrics and charts used by training and by the Streamlit app."""
import matplotlib
matplotlib.use("Agg")  # no display needed when saving files
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, precision_recall_fscore_support)


def evaluate_predictions(y_true, y_pred, labels):
    """Return accuracy, macro precision/recall/F1, confusion matrix and per-class report."""
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "report": classification_report(y_true, y_pred, labels=labels,
                                        output_dict=True, zero_division=0),
    }


def save_confusion_matrix_png(matrix, labels, title, output_path):
    """Matplotlib confusion matrix saved as an image (useful for the report)."""
    matrix = np.array(matrix)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, matrix[i, j], ha="center", va="center",
                    color="white" if matrix[i, j] > matrix.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def comparison_figure(results):
    """Plotly grouped bar chart. results = {model_name: metrics_dict}."""
    import plotly.graph_objects as go
    metric_names = {"accuracy": "Accuracy", "precision_macro": "Precision (macro)",
                    "recall_macro": "Recall (macro)", "f1_macro": "F1 (macro)"}
    fig = go.Figure()
    for key, nice_name in metric_names.items():
        fig.add_bar(name=nice_name, x=list(results.keys()),
                    y=[results[m][key] for m in results])
    fig.update_layout(barmode="group", yaxis_title="Score", yaxis_range=[0, 1],
                      margin=dict(t=30, b=30))
    return fig


def confusion_matrix_figure(matrix, labels):
    """Plotly heatmap of a confusion matrix."""
    import plotly.express as px
    fig = px.imshow(matrix, x=labels, y=labels, text_auto=True,
                    color_continuous_scale="Blues",
                    labels=dict(x="Predicted", y="Actual", color="Count"))
    fig.update_layout(margin=dict(t=30, b=30))
    return fig
