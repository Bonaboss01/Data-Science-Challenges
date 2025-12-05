import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve
)
import numpy as np


def evaluate_classification(model, X_test, y_true, save_plots=False, model_name="model"):
    """
    Advanced evaluation of classification performance.

    Parameters
    ----------
    model : fitted model
        Must have .predict() and .predict_proba() (optional for ROC)
    X_test : array-like
        Test features
    y_true : array-like
        True labels
    save_plots : bool
        Whether to save confusion matrix, ROC and PR curves
    model_name : str
        Used for saving plot filenames

    Returns
    -------
    metrics : dict
        Dictionary containing accuracy, precision, recall, f1, auc (if possible)
    """

    # ---------------------------
    # 1. Generate predictions
    # ---------------------------
    y_pred = model.predict(X_test)

    # Some models do not have predict_proba (e.g., SVM with no probability=True)
    has_proba = hasattr(model, "predict_proba")
    y_proba = model.predict_proba(X_test)[:, 1] if has_proba else None

    # ---------------------------
    # 2. Compute metrics
    # ---------------------------
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="binary" if len(np.unique(y_true)) == 2 else "macro"),
        "recall": recall_score(y_true, y_pred, average="binary" if len(np.unique(y_true)) == 2 else "macro"),
        "f1_score": f1_score(y_true, y_pred, average="binary" if len(np.unique(y_true)) == 2 else "macro"),
    }

    # Add AUC only for binary classification with predict_proba
    if has_proba and len(np.unique(y_

