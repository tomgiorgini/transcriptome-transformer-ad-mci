from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def positive_scores(estimator, x) -> np.ndarray:
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(x)[:, 1]
    if hasattr(estimator, "decision_function"):
        return estimator.decision_function(x)
    return estimator.predict(x)


def binary_predictions(scores: np.ndarray) -> np.ndarray:
    return (scores >= 0.5).astype(int)


def evaluate_binary(y_true: np.ndarray, scores: np.ndarray, y_pred: np.ndarray | None = None) -> dict[str, float]:
    if y_pred is None:
        y_pred = binary_predictions(scores)
    metrics = {
        "pr_auc": float(average_precision_score(y_true, scores)),
        "roc_auc": float(roc_auc_score(y_true, scores)) if len(np.unique(y_true)) == 2 else float("nan"),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
    }
    try:
        prob = np.column_stack([1.0 - scores, scores])
        metrics["log_loss"] = float(log_loss(y_true, prob, labels=[0, 1]))
    except ValueError:
        metrics["log_loss"] = float("nan")
    return metrics


def predictions_frame(sample_ids: pd.Index, y_true: np.ndarray, scores: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": sample_ids.astype(str),
            "y_true": y_true.astype(int),
            "y_pred": y_pred.astype(int),
            "score_positive_class": scores.astype(float),
        }
    )


def confusion_frame(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return pd.DataFrame(cm, index=["true_class_0", "true_class_1"], columns=["pred_class_0", "pred_class_1"])


def classification_report_frame(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    report = classification_report(y_true, y_pred, target_names=["class_0", "class_1"], output_dict=True, zero_division=0)
    return pd.DataFrame(report).transpose()
