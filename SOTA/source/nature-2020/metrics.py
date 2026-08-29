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


def binary_predictions(scores: np.ndarray) -> np.ndarray:
    return (np.asarray(scores) >= 0.5).astype(int)


def evaluate_binary(y_true: np.ndarray, scores: np.ndarray, y_pred: np.ndarray | None = None) -> dict[str, float]:
    scores = np.asarray(scores, dtype=float)
    if y_pred is None:
        y_pred = binary_predictions(scores)
    metrics = {
        "pr_auc": float(average_precision_score(y_true, scores)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, scores))
    except ValueError:
        metrics["roc_auc"] = float("nan")
    try:
        metrics["log_loss"] = float(log_loss(y_true, np.column_stack([1.0 - scores, scores]), labels=[0, 1]))
    except ValueError:
        metrics["log_loss"] = float("nan")
    return metrics


def predictions_frame(
    sample_ids: pd.Index,
    y_true: np.ndarray,
    scores: np.ndarray,
    y_pred: np.ndarray,
    class_names: tuple[str, str] = ("class_0", "class_1"),
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": sample_ids.astype(str),
            "y_true": y_true.astype(int),
            "y_pred": y_pred.astype(int),
            "score_positive": np.asarray(scores, dtype=float),
            "negative_class": class_names[0],
            "positive_class": class_names[1],
        }
    )


def confusion_frame(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: tuple[str, str] = ("class_0", "class_1"),
) -> pd.DataFrame:
    negative, positive = class_names
    return pd.DataFrame(
        confusion_matrix(y_true, y_pred, labels=[0, 1]),
        index=[f"true_{negative}", f"true_{positive}"],
        columns=[f"pred_{negative}", f"pred_{positive}"],
    )


def classification_report_frame(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: tuple[str, str] = ("class_0", "class_1"),
) -> pd.DataFrame:
    return pd.DataFrame(
        classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=list(class_names),
            output_dict=True,
            zero_division=0,
        )
    ).T
