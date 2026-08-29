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


def evaluate_binary(y_true: np.ndarray, scores: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    values: dict[str, float] = {
        "pr_auc": float(average_precision_score(y_true, scores)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }
    try:
        values["roc_auc"] = float(roc_auc_score(y_true, scores))
    except ValueError:
        values["roc_auc"] = float("nan")
    try:
        values["log_loss"] = float(log_loss(y_true, np.column_stack([1.0 - scores, scores]), labels=[0, 1]))
    except ValueError:
        values["log_loss"] = float("nan")
    return values


def predictions_frame(sample_ids: pd.Index, y_true: np.ndarray, scores: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": sample_ids.astype(str),
            "y_true": y_true.astype(int),
            "y_score": scores.astype(float),
            "y_pred": y_pred.astype(int),
        }
    )


def confusion_frame(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(confusion_matrix(y_true, y_pred, labels=[0, 1]), index=["true_0", "true_1"], columns=["pred_0", "pred_1"])


def classification_report_frame(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(classification_report(y_true, y_pred, labels=[0, 1], output_dict=True, zero_division=0)).T
