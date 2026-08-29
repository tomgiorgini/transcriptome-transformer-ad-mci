from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from sklearn.base import clone
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold


STRICT_V2_FS_SCORE = "0.5*roc_auc + 0.5*macro_f1"
STRICT_V2_CHECKPOINT_SCORE = "0.5*val_roc_auc + 0.5*val_macro_f1 - 0.25*val_loss"


def finite_scores(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    if scores.ndim == 2:
        scores = scores[:, 1] if scores.shape[1] > 1 else scores[:, 0]
    scores = scores.reshape(-1)
    scores = np.nan_to_num(scores, nan=0.5, posinf=1.0, neginf=0.0)
    if scores.size and (scores.min() < 0.0 or scores.max() > 1.0):
        denom = max(float(scores.max() - scores.min()), 1e-12)
        scores = (scores - scores.min()) / denom
    return np.clip(scores, 0.0, 1.0)


def positive_scores(estimator: Any, x: Any) -> np.ndarray:
    if hasattr(estimator, "predict_proba"):
        return finite_scores(estimator.predict_proba(x)[:, 1])
    if hasattr(estimator, "decision_function"):
        return finite_scores(estimator.decision_function(x))
    return finite_scores(estimator.predict(x))


def strict_v2_fs_score(y_true: np.ndarray, scores: np.ndarray) -> float:
    scores = finite_scores(scores)
    y_pred = (scores >= 0.5).astype(int)
    try:
        roc = float(roc_auc_score(y_true, scores))
    except ValueError:
        roc = 0.5
    macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    return 0.5 * roc + 0.5 * macro


def calibrate_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    metric: str = "macro_f1",
    thresholds: np.ndarray | None = None,
) -> tuple[float, dict[str, float]]:
    """Choose a classification threshold on validation data only."""
    y_true = np.asarray(y_true, dtype=int)
    scores = finite_scores(scores)
    thresholds = thresholds if thresholds is not None else np.linspace(0.05, 0.95, 91)
    best_threshold = 0.5
    best_primary = float("-inf")
    best_accuracy = float("-inf")
    best_macro_f1 = float("-inf")
    for threshold in thresholds:
        pred = (scores >= float(threshold)).astype(int)
        macro_f1 = float(f1_score(y_true, pred, average="macro", zero_division=0))
        accuracy = float(accuracy_score(y_true, pred))
        if metric == "accuracy_macro_f1":
            primary = 0.5 * accuracy + 0.5 * macro_f1
        elif metric == "accuracy":
            primary = accuracy
        elif metric == "macro_f1":
            primary = macro_f1
        else:
            raise ValueError(f"Unsupported threshold calibration metric: {metric}")
        if primary > best_primary or (
            np.isclose(primary, best_primary) and (macro_f1, accuracy) > (best_macro_f1, best_accuracy)
        ):
            best_threshold = float(threshold)
            best_primary = primary
            best_accuracy = accuracy
            best_macro_f1 = macro_f1
    try:
        val_roc_auc = float(roc_auc_score(y_true, scores))
    except ValueError:
        val_roc_auc = float("nan")
    return best_threshold, {
        "threshold": best_threshold,
        "threshold_metric": metric,
        "validation_macro_f1_at_threshold": best_macro_f1,
        "validation_accuracy_at_threshold": best_accuracy,
        "validation_roc_auc": val_roc_auc,
    }


def cv_score_estimator(estimator: Any, x: Any, y: np.ndarray, folds: int, seed: int) -> tuple[float, list[float]]:
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    scores: list[float] = []
    for train_idx, val_idx in cv.split(x, y):
        model = clone(estimator)
        x_train = x.iloc[train_idx] if hasattr(x, "iloc") else x[train_idx]
        x_val = x.iloc[val_idx] if hasattr(x, "iloc") else x[val_idx]
        model.fit(x_train, y[train_idx])
        scores.append(strict_v2_fs_score(y[val_idx], positive_scores(model, x_val)))
    return float(np.mean(scores)), scores


def keras_strict_v2_callback(checkpoint_path: Path, patience: int):
    import tensorflow as tf
    from tensorflow import keras

    class StrictV2Checkpoint(keras.callbacks.Callback):
        def __init__(self) -> None:
            super().__init__()
            self.best = float("-inf")
            self.wait = 0
            self.best_epoch = 0
            self.history_rows: list[dict[str, float]] = []
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
            logs = logs or {}
            val_loss = float(logs.get("val_loss", np.nan))
            val_auc = float(logs.get("val_auc", logs.get("val_AUC", np.nan)))
            y_val = getattr(self.model, "_strict_v2_y_val", None)
            x_val = getattr(self.model, "_strict_v2_x_val", None)
            if y_val is not None and x_val is not None:
                pred = self.model.predict(x_val, verbose=0)
                if pred.ndim == 2 and pred.shape[1] > 1:
                    pred = pred[:, 1]
                pred = finite_scores(pred.reshape(-1))
                val_macro_f1 = float(f1_score(y_val, (pred >= 0.5).astype(int), average="macro", zero_division=0))
                try:
                    val_auc = float(roc_auc_score(y_val, pred))
                except ValueError:
                    val_auc = 0.5
            else:
                val_macro_f1 = float(logs.get("val_macro_f1", np.nan))
            if np.isnan(val_auc):
                val_auc = 0.5
            if np.isnan(val_macro_f1):
                val_macro_f1 = 0.0
            if np.isnan(val_loss):
                val_loss = 0.0
            score = 0.5 * val_auc + 0.5 * val_macro_f1 - 0.25 * val_loss
            self.history_rows.append(
                {
                    "epoch": float(epoch + 1),
                    "val_loss": val_loss,
                    "val_roc_auc": val_auc,
                    "val_macro_f1": val_macro_f1,
                    "val_strict_v2_checkpoint_score": score,
                }
            )
            if score >= self.best:
                self.best = score
                self.wait = 0
                self.best_epoch = epoch + 1
                self.model.save(checkpoint_path, overwrite=True)
            else:
                self.wait += 1
                if patience > 0 and self.wait >= patience:
                    self.model.stop_training = True

    return StrictV2Checkpoint()


def attach_validation_data(model: Any, x_val: Any, y_val: np.ndarray) -> None:
    model._strict_v2_x_val = x_val
    model._strict_v2_y_val = np.asarray(y_val, dtype=int)
