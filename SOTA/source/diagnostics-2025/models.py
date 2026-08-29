from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC

from metrics import evaluate_binary, positive_scores

COMMON_DIR = Path(__file__).resolve().parents[1]
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

from strict_v2_utils import calibrate_threshold
from strict_v2_utils import STRICT_V2_CHECKPOINT_SCORE, attach_validation_data, keras_strict_v2_callback


def fit_predict_sklearn(
    model_name: str,
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_test: pd.DataFrame,
    y_test: np.ndarray,
    seed: int,
    x_val: pd.DataFrame | None = None,
    y_val: np.ndarray | None = None,
    threshold_mode: str = "fixed_0_5",
) -> tuple[np.ndarray, np.ndarray, dict[str, float], dict[str, Any]]:
    if model_name == "svm":
        estimator = SVC(C=1.0, kernel="rbf", gamma="scale", probability=True, random_state=seed)
        metadata = {"model": "SVM RBF", "C": 1.0, "gamma": "scale", "probability": True}
    elif model_name == "gbm":
        estimator = GradientBoostingClassifier(random_state=seed)
        metadata = {"model": "GradientBoostingClassifier", "random_state": seed}
    elif model_name == "rf":
        estimator = RandomForestClassifier(n_estimators=500, random_state=seed, n_jobs=2)
        metadata = {"model": "RandomForestClassifier", "n_estimators": 500, "random_state": seed}
    else:
        raise ValueError(f"Unsupported sklearn model: {model_name}")
    estimator.fit(x_train, y_train)
    scores = np.nan_to_num(positive_scores(estimator, x_test).astype(float), nan=0.5, posinf=1.0, neginf=0.0)
    if threshold_mode == "fixed_0_5" or x_val is None or y_val is None:
        threshold = 0.5
        threshold_meta = {"threshold": threshold, "threshold_mode": "fixed_0_5"}
    else:
        metric = threshold_mode.replace("validation_", "")
        threshold, threshold_meta = calibrate_threshold(y_val, positive_scores(estimator, x_val), metric=metric)
        threshold_meta["threshold_mode"] = threshold_mode
    y_pred = (scores >= threshold).astype(int)
    metadata["threshold_calibration"] = threshold_meta
    return scores, y_pred, evaluate_binary(y_test, scores, y_pred), metadata


def fit_predict_dl(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_val: pd.DataFrame,
    y_val: np.ndarray,
    x_test: pd.DataFrame,
    y_test: np.ndarray,
    seed: int,
    max_epochs: int,
    patience: int,
    batch_size: int,
    output_dir: Path,
    threshold_mode: str = "fixed_0_5",
) -> tuple[np.ndarray, np.ndarray, dict[str, float], dict[str, Any]]:
    import tensorflow as tf
    from tensorflow import keras

    tf.keras.utils.set_random_seed(seed)
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(x_train.shape[1],)),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dropout(0.20),
            keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="auc")],
    )
    checkpoint_path = output_dir / "dl_model.keras"
    attach_validation_data(
        model,
        x_val.to_numpy(dtype=np.float32),
        y_val,
    )
    checkpoint_callback = keras_strict_v2_callback(checkpoint_path, patience)
    callbacks = [checkpoint_callback]
    history = model.fit(
        x_train.to_numpy(dtype=np.float32),
        y_train.astype(np.float32),
        validation_data=(x_val.to_numpy(dtype=np.float32), y_val.astype(np.float32)),
        epochs=max_epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=0,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    history_frame = pd.DataFrame(checkpoint_callback.history_rows)
    history_frame.to_csv(output_dir / "checkpoint_history.csv", index=False)
    if checkpoint_path.exists():
        model = keras.models.load_model(checkpoint_path)
    else:
        model.save(checkpoint_path)
    val_scores = model.predict(x_val.to_numpy(dtype=np.float32), verbose=0).ravel()
    scores = model.predict(x_test.to_numpy(dtype=np.float32), verbose=0).ravel()
    scores = np.nan_to_num(scores.astype(float), nan=0.5, posinf=1.0, neginf=0.0)
    if threshold_mode == "fixed_0_5":
        threshold = 0.5
        threshold_meta = {"threshold": threshold, "threshold_mode": "fixed_0_5"}
    else:
        metric = threshold_mode.replace("validation_", "")
        threshold, threshold_meta = calibrate_threshold(y_val, val_scores, metric=metric)
        threshold_meta["threshold_mode"] = threshold_mode
    y_pred = (scores >= threshold).astype(int)
    metadata = {
        "model": "Dense DL",
        "optimizer": "Adam",
        "learning_rate": 0.001,
        "loss": "binary_crossentropy",
        "hidden_units": 128,
        "activation": "relu",
        "dropout": 0.20,
        "output_activation": "sigmoid",
        "max_epochs": int(max_epochs),
        "epochs_ran": int(len(history.history.get("loss", []))),
        "checkpoint_score": STRICT_V2_CHECKPOINT_SCORE,
        "best_epoch": int(checkpoint_callback.best_epoch),
        "best_checkpoint_score": float(checkpoint_callback.best),
        "best_val_auc": float(history_frame["val_roc_auc"].max()) if not history_frame.empty else float("nan"),
        "patience": int(patience),
        "batch_size": int(batch_size),
        "saved_model": str(checkpoint_path),
        "tensorflow_version": tf.__version__,
        "threshold_calibration": threshold_meta,
    }
    return scores, y_pred, evaluate_binary(y_test, scores, y_pred), metadata
