from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.svm import SVC

from metrics import evaluate_binary, positive_scores

COMMON_DIR = Path(__file__).resolve().parents[1]
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

from strict_v2_utils import attach_validation_data, calibrate_threshold, keras_strict_v2_callback


def fit_predict_sklearn(
    model_name: str,
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_test: pd.DataFrame,
    y_test: np.ndarray,
    seed: int,
    n_jobs: int = 2,
    x_val: pd.DataFrame | None = None,
    y_val: np.ndarray | None = None,
    threshold_mode: str = "fixed_0_5",
) -> tuple[np.ndarray, np.ndarray, dict[str, float], dict[str, Any]]:
    if model_name == "svm":
        estimator = SVC(C=1.0, kernel="rbf", gamma="scale", probability=True, random_state=seed)
        metadata = {
            "model": "SVM RBF",
            "C": 1.0,
            "gamma": "scale",
            "probability": True,
            "paper_reported": "Gaussian/RBF kernel",
            "implementation_assumptions": ["The paper does not disclose C or gamma."],
        }
    elif model_name == "rf":
        estimator = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=n_jobs)
        metadata = {
            "model": "RandomForestClassifier",
            "n_estimators": 100,
            "random_state": seed,
            "paper_reported_n_estimators": 100,
            "implementation_assumptions": ["All RandomForest parameters other than n_estimators are sklearn defaults."],
        }
    elif model_name == "adaboost":
        estimator = AdaBoostClassifier(n_estimators=200, learning_rate=1.0, random_state=seed, algorithm="SAMME")
        metadata = {
            "model": "AdaBoostClassifier",
            "n_estimators": 200,
            "learning_rate": 1.0,
            "paper_reported_n_estimators": 200,
            "implementation_assumptions": ["The paper does not disclose learning_rate or base-estimator details."],
        }
    elif model_name == "xgboost":
        from xgboost import XGBClassifier

        estimator = XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=seed,
            n_jobs=n_jobs,
        )
        metadata = {
            "model": "XGBClassifier",
            "n_estimators": 100,
            "paper_reported_n_estimators": 100,
            "max_depth": 3,
            "learning_rate": 0.05,
            "tree_method": "hist",
            "implementation_assumptions": [
                "The paper reports only n_estimators=100; max_depth, learning_rate, subsampling, and tree_method are implementation assumptions."
            ],
        }
    else:
        raise ValueError(f"Unsupported sklearn model: {model_name}")
    estimator.fit(x_train, y_train)
    scores = np.nan_to_num(positive_scores(estimator, x_test).astype(float), nan=0.5, posinf=1.0, neginf=0.0)
    if scores.min() < 0.0 or scores.max() > 1.0:
        scores = (scores - scores.min()) / max(scores.max() - scores.min(), 1e-12)
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


def _paper_dnn_model(keras, n_features: int, dropout: float):
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(n_features,)),
            keras.layers.Dense(7, activation="relu"),
            keras.layers.Dropout(dropout),
            keras.layers.Dense(6, activation="relu"),
            keras.layers.Dropout(dropout),
            keras.layers.Dense(6, activation="relu"),
            keras.layers.Dropout(dropout),
            keras.layers.Dense(6, activation="relu"),
            keras.layers.Dense(5, activation="relu"),
            keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    return model


def _paper_cnn_model(keras, n_features: int, dropout: float):
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(n_features, 1)),
            keras.layers.Conv1D(filters=4, kernel_size=3, activation="relu", padding="valid"),
            keras.layers.MaxPooling1D(pool_size=2),
            keras.layers.Conv1D(filters=3, kernel_size=3, activation="relu", padding="valid"),
            keras.layers.MaxPooling1D(pool_size=2),
            keras.layers.Conv1D(filters=3, kernel_size=3, activation="relu", padding="valid"),
            keras.layers.MaxPooling1D(pool_size=2),
            keras.layers.Flatten(),
            keras.layers.Dropout(dropout),
            keras.layers.Dense(4, activation="relu"),
            keras.layers.Dense(2, activation="softmax"),
        ]
    )
    return model


def fit_predict_deep(
    model_name: str,
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
    learning_rate: float,
    dropout: float,
    output_dir: Path,
    threshold_mode: str = "fixed_0_5",
) -> tuple[np.ndarray, np.ndarray, dict[str, float], dict[str, Any]]:
    import tensorflow as tf
    from tensorflow import keras

    tf.keras.utils.set_random_seed(seed)
    if model_name == "dnn":
        model = _paper_dnn_model(keras, x_train.shape[1], dropout)
        train_x = x_train.to_numpy(dtype=np.float32)
        val_x = x_val.to_numpy(dtype=np.float32)
        test_x = x_test.to_numpy(dtype=np.float32)
        architecture = "DNN Dense 7-6-6-6-5-sigmoid with dropout, adapted from Table 1"
    elif model_name == "cnn":
        if x_train.shape[1] < 30:
            raise ValueError("1D-CNN requires at least 30 selected genes for the current paper-like architecture.")
        model = _paper_cnn_model(keras, x_train.shape[1], dropout)
        train_x = x_train.to_numpy(dtype=np.float32)[..., None]
        val_x = x_val.to_numpy(dtype=np.float32)[..., None]
        test_x = x_test.to_numpy(dtype=np.float32)[..., None]
        architecture = "1D-CNN Conv4-Conv3-Conv3-Dense4-Dense2-softmax, following Table 2"
    else:
        raise ValueError(f"Unsupported deep model: {model_name}")

    if model_name == "cnn":
        loss_name = "sparse_categorical_crossentropy"
        compile_metrics = [keras.metrics.SparseCategoricalAccuracy(name="accuracy")]
        train_targets = y_train.astype(np.int32)
        val_targets = y_val.astype(np.int32)
        output_activation = "softmax"
        output_units = 2
    else:
        loss_name = "binary_crossentropy"
        compile_metrics = [keras.metrics.AUC(name="auc"), keras.metrics.AUC(name="pr_auc", curve="PR")]
        train_targets = y_train.astype(np.float32)
        val_targets = y_val.astype(np.float32)
        output_activation = "sigmoid"
        output_units = 1
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss_name,
        metrics=compile_metrics,
    )
    model_path = output_dir / f"{model_name}_model.keras"
    attach_validation_data(model, val_x, y_val)
    callbacks = [keras.callbacks.TerminateOnNaN(), keras_strict_v2_callback(model_path, patience)]
    history = model.fit(
        train_x,
        train_targets,
        validation_data=(val_x, val_targets),
        epochs=max_epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=0,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    if model_path.exists():
        model = keras.models.load_model(model_path)
    else:
        model.save(model_path)
    val_output = np.asarray(model.predict(val_x, verbose=0))
    test_output = np.asarray(model.predict(test_x, verbose=0))
    val_scores_raw = val_output[:, 1] if val_output.ndim == 2 and val_output.shape[1] == 2 else val_output.reshape(-1)
    scores_raw = test_output[:, 1] if test_output.ndim == 2 and test_output.shape[1] == 2 else test_output.reshape(-1)
    val_scores = np.nan_to_num(val_scores_raw.astype(float), nan=0.5, posinf=1.0, neginf=0.0)
    scores = np.nan_to_num(scores_raw.astype(float), nan=0.5, posinf=1.0, neginf=0.0)
    if threshold_mode == "fixed_0_5":
        threshold = 0.5
        threshold_meta = {"threshold": threshold, "threshold_mode": "fixed_0_5"}
    else:
        metric = threshold_mode.replace("validation_", "")
        threshold, threshold_meta = calibrate_threshold(y_val, val_scores, metric=metric)
        threshold_meta["threshold_mode"] = threshold_mode
    y_pred = (scores >= threshold).astype(int)
    metadata = {
        "model": model_name,
        "architecture": architecture,
        "optimizer": "Adam",
        "learning_rate": float(learning_rate),
        "loss": loss_name,
        "output_activation": output_activation,
        "output_units": output_units,
        "paper_adaptation": "The paper text inconsistently mentions three DNN outputs, while Tables 1-2 specify binary outputs. This implementation follows the tables: one sigmoid unit for DNN and two-way softmax for CNN.",
        "dropout": float(dropout),
        "max_epochs": int(max_epochs),
        "epochs_ran": int(len(history.history.get("loss", []))),
        "checkpoint_score": "0.5*val_roc_auc + 0.5*val_macro_f1 - 0.25*val_loss",
        "best_val_auc": float(max(history.history.get("val_auc", [float("nan")]))),
        "best_val_pr_auc": float(max(history.history.get("val_pr_auc", [float("nan")]))),
        "patience": int(patience),
        "batch_size": int(batch_size),
        "saved_model": str(model_path),
        "tensorflow_version": tf.__version__,
        "threshold_calibration": threshold_meta,
    }
    return scores, y_pred, evaluate_binary(y_test, scores, y_pred), metadata
