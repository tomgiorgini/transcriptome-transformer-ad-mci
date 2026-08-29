from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import numpy as np

from metrics import binary_predictions, evaluate_binary

COMMON_DIR = Path(__file__).resolve().parents[1]
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

from strict_v2_utils import attach_validation_data, calibrate_threshold, keras_strict_v2_callback


def set_tf_seed(seed: int) -> None:
    import tensorflow as tf

    tf.keras.utils.set_random_seed(seed)


def _regularized_dense(layers, regularizers, x, units: int):
    return layers.Dense(
        units,
        activation="relu",
        kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
        bias_regularizer=regularizers.l2(1e-4),
        activity_regularizer=regularizers.l2(1e-5),
    )(x)


def build_fnn(input_dim: int, seed: int, learning_rate: float = 1e-4):
    import tensorflow as tf
    from tensorflow.keras import layers, regularizers

    set_tf_seed(seed)
    inputs = tf.keras.Input(shape=(input_dim,), name="gene_input")
    x = layers.BatchNormalization()(inputs)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dense(64, activation="relu")(x)
    x = _regularized_dense(layers, regularizers, x, 32)
    x = layers.Dropout(0.4)(x)
    x = _regularized_dense(layers, regularizers, x, 32)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(1, activation="sigmoid", name="positive_class_probability")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="one2m_fnn")
    _compile_binary_model(model, optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))
    return model


def build_cnn(image_shape: tuple[int, int, int], seed: int, learning_rate: float = 1e-4):
    import tensorflow as tf
    from tensorflow.keras import layers, regularizers

    set_tf_seed(seed)
    inputs = tf.keras.Input(shape=image_shape, name="image_input")
    x = layers.BatchNormalization()(inputs)
    x = layers.Conv2D(32, (3, 3), activation="relu")(x)
    x = layers.Conv2D(32, (3, 3), activation="relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation="relu")(x)
    x = layers.Conv2D(64, (3, 3), activation="relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, (3, 3), activation="relu")(x)
    x = layers.Conv2D(128, (3, 3), activation="relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = _regularized_dense(layers, regularizers, x, 32)
    x = layers.Dropout(0.4)(x)
    x = _regularized_dense(layers, regularizers, x, 32)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(1, activation="sigmoid", name="positive_class_probability")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="one2m_cnn")
    _compile_binary_model(model, optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))
    return model


def build_fusion(input_dim: int, image_shape: tuple[int, int, int], seed: int, learning_rate: float = 1e-4):
    import tensorflow as tf
    from tensorflow.keras import layers, regularizers

    set_tf_seed(seed)
    gene_input = tf.keras.Input(shape=(input_dim,), name="gene_input")
    gx = layers.BatchNormalization()(gene_input)
    gx = layers.Dense(128, activation="relu")(gx)
    gx = layers.Dense(64, activation="relu")(gx)
    gx = _regularized_dense(layers, regularizers, gx, 32)
    gx = layers.Dropout(0.4)(gx)
    gx = _regularized_dense(layers, regularizers, gx, 32)
    gx = layers.Dropout(0.4)(gx)

    image_input = tf.keras.Input(shape=image_shape, name="image_input")
    ix = layers.BatchNormalization()(image_input)
    ix = layers.Conv2D(32, (3, 3), activation="relu")(ix)
    ix = layers.Conv2D(32, (3, 3), activation="relu")(ix)
    ix = layers.MaxPooling2D((2, 2))(ix)
    ix = layers.Conv2D(64, (3, 3), activation="relu")(ix)
    ix = layers.Conv2D(64, (3, 3), activation="relu")(ix)
    ix = layers.MaxPooling2D((2, 2))(ix)
    ix = layers.BatchNormalization()(ix)
    ix = layers.Conv2D(128, (3, 3), activation="relu")(ix)
    ix = layers.Conv2D(128, (3, 3), activation="relu")(ix)
    ix = layers.MaxPooling2D((2, 2))(ix)
    ix = layers.GlobalAveragePooling2D()(ix)
    ix = _regularized_dense(layers, regularizers, ix, 32)
    ix = layers.Dropout(0.4)(ix)
    ix = _regularized_dense(layers, regularizers, ix, 32)
    ix = layers.Dropout(0.4)(ix)

    merged = layers.concatenate([gx, ix], name="fusion_concat")
    merged = layers.Dense(32, activation="relu", name="fusion_dense_1")(merged)
    merged = layers.Dropout(0.4)(merged)
    merged = layers.Dense(32, activation="relu", name="fusion_dense_2")(merged)
    outputs = layers.Dense(1, activation="sigmoid", name="positive_class_probability")(merged)
    model = tf.keras.Model(inputs=[image_input, gene_input], outputs=outputs, name="one2mfusion")
    _compile_binary_model(model, optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))
    return model


def _compile_binary_model(model, optimizer: str | Any = "adam") -> None:
    import tensorflow as tf

    model.compile(
        loss="binary_crossentropy",
        optimizer=optimizer,
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc", curve="ROC"),
            tf.keras.metrics.AUC(name="pr_auc", curve="PR"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )


def train_and_evaluate(
    model_name: str,
    x_train_gene: np.ndarray,
    x_val_gene: np.ndarray,
    x_test_gene: np.ndarray,
    x_train_image: np.ndarray,
    x_val_image: np.ndarray,
    x_test_image: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    seed: int,
    epochs: int,
    batch_size: int,
    patience: int,
    checkpoint_path: Path,
    early_stopping_monitor: str = "val_loss",
    start_from_epoch: int = 0,
    threshold_mode: str = "fixed_0_5",
    learning_rate: float = 1e-4,
    implementation_profile: str = "paper",
) -> tuple[np.ndarray, np.ndarray, dict[str, float], dict[str, Any]]:
    import tensorflow as tf

    if model_name == "fnn":
        model = build_fnn(x_train_gene.shape[1], seed, learning_rate)
        train_x = _finite_array(x_train_gene)
        val_x = _finite_array(x_val_gene)
        test_x = _finite_array(x_test_gene)
    elif model_name == "cnn":
        model = build_cnn(tuple(x_train_image.shape[1:]), seed, learning_rate)
        train_x = _finite_array(x_train_image)
        val_x = _finite_array(x_val_image)
        test_x = _finite_array(x_test_image)
    elif model_name == "one2mfusion":
        model = build_fusion(x_train_gene.shape[1], tuple(x_train_image.shape[1:]), seed, learning_rate)
        train_x = [_finite_array(x_train_image), _finite_array(x_train_gene)]
        val_x = [_finite_array(x_val_image), _finite_array(x_val_gene)]
        test_x = [_finite_array(x_test_image), _finite_array(x_test_gene)]
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    if early_stopping_monitor == "strict_v2":
        attach_validation_data(model, val_x, y_val)
        strict_callback = keras_strict_v2_callback(checkpoint_path, patience)
        callbacks = [tf.keras.callbacks.TerminateOnNaN(), strict_callback]
    else:
        callback_kwargs: dict[str, Any] = {
            "monitor": early_stopping_monitor,
            "patience": patience,
            "restore_best_weights": True,
        }
        if start_from_epoch > 0:
            callback_kwargs["start_from_epoch"] = start_from_epoch
        callbacks = [
            tf.keras.callbacks.TerminateOnNaN(),
            tf.keras.callbacks.EarlyStopping(**callback_kwargs),
            tf.keras.callbacks.ModelCheckpoint(str(checkpoint_path), monitor=early_stopping_monitor, save_best_only=True),
        ]
    history = model.fit(
        train_x,
        y_train.astype(np.float32),
        validation_data=(val_x, y_val.astype(np.float32)),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=0,
    )
    if checkpoint_path.exists():
        model = tf.keras.models.load_model(str(checkpoint_path))
    val_scores = model.predict(val_x, batch_size=batch_size, verbose=0).reshape(-1).astype(float)
    scores = model.predict(test_x, batch_size=batch_size, verbose=0).reshape(-1).astype(float)
    had_nonfinite_scores = bool((~np.isfinite(scores)).any())
    scores = np.nan_to_num(scores, nan=0.5, posinf=1.0, neginf=0.0)
    if threshold_mode == "fixed_0_5":
        threshold = 0.5
        threshold_meta = {"threshold": threshold, "threshold_mode": "fixed_0_5"}
    else:
        metric = threshold_mode.replace("validation_", "")
        threshold, threshold_meta = calibrate_threshold(y_val, val_scores, metric=metric)
        threshold_meta["threshold_mode"] = threshold_mode
    y_pred = (scores >= threshold).astype(int)
    metrics = evaluate_binary(y_test, scores, y_pred)
    metadata = {
        "model_name": model_name,
        "epochs_ran": len(history.history["loss"]),
        "checkpoint": str(checkpoint_path),
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "implementation_profile": implementation_profile,
        "gene_input_features": int(x_train_gene.shape[1]),
        "gene_input_scope": "all_genes" if model_name == "fnn" else "lasso_selected",
        "patience": patience,
        "early_stopping_monitor": early_stopping_monitor,
        "checkpoint_score": "0.5*val_roc_auc + 0.5*val_macro_f1 - 0.25*val_loss" if early_stopping_monitor == "strict_v2" else early_stopping_monitor,
        "start_from_epoch": start_from_epoch,
        "had_nonfinite_scores": had_nonfinite_scores,
        "threshold_calibration": threshold_meta,
    }
    return scores, y_pred, metrics, metadata


def _finite_array(x: np.ndarray) -> np.ndarray:
    return np.nan_to_num(x.astype(np.float32, copy=False), nan=0.0, posinf=1.0, neginf=0.0)
