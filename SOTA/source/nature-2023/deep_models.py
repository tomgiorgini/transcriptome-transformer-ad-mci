from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from metrics import evaluate_binary


def _to_categorical(y: np.ndarray):
    import tensorflow as tf

    return tf.keras.utils.to_categorical(y, num_classes=2)


def _scores_from_model(model, x: np.ndarray, batch_size: int) -> np.ndarray:
    pred = model.predict(x, batch_size=batch_size, verbose=0)
    return pred[:, 1].astype(float)


def train_vae_classifier(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    seed: int,
    epochs: int,
    batch_size: int,
    patience: int,
    checkpoint_path: Path,
    architecture: str,
    learning_rate: float,
) -> tuple[np.ndarray, dict[str, float], dict[str, Any]]:
    import tensorflow as tf

    tf.keras.utils.set_random_seed(seed)
    feature_num = x_train.shape[1]
    inputs = tf.keras.Input(shape=(feature_num,))

    def dense_block(x, units: int, activation: str = "relu", dropout: bool = False):
        x = tf.keras.layers.Dense(units, activation=activation)(x)
        if architecture in {"batchnorm", "batchnorm_dropout"}:
            x = tf.keras.layers.BatchNormalization()(x)
        if dropout and architecture == "batchnorm_dropout":
            x = tf.keras.layers.Dropout(0.2)(x)
        return x

    x = dense_block(inputs, 4096)
    x = dense_block(x, 1024, dropout=True)
    x = dense_block(x, 512, dropout=True)
    x = tf.keras.layers.Dense(128, name="latent_mu")(x)
    if architecture in {"batchnorm", "batchnorm_dropout"}:
        x = tf.keras.layers.BatchNormalization()(x)
    x = dense_block(x, 128)
    x = dense_block(x, 64, dropout=True)
    outputs = tf.keras.layers.Dense(2, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs, name=f"vae_classifier_{architecture}")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="AUC", curve="ROC"),
            tf.keras.metrics.AUC(name="PR", curve="PR"),
        ],
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(str(checkpoint_path), monitor="val_loss", save_best_only=True),
    ]
    history = model.fit(
        x_train,
        _to_categorical(y_train),
        validation_data=(x_val, _to_categorical(y_val)),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=0,
    )
    if checkpoint_path.exists():
        model = tf.keras.models.load_model(str(checkpoint_path))
    scores = _scores_from_model(model, x_test, batch_size)
    y_pred = (scores >= 0.5).astype(int)
    metrics = evaluate_binary(y_test, scores, y_pred)
    return scores, metrics, {
        "epochs_ran": len(history.history["loss"]),
        "checkpoint": str(checkpoint_path),
        "architecture": architecture,
        "learning_rate": learning_rate,
    }


def _reshape_for_cnn(x: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    rows = 100
    cols = int(np.ceil(x.shape[1] / rows))
    padded_features = rows * cols
    pad = padded_features - x.shape[1]
    if pad:
        x = np.concatenate([x, np.zeros((x.shape[0], pad), dtype=x.dtype)], axis=1)
    reshaped = x.reshape((-1, rows, cols, 1)).astype("float32")
    return reshaped, {"rows": rows, "cols": cols, "padded_features": padded_features, "pad_added": pad}


def train_cnn_classifier(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    seed: int,
    epochs: int,
    batch_size: int,
    patience: int,
    checkpoint_path: Path,
) -> tuple[np.ndarray, dict[str, float], dict[str, Any]]:
    import tensorflow as tf

    tf.keras.utils.set_random_seed(seed)
    x_train_img, shape_meta = _reshape_for_cnn(x_train)
    x_val_img, _ = _reshape_for_cnn(x_val)
    x_test_img, _ = _reshape_for_cnn(x_test)
    input_shape = x_train_img.shape[1:]
    kernel_width = min(100, input_shape[1])

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.Conv2D(16, kernel_size=(1, kernel_width), strides=(1, 1), activation="relu"),
            tf.keras.layers.MaxPooling2D(pool_size=(1, 1), strides=(2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(2, activation="softmax"),
        ]
    )
    model.compile(loss="categorical_crossentropy", optimizer="sgd", metrics=["categorical_accuracy"])
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(str(checkpoint_path), monitor="val_loss", save_best_only=True),
    ]
    history = model.fit(
        x_train_img,
        _to_categorical(y_train),
        validation_data=(x_val_img, _to_categorical(y_val)),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        verbose=0,
    )
    if checkpoint_path.exists():
        model = tf.keras.models.load_model(str(checkpoint_path))
    scores = _scores_from_model(model, x_test_img, batch_size)
    y_pred = (scores >= 0.5).astype(int)
    metrics = evaluate_binary(y_test, scores, y_pred)
    return scores, metrics, {
        "epochs_ran": len(history.history["loss"]),
        "checkpoint": str(checkpoint_path),
        "filters": 16,
        "kernel_size": [1, kernel_width],
        "dense_layer_size": 64,
        **shape_meta,
    }
