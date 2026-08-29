from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


def dnn_epochs(n_features: int) -> int:
    return int(min(3000, max(200, n_features * 3)))


def dnn_hidden_sizes(n_features: int) -> tuple[int, int]:
    return max(n_features // 2, 10), max(n_features // 4, 5)


def fit_predict_dnn(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    seed: int,
    max_epochs: int | None,
    batch_size: int,
    patience: int,
    class_weight: dict[int, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    import tensorflow as tf
    from tensorflow import keras

    tf.keras.utils.set_random_seed(seed)
    n_features = int(x_train.shape[1])
    hidden1, hidden2 = dnn_hidden_sizes(n_features)
    epochs = dnn_epochs(n_features) if max_epochs is None else min(max_epochs, dnn_epochs(n_features))
    optimizer_name = "adagrad" if n_features >= 800 else "adam"
    optimizer = keras.optimizers.Adagrad(learning_rate=0.001) if optimizer_name == "adagrad" else keras.optimizers.Adam(learning_rate=0.001)

    model = keras.Sequential(
        [
            keras.Input(shape=(n_features,)),
            keras.layers.Dense(hidden1, activation="tanh"),
            keras.layers.Dense(hidden2, activation="tanh"),
            keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(curve="PR", name="pr_auc"), keras.metrics.AUC(curve="ROC", name="roc_auc")],
    )
    callbacks = [keras.callbacks.EarlyStopping(monitor="val_pr_auc", mode="max", patience=patience, restore_best_weights=True)]
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=0,
    )
    val_scores = model.predict(x_val, verbose=0).reshape(-1)
    test_scores = model.predict(x_test, verbose=0).reshape(-1)
    val_pred = (val_scores >= 0.5).astype(int)
    try:
        val_roc_auc = float(roc_auc_score(y_val, val_scores))
    except ValueError:
        val_roc_auc = float("nan")
    meta = {
        "hidden1": hidden1,
        "hidden2": hidden2,
        "activation": "tanh",
        "output_activation": "sigmoid",
        "optimizer": optimizer_name,
        "learning_rate": 0.001,
        "paper_epochs": dnn_epochs(n_features),
        "epochs_configured": epochs,
        "epochs_ran": len(history.history.get("loss", [])),
        "early_stopping_monitor": "val_pr_auc",
        "early_stopping_patience": patience,
        "class_weight": class_weight,
        "val_pr_auc": float(average_precision_score(y_val, val_scores)),
        "val_roc_auc": val_roc_auc,
        "val_macro_f1": float(f1_score(y_val, val_pred, average="macro", zero_division=0)),
    }
    return test_scores, (test_scores >= 0.5).astype(int), meta
