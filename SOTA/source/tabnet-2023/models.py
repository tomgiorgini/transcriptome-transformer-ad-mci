from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from pytorch_tabnet.tab_model import TabNetClassifier

from metrics import evaluate_binary
import sys
COMMON_DIR = Path(__file__).resolve().parents[1]
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))
from strict_v2_utils import calibrate_threshold


def default_tabnet_params(seed: int) -> dict[str, Any]:
    return {
        "n_d": 32,
        "n_a": 36,
        "n_steps": 4,
        "gamma": 1.3,
        "n_shared": 2,
        "lambda_sparse": 0.0011,
        "optimizer_fn": torch.optim.Adam,
        "optimizer_params": {"lr": 1e-3},
        "seed": seed,
        "verbose": 1,
        "device_name": "auto",
    }


def train_dgs_tabnet(
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
) -> tuple[np.ndarray, np.ndarray, dict[str, float], pd.DataFrame, dict[str, Any]]:
    params = default_tabnet_params(seed)
    model = TabNetClassifier(**params)
    x_train_np = x_train.to_numpy(dtype=np.float32)
    x_val_np = x_val.to_numpy(dtype=np.float32)
    x_test_np = x_test.to_numpy(dtype=np.float32)
    y_train_np = y_train.astype(int)
    y_val_np = y_val.astype(int)
    model.fit(
        X_train=x_train_np,
        y_train=y_train_np,
        eval_set=[(x_val_np, y_val_np)],
        eval_name=["val"],
        eval_metric=["auc"],
        max_epochs=max_epochs,
        patience=patience,
        batch_size=batch_size,
        virtual_batch_size=batch_size,
        drop_last=True,
    )
    val_proba = model.predict_proba(x_val_np)
    if threshold_mode == "fixed_0_5":
        threshold = 0.5
        threshold_meta = {"threshold": threshold, "threshold_mode": "fixed_0_5"}
    else:
        metric = threshold_mode.replace("validation_", "")
        threshold, threshold_meta = calibrate_threshold(y_val, val_proba[:, 1], metric=metric)
        threshold_meta["threshold_mode"] = threshold_mode
    proba = model.predict_proba(x_test_np)
    scores = np.nan_to_num(proba[:, 1].astype(float), nan=0.5, posinf=1.0, neginf=0.0)
    y_pred = (scores >= threshold).astype(int)
    metrics = evaluate_binary(y_test, scores, y_pred)
    importances = np.asarray(model.feature_importances_, dtype=float)
    importance_frame = pd.DataFrame({"gene": x_train.columns.astype(str), "importance": importances})
    importance_frame = importance_frame.sort_values("importance", ascending=False)
    model_path = output_dir / "dgs_tabnet_model"
    saved_path = model.save_model(str(model_path))
    metadata = {
        "model": "DGS-TabNet",
        "library": "pytorch-tabnet",
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "max_epochs": max_epochs,
        "patience": patience,
        "batch_size": batch_size,
        "drop_last": True,
        "checkpoint_score": "pytorch-tabnet native early stopping on validation AUC; strict_v2 composite not directly supported by library callback in this wrapper",
        "hyperparameters": {
            key: value
            for key, value in params.items()
            if key not in {"optimizer_fn"}
        },
        "optimizer": "Adam",
        "saved_model": saved_path,
        "threshold_calibration": threshold_meta,
    }
    return scores, y_pred, metrics, importance_frame, metadata
