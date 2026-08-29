from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class TrainingBalanceResult:
    x_train: pd.DataFrame
    y_train: np.ndarray
    manifest: dict[str, Any]


def _counts(y: np.ndarray) -> dict[str, int]:
    return {
        str(label): int(count)
        for label, count in pd.Series(y).value_counts().sort_index().to_dict().items()
    }


def balance_training_data(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    mode: str,
    seed: int,
) -> TrainingBalanceResult:
    y_values = np.asarray(y_train, dtype=np.int32)
    if len(x_train) != len(y_values):
        raise ValueError("x_train and y_train have different lengths.")
    if mode == "none":
        return TrainingBalanceResult(
            x_train=x_train.copy(),
            y_train=y_values.copy(),
            manifest={
                "training_balance": "none",
                "balance_scope": "not_applied",
                "seed": int(seed),
                "n_before": int(len(y_values)),
                "n_after": int(len(y_values)),
                "class_counts_before": _counts(y_values),
                "class_counts_after": _counts(y_values),
                "validation_and_test_modified": False,
            },
        )
    if mode != "undersample":
        raise ValueError(f"Unsupported training balance mode: {mode}")

    labels, counts = np.unique(y_values, return_counts=True)
    if len(labels) < 2:
        raise ValueError("Training-only undersampling requires at least two classes.")
    target_count = int(counts.min())
    rng = np.random.default_rng(seed)
    kept_positions: list[np.ndarray] = []
    for label in labels:
        positions = np.flatnonzero(y_values == label)
        if len(positions) > target_count:
            positions = rng.choice(positions, size=target_count, replace=False)
        kept_positions.append(np.asarray(positions, dtype=int))
    selected = np.concatenate(kept_positions)
    selected = rng.permutation(selected)
    x_balanced = x_train.iloc[selected].copy()
    y_balanced = y_values[selected]
    return TrainingBalanceResult(
        x_train=x_balanced,
        y_train=y_balanced,
        manifest={
            "training_balance": "undersample",
            "balance_scope": "train_inner_only",
            "implementation": "deterministic random undersampling of every class to the train-inner minority count",
            "paper_alignment": "The paper undersamples AD to the CTL count before evaluation; this shared-test adaptation undersamples train_inner only so validation and test membership remain unchanged.",
            "seed": int(seed),
            "target_count_per_class": target_count,
            "n_before": int(len(y_values)),
            "n_after": int(len(y_balanced)),
            "class_counts_before": _counts(y_values),
            "class_counts_after": _counts(y_balanced),
            "selected_train_ids": x_balanced.index.astype(str).tolist(),
            "validation_and_test_modified": False,
        },
    )
