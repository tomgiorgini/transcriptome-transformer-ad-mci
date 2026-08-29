from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from imblearn.over_sampling import BorderlineSMOTE


def apply_sampling(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    sampling: str,
    seed: int,
) -> tuple[pd.DataFrame, np.ndarray, dict[str, Any]]:
    before = {str(k): int(v) for k, v in zip(*np.unique(y_train, return_counts=True))}
    if sampling == "no_smote":
        return x_train.copy(), y_train.copy(), {"sampling": sampling, "scope": "none", "class_counts_before": before, "class_counts_after": before}
    if sampling != "borderline_smote":
        raise ValueError(f"Unsupported sampling mode: {sampling}")
    counts = np.bincount(y_train.astype(int), minlength=2)
    minority = int(counts.min())
    if minority < 3:
        return (
            x_train.copy(),
            y_train.copy(),
            {
                "sampling": sampling,
                "scope": "skipped",
                "reason": "too_few_minority_samples",
                "class_counts_before": before,
                "class_counts_after": before,
            },
        )
    k_neighbors = max(1, min(5, minority - 1))
    sampler = BorderlineSMOTE(random_state=seed, k_neighbors=k_neighbors)
    x_res, y_res = sampler.fit_resample(x_train, y_train)
    after = {str(k): int(v) for k, v in zip(*np.unique(y_res, return_counts=True))}
    return (
        pd.DataFrame(x_res, columns=x_train.columns),
        y_res.astype(int),
        {
            "sampling": sampling,
            "scope": "train_inner_only",
            "sampler": "BorderlineSMOTE",
            "k_neighbors": int(k_neighbors),
            "class_counts_before": before,
            "class_counts_after": after,
        },
    )
