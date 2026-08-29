from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


@dataclass(frozen=True)
class FoldSplit:
    repeat: int
    fold: int
    train_inner_idx: np.ndarray
    val_inner_idx: np.ndarray
    outer_test_idx: np.ndarray
    seed: int


def load_ad_mci_dataset(x_file: Path, y_file: Path) -> tuple[pd.DataFrame, pd.Series]:
    x_df = pd.read_csv(x_file)
    y_df = pd.read_csv(y_file)

    if "sample_id" not in x_df.columns:
        raise ValueError(f"{x_file} must contain sample_id.")
    if "sample_id" not in y_df.columns or "label" not in y_df.columns:
        raise ValueError(f"{y_file} must contain sample_id and label.")

    x_df["sample_id"] = x_df["sample_id"].astype(str).str.strip()
    y_df["sample_id"] = y_df["sample_id"].astype(str).str.strip()
    if x_df["sample_id"].duplicated().any():
        raise ValueError("X file contains duplicated sample_id values.")
    if y_df["sample_id"].duplicated().any():
        raise ValueError("y file contains duplicated sample_id values.")

    merged = x_df.merge(y_df[["sample_id", "label"]], on="sample_id", how="inner", validate="one_to_one")
    if len(merged) != len(x_df) or len(merged) != len(y_df):
        raise ValueError("X and y sample_id sets do not match.")

    labels = pd.to_numeric(merged["label"], errors="raise").astype(int)
    unique_labels = sorted(labels.unique().tolist())
    if unique_labels != [0, 1]:
        raise ValueError(f"Expected binary labels [0, 1], found {unique_labels}.")

    gene_cols = [col for col in merged.columns if col not in {"sample_id", "label"}]
    x = merged[gene_cols].apply(pd.to_numeric, errors="coerce")
    x.index = merged["sample_id"].astype(str)
    y = pd.Series(labels.to_numpy(dtype=np.int64), index=x.index, name="label")
    return x, y


def repeated_stratified_nested_splits(
    y: pd.Series,
    repeats: int,
    outer_folds: int,
    inner_val_ratio: float,
    seed: int,
    max_outer_folds: int | None = None,
) -> list[FoldSplit]:
    splits: list[FoldSplit] = []
    y_values = y.to_numpy()
    all_idx = np.arange(len(y_values))

    for repeat in range(1, repeats + 1):
        repeat_seed = seed + repeat - 1
        outer_cv = StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=repeat_seed)
        for fold, (pool_idx, outer_test_idx) in enumerate(outer_cv.split(all_idx, y_values), start=1):
            train_inner_idx, val_inner_idx = train_test_split(
                pool_idx,
                test_size=inner_val_ratio,
                random_state=repeat_seed * 100 + fold,
                stratify=y_values[pool_idx],
            )
            splits.append(
                FoldSplit(
                    repeat=repeat,
                    fold=fold,
                    train_inner_idx=np.asarray(train_inner_idx, dtype=np.int64),
                    val_inner_idx=np.asarray(val_inner_idx, dtype=np.int64),
                    outer_test_idx=np.asarray(outer_test_idx, dtype=np.int64),
                    seed=repeat_seed,
                )
            )
            if max_outer_folds is not None and fold >= max_outer_folds:
                break
    return splits


def assert_no_split_overlap(split: FoldSplit) -> None:
    train = set(split.train_inner_idx.tolist())
    val = set(split.val_inner_idx.tolist())
    test = set(split.outer_test_idx.tolist())
    if train & val or train & test or val & test:
        raise ValueError(f"Split overlap detected for repeat={split.repeat}, fold={split.fold}.")
