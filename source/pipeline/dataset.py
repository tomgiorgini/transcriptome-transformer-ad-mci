from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class ArrayDataset(Dataset):
    def __init__(self, gene_x: np.ndarray, y: np.ndarray):
        self.gene_x = torch.tensor(gene_x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.gene_x[idx], self.y[idx]


@dataclass
class PreparedDataset:
    class_names: list[str]
    gene_names: list[str]
    train_ids: np.ndarray
    val_ids: np.ndarray
    test_ids: np.ndarray
    train_gene_x: np.ndarray
    val_gene_x: np.ndarray
    test_gene_x: np.ndarray
    train_y: np.ndarray
    val_y: np.ndarray
    test_y: np.ndarray


def load_multiclass_dataset(x_file: Path, y_file: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[str]]:
    x_df = pd.read_csv(x_file)
    y_df = pd.read_csv(y_file)

    if "sample_id" not in x_df.columns:
        raise ValueError("X.csv must contain a sample_id column.")
    if "sample_id" not in y_df.columns or "label" not in y_df.columns:
        raise ValueError("y.csv must contain sample_id and label columns.")

    x_df["sample_id"] = x_df["sample_id"].astype(str).str.strip()
    y_df["sample_id"] = y_df["sample_id"].astype(str).str.strip()

    duplicate_x = x_df.loc[x_df["sample_id"].duplicated(keep=False), "sample_id"].unique().tolist()
    if duplicate_x:
        preview = ", ".join(duplicate_x[:10])
        raise ValueError(f"X.csv contains duplicated sample_id values. Example: {preview}")

    duplicate_y = y_df.loc[y_df["sample_id"].duplicated(keep=False), "sample_id"].unique().tolist()
    if duplicate_y:
        preview = ", ".join(duplicate_y[:10])
        raise ValueError(f"y.csv contains duplicated sample_id values. Example: {preview}")

    missing_in_y = pd.Index(x_df["sample_id"]).difference(pd.Index(y_df["sample_id"]))
    missing_in_x = pd.Index(y_df["sample_id"]).difference(pd.Index(x_df["sample_id"]))
    if len(missing_in_y) > 0 or len(missing_in_x) > 0:
        problems = []
        if len(missing_in_y) > 0:
            problems.append(f"present in X only: {', '.join(missing_in_y[:10])}")
        if len(missing_in_x) > 0:
            problems.append(f"present in y only: {', '.join(missing_in_x[:10])}")
        raise ValueError("X.csv and y.csv do not contain the same sample_id set: " + " | ".join(problems))

    y_cols = ["sample_id", "label"]
    if "label_name" in y_df.columns:
        y_cols.append("label_name")

    merged = x_df.merge(y_df[y_cols], on="sample_id", how="inner", validate="one_to_one")

    gene_names = [column for column in merged.columns if column not in {"sample_id", "label", "label_name"}]
    x = merged[gene_names].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)

    label_values = pd.to_numeric(merged["label"], errors="raise").to_numpy(dtype=np.int64)
    unique_labels = np.sort(np.unique(label_values))
    label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
    y = np.array([label_to_index[label] for label in label_values], dtype=np.int64)

    if "label_name" in merged.columns:
        class_names: list[str] = []
        for label in unique_labels:
            names = merged.loc[merged["label"] == label, "label_name"].dropna().astype(str).unique().tolist()
            class_names.append(names[0] if names else f"class_{label}")
    else:
        class_names = [f"class_{label}" for label in unique_labels]

    sample_ids = merged["sample_id"].to_numpy()
    return sample_ids, x, y, gene_names, class_names


def stratified_split(y: np.ndarray, val_ratio: float, test_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []

    for label in np.unique(y):
        idx = np.where(y == label)[0]
        idx = rng.permutation(idx)
        n = len(idx)
        n_test = int(round(n * test_ratio))
        n_val = int(round(n * val_ratio))

        while n_test + n_val >= n and (n_test > 0 or n_val > 0):
            if n_test >= n_val and n_test > 0:
                n_test -= 1
            elif n_val > 0:
                n_val -= 1

        test_parts.append(idx[:n_test])
        val_parts.append(idx[n_test : n_test + n_val])
        train_parts.append(idx[n_test + n_val :])

    train_idx = np.random.default_rng(seed).permutation(np.concatenate(train_parts))
    val_idx = np.random.default_rng(seed + 1).permutation(np.concatenate(val_parts))
    test_idx = np.random.default_rng(seed + 2).permutation(np.concatenate(test_parts))
    return train_idx, val_idx, test_idx


def random_split(n_samples: int, val_ratio: float, test_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if n_samples < 3:
        raise ValueError("At least three samples are required for random train/val/test split.")
    if not 0.0 <= val_ratio < 1.0 or not 0.0 <= test_ratio < 1.0 or val_ratio + test_ratio >= 1.0:
        raise ValueError("val_ratio and test_ratio must be non-negative and sum to less than 1.")

    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_samples)
    n_test = int(round(n_samples * test_ratio))
    n_val = int(round(n_samples * val_ratio))
    n_test = min(max(n_test, 1), n_samples - 2)
    n_val = min(max(n_val, 1), n_samples - n_test - 1)

    test_idx = indices[:n_test]
    val_idx = indices[n_test : n_test + n_val]
    train_idx = indices[n_test + n_val :]
    return train_idx, val_idx, test_idx


def build_split_manifest(sample_ids: np.ndarray, train_idx: np.ndarray, val_idx: np.ndarray, test_idx: np.ndarray) -> pd.DataFrame:
    rows = (
        [{"sample_id": sample_ids[idx], "split": "train"} for idx in train_idx]
        + [{"sample_id": sample_ids[idx], "split": "val"} for idx in val_idx]
        + [{"sample_id": sample_ids[idx], "split": "test"} for idx in test_idx]
    )
    return pd.DataFrame(rows)


def load_split_indices(split_file: Path, sample_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    split_df = pd.read_csv(split_file)
    if "sample_id" not in split_df.columns or "split" not in split_df.columns:
        raise ValueError("split file must contain sample_id and split columns.")

    split_df["sample_id"] = split_df["sample_id"].astype(str).str.strip()
    split_df["split"] = split_df["split"].astype(str).str.strip().str.lower()

    duplicate_ids = split_df.loc[split_df["sample_id"].duplicated(keep=False), "sample_id"].unique().tolist()
    if duplicate_ids:
        preview = ", ".join(duplicate_ids[:10])
        raise ValueError(f"split file contains duplicated sample_id values. Example: {preview}")

    invalid_splits = sorted(set(split_df["split"]) - {"train", "val", "test"})
    if invalid_splits:
        raise ValueError(f"split file contains invalid split names: {', '.join(invalid_splits)}")

    expected_ids = pd.Index(sample_ids.astype(str))
    received_ids = pd.Index(split_df["sample_id"])

    missing_ids = expected_ids.difference(received_ids)
    extra_ids = received_ids.difference(expected_ids)
    if len(missing_ids) > 0 or len(extra_ids) > 0:
        problems = []
        if len(missing_ids) > 0:
            problems.append(f"missing sample ids: {', '.join(missing_ids[:10])}")
        if len(extra_ids) > 0:
            problems.append(f"unexpected sample ids: {', '.join(extra_ids[:10])}")
        raise ValueError("split file does not align with dataset sample_id values: " + " | ".join(problems))

    sample_to_idx = {sample_id: idx for idx, sample_id in enumerate(sample_ids.astype(str))}
    train_idx = np.array([sample_to_idx[sample_id] for sample_id in split_df.loc[split_df["split"] == "train", "sample_id"]], dtype=np.int64)
    val_idx = np.array([sample_to_idx[sample_id] for sample_id in split_df.loc[split_df["split"] == "val", "sample_id"]], dtype=np.int64)
    test_idx = np.array([sample_to_idx[sample_id] for sample_id in split_df.loc[split_df["split"] == "test", "sample_id"]], dtype=np.int64)
    return train_idx, val_idx, test_idx


def impute_missing_from_train(train_x: np.ndarray, other_arrays: list[np.ndarray]) -> tuple[np.ndarray, list[np.ndarray]]:
    medians = np.nanmedian(train_x, axis=0)
    medians = np.where(np.isnan(medians), 0.0, medians)

    arrays = [train_x, *other_arrays]
    imputed_arrays: list[np.ndarray] = []
    for array in arrays:
        array_copy = array.copy()
        rows, cols = np.where(np.isnan(array_copy))
        if len(rows) > 0:
            array_copy[rows, cols] = medians[cols]
        imputed_arrays.append(array_copy.astype(np.float32))

    return imputed_arrays[0], imputed_arrays[1:]


def select_top_variable_genes(
    train_x: np.ndarray,
    other_arrays: list[np.ndarray],
    gene_names: list[str],
    max_genes: int | None,
    train_y: np.ndarray | None = None,
    selection: str = "variance",
) -> tuple[np.ndarray, list[np.ndarray], list[str]]:
    if max_genes is None or max_genes <= 0 or train_x.shape[1] <= max_genes:
        return train_x.astype(np.float32), [array.astype(np.float32) for array in other_arrays], gene_names

    if selection == "variance":
        scores = np.var(train_x, axis=0)
    elif selection == "mad":
        medians = np.median(train_x, axis=0)
        scores = np.median(np.abs(train_x - medians), axis=0)
    elif selection == "class_aware_variance":
        if train_y is None:
            raise ValueError("train_y is required for class-aware variance gene selection.")
        overall_mean = train_x.mean(axis=0)
        scores = np.zeros(train_x.shape[1], dtype=np.float64)
        for label in np.unique(train_y):
            class_x = train_x[train_y == label]
            if class_x.size == 0:
                continue
            class_mean = class_x.mean(axis=0)
            scores += class_x.shape[0] * np.square(class_mean - overall_mean)
        scores /= max(train_x.shape[0], 1)
    else:
        raise ValueError(f"Unsupported gene selection mode: {selection}")

    selected_idx = np.argpartition(scores, -max_genes)[-max_genes:]
    selected_idx = selected_idx[np.argsort(scores[selected_idx])[::-1]]

    selected_genes = [gene_names[idx] for idx in selected_idx]
    filtered_train = train_x[:, selected_idx].astype(np.float32)
    filtered_others = [array[:, selected_idx].astype(np.float32) for array in other_arrays]
    return filtered_train, filtered_others, selected_genes


def scale_arrays(
    train_x: np.ndarray,
    other_arrays: list[np.ndarray],
    scaler: str,
    fit_scope: str = "train",
) -> tuple[np.ndarray, list[np.ndarray]]:
    if scaler == "none":
        return train_x.astype(np.float32), [array.astype(np.float32) for array in other_arrays]

    if fit_scope == "train":
        fit_x = train_x
    elif fit_scope == "all":
        fit_x = np.concatenate([train_x, *other_arrays], axis=0)
    else:
        raise ValueError(f"Unsupported scaler fit scope: {fit_scope}")

    if scaler == "minmax":
        min_v = fit_x.min(axis=0)
        max_v = fit_x.max(axis=0)
        scale = np.where((max_v - min_v) == 0, 1.0, max_v - min_v)
        scaled_train = (train_x - min_v) / scale
        scaled_others = [(array - min_v) / scale for array in other_arrays]
        return scaled_train.astype(np.float32), [array.astype(np.float32) for array in scaled_others]

    if scaler == "standard":
        mean = fit_x.mean(axis=0)
        std = fit_x.std(axis=0)
        std = np.where(std == 0, 1.0, std)
        scaled_train = (train_x - mean) / std
        scaled_others = [(array - mean) / std for array in other_arrays]
        return scaled_train.astype(np.float32), [array.astype(np.float32) for array in scaled_others]

    raise ValueError(f"Unsupported scaler: {scaler}")


def prepare_dataset(
    x_file: Path,
    y_file: Path,
    seed: int,
    val_ratio: float,
    test_ratio: float,
    max_genes: int | None,
    scaler: str,
    scaler_fit_scope: str = "train",
    split_file: Path | None = None,
    split_seed: int | None = None,
    split_mode: str = "stratified",
    gene_selection: str = "variance",
    candidate_genes: list[str] | None = None,
) -> PreparedDataset:
    sample_ids, gene_x, y, gene_names, class_names = load_multiclass_dataset(x_file, y_file)
    if candidate_genes is not None:
        candidate_set = {str(gene).strip() for gene in candidate_genes}
        keep_idx = [idx for idx, gene in enumerate(gene_names) if str(gene).strip() in candidate_set]
        if not keep_idx:
            raise ValueError("No expression genes overlap the requested candidate gene set.")
        gene_x = gene_x[:, keep_idx]
        gene_names = [gene_names[idx] for idx in keep_idx]

    if split_file is not None:
        train_idx, val_idx, test_idx = load_split_indices(split_file, sample_ids)
    else:
        effective_split_seed = seed if split_seed is None else split_seed
        if split_mode == "stratified":
            train_idx, val_idx, test_idx = stratified_split(y, val_ratio, test_ratio, effective_split_seed)
        elif split_mode == "random":
            train_idx, val_idx, test_idx = random_split(len(y), val_ratio, test_ratio, effective_split_seed)
        else:
            raise ValueError(f"Unsupported split_mode: {split_mode}")

    if len(val_idx) == 0:
        raise ValueError("Validation split is empty. Adjust the split configuration.")

    train_gene_x = gene_x[train_idx]
    val_gene_x = gene_x[val_idx]
    test_gene_x = gene_x[test_idx]
    train_y = y[train_idx]
    val_y = y[val_idx]
    test_y = y[test_idx]

    train_gene_x, [val_gene_x, test_gene_x] = impute_missing_from_train(train_gene_x, [val_gene_x, test_gene_x])
    train_gene_x, [val_gene_x, test_gene_x], gene_names = select_top_variable_genes(
        train_gene_x,
        [val_gene_x, test_gene_x],
        gene_names,
        max_genes,
        train_y=train_y,
        selection=gene_selection,
    )
    train_gene_x, [val_gene_x, test_gene_x] = scale_arrays(
        train_gene_x,
        [val_gene_x, test_gene_x],
        scaler,
        fit_scope=scaler_fit_scope,
    )

    return PreparedDataset(
        class_names=class_names,
        gene_names=gene_names,
        train_ids=sample_ids[train_idx],
        val_ids=sample_ids[val_idx],
        test_ids=sample_ids[test_idx],
        train_gene_x=train_gene_x,
        val_gene_x=val_gene_x,
        test_gene_x=test_gene_x,
        train_y=train_y,
        val_y=val_y,
        test_y=test_y,
    )
