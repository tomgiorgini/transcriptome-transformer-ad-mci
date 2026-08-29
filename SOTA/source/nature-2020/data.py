from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


@dataclass(frozen=True)
class BatchSplit:
    scenario: str
    repeat: int
    fold: int
    seed: int
    train_inner_idx: np.ndarray
    val_inner_idx: np.ndarray
    outer_test_idx: np.ndarray


def load_binary_dataset(x_file: Path, y_file: Path) -> tuple[pd.DataFrame, pd.Series]:
    x_df = pd.read_csv(x_file)
    y_df = pd.read_csv(y_file)
    if "sample_id" not in x_df.columns:
        raise ValueError(f"{x_file} must contain sample_id.")
    if "sample_id" not in y_df.columns or "label" not in y_df.columns:
        raise ValueError(f"{y_file} must contain sample_id and label.")
    x_df["sample_id"] = x_df["sample_id"].astype(str).str.strip()
    y_df["sample_id"] = y_df["sample_id"].astype(str).str.strip()
    merged = x_df.merge(y_df[["sample_id", "label"]], on="sample_id", how="inner", validate="one_to_one")
    if len(merged) != len(x_df) or len(merged) != len(y_df):
        raise ValueError("X and y sample_id sets do not match.")
    y = pd.Series(merged["label"].astype(int).to_numpy(), index=merged["sample_id"].astype(str), name="label")
    if sorted(y.unique().tolist()) != [0, 1]:
        raise ValueError(f"Expected binary labels encoded as 0 and 1, got {sorted(y.unique().tolist())}.")
    gene_cols = [c for c in merged.columns if c not in {"sample_id", "label"}]
    x = merged[gene_cols].apply(pd.to_numeric, errors="coerce")
    x.index = y.index
    return x, y


# Backwards-compatible alias for scripts written before the runner supported all
# three pairwise tasks.
load_ad_mci_dataset = load_binary_dataset


def read_geo_metadata_sample_ids(path: Path) -> set[str]:
    md = pd.read_csv(path, sep="\t", compression="infer", dtype=str)
    for candidate in ("sample_id", "geo_accession"):
        if candidate in md.columns:
            return set(md[candidate].astype(str).str.strip())
    raise ValueError(f"{path} must contain sample_id or geo_accession.")


def infer_batch_labels(sample_ids: pd.Index, gse63060_metadata: Path, gse63061_metadata: Path) -> pd.Series:
    ids_63060 = read_geo_metadata_sample_ids(gse63060_metadata)
    ids_63061 = read_geo_metadata_sample_ids(gse63061_metadata)
    labels: dict[str, str] = {}
    for sample_id in sample_ids.astype(str):
        gsm_id = sample_id.split("_", 1)[0]
        if gsm_id in ids_63060:
            labels[sample_id] = "GSE63060"
        elif gsm_id in ids_63061:
            labels[sample_id] = "GSE63061"
        else:
            labels[sample_id] = "unknown"
    batch = pd.Series(labels, index=sample_ids, name="batch")
    unknown = batch[batch == "unknown"]
    if not unknown.empty:
        preview = ", ".join(unknown.index[:10].astype(str))
        raise ValueError(f"Could not map {len(unknown)} samples to batches. Examples: {preview}")
    return batch


def make_batch_holdout_splits(
    y: pd.Series,
    batch: pd.Series,
    scenarios: list[str],
    repeats: int,
    inner_val_ratio: float,
    shared_test_size: float,
    seed: int,
) -> list[BatchSplit]:
    y_values = y.to_numpy(dtype=np.int64)
    batch_values = batch.to_numpy()
    all_idx = np.arange(len(y_values))
    splits: list[BatchSplit] = []
    for repeat in range(1, repeats + 1):
        repeat_seed = seed + repeat - 1
        for scenario_idx, scenario in enumerate(scenarios, start=1):
            if scenario == "test_gse63060":
                pool_idx = all_idx[batch_values == "GSE63061"]
                test_idx = all_idx[batch_values == "GSE63060"]
            elif scenario == "test_gse63061":
                pool_idx = all_idx[batch_values == "GSE63060"]
                test_idx = all_idx[batch_values == "GSE63061"]
            elif scenario == "shared_test":
                train_parts: list[np.ndarray] = []
                test_parts: list[np.ndarray] = []
                for batch_name in ("GSE63060", "GSE63061"):
                    batch_idx = all_idx[batch_values == batch_name]
                    train_part, test_part = train_test_split(
                        batch_idx,
                        test_size=shared_test_size,
                        random_state=repeat_seed,
                        stratify=y_values[batch_idx],
                    )
                    train_parts.append(np.asarray(train_part, dtype=np.int64))
                    test_parts.append(np.asarray(test_part, dtype=np.int64))
                pool_idx = np.concatenate(train_parts)
                test_idx = np.concatenate(test_parts)
            else:
                raise ValueError(f"Unsupported scenario: {scenario}")
            train_idx, val_idx = train_test_split(
                np.asarray(pool_idx, dtype=np.int64),
                test_size=inner_val_ratio,
                random_state=repeat_seed * 100 + scenario_idx,
                stratify=y_values[pool_idx],
            )
            splits.append(
                BatchSplit(
                    scenario=scenario,
                    repeat=repeat,
                    fold=1,
                    seed=repeat_seed * 1000 + scenario_idx * 100 + repeat,
                    train_inner_idx=np.asarray(train_idx, dtype=np.int64),
                    val_inner_idx=np.asarray(val_idx, dtype=np.int64),
                    outer_test_idx=np.asarray(test_idx, dtype=np.int64),
                )
            )
    return splits


def make_stratified_5cv_splits(
    y: pd.Series,
    repeats: int,
    inner_val_ratio: float,
    seed: int,
    outer_folds: int = 5,
) -> list[BatchSplit]:
    y_values = y.to_numpy(dtype=np.int64)
    all_idx = np.arange(len(y_values))
    splits: list[BatchSplit] = []
    for repeat in range(1, repeats + 1):
        repeat_seed = seed + repeat - 1
        cv = StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=repeat_seed)
        for fold, (pool_idx, test_idx) in enumerate(cv.split(all_idx, y_values), start=1):
            train_idx, val_idx = train_test_split(
                np.asarray(pool_idx, dtype=np.int64),
                test_size=inner_val_ratio,
                random_state=repeat_seed * 100 + fold,
                stratify=y_values[pool_idx],
            )
            splits.append(
                BatchSplit(
                    scenario="shared_test",
                    repeat=repeat,
                    fold=fold,
                    seed=repeat_seed * 1000 + fold,
                    train_inner_idx=np.asarray(train_idx, dtype=np.int64),
                    val_inner_idx=np.asarray(val_idx, dtype=np.int64),
                    outer_test_idx=np.asarray(test_idx, dtype=np.int64),
                )
            )
    return splits


def assert_no_split_overlap(split: BatchSplit) -> None:
    train = set(split.train_inner_idx.tolist())
    val = set(split.val_inner_idx.tolist())
    test = set(split.outer_test_idx.tolist())
    if train & val or train & test or val & test:
        raise ValueError(f"Split overlap detected for {split.scenario} repeat={split.repeat}.")
