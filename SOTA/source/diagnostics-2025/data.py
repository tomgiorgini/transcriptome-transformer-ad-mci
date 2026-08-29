from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


def load_ad_mci_dataset(x_file: Path, y_file: Path) -> tuple[pd.DataFrame, pd.Series]:
    x = pd.read_csv(x_file, index_col=0)
    y_df = pd.read_csv(y_file, index_col=0)
    if "label" not in y_df.columns:
        raise ValueError(f"{y_file} must contain a 'label' column.")
    common = x.index.intersection(y_df.index)
    if len(common) == 0:
        raise ValueError("X and y have no overlapping sample IDs.")
    x = x.loc[common].copy()
    y = y_df.loc[common, "label"].astype(int)
    if set(y.unique()) != {0, 1}:
        raise ValueError(f"Expected binary labels 0/1, got {sorted(y.unique())}.")
    return x, y


def read_geo_metadata_sample_ids(path: Path) -> set[str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing GEO metadata file: {path}")
    md = pd.read_csv(path, sep="\t", compression="infer", dtype=str)
    for candidate in ("sample_id", "geo_accession"):
        if candidate in md.columns:
            return set(md[candidate].astype(str).str.strip())
    raise ValueError(f"{path} must contain sample_id or geo_accession.")


def infer_batch_labels(sample_ids: pd.Index, gse63060_metadata: Path, gse63061_metadata: Path) -> pd.Series:
    gse63060_ids = read_geo_metadata_sample_ids(gse63060_metadata)
    gse63061_ids = read_geo_metadata_sample_ids(gse63061_metadata)
    labels: dict[str, str] = {}
    for sample_id in sample_ids.astype(str):
        gsm_id = sample_id.split("_", 1)[0].strip()
        if gsm_id in gse63060_ids:
            labels[sample_id] = "GSE63060"
        elif gsm_id in gse63061_ids:
            labels[sample_id] = "GSE63061"
        else:
            labels[sample_id] = "unknown"
    batch = pd.Series(labels, index=sample_ids, name="batch")
    unknown = batch[batch == "unknown"]
    if not unknown.empty:
        preview = ", ".join(unknown.index.astype(str)[:10])
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
) -> list[dict[str, Any]]:
    y_values = y.to_numpy(dtype=np.int64)
    all_idx = np.arange(len(y_values))
    batch_values = batch.to_numpy()
    splits: list[dict[str, Any]] = []
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
                raise ValueError(f"Unsupported batch scenario: {scenario}")
            train_idx, val_idx = train_test_split(
                np.asarray(pool_idx, dtype=np.int64),
                test_size=inner_val_ratio,
                random_state=repeat_seed * 100 + scenario_idx,
                stratify=y_values[pool_idx],
            )
            splits.append(
                {
                    "scenario": scenario,
                    "scenario_index": scenario_idx,
                    "repeat": repeat,
                    "fold": 1,
                    "seed": repeat_seed,
                    "train_inner_idx": np.asarray(train_idx, dtype=np.int64),
                    "val_inner_idx": np.asarray(val_idx, dtype=np.int64),
                    "outer_test_idx": np.asarray(test_idx, dtype=np.int64),
                }
            )
    return splits


def make_stratified_5cv_splits(
    y: pd.Series,
    repeats: int,
    inner_val_ratio: float,
    seed: int,
    outer_folds: int = 5,
) -> list[dict[str, Any]]:
    y_values = y.to_numpy(dtype=np.int64)
    all_idx = np.arange(len(y_values))
    splits: list[dict[str, Any]] = []
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
                {
                    "scenario": "shared_test",
                    "scenario_index": 1,
                    "repeat": repeat,
                    "fold": fold,
                    "seed": repeat_seed,
                    "train_inner_idx": np.asarray(train_idx, dtype=np.int64),
                    "val_inner_idx": np.asarray(val_idx, dtype=np.int64),
                    "outer_test_idx": np.asarray(test_idx, dtype=np.int64),
                }
            )
    return splits


def assert_no_overlap(split: dict[str, Any]) -> None:
    train = set(split["train_inner_idx"].tolist())
    val = set(split["val_inner_idx"].tolist())
    test = set(split["outer_test_idx"].tolist())
    if train & val or train & test or val & test:
        raise ValueError(f"Split overlap detected for scenario={split['scenario']} repeat={split['repeat']}.")
