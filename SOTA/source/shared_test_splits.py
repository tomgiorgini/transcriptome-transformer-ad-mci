from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


TXT_SHARED_SEEDS: tuple[int, ...] = tuple(range(101, 111))
VALID_SPLITS = ("train", "val", "test")


def _load_labels(path: Path) -> pd.DataFrame:
    labels = pd.read_csv(path)
    required = {"sample_id", "label"}
    missing = required.difference(labels.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    labels = labels[["sample_id", "label"]].copy()
    labels["sample_id"] = labels["sample_id"].astype(str).str.strip()
    if labels["sample_id"].duplicated().any():
        duplicated = labels.loc[labels["sample_id"].duplicated(), "sample_id"].iloc[0]
        raise ValueError(f"{path} contains duplicated sample_id={duplicated!r}.")
    labels["label"] = pd.to_numeric(labels["label"], errors="raise").astype(int)
    return labels


def write_txt_shared_splits(
    shared_y_file: Path,
    output_dir: Path,
    seeds: Sequence[int] = TXT_SHARED_SEEDS,
    *,
    overwrite: bool = False,
) -> list[Path]:
    """Write the exact 70/10/20 manifests used by the TxT seed pipeline.

    The two calls and random states intentionally mirror
    ``write_seed_splits`` in the TxT paper-comparison runner.  Manifests are
    generated on the shared three-class cohort; pairwise SOTA runners filter
    them by sample_id so every method sees exactly the same held-out samples.
    """

    shared_y_file = Path(shared_y_file).resolve()
    output_dir = Path(output_dir).resolve()
    labels_df = _load_labels(shared_y_file)
    sample_ids = labels_df["sample_id"]
    output_dir.mkdir(parents=True, exist_ok=True)

    paths: list[Path] = []
    for raw_seed in seeds:
        seed = int(raw_seed)
        path = output_dir / f"seed_{seed}.csv"
        expected_manifest = _expected_txt_manifest(labels_df, seed)
        if path.exists() and not overwrite:
            _validate_full_manifest(path, set(sample_ids), expected_manifest=expected_manifest)
            paths.append(path)
            continue
        expected_manifest.to_csv(path, index=False)
        _validate_full_manifest(path, set(sample_ids), expected_manifest=expected_manifest)
        paths.append(path)
    return paths


def _expected_txt_manifest(labels_df: pd.DataFrame, seed: int) -> pd.DataFrame:
    sample_ids = labels_df["sample_id"]
    labels = labels_df["label"].to_numpy()
    all_idx = labels_df.index.to_numpy()
    train_idx, holdout_idx = train_test_split(
        all_idx,
        test_size=0.30,
        random_state=seed,
        stratify=labels,
    )
    val_idx, test_idx = train_test_split(
        holdout_idx,
        test_size=2.0 / 3.0,
        random_state=seed + 1000,
        stratify=labels[holdout_idx],
    )
    return pd.DataFrame(
        [{"sample_id": sample_ids.iloc[idx], "split": "train"} for idx in train_idx]
        + [{"sample_id": sample_ids.iloc[idx], "split": "val"} for idx in val_idx]
        + [{"sample_id": sample_ids.iloc[idx], "split": "test"} for idx in test_idx]
    )


def _read_manifest(path: Path) -> pd.DataFrame:
    manifest = pd.read_csv(path, dtype={"sample_id": str, "split": str})
    required = {"sample_id", "split"}
    missing = required.difference(manifest.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    manifest = manifest[["sample_id", "split"]].copy()
    manifest["sample_id"] = manifest["sample_id"].astype(str).str.strip()
    manifest["split"] = manifest["split"].astype(str).str.strip().str.lower()
    if manifest["sample_id"].duplicated().any():
        duplicated = manifest.loc[manifest["sample_id"].duplicated(), "sample_id"].iloc[0]
        raise ValueError(f"{path} contains duplicated sample_id={duplicated!r}.")
    invalid = sorted(set(manifest["split"]).difference(VALID_SPLITS))
    if invalid:
        raise ValueError(f"{path} contains invalid split values: {invalid}")
    return manifest


def _validate_full_manifest(
    path: Path,
    expected_ids: set[str],
    *,
    expected_manifest: pd.DataFrame | None = None,
) -> None:
    manifest = _read_manifest(path)
    found_ids = set(manifest["sample_id"])
    if found_ids != expected_ids:
        missing = sorted(expected_ids.difference(found_ids))[:5]
        extra = sorted(found_ids.difference(expected_ids))[:5]
        raise ValueError(f"{path} sample IDs do not match the shared cohort; missing={missing}, extra={extra}.")
    present_splits = set(manifest["split"])
    if present_splits != set(VALID_SPLITS):
        raise ValueError(f"{path} must contain train, val, and test rows; found {sorted(present_splits)}.")
    if expected_manifest is not None:
        actual_map = manifest.set_index("sample_id")["split"].sort_index()
        expected_map = expected_manifest.set_index("sample_id")["split"].sort_index()
        if not actual_map.equals(expected_map):
            mismatched = actual_map.index[actual_map.ne(expected_map)].tolist()[:5]
            raise ValueError(
                f"{path} is not the canonical TxT 70/10/20 membership for its seed; "
                f"mismatched sample examples={mismatched}. Use an empty --split-manifest-dir or remove the stale manifests."
            )


def _seed_from_path(path: Path) -> int:
    match = re.fullmatch(r"seed_(\d+)\.csv", path.name)
    if not match:
        raise ValueError(f"Split manifest must be named seed_<integer>.csv, got {path.name!r}.")
    return int(match.group(1))


def manifest_paths(
    split_manifest_dir: Path,
    repeats: int | None = None,
    *,
    expected_seeds: Sequence[int] | None = None,
) -> list[Path]:
    split_manifest_dir = Path(split_manifest_dir).resolve()
    if not split_manifest_dir.is_dir():
        raise FileNotFoundError(f"Split manifest directory does not exist: {split_manifest_dir}")
    if expected_seeds is not None:
        seeds = [int(seed) for seed in expected_seeds]
        if repeats is not None and len(seeds) != repeats:
            raise ValueError(f"Expected-seed count {len(seeds)} does not equal requested repeats={repeats}.")
        paths = [split_manifest_dir / f"seed_{seed}.csv" for seed in seeds]
        missing = [path for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing requested split manifests: {[str(path) for path in missing]}")
        return paths
    paths = sorted(split_manifest_dir.glob("seed_*.csv"), key=_seed_from_path)
    if repeats is not None:
        if len(paths) < repeats:
            raise ValueError(f"Requested {repeats} repeats but only {len(paths)} split manifests exist in {split_manifest_dir}.")
        paths = paths[:repeats]
    if not paths:
        raise ValueError(f"No seed_*.csv split manifests found in {split_manifest_dir}.")
    return paths


def load_task_splits(
    sample_ids: Iterable[str],
    y: Sequence[int] | pd.Series,
    split_manifest_dir: Path,
    *,
    repeats: int | None = None,
    expected_seeds: Sequence[int] | None = None,
) -> list[dict[str, Any]]:
    """Load shared TxT manifests and map their sample IDs to task row indices."""

    task_ids = pd.Index([str(value).strip() for value in sample_ids])
    if task_ids.has_duplicates:
        duplicated = task_ids[task_ids.duplicated()][0]
        raise ValueError(f"Task data contain duplicated sample_id={duplicated!r}.")
    y_values = np.asarray(y, dtype=np.int64)
    if len(y_values) != len(task_ids):
        raise ValueError("sample_ids and y must have the same length.")
    id_to_idx = {sample_id: idx for idx, sample_id in enumerate(task_ids)}
    task_id_set = set(task_ids)

    splits: list[dict[str, Any]] = []
    for repeat, path in enumerate(
        manifest_paths(split_manifest_dir, repeats, expected_seeds=expected_seeds),
        start=1,
    ):
        manifest = _read_manifest(path)
        manifest_ids = set(manifest["sample_id"])
        missing = task_id_set.difference(manifest_ids)
        if missing:
            raise ValueError(f"{path} is missing {len(missing)} task samples; examples={sorted(missing)[:5]}.")
        task_manifest = manifest.loc[manifest["sample_id"].isin(task_id_set)].copy()
        if len(task_manifest) != len(task_ids):
            raise ValueError(f"{path} did not map one-to-one to all {len(task_ids)} task samples.")

        indices: dict[str, np.ndarray] = {}
        for split_name in VALID_SPLITS:
            values = task_manifest.loc[task_manifest["split"] == split_name, "sample_id"]
            idx = np.asarray([id_to_idx[value] for value in values], dtype=np.int64)
            if len(idx) == 0:
                raise ValueError(f"{path} has no {split_name} samples for this task.")
            if len(np.unique(y_values[idx])) != 2:
                raise ValueError(f"{path} {split_name} partition does not contain both binary classes for this task.")
            indices[split_name] = idx

        if set(indices["train"]) & set(indices["val"]) or set(indices["train"]) & set(indices["test"]) or set(indices["val"]) & set(indices["test"]):
            raise ValueError(f"Split overlap detected in {path}.")
        if set(np.concatenate(list(indices.values())).tolist()) != set(range(len(task_ids))):
            raise ValueError(f"{path} does not assign every task sample exactly once.")

        seed = _seed_from_path(path)
        splits.append(
            {
                "scenario": "shared_test",
                "scenario_index": 1,
                "repeat": repeat,
                "fold": 1,
                "seed": seed,
                "train_inner_idx": indices["train"],
                "val_inner_idx": indices["val"],
                "outer_test_idx": indices["test"],
                "pool_idx": np.concatenate([indices["train"], indices["val"]]),
                "split_manifest_path": str(path),
                "split_source": "txt_shared_multiclass_70_10_20",
            }
        )
    return splits


def split_counts(splits: Sequence[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "scenario": split["scenario"],
                "repeat": split["repeat"],
                "seed": split["seed"],
                "n_train_inner": len(split["train_inner_idx"]),
                "n_val_inner": len(split["val_inner_idx"]),
                "n_outer_test": len(split["outer_test_idx"]),
                "split_manifest_path": split.get("split_manifest_path", ""),
            }
            for split in splits
        ]
    )
