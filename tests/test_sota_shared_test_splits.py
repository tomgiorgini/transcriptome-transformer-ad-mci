from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
SOTA_SOURCE = ROOT / "SOTA" / "source"
if str(SOTA_SOURCE) not in sys.path:
    sys.path.insert(0, str(SOTA_SOURCE))

from shared_test_splits import load_task_splits, manifest_paths, write_txt_shared_splits


def _synthetic_labels() -> pd.DataFrame:
    rows = []
    for label, name in enumerate(("ctl", "mci", "ad")):
        rows.extend({"sample_id": f"{name}_{idx:03d}", "label": label} for idx in range(100))
    return pd.DataFrame(rows)


def test_shared_manifests_are_exact_70_10_20_and_deterministic(tmp_path: Path) -> None:
    labels = _synthetic_labels()
    y_file = tmp_path / "y.csv"
    labels.to_csv(y_file, index=False)

    first = write_txt_shared_splits(y_file, tmp_path / "first", seeds=[101, 102])
    second = write_txt_shared_splits(y_file, tmp_path / "second", seeds=[101, 102])

    for path_a, path_b in zip(first, second, strict=True):
        manifest_a = pd.read_csv(path_a)
        manifest_b = pd.read_csv(path_b)
        pd.testing.assert_frame_equal(manifest_a, manifest_b)
        assert manifest_a["split"].value_counts().to_dict() == {"train": 210, "test": 60, "val": 30}


def test_pairwise_tasks_filter_the_same_multiclass_membership(tmp_path: Path) -> None:
    labels = _synthetic_labels()
    y_file = tmp_path / "y.csv"
    labels.to_csv(y_file, index=False)
    paths = write_txt_shared_splits(y_file, tmp_path / "splits", seeds=[101])

    pairwise = labels.loc[labels["label"].isin([1, 2])].copy()
    pairwise["label"] = (pairwise["label"] == 2).astype(int)
    splits = load_task_splits(pairwise["sample_id"], pairwise["label"], tmp_path / "splits", repeats=1)
    split = splits[0]

    manifest = pd.read_csv(paths[0]).set_index("sample_id")["split"]
    task_ids = pairwise["sample_id"].reset_index(drop=True)
    for manifest_name, index_name in (
        ("train", "train_inner_idx"),
        ("val", "val_inner_idx"),
        ("test", "outer_test_idx"),
    ):
        expected_ids = set(task_ids[manifest.loc[task_ids].to_numpy() == manifest_name])
        actual_ids = set(task_ids.iloc[split[index_name]])
        assert actual_ids == expected_ids

    assigned = np.concatenate(
        [split["train_inner_idx"], split["val_inner_idx"], split["outer_test_idx"]]
    )
    assert set(assigned.tolist()) == set(range(len(pairwise)))


def test_existing_noncanonical_manifest_is_rejected(tmp_path: Path) -> None:
    labels = _synthetic_labels()
    y_file = tmp_path / "y.csv"
    labels.to_csv(y_file, index=False)
    split_dir = tmp_path / "splits"
    [path] = write_txt_shared_splits(y_file, split_dir, seeds=[101])
    manifest = pd.read_csv(path)
    train_row = manifest.index[manifest["split"] == "train"][0]
    test_row = manifest.index[manifest["split"] == "test"][0]
    manifest.loc[train_row, "split"] = "test"
    manifest.loc[test_row, "split"] = "train"
    manifest.to_csv(path, index=False)

    with pytest.raises(ValueError, match="not the canonical TxT"):
        write_txt_shared_splits(y_file, split_dir, seeds=[101], overwrite=False)


def test_expected_seed_list_cannot_be_displaced_by_extra_manifest(tmp_path: Path) -> None:
    labels = _synthetic_labels()
    y_file = tmp_path / "y.csv"
    labels.to_csv(y_file, index=False)
    split_dir = tmp_path / "splits"
    write_txt_shared_splits(y_file, split_dir, seeds=[1, 101, 102])

    selected = manifest_paths(split_dir, repeats=2, expected_seeds=[101, 102])
    assert [path.name for path in selected] == ["seed_101.csv", "seed_102.csv"]
