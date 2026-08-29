from __future__ import annotations

import copy
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn


SOURCE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SOURCE_ROOT.parent
DEFAULT_RAW_DATA_DIR = REPO_ROOT / "task_dataset"
DEFAULT_GROUPS_DIR = DEFAULT_RAW_DATA_DIR
DEFAULT_PROCESSED_DATA_DIR = REPO_ROOT / "task_dataset" / "processed" / "alzheimer_multiclass"
DEFAULT_X_FILE = DEFAULT_PROCESSED_DATA_DIR / "X.csv"
DEFAULT_Y_FILE = DEFAULT_PROCESSED_DATA_DIR / "y.csv"
DEFAULT_SPLIT_DIR = DEFAULT_PROCESSED_DATA_DIR / "splits"
DEFAULT_OFFICIAL_SPLIT_FILE = DEFAULT_SPLIT_DIR / "official_seed42.csv"


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_device(requested_device: str) -> torch.device:
    if requested_device == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested_device == "mps":
        mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        return torch.device("mps" if mps_available else "cpu")
    return torch.device("cpu")


def compute_balanced_class_weights(y: np.ndarray) -> np.ndarray:
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) == 0:
        raise ValueError("Cannot compute class weights from an empty target array.")
    total = counts.sum()
    weights = total / (len(classes) * counts.astype(np.float32))
    ordered = np.zeros(int(classes.max()) + 1, dtype=np.float32)
    ordered[classes.astype(int)] = weights
    return ordered


def get_clones(module: nn.Module, count: int) -> nn.ModuleList:
    return nn.ModuleList([copy.deepcopy(module) for _ in range(count)])


def namespace_to_dict(namespace: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in vars(namespace).items():
        if isinstance(value, Path):
            result[key] = str(value)
        else:
            result[key] = value
    return result


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
