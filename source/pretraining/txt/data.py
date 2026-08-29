from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class PretrainingMatrix:
    sample_ids: np.ndarray
    gene_names: list[str]
    values: np.ndarray


@dataclass(frozen=True)
class MaskedGeneBatch:
    input_values: torch.Tensor
    target_values: torch.Tensor
    gene_indices: torch.Tensor
    masked_positions: torch.Tensor
    attention_mask: torch.Tensor


class ExpressionMatrixDataset(Dataset):
    def __init__(self, values: np.ndarray):
        self.values = torch.tensor(values, dtype=torch.float32)

    def __len__(self) -> int:
        return self.values.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.values[idx]


def load_pretraining_matrix(path: Path) -> PretrainingMatrix:
    df = pd.read_csv(path)
    if "sample_id" not in df.columns:
        raise ValueError("Pretraining matrix must contain a sample_id column.")

    sample_ids = df["sample_id"].astype(str).str.strip()
    duplicates = sample_ids[sample_ids.duplicated()].unique().tolist()
    if duplicates:
        preview = ", ".join(duplicates[:10])
        raise ValueError(f"Pretraining matrix contains duplicated sample_id values. Example: {preview}")

    gene_names = [column for column in df.columns if column != "sample_id"]
    if not gene_names:
        raise ValueError("Pretraining matrix must contain at least one gene column.")
    if len(set(gene_names)) != len(gene_names):
        raise ValueError("Pretraining matrix contains duplicated gene columns.")

    values = df[gene_names].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    if values.shape[0] == 0:
        raise ValueError("Pretraining matrix contains no samples.")
    if np.isnan(values).all(axis=0).any():
        bad_idx = np.where(np.isnan(values).all(axis=0))[0][:10]
        bad_genes = ", ".join(gene_names[int(idx)] for idx in bad_idx)
        raise ValueError(f"Pretraining matrix has genes with all missing values: {bad_genes}")

    values = np.nan_to_num(values, nan=0.0).astype(np.float32)
    return PretrainingMatrix(sample_ids=sample_ids.to_numpy(), gene_names=gene_names, values=values)


def split_train_val_indices(n_samples: int, val_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if n_samples < 2:
        raise ValueError("At least two samples are required for train/val split.")
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1.")

    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_samples)
    n_val = int(round(n_samples * val_ratio))
    n_val = min(max(n_val, 1), n_samples - 1)
    return indices[n_val:], indices[:n_val]


def build_masked_gene_batch(
    rows: Sequence[torch.Tensor],
    gene_subset_size: int,
    mask_ratio: float,
    generator: torch.Generator | None = None,
) -> MaskedGeneBatch:
    values = torch.stack([row.float() for row in rows], dim=0)
    batch_size, n_genes = values.shape
    if not 0.0 < mask_ratio < 1.0:
        raise ValueError("mask_ratio must be between 0 and 1.")

    subset_size = n_genes if gene_subset_size <= 0 else min(gene_subset_size, n_genes)
    mask_count = min(max(int(round(subset_size * mask_ratio)), 1), subset_size)

    gene_indices = torch.empty((batch_size, subset_size), dtype=torch.long)
    masked_positions = torch.zeros((batch_size, subset_size), dtype=torch.bool)
    for row_idx in range(batch_size):
        perm = torch.randperm(n_genes, generator=generator)
        selected = perm[:subset_size]
        gene_indices[row_idx] = selected
        mask_perm = torch.randperm(subset_size, generator=generator)
        masked_positions[row_idx, mask_perm[:mask_count]] = True

    target_values = values.gather(1, gene_indices)
    input_values = target_values.clone()
    input_values[masked_positions] = 0.0
    attention_mask = (~masked_positions).view(batch_size, 1, 1, subset_size)

    return MaskedGeneBatch(
        input_values=input_values,
        target_values=target_values,
        gene_indices=gene_indices,
        masked_positions=masked_positions,
        attention_mask=attention_mask,
    )


def make_masked_collate_fn(
    gene_subset_size: int,
    mask_ratio: float,
    generator: torch.Generator | None = None,
) -> Callable[[list[torch.Tensor]], MaskedGeneBatch]:
    def collate(rows: list[torch.Tensor]) -> MaskedGeneBatch:
        return build_masked_gene_batch(rows, gene_subset_size, mask_ratio, generator)

    return collate


def global_zscore_report(values: np.ndarray) -> dict[str, float]:
    means = values.mean(axis=0)
    stds = values.std(axis=0, ddof=1)
    return {
        "max_abs_gene_mean": float(np.max(np.abs(means))),
        "mean_gene_std": float(np.mean(stds)),
        "min_gene_std": float(np.min(stds)),
        "max_gene_std": float(np.max(stds)),
    }
