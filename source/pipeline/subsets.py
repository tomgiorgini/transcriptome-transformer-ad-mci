from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd


AD_MCI_BINARY_LABEL_MAP = {"MCI": 0, "AD": 1}
VALID_SPLIT_NAMES = {"train", "val", "test"}


@dataclass
class SubsetBuildResult:
    x_df: pd.DataFrame
    y_df: pd.DataFrame
    split_df: pd.DataFrame
    selected_gene_names: list[str]
    class_counts: dict[str, int]


def _require_file(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {description}: {path}")


def _normalize_sample_id(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip()


def _ensure_unique_sample_ids(df: pd.DataFrame, *, sample_col: str, file_name: str) -> None:
    duplicated = df.loc[df[sample_col].duplicated(keep=False), sample_col].unique().tolist()
    if duplicated:
        preview = ", ".join(duplicated[:10])
        raise ValueError(f"{file_name} contains duplicated {sample_col} values. Example: {preview}")


def _resolve_label_names(y_df: pd.DataFrame) -> pd.Series:
    if "label_name" in y_df.columns:
        label_names = y_df["label_name"].astype(str).str.strip()
        if (label_names == "").any():
            raise ValueError("y.csv contains empty label_name values.")
        return label_names

    if "label" not in y_df.columns:
        raise ValueError("y.csv must contain either label_name or label.")

    # Canonical project encoding from build_alzheimer_dataset.py.
    canonical_names = y_df["label"].map({0: "Control", 1: "MCI", 2: "AD"})
    if canonical_names.isna().any():
        unknown = sorted(y_df.loc[canonical_names.isna(), "label"].astype(str).unique().tolist())
        raise ValueError(f"Unable to infer label_name from label values: {', '.join(unknown)}")
    return canonical_names.astype(str)


def _filter_split_manifest(split_df: pd.DataFrame, kept_sample_ids: set[str]) -> pd.DataFrame:
    if "sample_id" not in split_df.columns or "split" not in split_df.columns:
        raise ValueError("split file must contain sample_id and split columns.")

    split_df = split_df.copy()
    split_df["sample_id"] = _normalize_sample_id(split_df["sample_id"])
    split_df["split"] = split_df["split"].astype(str).str.strip().str.lower()

    _ensure_unique_sample_ids(split_df, sample_col="sample_id", file_name="split file")

    invalid_split_names = sorted(set(split_df["split"]) - VALID_SPLIT_NAMES)
    if invalid_split_names:
        raise ValueError(f"split file contains invalid split names: {', '.join(invalid_split_names)}")

    filtered = split_df[split_df["sample_id"].isin(kept_sample_ids)].copy()
    missing_sample_ids = sorted(kept_sample_ids - set(filtered["sample_id"]))
    if missing_sample_ids:
        preview = ", ".join(missing_sample_ids[:10])
        raise ValueError(f"split file is missing {len(missing_sample_ids)} AD/MCI sample ids. Example: {preview}")

    split_counts = filtered["split"].value_counts().to_dict()
    empty_splits = [name for name in ("train", "val", "test") if split_counts.get(name, 0) == 0]
    if empty_splits:
        raise ValueError(f"Filtered split manifest has empty splits: {', '.join(empty_splits)}")

    return filtered


def _prepare_ordered_genes(ordered_genes: Sequence[str], available_gene_columns: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    requested_genes: list[str] = []
    for gene in ordered_genes:
        gene_name = str(gene).strip()
        if not gene_name or gene_name == "nan" or gene_name in seen:
            continue
        requested_genes.append(gene_name)
        seen.add(gene_name)

    available_genes = set(available_gene_columns)
    selected = [gene for gene in requested_genes if gene in available_genes]
    if not selected:
        raise ValueError("None of the requested genes were found in X.csv.")
    return selected


def build_ad_mci_subset(
    *,
    x_file: Path,
    y_file: Path,
    split_file: Path,
    ordered_genes: Sequence[str] | None = None,
) -> SubsetBuildResult:
    _require_file(x_file, "X file")
    _require_file(y_file, "y file")
    _require_file(split_file, "split file")

    x_df = pd.read_csv(x_file)
    y_df = pd.read_csv(y_file)
    split_df = pd.read_csv(split_file)

    if "sample_id" not in x_df.columns or "sample_id" not in y_df.columns:
        raise ValueError("Both X and y files must contain a sample_id column.")
    if "label" not in y_df.columns and "label_name" not in y_df.columns:
        raise ValueError("y.csv must contain at least one of label or label_name columns.")

    x_df = x_df.copy()
    y_df = y_df.copy()
    x_df["sample_id"] = _normalize_sample_id(x_df["sample_id"])
    y_df["sample_id"] = _normalize_sample_id(y_df["sample_id"])

    _ensure_unique_sample_ids(x_df, sample_col="sample_id", file_name="X file")
    _ensure_unique_sample_ids(y_df, sample_col="sample_id", file_name="y file")

    missing_in_y = sorted(set(x_df["sample_id"]) - set(y_df["sample_id"]))
    missing_in_x = sorted(set(y_df["sample_id"]) - set(x_df["sample_id"]))
    if missing_in_y or missing_in_x:
        messages: list[str] = []
        if missing_in_y:
            messages.append(f"present in X only: {', '.join(missing_in_y[:10])}")
        if missing_in_x:
            messages.append(f"present in y only: {', '.join(missing_in_x[:10])}")
        raise ValueError("X and y files do not share the same sample_id set: " + " | ".join(messages))

    label_names = _resolve_label_names(y_df)
    y_df["label_name"] = label_names

    y_binary = y_df[y_df["label_name"].isin(AD_MCI_BINARY_LABEL_MAP)].copy()
    if y_binary.empty:
        raise ValueError("No AD/MCI rows found in y.csv.")

    y_binary["label"] = y_binary["label_name"].map(AD_MCI_BINARY_LABEL_MAP).astype(int)
    y_binary = y_binary[["sample_id", "label", "label_name"]]

    kept_sample_ids = set(y_binary["sample_id"])
    x_binary = x_df[x_df["sample_id"].isin(kept_sample_ids)].copy()

    y_binary = y_binary.sort_values("sample_id").reset_index(drop=True)
    x_binary = x_binary.sort_values("sample_id").reset_index(drop=True)

    if x_binary["sample_id"].tolist() != y_binary["sample_id"].tolist():
        raise ValueError("Filtered X and y are not aligned after AD/MCI filtering.")

    gene_columns = [column for column in x_binary.columns if column != "sample_id"]
    if ordered_genes is not None:
        selected_genes = _prepare_ordered_genes(ordered_genes, gene_columns)
    else:
        selected_genes = gene_columns

    x_binary = x_binary[["sample_id", *selected_genes]].copy()
    split_binary = _filter_split_manifest(split_df, kept_sample_ids)

    class_counts = {
        class_name: int(count)
        for class_name, count in y_binary["label_name"].value_counts().sort_index().to_dict().items()
    }
    return SubsetBuildResult(
        x_df=x_binary,
        y_df=y_binary,
        split_df=split_binary,
        selected_gene_names=selected_genes,
        class_counts=class_counts,
    )