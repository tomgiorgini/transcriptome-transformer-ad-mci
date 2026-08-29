#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_OUTPUT_DIR = ROOT / "task_dataset" / "processed" / "optuna_current"
DEG_ROOT = ROOT / "deg_analysis" / "Results" / "DEG"
GEO_ROOT = ROOT / "pretraining_dataset" / "geo_downloads"

LABELS = {
    "Control": 0,
    "MCI": 1,
    "AD": 2,
}
TASK_LABELS = {
    "ad_mci": {"MCI": 0, "AD": 1},
    "ad_mci_ctl": LABELS,
}


@dataclass(frozen=True)
class GeoDatasetConfig:
    gse_id: str
    platform_id: str
    eset_dir: str
    status_col: str
    status_map: dict[str, str]


GEO_DATASETS = [
    GeoDatasetConfig(
        gse_id="GSE63060",
        platform_id="GPL6947",
        eset_dir="eset_1_GPL6947",
        status_col="status:ch1",
        status_map={"AD": "AD", "MCI": "MCI", "CTL": "Control"},
    ),
    GeoDatasetConfig(
        gse_id="GSE63061",
        platform_id="GPL10558",
        eset_dir="eset_1_GPL10558",
        status_col="status:ch1",
        status_map={"AD": "AD", "MCI": "MCI", "CTL": "Control"},
    ),
    GeoDatasetConfig(
        gse_id="GSE140829",
        platform_id="GPL15988",
        eset_dir="eset_1_GPL15988",
        status_col="diagnosis:ch1",
        status_map={"AD": "AD", "MCI": "MCI", "Control": "Control"},
    ),
]


def clean_text(value: object) -> str:
    return str(value).replace("\ufeff", "").strip().strip("\"'")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build legacy and GSE140829-augmented datasets for current Optuna fine-tuning experiments."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dataset", choices=["legacy", "augmented", "all"], default="all")
    parser.add_argument("--task", choices=["ad_mci", "ad_mci_ctl", "all"], default="all")
    parser.add_argument("--force", action="store_true", help="Overwrite existing outputs.")
    return parser.parse_args()


def read_deg(comparison: str) -> pd.DataFrame:
    path = DEG_ROOT / comparison / "DEG.txt"
    if not path.exists():
        raise FileNotFoundError(f"Missing DEG file: {path}")
    df = pd.read_csv(path, sep="\t")
    required = {"GeneSymbol", "pval", "logFC"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(sorted(missing))}")
    df = df.copy()
    df["GeneSymbol"] = df["GeneSymbol"].map(clean_text)
    df["pval"] = pd.to_numeric(df["pval"], errors="coerce")
    if "adj_pval" in df.columns:
        df["adj_pval"] = pd.to_numeric(df["adj_pval"], errors="coerce")
    else:
        df["adj_pval"] = np.nan
    df["logFC"] = pd.to_numeric(df["logFC"], errors="coerce")
    df["abs_logFC"] = df["logFC"].abs()
    return df.dropna(subset=["GeneSymbol"]).drop_duplicates("GeneSymbol", keep="first")


def sort_deg(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(
        ["adj_pval", "pval", "abs_logFC", "GeneSymbol"],
        ascending=[True, True, False, True],
        na_position="last",
    )


def dedupe_keep_order(genes: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for gene in genes:
        gene = clean_text(gene)
        if gene and gene not in seen:
            out.append(gene)
            seen.add(gene)
    return out


def select_ad_mci_genes(available_genes: set[str]) -> tuple[list[str], dict[str, object]]:
    ad_mci = read_deg("AD_vs_MCI")
    genes = dedupe_keep_order(ad_mci["GeneSymbol"].tolist())
    available = [gene for gene in genes if gene in available_genes]
    missing = [gene for gene in genes if gene not in available_genes]
    return available, {
        "requested_genes": len(genes),
        "selected_genes": len(available),
        "missing_genes": len(missing),
        "missing_gene_examples": missing[:20],
    }


def select_multiclass_genes(available_genes: set[str]) -> tuple[list[str], dict[str, object]]:
    ad_mci = sort_deg(read_deg("AD_vs_MCI"))
    ad_ctl = sort_deg(read_deg("AD_vs_CTL"))
    mci_ctl = sort_deg(read_deg("MCI_vs_CTL"))

    ad_mci_available = [gene for gene in ad_mci["GeneSymbol"].tolist() if gene in available_genes]
    ad_mci_top = dedupe_keep_order(ad_mci_available[:500])

    ad_ctl_rank = {gene: idx for idx, gene in enumerate(ad_ctl["GeneSymbol"].tolist(), start=1)}
    mci_ctl_rank = {gene: idx for idx, gene in enumerate(mci_ctl["GeneSymbol"].tolist(), start=1)}
    ctl_common = sorted(
        (set(ad_ctl_rank) & set(mci_ctl_rank) & available_genes) - set(ad_mci_top),
        key=lambda gene: (ad_ctl_rank[gene] + mci_ctl_rank[gene], max(ad_ctl_rank[gene], mci_ctl_rank[gene]), gene),
    )
    ctl_top = ctl_common[:500]

    selected = dedupe_keep_order([*ad_mci_top, *ctl_top])
    if len(selected) < 1000:
        selected_set = set(selected)
        union_candidates = sorted(
            ((set(ad_ctl_rank) | set(mci_ctl_rank)) & available_genes) - selected_set,
            key=lambda gene: (
                min(ad_ctl_rank.get(gene, 10**9), mci_ctl_rank.get(gene, 10**9)),
                ad_ctl_rank.get(gene, 10**9) + mci_ctl_rank.get(gene, 10**9),
                gene,
            ),
        )
        selected = dedupe_keep_order([*selected, *union_candidates[: 1000 - len(selected)]])

    return selected, {
        "requested_genes": 1000,
        "selected_genes": len(selected),
        "ad_mci_component": len(ad_mci_top),
        "ctl_common_component": len([gene for gene in ctl_top if gene in selected]),
        "available_ad_mci_deg": len(ad_mci_available),
        "available_ctl_common_deg": len(ctl_common),
        "filled_from_ctl_union": max(0, len(selected) - len(ad_mci_top) - len(ctl_top)),
    }


def select_genes(task: str, available_genes: set[str]) -> tuple[list[str], dict[str, object]]:
    if task == "ad_mci":
        return select_ad_mci_genes(available_genes)
    if task == "ad_mci_ctl":
        return select_multiclass_genes(available_genes)
    raise ValueError(f"Unsupported task: {task}")


def read_expression_matrix(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", compression="infer", low_memory=False)
    if "feature_id" not in df.columns:
        df = df.rename(columns={df.columns[0]: "feature_id"})
    df["feature_id"] = df["feature_id"].map(clean_text)
    return df.set_index("feature_id")


def read_row_map(path: Path) -> pd.DataFrame:
    row_map = pd.read_csv(path, sep="\t")
    required = {"feature_id", "gene_symbol", "keep_for_gene_symbol_matrix"}
    missing = required.difference(row_map.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(sorted(missing))}")
    row_map = row_map.copy()
    row_map["feature_id"] = row_map["feature_id"].map(clean_text)
    row_map["gene_symbol"] = row_map["gene_symbol"].map(clean_text)
    keep_raw = row_map["keep_for_gene_symbol_matrix"]
    if keep_raw.dtype == bool:
        row_map["_keep"] = keep_raw
    else:
        row_map["_keep"] = keep_raw.astype(str).str.lower().isin({"true", "1", "t", "yes"})
    return row_map


def collapse_to_gene_symbols(expr: pd.DataFrame, row_map: pd.DataFrame) -> pd.DataFrame:
    mapping = row_map.loc[row_map["_keep"] & row_map["gene_symbol"].ne(""), ["feature_id", "gene_symbol"]]
    expr = expr.loc[expr.index.intersection(mapping["feature_id"])]
    symbol_by_feature = mapping.drop_duplicates("feature_id").set_index("feature_id")["gene_symbol"]
    expr = expr.apply(pd.to_numeric, errors="coerce")
    expr["_gene_symbol"] = symbol_by_feature.loc[expr.index].to_numpy()
    collapsed = expr.groupby("_gene_symbol", sort=False).mean(numeric_only=True)
    collapsed.index = collapsed.index.map(clean_text)
    return collapsed.loc[~collapsed.index.duplicated(keep="first")]


def minmax_by_gene(expr_genes_x_samples: pd.DataFrame) -> pd.DataFrame:
    values = expr_genes_x_samples.to_numpy(dtype=np.float64)
    medians = np.nanmedian(values, axis=1)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    rows, cols = np.where(~np.isfinite(values))
    if len(rows) > 0:
        values[rows, cols] = medians[rows]
    min_v = values.min(axis=1, keepdims=True)
    max_v = values.max(axis=1, keepdims=True)
    scale = np.where(max_v > min_v, max_v - min_v, 1.0)
    scaled = (values - min_v) / scale
    return pd.DataFrame(scaled.astype(np.float32), index=expr_genes_x_samples.index, columns=expr_genes_x_samples.columns)


def load_geo_dataset(config: GeoDatasetConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = GEO_ROOT / config.gse_id / config.eset_dir
    expr = read_expression_matrix(base / f"{config.gse_id}_expr_matrix.tsv.gz")
    row_map = read_row_map(base / "matrix_row_to_gene_symbol.tsv")
    collapsed = collapse_to_gene_symbols(expr, row_map)

    metadata = pd.read_csv(base / f"{config.gse_id}_sample_metadata.tsv.gz", sep="\t")
    if config.status_col not in metadata.columns:
        raise ValueError(f"{base} metadata does not contain {config.status_col}")
    id_col = "sample_id" if "sample_id" in metadata.columns else "geo_accession"
    metadata = metadata.copy()
    metadata["original_sample_id"] = metadata[id_col].map(clean_text)
    metadata["raw_status"] = metadata[config.status_col].map(clean_text)
    metadata["label_name"] = metadata["raw_status"].map(config.status_map)
    metadata = metadata.loc[metadata["label_name"].isin(["AD", "MCI", "Control"])].copy()
    metadata["sample_id"] = (
        config.gse_id + "__" + config.platform_id + "__" + metadata["original_sample_id"].astype(str)
    )
    metadata["gse_id"] = config.gse_id
    metadata["platform_id"] = config.platform_id

    present = [sample for sample in metadata["original_sample_id"].tolist() if sample in collapsed.columns]
    metadata = metadata.loc[metadata["original_sample_id"].isin(present)].copy()
    metadata = metadata.drop_duplicates("original_sample_id", keep="first")
    metadata = metadata.set_index("original_sample_id").loc[present].reset_index()
    matrix = collapsed.loc[:, present].copy()
    matrix.columns = metadata["sample_id"].tolist()

    return matrix, metadata[["sample_id", "original_sample_id", "gse_id", "platform_id", "label_name", "raw_status"]]


def read_geo_combined_dataset(configs: list[GeoDatasetConfig]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    loaded = [load_geo_dataset(config) for config in configs]
    common_genes = set(loaded[0][0].index)
    for matrix, _ in loaded[1:]:
        common_genes &= set(matrix.index)
    common_gene_list = sorted(common_genes)
    if not common_gene_list:
        names = ", ".join(config.gse_id for config in configs)
        raise ValueError(f"No common genes across {names}.")

    scaled_matrices = []
    metadata_frames = []
    counts: dict[str, dict[str, int]] = {}
    for config, (matrix, metadata) in zip(GEO_DATASETS, loaded):
        matrix = matrix.loc[common_gene_list]
        scaled = minmax_by_gene(matrix)
        scaled_matrices.append(scaled)
        metadata_frames.append(metadata)
        counts[config.gse_id] = metadata["label_name"].value_counts().sort_index().astype(int).to_dict()

    combined = pd.concat(scaled_matrices, axis=1)
    metadata = pd.concat(metadata_frames, ignore_index=True)
    x_df = combined.transpose().reset_index().rename(columns={"index": "sample_id"})
    y_df = metadata[["sample_id", "label_name"]].copy()
    y_df["label"] = y_df["label_name"].map(LABELS).astype(int)
    y_df = y_df[["sample_id", "label", "label_name"]]
    return x_df, y_df, {
        "source": " + ".join(config.gse_id for config in configs),
        "normalization": "per dataset, per gene min-max before concatenation",
        "common_genes_before_task_selection": len(common_gene_list),
        "sample_counts_by_dataset": counts,
    }


def read_legacy_dataset() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    return read_geo_combined_dataset(GEO_DATASETS[:2])


def read_augmented_dataset() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    return read_geo_combined_dataset(GEO_DATASETS)


def build_task_dataset(dataset_name: str, task: str, output_dir: Path, force: bool) -> dict[str, object]:
    x_out = output_dir / f"X_{dataset_name}_{task}.csv"
    y_out = output_dir / f"y_{dataset_name}_{task}.csv"
    genes_out = output_dir / f"selected_genes_{dataset_name}_{task}.txt"
    summary_out = output_dir / f"dataset_summary_{dataset_name}_{task}.csv"
    summary_json_out = output_dir / f"dataset_summary_{dataset_name}_{task}.json"
    if not force and x_out.exists() and y_out.exists() and genes_out.exists() and summary_out.exists():
        return {
            "dataset": dataset_name,
            "task": task,
            "skipped_existing": True,
            "x_file": str(x_out),
            "y_file": str(y_out),
            "selected_genes_file": str(genes_out),
            "summary_file": str(summary_out),
            "summary_json_file": str(summary_json_out),
        }

    if dataset_name == "legacy":
        x_df, y_df, dataset_summary = read_legacy_dataset()
    elif dataset_name == "augmented":
        x_df, y_df, dataset_summary = read_augmented_dataset()
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    x_df["sample_id"] = x_df["sample_id"].map(clean_text)
    y_df["sample_id"] = y_df["sample_id"].map(clean_text)
    if x_df["sample_id"].duplicated().any():
        raise ValueError(f"{dataset_name} X contains duplicated sample IDs.")
    if y_df["sample_id"].duplicated().any():
        raise ValueError(f"{dataset_name} y contains duplicated sample IDs.")

    allowed = set(TASK_LABELS[task])
    y_task = y_df.loc[y_df["label_name"].isin(allowed), ["sample_id", "label_name"]].copy()
    y_task["label"] = y_task["label_name"].map(TASK_LABELS[task]).astype(int)
    y_task = y_task[["sample_id", "label", "label_name"]]
    sample_ids = set(y_task["sample_id"])
    x_task_base = x_df.loc[x_df["sample_id"].isin(sample_ids)].copy()
    x_task_base = x_task_base.set_index("sample_id").loc[y_task["sample_id"]].reset_index()

    available_genes = set(x_task_base.columns) - {"sample_id"}
    selected_genes, gene_summary = select_genes(task, available_genes)
    if dataset_name == "legacy" and task == "ad_mci" and len(selected_genes) != 788:
        raise ValueError(
            f"{dataset_name}/{task} expected 788 AD-vs-MCI genes but found {len(selected_genes)} available genes."
        )
    if dataset_name == "legacy" and task == "ad_mci_ctl" and len(selected_genes) != 1000:
        raise ValueError(
            f"{dataset_name}/{task} expected 1000 multiclass genes but found {len(selected_genes)} available genes."
        )

    x_task = x_task_base[["sample_id", *selected_genes]].copy()
    output_dir.mkdir(parents=True, exist_ok=True)
    x_task.to_csv(x_out, index=False)
    y_task.to_csv(y_out, index=False)
    genes_out.write_text("\n".join(selected_genes) + "\n", encoding="utf-8")

    summary = {
        "dataset": dataset_name,
        "task": task,
        "x_file": str(x_out),
        "y_file": str(y_out),
        "selected_genes_file": str(genes_out),
        "samples": int(len(y_task)),
        "genes": int(len(selected_genes)),
        "class_counts": y_task["label_name"].value_counts().sort_index().astype(int).to_dict(),
        "dataset_summary": dataset_summary,
        "gene_selection": gene_summary,
    }
    summary_rows = [
        {"metric": "dataset", "value": dataset_name},
        {"metric": "task", "value": task},
        {"metric": "samples", "value": int(len(y_task))},
        {"metric": "genes", "value": int(len(selected_genes))},
        {"metric": "x_file", "value": str(x_out)},
        {"metric": "y_file", "value": str(y_out)},
        {"metric": "selected_genes_file", "value": str(genes_out)},
        {"metric": "class_counts", "value": json.dumps(summary["class_counts"], sort_keys=True)},
        {"metric": "dataset_summary", "value": json.dumps(dataset_summary, sort_keys=True)},
        {"metric": "gene_selection", "value": json.dumps(gene_summary, sort_keys=True)},
    ]
    pd.DataFrame(summary_rows).to_csv(summary_out, index=False)
    summary_json_out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        **summary,
        "summary_file": str(summary_out),
        "summary_json_file": str(summary_json_out),
        "skipped_existing": False,
    }


def main() -> None:
    args = parse_args()
    datasets = ["legacy", "augmented"] if args.dataset == "all" else [args.dataset]
    tasks = ["ad_mci", "ad_mci_ctl"] if args.task == "all" else [args.task]
    outputs = []
    for dataset_name in datasets:
        for task in tasks:
            result = build_task_dataset(dataset_name, task, args.output_dir, args.force)
            outputs.append(result)
            print(
                f"{dataset_name}/{task}: samples={result.get('samples', 'existing')} "
                f"genes={result.get('genes', 'existing')} -> {result['x_file']}",
                flush=True,
            )
    print(json.dumps(outputs, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
