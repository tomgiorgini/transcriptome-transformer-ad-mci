#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MATRIX = ROOT / "task_dataset" / "matrix.txt"
DEFAULT_Y = ROOT / "task_dataset" / "processed" / "ad_mci_binary" / "y_ad_mci.csv"
DEFAULT_RESULT_ROOT = ROOT / "results" / "deg" / "raw_matrix_train_only_batchholdout"
DEFAULT_GSE63060_METADATA = ROOT / "pretraining_dataset" / "geo_downloads" / "GSE63060" / "eset_1_GPL6947" / "GSE63060_sample_metadata.tsv.gz"
DEFAULT_GSE63061_METADATA = ROOT / "pretraining_dataset" / "geo_downloads" / "GSE63061" / "eset_1_GPL10558" / "GSE63061_sample_metadata.tsv.gz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute train-only AD vs MCI DEG on task_dataset/matrix.txt, matching deg_analysis DEG logic."
    )
    parser.add_argument("--matrix-file", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--y-file", type=Path, default=DEFAULT_Y)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(101, 111)))
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=["shared_test", "test_gse63060", "test_gse63061"],
        choices=["shared_test", "test_gse63060", "test_gse63061", "pooled_stratified"],
    )
    parser.add_argument("--pooled-test-size", type=float, default=0.15)
    parser.add_argument("--full-dataset", action="store_true", help="Compute one AD vs MCI DEG table using all available AD/MCI samples.")
    parser.add_argument("--shared-test-size", type=float, default=0.2)
    parser.add_argument("--inner-val-ratio", type=float, default=0.15)
    parser.add_argument("--adj-pval-threshold", type=float, default=0.05)
    parser.add_argument("--min-abs-logfc", type=float, default=0.0)
    parser.add_argument("--iqr-quantile", type=float, default=0.10)
    parser.add_argument("--log-transform", choices=["log2_x_plus_1", "none"], default="log2_x_plus_1")
    parser.add_argument("--gse63060-metadata", type=Path, default=DEFAULT_GSE63060_METADATA)
    parser.add_argument("--gse63061-metadata", type=Path, default=DEFAULT_GSE63061_METADATA)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else (ROOT / path).resolve()


def read_geo_metadata_sample_ids(path: Path) -> set[str]:
    metadata = pd.read_csv(path, sep="\t", compression="gzip", dtype=str)
    for candidate in ("sample_id", "geo_accession"):
        if candidate in metadata.columns:
            return set(metadata[candidate].dropna().astype(str).str.strip())
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
        raise ValueError(f"Could not map {len(unknown)} samples to GSE63060/GSE63061. Examples: {preview}")
    return batch


def make_split(
    y: pd.Series,
    batch: pd.Series,
    scenario: str,
    seed: int,
    shared_test_size: float,
    inner_val_ratio: float,
    pooled_test_size: float,
) -> dict[str, Any]:
    y_values = y.to_numpy(dtype=np.int64)
    batch_values = batch.to_numpy()
    all_idx = np.arange(len(y_values))
    if scenario == "pooled_stratified":
        pool_idx, test_idx = train_test_split(
            all_idx,
            test_size=pooled_test_size,
            random_state=seed,
            stratify=y_values,
        )
    elif scenario == "test_gse63060":
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
                random_state=seed,
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
        random_state=seed * 100 + 1,
        stratify=y_values[pool_idx],
    )
    return {
        "train_idx": np.asarray(train_idx, dtype=np.int64),
        "val_idx": np.asarray(val_idx, dtype=np.int64),
        "test_idx": np.asarray(test_idx, dtype=np.int64),
    }


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=np.float64)
    n = p.size
    order = np.argsort(p)
    ranked = p[order]
    adjusted = ranked * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    out = np.empty_like(adjusted)
    out[order] = adjusted
    return out


def compute_deg(
    matrix: pd.DataFrame,
    y: pd.Series,
    train_sample_ids: list[str],
    output_dir: Path,
    adj_pval_threshold: float,
    min_abs_logfc: float,
    iqr_quantile: float,
    log_transform: str,
) -> dict[str, Any]:
    train_y = y.loc[train_sample_ids]
    missing_samples = sorted(set(train_sample_ids) - set(matrix.columns))
    if missing_samples:
        raise ValueError(f"Training samples missing from matrix. Examples: {missing_samples[:10]}")

    train_x = matrix.loc[:, train_sample_ids].transpose().apply(pd.to_numeric, errors="coerce")
    ad_ids = train_y.index[train_y.eq(1)].tolist()
    mci_ids = train_y.index[train_y.eq(0)].tolist()
    if len(ad_ids) < 2 or len(mci_ids) < 2:
        raise ValueError(f"Not enough training samples for DEG: AD={len(ad_ids)}, MCI={len(mci_ids)}")

    raw_values = train_x.to_numpy(dtype=np.float64)
    overall_mean = np.nanmean(raw_values, axis=0)
    keep_nonzero_mean = np.isfinite(overall_mean) & (overall_mean != 0)
    filtered_train = train_x.loc[:, keep_nonzero_mean].copy()
    after_zero_mean_genes = int(filtered_train.shape[1])

    values_for_test = filtered_train.to_numpy(dtype=np.float64)
    if log_transform == "log2_x_plus_1":
        values_for_test = np.log2(values_for_test + 1.0)
        log_transform_label = "log2(x + 1)"
    elif log_transform == "none":
        log_transform_label = "none"
    else:
        raise ValueError(f"Unsupported log_transform: {log_transform}")
    variation = stats.iqr(values_for_test, axis=0, nan_policy="omit")
    iqr_threshold = float(np.nanquantile(variation, iqr_quantile))
    keep_iqr = np.isfinite(variation) & (variation > iqr_threshold)
    filtered_train = filtered_train.loc[:, keep_iqr].copy()
    genes = np.asarray(filtered_train.columns.astype(str))
    values_for_test = filtered_train.to_numpy(dtype=np.float64)
    if log_transform == "log2_x_plus_1":
        values_for_test = np.log2(values_for_test + 1.0)

    train_index = filtered_train.index.to_numpy()
    ad_mask = np.isin(train_index, ad_ids)
    mci_mask = np.isin(train_index, mci_ids)
    ad_x = values_for_test[ad_mask, :]
    mci_x = values_for_test[mci_mask, :]
    logfc = np.nanmean(ad_x, axis=0) - np.nanmean(mci_x, axis=0)
    t_stat, pval = stats.ttest_ind(ad_x, mci_x, axis=0, equal_var=False, nan_policy="omit")
    pval = np.where(np.isfinite(pval), pval, 1.0)
    t_stat = np.where(np.isfinite(t_stat), t_stat, np.nan)
    adj_pval = benjamini_hochberg(pval)

    all_genes = pd.DataFrame(
        {
            "GeneSymbol": genes,
            "pval": pval,
            "adj_pval": adj_pval,
            "logFC": logfc,
            "abs_logFC": np.abs(logfc),
            "t_statistic": t_stat,
            "mean_AD": np.nanmean(ad_x, axis=0),
            "mean_MCI": np.nanmean(mci_x, axis=0),
        }
    )
    all_genes["direction"] = np.where(all_genes["logFC"] >= 0, "UP", "DOWN")
    ranked_all = all_genes.sort_values(
        ["adj_pval", "pval", "abs_logFC", "GeneSymbol"],
        ascending=[True, True, False, True],
        na_position="last",
    )
    selected = all_genes.loc[
        all_genes["adj_pval"].lt(adj_pval_threshold) & all_genes["abs_logFC"].ge(min_abs_logfc)
    ].copy()
    selected = selected.sort_values("logFC", ascending=False)

    output_dir.mkdir(parents=True, exist_ok=True)
    ranked_all.to_csv(output_dir / "DEG_all_genes.tsv", sep="\t", index=False)
    selected.to_csv(output_dir / "DEG_selected.tsv", sep="\t", index=False)
    selected_genes = selected["GeneSymbol"].astype(str).tolist()
    (output_dir / "selected_genes.txt").write_text("\n".join(selected_genes) + ("\n" if selected_genes else ""), encoding="utf-8")

    summary: dict[str, Any] = {
        "train_samples": int(len(train_sample_ids)),
        "train_ad": int(len(ad_ids)),
        "train_mci": int(len(mci_ids)),
        "input_genes": int(train_x.shape[1]),
        "after_zero_mean_filter_genes": after_zero_mean_genes,
        "tested_genes": int(len(ranked_all)),
        "selected_genes": int(len(selected_genes)),
        "selected_up": int((selected["direction"] == "UP").sum()) if not selected.empty else 0,
        "selected_down": int((selected["direction"] == "DOWN").sum()) if not selected.empty else 0,
        "adj_pval_threshold": float(adj_pval_threshold),
        "min_abs_logfc": float(min_abs_logfc),
        "log_transform": log_transform_label,
        "iqr_quantile_filter": float(iqr_quantile),
        "iqr_threshold": iqr_threshold,
        "test": "Welch t-test matching R stats::t.test default",
        "p_adjust": "fdr",
        "selection_order": "logFC decreasing, matching deg_analysis DEG.txt",
    }
    (output_dir / "deg_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    args.matrix_file = resolve(args.matrix_file)
    args.y_file = resolve(args.y_file)
    args.result_root = resolve(args.result_root)
    args.gse63060_metadata = resolve(args.gse63060_metadata)
    args.gse63061_metadata = resolve(args.gse63061_metadata)

    if args.overwrite and args.result_root.exists():
        import shutil

        shutil.rmtree(args.result_root)
    args.result_root.mkdir(parents=True, exist_ok=True)

    matrix = pd.read_csv(args.matrix_file, sep="\t", index_col=0)
    y_df = pd.read_csv(args.y_file)
    y_df["sample_id"] = y_df["sample_id"].astype(str)
    y = y_df.set_index("sample_id")["label"].astype(int)
    common_samples = y.index.intersection(matrix.columns.astype(str))
    y = y.loc[common_samples]
    matrix = matrix.loc[:, common_samples]
    batch = infer_batch_labels(y.index, args.gse63060_metadata, args.gse63061_metadata)

    rows: list[dict[str, Any]] = []
    if args.full_dataset:
        out_dir = args.result_root / "full_dataset" / "deg_ad_vs_mci"
        sample_ids = y.index.astype(str).tolist()
        summary = compute_deg(
            matrix,
            y,
            sample_ids,
            out_dir,
            args.adj_pval_threshold,
            args.min_abs_logfc,
            args.iqr_quantile,
            args.log_transform,
        )
        manifest = {
            "scenario": "full_dataset",
            "selection_scope": "full_dataset",
            "matrix_file": str(args.matrix_file),
            "sample_ids": sample_ids,
            "batches": sorted(batch.loc[sample_ids].unique().tolist()),
        }
        (out_dir / "full_dataset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        rows.append(
            {
                "seed": None,
                "scenario": "full_dataset",
                "samples": len(sample_ids),
                "batches": "|".join(manifest["batches"]),
                **summary,
                "output_dir": str(out_dir),
            }
        )
        summary_df = pd.DataFrame(rows)
        summary_df.to_csv(args.result_root / "summary.csv", index=False)
        print(
            f"full_dataset selected_genes={summary['selected_genes']} "
            f"AD={summary['train_ad']} MCI={summary['train_mci']}",
            flush=True,
        )
        print(f"Finished. Results written to {args.result_root}", flush=True)
        return

    for seed in args.seeds:
        for scenario in args.scenarios:
            split = make_split(
                y,
                batch,
                scenario,
                seed,
                args.shared_test_size,
                args.inner_val_ratio,
                args.pooled_test_size,
            )
            train_sample_ids = y.index[split["train_idx"]].astype(str).tolist()
            val_sample_ids = y.index[split["val_idx"]].astype(str).tolist()
            test_sample_ids = y.index[split["test_idx"]].astype(str).tolist()
            out_dir = args.result_root / scenario / f"seed_{seed}" / "deg_train_only"
            summary = compute_deg(
                matrix,
                y,
                train_sample_ids,
                out_dir,
                args.adj_pval_threshold,
                args.min_abs_logfc,
                args.iqr_quantile,
                args.log_transform,
            )
            split_manifest = {
                "seed": int(seed),
                "scenario": scenario,
                "selection_scope": "train_only",
                "matrix_file": str(args.matrix_file),
                "train_sample_ids": train_sample_ids,
                "val_sample_ids": val_sample_ids,
                "test_sample_ids": test_sample_ids,
                "train_batches": sorted(batch.loc[train_sample_ids].unique().tolist()),
                "val_batches": sorted(batch.loc[val_sample_ids].unique().tolist()),
                "test_batches": sorted(batch.loc[test_sample_ids].unique().tolist()),
            }
            (out_dir / "split_manifest.json").write_text(json.dumps(split_manifest, indent=2), encoding="utf-8")
            rows.append(
                {
                    "seed": int(seed),
                    "scenario": scenario,
                    "train_samples": len(train_sample_ids),
                    "val_samples": len(val_sample_ids),
                    "test_samples": len(test_sample_ids),
                    "train_ad": int(y.loc[train_sample_ids].sum()),
                    "train_mci": int((y.loc[train_sample_ids] == 0).sum()),
                    "train_batches": "|".join(split_manifest["train_batches"]),
                    "test_batches": "|".join(split_manifest["test_batches"]),
                    **summary,
                    "output_dir": str(out_dir),
                }
            )
            print(
                f"seed={seed} scenario={scenario} selected_genes={summary['selected_genes']} "
                f"train={summary['train_samples']} AD={summary['train_ad']} MCI={summary['train_mci']}",
                flush=True,
            )

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(args.result_root / "summary.csv", index=False)
    aggregate = (
        summary_df.groupby("scenario")["selected_genes"]
        .agg(["count", "mean", "std", "min", "median", "max"])
        .reset_index()
    )
    aggregate.to_csv(args.result_root / "summary_by_scenario.csv", index=False)
    print(f"Finished. Results written to {args.result_root}", flush=True)


if __name__ == "__main__":
    main()
