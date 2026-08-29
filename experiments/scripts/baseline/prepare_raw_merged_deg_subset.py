#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT_DIR = ROOT / "task_dataset" / "processed" / "ad_mci_merged_raw"
DEFAULT_X_FILE = DEFAULT_INPUT_DIR / "X_ad_mci_merged_raw.csv"
DEFAULT_Y_FILE = DEFAULT_INPUT_DIR / "y_ad_mci_merged_raw.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute AD vs MCI DEG-like subsets directly on the raw merged, non batch-corrected dataset."
    )
    parser.add_argument("--x-file", type=Path, default=DEFAULT_X_FILE)
    parser.add_argument("--y-file", type=Path, default=DEFAULT_Y_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--top-n", type=int, default=269)
    parser.add_argument("--nominal-p-cutoff", type=float, default=0.05)
    return parser.parse_args()


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    p_values = np.asarray(p_values, dtype=float)
    adjusted = np.full_like(p_values, np.nan, dtype=float)
    valid = np.isfinite(p_values)
    valid_p = p_values[valid]
    order = np.argsort(valid_p)
    ranked = valid_p[order] * len(valid_p) / np.arange(1, len(valid_p) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    ranked = np.clip(ranked, 0.0, 1.0)
    valid_adjusted = np.empty_like(valid_p)
    valid_adjusted[order] = ranked
    adjusted[valid] = valid_adjusted
    return adjusted


def make_subset(x_df: pd.DataFrame, genes: list[str]) -> pd.DataFrame:
    return x_df[["sample_id", *genes]].copy()


def main() -> None:
    args = parse_args()
    x_df = pd.read_csv(args.x_file)
    y_df = pd.read_csv(args.y_file)

    if "sample_id" not in x_df.columns:
        raise ValueError(f"{args.x_file} must contain a sample_id column.")
    if "sample_id" not in y_df.columns or "label_name" not in y_df.columns:
        raise ValueError(f"{args.y_file} must contain sample_id and label_name columns.")

    merged = x_df.merge(y_df[["sample_id", "label_name"]], on="sample_id", how="inner")
    labels = merged["label_name"].astype(str).str.upper()
    gene_columns = [column for column in x_df.columns if column != "sample_id"]

    ad_values = merged.loc[labels.eq("AD"), gene_columns].to_numpy(dtype=float)
    mci_values = merged.loc[labels.eq("MCI"), gene_columns].to_numpy(dtype=float)
    if ad_values.size == 0 or mci_values.size == 0:
        raise ValueError("Both AD and MCI samples are required to compute DEG statistics.")

    statistic, p_values = ttest_ind(ad_values, mci_values, axis=0, equal_var=False, nan_policy="omit")
    adj_p_values = benjamini_hochberg(p_values)
    ad_mean = np.nanmean(ad_values, axis=0)
    mci_mean = np.nanmean(mci_values, axis=0)
    log_fc = ad_mean - mci_mean

    stats_df = pd.DataFrame(
        {
            "GeneSymbol": gene_columns,
            "statistic": statistic,
            "pval": p_values,
            "adj_pval": adj_p_values,
            "logFC_AD_minus_MCI": log_fc,
            "abs_logFC": np.abs(log_fc),
            "mean_AD": ad_mean,
            "mean_MCI": mci_mean,
        }
    ).sort_values(["pval", "abs_logFC"], ascending=[True, False])

    nominal_genes = stats_df.loc[stats_df["pval"] < args.nominal_p_cutoff, "GeneSymbol"].tolist()
    top_genes = stats_df["GeneSymbol"].head(args.top_n).tolist()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stats_path = args.output_dir / "deg_stats_ad_mci_merged_raw.tsv"
    nominal_x_path = args.output_dir / "X_deg_nominal_p005_ad_mci_merged_raw.csv"
    top_x_path = args.output_dir / f"X_deg_top{args.top_n}_ad_mci_merged_raw.csv"
    nominal_genes_path = args.output_dir / "deg_nominal_p005_genes_ad_mci_merged_raw.txt"
    top_genes_path = args.output_dir / f"deg_top{args.top_n}_genes_ad_mci_merged_raw.txt"

    stats_df.to_csv(stats_path, sep="\t", index=False)
    make_subset(x_df, nominal_genes).to_csv(nominal_x_path, index=False)
    make_subset(x_df, top_genes).to_csv(top_x_path, index=False)
    pd.Series(nominal_genes).to_csv(nominal_genes_path, index=False, header=False)
    pd.Series(top_genes).to_csv(top_genes_path, index=False, header=False)

    print(f"Input samples: {len(merged)}")
    print(f"AD samples: {int(labels.eq('AD').sum())}")
    print(f"MCI samples: {int(labels.eq('MCI').sum())}")
    print(f"Input genes: {len(gene_columns)}")
    print(f"FDR < 0.05 genes: {int((stats_df['adj_pval'] < 0.05).sum())}")
    print(f"Nominal p < {args.nominal_p_cutoff:g} genes: {len(nominal_genes)}")
    print(f"Top-N DEG-like genes: {len(top_genes)}")
    print(f"Wrote: {stats_path}")
    print(f"Wrote: {nominal_x_path}")
    print(f"Wrote: {top_x_path}")


if __name__ == "__main__":
    main()
