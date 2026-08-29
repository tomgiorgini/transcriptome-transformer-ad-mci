#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from source.pipeline.subsets import build_ad_mci_subset
from source.pipeline.utils import DEFAULT_OFFICIAL_SPLIT_FILE, DEFAULT_PROCESSED_DATA_DIR


DEFAULT_DEG_FILE = ROOT / "deg_analysis" / "Results" / "DEG" / "AD_vs_MCI" / "DEG.txt"
DEFAULT_OUTPUT_DIR = ROOT / "task_dataset" / "processed" / "ad_mci_deg"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create an AD vs MCI subset filtered by the AD_vs_MCI DEG list (GeneSymbol order preserved)."
    )
    parser.add_argument("--x-file", type=Path, default=DEFAULT_PROCESSED_DATA_DIR / "X.csv")
    parser.add_argument("--y-file", type=Path, default=DEFAULT_PROCESSED_DATA_DIR / "y.csv")
    parser.add_argument("--split-file", type=Path, default=DEFAULT_OFFICIAL_SPLIT_FILE)
    parser.add_argument("--deg-file", type=Path, default=DEFAULT_DEG_FILE, help="TSV file containing a GeneSymbol column.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def load_ordered_deg_genes(deg_file: Path) -> list[str]:
    if not deg_file.exists():
        raise FileNotFoundError(f"DEG file not found: {deg_file}")

    deg_df = pd.read_csv(deg_file, sep="\t")
    if "GeneSymbol" not in deg_df.columns:
        raise ValueError("The DEG file must contain a GeneSymbol column.")

    seen: set[str] = set()
    ordered_genes: list[str] = []
    for value in deg_df["GeneSymbol"].dropna():
        gene_name = str(value).strip()
        if not gene_name or gene_name == "nan" or gene_name in seen:
            continue
        ordered_genes.append(gene_name)
        seen.add(gene_name)
    if not ordered_genes:
        raise ValueError(f"No valid genes found in DEG file: {deg_file}")
    return ordered_genes


def main() -> None:
    args = parse_args()
    ordered_deg_genes = load_ordered_deg_genes(args.deg_file)

    subset = build_ad_mci_subset(
        x_file=args.x_file,
        y_file=args.y_file,
        split_file=args.split_file,
        ordered_genes=ordered_deg_genes,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    split_out_dir = args.output_dir / "splits"
    split_out_dir.mkdir(parents=True, exist_ok=True)

    x_out = args.output_dir / "X_deg_ad_mci.csv"
    y_out = args.output_dir / "y_ad_mci.csv"
    split_out = split_out_dir / args.split_file.name

    subset.x_df.to_csv(x_out, index=False)
    subset.y_df.to_csv(y_out, index=False)
    subset.split_df.to_csv(split_out, index=False)

    split_counts = subset.split_df["split"].value_counts().to_dict()
    print(f"Output directory: {args.output_dir}")
    print(f"DEG input genes: {len(ordered_deg_genes)}")
    print(f"Matched DEG genes in X: {len(subset.selected_gene_names)}")
    print(f"Samples (AD + MCI): {len(subset.y_df)}")
    print(f"Class counts: {subset.class_counts}")
    print(f"Split counts: {split_counts}")
    print(f"Wrote: {x_out}")
    print(f"Wrote: {y_out}")
    print(f"Wrote: {split_out}")


if __name__ == "__main__":
    main()
