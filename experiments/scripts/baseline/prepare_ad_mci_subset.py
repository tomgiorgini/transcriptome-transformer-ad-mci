#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from source.pipeline.subsets import build_ad_mci_subset
from source.pipeline.utils import DEFAULT_OFFICIAL_SPLIT_FILE, DEFAULT_PROCESSED_DATA_DIR


DEFAULT_OUTPUT_DIR = ROOT / "task_dataset" / "processed" / "ad_mci_binary"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create an AD vs MCI subset from the canonical multiclass dataset using all genes."
    )
    parser.add_argument("--x-file", type=Path, default=DEFAULT_PROCESSED_DATA_DIR / "X.csv")
    parser.add_argument("--y-file", type=Path, default=DEFAULT_PROCESSED_DATA_DIR / "y.csv")
    parser.add_argument("--split-file", type=Path, default=DEFAULT_OFFICIAL_SPLIT_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    subset = build_ad_mci_subset(
        x_file=args.x_file,
        y_file=args.y_file,
        split_file=args.split_file,
        ordered_genes=None,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    split_out_dir = args.output_dir / "splits"
    split_out_dir.mkdir(parents=True, exist_ok=True)

    x_out = args.output_dir / "X_ad_mci.csv"
    y_out = args.output_dir / "y_ad_mci.csv"
    split_out = split_out_dir / args.split_file.name

    subset.x_df.to_csv(x_out, index=False)
    subset.y_df.to_csv(y_out, index=False)
    subset.split_df.to_csv(split_out, index=False)

    split_counts = subset.split_df["split"].value_counts().to_dict()

    print(f"Output directory: {args.output_dir}")
    print(f"Samples (AD + MCI): {len(subset.y_df)}")
    print(f"Genes kept: {len(subset.selected_gene_names)}")
    print(f"Class counts: {subset.class_counts}")
    print(f"Split counts: {split_counts}")
    print(f"Wrote: {x_out}")
    print(f"Wrote: {y_out}")
    print(f"Wrote: {split_out}")


if __name__ == "__main__":
    main()
