#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from source.pipeline.dataset import build_split_manifest, stratified_split
from source.pipeline.utils import DEFAULT_OFFICIAL_SPLIT_FILE, DEFAULT_SPLIT_DIR, DEFAULT_Y_FILE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the official split manifest for Alzheimer experiments.")
    parser.add_argument("--y-file", type=Path, default=DEFAULT_Y_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    return parser.parse_args()


def save_manifest(path: Path, manifest_df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest_df.to_csv(path, index=False)
    print(f"Wrote {path}")


def build_official_manifest(y_df: pd.DataFrame, seed: int, val_ratio: float, test_ratio: float) -> pd.DataFrame:
    sample_ids = y_df["sample_id"].to_numpy(dtype=str)
    labels = y_df["label"].to_numpy(dtype=int)
    train_idx, val_idx, test_idx = stratified_split(labels, val_ratio=val_ratio, test_ratio=test_ratio, seed=seed)
    return build_split_manifest(sample_ids, train_idx, val_idx, test_idx)


def main() -> None:
    args = parse_args()

    y_df = pd.read_csv(args.y_file)
    if "sample_id" not in y_df.columns or "label" not in y_df.columns:
        raise ValueError("y.csv must contain sample_id and label columns.")
    y_df["sample_id"] = y_df["sample_id"].astype(str).str.strip()

    official_manifest = build_official_manifest(y_df, seed=args.seed, val_ratio=args.val_ratio, test_ratio=args.test_ratio)
    save_manifest(args.output_dir / DEFAULT_OFFICIAL_SPLIT_FILE.name, official_manifest)


if __name__ == "__main__":
    main()
