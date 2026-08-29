#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


METRIC_COLUMNS = [
    "loss",
    "accuracy",
    "macro_f1",
    "weighted_f1",
    "balanced_accuracy",
    "roc_auc_ovr_macro",
    "roc_auc_MCI",
    "roc_auc_AD",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate metrics across multiple run directories that differ only by split seed."
    )
    parser.add_argument("--runs-root", type=Path, required=True, help="Root directory containing one subdirectory per split-seed run.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where the aggregated CSV files are written. Default: <runs-root>/seed_summary",
    )
    return parser.parse_args()


def load_args_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def discover_run_dirs(runs_root: Path) -> list[Path]:
    run_dirs: list[Path] = []
    for metrics_path in sorted(runs_root.rglob("metrics_summary.csv")):
        run_dir = metrics_path.parent
        if (run_dir / "args.json").exists():
            run_dirs.append(run_dir)
    unique_run_dirs = sorted(set(run_dirs))
    if not unique_run_dirs:
        raise FileNotFoundError(f"No run directories with args.json + metrics_summary.csv found under: {runs_root}")
    return unique_run_dirs


def build_per_run_dataframe(runs_root: Path, run_dirs: list[Path]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for run_dir in run_dirs:
        args_payload = load_args_json(run_dir / "args.json")
        metrics_df = pd.read_csv(run_dir / "metrics_summary.csv")

        split_seed = args_payload.get("split_seed")
        if split_seed is None:
            split_seed = args_payload.get("seed")

        relative_run_dir = str(run_dir.relative_to(runs_root))
        for metric_row in metrics_df.itertuples(index=False):
            row = {
                "run_dir": str(run_dir),
                "relative_run_dir": relative_run_dir,
                "split": str(metric_row.split),
                "samples": int(metric_row.samples),
                "seed": args_payload.get("seed"),
                "split_seed": split_seed,
                "split_mode": args_payload.get("split_mode"),
                "max_genes": args_payload.get("max_genes"),
                "x_file": args_payload.get("x_file"),
                "y_file": args_payload.get("y_file"),
            }
            for column in METRIC_COLUMNS:
                row[column] = float(getattr(metric_row, column)) if hasattr(metric_row, column) else float("nan")
            rows.append(row)

    return pd.DataFrame(rows)


def build_summary_dataframe(per_run_df: pd.DataFrame) -> pd.DataFrame:
    aggregations = {"runs": ("relative_run_dir", "nunique"), "samples_mean": ("samples", "mean")}
    for column in METRIC_COLUMNS:
        if column in per_run_df.columns:
            aggregations[f"{column}_mean"] = (column, "mean")
            aggregations[f"{column}_std"] = (column, "std")
    summary_df = per_run_df.groupby("split", dropna=False).agg(**aggregations).reset_index()
    return summary_df.sort_values("split").reset_index(drop=True)


def main() -> None:
    args = parse_args()
    runs_root = args.runs_root.resolve()
    if not runs_root.exists():
        raise FileNotFoundError(f"Runs root not found: {runs_root}")

    output_dir = (args.output_dir.resolve() if args.output_dir is not None else runs_root / "seed_summary")
    output_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = discover_run_dirs(runs_root)
    per_run_df = build_per_run_dataframe(runs_root, run_dirs)
    summary_df = build_summary_dataframe(per_run_df)

    per_run_path = output_dir / "per_run_metrics.csv"
    summary_path = output_dir / "summary_by_split.csv"
    per_run_df.to_csv(per_run_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print(f"Runs root: {runs_root}")
    print(f"Run directories found: {len(run_dirs)}")
    print(f"Wrote: {per_run_path}")
    print(f"Wrote: {summary_path}")
    if not summary_df.empty:
        print("\nSummary by split:")
        print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
