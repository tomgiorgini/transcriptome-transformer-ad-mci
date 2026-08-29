#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


TASKS = ["AD_vs_MCI", "AD_vs_CTL", "MCI_vs_CTL"]
METRICS = {
    "roc_auc": "ROC-AUC",
    "macro_f1": "Macro-F1",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot validation vs test epoch curves from TxT multitask training_log.csv files."
    )
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--tasks", nargs="+", choices=TASKS + ["mean"], default=TASKS + ["mean"])
    return parser.parse_args()


def seed_from_path(path: Path) -> int | None:
    for part in path.parts:
        match = re.fullmatch(r"seed_(\d+)", part)
        if match:
            return int(match.group(1))
    return None


def load_epoch_rows(result_root: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for log_path in sorted(result_root.rglob("training_log.csv")):
        seed = seed_from_path(log_path)
        if seed is None:
            continue
        df = pd.read_csv(log_path)
        if "test_multitask_auc_mean" not in df.columns:
            continue
        config = log_path.parent.name
        for _, row in df.iterrows():
            epoch = int(row["epoch"])
            for task in TASKS:
                for metric in METRICS:
                    val_col = f"val_{task}_{metric}"
                    test_col = f"test_{task}_{metric}"
                    if val_col in df.columns and test_col in df.columns:
                        rows.append(
                            {
                                "config": config,
                                "seed": seed,
                                "epoch": epoch,
                                "task": task,
                                "metric": metric,
                                "val": row[val_col],
                                "test": row[test_col],
                            }
                        )
            for metric in METRICS:
                val_col = f"val_multitask_{'auc_mean' if metric == 'roc_auc' else 'macro_f1_mean'}"
                test_col = f"test_multitask_{'auc_mean' if metric == 'roc_auc' else 'macro_f1_mean'}"
                if val_col in df.columns and test_col in df.columns:
                    rows.append(
                        {
                            "config": config,
                            "seed": seed,
                            "epoch": epoch,
                            "task": "mean",
                            "metric": metric,
                            "val": row[val_col],
                            "test": row[test_col],
                        }
                    )
    if not rows:
        raise FileNotFoundError(
            f"No training_log.csv files with test epoch metrics found under {result_root}. "
            "Run training with --evaluate-test-each-epoch on."
        )
    return pd.DataFrame(rows)


def summarize_epoch_selection(long_df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        long_df.groupby(["task", "metric", "epoch"], as_index=False)
        .agg(val_mean=("val", "mean"), val_std=("val", "std"), test_mean=("test", "mean"), test_std=("test", "std"))
        .sort_values(["task", "metric", "epoch"])
    )
    rows = []
    for (task, metric), df in grouped.groupby(["task", "metric"]):
        best_val = df.loc[df["val_mean"].idxmax()]
        best_test = df.loc[df["test_mean"].idxmax()]
        rows.append(
            {
                "task": task,
                "metric": metric,
                "best_val_epoch": int(best_val["epoch"]),
                "best_val_mean": best_val["val_mean"],
                "test_at_best_val_epoch": best_val["test_mean"],
                "best_test_epoch": int(best_test["epoch"]),
                "best_test_mean": best_test["test_mean"],
                "val_at_best_test_epoch": best_test["val_mean"],
            }
        )
    return pd.DataFrame(rows).sort_values(["task", "metric"])


def plot_curves(long_df: pd.DataFrame, output_dir: Path, tasks: list[str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    grouped = (
        long_df.groupby(["task", "metric", "epoch"], as_index=False)
        .agg(val_mean=("val", "mean"), val_std=("val", "std"), test_mean=("test", "mean"), test_std=("test", "std"))
        .sort_values(["task", "metric", "epoch"])
    )
    for task in tasks:
        for metric, label in METRICS.items():
            df = grouped[(grouped["task"] == task) & (grouped["metric"] == metric)]
            if df.empty:
                continue
            fig, ax = plt.subplots(figsize=(8, 4.8), dpi=150)
            epochs = df["epoch"].to_numpy()
            val_mean = df["val_mean"].to_numpy()
            test_mean = df["test_mean"].to_numpy()
            val_std = df["val_std"].fillna(0.0).to_numpy()
            test_std = df["test_std"].fillna(0.0).to_numpy()
            ax.plot(epochs, val_mean, label="validation", color="#1f77b4", linewidth=2)
            ax.fill_between(epochs, val_mean - val_std, val_mean + val_std, color="#1f77b4", alpha=0.12)
            ax.plot(epochs, test_mean, label="test", color="#d62728", linewidth=2)
            ax.fill_between(epochs, test_mean - test_std, test_mean + test_std, color="#d62728", alpha=0.12)
            best_val_epoch = int(df.loc[df["val_mean"].idxmax(), "epoch"])
            best_test_epoch = int(df.loc[df["test_mean"].idxmax(), "epoch"])
            ax.axvline(best_val_epoch, color="#1f77b4", linestyle="--", alpha=0.45, label="best val epoch")
            ax.axvline(best_test_epoch, color="#d62728", linestyle=":", alpha=0.45, label="best test epoch")
            ax.set_title(f"{task} - {label}: validation vs test")
            ax.set_xlabel("Epoch")
            ax.set_ylabel(label)
            ax.grid(True, alpha=0.25)
            ax.legend()
            fig.tight_layout()
            fig.savefig(output_dir / f"{task.lower()}_{metric}_val_vs_test.png")
            plt.close(fig)


def main() -> None:
    args = parse_args()
    result_root = args.result_root.resolve()
    output_dir = args.output_dir.resolve() if args.output_dir is not None else result_root / "epoch_val_test_plots"
    long_df = load_epoch_rows(result_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    long_df.to_csv(output_dir / "epoch_val_test_metrics_long.csv", index=False)
    summary = summarize_epoch_selection(long_df)
    summary.to_csv(output_dir / "epoch_checkpoint_diagnostic_summary.csv", index=False)
    plot_curves(long_df, output_dir, args.tasks)
    print(f"Wrote epoch diagnostics to: {output_dir}")


if __name__ == "__main__":
    main()
