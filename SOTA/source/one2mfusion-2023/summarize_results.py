from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


METRIC_COLS = ["pr_auc", "roc_auc", "accuracy", "balanced_accuracy", "precision", "recall", "macro_f1", "weighted_f1", "log_loss"]


def summarize(result_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics_files = sorted(result_root.glob("runs/**/metrics.csv"))
    rows = []
    for path in metrics_files:
        df = pd.read_csv(path)
        if not df.empty:
            rows.extend(df.to_dict("records"))
    all_metrics = pd.DataFrame(rows)
    if all_metrics.empty:
        return all_metrics, pd.DataFrame()
    numeric = [col for col in METRIC_COLS if col in all_metrics.columns]
    for col in numeric:
        all_metrics[col] = pd.to_numeric(all_metrics[col], errors="coerce")
    group_cols = [col for col in ["protocol", "scenario", "model"] if col in all_metrics.columns]
    summary = (
        all_metrics.groupby(group_cols, dropna=False)[numeric]
        .agg(["mean", "std", "count"])
        .sort_values(("pr_auc", "mean"), ascending=False)
        .reset_index()
    )
    summary.columns = ["_".join([str(part) for part in col if part]) for col in summary.columns.to_flat_index()]
    return all_metrics, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize One2MFusion reproduction results.")
    parser.add_argument("--result-root", type=Path, default=Path("results/SOTA/one2mfusion-2023"))
    args = parser.parse_args()
    all_metrics, summary = summarize(args.result_root)
    args.result_root.mkdir(parents=True, exist_ok=True)
    all_metrics.to_csv(args.result_root / "all_metrics.csv", index=False)
    summary.to_csv(args.result_root / "summary_by_method.csv", index=False)
    summary.to_csv(args.result_root / "ranking_by_pr_auc.csv", index=False)
    if not summary.empty:
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
