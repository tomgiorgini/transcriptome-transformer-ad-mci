from __future__ import annotations

from pathlib import Path

import pandas as pd


METRIC_COLS = [
    "pr_auc",
    "roc_auc",
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "macro_f1",
    "weighted_f1",
    "log_loss",
]


def summarize(result_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = [pd.read_csv(path) for path in result_root.rglob("metrics.csv")]
    if not frames:
        return pd.DataFrame(), pd.DataFrame()
    all_metrics = pd.concat(frames, ignore_index=True)
    group_cols = [col for col in ["protocol", "scenario", "model"] if col in all_metrics.columns]
    summary = all_metrics.groupby(group_cols, dropna=False)[METRIC_COLS].agg(["mean", "std", "count"]).reset_index()
    summary.columns = ["_".join([str(part) for part in col if part]) for col in summary.columns.to_flat_index()]
    if "pr_auc_mean" in summary.columns:
        summary = summary.sort_values("pr_auc_mean", ascending=False)
    return all_metrics, summary
