from __future__ import annotations

from pathlib import Path

import pandas as pd


def summarize(result_root: Path) -> None:
    frames = [pd.read_csv(path) for path in result_root.rglob("metrics.csv")]
    if not frames:
        return
    all_metrics = pd.concat(frames, ignore_index=True)
    all_metrics.to_csv(result_root / "all_metrics.csv", index=False)
    metric_cols = ["pr_auc", "roc_auc", "accuracy", "balanced_accuracy", "precision", "recall", "macro_f1", "weighted_f1", "log_loss"]
    group_cols = ["protocol", "scenario", "feature_set", "model"]
    summary = all_metrics.groupby(group_cols, dropna=False)[metric_cols].agg(["mean", "std", "count"]).reset_index()
    summary.columns = ["_".join([str(part) for part in col if part]) for col in summary.columns.to_flat_index()]
    summary = summary.sort_values("roc_auc_mean", ascending=False)
    summary.to_csv(result_root / "summary_by_method.csv", index=False)
    summary.sort_values("pr_auc_mean", ascending=False).to_csv(result_root / "ranking_by_pr_auc.csv", index=False)
    summary.sort_values("roc_auc_mean", ascending=False).to_csv(result_root / "ranking_by_roc_auc.csv", index=False)
    summary.sort_values("macro_f1_mean", ascending=False).to_csv(result_root / "ranking_by_macro_f1.csv", index=False)
    summary.sort_values("accuracy_mean", ascending=False).to_csv(result_root / "ranking_by_accuracy.csv", index=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("result_root", type=Path)
    summarize(parser.parse_args().result_root)
