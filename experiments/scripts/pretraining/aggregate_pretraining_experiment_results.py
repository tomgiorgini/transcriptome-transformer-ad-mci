#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd


TXT_BASELINE_ACCURACY = 0.6845070422535211
TXT_BASELINE_MACRO_F1 = 0.666539112072838
TGEM_BASELINE_ACCURACY = 0.7042253521126761
TGEM_BASELINE_MACRO_F1 = 0.6676665703511908


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate TxT pretraining/fine-tuning runs into presentation-ready comparison tables."
    )
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: <results-root>/final_tables",
    )
    parser.add_argument("--txt-baseline-accuracy", type=float, default=TXT_BASELINE_ACCURACY)
    parser.add_argument("--txt-baseline-macro-f1", type=float, default=TXT_BASELINE_MACRO_F1)
    parser.add_argument("--tgem-baseline-accuracy", type=float, default=TGEM_BASELINE_ACCURACY)
    parser.add_argument("--tgem-baseline-macro-f1", type=float, default=TGEM_BASELINE_MACRO_F1)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


def safe_float(value: Any) -> float:
    if value is None:
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        if isinstance(value, float) and math.isnan(value):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def metric_for_split(metrics_df: pd.DataFrame, split: str, column: str) -> float:
    if "split" not in metrics_df.columns or column not in metrics_df.columns:
        return math.nan
    rows = metrics_df.loc[metrics_df["split"].astype(str) == split, column]
    if rows.empty:
        return math.nan
    return safe_float(rows.iloc[0])


def infer_architecture(args_payload: dict[str, Any], checkpoint_path: str) -> str:
    pretrained_config = args_payload.get("pretrained_config")
    if not isinstance(pretrained_config, dict):
        pretrained_config = {}

    n_layers = safe_int(pretrained_config.get("n_layers"))
    n_heads = safe_int(pretrained_config.get("n_heads"))
    d_model = safe_int(pretrained_config.get("d_model"))
    lower_path = checkpoint_path.lower()

    if "baseline_arch" in lower_path:
        return "baseline_arch"
    if "4layer" in lower_path or "4_layer" in lower_path or "400ep" in lower_path:
        return "larger_4layer"
    if n_layers == 1 and n_heads == 2 and d_model == 256:
        return "baseline_arch"
    if n_layers is not None and n_layers >= 4:
        return "larger_4layer"
    return "unknown"


def infer_reference_policy(args_payload: dict[str, Any], checkpoint_path: str) -> str:
    candidates: list[str] = [checkpoint_path]
    pretrained_config = args_payload.get("pretrained_config")
    if isinstance(pretrained_config, dict):
        matrix_file = pretrained_config.get("matrix_file")
        if matrix_file is not None:
            candidates.append(str(matrix_file))

    joined = " ".join(candidates).lower()
    if "no_reference" in joined or "noreference" in joined:
        return "no_reference"
    if "with_reference" in joined or "reference" in joined:
        return "with_reference"
    return "unknown"


def missing_gene_count(transfer_report: dict[str, Any]) -> int | None:
    explicit_count = safe_int(transfer_report.get("missing_genes"))
    if explicit_count is not None:
        return explicit_count
    names = transfer_report.get("missing_gene_names")
    if isinstance(names, list):
        return len(names)
    return None


def training_log_stats(path: Path) -> dict[str, float | int]:
    if not path.exists():
        return {
            "training_epochs": 0,
            "val_macro_f1_max": math.nan,
            "collapse_epochs_near_0377": 0,
            "collapse_fraction_near_0377": math.nan,
        }
    log_df = pd.read_csv(path)
    if "val_macro_f1" not in log_df.columns:
        return {
            "training_epochs": int(len(log_df)),
            "val_macro_f1_max": math.nan,
            "collapse_epochs_near_0377": 0,
            "collapse_fraction_near_0377": math.nan,
        }
    val_macro = pd.to_numeric(log_df["val_macro_f1"], errors="coerce").dropna()
    if val_macro.empty:
        return {
            "training_epochs": int(len(log_df)),
            "val_macro_f1_max": math.nan,
            "collapse_epochs_near_0377": 0,
            "collapse_fraction_near_0377": math.nan,
        }
    collapse_epochs = int(((val_macro - 0.3772).abs() <= 0.005).sum())
    return {
        "training_epochs": int(len(log_df)),
        "val_macro_f1_max": float(val_macro.max()),
        "collapse_epochs_near_0377": collapse_epochs,
        "collapse_fraction_near_0377": float(collapse_epochs / len(val_macro)),
    }


def build_row(results_root: Path, metrics_path: Path) -> dict[str, Any]:
    run_dir = metrics_path.parent
    metrics_df = pd.read_csv(metrics_path)
    metadata = load_json(run_dir / "experiment_metadata.json")
    args_payload = load_json(run_dir / "args.json")
    transfer_report = load_json(run_dir / "transfer_report.json")
    model_summary = load_json(run_dir / "model_summary.json")
    log_stats = training_log_stats(run_dir / "training_log.csv")

    checkpoint_path = str(
        metadata.get("checkpoint_path")
        or args_payload.get("pretrained_checkpoint")
        or ""
    )
    transfer_mode = str(metadata.get("transfer_mode") or args_payload.get("transfer_mode") or "unknown")
    split_seed = safe_int(metadata.get("split_seed") or args_payload.get("split_seed") or args_payload.get("seed"))
    pretraining_arch = str(
        metadata.get("pretraining_arch") or infer_architecture(args_payload, checkpoint_path)
    )
    reference_policy = str(
        metadata.get("reference_policy") or infer_reference_policy(args_payload, checkpoint_path)
    )
    training_summary = model_summary.get("training_summary")
    if not isinstance(training_summary, dict):
        training_summary = {}

    return {
        "pretraining_arch": pretraining_arch,
        "reference_policy": reference_policy,
        "transfer_mode": transfer_mode,
        "split_seed": split_seed,
        "test_accuracy": metric_for_split(metrics_df, "test", "accuracy"),
        "test_macro_f1": metric_for_split(metrics_df, "test", "macro_f1"),
        "val_macro_f1": metric_for_split(metrics_df, "val", "macro_f1"),
        "matched_genes": safe_int(transfer_report.get("matched_genes")),
        "missing_genes": missing_gene_count(transfer_report),
        "best_epoch": safe_int(training_summary.get("best_epoch")),
        **log_stats,
        "val_accuracy": metric_for_split(metrics_df, "val", "accuracy"),
        "train_accuracy": metric_for_split(metrics_df, "train", "accuracy"),
        "train_macro_f1": metric_for_split(metrics_df, "train", "macro_f1"),
        "split_mode": metadata.get("split_mode") or args_payload.get("split_mode"),
        "scaler": metadata.get("scaler") or args_payload.get("scaler"),
        "class_weighting": metadata.get("class_weighting") or args_payload.get("class_weighting"),
        "learning_rate": safe_float(metadata.get("learning_rate") or args_payload.get("lr")),
        "freeze_encoder_epochs": safe_int(
            metadata.get("freeze_encoder_epochs") or args_payload.get("freeze_encoder_epochs")
        ),
        "x_file": metadata.get("x_file") or args_payload.get("x_file"),
        "random_gene_count": safe_int(
            metadata.get("random_gene_count")
            or args_payload.get("random_gene_count")
            or transfer_report.get("selected_gene_count")
        ),
        "random_gene_sampling": metadata.get("random_gene_sampling")
        or args_payload.get("random_gene_sampling")
        or metadata.get("gene_selection_mode")
        or transfer_report.get("gene_selection_mode"),
        "random_gene_pool": metadata.get("random_gene_pool")
        or args_payload.get("random_gene_pool")
        or transfer_report.get("gene_selection_pool"),
        "eval_gene_samples": safe_int(metadata.get("eval_gene_samples") or args_payload.get("eval_gene_samples")),
        "filter_to_pretrained_genes": metadata.get("filter_to_pretrained_genes")
        if "filter_to_pretrained_genes" in metadata
        else args_payload.get("filter_to_pretrained_genes"),
        "checkpoint_path": checkpoint_path,
        "relative_run_dir": str(run_dir.relative_to(results_root)),
        "run_dir": str(run_dir),
    }


def discover_rows(results_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for metrics_path in sorted(results_root.rglob("metrics_summary.csv")):
        if "final_tables" in metrics_path.parts or "seed_summary" in metrics_path.parts:
            continue
        rows.append(build_row(results_root, metrics_path))
    if not rows:
        raise FileNotFoundError(f"No metrics_summary.csv files found under: {results_root}")
    return pd.DataFrame(rows)


def build_summary(runs_df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        runs_df.groupby(["pretraining_arch", "reference_policy", "transfer_mode"], dropna=False)
        .agg(
            runs=("relative_run_dir", "nunique"),
            split_seeds=("split_seed", lambda values: ",".join(str(int(v)) for v in sorted(values.dropna().unique()))),
            test_accuracy_mean=("test_accuracy", "mean"),
            test_accuracy_std=("test_accuracy", "std"),
            test_macro_f1_mean=("test_macro_f1", "mean"),
            test_macro_f1_std=("test_macro_f1", "std"),
            val_macro_f1_mean=("val_macro_f1", "mean"),
            val_macro_f1_std=("val_macro_f1", "std"),
            matched_genes_mean=("matched_genes", "mean"),
            missing_genes_mean=("missing_genes", "mean"),
            best_epoch_mean=("best_epoch", "mean"),
            val_macro_f1_max_mean=("val_macro_f1_max", "mean"),
            collapse_fraction_near_0377_mean=("collapse_fraction_near_0377", "mean"),
        )
        .reset_index()
    )
    return grouped.sort_values(
        ["test_macro_f1_mean", "test_accuracy_mean"],
        ascending=[False, False],
        na_position="last",
    ).reset_index(drop=True)


def add_baseline_deltas(summary_df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    comparison_df = summary_df.copy()
    comparison_df["txt_baseline_accuracy"] = args.txt_baseline_accuracy
    comparison_df["txt_baseline_macro_f1"] = args.txt_baseline_macro_f1
    comparison_df["tgem_baseline_accuracy"] = args.tgem_baseline_accuracy
    comparison_df["tgem_baseline_macro_f1"] = args.tgem_baseline_macro_f1
    comparison_df["delta_accuracy_vs_txt_baseline"] = (
        comparison_df["test_accuracy_mean"] - args.txt_baseline_accuracy
    )
    comparison_df["delta_macro_f1_vs_txt_baseline"] = (
        comparison_df["test_macro_f1_mean"] - args.txt_baseline_macro_f1
    )
    comparison_df["delta_accuracy_vs_tgem_baseline"] = (
        comparison_df["test_accuracy_mean"] - args.tgem_baseline_accuracy
    )
    comparison_df["delta_macro_f1_vs_tgem_baseline"] = (
        comparison_df["test_macro_f1_mean"] - args.tgem_baseline_macro_f1
    )
    return comparison_df


def main() -> None:
    args = parse_args()
    results_root = args.results_root.resolve()
    if not results_root.exists():
        raise FileNotFoundError(f"Results root not found: {results_root}")

    output_dir = args.output_dir.resolve() if args.output_dir is not None else results_root / "final_tables"
    output_dir.mkdir(parents=True, exist_ok=True)

    runs_df = discover_rows(results_root)
    summary_df = build_summary(runs_df)
    comparison_df = add_baseline_deltas(summary_df, args)

    runs_path = output_dir / "pretraining_finetune_runs.csv"
    summary_path = output_dir / "pretraining_finetune_summary.csv"
    comparison_path = output_dir / "baseline_comparison.csv"
    runs_df.to_csv(runs_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    comparison_df.to_csv(comparison_path, index=False)

    print(f"Results root: {results_root}")
    print(f"Runs found: {len(runs_df)}")
    print(f"Wrote: {runs_path}")
    print(f"Wrote: {summary_path}")
    print(f"Wrote: {comparison_path}")
    print("")
    print("Top configurations by test macro F1:")
    columns = [
        "pretraining_arch",
        "reference_policy",
        "transfer_mode",
        "runs",
        "test_accuracy_mean",
        "test_accuracy_std",
        "test_macro_f1_mean",
        "test_macro_f1_std",
        "val_macro_f1_mean",
    ]
    print(summary_df.loc[:, columns].to_string(index=False))


if __name__ == "__main__":
    main()
