#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import optuna
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.scripts.finetuning.optuna_finetuning import (
    command_for_trial,
    np_mean,
    read_json,
    read_split_metrics,
    run_worker,
)


ORIGINAL_BOUNDS = {
    "lr_encoder": (1e-6, 5e-5),
    "lr_head": (5e-5, 1e-3),
    "weight_decay": (1e-6, 1e-3),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refine top Optuna fine-tuning trials on multiple split seeds.")
    parser.add_argument("--source-trials-csv", type=Path, required=True)
    parser.add_argument("--pretrained-checkpoint", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--trials-per-top", type=int, default=10)
    parser.add_argument("--tune-split-seeds", type=int, nargs="+", default=[101, 102, 103])
    parser.add_argument("--transfer-mode", choices=["full", "embedding_only", "random_init"], default="full")
    parser.add_argument("--dataset-mode", choices=["deg", "deg_pretrained_overlap"], default="deg_pretrained_overlap")
    parser.add_argument("--split-mode", choices=["stratified", "random"], default="stratified")
    parser.add_argument("--scaler", choices=["none", "minmax", "standard"], default="minmax")
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--early-stopping-patience", type=int, default=70)
    parser.add_argument("--device", choices=["cuda", "cpu", "mps"], default="cuda")
    parser.add_argument("--python-exe", type=str, default=sys.executable)
    parser.add_argument("--objective-metric", choices=["val_loss", "val_macro_f1", "val_roc_auc_ovr_macro"], default="val_macro_f1")
    parser.add_argument("--source-direction", choices=["maximize", "minimize"], default="maximize")
    parser.add_argument("--checkpoint-metric", choices=["val_loss", "val_macro_f1", "val_roc_auc_ovr_macro"], default="val_loss")
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--fixed-n-heads", type=int, default=2)
    parser.add_argument("--fixed-n-layers", type=int, default=1)
    parser.add_argument("--fixed-d-model", type=int, default=256)
    parser.add_argument("--fixed-d-ff", type=int, default=1024)
    parser.add_argument("--freeze-encoder-layers", type=str, default="")
    parser.add_argument("--freeze-embeddings", action="store_true")
    parser.add_argument("--freeze-tupe", action="store_true")
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    return parser.parse_args()


def normalize_param_name(column: str) -> str:
    return column.removeprefix("params_")


def read_top_params(source_csv: Path, top_k: int, source_direction: str) -> list[dict[str, Any]]:
    df = pd.read_csv(source_csv)
    if "value" not in df.columns:
        raise ValueError(f"{source_csv} must contain a value column.")
    df = df.loc[df.get("state", "COMPLETE").astype(str).str.upper().eq("COMPLETE")].copy()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"]).sort_values("value", ascending=source_direction == "minimize").head(top_k)
    if df.empty:
        raise ValueError(f"No completed trials found in {source_csv}.")

    top_params: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        params = {"source_trial": int(row["number"]), "source_value": float(row["value"])}
        for column in df.columns:
            if not column.startswith("params_"):
                continue
            name = normalize_param_name(column)
            value = row[column]
            if pd.isna(value):
                continue
            params[name] = parse_scalar(value)
        if "d_hidden2" in params and "d_hidden1" in params and int(params["d_hidden2"]) > int(params["d_hidden1"]):
            params["d_hidden2"] = int(params["d_hidden1"])
        top_params.append(params)
    return top_params


def parse_scalar(value: Any) -> Any:
    if isinstance(value, str):
        text = value.strip()
        if text.lower() in {"true", "false"}:
            return text.lower() == "true"
        try:
            number = float(text)
        except ValueError:
            return text
        if number.is_integer():
            return int(number)
        return number
    return value


def numeric_window(name: str, base: float, factor: float = 3.0) -> tuple[float, float]:
    lo, hi = ORIGINAL_BOUNDS[name]
    return max(lo, base / factor), min(hi, base * factor)


def categorical_window(base: Any, values: list[Any], neighbors: int = 1) -> list[Any]:
    if base not in values:
        return values
    idx = values.index(base)
    start = max(0, idx - neighbors)
    end = min(len(values), idx + neighbors + 1)
    return values[start:end]


def sample_near_base(trial: optuna.Trial, base: dict[str, Any]) -> dict[str, Any]:
    batch_values = categorical_window(int(base["batch_size"]), [8, 16, 32])
    clip_values = categorical_window(float(base["grad_clip_norm"]), [0.5, 1.0, 2.0])
    dropout_values = categorical_window(float(base["dropout"]), [0.10, 0.20, 0.30, 0.40])
    hidden1_values = categorical_window(int(base["d_hidden1"]), [64, 128, 256])
    hidden2_values = categorical_window(int(base["d_hidden2"]), [32, 64, 128])

    d_hidden1 = trial.suggest_categorical("d_hidden1", hidden1_values)
    d_hidden2 = trial.suggest_categorical("d_hidden2", hidden2_values)
    if d_hidden2 > d_hidden1:
        d_hidden2 = d_hidden1

    lr_encoder_low, lr_encoder_high = numeric_window("lr_encoder", float(base["lr_encoder"]))
    lr_head_low, lr_head_high = numeric_window("lr_head", float(base["lr_head"]))
    weight_decay_low, weight_decay_high = numeric_window("weight_decay", float(base["weight_decay"]))

    params = {
        "batch_size": trial.suggest_categorical("batch_size", batch_values),
        "lr_encoder": trial.suggest_float("lr_encoder", lr_encoder_low, lr_encoder_high, log=True),
        "lr_head": trial.suggest_float("lr_head", lr_head_low, lr_head_high, log=True),
        "weight_decay": trial.suggest_float("weight_decay", weight_decay_low, weight_decay_high, log=True),
        "grad_clip_norm": trial.suggest_categorical("grad_clip_norm", clip_values),
        "dropout": trial.suggest_categorical("dropout", dropout_values),
        "d_hidden1": d_hidden1,
        "d_hidden2": d_hidden2,
    }
    if "transfer_mode" in base:
        params["transfer_mode"] = base["transfer_mode"]
    if "freeze_encoder_layers" in base:
        params["freeze_encoder_layers"] = str(base["freeze_encoder_layers"])
    return params


def score_from_metrics(args: argparse.Namespace, val_metrics: dict[str, float], training_summary: dict[str, Any]) -> float:
    del training_summary
    return val_metrics[args.objective_metric.removeprefix("val_")]


def main() -> None:
    args = parse_args()
    args.source_trials_csv = (ROOT / args.source_trials_csv).resolve() if not args.source_trials_csv.is_absolute() else args.source_trials_csv
    args.pretrained_checkpoint = (ROOT / args.pretrained_checkpoint).resolve() if not args.pretrained_checkpoint.is_absolute() else args.pretrained_checkpoint
    args.result_root = (ROOT / args.result_root).resolve() if not args.result_root.is_absolute() else args.result_root
    args.result_root.mkdir(parents=True, exist_ok=True)

    top_params = read_top_params(args.source_trials_csv, args.top_k, args.source_direction)
    with (args.result_root / "source_top_trials.json").open("w", encoding="utf-8") as handle:
        json.dump(top_params, handle, indent=2)

    all_rows: list[dict[str, Any]] = []
    best_overall: dict[str, Any] | None = None

    for top_index, base in enumerate(top_params, start=1):
        study_name = f"refine_top{top_index}_source{base['source_trial']}"
        direction = "minimize" if args.objective_metric == "val_loss" else "maximize"
        study = optuna.create_study(
            study_name=study_name,
            direction=direction,
            sampler=optuna.samplers.TPESampler(seed=1000 + top_index, multivariate=True),
        )

        def objective(trial: optuna.Trial) -> float:
            params = sample_near_base(trial, base)
            scores: list[float] = []
            split_rows: list[dict[str, Any]] = []
            for split_seed in args.tune_split_seeds:
                split_dir = args.result_root / f"top_{top_index}_source_{base['source_trial']}" / f"trial_{trial.number:04d}" / f"split_seed_{split_seed}"
                cmd = command_for_trial(args, params, split_seed, split_dir)
                print("\n" + "=" * 80, flush=True)
                print(
                    f"Refine top {top_index}/{len(top_params)} source_trial={base['source_trial']} "
                    f"local_trial={trial.number} split_seed={split_seed}",
                    flush=True,
                )
                print(f"params={params}", flush=True)
                print("=" * 80, flush=True)
                run_worker(cmd, split_dir / "finetune.log")

                metrics_path = split_dir / "metrics_summary.csv"
                summary_path = split_dir / "model_summary.json"
                val_metrics = read_split_metrics(metrics_path, "val")
                test_metrics = read_split_metrics(metrics_path, "test")
                training_summary = read_json(summary_path).get("training_summary", {})
                score = score_from_metrics(args, val_metrics, training_summary)
                scores.append(score)
                split_rows.append(
                    {
                        "top_index": top_index,
                        "source_trial": base["source_trial"],
                        "source_value": base["source_value"],
                        "local_trial": trial.number,
                        "split_seed": split_seed,
                        "objective_score": score,
                        "val_macro_f1": val_metrics["macro_f1"],
                        "val_balanced_accuracy": val_metrics["balanced_accuracy"],
                        "val_roc_auc": val_metrics["roc_auc"],
                        "test_macro_f1": test_metrics["macro_f1"],
                        "test_accuracy": test_metrics["accuracy"],
                        "test_balanced_accuracy": test_metrics["balanced_accuracy"],
                        "test_roc_auc": test_metrics["roc_auc"],
                        "test_pr_auc": test_metrics["pr_auc"],
                        "test_recall_mci": test_metrics["recall_mci"],
                        "test_recall_ad": test_metrics["recall_ad"],
                        "best_epoch": training_summary.get("best_epoch", math.nan),
                        **params,
                    }
                )
            all_rows.extend(split_rows)
            pd.DataFrame(all_rows).to_csv(args.result_root / "refinement_trial_splits.csv", index=False)
            value = np_mean(scores)
            nonlocal best_overall
            is_better = value < best_overall["value"] if args.objective_metric == "val_loss" and best_overall is not None else True
            if args.objective_metric != "val_loss" and best_overall is not None:
                is_better = value > best_overall["value"]
            if best_overall is None or is_better:
                best_overall = {
                    "top_index": top_index,
                    "source_trial": base["source_trial"],
                    "local_trial": trial.number,
                    "value": value,
                    "params": params,
                }
                with (args.result_root / "best_refinement.json").open("w", encoding="utf-8") as handle:
                    json.dump(best_overall, handle, indent=2)
            return value

        study.optimize(objective, n_trials=args.trials_per_top, gc_after_trial=True)
        study.trials_dataframe().to_csv(args.result_root / f"refine_top{top_index}_trials.csv", index=False)

    pd.DataFrame(all_rows).to_csv(args.result_root / "refinement_trial_splits.csv", index=False)
    if all_rows:
        summary = (
            pd.DataFrame(all_rows)
            .groupby(["top_index", "source_trial", "local_trial"], as_index=False)
            .agg(
                objective_score_mean=("objective_score", "mean"),
                val_macro_f1_mean=("val_macro_f1", "mean"),
                test_macro_f1_mean=("test_macro_f1", "mean"),
                test_accuracy_mean=("test_accuracy", "mean"),
                test_balanced_accuracy_mean=("test_balanced_accuracy", "mean"),
                test_roc_auc_mean=("test_roc_auc", "mean"),
                test_pr_auc_mean=("test_pr_auc", "mean"),
                test_recall_mci_mean=("test_recall_mci", "mean"),
                test_recall_ad_mean=("test_recall_ad", "mean"),
            )
            .sort_values("objective_score_mean", ascending=args.objective_metric == "val_loss")
        )
        summary.to_csv(args.result_root / "refinement_summary.csv", index=False)
    print(f"Refinement complete: {args.result_root}", flush=True)


if __name__ == "__main__":
    main()
