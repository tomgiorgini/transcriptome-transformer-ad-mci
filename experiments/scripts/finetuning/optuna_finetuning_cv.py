#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import optuna
import pandas as pd
from sklearn.model_selection import StratifiedKFold


ROOT = Path(__file__).resolve().parents[3]
WORKER = ROOT / "experiments" / "scripts" / "finetuning" / "finetune_txt.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="5-fold CV Optuna tuning for TxT fine-tuning.")
    parser.add_argument("--pretrained-checkpoint", type=Path, required=True)
    parser.add_argument("--x-file", type=Path, default=ROOT / "task_dataset" / "processed" / "ad_mci_deg" / "X_deg_ad_mci.csv")
    parser.add_argument("--y-file", type=Path, default=ROOT / "task_dataset" / "processed" / "ad_mci_deg" / "y_ad_mci.csv")
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--study-name", default="txt_finetuning_5cv")
    parser.add_argument("--storage", default="")
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--cv-seed", type=int, default=42)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--transfer-modes", choices=["random_init", "embedding_only", "full"], nargs="+", default=["random_init", "embedding_only", "full"])
    parser.add_argument("--dataset-mode", choices=["deg", "deg_pretrained_overlap"], default="deg_pretrained_overlap")
    parser.add_argument("--max-genes", type=int, default=0)
    parser.add_argument("--gene-selection", choices=["variance", "mad", "class_aware_variance"], default="variance")
    parser.add_argument("--scaler", choices=["none", "minmax", "standard"], default="minmax")
    parser.add_argument("--scaler-fit-scope", choices=["train", "all"], default="train")
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--early-stopping-patience", type=int, default=45)
    parser.add_argument("--device", choices=["cuda", "cpu", "mps"], default="cuda")
    parser.add_argument("--python-exe", default=sys.executable)
    parser.add_argument(
        "--objective-metric",
        choices=[
            "val_macro_f1",
            "val_roc_auc_ovr_macro",
            "val_auc_f1_50_50",
            "val_auc_f1_50_50_minus_025_loss",
        ],
        default="val_auc_f1_50_50",
    )
    parser.add_argument(
        "--checkpoint-metric",
        choices=[
            "val_macro_f1",
            "val_roc_auc_ovr_macro",
            "val_auc_f1_50_50",
            "val_auc_f1_50_50_minus_025_loss",
        ],
        default="val_auc_f1_50_50",
    )
    parser.add_argument("--class-weighting", choices=["on", "off"], default="off")
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--n-heads", type=int, default=2)
    parser.add_argument("--n-layers", type=int, default=1)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--d-ff", type=int, default=1024)
    parser.add_argument("--d-hidden1", type=int, default=128)
    parser.add_argument("--d-hidden2", type=int, default=64)
    parser.add_argument("--aggfunc", choices=["Flatten", "Avgpool"], default="Avgpool")
    parser.add_argument("--lr-low", type=float, default=None)
    parser.add_argument("--lr-high", type=float, default=None)
    parser.add_argument("--weight-decay-low", type=float, default=1e-6)
    parser.add_argument("--weight-decay-high", type=float, default=5e-4)
    parser.add_argument("--dropout-options", type=float, nargs="+", default=[0.2, 0.3, 0.4, 0.5])
    parser.add_argument("--include-head-hidden", action="store_true")
    parser.add_argument("--include-weight-decay", action="store_true")
    parser.add_argument("--include-grad-clip", action="store_true")
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else (ROOT / path).resolve()


def load_labels(y_file: Path) -> pd.DataFrame:
    y_df = pd.read_csv(y_file)
    if "sample_id" not in y_df.columns:
        raise ValueError(f"{y_file} must contain sample_id.")
    label_col = "label" if "label" in y_df.columns else "label_name"
    if label_col not in y_df.columns:
        raise ValueError(f"{y_file} must contain label or label_name.")
    return y_df[["sample_id", label_col]].rename(columns={label_col: "label"})


def write_cv_splits(y_df: pd.DataFrame, output_dir: Path, n_folds: int, seed: int) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    split_paths: list[Path] = []
    labels = y_df["label"].astype(str).to_numpy()
    sample_ids = y_df["sample_id"].astype(str).to_numpy()
    for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(sample_ids, labels), start=1):
        split_df = pd.DataFrame(
            [{"sample_id": sample_ids[idx], "split": "train"} for idx in train_idx]
            + [{"sample_id": sample_ids[idx], "split": "val"} for idx in val_idx]
        )
        split_path = output_dir / f"fold_{fold_idx:02d}.csv"
        split_df.to_csv(split_path, index=False)
        split_paths.append(split_path)
    return split_paths


def sample_params(trial: optuna.Trial, args: argparse.Namespace) -> dict[str, Any]:
    transfer_mode = trial.suggest_categorical("transfer_mode", args.transfer_modes)
    if args.lr_low is not None and args.lr_high is not None:
        lr_low = min(args.lr_low, args.lr_high)
        lr_high = max(args.lr_low, args.lr_high)
    else:
        if transfer_mode == "full":
            lr_low, lr_high = 1e-5, 8e-5
        elif transfer_mode == "embedding_only":
            lr_low, lr_high = 8e-5, 3e-4
        else:
            lr_low, lr_high = 1.2e-4, 2.5e-4

    params: dict[str, Any] = {
        "transfer_mode": transfer_mode,
        "batch_size": trial.suggest_categorical("batch_size", [8, 16, 32]),
        "lr": trial.suggest_float("lr", lr_low, lr_high, log=True),
        "dropout": trial.suggest_categorical("dropout", args.dropout_options),
        "weight_decay": 1e-4,
        "grad_clip_norm": 1.0,
        "d_hidden1": args.d_hidden1,
        "d_hidden2": min(args.d_hidden1, args.d_hidden2),
    }
    if args.include_weight_decay:
        wd_low = min(args.weight_decay_low, args.weight_decay_high)
        wd_high = max(args.weight_decay_low, args.weight_decay_high)
        params["weight_decay"] = trial.suggest_float("weight_decay", wd_low, wd_high, log=True)
    if args.include_grad_clip:
        params["grad_clip_norm"] = trial.suggest_categorical("grad_clip_norm", [0.5, 1.0, 2.0])
    if args.include_head_hidden:
        d_hidden1 = trial.suggest_categorical("d_hidden1", [128, 256, 512])
        d_hidden2 = trial.suggest_categorical("d_hidden2", [64, 128, 256])
        params["d_hidden1"] = d_hidden1
        params["d_hidden2"] = min(d_hidden1, d_hidden2)
    return params


def command_for_fold(args: argparse.Namespace, params: dict[str, Any], split_file: Path, fold_dir: Path) -> list[str]:
    dataset_mode = args.dataset_mode
    if params["transfer_mode"] == "random_init":
        dataset_mode = "deg"
    return [
        args.python_exe,
        "-u",
        str(WORKER),
        "--pretrained-checkpoint", str(args.pretrained_checkpoint),
        "--transfer-mode", params["transfer_mode"],
        "--x-file", str(args.x_file),
        "--y-file", str(args.y_file),
        "--result-dir", str(fold_dir),
        "--seed", str(args.seed),
        "--split-file", str(split_file),
        "--dataset-mode", dataset_mode,
        "--max-genes", str(args.max_genes),
        "--gene-selection", args.gene_selection,
        "--scaler", args.scaler,
        "--scaler-fit-scope", args.scaler_fit_scope,
        "--batch-size", str(params["batch_size"]),
        "--epochs", str(args.epochs),
        "--early-stopping-patience", str(args.early_stopping_patience),
        "--lr-encoder", str(params["lr"]),
        "--lr-head", str(params["lr"]),
        "--weight-decay", str(params["weight_decay"]),
        "--label-smoothing", str(args.label_smoothing),
        "--class-weighting", args.class_weighting,
        "--checkpoint-metric", args.checkpoint_metric,
        "--evaluate-test", "off",
        "--grad-clip-norm", str(params["grad_clip_norm"]),
        "--device", args.device,
        "--n-layers", str(args.n_layers),
        "--n-heads", str(args.n_heads),
        "--d-model", str(args.d_model),
        "--d-ff", str(args.d_ff),
        "--dropout", str(params["dropout"]),
        "--aggfunc", args.aggfunc,
        "--d-hidden1", str(params["d_hidden1"]),
        "--d-hidden2", str(params["d_hidden2"]),
    ]


def run_worker(cmd: list[str], log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            cmd,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            handle.write(line)
        return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"Worker failed with exit code {return_code}. See log: {log_file}")


def read_val_metrics(metrics_path: Path) -> dict[str, float]:
    df = pd.read_csv(metrics_path)
    row = df.loc[df["split"] == "val"].iloc[0].to_dict()
    metrics = {key: float(value) for key, value in row.items() if key != "split"}
    val_auc = metrics.get("roc_auc_ovr_macro", metrics.get("roc_auc"))
    if val_auc is not None and "macro_f1" in metrics:
        metrics["auc_f1_50_50"] = 0.5 * float(val_auc) + 0.5 * float(metrics["macro_f1"])
        if "loss" in metrics:
            metrics["auc_f1_50_50_minus_025_loss"] = metrics["auc_f1_50_50"] - 0.25 * float(metrics["loss"])
    return metrics


def metric_key(objective_metric: str) -> str:
    if objective_metric == "val_auc_f1_50_50":
        return "auc_f1_50_50"
    if objective_metric == "val_auc_f1_50_50_minus_025_loss":
        return "auc_f1_50_50_minus_025_loss"
    return objective_metric.removeprefix("val_")


def main() -> None:
    args = parse_args()
    args.pretrained_checkpoint = resolve_path(args.pretrained_checkpoint)
    args.x_file = resolve_path(args.x_file)
    args.y_file = resolve_path(args.y_file)
    args.result_root = resolve_path(args.result_root)
    args.result_root.mkdir(parents=True, exist_ok=True)
    if not args.pretrained_checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.pretrained_checkpoint}")

    y_df = load_labels(args.y_file)
    split_paths = write_cv_splits(y_df, args.result_root / "cv_splits", args.n_folds, args.cv_seed)

    storage = args.storage.strip() or f"sqlite:///{(args.result_root / 'optuna_5cv.db').as_posix()}"
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        direction="maximize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=args.cv_seed, multivariate=True),
    )
    split_rows: list[dict[str, Any]] = []

    def objective(trial: optuna.Trial) -> float:
        params = sample_params(trial, args)
        fold_scores: list[float] = []
        for fold_idx, split_path in enumerate(split_paths, start=1):
            fold_dir = args.result_root / f"trial_{trial.number:04d}" / f"fold_{fold_idx:02d}"
            cmd = command_for_fold(args, params, split_path, fold_dir)
            print("\n" + "=" * 80, flush=True)
            print(f"Trial {trial.number} | fold={fold_idx}/{args.n_folds} | params={params}", flush=True)
            print("=" * 80, flush=True)
            run_worker(cmd, fold_dir / "finetune.log")
            val = read_val_metrics(fold_dir / "metrics_summary.csv")
            score = val[metric_key(args.objective_metric)]
            fold_scores.append(score)
            split_rows.append(
                {
                    "trial": trial.number,
                    "fold": fold_idx,
                    "objective_score": score,
                    "val_macro_f1": val.get("macro_f1"),
                    "val_auc": val.get("roc_auc_ovr_macro", val.get("roc_auc")),
                    "val_loss": val.get("loss"),
                    "val_recall_mci": val.get("recall_mci"),
                    "val_recall_ad": val.get("recall_ad"),
                    **params,
                }
            )
            pd.DataFrame(split_rows).to_csv(args.result_root / "optuna_cv_folds.csv", index=False)

        mean_score = float(sum(fold_scores) / max(len(fold_scores), 1))
        std_score = float(pd.Series(fold_scores).std(ddof=1)) if len(fold_scores) > 1 else 0.0
        trial.set_user_attr("cv_mean", mean_score)
        trial.set_user_attr("cv_std", std_score)
        study.trials_dataframe(attrs=("number", "value", "params", "user_attrs", "state")).to_csv(
            args.result_root / "optuna_trials.csv",
            index=False,
        )
        return mean_score

    study.optimize(objective, n_trials=args.n_trials, gc_after_trial=True)
    trials_df = study.trials_dataframe(attrs=("number", "value", "params", "user_attrs", "state"))
    trials_df.to_csv(args.result_root / "optuna_trials.csv", index=False)
    with (args.result_root / "best_trial.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "best_trial": study.best_trial.number,
                "best_value": study.best_value,
                "best_params": study.best_params,
                "objective_metric": args.objective_metric,
                "n_trials": args.n_trials,
                "n_folds": args.n_folds,
                "pretrained_checkpoint": str(args.pretrained_checkpoint),
            },
            handle,
            indent=2,
        )
    print("\nBest trial:", study.best_trial.number, flush=True)
    print("Best value:", study.best_value, flush=True)
    print("Best params:", study.best_params, flush=True)
    print(f"Optuna outputs: {args.result_root}", flush=True)


if __name__ == "__main__":
    main()
