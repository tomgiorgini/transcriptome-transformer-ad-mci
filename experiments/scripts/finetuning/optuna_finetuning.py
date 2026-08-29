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


ROOT = Path(__file__).resolve().parents[3]
WORKER = ROOT / "experiments" / "scripts" / "finetuning" / "finetune_txt.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optuna tuning for TxT DEG-overlap fine-tuning.")
    parser.add_argument("--pretrained-checkpoint", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--study-name", type=str, default="txt_deg_finetuning_optuna")
    parser.add_argument("--storage", type=str, default="")
    parser.add_argument("--n-trials", type=int, default=60)
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--tune-split-seeds", type=int, nargs="+", default=[101])
    parser.add_argument("--transfer-mode", choices=["full", "embedding_only", "random_init"], default="full")
    parser.add_argument("--transfer-modes", choices=["full", "embedding_only", "random_init"], nargs="+", default=None)
    parser.add_argument("--dataset-mode", choices=["deg", "deg_pretrained_overlap"], default="deg_pretrained_overlap")
    parser.add_argument("--split-mode", choices=["stratified", "random"], default="stratified")
    parser.add_argument("--scaler", choices=["none", "minmax", "standard"], default="minmax")
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--early-stopping-patience", type=int, default=70)
    parser.add_argument("--device", choices=["cuda", "cpu", "mps"], default="cuda")
    parser.add_argument("--python-exe", type=str, default=sys.executable)
    parser.add_argument("--objective-metric", choices=["val_loss", "val_macro_f1", "val_roc_auc_ovr_macro"], default="val_macro_f1")
    parser.add_argument("--checkpoint-metric", choices=["val_loss", "val_macro_f1", "val_roc_auc_ovr_macro"], default="val_loss")
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--fixed-n-heads", type=int, default=None)
    parser.add_argument("--fixed-n-layers", type=int, default=None)
    parser.add_argument("--fixed-d-model", type=int, default=None)
    parser.add_argument("--fixed-d-ff", type=int, default=None)
    parser.add_argument("--freeze-encoder-layers", type=str, default="")
    parser.add_argument("--freeze-encoder-layer-options", type=str, nargs="+", default=None)
    parser.add_argument("--freeze-embeddings", action="store_true")
    parser.add_argument("--freeze-tupe", action="store_true")
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    return parser.parse_args()


def read_split_metrics(path: Path, split: str) -> dict[str, float]:
    rows = pd.read_csv(path)
    row = rows.loc[rows["split"] == split]
    if row.empty:
        raise ValueError(f"Missing split {split!r} in {path}")
    return {key: float(value) for key, value in row.iloc[0].to_dict().items() if key not in {"split"}}


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sample_params(trial: optuna.Trial, args: argparse.Namespace) -> dict[str, Any]:
    d_hidden1 = trial.suggest_categorical("d_hidden1", [64, 128, 256])
    d_hidden2 = trial.suggest_categorical("d_hidden2", [32, 64, 128])
    if d_hidden2 > d_hidden1:
        d_hidden2 = d_hidden1
    params = {
        "batch_size": trial.suggest_categorical("batch_size", [8, 16, 32]),
        "lr_encoder": trial.suggest_float("lr_encoder", 1e-6, 5e-5, log=True),
        "lr_head": trial.suggest_float("lr_head", 5e-5, 1e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        "grad_clip_norm": trial.suggest_categorical("grad_clip_norm", [0.5, 1.0, 2.0]),
        "dropout": trial.suggest_categorical("dropout", [0.10, 0.20, 0.30, 0.40]),
        "d_hidden1": d_hidden1,
        "d_hidden2": d_hidden2,
    }
    if args.transfer_modes:
        params["transfer_mode"] = trial.suggest_categorical("transfer_mode", args.transfer_modes)
    if args.freeze_encoder_layer_options:
        params["freeze_encoder_layers"] = trial.suggest_categorical(
            "freeze_encoder_layers",
            args.freeze_encoder_layer_options,
        )
    return params


def command_for_trial(args: argparse.Namespace, params: dict[str, Any], split_seed: int, split_dir: Path) -> list[str]:
    cmd = [
        args.python_exe,
        "-u",
        str(WORKER),
        "--pretrained-checkpoint",
        str(args.pretrained_checkpoint),
        "--transfer-mode",
        str(params.get("transfer_mode", args.transfer_mode)),
        "--result-dir",
        str(split_dir),
        "--split-mode",
        args.split_mode,
        "--split-seed",
        str(split_seed),
        "--dataset-mode",
        args.dataset_mode,
        "--scaler",
        args.scaler,
        "--epochs",
        str(args.epochs),
        "--early-stopping-patience",
        str(args.early_stopping_patience),
        "--batch-size",
        str(params["batch_size"]),
        "--lr-encoder",
        str(params["lr_encoder"]),
        "--lr-head",
        str(params["lr_head"]),
        "--weight-decay",
        str(params["weight_decay"]),
        "--label-smoothing",
        str(args.label_smoothing),
        "--checkpoint-metric",
        args.checkpoint_metric,
        "--grad-clip-norm",
        str(params["grad_clip_norm"]),
        "--dropout",
        str(params["dropout"]),
        "--d-hidden1",
        str(params["d_hidden1"]),
        "--d-hidden2",
        str(params["d_hidden2"]),
        "--device",
        args.device,
    ]
    if args.fixed_n_heads is not None:
        cmd += ["--n-heads", str(args.fixed_n_heads)]
    if args.fixed_n_layers is not None:
        cmd += ["--n-layers", str(args.fixed_n_layers)]
    if args.fixed_d_model is not None:
        cmd += ["--d-model", str(args.fixed_d_model)]
    if args.fixed_d_ff is not None:
        cmd += ["--d-ff", str(args.fixed_d_ff)]
    freeze_encoder_layers = str(params.get("freeze_encoder_layers", args.freeze_encoder_layers))
    if freeze_encoder_layers:
        cmd += ["--freeze-encoder-layers", freeze_encoder_layers]
    if args.freeze_embeddings:
        cmd += ["--freeze-embeddings"]
    if args.freeze_tupe:
        cmd += ["--freeze-tupe"]
    if args.max_train_batches is not None:
        cmd += ["--max-train-batches", str(args.max_train_batches)]
    if args.max_val_batches is not None:
        cmd += ["--max-val-batches", str(args.max_val_batches)]
    return cmd


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


def main() -> None:
    args = parse_args()
    args.pretrained_checkpoint = (ROOT / args.pretrained_checkpoint).resolve() if not args.pretrained_checkpoint.is_absolute() else args.pretrained_checkpoint
    args.result_root = (ROOT / args.result_root).resolve() if not args.result_root.is_absolute() else args.result_root
    args.result_root.mkdir(parents=True, exist_ok=True)
    if not args.pretrained_checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.pretrained_checkpoint}")

    storage = args.storage.strip() or f"sqlite:///{(args.result_root / 'optuna_study.db').as_posix()}"
    sampler = optuna.samplers.TPESampler(seed=42, multivariate=True)
    direction = "minimize" if args.objective_metric == "val_loss" else "maximize"
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        direction=direction,
        load_if_exists=True,
        sampler=sampler,
    )

    trial_rows: list[dict[str, Any]] = []

    def objective(trial: optuna.Trial) -> float:
        params = sample_params(trial, args)
        split_scores: list[float] = []
        split_rows: list[dict[str, Any]] = []
        for split_seed in args.tune_split_seeds:
            split_dir = args.result_root / f"trial_{trial.number:04d}" / f"split_seed_{split_seed}"
            cmd = command_for_trial(args, params, split_seed, split_dir)
            print("\n" + "=" * 80, flush=True)
            print(f"Trial {trial.number} | split_seed={split_seed} | params={params}", flush=True)
            print("=" * 80, flush=True)
            run_worker(cmd, split_dir / "finetune.log")

            metrics_path = split_dir / "metrics_summary.csv"
            summary_path = split_dir / "model_summary.json"
            val_metrics = read_split_metrics(metrics_path, "val")
            test_metrics = read_split_metrics(metrics_path, "test")
            model_summary = read_json(summary_path)
            training_summary = model_summary.get("training_summary", {})
            score = val_metrics[args.objective_metric.removeprefix("val_")]
            split_scores.append(score)
            split_rows.append(
                {
                    "trial": trial.number,
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

        mean_score = float(np_mean(split_scores))
        trial.set_user_attr("mean_objective_score", mean_score)
        trial.set_user_attr("tune_split_seeds", ",".join(str(seed) for seed in args.tune_split_seeds))
        trial_rows.extend(split_rows)
        pd.DataFrame(trial_rows).to_csv(args.result_root / "optuna_trial_splits.csv", index=False)
        study.trials_dataframe(attrs=("number", "value", "params", "user_attrs", "state")).to_csv(
            args.result_root / "optuna_trials.csv",
            index=False,
        )
        return mean_score

    study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout, gc_after_trial=True)
    study.trials_dataframe(attrs=("number", "value", "params", "user_attrs", "state")).to_csv(
        args.result_root / "optuna_trials.csv",
        index=False,
    )
    with (args.result_root / "best_trial.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "best_trial": study.best_trial.number,
                "best_value": study.best_value,
                "best_params": study.best_params,
                "objective_metric": args.objective_metric,
                "tune_split_seeds": args.tune_split_seeds,
                "pretrained_checkpoint": str(args.pretrained_checkpoint),
            },
            handle,
            indent=2,
        )
    print("\nBest trial:", study.best_trial.number, flush=True)
    print("Best value:", study.best_value, flush=True)
    print("Best params:", study.best_params, flush=True)
    print(f"Optuna outputs: {args.result_root}", flush=True)


def np_mean(values: list[float]) -> float:
    if not values:
        return math.nan
    return float(sum(values) / len(values))


if __name__ == "__main__":
    main()
