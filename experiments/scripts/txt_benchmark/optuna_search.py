#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import optuna
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[3]
RUNNER = Path(__file__).resolve().with_name("run_experiments.py")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optuna architecture search for the TxT benchmark.")
    parser.add_argument("--x-file", type=Path, default=ROOT / "task_dataset" / "processed" / "ad_mci_binary" / "X_ad_mci.csv")
    parser.add_argument("--y-file", type=Path, default=ROOT / "task_dataset" / "processed" / "ad_mci_binary" / "y_ad_mci.csv")
    parser.add_argument("--result-root", type=Path, default=ROOT / "results" / "TxT" / "optuna")
    parser.add_argument("--study-name", default="txt_architecture_search")
    parser.add_argument("--storage", type=Path, default=None)
    parser.add_argument("--n-trials", type=int, default=192)
    parser.add_argument("--sampler", choices=["grid", "tpe"], default="grid")
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--split-protocol", choices=["stratified_holdout", "batch_holdout"], default="stratified_holdout")
    parser.add_argument("--train-size", type=float, default=0.7)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--batch-scenarios", nargs="+", default=["shared_test"], choices=["shared_test", "test_gse63060", "test_gse63061"])
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--gene-source", choices=["top_variance", "top_mad", "input", "nature2020"], default="top_variance")
    parser.add_argument("--top-variance-genes", type=int, default=1000)
    parser.add_argument("--top-mad-genes", type=int, default=1000)
    parser.add_argument("--scaler", choices=["minmax", "standard", "robust", "none"], default="minmax")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--early-stopping-patience", type=int, default=70)
    parser.add_argument("--device", choices=["cuda", "cpu", "mps"], default="cuda")
    parser.add_argument("--aggfunc", choices=["Flatten", "Avgpool", "search"], default="search")
    parser.add_argument("--d-model-choices", type=int, nargs="+", default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--smoke", action="store_true", help="Run one tiny trial to validate the plumbing.")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else (ROOT / path).resolve()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def load_binary_dataset(x_file: Path, y_file: Path) -> tuple[pd.DataFrame, pd.Series]:
    x = pd.read_csv(x_file, index_col=0)
    y_df = pd.read_csv(y_file, index_col=0)
    if "label" not in y_df.columns:
        raise ValueError(f"{y_file} must contain a label column.")
    common = x.index.intersection(y_df.index)
    x = x.loc[common].copy()
    y = y_df.loc[common, "label"].astype(int)
    if set(y.unique()) != {0, 1}:
        raise ValueError(f"Expected binary labels 0/1, got {sorted(y.unique())}")
    return x, y


def stratified_train_indices(y: pd.Series, train_size: float, val_size: float, test_size: float, seed: int) -> np.ndarray:
    total = train_size + val_size + test_size
    if not np.isclose(total, 1.0):
        raise ValueError(f"train_size + val_size + test_size must be 1.0, got {total}")
    y_values = y.to_numpy(dtype=np.int64)
    all_idx = np.arange(len(y_values))
    train_val_idx, _ = train_test_split(
        all_idx,
        test_size=test_size,
        random_state=seed,
        stratify=y_values,
    )
    val_fraction_of_train_val = val_size / (train_size + val_size)
    train_idx, _ = train_test_split(
        np.asarray(train_val_idx, dtype=np.int64),
        test_size=val_fraction_of_train_val,
        random_state=seed,
        stratify=y_values[train_val_idx],
    )
    return np.asarray(train_idx, dtype=np.int64)


def median_absolute_deviation(frame: pd.DataFrame) -> pd.Series:
    median = frame.median(axis=0)
    return frame.sub(median, axis=1).abs().median(axis=0)


def prepare_fixed_gene_input(args: argparse.Namespace) -> tuple[Path, str, dict[str, Any]]:
    if args.split_protocol != "stratified_holdout" or args.gene_source not in {"top_variance", "top_mad"}:
        return args.x_file, args.gene_source, {
            "fixed_gene_set": False,
            "reason": "fixed top-k selection is only applied for stratified_holdout with top_variance/top_mad",
        }
    x, y = load_binary_dataset(args.x_file, args.y_file)
    train_idx = stratified_train_indices(y, args.train_size, args.val_size, args.test_size, args.seed)
    x_train = x.iloc[train_idx]
    if args.gene_source == "top_variance":
        scores = x_train.var(axis=0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        top_k = args.top_variance_genes
        score_name = "variance"
    else:
        scores = median_absolute_deviation(x_train).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        top_k = args.top_mad_genes
        score_name = "mad"
    selected_genes = scores.sort_values(ascending=False).head(min(top_k, x.shape[1])).index.astype(str).tolist()
    selection_dir = args.result_root / f"fixed_top{len(selected_genes)}_{score_name}_seed{args.seed}"
    selection_dir.mkdir(parents=True, exist_ok=True)
    x_subset_file = selection_dir / f"X_top{len(selected_genes)}_{score_name}_seed{args.seed}.csv"
    score_file = selection_dir / "gene_scores.csv"
    (selection_dir / "selected_genes.txt").write_text("\n".join(selected_genes) + "\n", encoding="utf-8")
    scores.loc[selected_genes].rename(score_name).to_csv(score_file, header=True)
    x.loc[:, selected_genes].to_csv(x_subset_file)
    manifest = {
        "fixed_gene_set": True,
        "selection_scope": "train_split_only",
        "split_protocol": args.split_protocol,
        "seed": args.seed,
        "train_size": args.train_size,
        "val_size": args.val_size,
        "test_size": args.test_size,
        "source_x_file": str(args.x_file),
        "gene_source": args.gene_source,
        "score": score_name,
        "requested_top_k": int(top_k),
        "n_selected_genes": len(selected_genes),
        "selected_genes_file": str(selection_dir / "selected_genes.txt"),
        "x_subset_file": str(x_subset_file),
        "score_file": str(score_file),
    }
    write_json(selection_dir / "feature_selection_manifest.json", manifest)
    return x_subset_file, "input", manifest


def best_validation_score(trial_root: Path) -> float:
    logs = sorted(trial_root.rglob("training_log.csv"))
    if not logs:
        raise RuntimeError(f"No training_log.csv found under {trial_root}")
    scores: list[float] = []
    for log_path in logs:
        history = pd.read_csv(log_path)
        if history.empty:
            continue
        if "val_checkpoint_score" in history.columns:
            series = pd.to_numeric(history["val_checkpoint_score"], errors="coerce")
        else:
            required = {"val_roc_auc", "val_macro_f1", "val_loss"}
            missing = required - set(history.columns)
            if missing:
                raise RuntimeError(f"{log_path} is missing columns: {sorted(missing)}")
            series = (
                0.5 * pd.to_numeric(history["val_roc_auc"], errors="coerce")
                + 0.5 * pd.to_numeric(history["val_macro_f1"], errors="coerce")
                - 0.25 * pd.to_numeric(history["val_loss"], errors="coerce")
            )
        finite = series.dropna()
        if not finite.empty:
            scores.append(float(finite.max()))
    if not scores:
        raise RuntimeError(f"No finite validation checkpoint scores found under {trial_root}")
    return max(scores)


BASE_SEARCH_SPACE = {
    "n_heads": [2, 4],
    "n_layers": [1, 2],
    "d_model": [16, 32, 64, 128],
    "dropout": [0.2, 0.3, 0.4, 0.5],
    "batch_size": [16, 32],
}


def search_space(args: argparse.Namespace) -> dict[str, list[Any]]:
    space = dict(BASE_SEARCH_SPACE)
    if args.d_model_choices is not None:
        space["d_model"] = [int(value) for value in args.d_model_choices]
    if args.aggfunc == "search":
        space["aggfunc"] = ["Flatten", "Avgpool"]
    return space


def run_trial(args: argparse.Namespace, trial: optuna.Trial) -> float:
    space = search_space(args)
    n_heads = trial.suggest_categorical("n_heads", space["n_heads"])
    n_layers = trial.suggest_categorical("n_layers", space["n_layers"])
    d_model = trial.suggest_categorical("d_model", space["d_model"])
    dropout = trial.suggest_categorical("dropout", space["dropout"])
    batch_size = trial.suggest_categorical("batch_size", space["batch_size"])
    aggfunc = args.aggfunc if args.aggfunc != "search" else trial.suggest_categorical("aggfunc", space["aggfunc"])
    d_ff = 4 * int(d_model)

    trial_root = args.result_root / "trials" / f"trial_{trial.number:03d}"
    if trial_root.exists():
        shutil.rmtree(trial_root)
    trial_root.mkdir(parents=True, exist_ok=True)

    params = {
        "n_heads": n_heads,
        "n_layers": n_layers,
        "d_model": d_model,
        "embed_dim": d_model,
        "d_ff": d_ff,
        "dropout": dropout,
        "batch_size": batch_size,
        "aggfunc": aggfunc,
        "lr": 1e-4,
        "weight_decay": 1e-4,
    }
    write_json(trial_root / "trial_params.json", params)

    command = [
        args.python,
        str(RUNNER),
        "--x-file",
        str(args.x_file),
        "--y-file",
        str(args.y_file),
        "--result-root",
        str(trial_root),
        "--gene-source",
        args.runner_gene_source,
        "--top-variance-genes",
        str(args.top_variance_genes),
        "--top-mad-genes",
        str(args.top_mad_genes),
        "--split-protocol",
        args.split_protocol,
        "--train-size",
        str(args.train_size),
        "--val-size",
        str(args.val_size),
        "--test-size",
        str(args.test_size),
        "--batch-scenarios",
        *args.batch_scenarios,
        "--repeats",
        str(args.repeats),
        "--scaler",
        args.scaler,
        "--checkpoint-metric",
        "strict_v2",
        "--threshold-mode",
        "fixed_0_5",
        "--device",
        args.device,
        "--epochs",
        str(args.epochs),
        "--early-stopping-patience",
        str(args.early_stopping_patience),
        "--seed",
        str(args.seed),
        "--lr",
        "1e-4",
        "--weight-decay",
        "1e-4",
        "--n-heads",
        str(n_heads),
        "--n-layers",
        str(n_layers),
        "--d-model",
        str(d_model),
        "--embed-dim",
        str(d_model),
        "--d-ff",
        str(d_ff),
        "--dropout",
        str(dropout),
        "--batch-size",
        str(batch_size),
        "--aggfunc",
        aggfunc,
    ]
    if args.max_train_batches is not None:
        command.extend(["--max-train-batches", str(args.max_train_batches)])
    if args.max_val_batches is not None:
        command.extend(["--max-val-batches", str(args.max_val_batches)])

    log_path = trial_root / "runner.log"
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    print("", flush=True)
    print(f"===== TxT Optuna trial {trial.number:03d} =====", flush=True)
    print(f"Params: {params}", flush=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            env=env,
        )
        if process.stdout is None:
            raise RuntimeError("Subprocess stdout was not captured.")
        for line in process.stdout:
            print(line, end="", flush=True)
            log_file.write(line)
            log_file.flush()
        returncode = process.wait()
    print(f"===== End TxT Optuna trial {trial.number:03d} | exit={returncode} =====", flush=True)
    write_json(trial_root / "command.json", {"command": command, "returncode": returncode})
    if returncode != 0:
        raise RuntimeError(f"Trial {trial.number} failed with exit code {returncode}. See {log_path}")

    score = best_validation_score(trial_root)
    trial.set_user_attr("trial_root", str(trial_root))
    trial.set_user_attr("d_ff", d_ff)
    return score


def main() -> None:
    args = parse_args()
    args.x_file = resolve(args.x_file)
    args.y_file = resolve(args.y_file)
    args.result_root = resolve(args.result_root)
    if args.storage is None:
        args.storage = args.result_root / "optuna.db"
    else:
        args.storage = resolve(args.storage)
    if args.smoke:
        args.n_trials = 1
        args.epochs = min(args.epochs, 2)
        args.early_stopping_patience = min(args.early_stopping_patience, 1)
        args.max_train_batches = 2
        args.max_val_batches = 1
    args.result_root.mkdir(parents=True, exist_ok=True)
    fixed_x_file, runner_gene_source, fixed_gene_manifest = prepare_fixed_gene_input(args)
    args.x_file = fixed_x_file
    args.runner_gene_source = runner_gene_source
    write_json(args.result_root / "optuna_args.json", vars(args))
    write_json(args.result_root / "fixed_gene_selection_manifest.json", fixed_gene_manifest)

    if args.sampler == "grid":
        sampler = optuna.samplers.GridSampler(search_space(args), seed=args.seed)
    else:
        sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(
        study_name=args.study_name,
        direction="maximize",
        sampler=sampler,
        storage=f"sqlite:///{args.storage.as_posix()}",
        load_if_exists=True,
    )
    study.optimize(lambda trial: run_trial(args, trial), n_trials=args.n_trials, timeout=args.timeout)

    trials_df = study.trials_dataframe(attrs=("number", "value", "state", "params", "user_attrs"))
    trials_df.to_csv(args.result_root / "study_trials.csv", index=False)
    best_params = dict(study.best_trial.params)
    best_params["embed_dim"] = best_params["d_model"]
    best_params["d_ff"] = 4 * int(best_params["d_model"])
    best_params["lr"] = 1e-4
    best_params["weight_decay"] = 1e-4
    best_payload = {
        "best_value": study.best_value,
        "best_trial": study.best_trial.number,
        "best_params": best_params,
        "objective": "max val_checkpoint_score = 0.5*val_roc_auc + 0.5*val_macro_f1 - 0.25*val_loss",
        "fixed_gene_selection": fixed_gene_manifest,
    }
    write_json(args.result_root / "best_params.json", best_payload)

    final_command = [
        "python",
        "experiments\\scripts\\txt_benchmark\\run_experiments.py",
        "--x-file",
        str(args.x_file),
        "--y-file",
        str(args.y_file),
        "--repeats",
        str(args.repeats),
        "--split-protocol",
        args.split_protocol,
        "--train-size",
        str(args.train_size),
        "--val-size",
        str(args.val_size),
        "--test-size",
        str(args.test_size),
        "--batch-scenarios",
        *args.batch_scenarios,
        "--gene-source",
        args.runner_gene_source,
        "--top-variance-genes",
        str(args.top_variance_genes),
        "--top-mad-genes",
        str(args.top_mad_genes),
        "--scaler",
        args.scaler,
        "--checkpoint-metric",
        "strict_v2",
        "--threshold-mode",
        "fixed_0_5",
        "--lr",
        "1e-4",
        "--weight-decay",
        "1e-4",
        "--seed",
        str(args.seed),
        "--n-heads",
        str(best_params["n_heads"]),
        "--n-layers",
        str(best_params["n_layers"]),
        "--d-model",
        str(best_params["d_model"]),
        "--embed-dim",
        str(best_params["embed_dim"]),
        "--d-ff",
        str(best_params["d_ff"]),
        "--dropout",
        str(best_params["dropout"]),
        "--batch-size",
        str(best_params["batch_size"]),
        "--aggfunc",
        str(best_params["aggfunc"]),
        "--result-root",
        "results\\TxT\\benchmark",
        "--overwrite-results",
    ]
    (args.result_root / "run_best_full.ps1").write_text(" ".join(final_command) + "\n", encoding="utf-8")
    print(f"Best value: {study.best_value:.6f}")
    print(f"Best params: {best_params}")
    print(f"Saved: {args.result_root / 'best_params.json'}")


if __name__ == "__main__":
    main()
