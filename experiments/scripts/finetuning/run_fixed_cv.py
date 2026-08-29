#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold


ROOT = Path(__file__).resolve().parents[3]
WORKER = ROOT / "experiments" / "scripts" / "finetuning" / "finetune_txt.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fixed-parameter TxT k-fold CV.")
    parser.add_argument("--pretrained-checkpoint", type=Path, required=True)
    parser.add_argument("--x-file", type=Path, default=ROOT / "task_dataset" / "processed" / "ad_mci_deg" / "X_deg_ad_mci.csv")
    parser.add_argument("--y-file", type=Path, default=ROOT / "task_dataset" / "processed" / "ad_mci_deg" / "y_ad_mci.csv")
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--python-exe", default=sys.executable)
    parser.add_argument("--device", choices=["cuda", "cpu", "mps"], default="cuda")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--cv-seed", type=int, default=42)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--transfer-mode", choices=["random_init", "embedding_only", "full"], default="random_init")
    parser.add_argument("--dataset-mode", choices=["deg", "deg_pretrained_overlap"], default="deg")
    parser.add_argument("--max-genes", type=int, default=0)
    parser.add_argument("--gene-selection", choices=["variance", "class_aware_variance"], default="variance")
    parser.add_argument("--scaler", choices=["none", "minmax", "standard"], default="minmax")
    parser.add_argument("--scaler-fit-scope", choices=["train", "all"], default="train")
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--weight-decay", type=float, required=True)
    parser.add_argument("--dropout", type=float, required=True)
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--early-stopping-patience", type=int, default=45)
    parser.add_argument("--class-weighting", choices=["on", "off"], default="off")
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--checkpoint-metric", choices=["val_macro_f1", "val_roc_auc_ovr_macro", "val_auc_f1_50_50", "val_loss"], default="val_macro_f1")
    parser.add_argument("--threshold-tuning", choices=["on", "off"], default="off")
    parser.add_argument("--final-threshold-mode", choices=["fixed", "tuned"], default="fixed")
    parser.add_argument("--final-threshold", type=float, default=0.5)
    parser.add_argument("--n-layers", type=int, default=1)
    parser.add_argument("--n-heads", type=int, default=2)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--d-ff", type=int, default=1024)
    parser.add_argument("--aggfunc", choices=["Avgpool", "Flatten"], default="Avgpool")
    parser.add_argument("--d-hidden1", type=int, default=256)
    parser.add_argument("--d-hidden2", type=int, default=128)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else (ROOT / path).resolve()


def load_labels(y_file: Path) -> pd.DataFrame:
    y_df = pd.read_csv(y_file)
    label_col = "label" if "label" in y_df.columns else "label_name"
    if "sample_id" not in y_df.columns or label_col not in y_df.columns:
        raise ValueError(f"{y_file} must contain sample_id and label/label_name.")
    return y_df[["sample_id", label_col]].rename(columns={label_col: "label"})


def write_splits(y_df: pd.DataFrame, output_dir: Path, n_folds: int, seed: int) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_ids = y_df["sample_id"].astype(str).to_numpy()
    labels = y_df["label"].astype(str).to_numpy()
    splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    paths: list[Path] = []
    for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(sample_ids, labels), start=1):
        split_df = pd.DataFrame(
            [{"sample_id": sample_ids[idx], "split": "train"} for idx in train_idx]
            + [{"sample_id": sample_ids[idx], "split": "val"} for idx in val_idx]
        )
        path = output_dir / f"fold_{fold_idx:02d}.csv"
        split_df.to_csv(path, index=False)
        paths.append(path)
    return paths


def run_command(cmd: list[str], log_file: Path) -> None:
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
    args.pretrained_checkpoint = resolve_path(args.pretrained_checkpoint)
    args.x_file = resolve_path(args.x_file)
    args.y_file = resolve_path(args.y_file)
    args.result_root = resolve_path(args.result_root)
    args.result_root.mkdir(parents=True, exist_ok=True)

    split_paths = write_splits(load_labels(args.y_file), args.result_root / "cv_splits", args.n_folds, args.cv_seed)
    rows = []
    for fold_idx, split_path in enumerate(split_paths, start=1):
        fold_dir = args.result_root / f"fold_{fold_idx:02d}"
        if args.skip_existing and (fold_dir / "metrics_summary.csv").exists():
            print(f"Skipping existing fold: {fold_dir}", flush=True)
        else:
            dataset_mode = "deg" if args.transfer_mode == "random_init" else args.dataset_mode
            cmd = [
                args.python_exe,
                "-u",
                str(WORKER),
                "--pretrained-checkpoint", str(args.pretrained_checkpoint),
                "--transfer-mode", args.transfer_mode,
                "--x-file", str(args.x_file),
                "--y-file", str(args.y_file),
                "--result-dir", str(fold_dir),
                "--seed", str(args.seed),
                "--split-file", str(split_path),
                "--dataset-mode", dataset_mode,
                "--max-genes", str(args.max_genes),
                "--gene-selection", args.gene_selection,
                "--scaler", args.scaler,
                "--scaler-fit-scope", args.scaler_fit_scope,
                "--batch-size", str(args.batch_size),
                "--epochs", str(args.epochs),
                "--early-stopping-patience", str(args.early_stopping_patience),
                "--lr-encoder", str(args.lr),
                "--lr-head", str(args.lr),
                "--weight-decay", str(args.weight_decay),
                "--label-smoothing", str(args.label_smoothing),
                "--class-weighting", args.class_weighting,
                "--checkpoint-metric", args.checkpoint_metric,
                "--threshold-tuning", args.threshold_tuning,
                "--final-threshold-mode", args.final_threshold_mode,
                "--final-threshold", str(args.final_threshold),
                "--evaluate-test", "off",
                "--grad-clip-norm", str(args.grad_clip_norm),
                "--device", args.device,
                "--n-layers", str(args.n_layers),
                "--n-heads", str(args.n_heads),
                "--d-model", str(args.d_model),
                "--d-ff", str(args.d_ff),
                "--dropout", str(args.dropout),
                "--aggfunc", args.aggfunc,
                "--d-hidden1", str(args.d_hidden1),
                "--d-hidden2", str(args.d_hidden2),
            ]
            print("\n" + "=" * 80, flush=True)
            print(f"Fixed CV fold {fold_idx}/{args.n_folds}: {fold_dir}", flush=True)
            print("=" * 80, flush=True)
            run_command(cmd, fold_dir / "finetune.log")

        metrics = pd.read_csv(fold_dir / "metrics_summary.csv")
        for _, row in metrics.iterrows():
            payload = row.to_dict()
            payload["fold"] = fold_idx
            payload["run_dir"] = str(fold_dir)
            rows.append(payload)

    runs = pd.DataFrame(rows)
    runs.to_csv(args.result_root / "fixed_cv_runs.csv", index=False)
    aggregations = {
        "folds": ("fold", "nunique"),
        "macro_f1_mean": ("macro_f1", "mean"),
        "macro_f1_std": ("macro_f1", "std"),
        "auc_mean": ("roc_auc_ovr_macro", "mean"),
        "auc_std": ("roc_auc_ovr_macro", "std"),
        "accuracy_mean": ("accuracy", "mean"),
        "balanced_accuracy_mean": ("balanced_accuracy", "mean"),
    }
    for column in runs.columns:
        if column.startswith("recall_"):
            aggregations[f"{column}_mean"] = (column, "mean")
    summary = runs.groupby("split", as_index=False).agg(**aggregations)
    summary.to_csv(args.result_root / "fixed_cv_summary.csv", index=False)
    config = vars(args).copy()
    config = {key: str(value) if isinstance(value, Path) else value for key, value in config.items()}
    with (args.result_root / "fixed_cv_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
    print(summary.to_string(index=False), flush=True)
    print(f"Fixed CV outputs: {args.result_root}", flush=True)


if __name__ == "__main__":
    main()
