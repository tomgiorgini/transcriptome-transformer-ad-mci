#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[3]
WORKER = ROOT / "experiments" / "scripts" / "finetuning" / "finetune_txt.py"
DEFAULT_X_FILE = ROOT / "task_dataset" / "processed" / "ad_mci_deg" / "X_deg_ad_mci.csv"
DEFAULT_Y_FILE = ROOT / "task_dataset" / "processed" / "ad_mci_deg" / "y_ad_mci.csv"
DEFAULT_CHECKPOINT = (
    ROOT
    / "results"
    / "pretraining"
    / "self_supervised"
    / "txt_gexbert"
    / "baseline_1l2h_d256_dff1024_dropout04_deg_with_reference_500ep_mask25"
    / "best_checkpoint.pt"
)
DEFAULT_RESULT_ROOT = ROOT / "results" / "paper_comparison" / "sota_ad_mci_788"


ARCHITECTURES = [
    {"model": "txt_1l2h_d128_ff128", "n_layers": 1, "n_heads": 2},
    {"model": "txt_2l2h_d128_ff128", "n_layers": 2, "n_heads": 2},
]


LITERATURE_ROWS = [
    {
        "method": "Multiple Feature Selection + SVM",
        "paper": "Kalkan et al., Biomolecules 2023, Table 5",
        "dataset": "GSE63060 + GSE63061 + GSE140829",
        "feature_selection": "paper-specific multiple feature selection",
        "protocol": "reported 5-fold cross-validation",
        "reported_auc": 0.58,
        "reported_accuracy": 0.61,
        "reported_f1": 0.48,
        "comparability_status": "not_directly_comparable",
        "comparability_note": "Uses a different combined dataset and unpublished folds; values are reported literature only.",
    },
    {
        "method": "LASSO + SVM",
        "paper": "Kalkan et al., Biomolecules 2023, Table 5",
        "dataset": "GSE63060 + GSE63061 + GSE140829",
        "feature_selection": "LASSO",
        "protocol": "reported 5-fold cross-validation",
        "reported_auc": 0.63,
        "reported_accuracy": 0.64,
        "reported_f1": 0.53,
        "comparability_status": "not_directly_comparable",
        "comparability_note": "Uses a different combined dataset and unpublished folds; values are reported literature only.",
    },
    {
        "method": "DeepInsight (tSNE + CNN)",
        "paper": "Kalkan et al., Biomolecules 2023, Table 5",
        "dataset": "GSE63060 + GSE63061 + GSE140829",
        "feature_selection": "paper-specific image mapping",
        "protocol": "reported 5-fold cross-validation",
        "reported_auc": 0.60,
        "reported_accuracy": 0.58,
        "reported_f1": 0.50,
        "comparability_status": "not_directly_comparable",
        "comparability_note": "Uses a different combined dataset and unpublished folds; values are reported literature only.",
    },
    {
        "method": "LDA-based imaging + CNN",
        "paper": "Kalkan et al., Biomolecules 2023, Table 5",
        "dataset": "GSE63060 + GSE63061 + GSE140829",
        "feature_selection": "paper-specific LDA image mapping",
        "protocol": "reported 5-fold cross-validation",
        "reported_auc": 0.62,
        "reported_accuracy": 0.65,
        "reported_f1": 0.52,
        "comparability_status": "not_directly_comparable",
        "comparability_note": "Uses a different combined dataset and unpublished folds; values are reported literature only.",
    },
    {
        "method": "One2MFusion",
        "paper": "Kalkan et al., Biomolecules 2023, Table 5",
        "dataset": "GSE63060 + GSE63061 + GSE140829",
        "feature_selection": "LASSO, 492 genes for AD vs MCI",
        "protocol": "reported 5-fold cross-validation",
        "reported_auc": 0.88,
        "reported_accuracy": 0.79,
        "reported_f1": 0.74,
        "comparability_status": "not_directly_comparable",
        "comparability_note": "Uses a different combined dataset, 492 LASSO genes, and unpublished folds; values are reported literature only.",
    },
    {
        "method": "Kalkan et al. cited in Sarma et al.",
        "paper": "Sarma et al., Discover Applied Sciences 2025, literature table",
        "dataset": "GSE63060 + GSE63061 + GSE140829",
        "feature_selection": "LASSO regression",
        "protocol": "reported from prior study",
        "reported_auc": 0.664,
        "reported_accuracy": None,
        "reported_f1": None,
        "comparability_status": "not_directly_comparable",
        "comparability_note": "Reported as MCI vs AD literature value, not a locally reproduced identical-test-set result.",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TxT AD vs MCI SOTA comparison with fixed 788 DEG input.")
    parser.add_argument("--python-exe", default=sys.executable)
    parser.add_argument("--pretrained-checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--x-file", type=Path, default=DEFAULT_X_FILE)
    parser.add_argument("--y-file", type=Path, default=DEFAULT_Y_FILE)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--device", choices=["cuda", "cpu", "mps"], default="cuda")
    parser.add_argument("--protocol", choices=["all", "protocol_matched", "paper_exact"], default="protocol_matched")
    parser.add_argument("--paper-exact-split-dir", type=Path, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Run one fold, one seed, one epoch on CPU unless overridden.")
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--early-stopping-patience", type=int, default=50)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--cv-seed", type=int, default=42)
    parser.add_argument("--seeds", type=int, nargs="+", default=[101, 102, 103, 104, 105, 106, 107, 108, 109, 110])
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else (ROOT / path).resolve()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def validate_fixed_deg_dataset(x_file: Path, y_file: Path) -> pd.DataFrame:
    x_df = pd.read_csv(x_file)
    y_df = pd.read_csv(y_file)
    if "sample_id" not in x_df.columns:
        raise ValueError(f"{x_file} must contain sample_id.")
    if "sample_id" not in y_df.columns or "label" not in y_df.columns:
        raise ValueError(f"{y_file} must contain sample_id and label.")
    gene_count = x_df.shape[1] - 1
    if gene_count != 788:
        raise ValueError(f"Expected exactly 788 DEG columns in {x_file}, found {gene_count}.")
    if len(x_df) != 473:
        raise ValueError(f"Expected exactly 473 AD/MCI samples in {x_file}, found {len(x_df)}.")
    if set(x_df["sample_id"].astype(str)) != set(y_df["sample_id"].astype(str)):
        raise ValueError("X and y sample_id sets do not match.")
    label_counts = y_df["label"].value_counts().sort_index().to_dict()
    if len(label_counts) != 2:
        raise ValueError(f"Expected binary AD/MCI labels, found labels: {label_counts}.")
    return y_df[["sample_id", "label"]].copy()


def write_cv_splits(y_df: pd.DataFrame, output_dir: Path, n_folds: int, seed: int) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_ids = y_df["sample_id"].astype(str).to_numpy()
    labels = y_df["label"].to_numpy()
    if n_folds == 1:
        train_idx, val_idx = train_test_split(
            range(len(sample_ids)),
            test_size=0.2,
            random_state=seed,
            stratify=labels,
        )
        split_df = pd.DataFrame(
            [{"sample_id": sample_ids[idx], "split": "train"} for idx in train_idx]
            + [{"sample_id": sample_ids[idx], "split": "val"} for idx in val_idx]
        )
        path = output_dir / "fold_01.csv"
        split_df.to_csv(path, index=False)
        return [path]
    splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    split_paths: list[Path] = []
    for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(sample_ids, labels), start=1):
        split_df = pd.DataFrame(
            [{"sample_id": sample_ids[idx], "split": "train"} for idx in train_idx]
            + [{"sample_id": sample_ids[idx], "split": "val"} for idx in val_idx]
        )
        path = output_dir / f"fold_{fold_idx:02d}.csv"
        split_df.to_csv(path, index=False)
        split_paths.append(path)
    return split_paths


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
        raise RuntimeError(f"Command failed with exit code {return_code}. See {log_file}")


def base_worker_command(args: argparse.Namespace, arch: dict[str, Any], run_dir: Path) -> list[str]:
    cmd = [
        args.python_exe,
        "-u",
        str(WORKER),
        "--pretrained-checkpoint", str(args.pretrained_checkpoint),
        "--transfer-mode", "random_init",
        "--x-file", str(args.x_file),
        "--y-file", str(args.y_file),
        "--result-dir", str(run_dir),
        "--dataset-mode", "deg",
        "--max-genes", "0",
        "--scaler", "minmax",
        "--scaler-fit-scope", "train",
        "--batch-size", "8",
        "--epochs", str(args.epochs),
        "--early-stopping-patience", str(args.early_stopping_patience),
        "--lr-encoder", "0.0001",
        "--lr-head", "0.0001",
        "--weight-decay", "0.0001",
        "--label-smoothing", "0.0",
        "--class-weighting", "off",
        "--checkpoint-metric", "val_roc_auc_ovr_macro",
        "--threshold-tuning", "off",
        "--final-threshold-mode", "fixed",
        "--final-threshold", "0.5",
        "--grad-clip-norm", "1.0",
        "--device", args.device,
        "--n-layers", str(arch["n_layers"]),
        "--n-heads", str(arch["n_heads"]),
        "--d-model", "128",
        "--d-ff", "128",
        "--dropout", "0.5",
        "--aggfunc", "Avgpool",
        "--d-hidden1", "128",
        "--d-hidden2", "64",
    ]
    if args.max_train_batches is not None:
        cmd += ["--max-train-batches", str(args.max_train_batches)]
    if args.max_val_batches is not None:
        cmd += ["--max-val-batches", str(args.max_val_batches)]
    return cmd


def normalize_artifact_names(run_dir: Path) -> None:
    test_predictions = run_dir / "test_predictions.csv"
    test_confusion = run_dir / "test_confusion_matrix.csv"
    if test_predictions.exists():
        shutil.copyfile(test_predictions, run_dir / "predictions.csv")
    if test_confusion.exists():
        shutil.copyfile(test_confusion, run_dir / "confusion_matrix.csv")


def load_training_summary(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "model_summary.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle).get("training_summary", {})


def collect_run_metrics(run_dir: Path, extra: dict[str, Any]) -> list[dict[str, Any]]:
    metrics_path = run_dir / "metrics_summary.csv"
    if not metrics_path.exists():
        return []
    training_summary = load_training_summary(run_dir)
    rows = []
    metrics = pd.read_csv(metrics_path)
    for _, row in metrics.iterrows():
        payload = {
            **extra,
            "split": row["split"],
            "samples": int(row["samples"]),
            "loss": row.get("loss"),
            "accuracy": row.get("accuracy"),
            "macro_f1": row.get("macro_f1"),
            "weighted_f1": row.get("weighted_f1"),
            "balanced_accuracy": row.get("balanced_accuracy"),
            "auc": row.get("roc_auc_ovr_macro", row.get("roc_auc")),
            "best_epoch": training_summary.get("best_epoch"),
            "checkpoint_metric": training_summary.get("checkpoint_metric"),
            "best_checkpoint_value": training_summary.get("best_checkpoint_value"),
            "run_dir": str(run_dir),
        }
        rows.append(payload)
    return rows


def run_protocol_matched(args: argparse.Namespace, y_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    cv_folds = 1 if args.smoke else args.cv_folds
    seeds = args.seeds[:1] if args.smoke else args.seeds
    split_paths = write_cv_splits(y_df, args.result_root / "splits" / "cv5", cv_folds, args.cv_seed)

    for arch in ARCHITECTURES:
        for fold_idx, split_path in enumerate(split_paths, start=1):
            run_dir = args.result_root / "protocol_matched" / "cv5_validation" / arch["model"] / f"fold_{fold_idx:02d}"
            if not (args.skip_existing and (run_dir / "metrics_summary.csv").exists()):
                cmd = base_worker_command(args, arch, run_dir)
                cmd += [
                    "--seed", str(args.cv_seed),
                    "--split-file", str(split_path),
                    "--evaluate-test", "off",
                ]
                print(f"\nRunning {arch['model']} CV fold {fold_idx}/{cv_folds}: {run_dir}", flush=True)
                run_command(cmd, run_dir / "finetune.log")
            rows.extend(
                collect_run_metrics(
                    run_dir,
                    {
                        "source": "ours",
                        "protocol": "protocol_matched_cv5_validation",
                        "comparability_status": "same_dataset_protocol_only",
                        "model": arch["model"],
                        "fold": fold_idx,
                        "seed": args.cv_seed,
                    },
                )
            )

        for seed in seeds:
            run_dir = args.result_root / "protocol_matched" / "seeds10_70_15_15" / arch["model"] / f"seed_{seed}"
            if not (args.skip_existing and (run_dir / "metrics_summary.csv").exists()):
                cmd = base_worker_command(args, arch, run_dir)
                cmd += [
                    "--seed", str(seed),
                    "--split-seed", str(seed),
                    "--split-mode", "stratified",
                    "--val-ratio", "0.15",
                    "--test-ratio", "0.15",
                    "--evaluate-test", "on",
                ]
                print(f"\nRunning {arch['model']} 70/15/15 seed {seed}: {run_dir}", flush=True)
                run_command(cmd, run_dir / "finetune.log")
                normalize_artifact_names(run_dir)
            rows.extend(
                collect_run_metrics(
                    run_dir,
                    {
                        "source": "ours",
                        "protocol": "protocol_matched_70_15_15",
                        "comparability_status": "same_dataset_protocol_only",
                        "model": arch["model"],
                        "fold": None,
                        "seed": seed,
                    },
                )
            )
    return pd.DataFrame(rows)


def run_paper_exact(args: argparse.Namespace) -> pd.DataFrame:
    if args.paper_exact_split_dir is None:
        audit = pd.DataFrame(
            [
                {
                    "protocol": "paper_exact",
                    "status": "not_run",
                    "reason": "No paper exact split directory was provided; the inspected papers do not publish fold/test sample IDs.",
                }
            ]
        )
        return audit
    split_dir = resolve_path(args.paper_exact_split_dir)
    split_paths = sorted(split_dir.glob("*.csv"))
    if not split_paths:
        raise ValueError(f"No CSV split files found in {split_dir}.")

    rows: list[dict[str, Any]] = []
    for arch in ARCHITECTURES:
        for fold_idx, split_path in enumerate(split_paths, start=1):
            run_dir = args.result_root / "paper_exact" / arch["model"] / f"split_{fold_idx:02d}"
            if not (args.skip_existing and (run_dir / "metrics_summary.csv").exists()):
                cmd = base_worker_command(args, arch, run_dir)
                cmd += [
                    "--seed", str(args.cv_seed),
                    "--split-file", str(split_path),
                    "--evaluate-test", "on",
                ]
                print(f"\nRunning {arch['model']} paper-exact split {fold_idx}: {run_dir}", flush=True)
                run_command(cmd, run_dir / "finetune.log")
                normalize_artifact_names(run_dir)
            rows.extend(
                collect_run_metrics(
                    run_dir,
                    {
                        "source": "ours",
                        "protocol": "paper_exact",
                        "comparability_status": "direct_same_test_set",
                        "model": arch["model"],
                        "fold": fold_idx,
                        "seed": args.cv_seed,
                    },
                )
            )
    return pd.DataFrame(rows)


def summarize_our_runs(our_runs: pd.DataFrame) -> pd.DataFrame:
    if our_runs.empty or "source" not in our_runs.columns:
        return pd.DataFrame()
    numeric_cols = ["auc", "accuracy", "macro_f1", "weighted_f1", "balanced_accuracy", "loss"]
    for col in numeric_cols:
        if col in our_runs.columns:
            our_runs[col] = pd.to_numeric(our_runs[col], errors="coerce")
    summary = (
        our_runs.groupby(["source", "protocol", "comparability_status", "model", "split"], dropna=False)
        .agg(
            n_runs=("run_dir", "nunique"),
            mean_auc=("auc", "mean"),
            std_auc=("auc", "std"),
            mean_accuracy=("accuracy", "mean"),
            std_accuracy=("accuracy", "std"),
            mean_macro_f1=("macro_f1", "mean"),
            std_macro_f1=("macro_f1", "std"),
            mean_balanced_accuracy=("balanced_accuracy", "mean"),
            std_balanced_accuracy=("balanced_accuracy", "std"),
        )
        .reset_index()
    )
    return summary


def build_comparison_table(our_summary: pd.DataFrame, literature: pd.DataFrame) -> pd.DataFrame:
    lit_rows = literature.rename(
        columns={
            "reported_auc": "mean_auc",
            "reported_accuracy": "mean_accuracy",
            "reported_f1": "mean_macro_f1",
        }
    ).copy()
    lit_rows["source"] = "literature_reported"
    lit_rows["model"] = lit_rows["method"]
    lit_rows["split"] = "reported"
    lit_rows["n_runs"] = None
    for col in ["std_auc", "std_accuracy", "std_macro_f1", "mean_balanced_accuracy", "std_balanced_accuracy"]:
        lit_rows[col] = None
    lit_rows = lit_rows[
        [
            "source",
            "protocol",
            "comparability_status",
            "model",
            "split",
            "n_runs",
            "mean_auc",
            "std_auc",
            "mean_accuracy",
            "std_accuracy",
            "mean_macro_f1",
            "std_macro_f1",
            "mean_balanced_accuracy",
            "std_balanced_accuracy",
            "paper",
            "dataset",
            "feature_selection",
            "comparability_note",
        ]
    ]
    if our_summary.empty:
        return lit_rows
    ours = our_summary.copy()
    for col in ["paper", "dataset", "feature_selection", "comparability_note"]:
        ours[col] = {
            "paper": "This work",
            "dataset": "task_dataset/processed/ad_mci_deg, 473 samples x 788 DEG",
            "feature_selection": "Fixed top 788 DEG already materialized; no per-run selection",
            "comparability_note": "Run locally on fixed 788 DEG; directly comparable only to methods run on identical split manifests.",
        }[col]
    frames = [frame for frame in [ours[lit_rows.columns], lit_rows] if not frame.empty]
    return pd.concat(frames, ignore_index=True)


def write_comparability_audit(args: argparse.Namespace, y_df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "item": "fixed_input_dataset",
                "status": "pass",
                "details": f"{len(y_df)} samples, 788 genes, binary AD/MCI labels.",
            },
            {
                "item": "gene_selection",
                "status": "pass",
                "details": "No runtime gene selection; --max-genes 0 and hard check for 788 input genes.",
            },
            {
                "item": "scaler",
                "status": "pass",
                "details": "All worker runs use --scaler minmax --scaler-fit-scope train.",
            },
            {
                "item": "paper_exact_splits",
                "status": "missing" if args.paper_exact_split_dir is None else "provided",
                "details": "The inspected papers do not publish fold/test sample IDs; provide --paper-exact-split-dir to run exact splits if obtained.",
            },
            {
                "item": "literature_values",
                "status": "reported_only",
                "details": "Values are copied from paper tables and are not marked as direct comparisons without identical split IDs.",
            },
        ]
    )


def main() -> None:
    args = parse_args()
    args.pretrained_checkpoint = resolve_path(args.pretrained_checkpoint)
    args.x_file = resolve_path(args.x_file)
    args.y_file = resolve_path(args.y_file)
    args.result_root = resolve_path(args.result_root)
    if args.paper_exact_split_dir is not None:
        args.paper_exact_split_dir = resolve_path(args.paper_exact_split_dir)
    if args.smoke:
        args.device = "cpu"
        args.epochs = min(args.epochs, 1)
        args.early_stopping_patience = min(args.early_stopping_patience, 1)
        args.max_train_batches = 1 if args.max_train_batches is None else args.max_train_batches
        args.max_val_batches = 1 if args.max_val_batches is None else args.max_val_batches

    args.result_root.mkdir(parents=True, exist_ok=True)
    if not args.pretrained_checkpoint.exists():
        raise FileNotFoundError(f"Pretrained checkpoint placeholder not found: {args.pretrained_checkpoint}")
    y_df = validate_fixed_deg_dataset(args.x_file, args.y_file)
    write_json(args.result_root / "args.json", {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})

    run_frames: list[pd.DataFrame] = []
    if args.protocol in {"all", "protocol_matched"}:
        run_frames.append(run_protocol_matched(args, y_df))
    if args.protocol in {"all", "paper_exact"}:
        paper_exact = run_paper_exact(args)
        if "source" in paper_exact.columns:
            run_frames.append(paper_exact)
        else:
            paper_exact.to_csv(args.result_root / "paper_exact_status.csv", index=False)

    our_runs = pd.concat([frame for frame in run_frames if not frame.empty], ignore_index=True) if run_frames else pd.DataFrame()
    our_runs.to_csv(args.result_root / "our_runs.csv", index=False)
    our_summary = summarize_our_runs(our_runs)
    our_summary.to_csv(args.result_root / "our_summary.csv", index=False)

    literature = pd.DataFrame(LITERATURE_ROWS)
    literature.to_csv(args.result_root / "literature_reported_values.csv", index=False)
    audit = write_comparability_audit(args, y_df)
    audit.to_csv(args.result_root / "comparability_audit.csv", index=False)

    comparison = build_comparison_table(our_summary, literature)
    comparison.to_csv(args.result_root / "sota_comparison_table.csv", index=False)

    print("\nSOTA comparison outputs:", args.result_root, flush=True)
    if not our_summary.empty:
        print("\nOur summary:", flush=True)
        print(our_summary.to_string(index=False), flush=True)
    print("\nComparability audit:", flush=True)
    print(audit.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
