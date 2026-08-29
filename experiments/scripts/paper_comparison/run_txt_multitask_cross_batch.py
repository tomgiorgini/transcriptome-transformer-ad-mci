#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[3]
WORKER = ROOT / "experiments" / "scripts" / "paper_comparison" / "train_txt_multitask.py"
DEFAULT_PAIRWISE_SHARED_DIR = ROOT / "task_dataset" / "processed" / "txt_pairwise_multitask" / "shared_ad_mci_ctl"
DEFAULT_GSE63060_METADATA = ROOT / "pretraining_dataset" / "geo_downloads" / "GSE63060" / "eset_1_GPL6947" / "GSE63060_sample_metadata.tsv.gz"
DEFAULT_GSE63061_METADATA = ROOT / "pretraining_dataset" / "geo_downloads" / "GSE63061" / "eset_1_GPL10558" / "GSE63061_sample_metadata.tsv.gz"

ARCHITECTURES = {
    "1l2h": {"n_layers": 1, "n_heads": 2},
    "1l4h": {"n_layers": 1, "n_heads": 4},
    "2l2h": {"n_layers": 2, "n_heads": 2},
    "2l4h": {"n_layers": 2, "n_heads": 4},
    "3l2h": {"n_layers": 3, "n_heads": 2},
    "3l4h": {"n_layers": 3, "n_heads": 4},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TxT multitask cross-batch GSE63060/GSE63061 evaluation.")
    parser.add_argument("--python-exe", default=sys.executable)
    parser.add_argument("--x-file", type=Path, default=DEFAULT_PAIRWISE_SHARED_DIR / "X.csv")
    parser.add_argument("--y-file", type=Path, default=DEFAULT_PAIRWISE_SHARED_DIR / "y.csv")
    parser.add_argument("--gse63060-metadata", type=Path, default=DEFAULT_GSE63060_METADATA)
    parser.add_argument("--gse63061-metadata", type=Path, default=DEFAULT_GSE63061_METADATA)
    parser.add_argument("--result-root", type=Path, default=ROOT / "results" / "paper_comparison" / "txt_multitask_cross_batch")
    parser.add_argument(
        "--model-configs",
        nargs="+",
        required=True,
        help="Config labels like 1l2h_d256_do0p3. d_ff defaults to 4*d_model and embed_dim=d_model.",
    )
    parser.add_argument("--directions", nargs="+", choices=["63060_to_63061", "63061_to_63060"], default=["63060_to_63061", "63061_to_63060"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[101, 102, 103, 104, 105, 106, 107, 108, 109, 110])
    parser.add_argument("--inner-val-ratio", type=float, default=0.15)
    parser.add_argument("--device", choices=["cuda", "cpu", "mps"], default="cuda")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--smoke", action="store_true")

    parser.add_argument("--max-genes", type=int, default=2000)
    parser.add_argument(
        "--gene-selection",
        choices=[
            "pairwise_anova_union",
            "ad_mci_priority_anova_union",
            "ad_mci_vs_ctl_anova_50_50",
            "task_deg_union",
            "variance",
            "mad",
        ],
        default="variance",
    )
    parser.add_argument("--ad-mci-gene-fraction", type=float, default=0.5)
    parser.add_argument("--scaler", choices=["none", "minmax", "standard"], default="minmax")
    parser.add_argument("--feature-selection-fit-scope", choices=["train", "all"], default="train")
    parser.add_argument("--scaler-fit-scope", choices=["train", "all"], default="train")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--train-sampling", choices=["random", "balanced_classes"], default="random")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--early-stopping-patience", type=int, default=80)
    parser.add_argument("--val-loss-stop-threshold", type=float, default=None)
    parser.add_argument("--val-loss-stop-patience", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-embedding", type=float, default=None)
    parser.add_argument("--freeze-embedding-epochs", type=int, default=0)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--d-hidden1", type=int, default=128)
    parser.add_argument("--d-hidden2", type=int, default=64)
    parser.add_argument("--task-specific-pooling", choices=["on", "off"], default="off")
    parser.add_argument("--head-norm", choices=["batch", "layer", "none"], default="batch")
    parser.add_argument("--mask-aware-heads", choices=["on", "off"], default="off")
    parser.add_argument("--encoder-sharing", choices=["shared", "separate"], default="shared")
    parser.add_argument(
        "--gradient-strategy",
        choices=["weighted_sum", "primary_protected_pcgrad"],
        default="weighted_sum",
    )
    parser.add_argument("--gradient-diagnostics", choices=["on", "off"], default="off")
    parser.add_argument("--gradient-diagnostic-interval", type=int, default=1)
    parser.add_argument("--class-weighting", choices=["on", "off"], default="off")
    parser.add_argument("--task-loss-weights", nargs=3, type=float, default=[1.0, 1.0, 1.0])
    parser.add_argument("--pooling-mode", choices=["average", "task_attention"], default="average")
    parser.add_argument("--attention-pooling-hidden-dim", type=int, default=16)
    parser.add_argument("--attention-pooling-dropout", type=float, default=0.1)
    parser.add_argument("--primary-adapter-dim", type=int, default=0)
    parser.add_argument(
        "--expression-residual", choices=["none", "additive_zero_init"], default="none"
    )
    parser.add_argument("--checkpoint-ensemble-size", type=int, default=1)
    parser.add_argument("--checkpoint-ensemble-min-gap", type=int, default=3)
    parser.add_argument(
        "--augmentation",
        choices=["none", "smote", "borderline_smote", "pca_neighbor_mci_ctl", "pca_neighbor_all_tasks", "ctgan", "gan"],
        default="none",
    )
    parser.add_argument("--smote-k-neighbors", type=int, default=5)
    parser.add_argument("--smote-m-neighbors", type=int, default=10)
    parser.add_argument("--smote-kind", choices=["borderline-1", "borderline-2"], default="borderline-1")
    parser.add_argument("--pca-neighbor-components", type=int, default=50)
    parser.add_argument("--pca-neighbor-k", type=int, default=5)
    parser.add_argument("--pca-neighbor-gap-fraction", type=float, default=0.5)
    parser.add_argument("--pca-neighbor-target-count", type=int, default=0)
    parser.add_argument("--ctgan-epochs", type=int, default=100)
    parser.add_argument("--ctgan-batch-size", type=int, default=128)
    parser.add_argument("--augmentation-target-multiplier", type=float, default=2.0)
    parser.add_argument("--gan-epochs", type=int, default=100)
    parser.add_argument("--gan-batch-size", type=int, default=64)
    parser.add_argument("--gan-latent-dim", type=int, default=64)
    parser.add_argument("--gan-learning-rate", type=float, default=0.001)
    parser.add_argument("--gan-target-multiplier", type=float, default=1.5)
    parser.add_argument("--gan-sampling-strategy", choices=["balanced", "minority", "proportional"], default="balanced")
    parser.add_argument("--embed-file", type=Path, default=None)
    parser.add_argument("--embedding-gene-policy", choices=["all", "mapped_only"], default="all")
    parser.add_argument("--candidate-gene-file", type=Path, default=None)
    parser.add_argument(
        "--embedding-rescale",
        choices=["none", "global_std", "per_dimension_std"],
        default="none",
    )
    parser.add_argument("--embedding-init-scale", type=float, default=0.02)
    parser.add_argument("--ppi-integration", choices=["direct", "gated_residual"], default="direct")
    parser.add_argument("--ppi-gate-init", type=float, default=0.0)
    parser.add_argument(
        "--checkpoint-metric",
        choices=[
            "val_multitask_auc_mean",
            "val_multitask_auc_minus_025_loss",
            "val_multitask_auc_f1_minus_loss",
            "val_multitask_macro_f1_mean",
            "val_primary_auc_mean",
            "val_primary_auc_minus_025_loss",
            "val_weighted_70_15_15_auc_minus_025_loss",
            "val_loss",
        ],
        default="val_multitask_auc_f1_minus_loss",
    )
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else (ROOT / path).resolve()


def load_labels(y_file: Path) -> pd.DataFrame:
    y_df = pd.read_csv(y_file)
    if "sample_id" not in y_df.columns or "label" not in y_df.columns:
        raise ValueError(f"{y_file} must contain sample_id and label columns.")
    y_df = y_df[["sample_id", "label"]].copy()
    y_df["sample_id"] = y_df["sample_id"].astype(str).str.strip()
    return y_df


def read_geo_metadata_sample_ids(path: Path) -> set[str]:
    md = pd.read_csv(path, sep="\t", compression="infer", dtype=str)
    for candidate in ("sample_id", "geo_accession"):
        if candidate in md.columns:
            return set(md[candidate].astype(str).str.strip())
    raise ValueError(f"{path} must contain sample_id or geo_accession.")


def infer_batch_labels(sample_ids: pd.Series, gse63060_metadata: Path, gse63061_metadata: Path) -> pd.Series:
    gse63060_ids = read_geo_metadata_sample_ids(gse63060_metadata)
    gse63061_ids = read_geo_metadata_sample_ids(gse63061_metadata)
    labels: list[str] = []
    clean_sample_ids = sample_ids.astype(str).str.strip()
    for sample_id in clean_sample_ids:
        gsm_id = sample_id.split("_", 1)[0].strip()
        if gsm_id in gse63060_ids:
            labels.append("GSE63060")
        elif gsm_id in gse63061_ids:
            labels.append("GSE63061")
        else:
            labels.append("unknown")
    batch = pd.Series(labels, index=sample_ids.index, name="batch")
    unknown = batch[batch == "unknown"]
    if not unknown.empty:
        preview = ", ".join(clean_sample_ids.loc[unknown.index].head(10))
        raise ValueError(f"Could not map {len(unknown)} samples to batches. Examples: {preview}")
    return batch


def parse_model_config(config: str) -> dict[str, Any]:
    match = re.fullmatch(r"(?P<arch>[123]l[24]h)_d(?P<d_model>\d+)_do(?P<dropout>\d+p\d+|\d+)", config)
    if match is None:
        raise ValueError(f"Invalid --model-configs entry '{config}'. Expected for example: 1l2h_d256_do0p3.")
    arch_key = match.group("arch")
    if arch_key not in ARCHITECTURES:
        raise ValueError(f"Unsupported architecture in '{config}'.")
    d_model = int(match.group("d_model"))
    dropout = float(match.group("dropout").replace("p", "."))
    arch = ARCHITECTURES[arch_key]
    return {
        "label": config,
        "arch_key": arch_key,
        "n_layers": arch["n_layers"],
        "n_heads": arch["n_heads"],
        "d_model": d_model,
        "embed_dim": d_model,
        "d_ff": 4 * d_model,
        "dropout": dropout,
    }


def split_rows(sample_ids: pd.Series, train_idx: np.ndarray, val_idx: np.ndarray, test_idx: np.ndarray) -> pd.DataFrame:
    rows = (
        [{"sample_id": sample_ids.iloc[int(idx)], "split": "train"} for idx in train_idx]
        + [{"sample_id": sample_ids.iloc[int(idx)], "split": "val"} for idx in val_idx]
        + [{"sample_id": sample_ids.iloc[int(idx)], "split": "test"} for idx in test_idx]
    )
    return pd.DataFrame(rows)


def write_cross_batch_splits(args: argparse.Namespace, y_df: pd.DataFrame, batch: pd.Series) -> list[dict[str, Any]]:
    output_dir = args.result_root / "splits"
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_ids = y_df["sample_id"].astype(str)
    labels = y_df["label"].to_numpy()
    all_idx = np.arange(len(y_df))
    split_specs: list[dict[str, Any]] = []
    direction_map = {
        "63060_to_63061": ("GSE63060", "GSE63061"),
        "63061_to_63060": ("GSE63061", "GSE63060"),
    }
    for direction in args.directions:
        train_batch, test_batch = direction_map[direction]
        pool_idx = all_idx[batch.to_numpy() == train_batch]
        test_idx = all_idx[batch.to_numpy() == test_batch]
        for seed in args.seeds:
            train_idx, val_idx = train_test_split(
                pool_idx,
                test_size=args.inner_val_ratio,
                random_state=seed,
                stratify=labels[pool_idx],
            )
            split_df = split_rows(sample_ids, train_idx, val_idx, test_idx)
            split_path = output_dir / f"{direction}_seed_{seed}.csv"
            split_df.to_csv(split_path, index=False)
            split_specs.append(
                {
                    "direction": direction,
                    "train_batch": train_batch,
                    "test_batch": test_batch,
                    "seed": seed,
                    "split_path": split_path,
                    "n_train": int(len(train_idx)),
                    "n_val": int(len(val_idx)),
                    "n_test": int(len(test_idx)),
                }
            )
    pd.DataFrame(split_specs).to_csv(output_dir / "cross_batch_split_manifest.csv", index=False)
    return split_specs


def build_worker_command(args: argparse.Namespace, config: dict[str, Any], split_path: Path, result_dir: Path, seed: int) -> list[str]:
    cmd = [
        args.python_exe,
        "-u",
        str(WORKER),
        "--x-file", str(args.x_file),
        "--y-file", str(args.y_file),
        "--split-mode", "custom",
        "--split-file", str(split_path),
        "--result-dir", str(result_dir),
        "--seed", str(seed),
        "--max-genes", str(args.max_genes),
        "--gene-selection", args.gene_selection,
        "--ad-mci-gene-fraction", str(args.ad_mci_gene_fraction),
        "--scaler", args.scaler,
        "--feature-selection-fit-scope", args.feature_selection_fit_scope,
        "--scaler-fit-scope", args.scaler_fit_scope,
        "--batch-size", str(args.batch_size),
        "--train-sampling", args.train_sampling,
        "--epochs", str(args.epochs),
        "--early-stopping-patience", str(args.early_stopping_patience),
        "--val-loss-stop-patience", str(args.val_loss_stop_patience),
        "--lr-encoder", str(args.lr),
        "--lr-head", str(args.lr),
        "--freeze-embedding-epochs", str(args.freeze_embedding_epochs),
        "--weight-decay", str(args.weight_decay),
        "--class-weighting", args.class_weighting,
        "--task-loss-weights", *[str(value) for value in args.task_loss_weights],
        "--pooling-mode", args.pooling_mode,
        "--attention-pooling-hidden-dim", str(args.attention_pooling_hidden_dim),
        "--attention-pooling-dropout", str(args.attention_pooling_dropout),
        "--primary-adapter-dim", str(args.primary_adapter_dim),
        "--expression-residual", args.expression_residual,
        "--checkpoint-ensemble-size", str(args.checkpoint_ensemble_size),
        "--checkpoint-ensemble-min-gap", str(args.checkpoint_ensemble_min_gap),
        "--augmentation", args.augmentation,
        "--smote-k-neighbors", str(args.smote_k_neighbors),
        "--smote-m-neighbors", str(args.smote_m_neighbors),
        "--smote-kind", args.smote_kind,
        "--pca-neighbor-components", str(args.pca_neighbor_components),
        "--pca-neighbor-k", str(args.pca_neighbor_k),
        "--pca-neighbor-gap-fraction", str(args.pca_neighbor_gap_fraction),
        "--pca-neighbor-target-count", str(args.pca_neighbor_target_count),
        "--ctgan-epochs", str(args.ctgan_epochs),
        "--ctgan-batch-size", str(args.ctgan_batch_size),
        "--augmentation-target-multiplier", str(args.augmentation_target_multiplier),
        "--gan-epochs", str(args.gan_epochs),
        "--gan-batch-size", str(args.gan_batch_size),
        "--gan-latent-dim", str(args.gan_latent_dim),
        "--gan-learning-rate", str(args.gan_learning_rate),
        "--gan-target-multiplier", str(args.gan_target_multiplier),
        "--gan-sampling-strategy", args.gan_sampling_strategy,
        "--checkpoint-metric", args.checkpoint_metric,
        "--evaluate-test", "on",
        "--device", args.device,
        "--n-layers", str(config["n_layers"]),
        "--n-heads", str(config["n_heads"]),
        "--d-model", str(config["d_model"]),
        "--embed-dim", str(config["embed_dim"]),
        "--d-ff", str(config["d_ff"]),
        "--dropout", str(config["dropout"]),
        "--aggfunc", "Avgpool",
        "--task-specific-pooling", args.task_specific_pooling,
        "--head-norm", args.head_norm,
        "--mask-aware-heads", args.mask_aware_heads,
        "--encoder-sharing", args.encoder_sharing,
        "--gradient-strategy", args.gradient_strategy,
        "--gradient-diagnostics", args.gradient_diagnostics,
        "--gradient-diagnostic-interval", str(args.gradient_diagnostic_interval),
        "--d-hidden1", str(args.d_hidden1),
        "--d-hidden2", str(args.d_hidden2),
        "--grad-clip-norm", "1.0",
        "--embedding-gene-policy", args.embedding_gene_policy,
        "--embedding-rescale", args.embedding_rescale,
        "--embedding-init-scale", str(args.embedding_init_scale),
        "--ppi-integration", args.ppi_integration,
        "--ppi-gate-init", str(args.ppi_gate_init),
    ]
    if args.lr_embedding is not None:
        cmd.extend(["--lr-embedding", str(args.lr_embedding)])
    if args.embed_file is not None:
        cmd.extend(["--embed-file", str(args.embed_file)])
    if args.candidate_gene_file is not None:
        cmd.extend(["--candidate-gene-file", str(args.candidate_gene_file)])
    if args.val_loss_stop_threshold is not None:
        cmd.extend(["--val-loss-stop-threshold", str(args.val_loss_stop_threshold)])
    if args.max_train_batches is not None:
        cmd.extend(["--max-train-batches", str(args.max_train_batches)])
    if args.max_val_batches is not None:
        cmd.extend(["--max-val-batches", str(args.max_val_batches)])
    return cmd


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


def print_repeat_metric_summary(run_dir: Path) -> None:
    metrics_path = run_dir / "metrics_summary.csv"
    if not metrics_path.exists():
        print(f"Metrics summary not found yet: {metrics_path}", flush=True)
        return
    metrics = pd.read_csv(metrics_path)
    metrics = metrics[metrics["split"].isin(["val", "test"])].copy()
    if metrics.empty:
        print(f"No val/test rows in: {metrics_path}", flush=True)
        return
    display_cols = ["split", "task", "samples", "loss", "accuracy", "macro_f1", "balanced_accuracy", "roc_auc"]
    available = [col for col in display_cols if col in metrics.columns]
    print("\nRepeat metric summary (val/test):", flush=True)
    print(
        metrics[available].sort_values(["split", "task"]).to_string(index=False, float_format=lambda value: f"{value:.4f}"),
        flush=True,
    )


def collect_metrics(result_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for metrics_path in sorted((result_root / "runs").glob("*/*/seed_*/metrics_summary.csv")):
        direction = metrics_path.parents[2].name
        model_config = metrics_path.parents[1].name
        seed_text = metrics_path.parent.name.replace("seed_", "")
        metrics = pd.read_csv(metrics_path)
        for _, row in metrics.iterrows():
            payload = row.to_dict()
            payload["direction"] = direction
            payload["model_config"] = model_config
            payload["seed"] = int(seed_text)
            payload["run_dir"] = str(metrics_path.parent)
            rows.append(payload)
    return pd.DataFrame(rows)


def summarize_metrics(runs: pd.DataFrame) -> pd.DataFrame:
    if runs.empty:
        return runs
    numeric_cols = [
        "samples",
        "loss",
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "balanced_accuracy",
        "roc_auc",
        "roc_auc_ovr_macro",
    ]
    available = [col for col in numeric_cols if col in runs.columns]
    grouped = runs.groupby(["direction", "model_config", "split", "task"], as_index=False)
    summary = grouped[available].agg(["mean", "std"])
    summary.columns = [
        "_".join([part for part in col if part]).rstrip("_") if isinstance(col, tuple) else col for col in summary.columns
    ]
    counts = grouped.size().rename(columns={"size": "n_runs"})
    return counts.merge(summary, on=["direction", "model_config", "split", "task"], how="left")


def apply_smoke_overrides(args: argparse.Namespace) -> None:
    if not args.smoke:
        return
    args.result_root = args.result_root / "smoke"
    args.device = "cpu"
    args.epochs = 1
    args.gan_epochs = min(args.gan_epochs, 1)
    args.early_stopping_patience = 1
    args.seeds = args.seeds[:1]
    args.directions = args.directions[:1]
    args.model_configs = args.model_configs[:1]
    args.max_train_batches = 2 if args.max_train_batches is None else args.max_train_batches
    args.max_val_batches = 2 if args.max_val_batches is None else args.max_val_batches


def main() -> None:
    args = parse_args()
    args.x_file = resolve_path(args.x_file)
    args.y_file = resolve_path(args.y_file)
    args.gse63060_metadata = resolve_path(args.gse63060_metadata)
    args.gse63061_metadata = resolve_path(args.gse63061_metadata)
    args.result_root = resolve_path(args.result_root)
    args.embed_file = resolve_path(args.embed_file) if args.embed_file is not None else None
    args.candidate_gene_file = resolve_path(args.candidate_gene_file) if args.candidate_gene_file is not None else None
    apply_smoke_overrides(args)
    args.result_root.mkdir(parents=True, exist_ok=True)
    if not WORKER.exists():
        raise FileNotFoundError(f"Worker not found: {WORKER}")

    y_df = load_labels(args.y_file)
    batch = infer_batch_labels(y_df["sample_id"], args.gse63060_metadata, args.gse63061_metadata)
    configs = [parse_model_config(config) for config in args.model_configs]
    for config in configs:
        candidate_tag = "_candfile" if args.candidate_gene_file is not None else ""
        loss_tag = "-".join(str(value).replace(".", "p") for value in args.task_loss_weights)
        pooling_tag = "avg" if args.pooling_mode == "average" else f"att{args.attention_pooling_hidden_dim}"
        expression_tag = "none" if args.expression_residual == "none" else "add"
        config["label"] = (
            f"{config['label']}_enc{args.encoder_sharing}_hn{args.head_norm}_er{args.embedding_rescale}{candidate_tag}_"
            f"lw{loss_tag}_pool{pooling_tag}_adp{args.primary_adapter_dim}_expr{expression_tag}_ens{args.checkpoint_ensemble_size}"
        )
    split_specs = write_cross_batch_splits(args, y_df, batch)

    config_payload = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config_payload["parsed_model_configs"] = configs
    config_payload["batch_counts"] = batch.value_counts().sort_index().to_dict()
    with (args.result_root / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config_payload, handle, indent=2, sort_keys=True)
        handle.write("\n")

    for config in configs:
        for split_spec in split_specs:
            run_dir = args.result_root / "runs" / split_spec["direction"] / config["label"] / f"seed_{split_spec['seed']}"
            if args.skip_existing and (run_dir / "metrics_summary.csv").exists():
                print(f"Skipping existing cross-batch run: {run_dir}", flush=True)
                continue
            print("\n" + "=" * 80, flush=True)
            print(
                "TxT multitask cross-batch | "
                f"model={config['label']} | direction={split_spec['direction']} | seed={split_spec['seed']}",
                flush=True,
            )
            print(
                f"train={split_spec['train_batch']} n={split_spec['n_train']} | "
                f"val={split_spec['train_batch']} n={split_spec['n_val']} | "
                f"test={split_spec['test_batch']} n={split_spec['n_test']}",
                flush=True,
            )
            print("=" * 80, flush=True)
            cmd = build_worker_command(args, config, split_spec["split_path"], run_dir, seed=split_spec["seed"])
            run_command(cmd, run_dir / "train.log")
            print_repeat_metric_summary(run_dir)

    runs = collect_metrics(args.result_root)
    runs.to_csv(args.result_root / "cross_batch_runs.csv", index=False)
    summarize_metrics(runs).to_csv(args.result_root / "cross_batch_summary.csv", index=False)
    print(f"TxT multitask cross-batch complete. Outputs: {args.result_root}", flush=True)


if __name__ == "__main__":
    main()
