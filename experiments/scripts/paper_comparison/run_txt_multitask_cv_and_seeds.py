#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


ROOT = Path(__file__).resolve().parents[3]
WORKER = ROOT / "experiments" / "scripts" / "paper_comparison" / "train_txt_multitask.py"
DEFAULT_PAIRWISE_SHARED_DIR = ROOT / "task_dataset" / "processed" / "txt_pairwise_multitask" / "shared_ad_mci_ctl"
DEFAULT_PPI_EDGE_FILE = ROOT / "pretraining_dataset" / "ppi_networks" / "hippie_highconf_edges.csv"


ARCHITECTURES = {
    "1l2h": {
        "n_layers": 1,
        "n_heads": 2,
    },
    "1l4h": {
        "n_layers": 1,
        "n_heads": 4,
    },
    "2l2h": {
        "n_layers": 2,
        "n_heads": 2,
    },
    "2l4h": {
        "n_layers": 2,
        "n_heads": 4,
    },
    "3l2h": {
        "n_layers": 3,
        "n_heads": 2,
    },
    "3l4h": {
        "n_layers": 3,
        "n_heads": 4,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TxT multitask CV or 10-repeat 70/10/20 evaluations.")
    parser.add_argument("--python-exe", default=sys.executable)
    parser.add_argument("--x-file", type=Path, default=DEFAULT_PAIRWISE_SHARED_DIR / "X.csv")
    parser.add_argument("--y-file", type=Path, default=DEFAULT_PAIRWISE_SHARED_DIR / "y.csv")
    parser.add_argument("--result-root", type=Path, default=ROOT / "results" / "paper_comparison" / "txt_multitask_eval")
    parser.add_argument("--run-mode", choices=["both", "cv", "seeds"], default="seeds")
    parser.add_argument("--architectures", nargs="+", choices=sorted(ARCHITECTURES), default=sorted(ARCHITECTURES))
    parser.add_argument("--device", choices=["cuda", "cpu", "mps"], default="cuda")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--post-model-construction-reseed",
        choices=["on", "off"],
        default="off",
    )
    parser.add_argument(
        "--model-variant",
        choices=["baseline", "ppi_volumetric"],
        default="baseline",
    )
    parser.add_argument("--ppi-edge-file", type=Path, default=DEFAULT_PPI_EDGE_FILE)
    parser.add_argument("--ppi-score-threshold", type=float, default=0.73)
    parser.add_argument("--volumetric-beta", type=float, default=1.0)
    parser.add_argument("--volumetric-eps", type=float, default=1e-8)
    parser.add_argument("--volumetric-gate-init", type=float, default=0.0)
    parser.add_argument(
        "--volumetric-volume-mode",
        choices=["raw", "l2"],
        default="raw",
    )
    parser.add_argument(
        "--volumetric-dropout",
        type=float,
        default=None,
        help="VMA-only attention dropout; omit to inherit the model dropout.",
    )
    parser.add_argument(
        "--volumetric-message-mode",
        choices=["legacy", "expression_contrast"],
        default="legacy",
    )
    parser.add_argument(
        "--volumetric-output-norm",
        choices=["none", "rms"],
        default="none",
    )
    parser.add_argument(
        "--volumetric-gate-mode",
        choices=["scalar", "per_head"],
        default="scalar",
    )
    parser.add_argument(
        "--volumetric-backbone-gradient-mode",
        choices=["coupled", "detached"],
        default="coupled",
    )
    parser.add_argument(
        "--evaluation-only-checkpoint",
        type=Path,
        default=None,
        help="Pass a checkpoint to the worker, skip optimization, and evaluate only test.",
    )
    parser.add_argument(
        "--skip-final-test",
        action="store_true",
        help="Train/select on validation without evaluating test after training.",
    )

    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--cv-seed", type=int, default=42)
    parser.add_argument("--seeds", nargs="+", type=int, default=[101, 102, 103, 104, 105, 106, 107, 108, 109, 110])
    parser.add_argument("--seed-train-ratio", type=float, default=0.70)
    parser.add_argument("--seed-val-ratio", type=float, default=0.10)
    parser.add_argument("--seed-test-ratio", type=float, default=0.20)

    parser.add_argument("--max-genes", type=int, default=512)
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
        default="pairwise_anova_union",
    )
    parser.add_argument("--ad-mci-gene-fraction", type=float, default=0.5)
    parser.add_argument("--scaler", choices=["none", "minmax", "standard"], default="minmax")
    parser.add_argument("--feature-selection-fit-scope", choices=["train", "all"], default="train")
    parser.add_argument("--scaler-fit-scope", choices=["train", "all"], default="train")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--train-sampling", choices=["random", "balanced_classes"], default="random")
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--early-stopping-patience", type=int, default=50)
    parser.add_argument("--val-loss-stop-threshold", type=float, default=None)
    parser.add_argument("--val-loss-stop-patience", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-embedding", type=float, default=None)
    parser.add_argument("--lr-volumetric", type=float, default=None)
    parser.add_argument("--freeze-embedding-epochs", type=int, default=0)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--d-ff", type=int, default=128)
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
    parser.add_argument(
        "--grad-clip-scope",
        choices=["joint", "separate_volumetric"],
        default="joint",
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
    parser.add_argument("--tupe-mode", choices=["on", "off"], default="on")
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
    parser.add_argument(
        "--embed-file",
        type=Path,
        default=None,
        help="Optional HIPPIE/PPI or other external gene embedding CSV passed through to the multitask worker.",
    )
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
        default="val_primary_auc_minus_025_loss",
    )
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--evaluate-test-each-epoch", choices=["on", "off"], default="off")
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else (ROOT / path).resolve()


def load_labels(y_file: Path) -> pd.DataFrame:
    y_df = pd.read_csv(y_file)
    if "sample_id" not in y_df.columns or "label" not in y_df.columns:
        raise ValueError(f"{y_file} must contain sample_id and label columns.")
    return y_df[["sample_id", "label"]].copy()


def split_rows(sample_ids: pd.Series, train_idx, val_idx, test_idx) -> pd.DataFrame:
    rows = (
        [{"sample_id": sample_ids.iloc[idx], "split": "train"} for idx in train_idx]
        + [{"sample_id": sample_ids.iloc[idx], "split": "val"} for idx in val_idx]
        + [{"sample_id": sample_ids.iloc[idx], "split": "test"} for idx in test_idx]
    )
    return pd.DataFrame(rows)


def write_cv_splits(y_df: pd.DataFrame, output_dir: Path, n_folds: int, seed: int) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = y_df["label"].to_numpy()
    sample_ids = y_df["sample_id"].astype(str)
    splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    split_paths: list[Path] = []
    for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(sample_ids, labels), start=1):
        split_df = pd.DataFrame(
            [{"sample_id": sample_ids.iloc[idx], "split": "train"} for idx in train_idx]
            + [{"sample_id": sample_ids.iloc[idx], "split": "val"} for idx in val_idx]
        )
        split_path = output_dir / f"fold_{fold_idx:02d}.csv"
        split_df.to_csv(split_path, index=False)
        split_paths.append(split_path)
    return split_paths


def write_seed_splits(
    y_df: pd.DataFrame,
    output_dir: Path,
    seeds: list[int],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> list[tuple[int, Path]]:
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-8:
        raise ValueError("seed split ratios must sum to 1.0.")
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = y_df["label"].to_numpy()
    sample_ids = y_df["sample_id"].astype(str)
    all_idx = y_df.index.to_numpy()
    split_paths: list[tuple[int, Path]] = []
    holdout_ratio = val_ratio + test_ratio
    test_fraction_of_holdout = test_ratio / holdout_ratio
    for seed in seeds:
        train_idx, holdout_idx = train_test_split(
            all_idx,
            test_size=holdout_ratio,
            random_state=seed,
            stratify=labels,
        )
        val_idx, test_idx = train_test_split(
            holdout_idx,
            test_size=test_fraction_of_holdout,
            random_state=seed + 1000,
            stratify=labels[holdout_idx],
        )
        split_df = split_rows(sample_ids, train_idx, val_idx, test_idx)
        split_path = output_dir / f"seed_{seed}.csv"
        split_df.to_csv(split_path, index=False)
        split_paths.append((seed, split_path))
    return split_paths


def build_worker_command(
    args: argparse.Namespace,
    arch_key: str,
    split_path: Path,
    result_dir: Path,
    seed: int,
    evaluate_test: str,
) -> list[str]:
    arch = ARCHITECTURES[arch_key]
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
        "--post-model-construction-reseed", getattr(
            args, "post_model_construction_reseed", "off"
        ),
        "--model-variant", getattr(args, "model_variant", "baseline"),
        "--ppi-edge-file", str(getattr(args, "ppi_edge_file", DEFAULT_PPI_EDGE_FILE)),
        "--ppi-score-threshold", str(getattr(args, "ppi_score_threshold", 0.73)),
        "--volumetric-beta", str(getattr(args, "volumetric_beta", 1.0)),
        "--volumetric-eps", str(getattr(args, "volumetric_eps", 1e-8)),
        "--volumetric-gate-init", str(getattr(args, "volumetric_gate_init", 0.0)),
        "--volumetric-volume-mode", getattr(args, "volumetric_volume_mode", "raw"),
        "--volumetric-message-mode", getattr(args, "volumetric_message_mode", "legacy"),
        "--volumetric-output-norm", getattr(args, "volumetric_output_norm", "none"),
        "--volumetric-gate-mode", getattr(args, "volumetric_gate_mode", "scalar"),
        "--volumetric-backbone-gradient-mode", getattr(
            args, "volumetric_backbone_gradient_mode", "coupled"
        ),
        "--max-genes", str(args.max_genes),
        "--gene-selection", args.gene_selection,
        "--ad-mci-gene-fraction", str(args.ad_mci_gene_fraction),
        "--scaler", args.scaler,
        "--feature-selection-fit-scope", args.feature_selection_fit_scope,
        "--scaler-fit-scope", args.scaler_fit_scope,
        "--batch-size", str(args.batch_size),
        "--train-sampling", getattr(args, "train_sampling", "random"),
        "--epochs", str(args.epochs),
        "--early-stopping-patience", str(args.early_stopping_patience),
        "--val-loss-stop-patience", str(args.val_loss_stop_patience),
        "--lr-encoder", str(args.lr),
        "--lr-head", str(args.lr),
        "--freeze-embedding-epochs", str(getattr(args, "freeze_embedding_epochs", 0)),
        "--weight-decay", str(args.weight_decay),
        "--class-weighting", args.class_weighting,
        "--task-loss-weights", *[str(value) for value in getattr(args, "task_loss_weights", [1.0, 1.0, 1.0])],
        "--pooling-mode", getattr(args, "pooling_mode", "average"),
        "--attention-pooling-hidden-dim", str(getattr(args, "attention_pooling_hidden_dim", 16)),
        "--attention-pooling-dropout", str(getattr(args, "attention_pooling_dropout", 0.1)),
        "--primary-adapter-dim", str(getattr(args, "primary_adapter_dim", 0)),
        "--expression-residual", getattr(args, "expression_residual", "none"),
        "--tupe-mode", getattr(args, "tupe_mode", "on"),
        "--checkpoint-ensemble-size", str(getattr(args, "checkpoint_ensemble_size", 1)),
        "--checkpoint-ensemble-min-gap", str(getattr(args, "checkpoint_ensemble_min_gap", 3)),
        "--augmentation", args.augmentation,
        "--smote-k-neighbors", str(args.smote_k_neighbors),
        "--smote-m-neighbors", str(args.smote_m_neighbors),
        "--smote-kind", args.smote_kind,
        "--pca-neighbor-components", str(getattr(args, "pca_neighbor_components", 50)),
        "--pca-neighbor-k", str(getattr(args, "pca_neighbor_k", 5)),
        "--pca-neighbor-gap-fraction", str(getattr(args, "pca_neighbor_gap_fraction", 0.5)),
        "--pca-neighbor-target-count", str(getattr(args, "pca_neighbor_target_count", 0)),
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
        "--evaluate-test", evaluate_test,
        "--evaluate-test-each-epoch", args.evaluate_test_each_epoch,
        "--device", args.device,
        "--n-layers", str(arch["n_layers"]),
        "--n-heads", str(arch["n_heads"]),
        "--d-model", str(args.d_model),
        "--embed-dim", str(args.d_model),
        "--d-ff", str(args.d_ff),
        "--dropout", str(args.dropout),
        "--aggfunc", "Avgpool",
        "--task-specific-pooling", args.task_specific_pooling,
        "--head-norm", getattr(args, "head_norm", "batch"),
        "--mask-aware-heads", getattr(args, "mask_aware_heads", "off"),
        "--encoder-sharing", getattr(args, "encoder_sharing", "shared"),
        "--gradient-strategy", getattr(args, "gradient_strategy", "weighted_sum"),
        "--gradient-diagnostics", getattr(args, "gradient_diagnostics", "off"),
        "--gradient-diagnostic-interval", str(getattr(args, "gradient_diagnostic_interval", 1)),
        "--d-hidden1", str(args.d_hidden1),
        "--d-hidden2", str(args.d_hidden2),
        "--grad-clip-norm", "1.0",
        "--grad-clip-scope", getattr(args, "grad_clip_scope", "joint"),
        "--embedding-gene-policy", args.embedding_gene_policy,
        "--embedding-rescale", getattr(args, "embedding_rescale", "none"),
        "--embedding-init-scale", str(getattr(args, "embedding_init_scale", 0.02)),
        "--ppi-integration", getattr(args, "ppi_integration", "direct"),
        "--ppi-gate-init", str(getattr(args, "ppi_gate_init", 0.0)),
    ]
    if getattr(args, "volumetric_dropout", None) is not None:
        cmd.extend(["--volumetric-dropout", str(args.volumetric_dropout)])
    if getattr(args, "lr_volumetric", None) is not None:
        cmd.extend(["--lr-volumetric", str(args.lr_volumetric)])
    if getattr(args, "lr_embedding", None) is not None:
        cmd.extend(["--lr-embedding", str(args.lr_embedding)])
    if args.embed_file is not None:
        cmd.extend(["--embed-file", str(args.embed_file)])
    if getattr(args, "candidate_gene_file", None) is not None:
        cmd.extend(["--candidate-gene-file", str(args.candidate_gene_file)])
    if getattr(args, "evaluation_only_checkpoint", None) is not None:
        cmd.extend(["--evaluation-only-checkpoint", str(args.evaluation_only_checkpoint)])
    if getattr(args, "skip_final_test", False):
        cmd.append("--skip-final-test")
    if args.val_loss_stop_threshold is not None:
        cmd.extend(["--val-loss-stop-threshold", str(args.val_loss_stop_threshold)])
    if args.max_train_batches is not None:
        cmd.extend(["--max-train-batches", str(args.max_train_batches)])
    if args.max_val_batches is not None:
        cmd.extend(["--max-val-batches", str(args.max_val_batches)])
    return cmd


def architecture_run_name(args: argparse.Namespace, arch_key: str) -> str:
    arch = ARCHITECTURES[arch_key]
    dropout_tag = str(args.dropout).replace(".", "p")
    pooling_tag = "taskpool" if args.task_specific_pooling == "on" else "sharedpool"
    augmentation_tag = args.augmentation
    embedding_tag = "ppi" if args.embed_file is not None else "random"
    encoder_tag = f"enc{getattr(args, 'encoder_sharing', 'shared')}"
    norm_tag = f"hn{getattr(args, 'head_norm', 'batch')}"
    rescale = getattr(args, "embedding_rescale", "none") if args.embed_file is not None else "none"
    embedding_scale_tag = f"er{rescale}"
    loss_tag = "lw" + "-".join(str(value).replace(".", "p") for value in getattr(args, "task_loss_weights", [1, 1, 1]))
    pooling_mode = getattr(args, "pooling_mode", "average")
    learned_pooling_tag = "poolavg" if pooling_mode == "average" else f"poolatt{getattr(args, 'attention_pooling_hidden_dim', 16)}"
    adapter_tag = f"adp{getattr(args, 'primary_adapter_dim', 0)}"
    expression_tag = "exprnone" if getattr(args, "expression_residual", "none") == "none" else "expradd"
    ensemble_tag = f"ens{getattr(args, 'checkpoint_ensemble_size', 1)}"
    mask_tag = "maskheads" if getattr(args, "mask_aware_heads", "off") == "on" else "allheads"
    gradient_tag = {
        "weighted_sum": "gradsum",
        "primary_protected_pcgrad": "gradppc",
    }[getattr(args, "gradient_strategy", "weighted_sum")]
    ppi_tag = "ppigate" if getattr(args, "ppi_integration", "direct") == "gated_residual" else "ppidirect"
    sampling_tag = "balsampler" if getattr(args, "train_sampling", "random") == "balanced_classes" else "randsampler"
    run_name = (
        f"txt_multitask_{arch['n_layers']}l{arch['n_heads']}h_"
        f"d{args.d_model}_ff{args.d_ff}_do{dropout_tag}_b{args.batch_size}_"
        f"{pooling_tag}_{augmentation_tag}_{embedding_tag}_{encoder_tag}_{norm_tag}_{embedding_scale_tag}_"
        f"{loss_tag}_{learned_pooling_tag}_{adapter_tag}_{expression_tag}_{ensemble_tag}_{mask_tag}_{gradient_tag}_{ppi_tag}_{sampling_tag}"
    )
    if getattr(args, "model_variant", "baseline") == "ppi_volumetric":
        beta_tag = str(getattr(args, "volumetric_beta", 1.0)).replace(".", "p")
        score_tag = str(getattr(args, "ppi_score_threshold", 0.73)).replace(".", "p")
        run_name += f"_ppi_volumetric_beta{beta_tag}_score{score_tag}"
        volume_mode = getattr(args, "volumetric_volume_mode", "raw")
        volumetric_dropout = getattr(args, "volumetric_dropout", None)
        if volume_mode != "raw":
            run_name += f"_volume{volume_mode}"
        if volumetric_dropout is not None:
            dropout_tag = str(volumetric_dropout).replace(".", "p")
            run_name += f"_vdo{dropout_tag}"
        message_mode = getattr(args, "volumetric_message_mode", "legacy")
        if message_mode != "legacy":
            run_name += f"_msg{message_mode}"
        output_norm = getattr(args, "volumetric_output_norm", "none")
        if output_norm != "none":
            run_name += f"_vnorm{output_norm}"
        gate_mode = getattr(args, "volumetric_gate_mode", "scalar")
        if gate_mode != "scalar":
            run_name += f"_gate{gate_mode}"
        gradient_mode = getattr(args, "volumetric_backbone_gradient_mode", "coupled")
        if gradient_mode != "coupled":
            run_name += f"_vmagrad{gradient_mode}"
    if getattr(args, "grad_clip_scope", "joint") != "joint":
        run_name += "_clipseparatevma"
    if getattr(args, "post_model_construction_reseed", "off") == "on":
        run_name += "_postmodelreseed"
    if getattr(args, "evaluation_only_checkpoint", None) is not None:
        run_name += "_eval_only"
    elif getattr(args, "skip_final_test", False):
        run_name += "_val_only"
    return run_name


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
    metrics["split"] = pd.Categorical(metrics["split"], categories=["val", "test"], ordered=True)
    display_cols = [
        "split",
        "task",
        "samples",
        "loss",
        "accuracy",
        "macro_f1",
        "balanced_accuracy",
        "roc_auc",
    ]
    available = [col for col in display_cols if col in metrics.columns]
    print("\nRepeat metric summary (val/test):", flush=True)
    print(
        metrics[available]
        .sort_values(["split", "task"])
        .to_string(index=False, float_format=lambda value: f"{value:.4f}"),
        flush=True,
    )


def collect_metrics(result_root: Path, evaluation: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for metrics_path in sorted((result_root / evaluation).glob("*/*/metrics_summary.csv")):
        arch = metrics_path.parent.parent.name
        run_name = metrics_path.parent.name
        metrics = pd.read_csv(metrics_path)
        for _, row in metrics.iterrows():
            payload = row.to_dict()
            payload["evaluation"] = evaluation
            payload["arch"] = arch
            payload["run"] = run_name
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
    grouped = runs.groupby(["evaluation", "arch", "split", "task"], as_index=False)
    summary = grouped[available].agg(["mean", "std"])
    summary.columns = [
        "_".join([part for part in col if part]).rstrip("_")
        if isinstance(col, tuple)
        else col
        for col in summary.columns
    ]
    counts = grouped.size().rename(columns={"size": "n_runs"})
    return counts.merge(summary, on=["evaluation", "arch", "split", "task"], how="left")


def run_cv(args: argparse.Namespace, y_df: pd.DataFrame) -> None:
    cv_root = args.result_root / "cv5"
    split_paths = write_cv_splits(y_df, cv_root / "splits", args.n_folds, args.cv_seed)
    for arch_key in args.architectures:
        arch_name = architecture_run_name(args, arch_key)
        for fold_idx, split_path in enumerate(split_paths, start=1):
            run_dir = cv_root / arch_name / f"fold_{fold_idx:02d}"
            if args.skip_existing and (run_dir / "metrics_summary.csv").exists():
                print(f"Skipping existing CV run: {run_dir}", flush=True)
                continue
            print("\n" + "=" * 80, flush=True)
            print(f"TxT multitask 5-CV | arch={arch_name} | fold={fold_idx}/{len(split_paths)}", flush=True)
            print("=" * 80, flush=True)
            cmd = build_worker_command(args, arch_key, split_path, run_dir, seed=args.cv_seed + fold_idx, evaluate_test="off")
            run_command(cmd, run_dir / "train.log")
            print_repeat_metric_summary(run_dir)
    runs = collect_metrics(args.result_root, "cv5")
    runs.to_csv(cv_root / "cv5_runs.csv", index=False)
    summarize_metrics(runs).to_csv(cv_root / "cv5_summary.csv", index=False)


def run_seeds(args: argparse.Namespace, y_df: pd.DataFrame) -> None:
    seed_root = args.result_root / "seeds10"
    split_paths = write_seed_splits(
        y_df,
        seed_root / "splits",
        args.seeds,
        args.seed_train_ratio,
        args.seed_val_ratio,
        args.seed_test_ratio,
    )
    for arch_key in args.architectures:
        arch_name = architecture_run_name(args, arch_key)
        for seed, split_path in split_paths:
            run_dir = seed_root / arch_name / f"seed_{seed}"
            if args.skip_existing and (run_dir / "metrics_summary.csv").exists():
                print(f"Skipping existing seed run: {run_dir}", flush=True)
                continue
            print("\n" + "=" * 80, flush=True)
            print(f"TxT multitask 10-seed | arch={arch_name} | seed={seed}", flush=True)
            print("=" * 80, flush=True)
            cmd = build_worker_command(args, arch_key, split_path, run_dir, seed=seed, evaluate_test="on")
            run_command(cmd, run_dir / "train.log")
            print_repeat_metric_summary(run_dir)
    runs = collect_metrics(args.result_root, "seeds10")
    runs.to_csv(seed_root / "seeds10_runs.csv", index=False)
    summarize_metrics(runs).to_csv(seed_root / "seeds10_summary.csv", index=False)


def apply_smoke_overrides(args: argparse.Namespace) -> None:
    if not args.smoke:
        return
    args.result_root = args.result_root / "smoke"
    args.device = "cpu"
    args.epochs = 1
    args.gan_epochs = min(args.gan_epochs, 1)
    args.early_stopping_patience = 1
    args.n_folds = 2
    args.seeds = args.seeds[:1]
    args.architectures = args.architectures[:1]
    args.max_train_batches = 2 if args.max_train_batches is None else args.max_train_batches
    args.max_val_batches = 2 if args.max_val_batches is None else args.max_val_batches


def main() -> None:
    args = parse_args()
    if args.volumetric_dropout is not None and (
        not math.isfinite(args.volumetric_dropout)
        or not 0.0 <= args.volumetric_dropout <= 1.0
    ):
        raise ValueError("--volumetric-dropout must be finite and in [0, 1].")
    if args.lr_volumetric is not None and (
        not math.isfinite(args.lr_volumetric) or args.lr_volumetric <= 0
    ):
        raise ValueError("--lr-volumetric must be finite and positive when provided.")
    args.x_file = resolve_path(args.x_file)
    args.y_file = resolve_path(args.y_file)
    args.result_root = resolve_path(args.result_root)
    args.embed_file = resolve_path(args.embed_file) if args.embed_file is not None else None
    args.candidate_gene_file = resolve_path(args.candidate_gene_file) if args.candidate_gene_file is not None else None
    args.ppi_edge_file = resolve_path(args.ppi_edge_file)
    args.evaluation_only_checkpoint = (
        resolve_path(args.evaluation_only_checkpoint)
        if args.evaluation_only_checkpoint is not None
        else None
    )
    apply_smoke_overrides(args)
    if args.evaluation_only_checkpoint is not None:
        if args.skip_final_test:
            raise ValueError("--skip-final-test cannot be combined with --evaluation-only-checkpoint.")
        if args.run_mode != "seeds" or len(args.architectures) != 1 or len(args.seeds) != 1:
            raise ValueError(
                "Runner evaluation-only mode requires --run-mode seeds, exactly one architecture, "
                "and exactly one seed so the checkpoint/split pairing is unambiguous."
            )
    args.result_root.mkdir(parents=True, exist_ok=True)
    if not WORKER.exists():
        raise FileNotFoundError(f"Worker not found: {WORKER}")
    y_df = load_labels(args.y_file)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    with (args.result_root / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, sort_keys=True)
        handle.write("\n")

    if args.run_mode in {"both", "cv"}:
        run_cv(args, y_df)
    if args.run_mode in {"both", "seeds"}:
        run_seeds(args, y_df)
    print(f"TxT multitask evaluation outputs: {args.result_root}", flush=True)


if __name__ == "__main__":
    main()
