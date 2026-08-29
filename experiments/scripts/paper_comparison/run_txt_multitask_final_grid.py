#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
RUNNER = ROOT / "experiments" / "scripts" / "paper_comparison" / "run_txt_multitask_cv_and_seeds.py"
DEFAULT_SHARED_DIR = ROOT / "task_dataset" / "processed" / "txt_pairwise_multitask" / "shared_ad_mci_ctl"
DEFAULT_PPI_TEMPLATE = (
    ROOT
    / "results"
    / "pretraining"
    / "ppi_init"
    / "hippie_highconf_dim{dim}_score0p73_seed42"
    / "ppi_node_embedding.csv"
)

RANKED_SUMMARY_COLUMNS = [
    "job_name",
    "evaluation",
    "arch",
    "split",
    "primary_task",
    "primary_roc_auc_mean",
    "primary_macro_f1_mean",
    "primary_balanced_accuracy_mean",
    "auxiliary_roc_auc_mean",
    "auxiliary_macro_f1_mean",
    "n_runs",
    "expected_runs",
    "is_complete",
    "summary_path",
]


GRID_PRESETS: dict[str, dict[str, list[Any]]] = {
    "pilot": {
        "embedding_sources": ["ppi"],
        "d_models": [128],
        "dropouts": [0.4],
        "max_genes_list": [1000, 1500],
        "gene_selections": ["mad", "ad_mci_priority_anova_union"],
        "ad_mci_gene_fractions": [0.5],
        "augmentations": ["none", "borderline_smote"],
    },
    "balanced": {
        "embedding_sources": ["ppi"],
        "d_models": [64, 128, 256],
        "dropouts": [0.3, 0.4, 0.5],
        "max_genes_list": [512, 1000, 1500, 2000],
        "gene_selections": ["mad", "ad_mci_priority_anova_union", "pairwise_anova_union"],
        "ad_mci_gene_fractions": [0.5, 0.6],
        "augmentations": ["none", "borderline_smote"],
    },
    "wide": {
        "embedding_sources": ["ppi", "random"],
        "d_models": [64, 128, 256],
        "dropouts": [0.2, 0.3, 0.4, 0.5],
        "max_genes_list": [512, 1000, 1500, 2000, 3000],
        "gene_selections": ["mad", "variance", "ad_mci_priority_anova_union", "pairwise_anova_union"],
        "ad_mci_gene_fractions": [0.5, 0.6, 0.7],
        "augmentations": ["none", "smote", "borderline_smote"],
    },
    "random_sanity": {
        "embedding_sources": ["random"],
        "d_models": [128],
        "dropouts": [0.4],
        "max_genes_list": [1000],
        "gene_selections": ["mad"],
        "ad_mci_gene_fractions": [0.5],
        "augmentations": ["none"],
    },
}


@dataclass(frozen=True)
class GridJob:
    job_name: str
    result_root: Path
    embedding_source: str
    embed_file: Path | None
    embedding_gene_policy: str
    gene_selection: str
    max_genes: int
    augmentation: str
    d_model: int
    d_ff: int
    dropout: float
    ad_mci_gene_fraction: float
    encoder_sharing: str = "shared"
    head_norm: str = "batch"
    embedding_rescale: str = "none"
    task_loss_weights: tuple[float, float, float] = (1.0, 1.0, 1.0)
    checkpoint_metric: str = "val_primary_auc_minus_025_loss"
    pooling_mode: str = "average"
    attention_pooling_hidden_dim: int = 16
    primary_adapter_dim: int = 0
    expression_residual: str = "none"
    checkpoint_ensemble_size: int = 1
    ppi_integration: str = "direct"
    pca_neighbor_target_count: int = 0
    train_sampling: str = "random"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the TxT multitask grid: shared or task-specific encoders, one gene set, three binary heads, "
            "optional PPI/HIPPIE initialization, train-only feature selection, and train-only augmentation."
        )
    )
    parser.add_argument("--python-exe", default=sys.executable)
    parser.add_argument("--x-file", type=Path, default=DEFAULT_SHARED_DIR / "X.csv")
    parser.add_argument("--y-file", type=Path, default=DEFAULT_SHARED_DIR / "y.csv")
    parser.add_argument("--result-root", type=Path, default=ROOT / "results" / "paper_comparison" / "txt_multitask_final_grid")
    parser.add_argument("--run-mode", choices=["cv", "seeds", "both"], default="seeds")
    parser.add_argument("--device", choices=["cuda", "cpu", "mps"], default="cuda")
    parser.add_argument("--architectures", nargs="+", default=["1l2h"])
    parser.add_argument(
        "--preset",
        choices=["pilot", "balanced", "wide", "random_sanity", "custom"],
        default="balanced",
        help=(
            "Grid preset. Explicit list arguments override only their corresponding preset list. "
            "Use custom when every list is passed manually."
        ),
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--collect-only", action="store_true", help="Only collect existing grid summaries and write grid_ranked_summary.csv.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")

    parser.add_argument("--embedding-sources", nargs="+", choices=["random", "ppi"], default=None)
    parser.add_argument("--ppi-embedding-template", type=Path, default=DEFAULT_PPI_TEMPLATE)
    parser.add_argument("--skip-missing-ppi", action="store_true")
    parser.add_argument("--ppi-gene-policy", choices=["all", "mapped_only"], default="mapped_only")
    parser.add_argument(
        "--candidate-gene-file",
        type=Path,
        default=None,
        help="Optional common candidate-gene CSV used by both random and PPI jobs for a controlled comparison.",
    )
    parser.add_argument(
        "--embedding-rescales",
        nargs="+",
        choices=["none", "global_std", "per_dimension_std"],
        default=["none"],
    )
    parser.add_argument("--embedding-init-scale", type=float, default=0.02)
    parser.add_argument(
        "--ppi-integrations", nargs="+", choices=["direct", "gated_residual"], default=["direct"]
    )
    parser.add_argument("--ppi-gate-init", type=float, default=0.0)

    parser.add_argument("--d-models", nargs="+", type=int, default=None)
    parser.add_argument("--d-ff-multiplier", type=int, default=4)
    parser.add_argument("--dropouts", nargs="+", type=float, default=None)
    parser.add_argument("--max-genes-list", nargs="+", type=int, default=None)
    parser.add_argument(
        "--gene-selections",
        nargs="+",
        choices=["mad", "variance", "pairwise_anova_union", "ad_mci_priority_anova_union", "ad_mci_vs_ctl_anova_50_50"],
        default=None,
    )
    parser.add_argument("--ad-mci-gene-fractions", nargs="+", type=float, default=None)
    parser.add_argument(
        "--augmentations",
        nargs="+",
        choices=["none", "smote", "borderline_smote", "pca_neighbor_mci_ctl", "pca_neighbor_all_tasks", "ctgan", "gan"],
        default=None,
    )

    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--train-samplings", nargs="+", choices=["random", "balanced_classes"], default=["random"]
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--early-stopping-patience", type=int, default=30)
    parser.add_argument("--val-loss-stop-threshold", type=float, default=None)
    parser.add_argument("--val-loss-stop-patience", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-embedding", type=float, default=None)
    parser.add_argument("--freeze-embedding-epochs", type=int, default=0)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--task-specific-pooling", choices=["on", "off"], default="off")
    parser.add_argument("--head-norms", nargs="+", choices=["batch", "layer", "none"], default=["batch"])
    parser.add_argument("--mask-aware-heads", choices=["on", "off"], default="off")
    parser.add_argument("--encoder-sharings", nargs="+", choices=["shared", "separate"], default=["shared"])
    parser.add_argument(
        "--gradient-strategy",
        choices=["weighted_sum", "primary_protected_pcgrad"],
        default="weighted_sum",
    )
    parser.add_argument("--gradient-diagnostics", choices=["on", "off"], default="off")
    parser.add_argument("--gradient-diagnostic-interval", type=int, default=1)
    parser.add_argument("--class-weighting", choices=["on", "off"], default="off")
    parser.add_argument("--checkpoint-metric", default="val_primary_auc_minus_025_loss")
    parser.add_argument("--checkpoint-metrics", nargs="+", default=None)
    parser.add_argument(
        "--task-loss-weight-sets",
        nargs="+",
        default=["1,1,1"],
        help="Comma-separated AD/MCI,AD/CTL,MCI/CTL weights; for example 0.5,0.25,0.25.",
    )
    parser.add_argument("--pooling-modes", nargs="+", choices=["average", "task_attention"], default=["average"])
    parser.add_argument("--attention-pooling-hidden-dims", nargs="+", type=int, default=[16])
    parser.add_argument("--attention-pooling-dropout", type=float, default=0.1)
    parser.add_argument("--primary-adapter-dims", nargs="+", type=int, default=[0])
    parser.add_argument(
        "--expression-residuals", nargs="+", choices=["none", "additive_zero_init"], default=["none"]
    )
    parser.add_argument("--checkpoint-ensemble-sizes", nargs="+", type=int, default=[1])
    parser.add_argument("--checkpoint-ensemble-min-gap", type=int, default=3)
    parser.add_argument("--evaluate-test-each-epoch", choices=["on", "off"], default="off")
    parser.add_argument("--smote-k-neighbors", type=int, default=5)
    parser.add_argument("--smote-m-neighbors", type=int, default=10)
    parser.add_argument("--smote-kind", choices=["borderline-1", "borderline-2"], default="borderline-1")
    parser.add_argument("--pca-neighbor-components", type=int, default=50)
    parser.add_argument("--pca-neighbor-k", type=int, default=5)
    parser.add_argument("--pca-neighbor-gap-fraction", type=float, default=0.5)
    parser.add_argument(
        "--pca-neighbor-target-counts",
        nargs="+",
        type=int,
        default=[0],
        help="Grid of desired final per-class counts for pca_neighbor_all_tasks; 0 uses the largest train class.",
    )
    parser.add_argument("--ctgan-epochs", type=int, default=100)
    parser.add_argument("--ctgan-batch-size", type=int, default=128)
    parser.add_argument("--augmentation-target-multiplier", type=float, default=2.0)
    parser.add_argument("--gan-epochs", type=int, default=100)
    parser.add_argument("--gan-batch-size", type=int, default=64)
    parser.add_argument("--gan-latent-dim", type=int, default=64)
    parser.add_argument("--gan-learning-rate", type=float, default=0.001)
    parser.add_argument("--gan-target-multiplier", type=float, default=1.5)
    parser.add_argument("--gan-sampling-strategy", choices=["balanced", "minority", "proportional"], default="balanced")
    parser.add_argument("--seeds", nargs="+", type=int, default=[101, 102, 103, 104, 105, 106, 107, 108, 109, 110])
    parser.add_argument("--seed-train-ratio", type=float, default=0.70)
    parser.add_argument("--seed-val-ratio", type=float, default=0.10)
    parser.add_argument("--seed-test-ratio", type=float, default=0.20)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else (ROOT / path).resolve()


def format_float(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def parse_task_loss_weight_set(value: str) -> tuple[float, float, float]:
    parts = [float(part.strip()) for part in str(value).split(",")]
    if len(parts) != 3 or any(part <= 0 for part in parts):
        raise ValueError(f"Invalid task loss weight set {value!r}; expected three positive comma-separated values.")
    return parts[0], parts[1], parts[2]


def render_ppi_path(template: Path, dim: int) -> Path:
    rendered = str(template).format(dim=dim)
    return resolve(Path(rendered))


def apply_grid_preset(args: argparse.Namespace) -> None:
    if args.preset == "custom":
        missing = [
            name
            for name in [
                "embedding_sources",
                "d_models",
                "dropouts",
                "max_genes_list",
                "gene_selections",
                "ad_mci_gene_fractions",
                "augmentations",
            ]
            if getattr(args, name) is None
        ]
        if missing:
            raise ValueError(f"--preset custom requires explicit values for: {', '.join(missing)}")
        return
    preset = GRID_PRESETS[args.preset]
    for key, value in preset.items():
        if getattr(args, key) is None:
            setattr(args, key, list(value))


def build_jobs(args: argparse.Namespace) -> tuple[list[GridJob], list[dict[str, Any]]]:
    jobs: list[GridJob] = []
    skipped: list[dict[str, Any]] = []
    weight_sets = [parse_task_loss_weight_set(value) for value in getattr(args, "task_loss_weight_sets", ["1,1,1"])]
    default_checkpoint_metric = getattr(args, "checkpoint_metric", "val_primary_auc_minus_025_loss")
    checkpoint_metrics = getattr(args, "checkpoint_metrics", None) or [default_checkpoint_metric]
    pca_target_counts = getattr(args, "pca_neighbor_target_counts", [0])
    for embedding_source, gene_selection, max_genes, augmentation, d_model, dropout, encoder_sharing, head_norm, embedding_rescale, task_loss_weights, checkpoint_metric, pooling_mode, attention_hidden_dim, primary_adapter_dim, expression_residual, ensemble_size, ppi_integration, pca_target_count, train_sampling in itertools.product(
        args.embedding_sources,
        args.gene_selections,
        args.max_genes_list,
        args.augmentations,
        args.d_models,
        args.dropouts,
        getattr(args, "encoder_sharings", ["shared"]),
        getattr(args, "head_norms", ["batch"]),
        getattr(args, "embedding_rescales", ["none"]),
        weight_sets,
        checkpoint_metrics,
        getattr(args, "pooling_modes", ["average"]),
        getattr(args, "attention_pooling_hidden_dims", [16]),
        getattr(args, "primary_adapter_dims", [0]),
        getattr(args, "expression_residuals", ["none"]),
        getattr(args, "checkpoint_ensemble_sizes", [1]),
        getattr(args, "ppi_integrations", ["direct"]),
        pca_target_counts,
        getattr(args, "train_samplings", ["random"]),
    ):
        if augmentation != "pca_neighbor_all_tasks" and pca_target_count != pca_target_counts[0]:
            continue
        effective_pca_target_count = int(pca_target_count) if augmentation == "pca_neighbor_all_tasks" else 0
        if embedding_source == "random" and ppi_integration != "direct":
            continue
        if embedding_source == "random" and embedding_rescale != "none":
            continue
        if ppi_integration == "gated_residual" and embedding_rescale != "none":
            continue
        if pooling_mode == "average" and attention_hidden_dim != getattr(args, "attention_pooling_hidden_dims", [16])[0]:
            continue
        fractions = args.ad_mci_gene_fractions if gene_selection == "ad_mci_priority_anova_union" else [0.5]
        for ad_mci_fraction in fractions:
            d_ff = int(d_model * args.d_ff_multiplier)
            embed_file = None
            embedding_policy = "all"
            if embedding_source == "ppi":
                embed_file = render_ppi_path(args.ppi_embedding_template, d_model)
                embedding_policy = args.ppi_gene_policy
                if not embed_file.exists():
                    row = {
                        "status": "missing_ppi_embedding",
                        "d_model": d_model,
                        "embed_file": str(embed_file),
                        "gene_selection": gene_selection,
                        "max_genes": max_genes,
                        "augmentation": augmentation,
                        "dropout": dropout,
                        "ad_mci_gene_fraction": ad_mci_fraction,
                    }
                    if args.skip_missing_ppi:
                        skipped.append(row)
                        continue
                    raise FileNotFoundError(
                        f"PPI embedding not found for d_model={d_model}: {embed_file}. "
                        "Run run_ppi_init.ps1 for each d_model or pass --skip-missing-ppi."
                    )
            selection_tag = gene_selection.replace("_", "-")
            job_name = (
                f"{embedding_source}_{selection_tag}_k{max_genes}_{augmentation}_"
                f"d{d_model}_ff{d_ff}_do{format_float(dropout)}_admci{format_float(ad_mci_fraction)}"
            )
            if getattr(args, "task_specific_pooling", "off") == "on":
                job_name = f"{job_name}_taskpool"
            if encoder_sharing != "shared":
                job_name = f"{job_name}_enc{encoder_sharing}"
            if head_norm != "batch":
                job_name = f"{job_name}_hn{head_norm}"
            if embedding_rescale != "none":
                job_name = f"{job_name}_er{embedding_rescale}"
            if getattr(args, "candidate_gene_file", None) is not None:
                job_name = f"{job_name}_candfile"
            if task_loss_weights != (1.0, 1.0, 1.0):
                weight_tag = "-".join(format_float(value) for value in task_loss_weights)
                job_name = f"{job_name}_lw{weight_tag}"
            if checkpoint_metric != default_checkpoint_metric or len(checkpoint_metrics) > 1:
                job_name = f"{job_name}_ckpt{checkpoint_metric.replace('val_', '').replace('_', '-')}"
            if pooling_mode == "task_attention":
                job_name = f"{job_name}_attpool{attention_hidden_dim}"
            if primary_adapter_dim > 0:
                job_name = f"{job_name}_adp{primary_adapter_dim}"
            if expression_residual != "none":
                job_name = f"{job_name}_expradd"
            if ensemble_size > 1:
                job_name = f"{job_name}_ens{ensemble_size}"
            if getattr(args, "mask_aware_heads", "off") == "on":
                job_name = f"{job_name}_maskheads"
            if getattr(args, "gradient_strategy", "weighted_sum") == "primary_protected_pcgrad":
                job_name = f"{job_name}_gradppc"
            if ppi_integration == "gated_residual":
                job_name = f"{job_name}_ppigate"
            if effective_pca_target_count > 0:
                job_name = f"{job_name}_pcatarget{effective_pca_target_count}"
            if train_sampling == "balanced_classes":
                job_name = f"{job_name}_balsampler"
            jobs.append(
                GridJob(
                    job_name=job_name,
                    result_root=args.result_root / job_name,
                    embedding_source=embedding_source,
                    embed_file=embed_file,
                    embedding_gene_policy=embedding_policy,
                    gene_selection=gene_selection,
                    max_genes=max_genes,
                    augmentation=augmentation,
                    d_model=d_model,
                    d_ff=d_ff,
                    dropout=dropout,
                    ad_mci_gene_fraction=ad_mci_fraction,
                    encoder_sharing=encoder_sharing,
                    head_norm=head_norm,
                    embedding_rescale=embedding_rescale,
                    task_loss_weights=task_loss_weights,
                    checkpoint_metric=checkpoint_metric,
                    pooling_mode=pooling_mode,
                    attention_pooling_hidden_dim=attention_hidden_dim,
                    primary_adapter_dim=primary_adapter_dim,
                    expression_residual=expression_residual,
                    checkpoint_ensemble_size=ensemble_size,
                    ppi_integration=ppi_integration,
                    pca_neighbor_target_count=effective_pca_target_count,
                    train_sampling=train_sampling,
                )
            )
    if args.limit is not None:
        jobs = jobs[: args.limit]
    return jobs, skipped


def build_command(args: argparse.Namespace, job: GridJob) -> list[str]:
    cmd = [
        args.python_exe,
        "-u",
        str(RUNNER),
        "--x-file", str(args.x_file),
        "--y-file", str(args.y_file),
        "--result-root", str(job.result_root),
        "--run-mode", args.run_mode,
        "--architectures", *args.architectures,
        "--device", args.device,
        "--max-genes", str(job.max_genes),
        "--gene-selection", job.gene_selection,
        "--ad-mci-gene-fraction", str(job.ad_mci_gene_fraction),
        "--scaler", "minmax",
        "--batch-size", str(args.batch_size),
        "--train-sampling", job.train_sampling,
        "--epochs", str(args.epochs),
        "--early-stopping-patience", str(args.early_stopping_patience),
        "--val-loss-stop-patience", str(args.val_loss_stop_patience),
        "--lr", str(args.lr),
        "--freeze-embedding-epochs", str(getattr(args, "freeze_embedding_epochs", 0)),
        "--weight-decay", str(args.weight_decay),
        "--dropout", str(job.dropout),
        "--d-model", str(job.d_model),
        "--d-ff", str(job.d_ff),
        "--d-hidden1", "128",
        "--d-hidden2", "64",
        "--task-specific-pooling", getattr(args, "task_specific_pooling", "off"),
        "--head-norm", job.head_norm,
        "--mask-aware-heads", getattr(args, "mask_aware_heads", "off"),
        "--encoder-sharing", job.encoder_sharing,
        "--gradient-strategy", getattr(args, "gradient_strategy", "weighted_sum"),
        "--gradient-diagnostics", getattr(args, "gradient_diagnostics", "off"),
        "--gradient-diagnostic-interval", str(getattr(args, "gradient_diagnostic_interval", 1)),
        "--class-weighting", args.class_weighting,
        "--task-loss-weights", *[str(value) for value in job.task_loss_weights],
        "--pooling-mode", job.pooling_mode,
        "--attention-pooling-hidden-dim", str(job.attention_pooling_hidden_dim),
        "--attention-pooling-dropout", str(getattr(args, "attention_pooling_dropout", 0.1)),
        "--primary-adapter-dim", str(job.primary_adapter_dim),
        "--expression-residual", job.expression_residual,
        "--checkpoint-ensemble-size", str(job.checkpoint_ensemble_size),
        "--checkpoint-ensemble-min-gap", str(getattr(args, "checkpoint_ensemble_min_gap", 3)),
        "--augmentation", job.augmentation,
        "--smote-k-neighbors", str(args.smote_k_neighbors),
        "--smote-m-neighbors", str(args.smote_m_neighbors),
        "--smote-kind", args.smote_kind,
        "--pca-neighbor-components", str(getattr(args, "pca_neighbor_components", 50)),
        "--pca-neighbor-k", str(getattr(args, "pca_neighbor_k", 5)),
        "--pca-neighbor-gap-fraction", str(getattr(args, "pca_neighbor_gap_fraction", 0.5)),
        "--pca-neighbor-target-count", str(job.pca_neighbor_target_count),
        "--ctgan-epochs", str(getattr(args, "ctgan_epochs", 100)),
        "--ctgan-batch-size", str(getattr(args, "ctgan_batch_size", 128)),
        "--augmentation-target-multiplier", str(getattr(args, "augmentation_target_multiplier", 2.0)),
        "--gan-epochs", str(args.gan_epochs),
        "--gan-batch-size", str(args.gan_batch_size),
        "--gan-latent-dim", str(args.gan_latent_dim),
        "--gan-learning-rate", str(args.gan_learning_rate),
        "--gan-target-multiplier", str(args.gan_target_multiplier),
        "--gan-sampling-strategy", args.gan_sampling_strategy,
        "--checkpoint-metric", job.checkpoint_metric,
        "--evaluate-test-each-epoch", args.evaluate_test_each_epoch,
        "--embedding-gene-policy", job.embedding_gene_policy,
        "--embedding-rescale", job.embedding_rescale,
        "--embedding-init-scale", str(getattr(args, "embedding_init_scale", 0.02)),
        "--ppi-integration", job.ppi_integration,
        "--ppi-gate-init", str(getattr(args, "ppi_gate_init", 0.0)),
        "--seed-train-ratio", str(getattr(args, "seed_train_ratio", 0.70)),
        "--seed-val-ratio", str(getattr(args, "seed_val_ratio", 0.10)),
        "--seed-test-ratio", str(getattr(args, "seed_test_ratio", 0.20)),
        "--seeds", *[str(seed) for seed in args.seeds],
    ]
    if job.embed_file is not None:
        cmd.extend(["--embed-file", str(job.embed_file)])
    if getattr(args, "candidate_gene_file", None) is not None:
        cmd.extend(["--candidate-gene-file", str(args.candidate_gene_file)])
    if getattr(args, "lr_embedding", None) is not None:
        cmd.extend(["--lr-embedding", str(args.lr_embedding)])
    if args.val_loss_stop_threshold is not None:
        cmd.extend(["--val-loss-stop-threshold", str(args.val_loss_stop_threshold)])
    if args.skip_existing:
        cmd.append("--skip-existing")
    if args.smoke:
        cmd.append("--smoke")
    return cmd


def run_command(cmd: list[str], log_file: Path) -> int:
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
        return int(process.wait())


def write_grid_manifest(
    result_root: Path,
    jobs: list[GridJob],
    skipped: list[dict[str, Any]],
    commands: dict[str, list[str]],
) -> None:
    result_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for job in jobs:
        row = asdict(job)
        row["result_root"] = str(job.result_root)
        row["embed_file"] = str(job.embed_file) if job.embed_file is not None else ""
        row["status"] = "planned"
        row["command"] = " ".join(commands[job.job_name])
        rows.append(row)
    for row in skipped:
        rows.append({**row, "job_name": "", "result_root": "", "command": ""})
    pd.DataFrame(rows).to_csv(result_root / "grid_jobs.csv", index=False)


def validate_grid_inputs(args: argparse.Namespace, jobs: list[GridJob]) -> None:
    if args.dry_run:
        return
    if not jobs:
        raise ValueError(
            "The final grid has zero runnable jobs. If this is because PPI embeddings are missing, "
            "run experiments/scripts/pretraining/run_ppi_init.ps1 for the requested dimensions, "
            "or use --embedding-sources random for a random-init control."
        )
    missing_inputs = [path for path in [args.x_file, args.y_file] if not path.exists()]
    if missing_inputs:
        missing_text = ", ".join(str(path) for path in missing_inputs)
        raise FileNotFoundError(
            "Shared multitask dataset is missing: "
            f"{missing_text}. Build it with experiments/scripts/paper_comparison/build_txt_pairwise_datasets.py "
            "after creating task_dataset/processed/alzheimer_multiclass/X.csv and y.csv."
        )


def collect_ranked_summary(result_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for summary_path in sorted(result_root.glob("*/cv5/cv5_summary.csv")) + sorted(result_root.glob("*/seeds10/seeds10_summary.csv")):
        job_dir = summary_path.parents[1]
        summary = pd.read_csv(summary_path)
        group_columns = [column for column in ["evaluation", "arch", "split"] if column in summary.columns]
        if "split" not in group_columns:
            continue
        for group_key, split_df in summary.groupby(group_columns, dropna=False):
            if not isinstance(group_key, tuple):
                group_key = (group_key,)
            group_values = dict(zip(group_columns, group_key))
            split = str(group_values.get("split", ""))
            if split not in {"val", "test"}:
                continue
            if split_df.empty:
                continue
            primary = split_df[split_df["task"] == "AD_vs_MCI"]
            if primary.empty:
                continue
            primary_row = primary.iloc[0]
            auxiliary = split_df[split_df["task"] != "AD_vs_MCI"]
            evaluation = str(group_values.get("evaluation", primary_row.get("evaluation", summary_path.parent.name)))
            expected_match = re.fullmatch(r"(?:cv|seeds)(\d+)", evaluation)
            expected_runs = int(expected_match.group(1)) if expected_match else None
            observed_counts = pd.to_numeric(split_df.get("n_runs"), errors="coerce") if "n_runs" in split_df else None
            n_runs = int(observed_counts.min()) if observed_counts is not None and observed_counts.notna().any() else None
            is_complete = bool(n_runs >= expected_runs) if n_runs is not None and expected_runs is not None else True
            rows.append(
                {
                    "job_name": job_dir.name,
                    "evaluation": evaluation,
                    "arch": str(group_values.get("arch", primary_row.get("arch", ""))),
                    "split": split,
                    "primary_task": "AD_vs_MCI",
                    "primary_roc_auc_mean": primary_row.get("roc_auc_mean", float("nan")),
                    "primary_macro_f1_mean": primary_row.get("macro_f1_mean", float("nan")),
                    "primary_balanced_accuracy_mean": primary_row.get("balanced_accuracy_mean", float("nan")),
                    "auxiliary_roc_auc_mean": auxiliary["roc_auc_mean"].mean() if "roc_auc_mean" in auxiliary else float("nan"),
                    "auxiliary_macro_f1_mean": auxiliary["macro_f1_mean"].mean() if "macro_f1_mean" in auxiliary else float("nan"),
                    "n_runs": n_runs,
                    "expected_runs": expected_runs,
                    "is_complete": is_complete,
                    "summary_path": str(summary_path),
                }
            )
    ranked = pd.DataFrame(rows)
    if ranked.empty:
        return ranked
    ranked = ranked[ranked["is_complete"]].copy()
    ranked = ranked.sort_values(
        ["split", "primary_roc_auc_mean", "primary_macro_f1_mean", "auxiliary_roc_auc_mean", "job_name", "arch"],
        ascending=[True, False, False, False, True, True],
    )
    return ranked


def write_ranked_summary(result_root: Path) -> Path:
    result_root.mkdir(parents=True, exist_ok=True)
    ranked = collect_ranked_summary(result_root)
    if ranked.empty:
        ranked = pd.DataFrame(columns=RANKED_SUMMARY_COLUMNS)
    output_path = result_root / "grid_ranked_summary.csv"
    ranked.to_csv(output_path, index=False)
    return output_path


def main() -> None:
    args = parse_args()
    args.x_file = resolve(args.x_file)
    args.y_file = resolve(args.y_file)
    args.result_root = resolve(args.result_root)
    args.ppi_embedding_template = resolve(args.ppi_embedding_template)
    args.candidate_gene_file = resolve(args.candidate_gene_file) if args.candidate_gene_file is not None else None
    if args.collect_only:
        output_path = write_ranked_summary(args.result_root)
        print(f"Collected existing grid summaries: {output_path}", flush=True)
        return
    apply_grid_preset(args)
    args.result_root.mkdir(parents=True, exist_ok=True)

    jobs, skipped = build_jobs(args)
    commands = {job.job_name: build_command(args, job) for job in jobs}
    write_grid_manifest(args.result_root, jobs, skipped, commands)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    (args.result_root / "grid_config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    validate_grid_inputs(args, jobs)

    if args.dry_run:
        print(f"Dry run complete. Planned jobs: {len(jobs)}. Skipped jobs: {len(skipped)}. Manifest: {args.result_root / 'grid_jobs.csv'}")
        return

    failures = []
    for index, job in enumerate(jobs, start=1):
        print("\n" + "=" * 80, flush=True)
        print(f"Final multitask grid job {index}/{len(jobs)}: {job.job_name}", flush=True)
        print("=" * 80, flush=True)
        return_code = run_command(commands[job.job_name], job.result_root / "grid_job.log")
        if return_code != 0:
            failures.append({"job_name": job.job_name, "return_code": return_code})
            pd.DataFrame(failures).to_csv(args.result_root / "grid_failures.csv", index=False)
            raise RuntimeError(f"Grid job failed: {job.job_name}. See {job.result_root / 'grid_job.log'}")

    write_ranked_summary(args.result_root)
    print(f"Final multitask grid complete. Outputs: {args.result_root}", flush=True)


if __name__ == "__main__":
    main()
