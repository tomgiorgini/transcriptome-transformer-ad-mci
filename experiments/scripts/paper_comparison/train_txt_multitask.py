#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import math
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Sampler

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from source.models.txt import TxT
from source.pipeline.dataset import PreparedDataset, prepare_dataset, scale_arrays
from source.pipeline.reporting import (
    build_report_dataframe,
    binary_roc_auc_np,
    one_vs_rest_roc_auc,
    report_to_text,
    softmax_np,
)
from source.pipeline.utils import (
    DEFAULT_OFFICIAL_SPLIT_FILE,
    DEFAULT_X_FILE,
    DEFAULT_Y_FILE,
    compute_balanced_class_weights,
    namespace_to_dict,
    resolve_device,
    save_json,
    set_seed,
)


DEFAULT_RESULT_DIR = ROOT / "results" / "paper_comparison" / "txt_multitask" / "default_run"
DEFAULT_PPI_EDGE_FILE = ROOT / "pretraining_dataset" / "ppi_networks" / "hippie_highconf_edges.csv"
TASK_NAMES = ["AD_vs_MCI", "AD_vs_CTL", "MCI_vs_CTL"]


@dataclass(frozen=True)
class TaskSpec:
    name: str
    class_names: list[str]
    source_to_task_label: dict[int, int]


class MultitaskDataset(Dataset):
    def __init__(self, gene_x: np.ndarray, task_y: np.ndarray, task_mask: np.ndarray):
        self.gene_x = torch.tensor(gene_x, dtype=torch.float32)
        self.task_y = torch.tensor(task_y, dtype=torch.long)
        self.task_mask = torch.tensor(task_mask, dtype=torch.bool)

    def __len__(self) -> int:
        return int(self.gene_x.shape[0])

    def __getitem__(self, idx: int):
        return self.gene_x[idx], self.task_y[idx], self.task_mask[idx]


class BalancedClassBatchSampler(Sampler[list[int]]):
    """Yield equal biological-class counts per batch without synthetic profiles."""

    def __init__(self, labels: np.ndarray, batch_size: int, seed: int):
        labels = np.asarray(labels, dtype=np.int64)
        self.classes = np.unique(labels)
        if len(self.classes) < 2:
            raise ValueError("balanced_classes sampling requires at least two training classes.")
        if batch_size % len(self.classes) != 0:
            raise ValueError(
                f"--batch-size must be divisible by {len(self.classes)} for balanced_classes sampling."
            )
        self.indices_by_class = {
            int(label): np.flatnonzero(labels == label).astype(np.int64) for label in self.classes
        }
        if any(len(indices) == 0 for indices in self.indices_by_class.values()):
            raise ValueError("balanced_classes sampling requires at least one sample from every class.")
        self.batch_size = int(batch_size)
        self.samples_per_class = self.batch_size // len(self.classes)
        self.n_batches = len(labels) // self.batch_size
        if self.n_batches < 1:
            raise ValueError("Training split is smaller than --batch-size.")
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.n_batches

    def __iter__(self):
        rng = np.random.default_rng(self.seed + 1009 * self.epoch)
        orders = {label: rng.permutation(indices) for label, indices in self.indices_by_class.items()}
        cursors = {label: 0 for label in self.indices_by_class}

        def draw(label: int, count: int) -> list[int]:
            selected: list[int] = []
            while len(selected) < count:
                order = orders[label]
                cursor = cursors[label]
                available = len(order) - cursor
                take = min(count - len(selected), available)
                selected.extend(int(value) for value in order[cursor : cursor + take])
                cursors[label] += take
                if cursors[label] >= len(order):
                    orders[label] = rng.permutation(self.indices_by_class[label])
                    cursors[label] = 0
            return selected

        for _ in range(self.n_batches):
            batch: list[int] = []
            for label in self.classes:
                batch.extend(draw(int(label), self.samples_per_class))
            rng.shuffle(batch)
            yield batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TxT as a 3-head multitask classifier on AD/MCI/CTL pairwise tasks.")
    parser.add_argument("--x-file", type=Path, default=DEFAULT_X_FILE)
    parser.add_argument("--y-file", type=Path, default=DEFAULT_Y_FILE)
    parser.add_argument("--split-mode", choices=["official", "custom", "stratified", "random"], default="official")
    parser.add_argument("--split-file", type=Path, default=None, help="Used only when --split-mode=custom.")
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--post-model-construction-reseed",
        choices=["on", "off"],
        default="off",
        help=(
            "Reset all RNGs to --seed after model construction/device transfer. Enable for "
            "paired architectures that consume different RNG amounts while being constructed."
        ),
    )
    parser.add_argument("--split-seed", type=int, default=None)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--max-genes", type=int, default=512)
    parser.add_argument(
        "--gene-selection",
        choices=[
            "variance",
            "mad",
            "pairwise_anova_union",
            "ad_mci_priority_anova_union",
            "ad_mci_vs_ctl_anova_50_50",
            "task_deg_union",
        ],
        default="pairwise_anova_union",
        help=(
            "`variance` and `mad` use simple train-only unsupervised top-k selectors. "
            "`pairwise_anova_union` ranks genes separately for the three binary tasks on train only "
            "with an ANOVA F-score and round-robin merges them into one shared fixed gene set. "
            "`ad_mci_priority_anova_union` reserves a configurable share for AD_vs_MCI and splits "
            "the remainder across AD_vs_CTL and MCI_vs_CTL. "
            "`ad_mci_vs_ctl_anova_50_50` alternates train-only ANOVA genes from AD_vs_MCI "
            "and pooled AD+MCI_vs_Control. "
            "`task_deg_union` is kept as a backwards-compatible alias."
        ),
    )
    parser.add_argument(
        "--ad-mci-gene-fraction",
        type=float,
        default=0.5,
        help="Share of --max-genes reserved for AD_vs_MCI when using ad_mci_priority_anova_union.",
    )
    parser.add_argument("--scaler", choices=["none", "minmax", "standard"], default="minmax")
    parser.add_argument(
        "--feature-selection-fit-scope",
        choices=["train", "all"],
        default="train",
        help=(
            "Scope used for unsupervised variance/MAD ranking. "
            "Use all only for explicit leakage ablations, because it includes validation/test samples."
        ),
    )
    parser.add_argument(
        "--scaler-fit-scope",
        choices=["train", "all"],
        default="train",
        help=(
            "Scope used to fit the scaler. "
            "Use all only for explicit leakage ablations, because it includes validation/test samples."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--train-sampling",
        choices=["random", "balanced_classes"],
        default="random",
        help="Training batch construction; validation and test always retain their natural distributions.",
    )
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--early-stopping-patience", type=int, default=50)
    parser.add_argument(
        "--val-loss-stop-threshold",
        type=float,
        default=None,
        help="Optional extra early stop threshold. When set, count epochs with val_loss above this value.",
    )
    parser.add_argument(
        "--val-loss-stop-patience",
        type=int,
        default=0,
        help="Stop after this many consecutive epochs with val_loss above --val-loss-stop-threshold.",
    )
    parser.add_argument("--lr-encoder", type=float, default=3e-4)
    parser.add_argument("--lr-head", type=float, default=3e-4)
    parser.add_argument(
        "--lr-embedding",
        type=float,
        default=None,
        help="Embedding learning rate. Defaults to --lr-encoder when omitted.",
    )
    parser.add_argument(
        "--lr-volumetric",
        type=float,
        default=None,
        help=(
            "Learning rate for parameters owned by the VMA augmentation. Defaults to "
            "--lr-encoder and preserves the historical optimizer grouping when omitted."
        ),
    )
    parser.add_argument(
        "--freeze-embedding-epochs",
        type=int,
        default=0,
        help="Keep gene embeddings frozen for the first N epochs, then unfreeze them.",
    )
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--class-weighting", choices=["on", "off"], default="off")
    parser.add_argument(
        "--task-loss-weights",
        nargs=3,
        type=float,
        metavar=("AD_MCI", "AD_CTL", "MCI_CTL"),
        default=[1.0, 1.0, 1.0],
        help="Relative multitask loss weights in TASK_NAMES order.",
    )
    parser.add_argument(
        "--augmentation",
        choices=[
            "none", "smote", "borderline_smote", "pca_neighbor_mci_ctl",
            "pca_neighbor_all_tasks", "ctgan", "gan",
        ],
        default="none",
        help="Train-only augmentation after feature selection and scaling. Validation and test are never augmented.",
    )
    parser.add_argument("--smote-k-neighbors", type=int, default=5)
    parser.add_argument("--smote-m-neighbors", type=int, default=10)
    parser.add_argument("--smote-kind", choices=["borderline-1", "borderline-2"], default="borderline-1")
    parser.add_argument("--pca-neighbor-components", type=int, default=50)
    parser.add_argument("--pca-neighbor-k", type=int, default=5)
    parser.add_argument(
        "--pca-neighbor-gap-fraction",
        type=float,
        default=0.5,
        help="Fraction of the train MCI-vs-Control class-count gap filled by task-specific synthetic MCI profiles.",
    )
    parser.add_argument(
        "--pca-neighbor-target-count",
        type=int,
        default=0,
        help=(
            "For pca_neighbor_all_tasks, desired count per biological class; "
            "0 uses the largest original training class."
        ),
    )
    parser.add_argument("--ctgan-epochs", type=int, default=100)
    parser.add_argument("--ctgan-batch-size", type=int, default=128)
    parser.add_argument(
        "--augmentation-target-multiplier",
        type=float,
        default=2.0,
        help="For CTGAN, target augmented train size as multiplier * original train size.",
    )
    parser.add_argument("--gan-epochs", type=int, default=100)
    parser.add_argument("--gan-batch-size", type=int, default=64)
    parser.add_argument("--gan-latent-dim", type=int, default=64)
    parser.add_argument("--gan-learning-rate", type=float, default=0.001)
    parser.add_argument("--gan-target-multiplier", type=float, default=1.5)
    parser.add_argument("--gan-sampling-strategy", choices=["balanced", "minority", "proportional"], default="balanced")
    parser.add_argument("--label-smoothing", type=float, default=0.0)
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
    parser.add_argument("--evaluate-test", choices=["on", "off"], default="on")
    parser.add_argument(
        "--skip-final-test",
        action="store_true",
        help=(
            "Do not evaluate the held-out test split after training. This is an explicit alias for "
            "--evaluate-test off intended for validation-only model/beta selection runs."
        ),
    )
    evaluation_checkpoint_group = parser.add_mutually_exclusive_group()
    evaluation_checkpoint_group.add_argument(
        "--evaluation-only-checkpoint",
        type=Path,
        default=None,
        help=(
            "Load this model state dict, skip optimization entirely, and write test-split "
            "artifacts. The full validation split is scored only for ensemble metadata. All "
            "data-selection, architecture, embedding, and graph arguments must match the "
            "checkpoint run."
        ),
    )
    evaluation_checkpoint_group.add_argument(
        "--evaluation-only-checkpoints",
        type=Path,
        nargs="+",
        default=None,
        metavar="PATH",
        help=(
            "Load multiple model state dicts, skip optimization entirely, and evaluate their "
            "arithmetic probability ensemble. Test artifacts are written and the full validation "
            "split is scored for ensemble metadata. Paths are ranked in the supplied order: rank "
            "1 is also copied to best_model.pt and every member is copied to "
            "ensemble_checkpoint_rankN.pt."
        ),
    )
    parser.add_argument(
        "--evaluate-test-each-epoch",
        choices=["on", "off"],
        default="off",
        help=(
            "When on, evaluate the held-out test split at every epoch and write test metrics "
            "to training_log.csv. This is for checkpoint diagnostics only; checkpoint selection "
            "must still use validation metrics."
        ),
    )
    parser.add_argument("--device", choices=["cuda", "cpu", "mps"], default="cuda")
    parser.add_argument("--n-heads", type=int, default=2)
    parser.add_argument("--n-layers", type=int, default=1)
    parser.add_argument(
        "--model-variant",
        choices=["baseline", "ppi_volumetric"],
        default="baseline",
        help="Use the unchanged TxT baseline or its zero-gated HIPPIE volumetric-attention extension.",
    )
    parser.add_argument(
        "--ppi-edge-file",
        type=Path,
        default=DEFAULT_PPI_EDGE_FILE,
        help="HIPPIE edge CSV used only by --model-variant ppi_volumetric.",
    )
    parser.add_argument("--ppi-score-threshold", type=float, default=0.73)
    parser.add_argument("--volumetric-beta", type=float, default=1.0)
    parser.add_argument("--volumetric-eps", type=float, default=1e-8)
    parser.add_argument("--volumetric-gate-init", type=float, default=0.0)
    parser.add_argument(
        "--volumetric-volume-mode",
        choices=["raw", "l2"],
        default="raw",
        help="Use the legacy raw volume or its L2-normalized dimensionless form.",
    )
    parser.add_argument(
        "--volumetric-dropout",
        type=float,
        default=None,
        help=(
            "Dropout applied only to sparse volumetric attention weights. When omitted, "
            "inherit --dropout for backward-compatible behavior."
        ),
    )
    parser.add_argument(
        "--volumetric-message-mode",
        choices=["legacy", "expression_contrast"],
        default="legacy",
        help=(
            "VMA message construction. legacy preserves the historical static self/neighbor "
            "mixture; expression_contrast builds a patient-specific neighbor-minus-self message."
        ),
    )
    parser.add_argument(
        "--volumetric-output-norm",
        choices=["none", "rms"],
        default="none",
        help="Optional parameter-free normalization of the VMA message before residual gating.",
    )
    parser.add_argument(
        "--volumetric-gate-mode",
        choices=["scalar", "per_head"],
        default="scalar",
        help="Use one residual gate for the whole VMA branch or one gate per attention head.",
    )
    parser.add_argument(
        "--volumetric-backbone-gradient-mode",
        choices=["coupled", "detached"],
        default="coupled",
        help=(
            "Whether VMA gradients may flow into the shared Q/K/V/TUPE tensors. detached keeps "
            "the VMA input path adapter-like while the ordinary dense path remains trainable."
        ),
    )
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument(
        "--embed-file",
        type=Path,
        default=None,
        help=(
            "Optional external gene embedding CSV, for example HIPPIE/PPI `ppi_node_embedding.csv` "
            "or `gene_embedding.csv`. Rows are aligned to the selected train-fold gene set; "
            "missing genes receive deterministic random initialization."
        ),
    )
    parser.add_argument(
        "--embedding-gene-policy",
        choices=["all", "mapped_only"],
        default="all",
        help=(
            "When mapped_only and --embed-file is set, feature selection is restricted to expression genes "
            "present in the embedding file. Use this with PPI node embeddings to match the TxT paper setup."
        ),
    )
    parser.add_argument(
        "--candidate-gene-file",
        type=Path,
        default=None,
        help=(
            "Optional CSV whose first column defines the candidate gene universe independently of initialization. "
            "Use the same PPI gene file for random and PPI runs to isolate the embedding effect."
        ),
    )
    parser.add_argument(
        "--embedding-rescale",
        choices=["none", "global_std", "per_dimension_std"],
        default="none",
        help=(
            "Optional external-embedding rescaling. global_std preserves pairwise geometry up to one "
            "translation and one scalar; per_dimension_std standardizes each embedding dimension."
        ),
    )
    parser.add_argument(
        "--embedding-init-scale",
        type=float,
        default=0.02,
        help="Target standard deviation for random initialization and rescaled external embeddings.",
    )
    parser.add_argument(
        "--ppi-integration",
        choices=["direct", "gated_residual"],
        default="direct",
        help=(
            "direct replaces the random embedding with --embed-file; gated_residual keeps a trainable random "
            "embedding and adds a fixed row-normalized PPI prior through a zero-initialized scalar gate."
        ),
    )
    parser.add_argument("--ppi-gate-init", type=float, default=0.0)
    parser.add_argument("--d-ff", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--norm-first", action="store_true")
    parser.add_argument("--aggfunc", choices=["Flatten", "Avgpool"], default="Avgpool")
    parser.add_argument(
        "--task-specific-pooling",
        choices=["on", "off"],
        default="off",
        help=(
            "When on, the shared TxT encoder processes the union of selected genes, "
            "but each binary head pools only the genes selected for that task. "
            "With ad_mci_vs_ctl_anova_50_50, AD_vs_MCI pools the AD-vs-MCI genes, "
            "while AD_vs_CTL and MCI_vs_CTL pool the AD+MCI-vs-Control genes."
        ),
    )
    parser.add_argument(
        "--head-norm",
        choices=["batch", "layer", "none"],
        default="batch",
        help="Normalization inside each task-specific MLP head.",
    )
    parser.add_argument(
        "--mask-aware-heads",
        choices=["on", "off"],
        default="off",
        help="When on, each task head and its BatchNorm see only samples valid for that binary task.",
    )
    parser.add_argument(
        "--gradient-strategy",
        choices=["weighted_sum", "primary_protected_pcgrad"],
        default="weighted_sum",
        help="Optional gradient surgery on shared encoder parameters; AD_vs_MCI is the protected primary task.",
    )
    parser.add_argument(
        "--gradient-diagnostics",
        choices=["on", "off"],
        default="off",
        help="Log per-epoch task-gradient cosine similarities, norms, and conflict fractions.",
    )
    parser.add_argument("--gradient-diagnostic-interval", type=int, default=1)
    parser.add_argument(
        "--encoder-sharing",
        choices=["shared", "separate"],
        default="shared",
        help="Use one shared TxT encoder or one independent TxT encoder per binary task.",
    )
    parser.add_argument(
        "--pooling-mode",
        choices=["average", "task_attention"],
        default="average",
        help="Shared average pooling or a learned attention pooling query for each task.",
    )
    parser.add_argument("--attention-pooling-hidden-dim", type=int, default=16)
    parser.add_argument("--attention-pooling-dropout", type=float, default=0.1)
    parser.add_argument(
        "--primary-adapter-dim",
        type=int,
        default=0,
        help="Bottleneck dimension of a zero-initialized residual adapter used only by AD_vs_MCI; 0 disables it.",
    )
    parser.add_argument(
        "--expression-residual",
        choices=["none", "additive_zero_init"],
        default="none",
        help="Optional direct expression-to-token residual, initialized to reproduce the baseline exactly.",
    )
    parser.add_argument(
        "--tupe-mode",
        choices=["on", "off"],
        default="on",
        help=(
            "Enable the historical expression-derived TUPE attention bias or replace its "
            "forward contribution with exact zeros. The TUPE parameters remain checkpoint-compatible."
        ),
    )
    parser.add_argument("--checkpoint-ensemble-size", type=int, default=1)
    parser.add_argument("--checkpoint-ensemble-min-gap", type=int, default=3)
    parser.add_argument("--d-hidden1", type=int, default=128)
    parser.add_argument("--d-hidden2", type=int, default=64)
    parser.add_argument("--slope", type=float, default=0.2)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument(
        "--grad-clip-scope",
        choices=["joint", "separate_volumetric"],
        default="joint",
        help=(
            "Clip all gradients jointly (legacy) or clip VMA-module and non-VMA gradients "
            "as two independent groups."
        ),
    )
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    return parser.parse_args()


def resolve_split_file(args: argparse.Namespace) -> Path | None:
    if args.split_mode == "official":
        return DEFAULT_OFFICIAL_SPLIT_FILE
    if args.split_mode == "custom":
        if args.split_file is None:
            raise ValueError("`--split-file` is required when `--split-mode=custom`.")
        return args.split_file
    return None


def normalize_class_name(value: str) -> str:
    return value.strip().lower().replace(" ", "").replace("_", "")


def resolve_class_indices(class_names: list[str]) -> dict[str, int]:
    normalized = {normalize_class_name(name): idx for idx, name in enumerate(class_names)}
    aliases = {
        "control": ["control", "ctl", "nc", "cn"],
        "mci": ["mci"],
        "ad": ["ad", "dementia", "alzheimer"],
    }
    resolved: dict[str, int] = {}
    for canonical, options in aliases.items():
        for option in options:
            if option in normalized:
                resolved[canonical] = normalized[option]
                break
    missing = [name for name in aliases if name not in resolved]
    if missing:
        raise ValueError(f"Unable to resolve required classes {missing} from class_names={class_names}.")
    return resolved


def build_task_specs(class_names: list[str]) -> list[TaskSpec]:
    class_idx = resolve_class_indices(class_names)
    ctl = class_idx["control"]
    mci = class_idx["mci"]
    ad = class_idx["ad"]
    return [
        TaskSpec("AD_vs_MCI", ["MCI", "AD"], {mci: 0, ad: 1}),
        TaskSpec("AD_vs_CTL", ["Control", "AD"], {ctl: 0, ad: 1}),
        TaskSpec("MCI_vs_CTL", ["Control", "MCI"], {ctl: 0, mci: 1}),
    ]


def make_task_targets(y: np.ndarray, task_specs: list[TaskSpec]) -> tuple[np.ndarray, np.ndarray]:
    labels = np.full((len(y), len(task_specs)), -1, dtype=np.int64)
    mask = np.zeros((len(y), len(task_specs)), dtype=bool)
    for task_idx, spec in enumerate(task_specs):
        for source_label, task_label in spec.source_to_task_label.items():
            keep = y == source_label
            labels[keep, task_idx] = task_label
            mask[keep, task_idx] = True
    return labels, mask


PCA_MCI_CTL_SYNTHETIC_PREFIX = "synthetic_pca_mci_ctl_"


def make_task_targets_for_samples(
    y: np.ndarray,
    task_specs: list[TaskSpec],
    sample_ids: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Create task labels/masks and apply explicit task scope for targeted synthetics."""
    labels, mask = make_task_targets(y, task_specs)
    if sample_ids is None:
        return labels, mask
    synthetic = np.asarray(
        [str(sample_id).startswith(PCA_MCI_CTL_SYNTHETIC_PREFIX) for sample_id in sample_ids],
        dtype=bool,
    )
    if bool(synthetic.any()):
        mci_ctl_idx = next(idx for idx, spec in enumerate(task_specs) if spec.name == "MCI_vs_CTL")
        mask[synthetic, :] = False
        mask[synthetic, mci_ctl_idx] = True
    return labels, mask


def build_embedding_dataframe(gene_names: list[str], embed_dim: int, seed: int, init_scale: float = 0.02) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    values = rng.normal(0.0, init_scale, size=(len(gene_names), embed_dim)).astype(np.float32)
    embed_df = pd.DataFrame(values, index=gene_names)
    embed_df.index.name = "Gene"
    return embed_df


def normalize_embedding_gene_key(value: object) -> str:
    return str(value).strip().upper()


def load_expression_genes(x_file: Path) -> list[str]:
    x_header = pd.read_csv(x_file, nrows=0)
    return [column for column in x_header.columns.astype(str).tolist() if column != "sample_id"]


def load_embedding_gene_keys(embed_file: Path) -> set[str]:
    if not embed_file.exists():
        raise FileNotFoundError(f"Embedding file not found: {embed_file}")
    genes = pd.read_csv(embed_file, usecols=[0]).iloc[:, 0].astype(str).str.strip()
    return {normalize_embedding_gene_key(gene) for gene in genes if gene}


def resolve_candidate_genes(args: argparse.Namespace) -> tuple[list[str] | None, dict[str, Any]]:
    manifest = {
        "embedding_gene_policy": args.embedding_gene_policy,
        "candidate_gene_filter_applied": False,
        "candidate_gene_source": None,
    }
    candidate_source = getattr(args, "candidate_gene_file", None)
    if candidate_source is not None:
        candidate_source = candidate_source if candidate_source.is_absolute() else (ROOT / candidate_source).resolve()
    elif args.embedding_gene_policy == "mapped_only":
        if args.embed_file is None:
            raise ValueError("--embedding-gene-policy mapped_only requires --embed-file or --candidate-gene-file.")
        candidate_source = args.embed_file

    if candidate_source is None:
        return None, manifest
    if not candidate_source.exists():
        raise FileNotFoundError(f"Candidate gene file not found: {candidate_source}")

    expression_genes = load_expression_genes(args.x_file)
    embedding_gene_keys = load_embedding_gene_keys(candidate_source)
    candidate_genes = [
        gene
        for gene in expression_genes
        if normalize_embedding_gene_key(gene) in embedding_gene_keys
    ]
    if not candidate_genes:
        raise ValueError(
            "No expression genes overlap the embedding file under mapped_only policy. "
            "Check gene symbols and whether the embedding file is a raw PPI node embedding."
        )
    manifest.update(
        {
            "candidate_gene_filter_applied": True,
            "candidate_gene_source": str(candidate_source),
            "expression_genes_before_filter": int(len(expression_genes)),
            "embedding_source_genes": int(len(embedding_gene_keys)),
            "expression_genes_after_filter": int(len(candidate_genes)),
            "candidate_gene_coverage": float(len(candidate_genes) / max(len(expression_genes), 1)),
        }
    )
    return candidate_genes, manifest


def build_embedding_for_selected_genes(
    gene_names: list[str],
    embed_dim: int,
    seed: int,
    embed_file: Path | None = None,
    init_scale: float = 0.02,
    rescale: str = "none",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if init_scale <= 0:
        raise ValueError("embedding init scale must be positive.")
    if rescale not in {"none", "global_std", "per_dimension_std"}:
        raise ValueError(f"Unsupported embedding rescale mode: {rescale}")
    if embed_file is None:
        embedding_df = build_embedding_dataframe(gene_names, embed_dim, seed, init_scale)
        return embedding_df, {
            "embedding_source": "random",
            "source_file": None,
            "embedding_dim": int(embed_dim),
            "target_genes": int(len(gene_names)),
            "matched_genes": 0,
            "missing_genes": int(len(gene_names)),
            "coverage": 0.0,
            "missing_gene_names": list(gene_names),
            "rescale": "not_applicable",
            "init_scale": float(init_scale),
            "initial_mean": float(embedding_df.to_numpy().mean()),
            "initial_std": float(embedding_df.to_numpy().std()),
        }

    if not embed_file.exists():
        raise FileNotFoundError(f"Embedding file not found: {embed_file}")

    source_df = pd.read_csv(embed_file, index_col=0)
    if source_df.empty:
        raise ValueError(f"Embedding file is empty: {embed_file}")
    source_df = source_df.copy()
    source_df.index = source_df.index.astype(str).str.strip()
    source_df = source_df.loc[~source_df.index.duplicated(keep="first")]
    source_df = source_df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    source_dim = int(source_df.shape[1])
    if source_dim != int(embed_dim):
        raise ValueError(
            f"Embedding dimension mismatch: file has {source_dim} columns but --embed-dim is {embed_dim}."
        )

    raw_values = source_df.to_numpy(dtype=np.float64, copy=True)
    raw_mean = float(raw_values.mean())
    raw_std = float(raw_values.std())
    if rescale == "global_std":
        # Translation plus one global scalar preserves all pairwise distances up to scale.
        centered = raw_values - raw_values.mean(axis=0, keepdims=True)
        centered_std = float(centered.std())
        if centered_std <= 0:
            raise ValueError(f"Cannot globally rescale constant embedding file: {embed_file}")
        source_df.iloc[:, :] = centered * (init_scale / centered_std)
    elif rescale == "per_dimension_std":
        centered = raw_values - raw_values.mean(axis=0, keepdims=True)
        dimension_std = centered.std(axis=0, keepdims=True)
        dimension_std = np.where(dimension_std <= 0, 1.0, dimension_std)
        source_df.iloc[:, :] = centered * (init_scale / dimension_std)

    normalized_to_index: dict[str, str | None] = {}
    for index_value in source_df.index:
        key = normalize_embedding_gene_key(index_value)
        if key in normalized_to_index:
            normalized_to_index[key] = None
        else:
            normalized_to_index[key] = index_value

    rng = np.random.default_rng(seed)
    rows: list[np.ndarray] = []
    missing: list[str] = []
    matched_exact = 0
    matched_normalized = 0
    for gene in gene_names:
        if gene in source_df.index:
            rows.append(source_df.loc[gene].to_numpy(dtype=np.float32, copy=True))
            matched_exact += 1
            continue
        normalized_index = normalized_to_index.get(normalize_embedding_gene_key(gene))
        if normalized_index is not None:
            rows.append(source_df.loc[normalized_index].to_numpy(dtype=np.float32, copy=True))
            matched_normalized += 1
            continue
        missing.append(gene)
        rows.append(rng.normal(0.0, init_scale, size=embed_dim).astype(np.float32))

    embedding_df = pd.DataFrame(np.vstack(rows), index=gene_names)
    embedding_df.index.name = "Gene"
    matched_genes = matched_exact + matched_normalized
    return embedding_df, {
        "embedding_source": "external_file",
        "source_file": str(embed_file),
        "embedding_dim": int(embed_dim),
        "source_genes": int(source_df.shape[0]),
        "target_genes": int(len(gene_names)),
        "matched_genes": int(matched_genes),
        "matched_exact": int(matched_exact),
        "matched_normalized": int(matched_normalized),
        "missing_genes": int(len(missing)),
        "coverage": float(matched_genes / max(len(gene_names), 1)),
        "missing_gene_names": missing,
        "missing_policy": "deterministic_random_normal_init",
        "rescale": rescale,
        "init_scale": float(init_scale),
        "source_raw_mean": raw_mean,
        "source_raw_std": raw_std,
        "source_transformed_mean": float(source_df.to_numpy().mean()),
        "source_transformed_std": float(source_df.to_numpy().std()),
        "selected_initial_mean": float(embedding_df.to_numpy().mean()),
        "selected_initial_std": float(embedding_df.to_numpy().std()),
    }


def build_row_normalized_ppi_prior(
    gene_names: list[str],
    embed_dim: int,
    embed_file: Path,
    init_scale: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build a fixed PPI prior with equal row norms and zeros for unmapped genes."""
    if not embed_file.exists():
        raise FileNotFoundError(f"PPI prior file not found: {embed_file}")
    source_df = pd.read_csv(embed_file, index_col=0)
    source_df.index = source_df.index.astype(str).str.strip()
    source_df = source_df.loc[~source_df.index.duplicated(keep="first")]
    source_df = source_df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    if int(source_df.shape[1]) != int(embed_dim):
        raise ValueError(
            f"PPI prior dimension mismatch: file has {source_df.shape[1]} columns but --embed-dim is {embed_dim}."
        )
    raw_values = source_df.to_numpy(dtype=np.float64, copy=True)
    centered_values = raw_values - raw_values.mean(axis=0, keepdims=True)
    source_df.iloc[:, :] = centered_values
    normalized_to_index: dict[str, str | None] = {}
    for index_value in source_df.index:
        key = normalize_embedding_gene_key(index_value)
        normalized_to_index[key] = index_value if key not in normalized_to_index else None

    target_norm = float(math.sqrt(embed_dim) * init_scale)
    rows: list[np.ndarray] = []
    missing: list[str] = []
    matched_exact = 0
    matched_normalized = 0
    for gene in gene_names:
        source_index: str | None
        if gene in source_df.index:
            source_index = gene
            matched_exact += 1
        else:
            source_index = normalized_to_index.get(normalize_embedding_gene_key(gene))
            if source_index is not None:
                matched_normalized += 1
        if source_index is None:
            rows.append(np.zeros(embed_dim, dtype=np.float32))
            missing.append(gene)
            continue
        row = source_df.loc[source_index].to_numpy(dtype=np.float64, copy=True)
        row_norm = float(np.linalg.norm(row))
        if row_norm <= 0:
            rows.append(np.zeros(embed_dim, dtype=np.float32))
            missing.append(gene)
            continue
        rows.append((row * (target_norm / row_norm)).astype(np.float32))

    prior_df = pd.DataFrame(np.vstack(rows), index=gene_names)
    prior_df.index.name = "Gene"
    matched = len(gene_names) - len(missing)
    nonzero = prior_df.to_numpy()[np.linalg.norm(prior_df.to_numpy(), axis=1) > 0]
    return prior_df, {
        "integration": "gated_residual",
        "source_file": str(embed_file),
        "source_genes": int(source_df.shape[0]),
        "embedding_dim": int(embed_dim),
        "target_genes": int(len(gene_names)),
        "matched_genes": int(matched),
        "matched_exact": int(matched_exact),
        "matched_normalized": int(matched_normalized),
        "missing_genes": int(len(missing)),
        "missing_gene_names": missing,
        "coverage": float(matched / max(len(gene_names), 1)),
        "centering": "per_dimension_source_mean",
        "normalization": "per_gene_l2",
        "target_row_l2_norm": target_norm,
        "nonzero_prior_std": float(nonzero.std()) if len(nonzero) else 0.0,
    }


def subset_prepared_dataset(dataset: PreparedDataset, keep_indices: np.ndarray) -> PreparedDataset:
    keep_indices = np.asarray(keep_indices, dtype=np.int64)
    return PreparedDataset(
        class_names=dataset.class_names,
        gene_names=[dataset.gene_names[int(idx)] for idx in keep_indices],
        train_ids=dataset.train_ids,
        val_ids=dataset.val_ids,
        test_ids=dataset.test_ids,
        train_gene_x=dataset.train_gene_x[:, keep_indices].astype(np.float32),
        val_gene_x=dataset.val_gene_x[:, keep_indices].astype(np.float32),
        test_gene_x=dataset.test_gene_x[:, keep_indices].astype(np.float32),
        train_y=dataset.train_y,
        val_y=dataset.val_y,
        test_y=dataset.test_y,
    )


def binary_anova_f_scores(train_x: np.ndarray, train_y: np.ndarray, class_a: int, class_b: int) -> np.ndarray:
    return binary_group_anova_f_scores(train_x, train_y == class_a, train_y == class_b, f"{class_a}", f"{class_b}")


def binary_group_anova_f_scores(
    train_x: np.ndarray,
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    name_a: str,
    name_b: str,
) -> np.ndarray:
    group_a = train_x[mask_a]
    group_b = train_x[mask_b]
    if len(group_a) == 0 or len(group_b) == 0:
        raise ValueError(f"Cannot rank genes: missing group pair {name_a}, {name_b} in train split.")
    n_a = len(group_a)
    n_b = len(group_b)
    mean_a = np.nanmean(group_a, axis=0)
    mean_b = np.nanmean(group_b, axis=0)
    overall_mean = np.nanmean(np.vstack([group_a, group_b]), axis=0)
    ss_between = n_a * np.square(mean_a - overall_mean) + n_b * np.square(mean_b - overall_mean)
    ss_within = np.nansum(np.square(group_a - mean_a), axis=0) + np.nansum(np.square(group_b - mean_b), axis=0)
    df_between = 1.0
    df_within = max(n_a + n_b - 2, 1)
    ms_between = ss_between / df_between
    ms_within = ss_within / df_within
    scores = np.divide(
        ms_between,
        ms_within,
        out=np.zeros_like(mean_a, dtype=np.float64),
        where=ms_within > 0,
    )
    return np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)


def simple_gene_scores(train_x: np.ndarray, method: str) -> np.ndarray:
    if method == "variance":
        scores = np.nanvar(train_x, axis=0)
    elif method == "mad":
        medians = np.nanmedian(train_x, axis=0)
        scores = np.nanmedian(np.abs(train_x - medians), axis=0)
    else:
        raise ValueError(f"Unsupported simple gene selection method: {method}")
    return np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)


def select_simple_gene_ranking(
    dataset: PreparedDataset,
    max_genes: int,
    method: str,
    fit_scope: str = "train",
) -> tuple[PreparedDataset, pd.DataFrame]:
    if fit_scope == "train":
        score_x = dataset.train_gene_x
    elif fit_scope == "all":
        score_x = np.concatenate([dataset.train_gene_x, dataset.val_gene_x, dataset.test_gene_x], axis=0)
    else:
        raise ValueError(f"Unsupported feature selection fit scope: {fit_scope}")
    scores = simple_gene_scores(score_x, method)
    order = np.lexsort((np.arange(len(scores)), -scores))
    if max_genes <= 0 or dataset.train_gene_x.shape[1] <= max_genes:
        keep_indices = order.astype(np.int64)
    else:
        keep_indices = order[:max_genes].astype(np.int64)
    details = pd.DataFrame(
        [
            {
                "gene": dataset.gene_names[int(gene_idx)],
                "selection_order": order_idx,
                "selected_by_task": f"{fit_scope}_{method}",
                "selection_method": method,
                "fit_scope": fit_scope,
                "leakage_ablation": bool(fit_scope == "all"),
                "selection_score": float(scores[int(gene_idx)]),
            }
            for order_idx, gene_idx in enumerate(keep_indices, start=1)
        ]
    )
    return subset_prepared_dataset(dataset, keep_indices), details


def select_pairwise_anova_union(
    dataset: PreparedDataset,
    task_specs: list[TaskSpec],
    max_genes: int,
) -> tuple[PreparedDataset, pd.DataFrame]:
    if max_genes <= 0 or dataset.train_gene_x.shape[1] <= max_genes:
        details = pd.DataFrame(
            {
                "gene": dataset.gene_names,
                "selection_order": np.arange(1, len(dataset.gene_names) + 1),
                "selected_by_task": "all_genes",
                "selection_method": "all_genes",
            }
        )
        return dataset, details

    binary_specs = [spec for spec in task_specs if len(spec.class_names) == 2]
    rankings: dict[str, list[int]] = {}
    score_by_task: dict[str, np.ndarray] = {}
    for spec in binary_specs:
        source_labels = list(spec.source_to_task_label.keys())
        scores = binary_anova_f_scores(dataset.train_gene_x, dataset.train_y, source_labels[0], source_labels[1])
        ranking = np.argsort(scores)[::-1].tolist()
        rankings[spec.name] = ranking
        score_by_task[spec.name] = scores

    selected: list[int] = []
    selected_set: set[int] = set()
    selected_by: dict[int, str] = {}
    rank_positions = {spec.name: 0 for spec in binary_specs}

    while len(selected) < max_genes and len(selected_set) < dataset.train_gene_x.shape[1]:
        progressed = False
        for spec in binary_specs:
            ranking = rankings[spec.name]
            pos = rank_positions[spec.name]
            while pos < len(ranking) and ranking[pos] in selected_set:
                pos += 1
            rank_positions[spec.name] = pos + 1
            if pos >= len(ranking):
                continue
            gene_idx = int(ranking[pos])
            selected.append(gene_idx)
            selected_set.add(gene_idx)
            selected_by[gene_idx] = spec.name
            progressed = True
            if len(selected) >= max_genes:
                break
        if not progressed:
            break

    keep_indices = np.asarray(selected, dtype=np.int64)
    details_rows = []
    for order, gene_idx in enumerate(keep_indices, start=1):
        row = {
            "gene": dataset.gene_names[int(gene_idx)],
            "selection_order": order,
            "selected_by_task": selected_by[int(gene_idx)],
            "selection_method": "pairwise_anova_union",
        }
        for task_name, scores in score_by_task.items():
            row[f"{task_name}_anova_f"] = float(scores[int(gene_idx)])
        details_rows.append(row)

    return subset_prepared_dataset(dataset, keep_indices), pd.DataFrame(details_rows)


def select_ad_mci_vs_ctl_anova_50_50(
    dataset: PreparedDataset,
    task_specs: list[TaskSpec],
    max_genes: int,
) -> tuple[PreparedDataset, pd.DataFrame]:
    if max_genes <= 0 or dataset.train_gene_x.shape[1] <= max_genes:
        details = pd.DataFrame(
            {
                "gene": dataset.gene_names,
                "selection_order": np.arange(1, len(dataset.gene_names) + 1),
                "selected_by_task": "all_genes",
                "selection_method": "all_genes",
            }
        )
        return dataset, details

    class_idx = resolve_class_indices(dataset.class_names)
    ctl = class_idx["control"]
    mci = class_idx["mci"]
    ad = class_idx["ad"]
    score_by_task = {
        "AD_vs_MCI": binary_anova_f_scores(dataset.train_gene_x, dataset.train_y, ad, mci),
        "AD_MCI_vs_CTL": binary_group_anova_f_scores(
            dataset.train_gene_x,
            np.isin(dataset.train_y, [ad, mci]),
            dataset.train_y == ctl,
            "AD_MCI",
            "Control",
        ),
    }
    rankings = {task_name: np.argsort(scores)[::-1].tolist() for task_name, scores in score_by_task.items()}
    task_order = ["AD_vs_MCI", "AD_MCI_vs_CTL"]
    rank_positions = {task_name: 0 for task_name in task_order}
    selected: list[int] = []
    selected_set: set[int] = set()
    selected_by: dict[int, str] = {}

    while len(selected) < max_genes and len(selected_set) < dataset.train_gene_x.shape[1]:
        progressed = False
        for task_name in task_order:
            ranking = rankings[task_name]
            pos = rank_positions[task_name]
            while pos < len(ranking) and ranking[pos] in selected_set:
                pos += 1
            rank_positions[task_name] = pos + 1
            if pos >= len(ranking):
                continue
            gene_idx = int(ranking[pos])
            selected.append(gene_idx)
            selected_set.add(gene_idx)
            selected_by[gene_idx] = task_name
            progressed = True
            if len(selected) >= max_genes:
                break
        if not progressed:
            break

    keep_indices = np.asarray(selected, dtype=np.int64)
    details_rows = []
    for order, gene_idx in enumerate(keep_indices, start=1):
        row = {
            "gene": dataset.gene_names[int(gene_idx)],
            "selection_order": order,
            "selected_by_task": selected_by[int(gene_idx)],
            "selection_method": "ad_mci_vs_ctl_anova_50_50",
        }
        for task_name, scores in score_by_task.items():
            row[f"{task_name}_anova_f"] = float(scores[int(gene_idx)])
        details_rows.append(row)

    return subset_prepared_dataset(dataset, keep_indices), pd.DataFrame(details_rows)


def split_gene_budget(total_genes: int, task_count: int) -> list[int]:
    if total_genes <= 0:
        raise ValueError("total_genes must be positive.")
    base = total_genes // task_count
    remainder = total_genes % task_count
    return [base + (1 if idx < remainder else 0) for idx in range(task_count)]


def select_weighted_pairwise_anova_union(
    dataset: PreparedDataset,
    task_specs: list[TaskSpec],
    max_genes: int,
    ad_mci_gene_fraction: float,
) -> tuple[PreparedDataset, pd.DataFrame]:
    if max_genes <= 0 or dataset.train_gene_x.shape[1] <= max_genes:
        details = pd.DataFrame(
            {
                "gene": dataset.gene_names,
                "selection_order": np.arange(1, len(dataset.gene_names) + 1),
                "selected_by_task": "all_genes",
                "selection_method": "all_genes",
            }
        )
        return dataset, details
    if not 0.0 < ad_mci_gene_fraction < 1.0:
        raise ValueError("--ad-mci-gene-fraction must be between 0 and 1 for ad_mci_priority_anova_union.")

    spec_by_name = {spec.name: spec for spec in task_specs if len(spec.class_names) == 2}
    task_order = ["AD_vs_MCI", "AD_vs_CTL", "MCI_vs_CTL"]
    missing = [task_name for task_name in task_order if task_name not in spec_by_name]
    if missing:
        raise ValueError(f"Missing binary tasks for weighted selection: {missing}")

    primary_quota = max(1, min(max_genes, int(math.ceil(max_genes * ad_mci_gene_fraction))))
    auxiliary_quotas = split_gene_budget(max_genes - primary_quota, 2) if max_genes > primary_quota else [0, 0]
    quotas = {
        "AD_vs_MCI": primary_quota,
        "AD_vs_CTL": auxiliary_quotas[0],
        "MCI_vs_CTL": auxiliary_quotas[1],
    }
    rankings: dict[str, list[int]] = {}
    score_by_task: dict[str, np.ndarray] = {}
    for task_name in task_order:
        source_labels = list(spec_by_name[task_name].source_to_task_label.keys())
        scores = binary_anova_f_scores(dataset.train_gene_x, dataset.train_y, source_labels[0], source_labels[1])
        rankings[task_name] = np.argsort(scores)[::-1].tolist()
        score_by_task[task_name] = scores

    selected: list[int] = []
    selected_set: set[int] = set()
    selected_by: dict[int, str] = {}
    rank_positions = {task_name: 0 for task_name in task_order}
    remaining = quotas.copy()

    while len(selected) < max_genes and any(value > 0 for value in remaining.values()):
        progressed = False
        for task_name in task_order:
            if remaining[task_name] <= 0:
                continue
            ranking = rankings[task_name]
            pos = rank_positions[task_name]
            while pos < len(ranking) and ranking[pos] in selected_set:
                pos += 1
            rank_positions[task_name] = pos + 1
            if pos >= len(ranking):
                remaining[task_name] = 0
                continue
            gene_idx = int(ranking[pos])
            selected.append(gene_idx)
            selected_set.add(gene_idx)
            selected_by[gene_idx] = task_name
            remaining[task_name] -= 1
            progressed = True
            if len(selected) >= max_genes:
                break
        if not progressed:
            break

    while len(selected) < max_genes and len(selected_set) < dataset.train_gene_x.shape[1]:
        progressed = False
        for task_name in task_order:
            ranking = rankings[task_name]
            pos = rank_positions[task_name]
            while pos < len(ranking) and ranking[pos] in selected_set:
                pos += 1
            rank_positions[task_name] = pos + 1
            if pos >= len(ranking):
                continue
            gene_idx = int(ranking[pos])
            selected.append(gene_idx)
            selected_set.add(gene_idx)
            selected_by[gene_idx] = f"{task_name}_quota_fill"
            progressed = True
            if len(selected) >= max_genes:
                break
        if not progressed:
            break

    keep_indices = np.asarray(selected, dtype=np.int64)
    details_rows = []
    for order, gene_idx in enumerate(keep_indices, start=1):
        row = {
            "gene": dataset.gene_names[int(gene_idx)],
            "selection_order": order,
            "selected_by_task": selected_by[int(gene_idx)],
            "selection_method": "ad_mci_priority_anova_union",
            "AD_vs_MCI_gene_fraction": float(ad_mci_gene_fraction),
        }
        for task_name, quota in quotas.items():
            row[f"{task_name}_quota"] = int(quota)
        for task_name, scores in score_by_task.items():
            row[f"{task_name}_anova_f"] = float(scores[int(gene_idx)])
        details_rows.append(row)

    return subset_prepared_dataset(dataset, keep_indices), pd.DataFrame(details_rows)


def select_pairwise_anova_task_specific_pooling(
    dataset: PreparedDataset,
    task_specs: list[TaskSpec],
    max_genes: int,
) -> tuple[PreparedDataset, pd.DataFrame, dict[str, list[int]]]:
    if max_genes <= 0:
        raise ValueError("task-specific pooling requires --max-genes > 0.")
    binary_specs = [spec for spec in task_specs if len(spec.class_names) == 2]
    quotas = split_gene_budget(max_genes, len(binary_specs))
    rankings: dict[str, list[int]] = {}
    score_by_task: dict[str, np.ndarray] = {}
    selected_original_by_task: dict[str, list[int]] = {}

    for spec, quota in zip(binary_specs, quotas):
        source_labels = list(spec.source_to_task_label.keys())
        scores = binary_anova_f_scores(dataset.train_gene_x, dataset.train_y, source_labels[0], source_labels[1])
        ranking = np.argsort(scores)[::-1].tolist()
        task_selected = ranking[: min(quota, len(ranking))]
        rankings[spec.name] = ranking
        score_by_task[spec.name] = scores
        selected_original_by_task[spec.name] = [int(idx) for idx in task_selected]

    union_original: list[int] = []
    union_set: set[int] = set()
    max_quota = max(len(values) for values in selected_original_by_task.values())
    for pos in range(max_quota):
        for spec in binary_specs:
            task_values = selected_original_by_task[spec.name]
            if pos >= len(task_values):
                continue
            gene_idx = task_values[pos]
            if gene_idx not in union_set:
                union_original.append(gene_idx)
                union_set.add(gene_idx)

    original_to_union = {gene_idx: union_idx for union_idx, gene_idx in enumerate(union_original)}
    task_gene_indices = {
        task_name: [original_to_union[gene_idx] for gene_idx in gene_indices if gene_idx in original_to_union]
        for task_name, gene_indices in selected_original_by_task.items()
    }

    details_rows = []
    for union_order, gene_idx in enumerate(union_original, start=1):
        selected_for_tasks = [task_name for task_name, values in selected_original_by_task.items() if gene_idx in set(values)]
        row = {
            "gene": dataset.gene_names[int(gene_idx)],
            "selection_order": union_order,
            "selected_by_task": ";".join(selected_for_tasks),
            "selection_method": "pairwise_anova_task_specific_pooling",
        }
        for task_name, scores in score_by_task.items():
            row[f"{task_name}_anova_f"] = float(scores[int(gene_idx)])
            row[f"{task_name}_in_pool"] = task_name in selected_for_tasks
        details_rows.append(row)

    return subset_prepared_dataset(dataset, np.asarray(union_original, dtype=np.int64)), pd.DataFrame(details_rows), task_gene_indices


def select_ad_mci_vs_ctl_anova_task_specific_pooling(
    dataset: PreparedDataset,
    max_genes: int,
) -> tuple[PreparedDataset, pd.DataFrame, dict[str, list[int]]]:
    if max_genes <= 0:
        raise ValueError("task-specific pooling requires --max-genes > 0.")

    class_idx = resolve_class_indices(dataset.class_names)
    ctl = class_idx["control"]
    mci = class_idx["mci"]
    ad = class_idx["ad"]
    score_by_pool = {
        "AD_vs_MCI": binary_anova_f_scores(dataset.train_gene_x, dataset.train_y, ad, mci),
        "AD_MCI_vs_CTL": binary_group_anova_f_scores(
            dataset.train_gene_x,
            np.isin(dataset.train_y, [ad, mci]),
            dataset.train_y == ctl,
            "AD_MCI",
            "Control",
        ),
    }
    rankings = {pool_name: np.argsort(scores)[::-1].tolist() for pool_name, scores in score_by_pool.items()}
    quotas = split_gene_budget(max_genes, 2)
    selected_original_by_pool = {
        "AD_vs_MCI": [int(idx) for idx in rankings["AD_vs_MCI"][: min(quotas[0], len(rankings["AD_vs_MCI"]))]],
        "AD_MCI_vs_CTL": [
            int(idx) for idx in rankings["AD_MCI_vs_CTL"][: min(quotas[1], len(rankings["AD_MCI_vs_CTL"]))]
        ],
    }

    union_original: list[int] = []
    union_set: set[int] = set()
    max_quota = max(len(values) for values in selected_original_by_pool.values())
    for pos in range(max_quota):
        for pool_name in ["AD_vs_MCI", "AD_MCI_vs_CTL"]:
            pool_values = selected_original_by_pool[pool_name]
            if pos >= len(pool_values):
                continue
            gene_idx = pool_values[pos]
            if gene_idx not in union_set:
                union_original.append(gene_idx)
                union_set.add(gene_idx)

    original_to_union = {gene_idx: union_idx for union_idx, gene_idx in enumerate(union_original)}
    ad_mci_pool = [
        original_to_union[gene_idx]
        for gene_idx in selected_original_by_pool["AD_vs_MCI"]
        if gene_idx in original_to_union
    ]
    disease_vs_ctl_pool = [
        original_to_union[gene_idx]
        for gene_idx in selected_original_by_pool["AD_MCI_vs_CTL"]
        if gene_idx in original_to_union
    ]
    task_gene_indices = {
        "AD_vs_MCI": ad_mci_pool,
        "AD_vs_CTL": disease_vs_ctl_pool,
        "MCI_vs_CTL": disease_vs_ctl_pool,
    }

    selected_set_by_pool = {pool_name: set(values) for pool_name, values in selected_original_by_pool.items()}
    details_rows = []
    for union_order, gene_idx in enumerate(union_original, start=1):
        selected_for_pools = [
            pool_name for pool_name, values in selected_set_by_pool.items() if int(gene_idx) in values
        ]
        row = {
            "gene": dataset.gene_names[int(gene_idx)],
            "selection_order": union_order,
            "selected_by_task": ";".join(selected_for_pools),
            "selection_method": "ad_mci_vs_ctl_anova_50_50_task_specific_pooling",
        }
        for pool_name, scores in score_by_pool.items():
            row[f"{pool_name}_anova_f"] = float(scores[int(gene_idx)])
            row[f"{pool_name}_in_pool"] = pool_name in selected_for_pools
        details_rows.append(row)

    return subset_prepared_dataset(dataset, np.asarray(union_original, dtype=np.int64)), pd.DataFrame(details_rows), task_gene_indices


def prepare_multitask_dataset(
    args: argparse.Namespace,
    resolved_split_file: Path | None,
) -> tuple[PreparedDataset, pd.DataFrame, dict[str, list[int]] | None, dict[str, Any]]:
    candidate_genes, candidate_gene_manifest = resolve_candidate_genes(args)
    if args.feature_selection_fit_scope == "all" and args.gene_selection not in {"variance", "mad"}:
        raise ValueError("--feature-selection-fit-scope all is implemented only for unsupervised variance/MAD ablations.")
    if args.gene_selection in {"variance", "mad"}:
        if args.task_specific_pooling == "on":
            raise ValueError("task-specific pooling requires ANOVA task-specific gene selection, not variance/MAD.")
        dataset = prepare_dataset(
            x_file=args.x_file,
            y_file=args.y_file,
            seed=args.seed,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            max_genes=0,
            scaler="none",
            scaler_fit_scope="train",
            split_file=resolved_split_file,
            split_seed=args.split_seed,
            split_mode="stratified" if args.split_mode in {"official", "custom", "stratified"} else "random",
            candidate_genes=candidate_genes,
        )
        dataset, details = select_simple_gene_ranking(
            dataset,
            args.max_genes,
            args.gene_selection,
            fit_scope=args.feature_selection_fit_scope,
        )
        dataset.train_gene_x, [dataset.val_gene_x, dataset.test_gene_x] = scale_arrays(
            dataset.train_gene_x,
            [dataset.val_gene_x, dataset.test_gene_x],
            args.scaler,
            fit_scope=args.scaler_fit_scope,
        )
        return dataset, details, None, candidate_gene_manifest

    dataset = prepare_dataset(
        x_file=args.x_file,
        y_file=args.y_file,
        seed=args.seed,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        max_genes=0,
        scaler="none",
        scaler_fit_scope="train",
        split_file=resolved_split_file,
        split_seed=args.split_seed,
        split_mode="stratified" if args.split_mode in {"official", "custom", "stratified"} else "random",
        candidate_genes=candidate_genes,
    )
    task_specs = build_task_specs(dataset.class_names)
    task_gene_indices = None
    if args.task_specific_pooling == "on":
        if args.gene_selection == "pairwise_anova_union":
            dataset, details, task_gene_indices = select_pairwise_anova_task_specific_pooling(
                dataset, task_specs, args.max_genes
            )
        elif args.gene_selection == "ad_mci_vs_ctl_anova_50_50":
            dataset, details, task_gene_indices = select_ad_mci_vs_ctl_anova_task_specific_pooling(
                dataset, args.max_genes
            )
        else:
            raise ValueError(
                "--task-specific-pooling on requires --gene-selection pairwise_anova_union "
                "or ad_mci_vs_ctl_anova_50_50."
            )
    elif args.gene_selection == "ad_mci_vs_ctl_anova_50_50":
        dataset, details = select_ad_mci_vs_ctl_anova_50_50(dataset, task_specs, args.max_genes)
    elif args.gene_selection == "ad_mci_priority_anova_union":
        dataset, details = select_weighted_pairwise_anova_union(
            dataset,
            task_specs,
            args.max_genes,
            args.ad_mci_gene_fraction,
        )
    else:
        dataset, details = select_pairwise_anova_union(dataset, task_specs, args.max_genes)
    dataset.train_gene_x, [dataset.val_gene_x, dataset.test_gene_x] = scale_arrays(
        dataset.train_gene_x,
        [dataset.val_gene_x, dataset.test_gene_x],
        args.scaler,
        fit_scope=args.scaler_fit_scope,
    )
    return dataset, details, task_gene_indices, candidate_gene_manifest


def prepare_volumetric_graph(
    args: argparse.Namespace,
    gene_names: list[str],
) -> Any | None:
    if args.model_variant == "baseline":
        return None
    from source.models.txt_volumetric import load_induced_ppi_graph

    graph = load_induced_ppi_graph(
        args.ppi_edge_file,
        gene_names,
        score_threshold=args.ppi_score_threshold,
    )
    graph.to_edge_frame(directed=False).to_csv(
        args.result_dir / "induced_ppi_edges.csv", index=False
    )
    manifest = {
        **graph.to_manifest(),
        "model_variant": args.model_variant,
        "volumetric_beta": float(args.volumetric_beta),
        "volumetric_eps": float(args.volumetric_eps),
        "volumetric_gate_init": float(args.volumetric_gate_init),
        "volumetric_volume_mode": args.volumetric_volume_mode,
        "volumetric_dropout": args.volumetric_dropout,
        "volumetric_dropout_effective": float(
            args.dropout if args.volumetric_dropout is None else args.volumetric_dropout
        ),
        "volumetric_message_mode": args.volumetric_message_mode,
        "volumetric_output_norm": args.volumetric_output_norm,
        "volumetric_gate_mode": args.volumetric_gate_mode,
        "volumetric_backbone_gradient_mode": args.volumetric_backbone_gradient_mode,
        "confidence_used_by_model": False,
    }
    save_json(args.result_dir / "ppi_graph_manifest.json", manifest)
    return graph


def model_ppi_gate_values(model: nn.Module) -> dict[str, float]:
    accessor = getattr(model, "ppi_gate_values", None)
    return dict(accessor()) if callable(accessor) else {}


def model_volumetric_gate_values(model: nn.Module) -> dict[str, float]:
    accessor = getattr(model, "volumetric_gate_values", None)
    if not callable(accessor):
        return {}
    return {str(key): float(value) for key, value in accessor().items()}


def _json_compatible(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        detached = value.detach().cpu()
        return detached.item() if detached.numel() == 1 else detached.tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    return value


def model_volumetric_diagnostics(model: nn.Module) -> dict[str, Any]:
    accessor = getattr(model, "volumetric_diagnostics", None)
    if not callable(accessor):
        return {}
    diagnostics = accessor()
    return _json_compatible(diagnostics) if diagnostics is not None else {}


def _aggregate_weighted_diagnostic_values(
    weighted_values: list[tuple[Any, int]],
) -> Any:
    """Aggregate nested diagnostic snapshots without assuming a fixed schema."""

    if not weighted_values:
        return None
    if all(isinstance(value, dict) for value, _ in weighted_values):
        keys: list[str] = []
        for value, _ in weighted_values:
            for key in value:
                if key not in keys:
                    keys.append(key)
        return {
            key: _aggregate_weighted_diagnostic_values(
                [(value[key], weight) for value, weight in weighted_values if key in value]
            )
            for key in keys
        }
    if all(
        isinstance(value, (int, float, np.number)) and not isinstance(value, bool)
        for value, _ in weighted_values
    ):
        total_weight = sum(weight for _, weight in weighted_values)
        if total_weight <= 0:
            return math.nan
        return float(
            sum(float(value) * weight for value, weight in weighted_values) / total_weight
        )
    json_values = [_json_compatible(value) for value, _ in weighted_values]
    if all(value == json_values[0] for value in json_values[1:]):
        return json_values[0]
    unique_values: list[Any] = []
    for value in json_values:
        if value not in unique_values:
            unique_values.append(value)
    return unique_values


def aggregate_volumetric_diagnostic_runs(
    diagnostic_runs: list[list[tuple[int, dict[str, Any]]]],
    *,
    split_name: str,
    batch_size: int,
    samples_evaluated: int,
    full_split: bool,
) -> dict[str, Any]:
    """Summarize every forward batch, first per member and then across members."""

    member_payloads: list[dict[str, Any]] = []
    all_snapshots: list[tuple[Any, int]] = []
    for rank, batches in enumerate(diagnostic_runs, start=1):
        snapshots = [(diagnostics, samples) for samples, diagnostics in batches]
        all_snapshots.extend(snapshots)
        member_payloads.append(
            {
                "rank": rank,
                "forward_batches": len(batches),
                "forwarded_samples": int(sum(samples for samples, _ in batches)),
                "diagnostics": _aggregate_weighted_diagnostic_values(snapshots) or {},
            }
        )
    return {
        "diagnostics": _aggregate_weighted_diagnostic_values(all_snapshots) or {},
        "member_diagnostics": member_payloads,
        "scope": {
            "split": split_name,
            "model_mode": "eval",
            "aggregation": (
                "sample_weighted_mean_across_batches_then_arithmetic_mean_across_members"
            ),
            "batch_size": int(batch_size),
            "samples_evaluated": int(samples_evaluated),
            "full_split": bool(full_split),
            "ensemble_members": len(diagnostic_runs),
            "forward_batches": int(sum(len(run) for run in diagnostic_runs)),
            "forwarded_samples": int(
                sum(samples for run in diagnostic_runs for samples, _ in run)
            ),
        },
    }


def _flatten_scalar_diagnostics(
    value: Any,
    prefix: str = "",
) -> dict[str, float]:
    flattened: dict[str, float] = {}
    if isinstance(value, dict):
        for key, item in value.items():
            safe_key = re.sub(r"[^0-9A-Za-z]+", "_", str(key)).strip("_")
            nested_prefix = f"{prefix}_{safe_key}" if prefix else safe_key
            flattened.update(_flatten_scalar_diagnostics(item, nested_prefix))
        return flattened
    if isinstance(value, torch.Tensor) and value.numel() == 1:
        value = value.detach().cpu().item()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (bool, int, float)) and not isinstance(value, bool):
        flattened[prefix] = float(value)
    return flattened


def volumetric_training_log_fields(model: nn.Module) -> dict[str, float]:
    fields = {
        f"volumetric_gamma_{name}": value
        for name, value in model_volumetric_gate_values(model).items()
    }
    for key, value in _flatten_scalar_diagnostics(model_volumetric_diagnostics(model)).items():
        fields[f"volumetric_{key}"] = value
    return fields


def class_counts_dict(y: np.ndarray, class_names: list[str]) -> dict[str, int]:
    return {class_names[idx]: int((y == idx).sum()) for idx in range(len(class_names))}


def metric_values_from_results(
    val_results: dict[str, dict[str, Any]],
    metric_name: str,
    exclude_task: str | None = None,
) -> list[float]:
    values = []
    for task_name, result in val_results.items():
        if exclude_task is not None and task_name == exclude_task:
            continue
        value = float(result[metric_name])
        if not math.isnan(value):
            values.append(value)
    return values


def compute_checkpoint_value(
    checkpoint_metric: str,
    val_results: dict[str, dict[str, Any]],
    val_loss: float,
    primary_task: str = "AD_vs_MCI",
) -> tuple[float, dict[str, Any]]:
    val_macro_values = metric_values_from_results(val_results, "macro_f1")
    val_auc_values = metric_values_from_results(val_results, "roc_auc")
    val_macro_f1_mean = float(np.mean(val_macro_values)) if val_macro_values else math.nan
    val_multitask_auc_mean = float(np.mean(val_auc_values)) if val_auc_values else math.nan

    primary_result = val_results.get(primary_task, {})
    primary_auc = float(primary_result.get("roc_auc", math.nan))
    primary_macro_f1 = float(primary_result.get("macro_f1", math.nan))
    if not math.isnan(primary_auc):
        primary_signal = primary_auc
        primary_signal_source = "roc_auc"
    elif not math.isnan(primary_macro_f1):
        primary_signal = primary_macro_f1
        primary_signal_source = "macro_f1_fallback"
    else:
        primary_signal = -float(val_loss)
        primary_signal_source = "negative_loss_fallback"
    auxiliary_auc_values = metric_values_from_results(val_results, "roc_auc", exclude_task=primary_task)
    auxiliary_auc_mean = float(np.mean(auxiliary_auc_values)) if auxiliary_auc_values else math.nan
    primary_auc_with_aux = primary_signal
    if not math.isnan(auxiliary_auc_mean):
        primary_auc_with_aux = 0.75 * primary_signal + 0.25 * auxiliary_auc_mean
    weighted_70_15_15_auc = math.nan
    task_weights = {"AD_vs_MCI": 0.70, "AD_vs_CTL": 0.15, "MCI_vs_CTL": 0.15}
    weighted_values = []
    for task_name, weight in task_weights.items():
        task_auc = float(val_results.get(task_name, {}).get("roc_auc", math.nan))
        if not math.isnan(task_auc):
            weighted_values.append(weight * task_auc)
    if len(weighted_values) == len(task_weights):
        weighted_70_15_15_auc = float(sum(weighted_values))

    details = {
        "val_multitask_auc_mean": val_multitask_auc_mean,
        "val_multitask_macro_f1_mean": val_macro_f1_mean,
        "val_primary_roc_auc": primary_auc,
        "val_primary_macro_f1": primary_macro_f1,
        "val_primary_checkpoint_signal": primary_signal,
        "val_primary_checkpoint_signal_source": primary_signal_source,
        "val_auxiliary_auc_mean": auxiliary_auc_mean,
        "val_primary_auc_with_auxiliary_mean": primary_auc_with_aux,
        "val_weighted_70_15_15_auc": weighted_70_15_15_auc,
    }
    if checkpoint_metric == "val_loss":
        return val_loss, details
    if checkpoint_metric == "val_multitask_auc_mean":
        return val_multitask_auc_mean, details
    if checkpoint_metric == "val_multitask_auc_minus_025_loss":
        return val_multitask_auc_mean - 0.25 * val_loss, details
    if checkpoint_metric == "val_multitask_auc_f1_minus_loss":
        auc_term = 0.0 if math.isnan(val_multitask_auc_mean) else 0.50 * val_multitask_auc_mean
        f1_term = 0.0 if math.isnan(val_macro_f1_mean) else 0.50 * val_macro_f1_mean
        return auc_term + f1_term - val_loss, details
    if checkpoint_metric == "val_primary_auc_mean":
        return primary_auc_with_aux, details
    if checkpoint_metric == "val_primary_auc_minus_025_loss":
        return primary_auc_with_aux - 0.25 * val_loss, details
    if checkpoint_metric == "val_weighted_70_15_15_auc_minus_025_loss":
        if math.isnan(weighted_70_15_15_auc):
            return primary_auc_with_aux - 0.25 * val_loss, details
        return weighted_70_15_15_auc - 0.25 * val_loss, details
    if checkpoint_metric == "val_multitask_macro_f1_mean":
        return val_macro_f1_mean, details
    raise ValueError(f"Unsupported checkpoint metric: {checkpoint_metric}")


def replace_train_split(
    dataset: PreparedDataset,
    train_gene_x: np.ndarray,
    train_y: np.ndarray,
    train_ids: np.ndarray,
) -> PreparedDataset:
    return PreparedDataset(
        class_names=dataset.class_names,
        gene_names=dataset.gene_names,
        train_ids=train_ids,
        val_ids=dataset.val_ids,
        test_ids=dataset.test_ids,
        train_gene_x=train_gene_x.astype(np.float32),
        val_gene_x=dataset.val_gene_x,
        test_gene_x=dataset.test_gene_x,
        train_y=train_y.astype(np.int64),
        val_y=dataset.val_y,
        test_y=dataset.test_y,
    )


def sanitize_augmented_features(features: np.ndarray, scaler: str) -> np.ndarray:
    values = np.nan_to_num(features.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if scaler == "minmax":
        values = np.nan_to_num(features.astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0)
        return np.clip(values, 0.0, 1.0).astype(np.float32)
    return values.astype(np.float32)


def apply_smote_augmentation(dataset: PreparedDataset, args: argparse.Namespace) -> tuple[PreparedDataset, dict[str, Any]]:
    from imblearn.over_sampling import SMOTE

    counts = pd.Series(dataset.train_y).value_counts().sort_index()
    minority_count = int(counts.min())
    if minority_count < 2 or int(counts.min()) == int(counts.max()):
        manifest = {
            "augmentation": "smote",
            "status": "skipped_not_enough_minority_or_already_balanced",
            "augmentation_scope": "train_after_train_only_feature_selection_and_scaling",
            "validation_and_test_scope": "not_augmented",
            "n_original_train": int(len(dataset.train_y)),
            "n_synthetic": 0,
            "n_augmented_train": int(len(dataset.train_y)),
            "original_class_counts": class_counts_dict(dataset.train_y, dataset.class_names),
            "augmented_class_counts": class_counts_dict(dataset.train_y, dataset.class_names),
        }
        return dataset, manifest

    effective_k = max(1, min(int(args.smote_k_neighbors), minority_count - 1))
    sampler = SMOTE(random_state=args.seed, k_neighbors=effective_k, sampling_strategy="auto")
    x_resampled, y_resampled = sampler.fit_resample(dataset.train_gene_x, dataset.train_y)
    x_resampled = sanitize_augmented_features(x_resampled, args.scaler)
    y_resampled = y_resampled.astype(np.int64)
    n_synthetic = int(len(y_resampled) - len(dataset.train_y))
    synthetic_ids = np.asarray([f"synthetic_smote_{idx:05d}" for idx in range(n_synthetic)], dtype=str)
    train_ids = np.concatenate([dataset.train_ids.astype(str), synthetic_ids])
    augmented = replace_train_split(dataset, x_resampled, y_resampled, train_ids)
    manifest = {
        "augmentation": "smote",
        "status": "completed",
        "augmentation_scope": "train_after_train_only_feature_selection_and_scaling",
        "validation_and_test_scope": "not_augmented",
        "sampling_strategy": "auto_multiclass_balance",
        "feature_value_policy": "clip_to_0_1_after_minmax" if args.scaler == "minmax" else "nan_inf_to_zero_no_clipping",
        "k_neighbors": int(args.smote_k_neighbors),
        "effective_k_neighbors": int(effective_k),
        "n_original_train": int(len(dataset.train_y)),
        "n_synthetic": n_synthetic,
        "n_augmented_train": int(len(y_resampled)),
        "original_class_counts": class_counts_dict(dataset.train_y, dataset.class_names),
        "augmented_class_counts": class_counts_dict(y_resampled, dataset.class_names),
    }
    return augmented, manifest


def apply_borderline_smote_augmentation(
    dataset: PreparedDataset,
    args: argparse.Namespace,
) -> tuple[PreparedDataset, dict[str, Any]]:
    counts = pd.Series(dataset.train_y).value_counts().sort_index()
    minority_count = int(counts.min())
    majority_count = int(counts.max())
    base_manifest = {
        "augmentation": "borderline_smote",
        "augmentation_scope": "train_after_train_only_feature_selection_and_scaling",
        "validation_and_test_scope": "not_augmented",
        "paper_reference": "Diagnostics 2025-style BorderlineSMOTE train-fold oversampling for HDLSS gene-expression classification.",
        "implementation_note": "Applied only after train-only feature selection and scaling; validation and test samples are never augmented.",
        "kind": args.smote_kind,
        "k_neighbors": int(args.smote_k_neighbors),
        "m_neighbors": int(args.smote_m_neighbors),
        "n_original_train": int(len(dataset.train_y)),
        "original_class_counts": class_counts_dict(dataset.train_y, dataset.class_names),
    }
    if minority_count < 2 or minority_count == majority_count:
        manifest = {
            **base_manifest,
            "status": "skipped_not_enough_minority_or_already_balanced",
            "n_synthetic": 0,
            "n_augmented_train": int(len(dataset.train_y)),
            "augmented_class_counts": class_counts_dict(dataset.train_y, dataset.class_names),
        }
        return dataset, manifest

    from imblearn.over_sampling import BorderlineSMOTE

    effective_k = max(1, min(int(args.smote_k_neighbors), minority_count - 1))
    effective_m = max(1, min(int(args.smote_m_neighbors), len(dataset.train_y) - 1))
    sampler = BorderlineSMOTE(
        random_state=args.seed,
        k_neighbors=effective_k,
        m_neighbors=effective_m,
        kind=args.smote_kind,
        sampling_strategy="auto",
    )
    x_resampled, y_resampled = sampler.fit_resample(dataset.train_gene_x, dataset.train_y)
    x_resampled = sanitize_augmented_features(x_resampled, args.scaler)
    y_resampled = y_resampled.astype(np.int64)
    n_synthetic = int(len(y_resampled) - len(dataset.train_y))
    synthetic_ids = np.asarray([f"synthetic_borderline_smote_{idx:05d}" for idx in range(n_synthetic)], dtype=str)
    train_ids = np.concatenate([dataset.train_ids.astype(str), synthetic_ids])
    augmented = replace_train_split(dataset, x_resampled, y_resampled, train_ids)
    manifest = {
        **base_manifest,
        "status": "completed",
        "sampling_strategy": "auto_multiclass_balance",
        "feature_value_policy": "clip_to_0_1_after_minmax" if args.scaler == "minmax" else "nan_inf_to_zero_no_clipping",
        "effective_k_neighbors": int(effective_k),
        "effective_m_neighbors": int(effective_m),
        "n_synthetic": n_synthetic,
        "n_augmented_train": int(len(y_resampled)),
        "augmented_class_counts": class_counts_dict(y_resampled, dataset.class_names),
    }
    return augmented, manifest


def ctgan_label_plan(y_train: np.ndarray, class_names: list[str], target_size: int) -> dict[str, int]:
    current = {idx: int((y_train == idx).sum()) for idx in range(len(class_names))}
    desired = {idx: 0 for idx in range(len(class_names))}
    remaining = max(0, target_size - len(y_train))
    while remaining > 0:
        label = min(current, key=lambda idx: current[idx] + desired[idx])
        desired[label] += 1
        remaining -= 1
    return {class_names[idx]: int(count) for idx, count in desired.items() if count > 0}


def sample_ctgan_rows(
    synthesizer: Any,
    label_plan: dict[str, int],
    feature_columns: list[str],
    class_names: list[str],
    seed: int,
) -> tuple[pd.DataFrame, np.ndarray, dict[str, int]]:
    rng = np.random.default_rng(seed)
    collected: list[pd.DataFrame] = []
    missing = dict(label_plan)
    attempts = 0
    max_attempts = 20
    while any(count > 0 for count in missing.values()) and attempts < max_attempts:
        attempts += 1
        request_size = max(256, int(sum(missing.values()) * 3))
        sampled = synthesizer.sample(num_rows=request_size)
        if "label" not in sampled.columns:
            raise ValueError("CTGAN sampled data does not contain the label column.")
        sampled["label"] = sampled["label"].astype(str)
        for class_name in class_names:
            need = missing.get(class_name, 0)
            if need <= 0:
                continue
            candidates = sampled[sampled["label"] == class_name]
            if candidates.empty:
                continue
            take = min(need, len(candidates))
            chosen_idx = rng.choice(candidates.index.to_numpy(), size=take, replace=False)
            chosen = candidates.loc[chosen_idx, feature_columns + ["label"]].copy()
            collected.append(chosen)
            missing[class_name] = need - take

    if not collected:
        return pd.DataFrame(columns=feature_columns), np.empty((0,), dtype=np.int64), missing

    synthetic = pd.concat(collected, axis=0, ignore_index=True)
    synthetic_x = synthetic[feature_columns].apply(pd.to_numeric, errors="coerce")
    synthetic_x = synthetic_x.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    synthetic_x = synthetic_x.clip(lower=0.0, upper=1.0)
    label_to_idx = {name: idx for idx, name in enumerate(class_names)}
    synthetic_y = synthetic["label"].map(label_to_idx).to_numpy(dtype=np.int64)
    return synthetic_x, synthetic_y, missing


def apply_ctgan_augmentation(dataset: PreparedDataset, args: argparse.Namespace) -> tuple[PreparedDataset, dict[str, Any]]:
    if args.scaler != "minmax":
        raise ValueError("CTGAN augmentation requires --scaler minmax because generated features are bounded to [0, 1].")
    try:
        from sdv.metadata import SingleTableMetadata
        from sdv.single_table import CTGANSynthesizer
    except ImportError as exc:
        raise ImportError(
            "CTGAN richiede il pacchetto `sdv`. Installa con: pip install sdv"
        ) from exc

    target_size = int(math.ceil(len(dataset.train_y) * float(args.augmentation_target_multiplier)))
    n_to_generate = max(0, target_size - len(dataset.train_y))
    if n_to_generate == 0:
        manifest = {
            "augmentation": "ctgan",
            "status": "skipped_target_size_already_met",
            "augmentation_scope": "train_after_train_only_feature_selection_and_scaling",
            "validation_and_test_scope": "not_augmented",
            "target_size": int(target_size),
            "n_original_train": int(len(dataset.train_y)),
            "n_synthetic": 0,
            "n_augmented_train": int(len(dataset.train_y)),
        }
        return dataset, manifest

    train_df = pd.DataFrame(dataset.train_gene_x, columns=dataset.gene_names)
    train_df["label"] = [dataset.class_names[int(label)] for label in dataset.train_y]
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(train_df)
    metadata.update_column("label", sdtype="categorical")

    effective_batch_size = max(10, int(math.ceil(int(args.ctgan_batch_size) / 10.0) * 10))
    kwargs = {
        "epochs": int(args.ctgan_epochs),
        "batch_size": int(effective_batch_size),
        "pac": 10,
        "enable_gpu": bool(args.device == "cuda" and torch.cuda.is_available()),
        "verbose": False,
    }
    try:
        synthesizer = CTGANSynthesizer(metadata, **kwargs)
    except TypeError:
        kwargs.pop("verbose", None)
        kwargs.pop("enable_gpu", None)
        kwargs["cuda"] = bool(args.device == "cuda" and torch.cuda.is_available())
        synthesizer = CTGANSynthesizer(metadata, **kwargs)
    synthesizer.fit(train_df)

    label_plan = ctgan_label_plan(dataset.train_y, dataset.class_names, target_size)
    synthetic_x_df, synthetic_y, missing = sample_ctgan_rows(
        synthesizer,
        label_plan,
        dataset.gene_names,
        dataset.class_names,
        args.seed + 991,
    )
    synthetic_x = synthetic_x_df.to_numpy(dtype=np.float32)
    train_gene_x = np.vstack([dataset.train_gene_x, synthetic_x]) if len(synthetic_y) else dataset.train_gene_x
    train_y = np.concatenate([dataset.train_y, synthetic_y]) if len(synthetic_y) else dataset.train_y
    synthetic_ids = np.asarray([f"synthetic_ctgan_{idx:05d}" for idx in range(len(synthetic_y))], dtype=str)
    train_ids = np.concatenate([dataset.train_ids.astype(str), synthetic_ids])
    augmented = replace_train_split(dataset, train_gene_x, train_y, train_ids)
    manifest = {
        "augmentation": "ctgan",
        "status": "completed" if len(synthetic_y) == n_to_generate else "completed_partial",
        "augmentation_scope": "train_after_train_only_feature_selection_and_scaling",
        "validation_and_test_scope": "not_augmented",
        "target_size": int(target_size),
        "target_multiplier": float(args.augmentation_target_multiplier),
        "ctgan_epochs": int(args.ctgan_epochs),
        "ctgan_batch_size": int(args.ctgan_batch_size),
        "ctgan_effective_batch_size": int(effective_batch_size),
        "ctgan_pac": 10,
        "requested_synthetic_by_class": label_plan,
        "missing_synthetic_by_class": {key: int(value) for key, value in missing.items()},
        "n_original_train": int(len(dataset.train_y)),
        "n_synthetic": int(len(synthetic_y)),
        "n_augmented_train": int(len(train_y)),
        "original_class_counts": class_counts_dict(dataset.train_y, dataset.class_names),
        "synthetic_class_counts": class_counts_dict(synthetic_y, dataset.class_names),
        "augmented_class_counts": class_counts_dict(train_y, dataset.class_names),
    }
    return augmented, manifest


def apply_pca_neighbor_mci_ctl_augmentation(
    dataset: PreparedDataset,
    args: argparse.Namespace,
) -> tuple[PreparedDataset, dict[str, Any]]:
    """Generate a small number of MCI profiles for the MCI-vs-Control task only.

    PCA is fitted train-only and is used solely for neighbour discovery. The
    convex interpolation itself is performed in the selected/scaled gene space.
    """
    from sklearn.decomposition import PCA
    from sklearn.neighbors import NearestNeighbors

    class_idx = resolve_class_indices(dataset.class_names)
    mci_label = class_idx["mci"]
    ctl_label = class_idx["control"]
    mci_indices = np.flatnonzero(dataset.train_y == mci_label)
    ctl_indices = np.flatnonzero(dataset.train_y == ctl_label)
    gap = max(0, int(len(ctl_indices) - len(mci_indices)))
    fraction = float(args.pca_neighbor_gap_fraction)
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("--pca-neighbor-gap-fraction must be between 0 and 1.")
    n_synthetic = int(math.ceil(gap * fraction))
    base_manifest = {
        "augmentation": "pca_neighbor_mci_ctl",
        "augmentation_scope": "train_only_after_feature_selection_and_scaling",
        "validation_and_test_scope": "not_augmented",
        "synthetic_task_scope": "MCI_vs_CTL_only",
        "synthetic_task_mask": [False, False, True],
        "pca_requested_components": int(args.pca_neighbor_components),
        "neighbor_k_requested": int(args.pca_neighbor_k),
        "gap_fraction": fraction,
        "mci_count": int(len(mci_indices)),
        "control_count": int(len(ctl_indices)),
        "class_count_gap": gap,
        "n_original_train": int(len(dataset.train_y)),
        "original_class_counts": class_counts_dict(dataset.train_y, dataset.class_names),
    }
    if n_synthetic <= 0 or len(mci_indices) < 2:
        return dataset, {
            **base_manifest,
            "status": "skipped_no_gap_or_not_enough_mci",
            "n_synthetic": 0,
            "n_augmented_train": int(len(dataset.train_y)),
            "augmented_class_counts": class_counts_dict(dataset.train_y, dataset.class_names),
        }

    pair_indices = np.concatenate([mci_indices, ctl_indices])
    max_components = min(len(pair_indices) - 1, dataset.train_gene_x.shape[1])
    effective_components = max(1, min(int(args.pca_neighbor_components), max_components))
    pca = PCA(n_components=effective_components, random_state=args.seed)
    pair_scores = pca.fit_transform(dataset.train_gene_x[pair_indices])
    mci_scores = pair_scores[: len(mci_indices)]
    effective_k = max(1, min(int(args.pca_neighbor_k), len(mci_indices) - 1))
    neighbor_model = NearestNeighbors(n_neighbors=effective_k + 1, metric="euclidean")
    neighbor_model.fit(mci_scores)
    neighbor_indices = neighbor_model.kneighbors(mci_scores, return_distance=False)[:, 1:]

    rng = np.random.default_rng(args.seed)
    synthetic_rows: list[np.ndarray] = []
    interpolation_coefficients: list[float] = []
    for synthetic_idx in range(n_synthetic):
        base_position = synthetic_idx % len(mci_indices)
        neighbor_position = int(rng.choice(neighbor_indices[base_position]))
        coefficient = float(rng.uniform(0.2, 0.8))
        base_x = dataset.train_gene_x[mci_indices[base_position]]
        neighbor_x = dataset.train_gene_x[mci_indices[neighbor_position]]
        synthetic_rows.append((base_x + coefficient * (neighbor_x - base_x)).astype(np.float32))
        interpolation_coefficients.append(coefficient)

    synthetic_x = np.vstack(synthetic_rows).astype(np.float32)
    synthetic_y = np.full(n_synthetic, mci_label, dtype=np.int64)
    synthetic_ids = np.asarray(
        [f"{PCA_MCI_CTL_SYNTHETIC_PREFIX}{idx:05d}" for idx in range(n_synthetic)], dtype=str
    )
    train_x = np.vstack([dataset.train_gene_x, synthetic_x]).astype(np.float32)
    train_y = np.concatenate([dataset.train_y, synthetic_y])
    train_ids = np.concatenate([dataset.train_ids.astype(str), synthetic_ids])
    augmented = replace_train_split(dataset, train_x, train_y, train_ids)
    return augmented, {
        **base_manifest,
        "status": "completed",
        "pca_effective_components": effective_components,
        "pca_explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
        "neighbor_k_effective": effective_k,
        "interpolation_lambda_min": float(min(interpolation_coefficients)),
        "interpolation_lambda_max": float(max(interpolation_coefficients)),
        "n_synthetic": n_synthetic,
        "n_augmented_train": int(len(train_y)),
        "augmented_class_counts": class_counts_dict(train_y, dataset.class_names),
    }


def apply_pca_neighbor_all_tasks_augmentation(
    dataset: PreparedDataset,
    args: argparse.Namespace,
) -> tuple[PreparedDataset, dict[str, Any]]:
    """Partially balance biological classes with train-only PCA-neighbour interpolation.

    PCA is fitted on the complete training split and is used only to identify
    within-class neighbours. Interpolation is performed in the selected/scaled
    gene space. Synthetic samples keep their biological class label, therefore
    each contributes to both pairwise tasks in which that class participates.
    """
    from sklearn.decomposition import PCA
    from sklearn.neighbors import NearestNeighbors

    fraction = float(args.pca_neighbor_gap_fraction)
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("--pca-neighbor-gap-fraction must be between 0 and 1.")
    class_counts = np.bincount(dataset.train_y, minlength=len(dataset.class_names))
    requested_target_count = int(getattr(args, "pca_neighbor_target_count", 0))
    if requested_target_count < 0:
        raise ValueError("--pca-neighbor-target-count must be non-negative.")
    largest_class_count = int(class_counts.max())
    target_count = requested_target_count or largest_class_count
    if target_count < largest_class_count:
        raise ValueError(
            "--pca-neighbor-target-count must be 0 or at least the largest training-class count "
            f"({largest_class_count} for this split)."
        )
    requested_by_class = {
        dataset.class_names[label]: int(math.ceil((target_count - int(count)) * fraction))
        for label, count in enumerate(class_counts)
    }
    n_requested = int(sum(requested_by_class.values()))
    base_manifest = {
        "augmentation": "pca_neighbor_all_tasks",
        "augmentation_scope": "train_only_after_feature_selection_and_scaling",
        "validation_and_test_scope": "not_augmented",
        "synthetic_task_scope": "normal_biological_class_masks_all_pairwise_tasks",
        "balancing_target": "fraction_of_gap_to_target_class_count",
        "target_count_source": "explicit" if requested_target_count else "largest_training_class",
        "target_class_count": target_count,
        "gap_fraction": fraction,
        "pca_requested_components": int(args.pca_neighbor_components),
        "neighbor_k_requested": int(args.pca_neighbor_k),
        "requested_synthetic_by_class": requested_by_class,
        "n_original_train": int(len(dataset.train_y)),
        "original_class_counts": class_counts_dict(dataset.train_y, dataset.class_names),
    }
    if n_requested <= 0:
        return dataset, {
            **base_manifest,
            "status": "skipped_classes_already_balanced",
            "n_synthetic": 0,
            "generated_synthetic_by_class": {name: 0 for name in dataset.class_names},
            "n_augmented_train": int(len(dataset.train_y)),
            "augmented_class_counts": class_counts_dict(dataset.train_y, dataset.class_names),
        }

    max_components = min(len(dataset.train_y) - 1, dataset.train_gene_x.shape[1])
    effective_components = max(1, min(int(args.pca_neighbor_components), max_components))
    pca = PCA(n_components=effective_components, random_state=args.seed)
    train_scores = pca.fit_transform(dataset.train_gene_x)
    rng = np.random.default_rng(args.seed)
    synthetic_rows: list[np.ndarray] = []
    synthetic_labels: list[int] = []
    synthetic_ids: list[str] = []
    coefficients: list[float] = []
    effective_k_by_class: dict[str, int] = {}
    generated_by_class = {name: 0 for name in dataset.class_names}

    for label, class_name in enumerate(dataset.class_names):
        n_synthetic = requested_by_class[class_name]
        class_indices = np.flatnonzero(dataset.train_y == label)
        if n_synthetic <= 0 or len(class_indices) < 2:
            effective_k_by_class[class_name] = 0
            continue
        effective_k = max(1, min(int(args.pca_neighbor_k), len(class_indices) - 1))
        effective_k_by_class[class_name] = effective_k
        neighbour_model = NearestNeighbors(n_neighbors=effective_k + 1, metric="euclidean")
        neighbour_model.fit(train_scores[class_indices])
        neighbours = neighbour_model.kneighbors(train_scores[class_indices], return_distance=False)[:, 1:]
        safe_class_name = re.sub(r"[^a-z0-9]+", "_", normalize_class_name(class_name)).strip("_")
        for synthetic_idx in range(n_synthetic):
            base_position = synthetic_idx % len(class_indices)
            neighbour_position = int(rng.choice(neighbours[base_position]))
            coefficient = float(rng.uniform(0.2, 0.8))
            base_x = dataset.train_gene_x[class_indices[base_position]]
            neighbour_x = dataset.train_gene_x[class_indices[neighbour_position]]
            synthetic_rows.append((base_x + coefficient * (neighbour_x - base_x)).astype(np.float32))
            synthetic_labels.append(label)
            synthetic_ids.append(f"synthetic_pca_all_tasks_{safe_class_name}_{synthetic_idx:05d}")
            coefficients.append(coefficient)
            generated_by_class[class_name] += 1

    if not synthetic_rows:
        return dataset, {
            **base_manifest,
            "status": "skipped_not_enough_within_class_neighbours",
            "n_synthetic": 0,
            "generated_synthetic_by_class": generated_by_class,
            "neighbor_k_effective_by_class": effective_k_by_class,
            "n_augmented_train": int(len(dataset.train_y)),
            "augmented_class_counts": class_counts_dict(dataset.train_y, dataset.class_names),
        }

    synthetic_x = np.vstack(synthetic_rows).astype(np.float32)
    synthetic_y = np.asarray(synthetic_labels, dtype=np.int64)
    train_x = np.vstack([dataset.train_gene_x, synthetic_x]).astype(np.float32)
    train_y = np.concatenate([dataset.train_y, synthetic_y])
    train_ids = np.concatenate([dataset.train_ids.astype(str), np.asarray(synthetic_ids, dtype=str)])
    augmented = replace_train_split(dataset, train_x, train_y, train_ids)
    return augmented, {
        **base_manifest,
        "status": "completed",
        "pca_effective_components": effective_components,
        "pca_explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
        "neighbor_k_effective_by_class": effective_k_by_class,
        "interpolation_lambda_min": float(min(coefficients)),
        "interpolation_lambda_max": float(max(coefficients)),
        "generated_synthetic_by_class": generated_by_class,
        "n_synthetic": int(len(synthetic_y)),
        "n_augmented_train": int(len(train_y)),
        "augmented_class_counts": class_counts_dict(train_y, dataset.class_names),
    }


def gan_realism_summary(real_x: np.ndarray, synthetic_x: np.ndarray) -> dict[str, float]:
    if synthetic_x.size == 0:
        return {
            "realism_mean_abs_feature_mean_diff": float("nan"),
            "realism_mean_abs_feature_std_diff": float("nan"),
            "realism_feature_mean_correlation": float("nan"),
        }
    real_mean = np.nanmean(real_x, axis=0)
    synthetic_mean = np.nanmean(synthetic_x, axis=0)
    real_std = np.nanstd(real_x, axis=0)
    synthetic_std = np.nanstd(synthetic_x, axis=0)
    if float(np.nanstd(real_mean)) > 0.0 and float(np.nanstd(synthetic_mean)) > 0.0:
        mean_correlation = float(np.corrcoef(real_mean, synthetic_mean)[0, 1])
    else:
        mean_correlation = float("nan")
    return {
        "realism_mean_abs_feature_mean_diff": float(np.nanmean(np.abs(real_mean - synthetic_mean))),
        "realism_mean_abs_feature_std_diff": float(np.nanmean(np.abs(real_std - synthetic_std))),
        "realism_feature_mean_correlation": mean_correlation,
    }


def gan_synthetic_labels(
    y_train: np.ndarray,
    n_to_generate: int,
    n_classes: int,
    strategy: str,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    class_indices = np.arange(n_classes, dtype=np.int64)
    counts = np.asarray([(y_train == idx).sum() for idx in class_indices], dtype=np.int64)
    observed = class_indices[counts > 0]
    if n_to_generate <= 0 or len(observed) == 0:
        return np.empty((0,), dtype=np.int64)

    if strategy == "proportional":
        probs = counts[observed].astype(np.float64)
        probs = probs / probs.sum()
        return rng.choice(observed, size=n_to_generate, replace=True, p=probs).astype(np.int64)

    if strategy == "minority":
        minority_label = int(observed[np.argmin(counts[observed])])
        return np.full(n_to_generate, minority_label, dtype=np.int64)

    if strategy != "balanced":
        raise ValueError(f"Unsupported GAN sampling strategy: {strategy}")

    augmented_counts = counts.copy()
    generated: list[int] = []
    for _ in range(n_to_generate):
        label = int(observed[np.argmin(augmented_counts[observed])])
        generated.append(label)
        augmented_counts[label] += 1
    return rng.permutation(np.asarray(generated, dtype=np.int64))


def build_gan_generator(tf: Any, keras: Any, latent_dim: int, n_features: int, n_classes: int) -> Any:
    noise = keras.layers.Input(shape=(latent_dim,), name="noise")
    label = keras.layers.Input(shape=(1,), dtype="int32", name="label")
    label_embed = keras.layers.Embedding(n_classes, latent_dim, name="label_embedding")(label)
    label_flat = keras.layers.Flatten()(label_embed)
    x = keras.layers.Concatenate()([noise, label_flat])
    x = keras.layers.Dense(128, activation="relu")(x)
    x = keras.layers.Dense(128, activation="relu")(x)
    out = keras.layers.Dense(n_features, activation="sigmoid")(x)
    return keras.Model([noise, label], out, name="multiclass_conditional_tabular_generator")


def build_gan_discriminator(keras: Any, n_features: int, n_classes: int) -> Any:
    features = keras.layers.Input(shape=(n_features,), name="features")
    label = keras.layers.Input(shape=(1,), dtype="int32", name="label")
    label_embed = keras.layers.Embedding(n_classes, n_features, name="label_embedding")(label)
    label_flat = keras.layers.Flatten()(label_embed)
    x = keras.layers.Concatenate()([features, label_flat])
    x = keras.layers.Dense(128, activation="relu")(x)
    x = keras.layers.Dense(128, activation="relu")(x)
    out = keras.layers.Dense(1, activation="sigmoid")(x)
    return keras.Model([features, label], out, name="multiclass_conditional_tabular_discriminator")


def train_multiclass_conditional_gan(
    x_train: np.ndarray,
    y_train: np.ndarray,
    args: argparse.Namespace,
    n_classes: int,
) -> tuple[Any, dict[str, Any]]:
    import tensorflow as tf
    from tensorflow import keras

    try:
        for gpu in tf.config.list_physical_devices("GPU"):
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass

    tf.keras.utils.set_random_seed(int(args.seed))
    generator = build_gan_generator(tf, keras, int(args.gan_latent_dim), x_train.shape[1], n_classes)
    discriminator = build_gan_discriminator(keras, x_train.shape[1], n_classes)
    d_optimizer = keras.optimizers.Adam(learning_rate=float(args.gan_learning_rate))
    g_optimizer = keras.optimizers.Adam(learning_rate=float(args.gan_learning_rate))
    loss_fn = keras.losses.BinaryCrossentropy()
    rng = np.random.default_rng(int(args.seed))
    losses: list[dict[str, float]] = []

    real_x = x_train.astype(np.float32)
    real_y = y_train.astype(np.int32).reshape(-1, 1)
    batch_size = max(1, int(args.gan_batch_size))
    epochs = max(1, int(args.gan_epochs))
    latent_dim = int(args.gan_latent_dim)
    for epoch in range(epochs):
        order = rng.permutation(len(real_x))
        epoch_d: list[float] = []
        epoch_g: list[float] = []
        for start in range(0, len(order), batch_size):
            idx = order[start : start + batch_size]
            if len(idx) == 0:
                continue
            batch_x = tf.convert_to_tensor(real_x[idx], dtype=tf.float32)
            batch_y = tf.convert_to_tensor(real_y[idx], dtype=tf.int32)
            current_batch = len(idx)
            noise = tf.convert_to_tensor(rng.normal(size=(current_batch, latent_dim)).astype(np.float32), dtype=tf.float32)

            with tf.GradientTape() as d_tape:
                fake_x = generator([noise, batch_y], training=True)
                real_logits = discriminator([batch_x, batch_y], training=True)
                fake_logits = discriminator([fake_x, batch_y], training=True)
                d_loss = loss_fn(tf.ones_like(real_logits), real_logits) + loss_fn(tf.zeros_like(fake_logits), fake_logits)
            d_grads = d_tape.gradient(d_loss, discriminator.trainable_variables)
            d_optimizer.apply_gradients(zip(d_grads, discriminator.trainable_variables))

            sampled_labels = tf.convert_to_tensor(
                rng.integers(0, n_classes, size=(current_batch, 1), dtype=np.int32),
                dtype=tf.int32,
            )
            noise = tf.convert_to_tensor(rng.normal(size=(current_batch, latent_dim)).astype(np.float32), dtype=tf.float32)
            with tf.GradientTape() as g_tape:
                fake_x = generator([noise, sampled_labels], training=True)
                fake_logits = discriminator([fake_x, sampled_labels], training=True)
                g_loss = loss_fn(tf.ones_like(fake_logits), fake_logits)
            g_grads = g_tape.gradient(g_loss, generator.trainable_variables)
            g_optimizer.apply_gradients(zip(g_grads, generator.trainable_variables))
            epoch_d.append(float(d_loss.numpy()))
            epoch_g.append(float(g_loss.numpy()))
        if epoch == epochs - 1 or epoch % max(1, epochs // 10) == 0:
            losses.append(
                {
                    "epoch": int(epoch),
                    "d_loss": float(np.mean(epoch_d)) if epoch_d else float("nan"),
                    "g_loss": float(np.mean(epoch_g)) if epoch_g else float("nan"),
                }
            )

    metadata = {
        "tensorflow_version": tf.__version__,
        "gan_type": "multiclass_conditional_tabular_gan",
        "latent_dim": int(args.gan_latent_dim),
        "generator_hidden_layers": [128, 128],
        "discriminator_hidden_layers": [128, 128],
        "batch_size": int(args.gan_batch_size),
        "epochs": int(args.gan_epochs),
        "learning_rate": float(args.gan_learning_rate),
        "loss_trace": losses,
    }
    return generator, metadata


def apply_gan_augmentation(dataset: PreparedDataset, args: argparse.Namespace) -> tuple[PreparedDataset, dict[str, Any]]:
    if args.scaler != "minmax":
        raise ValueError("GAN augmentation requires --scaler minmax because generated features are bounded to [0, 1].")
    if len(dataset.class_names) < 2:
        raise ValueError("GAN augmentation requires at least two classes.")

    target_size = int(math.ceil(len(dataset.train_y) * float(args.gan_target_multiplier)))
    n_to_generate = max(0, target_size - len(dataset.train_y))
    base_manifest = {
        "augmentation": "gan",
        "augmentation_scope": "train_after_train_only_feature_selection_and_scaling",
        "validation_and_test_scope": "not_augmented",
        "implementation_note": "Multiclass conditional tabular GAN fit only on train after train-only feature selection and minmax scaling.",
        "target_multiplier": float(args.gan_target_multiplier),
        "target_size": int(target_size),
        "sampling_strategy": args.gan_sampling_strategy,
        "n_original_train": int(len(dataset.train_y)),
        "original_class_counts": class_counts_dict(dataset.train_y, dataset.class_names),
    }
    if n_to_generate == 0:
        manifest = {
            **base_manifest,
            "status": "skipped_target_size_already_met",
            "n_synthetic": 0,
            "n_augmented_train": int(len(dataset.train_y)),
            "synthetic_class_counts": class_counts_dict(np.empty((0,), dtype=np.int64), dataset.class_names),
            "augmented_class_counts": class_counts_dict(dataset.train_y, dataset.class_names),
        }
        return dataset, manifest

    generator, gan_metadata = train_multiclass_conditional_gan(dataset.train_gene_x, dataset.train_y, args, len(dataset.class_names))
    synthetic_y = gan_synthetic_labels(
        dataset.train_y,
        n_to_generate,
        len(dataset.class_names),
        args.gan_sampling_strategy,
        int(args.seed) + 991,
    )
    rng = np.random.default_rng(int(args.seed) + 1991)
    noise = rng.normal(size=(len(synthetic_y), int(args.gan_latent_dim))).astype(np.float32)
    synthetic_x = generator.predict([noise, synthetic_y.reshape(-1, 1).astype(np.int32)], verbose=0).astype(np.float32)
    synthetic_x = sanitize_augmented_features(synthetic_x, args.scaler)

    try:
        import tensorflow as tf

        tf.keras.backend.clear_session()
    except Exception:
        pass

    train_gene_x = np.vstack([dataset.train_gene_x, synthetic_x])
    train_y = np.concatenate([dataset.train_y, synthetic_y.astype(np.int64)])
    synthetic_ids = np.asarray([f"synthetic_gan_{idx:05d}" for idx in range(len(synthetic_y))], dtype=str)
    train_ids = np.concatenate([dataset.train_ids.astype(str), synthetic_ids])
    augmented = replace_train_split(dataset, train_gene_x, train_y, train_ids)
    manifest = {
        **base_manifest,
        "status": "completed",
        "n_synthetic": int(len(synthetic_y)),
        "n_augmented_train": int(len(train_y)),
        "synthetic_class_counts": class_counts_dict(synthetic_y, dataset.class_names),
        "augmented_class_counts": class_counts_dict(train_y, dataset.class_names),
        "feature_value_policy": "sigmoid_generator_clip_to_0_1_after_minmax",
        **gan_realism_summary(dataset.train_gene_x, synthetic_x),
        **gan_metadata,
    }
    return augmented, manifest


def apply_train_augmentation(dataset: PreparedDataset, args: argparse.Namespace) -> tuple[PreparedDataset, dict[str, Any]]:
    if args.augmentation == "none":
        manifest = {
            "augmentation": "none",
            "status": "not_applied",
            "augmentation_scope": "not_applied",
            "validation_and_test_scope": "not_augmented",
            "n_original_train": int(len(dataset.train_y)),
            "n_synthetic": 0,
            "n_augmented_train": int(len(dataset.train_y)),
            "original_class_counts": class_counts_dict(dataset.train_y, dataset.class_names),
            "augmented_class_counts": class_counts_dict(dataset.train_y, dataset.class_names),
        }
        return dataset, manifest
    if args.augmentation == "smote":
        return apply_smote_augmentation(dataset, args)
    if args.augmentation == "borderline_smote":
        return apply_borderline_smote_augmentation(dataset, args)
    if args.augmentation == "pca_neighbor_mci_ctl":
        return apply_pca_neighbor_mci_ctl_augmentation(dataset, args)
    if args.augmentation == "pca_neighbor_all_tasks":
        return apply_pca_neighbor_all_tasks_augmentation(dataset, args)
    if args.augmentation == "ctgan":
        return apply_ctgan_augmentation(dataset, args)
    if args.augmentation == "gan":
        return apply_gan_augmentation(dataset, args)
    raise ValueError(f"Unsupported augmentation: {args.augmentation}")


def compute_task_class_weights(task_y: np.ndarray, task_mask: np.ndarray, task_specs: list[TaskSpec], enabled: bool) -> list[torch.Tensor | None]:
    weights: list[torch.Tensor | None] = []
    for task_idx, spec in enumerate(task_specs):
        if not enabled:
            weights.append(None)
            continue
        valid_y = task_y[task_mask[:, task_idx], task_idx]
        class_weights = compute_balanced_class_weights(valid_y)
        if len(class_weights) < len(spec.class_names):
            padded = np.ones(len(spec.class_names), dtype=np.float32)
            padded[: len(class_weights)] = class_weights
            class_weights = padded
        weights.append(torch.tensor(class_weights[: len(spec.class_names)], dtype=torch.float32))
    return weights


def multitask_loss(
    logits_by_task: list[torch.Tensor],
    labels: torch.Tensor,
    masks: torch.Tensor,
    criteria: list[nn.Module],
    task_loss_weights: list[float] | tuple[float, ...] | np.ndarray,
) -> tuple[torch.Tensor, dict[str, float]]:
    task_losses, loss_values = compute_task_losses(logits_by_task, labels, masks, criteria)
    losses: list[torch.Tensor] = []
    active_weights: list[float] = []
    for task_idx, task_loss in enumerate(task_losses):
        if task_loss is not None:
            losses.append(task_loss)
            active_weights.append(float(task_loss_weights[task_idx]))
    if not losses:
        raise ValueError("Batch has no valid samples for any task.")
    weights = torch.tensor(active_weights, dtype=losses[0].dtype, device=losses[0].device)
    return torch.sum(torch.stack(losses) * weights) / weights.sum(), loss_values


def compute_task_losses(
    logits_by_task: list[torch.Tensor],
    labels: torch.Tensor,
    masks: torch.Tensor,
    criteria: list[nn.Module],
) -> tuple[list[torch.Tensor | None], dict[str, float]]:
    task_losses: list[torch.Tensor | None] = []
    values: dict[str, float] = {}
    for task_idx, task_name in enumerate(TASK_NAMES):
        valid = masks[:, task_idx]
        if bool(valid.any()):
            task_loss = criteria[task_idx](logits_by_task[task_idx][valid], labels[valid, task_idx])
            task_losses.append(task_loss)
            values[f"{task_name}_loss"] = float(task_loss.detach().cpu().item())
        else:
            task_losses.append(None)
            values[f"{task_name}_loss"] = math.nan
    return task_losses, values


def task_gradient_vectors(
    task_losses: list[torch.Tensor | None],
    shared_parameters: list[nn.Parameter],
) -> list[list[torch.Tensor | None] | None]:
    gradients: list[list[torch.Tensor | None] | None] = []
    for task_loss in task_losses:
        if task_loss is None:
            gradients.append(None)
            continue
        task_grads = torch.autograd.grad(
            task_loss, shared_parameters, retain_graph=True, allow_unused=True
        )
        gradients.append(
            [gradient.detach().clone() if gradient is not None else None for gradient in task_grads]
        )
    return gradients


def gradient_dot(left: list[torch.Tensor | None], right: list[torch.Tensor | None]) -> torch.Tensor:
    terms = [
        torch.sum(left_value * right_value)
        for left_value, right_value in zip(left, right)
        if left_value is not None and right_value is not None
    ]
    if terms:
        return torch.stack(terms).sum()
    device = next((value.device for value in left if value is not None), torch.device("cpu"))
    return torch.tensor(0.0, device=device)


def gradient_cosine(left: list[torch.Tensor | None], right: list[torch.Tensor | None]) -> float:
    dot = gradient_dot(left, right)
    denominator = torch.sqrt(torch.clamp(gradient_dot(left, left), min=0.0)) * torch.sqrt(
        torch.clamp(gradient_dot(right, right), min=0.0)
    )
    if float(denominator.detach().cpu().item()) <= 0:
        return math.nan
    return float((dot / denominator).detach().cpu().item())


def summarize_task_gradients(
    gradients: list[list[torch.Tensor | None] | None],
) -> dict[str, float]:
    summary: dict[str, float] = {}
    for task_idx, task_name in enumerate(TASK_NAMES):
        task_gradient = gradients[task_idx]
        summary[f"grad_norm_{task_name}"] = (
            float(torch.sqrt(torch.clamp(gradient_dot(task_gradient, task_gradient), min=0.0)).cpu().item())
            if task_gradient is not None
            else math.nan
        )
    for left_idx in range(len(TASK_NAMES)):
        for right_idx in range(left_idx + 1, len(TASK_NAMES)):
            key = f"grad_cos_{TASK_NAMES[left_idx]}__{TASK_NAMES[right_idx]}"
            left = gradients[left_idx]
            right = gradients[right_idx]
            summary[key] = gradient_cosine(left, right) if left is not None and right is not None else math.nan
    return summary


def overwrite_shared_gradients_primary_protected(
    shared_parameters: list[nn.Parameter],
    gradients: list[list[torch.Tensor | None] | None],
    task_loss_weights: list[float],
) -> int:
    primary = gradients[0]
    if primary is None:
        return 0
    projected: list[list[torch.Tensor | None] | None] = [primary]
    projection_count = 0
    primary_norm_squared = gradient_dot(primary, primary)
    for auxiliary in gradients[1:]:
        if auxiliary is None:
            projected.append(None)
            continue
        dot = gradient_dot(auxiliary, primary)
        if float(dot.detach().cpu().item()) < 0 and float(primary_norm_squared.detach().cpu().item()) > 0:
            coefficient = dot / primary_norm_squared
            auxiliary = [
                aux_value - coefficient * primary_value
                if aux_value is not None and primary_value is not None
                else aux_value
                for aux_value, primary_value in zip(auxiliary, primary)
            ]
            projection_count += 1
        projected.append(auxiliary)
    active_weight_sum = sum(
        float(task_loss_weights[idx]) for idx, task_gradient in enumerate(projected) if task_gradient is not None
    )
    for parameter_idx, parameter in enumerate(shared_parameters):
        pieces = [
            float(task_loss_weights[task_idx]) * task_gradient[parameter_idx]
            for task_idx, task_gradient in enumerate(projected)
            if task_gradient is not None and task_gradient[parameter_idx] is not None
        ]
        parameter.grad = torch.stack(pieces).sum(dim=0) / active_weight_sum if pieces else None
    return projection_count


@torch.no_grad()
def collect_logits(
    model: TxT,
    gene_x: np.ndarray,
    task_y: np.ndarray,
    task_mask: np.ndarray,
    batch_size: int,
    device: torch.device,
    max_batches: int | None = None,
    mask_aware_heads: bool = False,
    diagnostic_batches: list[tuple[int, dict[str, Any]]] | None = None,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    loader = DataLoader(MultitaskDataset(gene_x, task_y, task_mask), batch_size=batch_size, shuffle=False)
    model.eval()
    logits_by_task: list[list[np.ndarray]] = [[] for _ in TASK_NAMES]
    labels: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    for batch_idx, (batch_x, batch_y, batch_mask) in enumerate(loader, start=1):
        outputs = model(
            batch_x.to(device),
            task_sample_mask=batch_mask.to(device) if mask_aware_heads else None,
        )
        if diagnostic_batches is not None:
            diagnostic_batches.append(
                (int(batch_x.size(0)), model_volumetric_diagnostics(model))
            )
        for task_idx, logits in enumerate(outputs):
            logits_by_task[task_idx].append(logits.detach().cpu().numpy())
        labels.append(batch_y.numpy())
        masks.append(batch_mask.numpy())
        if max_batches is not None and batch_idx >= max_batches:
            break
    return [np.concatenate(parts, axis=0) for parts in logits_by_task], np.concatenate(labels, axis=0), np.concatenate(masks, axis=0)


def task_metrics(y_true: np.ndarray, y_prob: np.ndarray, class_names: list[str]) -> dict[str, Any]:
    if len(y_true) == 0:
        report_df = pd.DataFrame(
            [
                {"class": class_name, "precision": math.nan, "recall": math.nan, "f1_score": math.nan, "support": 0}
                for class_name in class_names
            ]
            + [
                {"class": "macro avg", "precision": math.nan, "recall": math.nan, "f1_score": math.nan, "support": 0},
                {"class": "weighted avg", "precision": math.nan, "recall": math.nan, "f1_score": math.nan, "support": 0},
                {"class": "accuracy", "precision": math.nan, "recall": math.nan, "f1_score": math.nan, "support": 0},
            ]
        )
        return {
            "samples": 0,
            "accuracy": math.nan,
            "macro_f1": math.nan,
            "weighted_f1": math.nan,
            "balanced_accuracy": math.nan,
            "roc_auc": math.nan,
            "report_df": report_df,
            "y_pred": np.asarray([], dtype=np.int64),
        }
    y_pred = y_prob.argmax(axis=1).astype(np.int64)
    report_df = build_report_dataframe(y_true, y_pred, class_names)
    rows = {row["class"]: row for row in report_df.to_dict(orient="records")}
    recalls = [rows[name]["recall"] for name in class_names if name in rows]
    payload: dict[str, Any] = {
        "samples": int(len(y_true)),
        "accuracy": float(rows["accuracy"]["f1_score"]),
        "macro_f1": float(rows["macro avg"]["f1_score"]),
        "weighted_f1": float(rows["weighted avg"]["f1_score"]),
        "balanced_accuracy": float(np.mean(recalls)) if recalls else math.nan,
        "report_df": report_df,
        "y_pred": y_pred,
    }
    if len(class_names) == 2:
        payload["roc_auc"] = binary_roc_auc_np((y_true == 1).astype(np.int64), y_prob[:, 1])
    else:
        aucs = one_vs_rest_roc_auc(y_true, y_prob, class_names)
        payload.update(aucs)
        payload["roc_auc"] = aucs["roc_auc_ovr_macro"]
    return payload


def evaluate_multitask(
    model: TxT,
    dataset: PreparedDataset,
    split_name: str,
    gene_x: np.ndarray,
    source_y: np.ndarray,
    sample_ids: np.ndarray,
    task_specs: list[TaskSpec],
    batch_size: int,
    device: torch.device,
    criteria: list[nn.Module],
    task_loss_weights: list[float] | tuple[float, ...] | np.ndarray,
    max_batches: int | None = None,
    state_dicts: list[dict[str, torch.Tensor]] | None = None,
    mask_aware_heads: bool = False,
    volumetric_diagnostics_collector: dict[str, Any] | None = None,
) -> tuple[dict[str, dict[str, Any]], float]:
    task_y, task_mask = make_task_targets_for_samples(source_y, task_specs, sample_ids)
    probabilities_by_task: list[np.ndarray] | None = None
    diagnostic_runs: list[list[tuple[int, dict[str, Any]]]] = []
    if state_dicts and len(state_dicts) > 1:
        probability_runs: list[list[np.ndarray]] = []
        labels = masks = None
        for state_dict in state_dicts:
            model.load_state_dict(state_dict)
            diagnostic_batches: list[tuple[int, dict[str, Any]]] | None = (
                [] if volumetric_diagnostics_collector is not None else None
            )
            run_logits, run_labels, run_masks = collect_logits(
                model,
                gene_x,
                task_y,
                task_mask,
                batch_size,
                device,
                max_batches,
                mask_aware_heads,
                diagnostic_batches,
            )
            if diagnostic_batches is not None:
                diagnostic_runs.append(diagnostic_batches)
            probability_runs.append([softmax_np(task_logits) for task_logits in run_logits])
            if labels is None:
                labels, masks = run_labels, run_masks
        assert labels is not None and masks is not None
        probabilities_by_task = [
            np.mean([run[task_idx] for run in probability_runs], axis=0)
            for task_idx in range(len(TASK_NAMES))
        ]
        logits_by_task = [np.log(np.clip(probabilities, 1e-8, 1.0)) for probabilities in probabilities_by_task]
    else:
        if state_dicts:
            model.load_state_dict(state_dicts[0])
        diagnostic_batches = [] if volumetric_diagnostics_collector is not None else None
        logits_by_task, labels, masks = collect_logits(
            model,
            gene_x,
            task_y,
            task_mask,
            batch_size,
            device,
            max_batches,
            mask_aware_heads,
            diagnostic_batches,
        )
        if diagnostic_batches is not None:
            diagnostic_runs.append(diagnostic_batches)
    if volumetric_diagnostics_collector is not None:
        volumetric_diagnostics_collector.clear()
        volumetric_diagnostics_collector.update(
            aggregate_volumetric_diagnostic_runs(
                diagnostic_runs,
                split_name=split_name,
                batch_size=batch_size,
                samples_evaluated=len(labels),
                full_split=max_batches is None or len(labels) >= len(gene_x),
            )
        )
    evaluated_sample_ids = sample_ids[: len(labels)]
    split_results: dict[str, dict[str, Any]] = {}
    losses: list[float] = []
    loss_weights: list[float] = []
    for task_idx, spec in enumerate(task_specs):
        valid = masks[:, task_idx]
        valid_logits = torch.tensor(logits_by_task[task_idx][valid], dtype=torch.float32, device=device)
        valid_labels = torch.tensor(labels[valid, task_idx], dtype=torch.long, device=device)
        loss = float(criteria[task_idx](valid_logits, valid_labels).item()) if len(valid_labels) else math.nan
        if not math.isnan(loss):
            losses.append(loss)
            loss_weights.append(float(task_loss_weights[task_idx]))
        y_true = labels[valid, task_idx]
        if len(y_true) == 0:
            y_prob = np.empty((0, len(spec.class_names)), dtype=np.float32)
        else:
            y_prob = (
                probabilities_by_task[task_idx][valid]
                if probabilities_by_task is not None
                else softmax_np(logits_by_task[task_idx][valid])
            )
        metrics = task_metrics(y_true, y_prob, spec.class_names)
        confusion = np.zeros((len(spec.class_names), len(spec.class_names)), dtype=np.int64)
        for true_value, pred_value in zip(y_true, metrics["y_pred"]):
            confusion[int(true_value), int(pred_value)] += 1
        split_results[spec.name] = {
            "split": split_name,
            "task": spec.name,
            "sample_ids": evaluated_sample_ids[valid],
            "y_true": y_true,
            "y_pred": metrics["y_pred"],
            "y_prob": y_prob,
            "loss": loss,
            "confusion_matrix": confusion,
            "class_names": spec.class_names,
            **{key: value for key, value in metrics.items() if key not in {"report_df", "y_pred"}},
            "report_df": metrics["report_df"],
        }
    if not losses:
        return split_results, math.nan
    return split_results, float(np.average(losses, weights=loss_weights))


def checkpoint_improved(current: float, best: float, metric: str) -> bool:
    if metric == "val_loss":
        return current <= best
    return current >= best


def update_checkpoint_ensemble(
    entries: list[dict[str, Any]],
    current_value: float,
    epoch: int,
    state_dict: dict[str, torch.Tensor],
    metric: str,
    ensemble_size: int,
    min_gap: int,
) -> list[dict[str, Any]]:
    if math.isnan(float(current_value)):
        return entries
    candidate = {"value": float(current_value), "epoch": int(epoch), "state": copy.deepcopy(state_dict)}
    nearby = [idx for idx, entry in enumerate(entries) if abs(int(entry["epoch"]) - epoch) < min_gap]
    if nearby:
        best_nearby_idx = nearby[0]
        for idx in nearby[1:]:
            if checkpoint_improved(float(entries[idx]["value"]), float(entries[best_nearby_idx]["value"]), metric):
                best_nearby_idx = idx
        if checkpoint_improved(current_value, float(entries[best_nearby_idx]["value"]), metric):
            entries[best_nearby_idx] = candidate
    else:
        entries.append(candidate)
    reverse = metric != "val_loss"
    entries.sort(key=lambda entry: float(entry["value"]), reverse=reverse)
    return entries[:ensemble_size]


def partition_volumetric_parameters(
    model: nn.Module,
) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """Return deduplicated ``(non_volumetric, volumetric)`` model parameters."""

    all_parameters = list(model.parameters())
    volumetric_parameter_ids: set[int] = set()
    iterator = getattr(model, "iter_volumetric_augmentations", None)
    if callable(iterator):
        for _, augmentation in iterator():
            volumetric_parameter_ids.update(id(parameter) for parameter in augmentation.parameters())
    non_volumetric = [
        parameter for parameter in all_parameters if id(parameter) not in volumetric_parameter_ids
    ]
    volumetric = [
        parameter for parameter in all_parameters if id(parameter) in volumetric_parameter_ids
    ]
    return non_volumetric, volumetric


def clip_model_gradients(
    model: nn.Module,
    max_norm: float,
    scope: str,
    model_variant: str,
) -> None:
    """Apply legacy joint clipping or independent non-VMA/VMA clipping."""

    if scope == "separate_volumetric" and model_variant == "ppi_volumetric":
        non_volumetric, volumetric = partition_volumetric_parameters(model)
        if non_volumetric:
            torch.nn.utils.clip_grad_norm_(non_volumetric, max_norm)
        if volumetric:
            torch.nn.utils.clip_grad_norm_(volumetric, max_norm)
        return
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)


def train_model(
    model: TxT,
    dataset: PreparedDataset,
    task_specs: list[TaskSpec],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[
    list[dict[str, Any]],
    dict[str, torch.Tensor],
    list[dict[str, Any]],
    dict[str, Any],
    list[nn.Module],
]:
    train_task_y, train_task_mask = make_task_targets_for_samples(
        dataset.train_y, task_specs, dataset.train_ids
    )
    loss_task_weights = [float(value) for value in args.task_loss_weights]
    loader_generator = torch.Generator()
    loader_generator.manual_seed(args.seed)
    train_torch_dataset = MultitaskDataset(dataset.train_gene_x, train_task_y, train_task_mask)
    balanced_batch_sampler: BalancedClassBatchSampler | None = None
    if args.train_sampling == "balanced_classes":
        balanced_batch_sampler = BalancedClassBatchSampler(dataset.train_y, args.batch_size, args.seed)
        train_loader = DataLoader(train_torch_dataset, batch_sampler=balanced_batch_sampler)
    else:
        train_loader = DataLoader(
            train_torch_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=args.batch_size > 1 and len(dataset.train_y) >= args.batch_size,
            generator=loader_generator,
        )
    task_weights = compute_task_class_weights(train_task_y, train_task_mask, task_specs, args.class_weighting == "on")
    criteria = [
        nn.CrossEntropyLoss(
            weight=weight.to(device) if weight is not None else None,
            label_smoothing=args.label_smoothing,
        )
        for weight in task_weights
    ]
    embedding_params = list(model.embedding_parameters())
    encoder_params = list(model.encoder_parameters_without_embeddings())
    all_encoder_params = list(encoder_params)
    shared_params = [parameter for parameter in [*embedding_params, *all_encoder_params] if parameter.requires_grad]
    encoder_parameter_ids = {id(parameter) for parameter in [*embedding_params, *all_encoder_params]}
    task_specific_params = [parameter for parameter in model.parameters() if id(parameter) not in encoder_parameter_ids]
    optimizer_groups: list[dict[str, Any]] = [
            {
                "params": embedding_params,
                "lr": args.lr_encoder if args.lr_embedding is None else args.lr_embedding,
                "group_name": "embedding",
            },
    ]
    if args.lr_volumetric is None or args.model_variant != "ppi_volumetric":
        optimizer_groups.append(
            {"params": encoder_params, "lr": args.lr_encoder, "group_name": "encoder"}
        )
    else:
        _, volumetric_parameters = partition_volumetric_parameters(model)
        volumetric_ids = {id(parameter) for parameter in volumetric_parameters}
        encoder_params = [parameter for parameter in all_encoder_params if id(parameter) not in volumetric_ids]
        volumetric_encoder_params = [
            parameter for parameter in all_encoder_params if id(parameter) in volumetric_ids
        ]
        optimizer_groups.append(
            {"params": encoder_params, "lr": args.lr_encoder, "group_name": "encoder"}
        )
        optimizer_groups.append(
            {
                "params": volumetric_encoder_params,
                "lr": args.lr_volumetric,
                "group_name": "volumetric",
            }
        )
    optimizer_groups.append(
        {"params": task_specific_params, "lr": args.lr_head, "group_name": "task_specific"}
    )
    optimizer = torch.optim.AdamW(optimizer_groups, weight_decay=args.weight_decay)
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_value = float("inf") if args.checkpoint_metric == "val_loss" else float("-inf")
    epochs_without_improvement = 0
    epochs_over_val_loss_threshold = 0
    stop_reason = "max_epochs"
    history: list[dict[str, Any]] = []
    checkpoint_ensemble: list[dict[str, Any]] = []

    for epoch in range(1, args.epochs + 1):
        if balanced_batch_sampler is not None:
            balanced_batch_sampler.set_epoch(epoch - 1)
        embedding_frozen = epoch <= args.freeze_embedding_epochs
        for parameter in embedding_params:
            parameter.requires_grad_(not embedding_frozen)
        model.train()
        total_loss = 0.0
        total_samples = 0
        task_loss_accumulator: dict[str, list[float]] = {f"{name}_loss": [] for name in TASK_NAMES}
        gradient_accumulator: dict[str, list[float]] = {}
        epoch_projection_count = 0
        for batch_idx, (batch_x, batch_y, batch_mask) in enumerate(train_loader, start=1):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_mask = batch_mask.to(device)
            optimizer.zero_grad()
            outputs = model(
                batch_x,
                task_sample_mask=batch_mask if args.mask_aware_heads == "on" else None,
            )
            loss, task_loss_values = multitask_loss(outputs, batch_y, batch_mask, criteria, loss_task_weights)
            needs_task_gradients = (
                args.gradient_strategy == "primary_protected_pcgrad"
                or (
                    args.gradient_diagnostics == "on"
                    and (batch_idx - 1) % args.gradient_diagnostic_interval == 0
                )
            )
            gradients = None
            if needs_task_gradients:
                task_loss_tensors, _ = compute_task_losses(outputs, batch_y, batch_mask, criteria)
                active_shared_params = [parameter for parameter in shared_params if parameter.requires_grad]
                gradients = task_gradient_vectors(task_loss_tensors, active_shared_params)
                diagnostic_values = summarize_task_gradients(gradients)
                for key, value in diagnostic_values.items():
                    if not math.isnan(value):
                        gradient_accumulator.setdefault(key, []).append(value)
            loss.backward()
            if args.gradient_strategy == "primary_protected_pcgrad":
                assert gradients is not None
                active_shared_params = [parameter for parameter in shared_params if parameter.requires_grad]
                epoch_projection_count += overwrite_shared_gradients_primary_protected(
                    active_shared_params, gradients, loss_task_weights
                )
            if args.grad_clip_norm > 0:
                clip_model_gradients(
                    model,
                    args.grad_clip_norm,
                    getattr(args, "grad_clip_scope", "joint"),
                    args.model_variant,
                )
            optimizer.step()
            total_loss += float(loss.item()) * int(batch_x.size(0))
            total_samples += int(batch_x.size(0))
            for key, value in task_loss_values.items():
                if not math.isnan(value):
                    task_loss_accumulator[key].append(value)
            if args.max_train_batches is not None and batch_idx >= args.max_train_batches:
                break

        val_results, val_loss = evaluate_multitask(
            model,
            dataset,
            "val",
            dataset.val_gene_x,
            dataset.val_y,
            dataset.val_ids,
            task_specs,
            args.batch_size,
            device,
            criteria,
            loss_task_weights,
            max_batches=args.max_val_batches,
            mask_aware_heads=args.mask_aware_heads == "on",
        )
        checkpoint_value, checkpoint_details = compute_checkpoint_value(
            args.checkpoint_metric,
            val_results,
            val_loss,
            primary_task="AD_vs_MCI",
        )
        if args.val_loss_stop_threshold is not None and not math.isnan(val_loss) and val_loss > args.val_loss_stop_threshold:
            epochs_over_val_loss_threshold += 1
        else:
            epochs_over_val_loss_threshold = 0
        val_multitask_auc_mean = checkpoint_details["val_multitask_auc_mean"]
        val_macro_f1_mean = checkpoint_details["val_multitask_macro_f1_mean"]
        row = {
            "epoch": epoch,
            "embedding_frozen": embedding_frozen,
            "train_loss": total_loss / max(total_samples, 1),
            "val_loss": val_loss,
            "val_loss_stop_threshold": args.val_loss_stop_threshold,
            "epochs_over_val_loss_threshold": epochs_over_val_loss_threshold,
            **checkpoint_details,
            "checkpoint_metric": args.checkpoint_metric,
            "checkpoint_value": checkpoint_value,
            "gradient_strategy": args.gradient_strategy,
            "gradient_projection_count": epoch_projection_count,
        }
        for key, values in gradient_accumulator.items():
            row[key] = float(np.mean(values)) if values else math.nan
            if key.startswith("grad_cos_"):
                row[f"{key}_conflict_fraction"] = float(np.mean(np.asarray(values) < 0)) if values else math.nan
        for gate_name, gate_value in model_ppi_gate_values(model).items():
            row[f"ppi_gate_{gate_name}"] = gate_value
        row.update(volumetric_training_log_fields(model))
        for key, values in task_loss_accumulator.items():
            row[f"train_{key}"] = float(np.mean(values)) if values else math.nan
        for task_name, result in val_results.items():
            row[f"val_{task_name}_macro_f1"] = result["macro_f1"]
            row[f"val_{task_name}_accuracy"] = result["accuracy"]
            row[f"val_{task_name}_roc_auc"] = result["roc_auc"]
        if args.evaluate_test_each_epoch == "on":
            test_results, test_loss = evaluate_multitask(
                model,
                dataset,
                "test",
                dataset.test_gene_x,
                dataset.test_y,
                dataset.test_ids,
                task_specs,
                args.batch_size,
                device,
                criteria,
                loss_task_weights,
                mask_aware_heads=args.mask_aware_heads == "on",
            )
            test_auc_values = [
                result["roc_auc"]
                for result in test_results.values()
                if result.get("roc_auc") is not None and not math.isnan(result["roc_auc"])
            ]
            test_macro_f1_values = [
                result["macro_f1"]
                for result in test_results.values()
                if result.get("macro_f1") is not None and not math.isnan(result["macro_f1"])
            ]
            row["test_loss"] = test_loss
            row["test_multitask_auc_mean"] = float(np.mean(test_auc_values)) if test_auc_values else math.nan
            row["test_multitask_macro_f1_mean"] = (
                float(np.mean(test_macro_f1_values)) if test_macro_f1_values else math.nan
            )
            for task_name, result in test_results.items():
                row[f"test_{task_name}_macro_f1"] = result["macro_f1"]
                row[f"test_{task_name}_accuracy"] = result["accuracy"]
                row[f"test_{task_name}_roc_auc"] = result["roc_auc"]
        history.append(row)
        checkpoint_ensemble = update_checkpoint_ensemble(
            checkpoint_ensemble,
            checkpoint_value,
            epoch,
            model.state_dict(),
            args.checkpoint_metric,
            args.checkpoint_ensemble_size,
            args.checkpoint_ensemble_min_gap,
        )
        task_auc_text = " | ".join(
            f"val_auc_{task_name}={val_results[task_name]['roc_auc']:.4f}"
            for task_name in TASK_NAMES
        )
        print(
            f"Epoch {epoch:03d} | train_loss={row['train_loss']:.4f} | "
            f"val_loss={val_loss:.4f} | val_multitask_auc_mean={val_multitask_auc_mean:.4f} | "
            f"{task_auc_text} | val_multitask_macro_f1_mean={val_macro_f1_mean:.4f}",
            f"| checkpoint={checkpoint_value:.4f}",
            flush=True,
        )
        if checkpoint_improved(checkpoint_value, best_value, args.checkpoint_metric):
            best_value = checkpoint_value
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        stop_reasons = []
        if args.early_stopping_patience > 0 and epochs_without_improvement >= args.early_stopping_patience:
            stop_reasons.append("checkpoint_patience")
        if (
            args.val_loss_stop_threshold is not None
            and args.val_loss_stop_patience > 0
            and epochs_over_val_loss_threshold >= args.val_loss_stop_patience
        ):
            stop_reasons.append("val_loss_threshold_patience")
        if stop_reasons:
            stop_reason = "+".join(stop_reasons)
            print(f"Early stopping at epoch {epoch}: {stop_reason}", flush=True)
            break

    ensemble_metadata = [
        {"rank": rank, "epoch": entry["epoch"], "checkpoint_value": entry["value"]}
        for rank, entry in enumerate(checkpoint_ensemble, start=1)
    ]
    return history, best_state, checkpoint_ensemble, {
        "best_epoch": best_epoch,
        "checkpoint_metric": args.checkpoint_metric,
        "best_checkpoint_value": best_value,
        "val_loss_stop_threshold": args.val_loss_stop_threshold,
        "val_loss_stop_patience": args.val_loss_stop_patience,
        "epochs_ran": len(history),
        "stopped_early": len(history) < args.epochs,
        "stop_reason": stop_reason,
        "embedding_lr": args.lr_encoder if args.lr_embedding is None else args.lr_embedding,
        "volumetric_lr": args.lr_encoder if args.lr_volumetric is None else args.lr_volumetric,
        "freeze_embedding_epochs": args.freeze_embedding_epochs,
        "task_loss_weights": dict(zip(TASK_NAMES, loss_task_weights)),
        "train_sampling": args.train_sampling,
        "train_batches_per_epoch": len(train_loader),
        "balanced_samples_per_class_per_batch": (
            balanced_batch_sampler.samples_per_class if balanced_batch_sampler is not None else None
        ),
        "mask_aware_heads": args.mask_aware_heads,
        "gradient_strategy": args.gradient_strategy,
        "gradient_diagnostics": args.gradient_diagnostics,
        "gradient_diagnostic_interval": args.gradient_diagnostic_interval,
        "grad_clip_norm": float(args.grad_clip_norm),
        "grad_clip_scope": getattr(args, "grad_clip_scope", "joint"),
        "final_ppi_gate_values": model_ppi_gate_values(model),
        "final_volumetric_gate_values": model_volumetric_gate_values(model),
        "final_volumetric_diagnostics": model_volumetric_diagnostics(model),
        "checkpoint_ensemble": ensemble_metadata,
    }, criteria


def save_split_artifacts(result_dir: Path, split_results: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    predictions_dir = result_dir / "task_predictions"
    confusion_dir = result_dir / "task_confusion_matrices"
    reports_dir = result_dir / "task_classification_reports"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    confusion_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    metric_rows: list[dict[str, Any]] = []
    for task_name, result in split_results.items():
        split = result["split"]
        safe = task_name.lower()
        predictions = pd.DataFrame(
            {
                "sample_id": result["sample_ids"],
                "y_true": result["y_true"],
                "y_true_name": [result["class_names"][idx] for idx in result["y_true"]],
                "y_pred": result["y_pred"],
                "y_pred_name": [result["class_names"][idx] for idx in result["y_pred"]],
            }
        )
        for class_idx, class_name in enumerate(result["class_names"]):
            predictions[f"prob_{class_name}"] = result["y_prob"][:, class_idx]
        predictions.to_csv(predictions_dir / f"{split}_{safe}.csv", index=False)
        pd.DataFrame(
            result["confusion_matrix"],
            index=result["class_names"],
            columns=result["class_names"],
        ).to_csv(confusion_dir / f"{split}_{safe}.csv")
        result["report_df"].to_csv(reports_dir / f"{split}_{safe}.csv", index=False)
        with (reports_dir / f"{split}_{safe}.txt").open("w", encoding="utf-8") as handle:
            handle.write(report_to_text(result["report_df"]))
            handle.write("\n")
        metric_row = {
            "split": split,
            "task": task_name,
            "samples": result["samples"],
            "loss": result["loss"],
            "accuracy": result["accuracy"],
            "macro_f1": result["macro_f1"],
            "weighted_f1": result["weighted_f1"],
            "balanced_accuracy": result["balanced_accuracy"],
            "roc_auc": result["roc_auc"],
        }
        metric_row.update({key: value for key, value in result.items() if key.startswith("roc_auc_")})
        metric_rows.append(metric_row)
    return metric_rows


def save_label_mask_summary(result_dir: Path, dataset: PreparedDataset, task_specs: list[TaskSpec]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for split_name, labels, sample_ids in [
        ("train", dataset.train_y, dataset.train_ids),
        ("val", dataset.val_y, dataset.val_ids),
        ("test", dataset.test_y, dataset.test_ids),
    ]:
        task_y, task_mask = make_task_targets_for_samples(labels, task_specs, sample_ids)
        for task_idx, spec in enumerate(task_specs):
            valid_y = task_y[task_mask[:, task_idx], task_idx]
            counts = {spec.class_names[class_idx]: int((valid_y == class_idx).sum()) for class_idx in range(len(spec.class_names))}
            rows.append({"split": split_name, "task": spec.name, "valid_samples": int(task_mask[:, task_idx].sum()), **counts})
            summary[f"{split_name}_{spec.name}"] = {"valid_samples": int(task_mask[:, task_idx].sum()), "class_counts": counts}
    pd.DataFrame(rows).to_csv(result_dir / "label_mask_summary.csv", index=False)
    return summary


@torch.no_grad()
def save_trained_embedding_artifacts(model: nn.Module, gene_names: list[str], result_dir: Path) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    modules = model.encoder_modules()
    names = ["shared"] if model.encoder_sharing == "shared" else list(model.task_names)
    for name, transformer in zip(names, modules):
        values = transformer.encoder.embed.embed.weight.detach().cpu().numpy()
        safe_name = re.sub(r"[^0-9A-Za-z_]+", "_", name).strip("_").lower()
        output_name = "trained_gene_embedding.csv" if name == "shared" else f"trained_gene_embedding_{safe_name}.csv"
        frame = pd.DataFrame(values, index=gene_names)
        frame.index.name = "Gene"
        frame.to_csv(result_dir / output_name)
        row_norms = np.linalg.norm(values, axis=1)
        summaries.append(
            {
                "encoder": name,
                "file": output_name,
                "mean": float(values.mean()),
                "std": float(values.std()),
                "mean_row_l2_norm": float(row_norms.mean()),
                "median_row_l2_norm": float(np.median(row_norms)),
            }
        )
    save_json(result_dir / "trained_embedding_summary.json", summaries)
    return summaries


def build_txt_model(
    args: argparse.Namespace,
    dataset: PreparedDataset,
    task_specs: list[TaskSpec],
    task_gene_indices: dict[str, list[int]] | None,
    temp_embed_path: Path,
    temp_ppi_prior_path: Path | None,
    volumetric_graph: Any | None,
) -> nn.Module:
    model_kwargs = {
        "embed_file": str(temp_embed_path),
        "gene_list": dataset.gene_names,
        "n_heads": args.n_heads,
        "d_model": args.d_model,
        "dropout": args.dropout,
        "d_ff": args.d_ff,
        "norm_first": args.norm_first,
        "n_layers": args.n_layers,
        "aggfunc": args.aggfunc,
        "d_hidden1": args.d_hidden1,
        "d_hidden2": args.d_hidden2,
        "slope": args.slope,
        "d_output_dict": {spec.name: len(spec.class_names) for spec in task_specs},
        "task_gene_indices": task_gene_indices,
        "head_norm": args.head_norm,
        "encoder_sharing": args.encoder_sharing,
        "pooling_mode": args.pooling_mode,
        "attention_pooling_hidden_dim": args.attention_pooling_hidden_dim,
        "attention_pooling_dropout": args.attention_pooling_dropout,
        "primary_adapter_dim": args.primary_adapter_dim,
        "expression_residual": args.expression_residual,
        "tupe_mode": getattr(args, "tupe_mode", "on"),
        "ppi_prior_file": str(temp_ppi_prior_path) if temp_ppi_prior_path is not None else None,
        "ppi_gate_init": args.ppi_gate_init,
    }
    if args.model_variant == "baseline":
        return TxT(**model_kwargs)
    if volumetric_graph is None:
        raise RuntimeError("The ppi_volumetric model requires an induced PPI graph.")
    from source.models.txt_volumetric import TxTVolumetric

    return TxTVolumetric(
        **model_kwargs,
        edge_index=volumetric_graph.edge_index,
        ppi_edge_scores=volumetric_graph.edge_scores,
        ppi_score_threshold=args.ppi_score_threshold,
        volumetric_beta=args.volumetric_beta,
        volumetric_eps=args.volumetric_eps,
        volumetric_gate_init=args.volumetric_gate_init,
        volumetric_volume_mode=args.volumetric_volume_mode,
        volumetric_dropout=args.volumetric_dropout,
        volumetric_message_mode=args.volumetric_message_mode,
        volumetric_output_norm=args.volumetric_output_norm,
        volumetric_gate_mode=args.volumetric_gate_mode,
        volumetric_backbone_gradient_mode=args.volumetric_backbone_gradient_mode,
    )


def build_evaluation_criteria(
    dataset: PreparedDataset,
    task_specs: list[TaskSpec],
    args: argparse.Namespace,
    device: torch.device,
) -> list[nn.Module]:
    train_task_y, train_task_mask = make_task_targets_for_samples(
        dataset.train_y, task_specs, dataset.train_ids
    )
    task_weights = compute_task_class_weights(
        train_task_y,
        train_task_mask,
        task_specs,
        args.class_weighting == "on",
    )
    return [
        nn.CrossEntropyLoss(
            weight=weight.to(device) if weight is not None else None,
            label_smoothing=args.label_smoothing,
        )
        for weight in task_weights
    ]


def load_checkpoint_state(path: Path, device: torch.device) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Evaluation checkpoint not found: {resolved}")
    try:
        payload = torch.load(resolved, map_location=device, weights_only=True)
    except TypeError:
        payload = torch.load(resolved, map_location=device)
    if isinstance(payload, dict):
        for key in ("state_dict", "model_state_dict"):
            nested = payload.get(key)
            if isinstance(nested, dict):
                payload = nested
                break
    if not isinstance(payload, dict) or not all(isinstance(key, str) for key in payload):
        raise ValueError(f"Checkpoint {resolved} does not contain a model state dict.")
    return payload


def validate_checkpoint_graph(
    state_dict: dict[str, Any],
    volumetric_graph: Any | None,
) -> None:
    if volumetric_graph is None:
        return
    graph_entries = [
        (key, value)
        for key, value in state_dict.items()
        if key.endswith("edge_index") and isinstance(value, torch.Tensor)
    ]
    if not graph_entries:
        raise ValueError(
            "The ppi_volumetric checkpoint does not contain a persistent edge_index buffer."
        )
    expected = volumetric_graph.edge_index.detach().cpu()
    for key, checkpoint_edges in graph_entries:
        if not torch.equal(checkpoint_edges.detach().cpu(), expected):
            raise ValueError(
                "Evaluation checkpoint graph does not match the graph induced from the current "
                f"split/gene selection/PPI arguments (buffer: {key})."
            )


def load_source_model_summary(checkpoint_path: Path) -> dict[str, Any]:
    summary_path = checkpoint_path.resolve().parent / "model_summary.json"
    if not summary_path.exists():
        return {}
    import json

    with summary_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


def resolve_evaluation_checkpoint_paths(args: argparse.Namespace) -> list[Path]:
    multiple = getattr(args, "evaluation_only_checkpoints", None)
    if multiple:
        return [Path(path) for path in multiple]
    single = getattr(args, "evaluation_only_checkpoint", None)
    return [Path(single)] if single is not None else []


def source_checkpoint_member_metadata(
    checkpoint_path: Path,
    source_model_summary: dict[str, Any],
) -> dict[str, Any]:
    """Recover epoch/score metadata from old and new model summaries when possible."""

    resolved = checkpoint_path.resolve()
    training_summary = source_model_summary.get("training_summary", {})
    if not isinstance(training_summary, dict):
        training_summary = {}
    metric = training_summary.get("checkpoint_metric")
    members = training_summary.get(
        "checkpoint_ensemble_members",
        training_summary.get("checkpoint_ensemble", []),
    )
    if not isinstance(members, list):
        members = []

    matching_member: dict[str, Any] | None = None
    for member in members:
        if not isinstance(member, dict):
            continue
        for path_key in ("path", "source_path", "copied_path"):
            member_path = member.get(path_key)
            if member_path is None:
                continue
            try:
                if Path(member_path).resolve() == resolved:
                    matching_member = member
                    break
            except (OSError, TypeError, ValueError):
                continue
        if matching_member is not None:
            break

    rank_match = re.fullmatch(r"ensemble_checkpoint_rank(\d+)\.pt", resolved.name)
    if matching_member is None and rank_match is not None:
        source_rank = int(rank_match.group(1))
        matching_member = next(
            (
                member
                for member in members
                if isinstance(member, dict) and str(member.get("rank")) == str(source_rank)
            ),
            None,
        )

    if matching_member is None:
        epoch = training_summary.get("best_epoch", training_summary.get("source_best_epoch"))
        score = training_summary.get("best_checkpoint_value")
    else:
        epoch = matching_member.get("epoch")
        score = matching_member.get(
            "score",
            matching_member.get("checkpoint_value", matching_member.get("value")),
        )
        metric = matching_member.get("score_metric", metric)
    try:
        epoch_value = int(epoch) if epoch is not None else None
    except (TypeError, ValueError):
        epoch_value = None
    try:
        score_value = float(score) if score is not None else None
    except (TypeError, ValueError):
        score_value = None
    return {
        "path": str(resolved),
        "epoch": epoch_value,
        "score": score_value,
        "score_metric": str(metric) if metric is not None else None,
        "source_training_summary_found": bool(training_summary),
    }


def save_ensemble_checkpoint_artifacts(
    result_dir: Path,
    state_dicts: list[dict[str, Any]],
    source_paths: list[Path] | None = None,
) -> list[Path]:
    """Persist rank copies and make rank 1 the canonical best_model artifact."""

    if not state_dicts:
        raise ValueError("Cannot save an empty checkpoint ensemble.")
    result_dir.mkdir(parents=True, exist_ok=True)
    copied_paths: list[Path] = []
    for rank, state_dict in enumerate(state_dicts, start=1):
        destination = (result_dir / f"ensemble_checkpoint_rank{rank}.pt").resolve()
        source = source_paths[rank - 1].resolve() if source_paths is not None else None
        if source is None or source != destination:
            torch.save(state_dict, destination)
        copied_paths.append(destination)
    best_destination = (result_dir / "best_model.pt").resolve()
    rank_one_source = source_paths[0].resolve() if source_paths is not None else None
    if rank_one_source is None or rank_one_source != best_destination:
        torch.save(state_dicts[0], best_destination)
    return copied_paths


def validate_checkpoint_configuration(
    source_model_summary: dict[str, Any],
    args: argparse.Namespace,
    volumetric_graph: Any | None,
) -> None:
    if not source_model_summary:
        return
    architecture = source_model_summary.get("architecture", {})
    source_variant = source_model_summary.get(
        "model_variant", architecture.get("model_variant")
    )
    if source_variant is not None and source_variant != args.model_variant:
        raise ValueError(
            f"Checkpoint model_variant={source_variant!r} does not match {args.model_variant!r}."
        )
    source_tupe_mode = architecture.get("tupe_mode", "on")
    requested_tupe_mode = getattr(args, "tupe_mode", "on")
    if str(source_tupe_mode) != str(requested_tupe_mode):
        raise ValueError(
            f"Checkpoint tupe_mode={source_tupe_mode!r} does not match requested value "
            f"{requested_tupe_mode!r}."
        )
    if args.model_variant != "ppi_volumetric":
        return
    for key, current in (
        ("volumetric_beta", args.volumetric_beta),
        ("volumetric_eps", args.volumetric_eps),
    ):
        source_value = architecture.get(key)
        if source_value is not None and not math.isclose(
            float(source_value), float(current), rel_tol=1e-12, abs_tol=1e-12
        ):
            raise ValueError(
                f"Checkpoint {key}={source_value} does not match requested value {current}."
            )
    source_volume_mode = architecture.get("volumetric_volume_mode")
    if (
        source_volume_mode is not None
        and str(source_volume_mode) != args.volumetric_volume_mode
    ):
        raise ValueError(
            "Checkpoint volumetric_volume_mode="
            f"{source_volume_mode!r} does not match requested value "
            f"{args.volumetric_volume_mode!r}."
        )
    for key, current in (
        ("volumetric_message_mode", args.volumetric_message_mode),
        ("volumetric_output_norm", args.volumetric_output_norm),
        ("volumetric_gate_mode", args.volumetric_gate_mode),
        ("volumetric_backbone_gradient_mode", args.volumetric_backbone_gradient_mode),
    ):
        source_value = architecture.get(key)
        if source_value is not None and str(source_value) != str(current):
            raise ValueError(
                f"Checkpoint {key}={source_value!r} does not match requested value {current!r}."
            )
    source_volumetric_dropout = architecture.get("volumetric_dropout_effective")
    if source_volumetric_dropout is None and "volumetric_dropout" in architecture:
        source_volumetric_dropout = architecture["volumetric_dropout"]
        if source_volumetric_dropout is None:
            source_volumetric_dropout = architecture.get("dropout")
    if source_volumetric_dropout is not None:
        current_volumetric_dropout = (
            args.dropout if args.volumetric_dropout is None else args.volumetric_dropout
        )
        if not math.isclose(
            float(source_volumetric_dropout),
            float(current_volumetric_dropout),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "Checkpoint effective volumetric_dropout="
                f"{source_volumetric_dropout} does not match requested effective value "
                f"{current_volumetric_dropout}."
            )
    source_graph = source_model_summary.get("ppi_graph") or {}
    source_hash = source_graph.get("source_sha256")
    if (
        source_hash is not None
        and volumetric_graph is not None
        and source_hash != volumetric_graph.source_sha256
    ):
        raise ValueError("Checkpoint HIPPIE source hash does not match --ppi-edge-file.")


def main() -> None:
    args = parse_args()
    evaluation_checkpoint_paths = resolve_evaluation_checkpoint_paths(args)
    evaluation_only = bool(evaluation_checkpoint_paths)
    if args.freeze_embedding_epochs < 0:
        raise ValueError("--freeze-embedding-epochs must be non-negative.")
    if args.lr_volumetric is not None and (
        not math.isfinite(args.lr_volumetric) or args.lr_volumetric <= 0
    ):
        raise ValueError("--lr-volumetric must be finite and positive when provided.")
    if len(args.task_loss_weights) != len(TASK_NAMES) or any(weight <= 0 for weight in args.task_loss_weights):
        raise ValueError("--task-loss-weights requires three strictly positive values.")
    if args.primary_adapter_dim < 0:
        raise ValueError("--primary-adapter-dim must be non-negative.")
    if args.checkpoint_ensemble_size <= 0:
        raise ValueError("--checkpoint-ensemble-size must be positive.")
    if args.checkpoint_ensemble_min_gap < 0:
        raise ValueError("--checkpoint-ensemble-min-gap must be non-negative.")
    if args.gradient_diagnostic_interval <= 0:
        raise ValueError("--gradient-diagnostic-interval must be positive.")
    if not math.isfinite(args.volumetric_beta):
        raise ValueError("--volumetric-beta must be finite.")
    if not math.isfinite(args.volumetric_eps) or args.volumetric_eps <= 0:
        raise ValueError("--volumetric-eps must be finite and positive.")
    if not math.isfinite(args.volumetric_gate_init):
        raise ValueError("--volumetric-gate-init must be finite.")
    if args.volumetric_dropout is not None and (
        not math.isfinite(args.volumetric_dropout)
        or not 0.0 <= args.volumetric_dropout <= 1.0
    ):
        raise ValueError("--volumetric-dropout must be finite and in [0, 1].")
    if not math.isfinite(args.ppi_score_threshold):
        raise ValueError("--ppi-score-threshold must be finite.")
    if args.ppi_integration == "gated_residual" and args.embed_file is None:
        raise ValueError("--ppi-integration gated_residual requires --embed-file with the PPI embedding.")
    if args.gradient_strategy == "primary_protected_pcgrad" and args.encoder_sharing != "shared":
        raise ValueError("primary_protected_pcgrad requires --encoder-sharing shared.")
    if evaluation_only and args.skip_final_test:
        raise ValueError(
            "--skip-final-test cannot be combined with evaluation-only checkpoints, "
            "which explicitly evaluate test."
        )
    if not evaluation_only and args.skip_final_test:
        args.evaluate_test = "off"
    if not evaluation_only and args.evaluate_test_each_epoch == "on" and args.evaluate_test == "off":
        raise ValueError(
            "Test evaluation was disabled for model selection, but --evaluate-test-each-epoch is on."
        )
    if not args.ppi_edge_file.is_absolute():
        args.ppi_edge_file = (ROOT / args.ppi_edge_file).resolve()
    set_seed(args.seed)
    device = resolve_device(args.device)
    resolved_split_file = resolve_split_file(args)
    args.result_dir.mkdir(parents=True, exist_ok=True)

    dataset, gene_selection_details, task_gene_indices, candidate_gene_manifest = prepare_multitask_dataset(
        args,
        resolved_split_file,
    )
    volumetric_graph = prepare_volumetric_graph(args, dataset.gene_names)
    if evaluation_only:
        augmentation_manifest = {
            "augmentation": args.augmentation,
            "status": "skipped_evaluation_only",
            "augmentation_scope": "not_applied_in_evaluation_only_mode",
            "validation_and_test_scope": "not_augmented",
            "n_original_train": int(len(dataset.train_y)),
            "n_synthetic": 0,
            "n_augmented_train": int(len(dataset.train_y)),
            "original_class_counts": class_counts_dict(dataset.train_y, dataset.class_names),
            "augmented_class_counts": class_counts_dict(dataset.train_y, dataset.class_names),
        }
    else:
        dataset, augmentation_manifest = apply_train_augmentation(dataset, args)
    task_specs = build_task_specs(dataset.class_names)
    label_mask_summary = save_label_mask_summary(args.result_dir, dataset, task_specs)

    direct_embed_file = args.embed_file if args.ppi_integration == "direct" else None
    embedding_df, embedding_manifest = build_embedding_for_selected_genes(
        dataset.gene_names,
        args.embed_dim,
        args.seed,
        direct_embed_file,
        args.embedding_init_scale,
        args.embedding_rescale,
    )
    ppi_prior_df: pd.DataFrame | None = None
    ppi_prior_manifest: dict[str, Any] | None = None
    if args.ppi_integration == "gated_residual":
        assert args.embed_file is not None
        ppi_prior_df, ppi_prior_manifest = build_row_normalized_ppi_prior(
            dataset.gene_names,
            args.embed_dim,
            args.embed_file,
            args.embedding_init_scale,
        )
        ppi_prior_df.to_csv(args.result_dir / "ppi_prior_embedding.csv")
        embedding_manifest["ppi_prior"] = ppi_prior_manifest
        embedding_manifest["ppi_gate_init"] = float(args.ppi_gate_init)
    embedding_manifest["ppi_integration"] = args.ppi_integration
    embedding_df.to_csv(args.result_dir / "gene_embedding.csv")
    save_json(args.result_dir / "embedding_manifest.json", embedding_manifest)
    trained_embedding_summary: list[dict[str, Any]] = []
    parameter_counts: dict[str, int] = {}
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_embed_path = Path(temp_dir) / "embedding.csv"
        embedding_df.to_csv(temp_embed_path)
        temp_ppi_prior_path: Path | None = None
        if ppi_prior_df is not None:
            temp_ppi_prior_path = Path(temp_dir) / "ppi_prior.csv"
            ppi_prior_df.to_csv(temp_ppi_prior_path)
        model = build_txt_model(
            args,
            dataset,
            task_specs,
            task_gene_indices,
            temp_embed_path,
            temp_ppi_prior_path,
            volumetric_graph,
        ).to(device)
        reseed_enabled = getattr(args, "post_model_construction_reseed", "off") == "on"
        # Model construction consumes a variant-dependent number of random values.
        # Paired protocols explicitly reset here so baseline/VMA optimization
        # starts from the same RNG state; generic legacy runs leave it untouched.
        if reseed_enabled:
            set_seed(args.seed)
        effective_vma_dropout = float(
            args.dropout if args.volumetric_dropout is None else args.volumetric_dropout
        )
        grad_clip_scope = getattr(args, "grad_clip_scope", "joint")
        if args.model_variant == "ppi_volumetric":
            exact_zero_gate_first_step_pairing = bool(
                reseed_enabled
                and effective_vma_dropout == 0.0
                and float(args.volumetric_gate_init) == 0.0
                and grad_clip_scope == "separate_volumetric"
            )
            pairing_caveats = []
            if not reseed_enabled:
                pairing_caveats.append("post_model_construction_reseed_disabled")
            if effective_vma_dropout != 0.0:
                pairing_caveats.append("volumetric_dropout_consumes_training_rng")
            if float(args.volumetric_gate_init) != 0.0:
                pairing_caveats.append("volumetric_gate_not_zero_initialized")
            if grad_clip_scope != "separate_volumetric":
                pairing_caveats.append(
                    "joint_gradient_clipping_couples_vma_and_shared_gradient_norms"
                )
        else:
            exact_zero_gate_first_step_pairing = reseed_enabled
            pairing_caveats = (
                [] if reseed_enabled else ["post_model_construction_reseed_disabled"]
            )
        rng_pairing_policy = {
            "seed": int(args.seed),
            "model_variant": args.model_variant,
            "post_model_construction_reseed": reseed_enabled,
            "reseed_point": (
                "immediately_after_model_construction_and_device_transfer"
                if reseed_enabled
                else None
            ),
            "effective_vma_dropout": effective_vma_dropout,
            "grad_clip_scope": grad_clip_scope,
            "gate_init": float(args.volumetric_gate_init),
            "exact_zero_gate_first_step_pairing": exact_zero_gate_first_step_pairing,
            "purpose": (
                "exact_zero_gate_first_step_pairing"
                if exact_zero_gate_first_step_pairing
                else (
                    "post_construction_rng_reseed"
                    if reseed_enabled
                    else "legacy_rng_sequence_after_model_construction"
                )
            ),
            "caveats": pairing_caveats,
        }
        embedding_parameter_ids = {id(parameter) for parameter in model.embedding_parameters()}
        encoder_parameter_ids = {
            id(parameter) for parameter in model.encoder_parameters_without_embeddings()
        }
        parameter_counts = {
            "total": int(sum(parameter.numel() for parameter in model.parameters())),
            "embeddings": int(
                sum(parameter.numel() for parameter in model.parameters() if id(parameter) in embedding_parameter_ids)
            ),
            "encoders_without_embeddings": int(
                sum(parameter.numel() for parameter in model.parameters() if id(parameter) in encoder_parameter_ids)
            ),
            "task_specific": int(
                sum(
                    parameter.numel()
                    for parameter in model.parameters()
                    if id(parameter) not in embedding_parameter_ids | encoder_parameter_ids
                )
            ),
        }
        if evaluation_only:
            ensemble_states: list[dict[str, Any]] = []
            ensemble_members: list[dict[str, Any]] = []
            source_training_summaries: list[dict[str, Any]] = []
            for rank, checkpoint_path in enumerate(evaluation_checkpoint_paths, start=1):
                checkpoint_state = load_checkpoint_state(checkpoint_path, device)
                source_model_summary = load_source_model_summary(checkpoint_path)
                validate_checkpoint_configuration(source_model_summary, args, volumetric_graph)
                validate_checkpoint_graph(checkpoint_state, volumetric_graph)
                # Strict loading validates keys and tensor shapes for every member.
                model.load_state_dict(checkpoint_state)
                if args.model_variant == "ppi_volumetric":
                    loaded_volume_mode = str(
                        getattr(model, "volumetric_volume_mode", "raw")
                    )
                    if loaded_volume_mode != args.volumetric_volume_mode:
                        raise ValueError(
                            "Evaluation checkpoint restored volumetric_volume_mode="
                            f"{loaded_volume_mode!r}, which does not match requested value "
                            f"{args.volumetric_volume_mode!r}. Legacy checkpoints without this "
                            "extra-state field are raw-mode checkpoints."
                        )
                ensemble_states.append(checkpoint_state)
                member = source_checkpoint_member_metadata(
                    checkpoint_path, source_model_summary
                )
                member["rank"] = rank
                ensemble_members.append(member)
                source_training_summary = source_model_summary.get("training_summary", {})
                source_training_summaries.append(
                    source_training_summary
                    if isinstance(source_training_summary, dict)
                    else {}
                )

            artifact_state = ensemble_states[0]
            model.load_state_dict(artifact_state)
            source_training_summary = source_training_summaries[0]
            training_summary = {
                "mode": "evaluation_only",
                "trained": False,
                "evaluation_only_checkpoint": (
                    str(evaluation_checkpoint_paths[0].resolve())
                    if len(evaluation_checkpoint_paths) == 1
                    else None
                ),
                "evaluation_only_checkpoints": [
                    str(path.resolve()) for path in evaluation_checkpoint_paths
                ],
                "source_best_epoch": ensemble_members[0]["epoch"],
                "best_epoch": ensemble_members[0]["epoch"],
                "checkpoint_metric": source_training_summary.get(
                    "checkpoint_metric", args.checkpoint_metric
                ),
                "best_checkpoint_value": ensemble_members[0]["score"],
                "source_training_summary_found": bool(source_training_summary),
                "source_training_summaries_found": int(
                    sum(bool(summary) for summary in source_training_summaries)
                ),
                "best_ppi_gate_values": model_ppi_gate_values(model),
                "best_volumetric_gate_values": model_volumetric_gate_values(model),
            }
            history: list[dict[str, Any]] = []
            criteria = build_evaluation_criteria(dataset, task_specs, args, device)
            copied_paths = save_ensemble_checkpoint_artifacts(
                args.result_dir,
                ensemble_states,
                evaluation_checkpoint_paths,
            )
            for member, copied_path in zip(ensemble_members, copied_paths):
                member["copied_path"] = str(copied_path)
            ensemble_requested_k = len(evaluation_checkpoint_paths)
        else:
            history, best_state, checkpoint_ensemble, training_summary, criteria = train_model(
                model, dataset, task_specs, args, device
            )
            if checkpoint_ensemble:
                ensemble_states = [entry["state"] for entry in checkpoint_ensemble]
                ensemble_entries = checkpoint_ensemble
            else:
                # Keep the artifact/evaluation contract usable even when every
                # checkpoint score is NaN (for example in a tiny smoke split).
                ensemble_states = [best_state]
                ensemble_entries = [
                    {
                        "epoch": training_summary.get("best_epoch", 0),
                        "value": training_summary.get("best_checkpoint_value", math.nan),
                    }
                ]
            artifact_state = ensemble_states[0]
            copied_paths = save_ensemble_checkpoint_artifacts(
                args.result_dir, ensemble_states
            )
            ensemble_members = [
                {
                    "rank": rank,
                    "path": str(copied_path),
                    "epoch": int(entry["epoch"]) if entry.get("epoch") is not None else None,
                    "score": (
                        float(entry["value"])
                        if entry.get("value") is not None
                        else None
                    ),
                    "score_metric": args.checkpoint_metric,
                }
                for rank, (entry, copied_path) in enumerate(
                    zip(ensemble_entries, copied_paths), start=1
                )
            ]
            ensemble_requested_k = int(args.checkpoint_ensemble_size)
            model.load_state_dict(artifact_state)
            training_summary["best_ppi_gate_values"] = model_ppi_gate_values(model)
            training_summary["best_volumetric_gate_values"] = model_volumetric_gate_values(model)
        ensemble_effective_k = len(ensemble_states)
        training_summary["checkpoint_ensemble_requested_k"] = ensemble_requested_k
        training_summary["checkpoint_ensemble_effective_k"] = ensemble_effective_k
        training_summary["checkpoint_ensemble_aggregation"] = (
            "arithmetic_mean_softmax_probabilities"
        )
        training_summary["checkpoint_ensemble_members"] = ensemble_members
        training_summary["rng_pairing_policy"] = rng_pairing_policy
        model.load_state_dict(artifact_state)
        trained_embedding_summary = save_trained_embedding_artifacts(model, dataset.gene_names, args.result_dir)

        ensemble_validation_diagnostics: dict[str, Any] = {}
        ensemble_val_results, ensemble_val_loss = evaluate_multitask(
            model,
            dataset,
            "val",
            dataset.val_gene_x,
            dataset.val_y,
            dataset.val_ids,
            task_specs,
            args.batch_size,
            device,
            criteria,
            args.task_loss_weights,
            state_dicts=ensemble_states,
            mask_aware_heads=args.mask_aware_heads == "on",
            volumetric_diagnostics_collector=(
                ensemble_validation_diagnostics
                if args.model_variant == "ppi_volumetric"
                else None
            ),
        )
        ensemble_score_metric = str(
            training_summary.get("checkpoint_metric") or args.checkpoint_metric
        )
        ensemble_validation_score, ensemble_validation_details = compute_checkpoint_value(
            ensemble_score_metric,
            ensemble_val_results,
            ensemble_val_loss,
            primary_task="AD_vs_MCI",
        )
        training_summary["ensemble_validation_score"] = ensemble_validation_score
        training_summary["ensemble_validation_score_metric"] = ensemble_score_metric
        training_summary["ensemble_validation_loss"] = ensemble_val_loss
        training_summary["ensemble_validation_score_details"] = ensemble_validation_details
        training_summary["ensemble_validation_scope"] = "full_validation_split"
        training_summary["ensemble"] = {
            "requested_k": ensemble_requested_k,
            "effective_k": ensemble_effective_k,
            "members": ensemble_members,
            "aggregation": "arithmetic_mean_softmax_probabilities",
            "validation_score": ensemble_validation_score,
            "validation_score_metric": ensemble_score_metric,
            "validation_loss": ensemble_val_loss,
            "validation_scope": "full_validation_split",
        }

        all_metric_rows: list[dict[str, Any]] = []
        if evaluation_only:
            eval_splits = [("test", dataset.test_gene_x, dataset.test_y, dataset.test_ids)]
        else:
            eval_splits = [
                ("train", dataset.train_gene_x, dataset.train_y, dataset.train_ids),
                ("val", dataset.val_gene_x, dataset.val_y, dataset.val_ids),
            ]
            if args.evaluate_test == "on":
                eval_splits.append(("test", dataset.test_gene_x, dataset.test_y, dataset.test_ids))
        for split_name, gene_x, y, ids in eval_splits:
            split_results, _ = evaluate_multitask(
                model,
                dataset,
                split_name,
                gene_x,
                y,
                ids,
                task_specs,
                args.batch_size,
                device,
                criteria,
                args.task_loss_weights,
                state_dicts=ensemble_states,
                mask_aware_heads=args.mask_aware_heads == "on",
            )
            all_metric_rows.extend(save_split_artifacts(args.result_dir, split_results))
        # Ensemble evaluation loads each member in turn; rank 1 remains the
        # canonical artifact after all split evaluations.
        model.load_state_dict(artifact_state)
        if args.model_variant == "ppi_volumetric":
            member_diagnostics = ensemble_validation_diagnostics.get(
                "member_diagnostics", []
            )
            best_member_diagnostics = (
                member_diagnostics[0] if member_diagnostics else {}
            )
            ensemble_diagnostics_scope = {
                **ensemble_validation_diagnostics.get("scope", {}),
                "checkpoint": "ensemble_checkpoint_rank1..rankN",
            }
            best_diagnostics_scope = {
                **ensemble_validation_diagnostics.get("scope", {}),
                "checkpoint": "best_model",
                "aggregation": "sample_weighted_mean_across_all_validation_batches",
                "ensemble_members": 1,
                "forward_batches": int(
                    best_member_diagnostics.get("forward_batches", 0)
                ),
                "forwarded_samples": int(
                    best_member_diagnostics.get("forwarded_samples", 0)
                ),
            }
            training_summary["best_volumetric_gate_values"] = model_volumetric_gate_values(model)
            training_summary["best_volumetric_diagnostics"] = (
                best_member_diagnostics.get("diagnostics", {})
            )
            training_summary["best_volumetric_diagnostics_scope"] = best_diagnostics_scope
            training_summary["ensemble_volumetric_diagnostics"] = (
                ensemble_validation_diagnostics.get("diagnostics", {})
            )
            training_summary["ensemble_volumetric_diagnostics_scope"] = (
                ensemble_diagnostics_scope
            )
            save_json(
                args.result_dir / "volumetric_diagnostics.json",
                {
                    "model_variant": args.model_variant,
                    "volumetric_beta": float(args.volumetric_beta),
                    "volumetric_eps": float(args.volumetric_eps),
                    "volumetric_gate_init": float(args.volumetric_gate_init),
                    "volumetric_volume_mode": args.volumetric_volume_mode,
                    "volumetric_dropout": args.volumetric_dropout,
                    "volumetric_dropout_effective": float(
                        args.dropout
                        if args.volumetric_dropout is None
                        else args.volumetric_dropout
                    ),
                    "volumetric_message_mode": args.volumetric_message_mode,
                    "volumetric_output_norm": args.volumetric_output_norm,
                    "volumetric_gate_mode": args.volumetric_gate_mode,
                    "volumetric_backbone_gradient_mode": args.volumetric_backbone_gradient_mode,
                    "gate_values": model_volumetric_gate_values(model),
                    "diagnostics": best_member_diagnostics.get("diagnostics", {}),
                    "diagnostics_scope": best_diagnostics_scope,
                    "ensemble_diagnostics": ensemble_validation_diagnostics.get(
                        "diagnostics", {}
                    ),
                    "ensemble_diagnostics_scope": ensemble_diagnostics_scope,
                    "member_diagnostics": member_diagnostics,
                    "training_summary": {
                        key: value
                        for key, value in training_summary.items()
                        if "volumetric" in key
                    },
                },
            )

    history_frame = pd.DataFrame(history) if history else pd.DataFrame(columns=["epoch"])
    history_frame.to_csv(args.result_dir / "training_log.csv", index=False)
    diagnostic_columns = [
        column
        for column in history_frame.columns
        if column == "epoch"
        or column.startswith("grad_")
        or column.startswith("ppi_gate_")
        or column.startswith("volumetric_")
    ]
    if len(diagnostic_columns) > 1:
        history_frame[diagnostic_columns].to_csv(
            args.result_dir / "gradient_diagnostics.csv", index=False
        )
    pd.DataFrame(all_metric_rows).to_csv(args.result_dir / "metrics_summary.csv", index=False)
    pd.DataFrame({"gene": dataset.gene_names}).to_csv(args.result_dir / "selected_genes.csv", index=False)
    gene_selection_details.to_csv(args.result_dir / "gene_selection_details.csv", index=False)
    save_json(args.result_dir / "augmentation_manifest.json", augmentation_manifest)
    save_json(args.result_dir / "candidate_gene_manifest.json", candidate_gene_manifest)
    if task_gene_indices is not None:
        task_gene_rows = []
        for task_name, indices in task_gene_indices.items():
            for position, gene_idx in enumerate(indices, start=1):
                task_gene_rows.append(
                    {
                        "task": task_name,
                        "task_gene_order": position,
                        "union_gene_index": int(gene_idx),
                        "gene": dataset.gene_names[int(gene_idx)],
                    }
                )
        pd.DataFrame(task_gene_rows).to_csv(args.result_dir / "task_specific_gene_pools.csv", index=False)
    save_json(
        args.result_dir / "args.json",
        {
            **_json_compatible(namespace_to_dict(args)),
            "resolved_split_file": str(resolved_split_file) if resolved_split_file is not None else None,
            "device_resolved": str(device),
            "rng_pairing_policy": rng_pairing_policy,
        },
    )
    save_json(
        args.result_dir / "model_summary.json",
        {
            "model": "txt_multitask",
            "model_variant": args.model_variant,
            "execution_mode": "evaluation_only" if evaluation_only else "train_and_evaluate",
            "reproducibility": rng_pairing_policy,
            "num_genes": len(dataset.gene_names),
            "class_names": dataset.class_names,
            "tasks": {spec.name: spec.class_names for spec in task_specs},
            "gene_selection": {
                "method": args.gene_selection,
                "max_genes": args.max_genes,
                "selected_genes": len(dataset.gene_names),
                "feature_selection_fit_scope": args.feature_selection_fit_scope,
                "feature_selection_leakage_ablation": bool(args.feature_selection_fit_scope == "all"),
                "scaler": args.scaler,
                "scaler_fit_scope": args.scaler_fit_scope,
                "scaler_leakage_ablation": bool(args.scaler_fit_scope == "all"),
                "task_specific_pooling": args.task_specific_pooling,
                "ad_mci_gene_fraction": args.ad_mci_gene_fraction,
                "candidate_gene_filter": candidate_gene_manifest,
                "task_gene_counts": {
                    task_name: len(indices)
                    for task_name, indices in (task_gene_indices or {}).items()
                },
            },
            "augmentation": augmentation_manifest,
            "embedding": {
                **embedding_manifest,
                "trained": trained_embedding_summary,
            },
            "label_mask_summary": label_mask_summary,
            "training_summary": training_summary,
            "ppi_graph": volumetric_graph.to_manifest() if volumetric_graph is not None else None,
            "architecture": {
                "n_layers": args.n_layers,
                "n_heads": args.n_heads,
                "d_model": args.d_model,
                "d_ff": args.d_ff,
                "d_hidden1": args.d_hidden1,
                "d_hidden2": args.d_hidden2,
                "dropout": args.dropout,
                "train_sampling": args.train_sampling,
                "aggfunc": args.aggfunc,
                "head_norm": args.head_norm,
                "mask_aware_heads": args.mask_aware_heads,
                "encoder_sharing": args.encoder_sharing,
                "pooling_mode": args.pooling_mode,
                "attention_pooling_hidden_dim": args.attention_pooling_hidden_dim,
                "attention_pooling_dropout": args.attention_pooling_dropout,
                "primary_adapter_dim": args.primary_adapter_dim,
                "expression_residual": args.expression_residual,
                "tupe_mode": getattr(args, "tupe_mode", "on"),
                "ppi_integration": args.ppi_integration,
                "ppi_gate_init": args.ppi_gate_init,
                "model_variant": args.model_variant,
                "ppi_score_threshold": args.ppi_score_threshold,
                "volumetric_beta": args.volumetric_beta,
                "volumetric_eps": args.volumetric_eps,
                "volumetric_gate_init": args.volumetric_gate_init,
                "volumetric_volume_mode": args.volumetric_volume_mode,
                "volumetric_dropout": args.volumetric_dropout,
                "volumetric_dropout_effective": (
                    args.dropout
                    if args.volumetric_dropout is None
                    else args.volumetric_dropout
                ),
                "volumetric_message_mode": args.volumetric_message_mode,
                "volumetric_output_norm": args.volumetric_output_norm,
                "volumetric_gate_mode": args.volumetric_gate_mode,
                "volumetric_backbone_gradient_mode": args.volumetric_backbone_gradient_mode,
                "gradient_strategy": args.gradient_strategy,
                "grad_clip_norm": args.grad_clip_norm,
                "grad_clip_scope": args.grad_clip_scope,
                "task_loss_weights": dict(zip(TASK_NAMES, args.task_loss_weights)),
                "checkpoint_ensemble_size": args.checkpoint_ensemble_size,
                "checkpoint_ensemble_min_gap": args.checkpoint_ensemble_min_gap,
                "parameter_counts": parameter_counts,
            },
        },
    )
    print(f"Multitask TxT complete. Results: {args.result_dir}", flush=True)


if __name__ == "__main__":
    main()
