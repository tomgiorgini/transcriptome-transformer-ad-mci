#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import math
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from source.models.txt import TxT
from source.pipeline.dataset import ArrayDataset, PreparedDataset, prepare_dataset
from source.pipeline.reporting import build_report_dataframe, one_vs_rest_roc_auc, print_report, save_test_artifacts, softmax_np
from source.pipeline.utils import compute_balanced_class_weights, namespace_to_dict, resolve_device, save_json, set_seed
from source.pretraining.txt.transfer import remap_embedding_dataframe

try:
    from experiments.scripts.data_augmentation.augmentations import apply_offline_augmentation
except ModuleNotFoundError:
    def apply_offline_augmentation(
        dataset: PreparedDataset,
        method: str,
        seed: int,
        factor: float,
        pca_components: int,
        noise_scale: float,
    ) -> tuple[PreparedDataset, dict[str, Any]]:
        if method != "none":
            raise ModuleNotFoundError(
                "experiments.scripts.data_augmentation is unavailable; use --augmentation none "
                "or restore experiments/scripts/data_augmentation/augmentations.py."
            )
        return dataset, {
            "method": "none",
            "seed": seed,
            "factor": factor,
            "pca_components": pca_components,
            "noise_scale": noise_scale,
            "fallback_note": "data_augmentation module not found; no augmentation was applied.",
        }


DEFAULT_X_FILE = ROOT / "task_dataset" / "processed" / "ad_mci_deg" / "X_deg_ad_mci.csv"
DEFAULT_Y_FILE = ROOT / "task_dataset" / "processed" / "ad_mci_deg" / "y_ad_mci.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TxT DEG fine-tuning for pretrained or random-init models.")
    parser.add_argument("--pretrained-checkpoint", type=Path, required=True)
    parser.add_argument("--transfer-mode", choices=["full", "embedding_only", "random_init"], default="full")
    parser.add_argument("--x-file", type=Path, default=DEFAULT_X_FILE)
    parser.add_argument("--y-file", type=Path, default=DEFAULT_Y_FILE)
    parser.add_argument("--result-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-mode", choices=["stratified", "random"], default="stratified")
    parser.add_argument("--split-seed", type=int, default=101)
    parser.add_argument(
        "--split-file",
        type=Path,
        default=None,
        help="Optional CSV with sample_id,split columns. Supports train/val/test; test may be omitted when --evaluate-test off.",
    )
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--dataset-mode", choices=["deg", "deg_pretrained_overlap"], default="deg_pretrained_overlap")
    parser.add_argument(
        "--max-genes",
        type=int,
        default=0,
        help="Keep the top-N most variable genes from the training split before scaling. Use 0 to keep all genes in X.",
    )
    parser.add_argument(
        "--gene-selection",
        choices=["variance", "mad", "class_aware_variance"],
        default="variance",
        help=(
            "Feature ranking used when --max-genes is positive. "
            "variance and mad are unsupervised train-fold filters; "
            "class_aware_variance ranks genes by train-fold between-class variance."
        ),
    )
    parser.add_argument("--scaler", choices=["none", "minmax", "standard"], default="minmax")
    parser.add_argument(
        "--scaler-fit-scope",
        choices=["train", "all"],
        default="train",
        help="Use 'train' for proper train-only scaling or 'all' for exploratory leakage scaling across train/val/test.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--early-stopping-patience", type=int, default=100)
    parser.add_argument("--lr-encoder", type=float, default=1e-4)
    parser.add_argument("--lr-head", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--augmentation", choices=["none", "mixup", "minority_interp_balance", "pca_noise"], default="none")
    parser.add_argument("--augmentation-alpha", type=float, default=0.2)
    parser.add_argument("--augmentation-factor", type=float, default=0.5)
    parser.add_argument("--augmentation-pca-components", type=int, default=50)
    parser.add_argument("--augmentation-noise-scale", type=float, default=0.05)
    parser.add_argument("--augmentation-seed", type=int, default=None)
    parser.add_argument("--class-weighting", choices=["on", "off"], default="on")
    parser.add_argument(
        "--checkpoint-metric",
        choices=[
            "val_loss",
            "val_macro_f1",
            "val_macro_f1_tuned_threshold",
            "val_roc_auc_ovr_macro",
            "val_auc_f1_50_50",
            "val_auc_f1_50_50_minus_025_loss",
            "val_auc_loss_guarded",
            "val_auc_f1_50_50_loss_guarded",
        ],
        default="val_macro_f1",
    )
    parser.add_argument(
        "--auc-loss-guard-tolerance",
        type=float,
        default=0.10,
        help=(
            "For --checkpoint-metric val_auc_loss_guarded, an epoch can become the best AUC checkpoint "
            "only when val_loss is within this absolute tolerance from the best validation loss seen so far. "
            "Also applies to val_auc_f1_50_50_loss_guarded."
        ),
    )
    parser.add_argument(
        "--auc-loss-guard-max-loss",
        type=float,
        default=None,
        help=(
            "For guarded checkpoint metrics, use an absolute maximum validation loss. "
            "When set, an epoch is checkpoint-eligible only if val_loss is <= this value."
        ),
    )
    parser.add_argument("--threshold-tuning", choices=["on", "off"], default="off")
    parser.add_argument(
        "--threshold-selection",
        choices=["max_macro_f1", "stable_macro_f1"],
        default="stable_macro_f1",
        help=(
            "How to choose the validation threshold. `max_macro_f1` takes the best validation F1. "
            "`stable_macro_f1` chooses the threshold closest to --threshold-target among thresholds within "
            "--threshold-stability-tolerance of the best validation F1."
        ),
    )
    parser.add_argument("--threshold-min", type=float, default=0.05)
    parser.add_argument("--threshold-max", type=float, default=0.95)
    parser.add_argument("--threshold-steps", type=int, default=181)
    parser.add_argument("--threshold-target", type=float, default=0.5)
    parser.add_argument("--threshold-stability-tolerance", type=float, default=0.01)
    parser.add_argument(
        "--evaluate-test",
        choices=["on", "off"],
        default="on",
        help="Set to off to leave the test split as a pure holdout and save only train/validation metrics.",
    )
    parser.add_argument(
        "--final-threshold-mode",
        choices=["tuned", "fixed"],
        default="tuned",
        help="Use the validation-tuned threshold or a fixed threshold for final train/val/test metrics.",
    )
    parser.add_argument(
        "--final-threshold",
        type=float,
        default=0.5,
        help="Threshold used for final metrics when --final-threshold-mode fixed.",
    )
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--device", choices=["cuda", "cpu", "mps"], default="cuda")
    parser.add_argument("--n-heads", type=int, default=None)
    parser.add_argument("--n-layers", type=int, default=None)
    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--d-ff", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--norm-first", action="store_true")
    parser.add_argument("--aggfunc", choices=["Flatten", "Avgpool"], default="Avgpool")
    parser.add_argument("--d-hidden1", type=int, default=256)
    parser.add_argument("--d-hidden2", type=int, default=128)
    parser.add_argument("--slope", type=float, default=0.2)
    parser.add_argument(
        "--freeze-encoder-layers",
        type=str,
        default="",
        help="Comma-separated 1-based TxT encoder layer indices to freeze during fine-tuning, e.g. '1' or '1,2'.",
    )
    parser.add_argument("--freeze-embeddings", action="store_true")
    parser.add_argument("--freeze-tupe", action="store_true")
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    return parser.parse_args()


def default_result_dir() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return ROOT / "results" / "pretraining" / "finetuning" / f"run_{timestamp}"


def logits_fn(model: TxT, batch_gene_x: torch.Tensor) -> torch.Tensor:
    return model(batch_gene_x)[0]


def parse_freeze_layers(value: str, n_layers: int) -> list[int]:
    if not value.strip():
        return []
    layers: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        layer_idx = int(part)
        if layer_idx < 1 or layer_idx > n_layers:
            raise ValueError(f"freeze layer index {layer_idx} is outside valid range 1..{n_layers}.")
        layers.append(layer_idx - 1)
    return sorted(set(layers))


def apply_freezing(model: TxT, args: argparse.Namespace) -> dict[str, Any]:
    encoder = model.transformer.encoder
    frozen_layers = parse_freeze_layers(args.freeze_encoder_layers, len(encoder.layers))

    for layer_idx in frozen_layers:
        for parameter in encoder.layers[layer_idx].parameters():
            parameter.requires_grad = False

    if args.freeze_embeddings:
        for parameter in encoder.embed.parameters():
            parameter.requires_grad = False

    if args.freeze_tupe:
        for parameter in encoder.tupe.parameters():
            parameter.requires_grad = False

    trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    frozen_parameters = sum(parameter.numel() for parameter in model.parameters() if not parameter.requires_grad)
    return {
        "freeze_encoder_layers": [idx + 1 for idx in frozen_layers],
        "freeze_embeddings": bool(args.freeze_embeddings),
        "freeze_tupe": bool(args.freeze_tupe),
        "trainable_parameters": int(trainable_parameters),
        "frozen_parameters": int(frozen_parameters),
    }


def trainable_parameters(module: nn.Module):
    return [parameter for parameter in module.parameters() if parameter.requires_grad]


def soft_cross_entropy(
    logits: torch.Tensor,
    target_probabilities: torch.Tensor,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    log_probabilities = F.log_softmax(logits, dim=1)
    losses = -(target_probabilities * log_probabilities)
    if weight is not None:
        losses = losses * weight.unsqueeze(0)
    return losses.sum(dim=1).mean()


def smoothed_one_hot(labels: torch.Tensor, num_classes: int, smoothing: float) -> torch.Tensor:
    targets = F.one_hot(labels, num_classes=num_classes).to(dtype=torch.float32)
    if smoothing <= 0:
        return targets
    return targets * (1.0 - smoothing) + smoothing / float(num_classes)


def resolve_model_param(args_value: Any, config: dict[str, Any], key: str, fallback: Any) -> Any:
    if args_value is not None:
        return args_value
    value = config.get(key)
    return fallback if value is None else value


def subset_dataset(dataset: PreparedDataset, keep_indices: np.ndarray) -> PreparedDataset:
    keep_indices = np.asarray(keep_indices, dtype=np.int64)
    if keep_indices.size == 0:
        raise ValueError("Cannot subset dataset to zero genes.")
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


def filter_dataset_to_pretrained_genes(dataset: PreparedDataset, pretrained_gene_names: list[str]) -> PreparedDataset:
    pretrained_set = set(pretrained_gene_names)
    keep = np.array([idx for idx, gene in enumerate(dataset.gene_names) if gene in pretrained_set], dtype=np.int64)
    return subset_dataset(dataset, keep)


def load_transformer_weights(model: TxT, checkpoint_state: dict[str, torch.Tensor]) -> dict[str, Any]:
    model_state = model.state_dict()
    loaded: list[str] = []
    skipped: list[str] = []
    for key, value in checkpoint_state.items():
        if not key.startswith("transformer."):
            continue
        if key == "transformer.encoder.embed.embed.weight":
            skipped.append(key)
            continue
        if key in model_state and tuple(model_state[key].shape) == tuple(value.shape):
            model_state[key] = value
            loaded.append(key)
        else:
            skipped.append(key)
    model.load_state_dict(model_state)
    return {"loaded_transformer_keys": loaded, "skipped_pretrained_keys": skipped}


def safe_metric_name(value: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(value).strip())
    return "_".join(part for part in cleaned.split("_") if part)


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray, class_names: list[str]) -> dict[str, Any]:
    report_df = build_report_dataframe(y_true, y_pred, class_names)
    rows = {row["class"]: row for row in report_df.to_dict(orient="records")}
    try:
        roc_auc = float(roc_auc_score(y_true, y_prob[:, 1]))
    except ValueError:
        roc_auc = math.nan
    try:
        pr_auc = float(average_precision_score(y_true, y_prob[:, 1]))
    except ValueError:
        pr_auc = math.nan
    pr_auc_ovr: dict[str, float] = {}
    for class_idx, class_name in enumerate(class_names):
        metric_name = safe_metric_name(class_name)
        try:
            pr_auc_ovr[f"pr_auc_{metric_name}"] = float(
                average_precision_score((y_true == class_idx).astype(np.int64), y_prob[:, class_idx])
            )
        except ValueError:
            pr_auc_ovr[f"pr_auc_{metric_name}"] = math.nan
    valid_pr_auc = [value for value in pr_auc_ovr.values() if math.isfinite(float(value))]
    pr_auc_ovr["pr_auc_ovr_macro"] = float(np.mean(valid_pr_auc)) if valid_pr_auc else math.nan
    result = {
        "accuracy": float(rows["accuracy"]["f1_score"]),
        "macro_f1": float(rows["macro avg"]["f1_score"]),
        "weighted_f1": float(rows["weighted avg"]["f1_score"]),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "roc_auc": roc_auc,
        **one_vs_rest_roc_auc(y_true, y_prob, class_names),
        "pr_auc": pr_auc,
        **pr_auc_ovr,
        "report_df": report_df,
    }
    recalls = []
    for class_idx, class_name in enumerate(class_names):
        metric_name = safe_metric_name(class_name)
        recall_value = float(rows[class_name]["recall"])
        result[f"recall_{metric_name}"] = recall_value
        result[f"predicted_{metric_name}"] = int((y_pred == class_idx).sum())
        recalls.append(recall_value)
    result["min_recall"] = float(min(recalls)) if recalls else math.nan
    return result


@torch.no_grad()
def predict(model: TxT, x: np.ndarray, y: np.ndarray, batch_size: int, device: torch.device, max_batches: int | None = None):
    loader = DataLoader(ArrayDataset(x, y), batch_size=batch_size, shuffle=False)
    model.eval()
    ys: list[np.ndarray] = []
    logits_rows: list[np.ndarray] = []
    for batch_idx, (batch_x, batch_y) in enumerate(loader, start=1):
        logits = logits_fn(model, batch_x.to(device)).detach().cpu().numpy()
        ys.append(batch_y.numpy())
        logits_rows.append(logits)
        if max_batches is not None and batch_idx >= max_batches:
            break
    y_true = np.concatenate(ys)
    y_prob = softmax_np(np.concatenate(logits_rows))
    y_pred = np.argmax(y_prob, axis=1).astype(np.int64)
    return y_true, y_pred, y_prob


@torch.no_grad()
def evaluate_loss(
    model: TxT,
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    device: torch.device,
    criterion: nn.Module,
    max_batches: int | None = None,
) -> float:
    loader = DataLoader(ArrayDataset(x, y), batch_size=batch_size, shuffle=False)
    model.eval()
    total_loss = 0.0
    total_samples = 0
    for batch_idx, (batch_x, batch_y) in enumerate(loader, start=1):
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        logits = logits_fn(model, batch_x)
        loss = criterion(logits, batch_y)
        total_loss += float(loss.item()) * int(batch_x.size(0))
        total_samples += int(batch_x.size(0))
        if max_batches is not None and batch_idx >= max_batches:
            break
    return total_loss / max(total_samples, 1)


def checkpoint_improved(current: float, best: float, metric: str) -> bool:
    if not math.isfinite(float(current)):
        return False
    if metric == "val_loss":
        return current <= best
    return current >= best


def threshold_predictions(y_prob: np.ndarray, threshold: float) -> np.ndarray:
    if y_prob.shape[1] != 2:
        raise ValueError("Threshold tuning is only supported for binary classification.")
    return (y_prob[:, 1] >= threshold).astype(np.int64)


def tune_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: np.ndarray,
    selection: str,
    stability_tolerance: float,
    target_threshold: float,
    class_names: list[str],
) -> dict[str, Any]:
    evaluated: list[dict[str, Any]] = []
    for threshold in thresholds:
        y_pred = threshold_predictions(y_prob, float(threshold))
        metrics = classification_metrics(y_true, y_pred, y_prob, class_names)
        metrics.pop("report_df", None)
        evaluated.append({"threshold": float(threshold), **metrics})

    if not evaluated:
        raise ValueError("No thresholds were evaluated.")

    best_score = max(float(item["macro_f1"]) for item in evaluated)
    if selection == "stable_macro_f1":
        tolerance = max(float(stability_tolerance), 0.0)
        candidates = [item for item in evaluated if float(item["macro_f1"]) >= best_score - tolerance]
        selected = min(
            candidates,
            key=lambda item: (
                abs(float(item["threshold"]) - float(target_threshold)),
                -float(item["macro_f1"]),
            ),
        )
    else:
        selected = min(
            evaluated,
            key=lambda item: (
                -float(item["macro_f1"]),
                abs(float(item["threshold"]) - float(target_threshold)),
            ),
        )

    selected["threshold_best_macro_f1"] = best_score
    selected["threshold_selection"] = selection
    selected["threshold_stability_tolerance"] = float(stability_tolerance)
    selected["threshold_target"] = float(target_threshold)
    return selected


def train_finetuning(
    model: TxT,
    dataset: PreparedDataset,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, torch.Tensor], dict[str, Any]]:
    train_loader = DataLoader(
        ArrayDataset(dataset.train_gene_x, dataset.train_y),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=args.batch_size > 1 and len(dataset.train_y) >= args.batch_size,
    )
    class_weights = compute_balanced_class_weights(dataset.train_y) if args.class_weighting == "on" else None
    weight_tensor = None
    if class_weights is not None:
        weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor, label_smoothing=args.label_smoothing)
    transformer_parameters = trainable_parameters(model.transformer)
    head_parameters = trainable_parameters(model.task_specific_layers)
    if not transformer_parameters and not head_parameters:
        raise ValueError("No trainable parameters remain after applying freeze options.")
    parameter_groups = []
    if transformer_parameters:
        parameter_groups.append({"params": transformer_parameters, "lr": args.lr_encoder})
    if head_parameters:
        parameter_groups.append({"params": head_parameters, "lr": args.lr_head})
    optimizer = torch.optim.AdamW(
        parameter_groups,
        weight_decay=args.weight_decay,
    )
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_metric_value = float("inf") if args.checkpoint_metric == "val_loss" else float("-inf")
    best_threshold = 0.5
    best_val_macro_f1 = float("-inf")
    best_val_loss_seen = float("inf")
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []
    thresholds = np.linspace(args.threshold_min, args.threshold_max, args.threshold_steps)
    mixup_rng = np.random.default_rng(args.augmentation_seed)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_samples = 0
        for batch_idx, (batch_x, batch_y) in enumerate(train_loader, start=1):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            logits = logits_fn(model, batch_x)
            if args.augmentation == "mixup":
                alpha = float(args.augmentation_alpha)
                lam = float(mixup_rng.beta(alpha, alpha)) if alpha > 0 else 1.0
                permutation = torch.randperm(batch_x.size(0), device=device)
                mixed_x = lam * batch_x + (1.0 - lam) * batch_x[permutation]
                logits = logits_fn(model, mixed_x)
                targets_a = smoothed_one_hot(batch_y, logits.size(1), args.label_smoothing)
                targets_b = smoothed_one_hot(batch_y[permutation], logits.size(1), args.label_smoothing)
                mixed_targets = lam * targets_a + (1.0 - lam) * targets_b
                loss = soft_cross_entropy(logits, mixed_targets, weight_tensor)
            else:
                loss = criterion(logits, batch_y)
            loss.backward()
            if args.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            optimizer.step()
            total_loss += float(loss.item()) * int(batch_x.size(0))
            total_samples += int(batch_x.size(0))
            if args.max_train_batches is not None and batch_idx >= args.max_train_batches:
                break

        val_y_true, val_y_pred, val_y_prob = predict(
            model,
            dataset.val_gene_x,
            dataset.val_y,
            args.batch_size,
            device,
            max_batches=args.max_val_batches,
        )
        val_loss = evaluate_loss(
            model,
            dataset.val_gene_x,
            dataset.val_y,
            args.batch_size,
            device,
            criterion,
            max_batches=args.max_val_batches,
        )
        val_metrics = classification_metrics(val_y_true, val_y_pred, val_y_prob, dataset.class_names)
        val_metrics.pop("report_df", None)
        tuned_threshold = 0.5
        tuned_macro_f1 = math.nan
        threshold_best_macro_f1 = math.nan
        if args.threshold_tuning == "on":
            tuned = tune_threshold(
                val_y_true,
                val_y_prob,
                thresholds,
                args.threshold_selection,
                args.threshold_stability_tolerance,
                args.threshold_target,
                dataset.class_names,
            )
            tuned_threshold = float(tuned["threshold"])
            tuned_macro_f1 = float(tuned["macro_f1"])
            threshold_best_macro_f1 = float(tuned["threshold_best_macro_f1"])
        val_auc = float(val_metrics.get("roc_auc_ovr_macro", val_metrics.get("roc_auc", math.nan)))
        val_auc_f1_50_50 = 0.5 * val_auc + 0.5 * float(val_metrics["macro_f1"])
        val_auc_f1_50_50_minus_025_loss = val_auc_f1_50_50 - 0.25 * float(val_loss)
        best_val_loss_seen = min(best_val_loss_seen, float(val_loss))
        if args.auc_loss_guard_max_loss is not None:
            loss_guard_passed = float(val_loss) <= float(args.auc_loss_guard_max_loss)
        else:
            loss_guard_passed = float(val_loss) <= best_val_loss_seen + float(args.auc_loss_guard_tolerance)
        if args.checkpoint_metric == "val_loss":
            checkpoint_value = float(val_loss)
        elif args.checkpoint_metric == "val_macro_f1_tuned_threshold":
            checkpoint_value = float(tuned_macro_f1)
        elif args.checkpoint_metric == "val_auc_f1_50_50":
            checkpoint_value = float(val_auc_f1_50_50)
        elif args.checkpoint_metric == "val_auc_f1_50_50_minus_025_loss":
            checkpoint_value = float(val_auc_f1_50_50_minus_025_loss)
        elif args.checkpoint_metric == "val_auc_loss_guarded":
            checkpoint_value = float(val_auc) if loss_guard_passed else float("nan")
        elif args.checkpoint_metric == "val_auc_f1_50_50_loss_guarded":
            checkpoint_value = float(val_auc_f1_50_50) if loss_guard_passed else float("nan")
        else:
            checkpoint_value = float(val_metrics[args.checkpoint_metric.removeprefix("val_")])
        row = {
            "epoch": epoch,
            "train_loss": total_loss / max(total_samples, 1),
            "val_loss": val_loss,
            "checkpoint_metric": args.checkpoint_metric,
            "checkpoint_value": checkpoint_value,
            "val_threshold": tuned_threshold,
            "val_macro_f1_tuned_threshold": tuned_macro_f1,
            "val_threshold_best_macro_f1": threshold_best_macro_f1,
            "val_auc_f1_50_50": val_auc_f1_50_50,
            "val_auc_f1_50_50_minus_025_loss": val_auc_f1_50_50_minus_025_loss,
            "val_loss_guard_best": best_val_loss_seen,
            "val_loss_guard_tolerance": float(args.auc_loss_guard_tolerance),
            "val_loss_guard_max_loss": args.auc_loss_guard_max_loss,
            "val_loss_guard_passed": bool(loss_guard_passed),
            **{f"val_{k}": v for k, v in val_metrics.items() if isinstance(v, (int, float, np.integer, np.floating))},
        }
        history.append(row)
        print(
            f"Epoch {epoch:03d} | train_loss={row['train_loss']:.4f} | "
            f"val_loss={row['val_loss']:.4f} | "
            f"val_macro_f1={val_metrics['macro_f1']:.4f} | "
            f"val_auc_ovr={val_metrics.get('roc_auc_ovr_macro', math.nan):.4f} | "
            f"checkpoint_value={checkpoint_value:.4f}",
            flush=True,
        )
        if checkpoint_improved(checkpoint_value, best_metric_value, args.checkpoint_metric):
            best_metric_value = checkpoint_value
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            best_threshold = tuned_threshold
            best_val_macro_f1 = float(val_metrics["macro_f1"])
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if args.early_stopping_patience > 0 and epochs_without_improvement >= args.early_stopping_patience:
            break

    guarded_metrics = {"val_auc_loss_guarded", "val_auc_f1_50_50_loss_guarded"}
    if args.checkpoint_metric in guarded_metrics and best_epoch == 0:
        raise RuntimeError(
            "No checkpoint satisfied the guarded validation-loss constraint. "
            f"checkpoint_metric={args.checkpoint_metric}, "
            f"auc_loss_guard_max_loss={args.auc_loss_guard_max_loss}, "
            f"auc_loss_guard_tolerance={args.auc_loss_guard_tolerance}"
        )

    return history, best_state, {
        "best_epoch": best_epoch,
        "checkpoint_metric": args.checkpoint_metric,
        "best_checkpoint_value": best_metric_value,
        "best_threshold": best_threshold,
        "best_val_macro_f1": best_val_macro_f1,
        "threshold_tuning": args.threshold_tuning,
        "threshold_selection": args.threshold_selection,
        "threshold_target": args.threshold_target,
        "threshold_stability_tolerance": args.threshold_stability_tolerance,
        "auc_loss_guard_tolerance": args.auc_loss_guard_tolerance,
        "auc_loss_guard_max_loss": args.auc_loss_guard_max_loss,
        "best_val_loss_seen": best_val_loss_seen,
        "epochs_ran": len(history),
        "stopped_early": len(history) < args.epochs,
    }


def evaluate_split(model: TxT, x: np.ndarray, y: np.ndarray, ids: np.ndarray, args, device, class_names, threshold: float):
    y_true, y_pred, y_prob = predict(model, x, y, args.batch_size, device)
    if args.threshold_tuning == "on":
        y_pred = threshold_predictions(y_prob, threshold)
    metrics = classification_metrics(y_true, y_pred, y_prob, class_names)
    loss = evaluate_loss(model, x, y, args.batch_size, device, nn.CrossEntropyLoss(), max_batches=None)
    return {
        "sample_ids": ids,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "loss": loss,
        "confusion_matrix": np.asarray([[((y_true == i) & (y_pred == j)).sum() for j in range(len(class_names))] for i in range(len(class_names))]),
        **{k: v for k, v in metrics.items() if k != "report_df"},
        "report_df": build_report_dataframe(y_true, y_pred, class_names),
    }


def save_metrics_summary(result_dir: Path, split_results: dict[str, dict[str, Any]]) -> None:
    rows = []
    for split, result in split_results.items():
        rows.append(
            {
                "split": split,
                "samples": len(result["sample_ids"]),
                "loss": result["loss"],
                "accuracy": result["accuracy"],
                "macro_f1": result["macro_f1"],
                "weighted_f1": result["weighted_f1"],
                "balanced_accuracy": result["balanced_accuracy"],
                "roc_auc": result["roc_auc"],
                **{key: value for key, value in result.items() if key.startswith("roc_auc_")},
                "pr_auc": result["pr_auc"],
                **{key: value for key, value in result.items() if key.startswith("pr_auc_")},
                "min_recall": result["min_recall"],
                **{key: value for key, value in result.items() if key.startswith("recall_")},
                **{key: value for key, value in result.items() if key.startswith("predicted_")},
            }
        )
    pd.DataFrame(rows).to_csv(result_dir / "metrics_summary.csv", index=False)


def print_metric_summary(name: str, split_result: dict[str, Any]) -> None:
    auc = split_result.get("roc_auc_ovr_macro", split_result.get("roc_auc", math.nan))
    pr_auc = split_result.get("pr_auc_ovr_macro", split_result.get("pr_auc", math.nan))
    print(
        f"[{name}] loss={split_result['loss']:.4f} "
        f"macro_f1={split_result['macro_f1']:.4f} "
        f"balanced_accuracy={split_result['balanced_accuracy']:.4f} "
        f"auc_ovr_macro={float(auc):.4f} "
        f"pr_auc_ovr_macro={float(pr_auc):.4f}",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    if args.augmentation_seed is None:
        args.augmentation_seed = args.split_seed if args.split_seed is not None else args.seed
    set_seed(args.seed)
    device = resolve_device(args.device)
    result_dir = args.result_dir if args.result_dir is not None else default_result_dir()
    result_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(args.pretrained_checkpoint, map_location="cpu", weights_only=False)
    checkpoint_config = checkpoint.get("config", {})
    pretrained_gene_names = [str(gene) for gene in checkpoint["gene_names"]]
    pretrained_state = checkpoint["model_state_dict"]
    pretrained_embedding = pretrained_state["transformer.encoder.embed.embed.weight"].detach().cpu().numpy()
    source_embedding_df = pd.DataFrame(pretrained_embedding, index=pretrained_gene_names)

    dataset = prepare_dataset(
        x_file=args.x_file,
        y_file=args.y_file,
        seed=args.seed,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        max_genes=args.max_genes,
        scaler=args.scaler,
        scaler_fit_scope=args.scaler_fit_scope,
        split_file=args.split_file,
        split_seed=args.split_seed,
        split_mode=args.split_mode,
        gene_selection=args.gene_selection,
        candidate_genes=pretrained_gene_names if args.dataset_mode == "deg_pretrained_overlap" else None,
    )
    original_gene_count = len(dataset.gene_names)
    if args.dataset_mode == "deg_pretrained_overlap":
        dataset = filter_dataset_to_pretrained_genes(dataset, pretrained_gene_names)
    training_dataset, augmentation_summary = apply_offline_augmentation(
        dataset,
        method=args.augmentation,
        seed=args.augmentation_seed,
        factor=args.augmentation_factor,
        pca_components=args.augmentation_pca_components,
        noise_scale=args.augmentation_noise_scale,
    )
    augmentation_summary["mixup"] = {
        "enabled": bool(args.augmentation == "mixup"),
        "alpha": float(args.augmentation_alpha),
    }
    save_json(result_dir / "augmentation_summary.json", augmentation_summary)

    if args.transfer_mode == "random_init":
        rng = np.random.default_rng(args.seed)
        d_model_for_init = int(resolve_model_param(args.d_model, checkpoint_config, "d_model", pretrained_embedding.shape[1]))
        random_embedding = rng.normal(loc=0.0, scale=0.02, size=(len(dataset.gene_names), d_model_for_init)).astype(np.float32)
        remapped_embedding_df = pd.DataFrame(random_embedding, index=dataset.gene_names)
        remap_report = {
            "matched_genes": 0,
            "missing_genes": len(dataset.gene_names),
            "random_init_note": "No pretrained weights were loaded; checkpoint was used only for architecture and optional gene overlap.",
        }
    else:
        remapped_embedding_df, remap_report = remap_embedding_dataframe(source_embedding_df, dataset.gene_names, args.seed)
    embedding_path = result_dir / "gene_embedding_from_pretraining.csv"
    remapped_embedding_df.to_csv(embedding_path)

    n_heads = int(resolve_model_param(args.n_heads, checkpoint_config, "n_heads", 4))
    n_layers = int(resolve_model_param(args.n_layers, checkpoint_config, "n_layers", 3))
    d_model = int(resolve_model_param(args.d_model, checkpoint_config, "d_model", 128))
    d_ff = int(resolve_model_param(args.d_ff, checkpoint_config, "d_ff", 512))
    dropout = float(resolve_model_param(args.dropout, checkpoint_config, "dropout", 0.2))
    norm_first = bool(args.norm_first or checkpoint_config.get("norm_first", False))
    if args.transfer_mode in {"full", "embedding_only"}:
        checkpoint_d_model = checkpoint_config.get("d_model")
        checkpoint_d_ff = checkpoint_config.get("d_ff")
        if checkpoint_d_model is not None and args.d_model is not None and int(args.d_model) != int(checkpoint_d_model):
            print(
                f"Warning: ignoring --d-model {args.d_model} for {args.transfer_mode}; "
                f"using checkpoint d_model={checkpoint_d_model}.",
                flush=True,
            )
            d_model = int(checkpoint_d_model)
        if checkpoint_d_ff is not None and args.d_ff is not None and int(args.d_ff) != int(checkpoint_d_ff):
            print(
                f"Warning: ignoring --d-ff {args.d_ff} for {args.transfer_mode}; "
                f"using checkpoint d_ff={checkpoint_d_ff}.",
                flush=True,
            )
            d_ff = int(checkpoint_d_ff)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_embed_path = Path(temp_dir) / "embedding.csv"
        remapped_embedding_df.to_csv(temp_embed_path)
        model = TxT(
            embed_file=str(temp_embed_path),
            gene_list=dataset.gene_names,
            n_heads=n_heads,
            d_model=d_model,
            dropout=dropout,
            d_ff=d_ff,
            norm_first=norm_first,
            n_layers=n_layers,
            aggfunc=args.aggfunc,
            d_hidden1=args.d_hidden1,
            d_hidden2=args.d_hidden2,
            slope=args.slope,
            d_output_dict={"label": len(dataset.class_names)},
        ).to(device)
        if args.transfer_mode == "full":
            transfer_report = load_transformer_weights(model, pretrained_state)
        elif args.transfer_mode == "random_init":
            transfer_report = {"loaded_transformer_keys": [], "transfer_mode_note": "Random initialization; no pretrained embeddings or transformer weights were used."}
        else:
            transfer_report = {"loaded_transformer_keys": [], "transfer_mode_note": "Only pretrained gene embeddings were used."}
        freeze_report = apply_freezing(model, args)
        transfer_report.update(remap_report)
        transfer_report.update(
            {
                "transfer_mode": args.transfer_mode,
                "dataset_mode": args.dataset_mode,
                "task_genes_before_filter": original_gene_count,
                "task_genes_after_filter": len(dataset.gene_names),
                "freeze_report": freeze_report,
            }
        )
        save_json(result_dir / "transfer_report.json", transfer_report)

        history, best_state, training_summary = train_finetuning(model, training_dataset, args, device)
        pd.DataFrame(history).to_csv(result_dir / "training_log.csv", index=False)
        model.load_state_dict(best_state)
        torch.save(best_state, result_dir / "best_model.pt")
        final_threshold = (
            float(args.final_threshold)
            if args.final_threshold_mode == "fixed"
            else float(training_summary["best_threshold"])
        )
        final_eval_args = copy.copy(args)
        final_eval_args.threshold_tuning = "on" if len(dataset.class_names) == 2 else "off"
        training_summary["final_threshold_mode"] = args.final_threshold_mode
        training_summary["final_threshold"] = final_threshold

        split_results = {
            "train": evaluate_split(
                model,
                dataset.train_gene_x,
                dataset.train_y,
                dataset.train_ids,
                final_eval_args,
                device,
                dataset.class_names,
                final_threshold,
            ),
            "val": evaluate_split(
                model,
                dataset.val_gene_x,
                dataset.val_y,
                dataset.val_ids,
                final_eval_args,
                device,
                dataset.class_names,
                final_threshold,
            ),
        }
        if args.evaluate_test == "on":
            split_results["test"] = evaluate_split(
                model,
                dataset.test_gene_x,
                dataset.test_y,
                dataset.test_ids,
                final_eval_args,
                device,
                dataset.class_names,
                final_threshold,
            )

    for split_name, split_result in split_results.items():
        print_report(split_name, split_result["report_df"])
        print_metric_summary(split_name, split_result)

    save_metrics_summary(result_dir, split_results)
    if "test" in split_results:
        save_test_artifacts(result_dir, split_results["test"], dataset.class_names)
    pd.DataFrame({"gene": dataset.gene_names}).to_csv(result_dir / "selected_genes.csv", index=False)
    pd.DataFrame(history).to_csv(result_dir / "training_log.csv", index=False)
    save_json(
        result_dir / "args.json",
        {
            **namespace_to_dict(args),
            "result_dir": str(result_dir),
            "device_resolved": str(device),
            "pretrained_config": checkpoint_config,
            "resolved_model_config": {
                "n_heads": n_heads,
                "n_layers": n_layers,
                "d_model": d_model,
                "d_ff": d_ff,
                "dropout": dropout,
                "norm_first": norm_first,
            },
            "freeze_report": freeze_report,
        },
    )
    save_json(
        result_dir / "model_summary.json",
        {
            "model": "txt_deg_finetuning",
            "num_genes": len(dataset.gene_names),
            "num_classes": len(dataset.class_names),
            "freeze_report": freeze_report,
            "training_summary": training_summary,
        },
    )
    print(f"Fine-tuning complete. Results: {result_dir}", flush=True)


if __name__ == "__main__":
    main()
