from __future__ import annotations

import copy
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .dataset import ArrayDataset
from .reporting import binary_roc_auc_np, build_report_dataframe, classification_stats, one_vs_rest_roc_auc, softmax_np


TensorToLogits = Callable[[nn.Module, torch.Tensor], torch.Tensor]
EpochCallback = Callable[[dict[str, float]], None]


@torch.no_grad()
def predict_classifier(
    model: nn.Module,
    gene_x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    device: torch.device,
    logits_fn: TensorToLogits,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    loader = DataLoader(ArrayDataset(gene_x, y), batch_size=batch_size, shuffle=False)
    model.eval()
    all_targets: list[np.ndarray] = []
    all_logits: list[np.ndarray] = []

    for batch_gene_x, batch_y in loader:
        batch_gene_x = batch_gene_x.to(device)
        logits = logits_fn(model, batch_gene_x).detach().cpu().numpy()
        all_targets.append(batch_y.numpy())
        all_logits.append(logits)

    y_true = np.concatenate(all_targets)
    logits = np.concatenate(all_logits)
    y_prob = softmax_np(logits)
    y_pred = y_prob.argmax(axis=1)
    return y_true, y_pred, y_prob, logits


def train_classifier(
    model: nn.Module,
    train_gene_x: np.ndarray,
    train_y: np.ndarray,
    val_gene_x: np.ndarray,
    val_y: np.ndarray,
    batch_size: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    class_weights: np.ndarray | None,
    early_stopping_patience: int | None,
    device: torch.device,
    logits_fn: TensorToLogits,
    verbose: bool = True,
    epoch_callback: EpochCallback | None = None,
    max_train_batches: int | None = None,
    max_val_batches: int | None = None,
    checkpoint_metric: str = "macro_f1",
    checkpoint_auc_weight: float = 0.5,
) -> tuple[list[dict[str, float]], dict[str, torch.Tensor], dict[str, float | int | bool]]:
    drop_last = batch_size > 1 and len(train_y) >= batch_size
    train_loader = DataLoader(
        ArrayDataset(train_gene_x, train_y),
        batch_size=batch_size,
        shuffle=True,
        drop_last=drop_last,
    )
    val_loader = DataLoader(ArrayDataset(val_gene_x, val_y), batch_size=batch_size, shuffle=False)

    weight_tensor = None
    if class_weights is not None:
        weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)

    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    if checkpoint_metric not in {"macro_f1", "roc_auc_macro_f1", "strict_v2"}:
        raise ValueError(f"Unsupported checkpoint_metric: {checkpoint_metric}")

    best_val_checkpoint_score = float("-inf")
    best_val_macro_f1 = float("-inf")
    best_val_roc_auc = float("nan")
    best_state_dict = copy.deepcopy(model.state_dict())
    best_epoch = 0
    epochs_without_improvement = 0
    history_rows: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_samples = 0

        for batch_idx, (batch_gene_x, batch_y) in enumerate(train_loader, start=1):
            batch_gene_x = batch_gene_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            logits = logits_fn(model, batch_gene_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_gene_x.size(0)
            total_samples += batch_gene_x.size(0)
            if max_train_batches is not None and batch_idx >= max_train_batches:
                break

        model.eval()
        val_targets: list[np.ndarray] = []
        val_predictions: list[np.ndarray] = []
        val_probabilities: list[np.ndarray] = []
        val_loss = 0.0
        val_samples = 0
        with torch.no_grad():
            for batch_idx, (batch_gene_x, batch_y) in enumerate(val_loader, start=1):
                batch_gene_x = batch_gene_x.to(device)
                batch_y_device = batch_y.to(device)
                logits_tensor = logits_fn(model, batch_gene_x)
                loss = criterion(logits_tensor, batch_y_device)
                logits = logits_tensor.detach().cpu().numpy()
                val_loss += loss.item() * batch_gene_x.size(0)
                val_samples += batch_gene_x.size(0)
                val_targets.append(batch_y.numpy())
                y_prob = softmax_np(logits)
                val_predictions.append(y_prob.argmax(axis=1))
                val_probabilities.append(y_prob)
                if max_val_batches is not None and batch_idx >= max_val_batches:
                    break

        y_true = np.concatenate(val_targets)
        y_pred = np.concatenate(val_predictions)
        y_prob = np.concatenate(val_probabilities)
        val_accuracy, val_macro_f1, val_weighted_f1, _, _, _, _, _ = classification_stats(
            y_true, y_pred, len(np.unique(train_y))
        )
        if y_prob.shape[1] == 2:
            val_roc_auc = binary_roc_auc_np(y_true, y_prob[:, 1])
        else:
            aucs = one_vs_rest_roc_auc(y_true, y_prob, [str(idx) for idx in range(y_prob.shape[1])])
            val_roc_auc = aucs["roc_auc_ovr_macro"]

        val_loss_mean = val_loss / max(val_samples, 1)
        if checkpoint_metric == "strict_v2" and not np.isnan(val_roc_auc):
            val_checkpoint_score = 0.5 * val_roc_auc + 0.5 * val_macro_f1 - 0.25 * val_loss_mean
        elif checkpoint_metric == "macro_f1" or np.isnan(val_roc_auc):
            val_checkpoint_score = val_macro_f1
        else:
            val_checkpoint_score = checkpoint_auc_weight * val_roc_auc + (1.0 - checkpoint_auc_weight) * val_macro_f1

        row = {
            "epoch": float(epoch),
            "train_loss": total_loss / max(total_samples, 1),
            "val_loss": val_loss_mean,
            "val_accuracy": val_accuracy,
            "val_roc_auc": val_roc_auc,
            "val_macro_f1": val_macro_f1,
            "val_weighted_f1": val_weighted_f1,
            "val_checkpoint_score": val_checkpoint_score,
        }
        history_rows.append(row)
        if verbose:
            print(
                f"Epoch {epoch:03d} | "
                f"train_loss={row['train_loss']:.4f} | "
                f"val_loss={row['val_loss']:.4f} | "
                f"val_acc={val_accuracy:.4f} | "
                f"val_roc_auc={val_roc_auc:.4f} | "
                f"val_macro_f1={val_macro_f1:.4f} | "
                f"val_weighted_f1={val_weighted_f1:.4f} | "
                f"checkpoint={val_checkpoint_score:.4f}",
                flush=True,
            )
        if epoch_callback is not None:
            epoch_callback(row)

        if val_checkpoint_score >= best_val_checkpoint_score:
            best_val_checkpoint_score = val_checkpoint_score
            best_val_macro_f1 = val_macro_f1
            best_val_roc_auc = val_roc_auc
            best_state_dict = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if early_stopping_patience is not None and early_stopping_patience > 0 and epochs_without_improvement >= early_stopping_patience:
            break

    training_summary: dict[str, float | int | bool] = {
        "best_epoch": int(best_epoch),
        "checkpoint_metric": checkpoint_metric,
        "checkpoint_auc_weight": float(checkpoint_auc_weight),
        "best_val_checkpoint_score": float(best_val_checkpoint_score),
        "best_val_macro_f1": float(best_val_macro_f1),
        "best_val_roc_auc": float(best_val_roc_auc),
        "epochs_ran": int(len(history_rows)),
        "stopped_early": bool(len(history_rows) < epochs),
    }
    return history_rows, best_state_dict, training_summary


def evaluate_classifier(
    model: nn.Module,
    gene_x: np.ndarray,
    y: np.ndarray,
    sample_ids: np.ndarray,
    batch_size: int,
    device: torch.device,
    class_names: list[str],
    logits_fn: TensorToLogits,
) -> dict[str, object]:
    y_true, y_pred, y_prob, logits = predict_classifier(model, gene_x, y, batch_size, device, logits_fn)
    accuracy, macro_f1, weighted_f1, confusion, _, recall, _, support = classification_stats(
        y_true, y_pred, len(class_names)
    )
    report_df = build_report_dataframe(y_true, y_pred, class_names)
    loss = float(nn.CrossEntropyLoss()(torch.tensor(logits, dtype=torch.float32), torch.tensor(y_true, dtype=torch.long)).item())
    balanced_accuracy = float(np.mean(recall[support > 0])) if np.any(support > 0) else float("nan")
    return {
        "sample_ids": sample_ids,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "loss": loss,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "balanced_accuracy": balanced_accuracy,
        **one_vs_rest_roc_auc(y_true, y_prob, class_names),
        "confusion_matrix": confusion,
        "report_df": report_df,
    }
