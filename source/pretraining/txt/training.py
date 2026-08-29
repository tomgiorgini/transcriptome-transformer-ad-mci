from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .data import MaskedGeneBatch


RestorationEpochCallback = Callable[[int, nn.Module, dict[str, float]], None]


@dataclass(frozen=True)
class RestorationEpochResult:
    loss: float
    masked_values: int
    batches: int


def move_batch_to_device(batch: MaskedGeneBatch, device: torch.device) -> MaskedGeneBatch:
    return MaskedGeneBatch(
        input_values=batch.input_values.to(device),
        target_values=batch.target_values.to(device),
        gene_indices=batch.gene_indices.to(device),
        masked_positions=batch.masked_positions.to(device),
        attention_mask=batch.attention_mask.to(device),
    )


def masked_mse_loss(prediction: torch.Tensor, target: torch.Tensor, masked_positions: torch.Tensor) -> torch.Tensor:
    if masked_positions.sum().item() == 0:
        raise ValueError("No masked positions available for restoration loss.")
    return nn.functional.mse_loss(prediction[masked_positions], target[masked_positions])


def train_restoration_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_batches: int | None = None,
) -> RestorationEpochResult:
    model.train()
    total_loss = 0.0
    total_masked = 0
    batches = 0

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad()
        prediction = model(
            batch.input_values,
            gene_indices=batch.gene_indices,
            mask=batch.attention_mask,
            masked_positions=batch.masked_positions,
        )
        loss = masked_mse_loss(prediction, batch.target_values, batch.masked_positions)
        loss.backward()
        optimizer.step()

        masked_count = int(batch.masked_positions.sum().item())
        total_loss += float(loss.item()) * masked_count
        total_masked += masked_count
        batches += 1
        if max_batches is not None and batches >= max_batches:
            break

    return RestorationEpochResult(loss=total_loss / max(total_masked, 1), masked_values=total_masked, batches=batches)


@torch.no_grad()
def evaluate_restoration(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int | None = None,
) -> RestorationEpochResult:
    model.eval()
    total_loss = 0.0
    total_masked = 0
    batches = 0

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        prediction = model(
            batch.input_values,
            gene_indices=batch.gene_indices,
            mask=batch.attention_mask,
            masked_positions=batch.masked_positions,
        )
        loss = masked_mse_loss(prediction, batch.target_values, batch.masked_positions)

        masked_count = int(batch.masked_positions.sum().item())
        total_loss += float(loss.item()) * masked_count
        total_masked += masked_count
        batches += 1
        if max_batches is not None and batches >= max_batches:
            break

    return RestorationEpochResult(loss=total_loss / max(total_masked, 1), masked_values=total_masked, batches=batches)


def run_restoration_training(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
    max_train_batches: int | None = None,
    max_val_batches: int | None = None,
    early_stopping_patience: int = 0,
    epoch_callback: RestorationEpochCallback | None = None,
) -> tuple[list[dict[str, float]], dict[str, torch.Tensor], dict[str, float | int]]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_state_dict = copy.deepcopy(model.state_dict())
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        train_result = train_restoration_epoch(model, train_loader, optimizer, device, max_train_batches)
        val_result = evaluate_restoration(model, val_loader, device, max_val_batches)
        row = {
            "epoch": float(epoch),
            "train_loss": float(train_result.loss),
            "val_loss": float(val_result.loss),
            "train_masked_values": float(train_result.masked_values),
            "val_masked_values": float(val_result.masked_values),
            "train_batches": float(train_result.batches),
            "val_batches": float(val_result.batches),
        }
        history.append(row)
        print(
            f"Epoch {epoch:03d} | train_loss={train_result.loss:.6f} | "
            f"val_loss={val_result.loss:.6f}",
            flush=True,
        )

        if val_result.loss < best_val_loss:
            best_val_loss = val_result.loss
            best_state_dict = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epoch_callback is not None:
            epoch_callback(epoch, model, row)

        if early_stopping_patience > 0 and epochs_without_improvement >= early_stopping_patience:
            print(
                f"Early stopping at epoch {epoch:03d} | "
                f"best_epoch={best_epoch:03d} | best_val_loss={best_val_loss:.6f}",
                flush=True,
            )
            break

    summary: dict[str, float | int] = {
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val_loss),
        "epochs_ran": int(len(history)),
        "early_stopping_patience": int(early_stopping_patience),
    }
    return history, best_state_dict, summary
