#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from source.models.txt import TxT
from source.models.txt.layers import calculate_attention
from source.pipeline.dataset import PreparedDataset, prepare_dataset
from source.pipeline.reporting import (
    print_report,
    save_args,
    save_metrics_summary,
    save_selected_genes,
    save_split_assignments,
    save_test_artifacts,
    save_training_history,
)
from source.pipeline.training import evaluate_classifier, train_classifier
from source.pipeline.utils import compute_balanced_class_weights, namespace_to_dict, resolve_device, save_json, set_seed
from source.pretraining.txt.transfer import remap_embedding_dataframe


DEFAULT_X_FILE = ROOT / "task_dataset" / "processed" / "ad_mci_deg" / "X_deg_ad_mci.csv"
DEFAULT_Y_FILE = ROOT / "task_dataset" / "processed" / "ad_mci_deg" / "y_ad_mci.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune TxT from a restoration pretraining checkpoint.")
    parser.add_argument("--pretrained-checkpoint", type=Path, required=True)
    parser.add_argument("--transfer-mode", choices=["full", "embedding_only"], default="embedding_only")
    parser.add_argument("--x-file", type=Path, default=DEFAULT_X_FILE)
    parser.add_argument("--y-file", type=Path, default=DEFAULT_Y_FILE)
    parser.add_argument("--result-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-mode", choices=["stratified", "random"], default="stratified")
    parser.add_argument("--split-seed", type=int, default=101)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--max-genes", type=int, default=0)
    parser.add_argument("--filter-to-pretrained-genes", action="store_true")
    parser.add_argument(
        "--random-gene-count",
        type=int,
        default=0,
        help="If > 0, keep one fixed selected subset of K genes for the whole fine-tuning run.",
    )
    parser.add_argument(
        "--gene-selection-pool",
        choices=["pretrained_overlap", "dataset"],
        default="pretrained_overlap",
        help="Gene pool used when --random-gene-count > 0.",
    )
    parser.add_argument("--random-gene-pool", choices=["pretrained_overlap", "dataset"], default=None)
    parser.add_argument("--gene-selection-mode", choices=["random", "variance"], default="random")
    parser.add_argument("--skip-deg-validation", action="store_true")
    parser.add_argument("--scaler", choices=["none", "minmax", "standard"], default="minmax")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--early-stopping-patience", type=int, default=45)
    parser.add_argument("--lr", type=float, default=0.000187854264)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--class-weighting", choices=["on", "off"], default="off")
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
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--plot-finetuning-matrices", action="store_true")
    parser.add_argument(
        "--plot-attention-max-samples",
        type=int,
        default=32,
        help="Maximum number of train samples used to average attention plots.",
    )
    parser.add_argument(
        "--plot-input-max-samples",
        type=int,
        default=160,
        help="Maximum rows shown in the finetuning input matrix heatmap.",
    )
    return parser.parse_args()


def default_result_dir() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return ROOT / "results" / "pretraining" / "finetuning" / f"run_{timestamp}"


def logits_fn(model: TxT, batch_gene_x: torch.Tensor) -> torch.Tensor:
    return model(batch_gene_x)[0]


def _sample_evenly(n_items: int, max_items: int) -> np.ndarray:
    if max_items <= 0 or n_items <= max_items:
        return np.arange(n_items, dtype=np.int64)
    return np.linspace(0, n_items - 1, num=max_items, dtype=np.int64)


def save_finetuning_input_matrix_plot(
    result_dir: Path,
    dataset: PreparedDataset,
    max_samples: int,
    scale: str = "raw",
    cmap: str = "viridis",
    clip: float | None = None,
) -> None:
    plot_dir = result_dir / "plots" / "matrices"
    plot_dir.mkdir(parents=True, exist_ok=True)

    split_arrays = [
        ("train", dataset.train_gene_x, dataset.train_y, dataset.train_ids),
        ("val", dataset.val_gene_x, dataset.val_y, dataset.val_ids),
        ("test", dataset.test_gene_x, dataset.test_y, dataset.test_ids),
    ]
    rows: list[np.ndarray] = []
    row_metadata: list[dict[str, str | int]] = []
    for split_name, x_values, y_values, sample_ids in split_arrays:
        selected = _sample_evenly(x_values.shape[0], max_samples // len(split_arrays) if max_samples > 0 else 0)
        for idx in selected:
            rows.append(x_values[int(idx)])
            row_metadata.append(
                {
                    "sample_id": str(sample_ids[int(idx)]),
                    "split": split_name,
                    "label": int(y_values[int(idx)]),
                    "label_name": dataset.class_names[int(y_values[int(idx)])],
                }
            )

    order = np.lexsort(
        (
            np.array([str(row["sample_id"]) for row in row_metadata]),
            np.array([str(row["split"]) for row in row_metadata]),
            np.array([int(row["label"]) for row in row_metadata]),
        )
    )
    matrix = np.vstack(rows)[order]
    ordered_metadata = [row_metadata[int(idx)] for idx in order]
    labels = np.array([int(row["label"]) for row in ordered_metadata], dtype=np.int64)
    label_names = [dataset.class_names[int(label)] for label in labels]
    plot_matrix, colorbar_label = transform_input_matrix_for_plot(matrix, scale)
    vmin = None
    vmax = None
    if clip is not None and clip > 0:
        plot_matrix = np.clip(plot_matrix, -clip, clip)
        vmin = -clip
        vmax = clip

    pd.DataFrame(matrix, columns=dataset.gene_names).to_csv(plot_dir / "finetuning_input_matrix_plotted_values.csv", index=False)
    pd.DataFrame(ordered_metadata).to_csv(plot_dir / "finetuning_input_matrix_plotted_rows.csv", index=False)

    width = min(max(9.0, matrix.shape[1] / 55.0), 18.0)
    height = min(max(5.0, matrix.shape[0] / 20.0), 12.0)
    fig, (label_ax, ax) = plt.subplots(
        1,
        2,
        figsize=(width, height),
        width_ratios=[0.035, 1.0],
        constrained_layout=True,
    )
    label_ax.imshow(labels[:, None], aspect="auto", interpolation="nearest", cmap="tab10")
    label_ax.set_title("Class")
    label_ax.set_xticks([])
    label_ax.set_yticks([])

    image = ax.imshow(
        plot_matrix,
        aspect="auto",
        interpolation="nearest",
        cmap=resolve_colormap(cmap),
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_title(f"Fine-tuning input matrix grouped by class label ({scale})")
    ax.set_xlabel(f"Genes (n={len(dataset.gene_names)})")
    ax.set_ylabel(f"Samples (n={matrix.shape[0]})")
    ax.set_xticks([])

    class_boundaries = np.where(labels[1:] != labels[:-1])[0] + 1
    for boundary in class_boundaries:
        ax.axhline(boundary - 0.5, color="white", linewidth=1.0)
        label_ax.axhline(boundary - 0.5, color="white", linewidth=1.0)
    ytick_positions = []
    ytick_labels = []
    start = 0
    for boundary in [*class_boundaries.tolist(), len(labels)]:
        midpoint = (start + boundary - 1) / 2
        ytick_positions.append(midpoint)
        ytick_labels.append(label_names[start])
        start = boundary
    ax.set_yticks(ytick_positions)
    ax.set_yticklabels(ytick_labels)
    fig.colorbar(image, ax=ax, label=colorbar_label)
    fig.savefig(plot_dir / "finetuning_input_matrix.png", dpi=220)
    fig.savefig(plot_dir / "finetuning_input_matrix.pdf")
    plt.close(fig)


def transform_input_matrix_for_plot(matrix: np.ndarray, scale: str) -> tuple[np.ndarray, str]:
    if scale == "raw":
        return matrix, "Expression value after finetuning preprocessing"
    if scale == "gene_zscore":
        mean = matrix.mean(axis=0, keepdims=True)
        std = matrix.std(axis=0, keepdims=True)
        std = np.where(std == 0, 1.0, std)
        return (matrix - mean) / std, "Gene-wise z-score across plotted samples"
    if scale == "gene_robust_zscore":
        median = np.median(matrix, axis=0, keepdims=True)
        q25 = np.percentile(matrix, 25, axis=0, keepdims=True)
        q75 = np.percentile(matrix, 75, axis=0, keepdims=True)
        iqr = np.where((q75 - q25) == 0, 1.0, q75 - q25)
        return (matrix - median) / iqr, "Gene-wise robust z-score across plotted samples"
    raise ValueError(f"Unsupported input matrix scale: {scale}")


def resolve_colormap(cmap: str):
    if cmap == "blue_black_yellow":
        return LinearSegmentedColormap.from_list("blue_black_yellow", ["blue", "black", "yellow"])
    return cmap


def _attention_weights(
    attention_layer: torch.nn.Module,
    hidden: torch.Tensor,
    tupe: torch.Tensor,
    mask: torch.Tensor | None,
) -> torch.Tensor:
    batch_size = hidden.size(0)
    q = attention_layer.q_linear(hidden).view(batch_size, -1, attention_layer.n_heads, attention_layer.d_head).transpose(1, 2)
    k = attention_layer.k_linear(hidden).view(batch_size, -1, attention_layer.n_heads, attention_layer.d_head).transpose(1, 2)
    v = attention_layer.v_linear(hidden).view(batch_size, -1, attention_layer.n_heads, attention_layer.d_head).transpose(1, 2)
    _, attention = calculate_attention(q, k, v, tupe, mask, dropout=None)
    return attention


@torch.no_grad()
def collect_attention_matrices(
    model: TxT,
    gene_x: np.ndarray,
    device: torch.device,
    batch_size: int,
    max_samples: int,
) -> np.ndarray:
    selected = _sample_evenly(gene_x.shape[0], max_samples)
    values = torch.tensor(gene_x[selected], dtype=torch.float32)
    n_layers = len(model.transformer.encoder.layers)
    n_heads = model.transformer.encoder.layers[0].multi_head_attention_layer.n_heads
    n_genes = gene_x.shape[1]
    attention_sum = torch.zeros((n_layers, n_heads, n_genes, n_genes), dtype=torch.float64)
    sample_count = 0

    model.eval()
    for start in range(0, values.shape[0], batch_size):
        batch = values[start : start + batch_size].to(device)
        current_batch_size, current_n_genes = batch.size()
        gene_indices = torch.arange(current_n_genes, device=device).repeat(current_batch_size).view(current_batch_size, -1)
        hidden = model.transformer.encoder.embed(gene_indices)
        tupe = model.transformer.encoder.tupe(batch)

        for layer_idx, layer in enumerate(model.transformer.encoder.layers):
            attention_input = layer.layer_norm_1(hidden) if layer.norm_first else hidden
            attention = _attention_weights(layer.multi_head_attention_layer, attention_input, tupe, mask=None)
            attention_sum[layer_idx] += attention.detach().cpu().double().sum(dim=0)
            hidden = layer(hidden, tupe, mask=None)
        sample_count += current_batch_size

    if sample_count == 0:
        raise ValueError("No samples available for attention plotting.")
    return (attention_sum / sample_count).numpy()


@torch.no_grad()
def collect_attention_matrices_by_class(
    model: TxT,
    gene_x: np.ndarray,
    y: np.ndarray,
    class_names: list[str],
    device: torch.device,
    batch_size: int,
    max_samples_per_class: int,
) -> dict[str, np.ndarray]:
    class_attention: dict[str, np.ndarray] = {}
    for class_idx, class_name in enumerate(class_names):
        class_rows = np.where(y == class_idx)[0]
        if class_rows.size == 0:
            continue
        selected = class_rows[_sample_evenly(class_rows.size, max_samples_per_class)]
        class_attention[class_name] = collect_attention_matrices(
            model=model,
            gene_x=gene_x[selected],
            device=device,
            batch_size=batch_size,
            max_samples=max_samples_per_class,
        )
    if len(class_attention) < 2:
        raise ValueError("Need at least two classes with samples to compute class-specific attention.")
    return class_attention


def save_attention_matrix_plots(
    result_dir: Path,
    attention_by_layer_head: np.ndarray,
    gene_names: list[str],
    save_csv: bool = True,
    scale: str = "linear",
    vmax_percentile: float = 99.0,
    filename_suffix: str = "",
    linear_vmax: float | None = None,
) -> None:
    plot_dir = result_dir / "plots" / "attention"
    plot_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        plot_dir / "attention_matrices.npz",
        layer_head_attention=attention_by_layer_head,
        layer_mean_attention=attention_by_layer_head.mean(axis=1),
        gene_names=np.asarray(gene_names, dtype=object),
    )

    n_layers = attention_by_layer_head.shape[0]
    n_heads = attention_by_layer_head.shape[1]
    shared_vmax = float(linear_vmax) if linear_vmax is not None and linear_vmax > 0 else float(np.max(attention_by_layer_head))
    for layer_idx in range(n_layers):
        layer_attention = attention_by_layer_head[layer_idx].mean(axis=0)
        if save_csv:
            pd.DataFrame(layer_attention, index=gene_names, columns=gene_names).to_csv(
                plot_dir / f"attention_layer_{layer_idx + 1:02d}_mean_heads.csv"
            )
        fig, ax = plt.subplots(figsize=(8.5, 7.5), constrained_layout=True)
        plot_values, colorbar_label, vmin, vmax = transform_attention_for_plot(
            layer_attention,
            scale,
            vmax_percentile,
            shared_vmax,
            linear_vmax,
        )
        image = ax.imshow(plot_values, aspect="auto", interpolation="nearest", cmap="magma", vmin=vmin, vmax=vmax)
        ax.set_title(f"TxT attention layer {layer_idx + 1} - mean over heads and samples ({scale})")
        ax.set_xlabel(f"Key genes (n={len(gene_names)})")
        ax.set_ylabel(f"Query genes (n={len(gene_names)})")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(image, ax=ax, label=colorbar_label)
        fig.savefig(plot_dir / f"attention_layer_{layer_idx + 1:02d}_mean_heads{filename_suffix}.png", dpi=220)
        fig.savefig(plot_dir / f"attention_layer_{layer_idx + 1:02d}_mean_heads{filename_suffix}.pdf")
        plt.close(fig)

        for head_idx in range(n_heads):
            head_attention = attention_by_layer_head[layer_idx, head_idx]
            if save_csv:
                pd.DataFrame(head_attention, index=gene_names, columns=gene_names).to_csv(
                    plot_dir / f"attention_layer_{layer_idx + 1:02d}_head_{head_idx + 1:02d}.csv"
                )
            fig, ax = plt.subplots(figsize=(8.5, 7.5), constrained_layout=True)
            plot_values, colorbar_label, vmin, vmax = transform_attention_for_plot(
                head_attention,
                scale,
                vmax_percentile,
                shared_vmax,
                linear_vmax,
            )
            image = ax.imshow(plot_values, aspect="auto", interpolation="nearest", cmap="magma", vmin=vmin, vmax=vmax)
            ax.set_title(f"TxT attention layer {layer_idx + 1} head {head_idx + 1} ({scale})")
            ax.set_xlabel(f"Key genes (n={len(gene_names)})")
            ax.set_ylabel(f"Query genes (n={len(gene_names)})")
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(image, ax=ax, label=colorbar_label)
            fig.savefig(plot_dir / f"attention_layer_{layer_idx + 1:02d}_head_{head_idx + 1:02d}{filename_suffix}.png", dpi=220)
            fig.savefig(plot_dir / f"attention_layer_{layer_idx + 1:02d}_head_{head_idx + 1:02d}{filename_suffix}.pdf")
            plt.close(fig)

    all_layers_mean = attention_by_layer_head.mean(axis=(0, 1))
    if save_csv:
        pd.DataFrame(all_layers_mean, index=gene_names, columns=gene_names).to_csv(plot_dir / "attention_all_layers_mean.csv")
    fig, ax = plt.subplots(figsize=(8.5, 7.5), constrained_layout=True)
    plot_values, colorbar_label, vmin, vmax = transform_attention_for_plot(
        all_layers_mean,
        scale,
        vmax_percentile,
        shared_vmax,
        linear_vmax,
    )
    image = ax.imshow(plot_values, aspect="auto", interpolation="nearest", cmap="magma", vmin=vmin, vmax=vmax)
    ax.set_title(f"TxT attention - mean over layers, heads, and train samples ({scale})")
    ax.set_xlabel(f"Key genes (n={len(gene_names)})")
    ax.set_ylabel(f"Query genes (n={len(gene_names)})")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(image, ax=ax, label=colorbar_label)
    fig.savefig(plot_dir / f"attention_all_layers_mean{filename_suffix}.png", dpi=220)
    fig.savefig(plot_dir / f"attention_all_layers_mean{filename_suffix}.pdf")
    plt.close(fig)


def _class_delta_pair(class_attention: dict[str, np.ndarray]) -> tuple[str, str, np.ndarray]:
    if "AD" in class_attention and "MCI" in class_attention:
        positive_class = "AD"
        negative_class = "MCI"
    else:
        names = list(class_attention)
        negative_class = names[0]
        positive_class = names[1]
    return positive_class, negative_class, class_attention[positive_class] - class_attention[negative_class]


def _save_attention_image(
    path_base: Path,
    matrix: np.ndarray,
    title: str,
    gene_names: list[str],
    cmap: str,
    colorbar_label: str,
    vmin: float | None,
    vmax: float | None,
) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 7.5), constrained_layout=True)
    image = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel(f"Key genes (n={len(gene_names)})")
    ax.set_ylabel(f"Query genes (n={len(gene_names)})")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(image, ax=ax, label=colorbar_label)
    fig.savefig(path_base.with_suffix(".png"), dpi=220)
    fig.savefig(path_base.with_suffix(".pdf"))
    plt.close(fig)


def save_class_attention_matrix_plots(
    result_dir: Path,
    class_attention: dict[str, np.ndarray],
    gene_names: list[str],
    save_csv: bool = True,
    scale: str = "linear",
    vmax_percentile: float = 99.0,
    filename_suffix: str = "",
    linear_vmax: float | None = None,
) -> None:
    plot_dir = result_dir / "plots" / "attention_by_class"
    plot_dir.mkdir(parents=True, exist_ok=True)
    positive_class, negative_class, delta_attention = _class_delta_pair(class_attention)

    np.savez_compressed(
        plot_dir / "attention_matrices_by_class.npz",
        **{f"layer_head_attention_{name}": values for name, values in class_attention.items()},
        delta_positive_class=positive_class,
        delta_negative_class=negative_class,
        layer_head_attention_delta=delta_attention,
        gene_names=np.asarray(gene_names, dtype=object),
    )

    class_stack = np.stack(list(class_attention.values()), axis=0)
    shared_vmax = float(linear_vmax) if linear_vmax is not None and linear_vmax > 0 else float(np.max(class_stack))
    n_layers = class_stack.shape[1]
    n_heads = class_stack.shape[2]

    for layer_idx in range(n_layers):
        for class_name, attention_by_layer_head in class_attention.items():
            layer_attention = attention_by_layer_head[layer_idx].mean(axis=0)
            if save_csv:
                pd.DataFrame(layer_attention, index=gene_names, columns=gene_names).to_csv(
                    plot_dir / f"attention_{class_name}_layer_{layer_idx + 1:02d}_mean_heads.csv"
                )
            plot_values, colorbar_label, vmin, vmax = transform_attention_for_plot(
                layer_attention,
                scale,
                vmax_percentile,
                shared_vmax,
                linear_vmax,
            )
            _save_attention_image(
                plot_dir / f"attention_{class_name}_layer_{layer_idx + 1:02d}_mean_heads{filename_suffix}",
                plot_values,
                f"TxT attention {class_name} layer {layer_idx + 1} - mean heads ({scale})",
                gene_names,
                "magma",
                colorbar_label,
                vmin,
                vmax,
            )

        layer_delta = delta_attention[layer_idx].mean(axis=0)
        delta_vmax = float(np.percentile(np.abs(layer_delta), vmax_percentile))
        if delta_vmax <= 0:
            delta_vmax = float(np.max(np.abs(layer_delta)))
        if delta_vmax <= 0:
            delta_vmax = 1.0
        if save_csv:
            pd.DataFrame(layer_delta, index=gene_names, columns=gene_names).to_csv(
                plot_dir / f"attention_delta_{positive_class}_minus_{negative_class}_layer_{layer_idx + 1:02d}_mean_heads.csv"
            )
        _save_attention_image(
            plot_dir / f"attention_delta_{positive_class}_minus_{negative_class}_layer_{layer_idx + 1:02d}_mean_heads{filename_suffix}",
            np.clip(layer_delta, -delta_vmax, delta_vmax),
            f"TxT attention delta {positive_class} - {negative_class} layer {layer_idx + 1} - mean heads",
            gene_names,
            "coolwarm",
            f"Attention difference ({positive_class} - {negative_class})",
            -delta_vmax,
            delta_vmax,
        )

        for head_idx in range(n_heads):
            for class_name, attention_by_layer_head in class_attention.items():
                head_attention = attention_by_layer_head[layer_idx, head_idx]
                if save_csv:
                    pd.DataFrame(head_attention, index=gene_names, columns=gene_names).to_csv(
                        plot_dir / f"attention_{class_name}_layer_{layer_idx + 1:02d}_head_{head_idx + 1:02d}.csv"
                    )
                plot_values, colorbar_label, vmin, vmax = transform_attention_for_plot(
                    head_attention,
                    scale,
                    vmax_percentile,
                    shared_vmax,
                    linear_vmax,
                )
                _save_attention_image(
                    plot_dir / f"attention_{class_name}_layer_{layer_idx + 1:02d}_head_{head_idx + 1:02d}{filename_suffix}",
                    plot_values,
                    f"TxT attention {class_name} layer {layer_idx + 1} head {head_idx + 1} ({scale})",
                    gene_names,
                    "magma",
                    colorbar_label,
                    vmin,
                    vmax,
                )

            head_delta = delta_attention[layer_idx, head_idx]
            delta_vmax = float(np.percentile(np.abs(head_delta), vmax_percentile))
            if delta_vmax <= 0:
                delta_vmax = float(np.max(np.abs(head_delta)))
            if delta_vmax <= 0:
                delta_vmax = 1.0
            if save_csv:
                pd.DataFrame(head_delta, index=gene_names, columns=gene_names).to_csv(
                    plot_dir / f"attention_delta_{positive_class}_minus_{negative_class}_layer_{layer_idx + 1:02d}_head_{head_idx + 1:02d}.csv"
                )
            _save_attention_image(
                plot_dir / f"attention_delta_{positive_class}_minus_{negative_class}_layer_{layer_idx + 1:02d}_head_{head_idx + 1:02d}{filename_suffix}",
                np.clip(head_delta, -delta_vmax, delta_vmax),
                f"TxT attention delta {positive_class} - {negative_class} layer {layer_idx + 1} head {head_idx + 1}",
                gene_names,
                "coolwarm",
                f"Attention difference ({positive_class} - {negative_class})",
                -delta_vmax,
                delta_vmax,
            )

    for class_name, attention_by_layer_head in class_attention.items():
        all_layers_attention = attention_by_layer_head.mean(axis=(0, 1))
        if save_csv:
            pd.DataFrame(all_layers_attention, index=gene_names, columns=gene_names).to_csv(
                plot_dir / f"attention_{class_name}_all_layers_mean.csv"
            )
        plot_values, colorbar_label, vmin, vmax = transform_attention_for_plot(
            all_layers_attention,
            scale,
            vmax_percentile,
            shared_vmax,
            linear_vmax,
        )
        _save_attention_image(
            plot_dir / f"attention_{class_name}_all_layers_mean{filename_suffix}",
            plot_values,
            f"TxT attention {class_name} - all layers mean ({scale})",
            gene_names,
            "magma",
            colorbar_label,
            vmin,
            vmax,
        )

    all_layers_delta = delta_attention.mean(axis=(0, 1))
    delta_vmax = float(np.percentile(np.abs(all_layers_delta), vmax_percentile))
    if delta_vmax <= 0:
        delta_vmax = float(np.max(np.abs(all_layers_delta)))
    if delta_vmax <= 0:
        delta_vmax = 1.0
    if save_csv:
        pd.DataFrame(all_layers_delta, index=gene_names, columns=gene_names).to_csv(
            plot_dir / f"attention_delta_{positive_class}_minus_{negative_class}_all_layers_mean.csv"
        )
    _save_attention_image(
        plot_dir / f"attention_delta_{positive_class}_minus_{negative_class}_all_layers_mean{filename_suffix}",
        np.clip(all_layers_delta, -delta_vmax, delta_vmax),
        f"TxT attention delta {positive_class} - {negative_class} - all layers mean",
        gene_names,
        "coolwarm",
        f"Attention difference ({positive_class} - {negative_class})",
        -delta_vmax,
        delta_vmax,
    )


def transform_attention_for_plot(
    attention: np.ndarray,
    scale: str,
    vmax_percentile: float,
    shared_vmax: float,
    linear_vmax: float | None = None,
) -> tuple[np.ndarray, str, float | None, float | None]:
    if scale == "linear":
        if linear_vmax is not None and linear_vmax > 0:
            return np.clip(attention, 0.0, linear_vmax), "Attention weight", 0.0, linear_vmax
        return attention, "Attention weight", 0.0, shared_vmax
    if scale == "percentile":
        vmax = float(np.percentile(attention, vmax_percentile))
        if vmax <= 0:
            return attention, "Attention weight", None, None
        return np.clip(attention, 0.0, vmax), f"Attention weight clipped at p{vmax_percentile:g}", 0.0, vmax
    if scale == "log":
        return np.log10(attention + 1e-12), "log10(attention weight + 1e-12)", None, None
    if scale == "relative":
        baseline = 1.0 / attention.shape[1]
        return attention / baseline, "Attention weight / uniform baseline", None, None
    raise ValueError(f"Unsupported attention plot scale: {scale}")


def resolve_model_param(args_value: Any, config: dict[str, Any], key: str, fallback: Any) -> Any:
    if args_value is not None:
        return args_value
    value = config.get(key)
    return fallback if value is None else value


def validate_ad_mci_deg(dataset: PreparedDataset) -> None:
    if len(dataset.gene_names) != 788:
        raise ValueError(f"Expected 788 DEG genes, found {len(dataset.gene_names)}.")
    if dataset.class_names != ["MCI", "AD"]:
        raise ValueError(f"Expected class_names ['MCI', 'AD'], found {dataset.class_names}.")


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
    if keep.size == 0:
        raise ValueError("No task genes overlap with pretrained genes.")
    return subset_dataset(dataset, keep)


def fixed_gene_subset(
    dataset: PreparedDataset,
    pretrained_gene_names: list[str],
    count: int,
    pool: str,
    mode: str,
    seed: int,
) -> tuple[PreparedDataset, dict[str, Any]]:
    if count <= 0:
        return dataset, {
            "selected_gene_count": 0,
            "gene_selection_pool": "none",
            "gene_selection_pool_size": 0,
            "gene_selection_mode": "none",
        }

    pretrained_set = set(pretrained_gene_names)
    if pool == "pretrained_overlap":
        pool_indices = np.array([idx for idx, gene in enumerate(dataset.gene_names) if gene in pretrained_set], dtype=np.int64)
    else:
        pool_indices = np.arange(len(dataset.gene_names), dtype=np.int64)

    if pool_indices.size == 0:
        raise ValueError("Random gene pool is empty.")
    if count > pool_indices.size:
        raise ValueError(f"random_gene_count={count} is larger than pool size {pool_indices.size}.")

    if mode == "random":
        rng = np.random.default_rng(seed)
        selected = rng.choice(pool_indices, size=count, replace=False).astype(np.int64)
    else:
        variances = np.var(dataset.train_gene_x[:, pool_indices], axis=0)
        order = np.argsort(variances)[::-1]
        selected = pool_indices[order[:count]].astype(np.int64)
    selected.sort()
    return subset_dataset(dataset, selected), {
        "random_gene_count": int(count),
        "random_gene_pool": pool,
        "random_gene_sampling": "fixed_once_per_run" if mode == "random" else "top_train_variance",
        "selected_gene_count": int(count),
        "gene_selection_pool": pool,
        "gene_selection_pool_size": int(pool_indices.size),
        "gene_selection_mode": mode,
        "selected_gene_names": [dataset.gene_names[int(idx)] for idx in selected],
    }


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


def main() -> None:
    args = parse_args()
    if args.random_gene_pool is not None:
        args.gene_selection_pool = args.random_gene_pool
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
        split_file=None,
        split_seed=args.split_seed,
        split_mode=args.split_mode,
    )
    original_gene_count = len(dataset.gene_names)
    filter_applied = False
    if args.filter_to_pretrained_genes:
        dataset = filter_dataset_to_pretrained_genes(dataset, pretrained_gene_names)
        filter_applied = True
        print(
            "Filtered task genes to pretrained overlap | "
            f"before={original_gene_count} | after={len(dataset.gene_names)}",
            flush=True,
        )

    gene_selection_report: dict[str, Any]
    dataset, gene_selection_report = fixed_gene_subset(
        dataset=dataset,
        pretrained_gene_names=pretrained_gene_names,
        count=args.random_gene_count,
        pool=args.gene_selection_pool,
        mode=args.gene_selection_mode,
        seed=args.seed + args.split_seed,
    )

    if args.random_gene_count <= 0 and not args.skip_deg_validation and not filter_applied:
        validate_ad_mci_deg(dataset)

    remapped_embedding_df, remap_report = remap_embedding_dataframe(
        source_embedding_df,
        target_genes=dataset.gene_names,
        seed=args.seed,
    )
    embedding_path = result_dir / "gene_embedding_from_pretraining.csv"
    remapped_embedding_df.to_csv(embedding_path)

    n_heads = int(resolve_model_param(args.n_heads, checkpoint_config, "n_heads", 4))
    n_layers = int(resolve_model_param(args.n_layers, checkpoint_config, "n_layers", 3))
    d_model = int(resolve_model_param(args.d_model, checkpoint_config, "d_model", 128))
    d_ff = int(resolve_model_param(args.d_ff, checkpoint_config, "d_ff", 512))
    dropout = float(resolve_model_param(args.dropout, checkpoint_config, "dropout", 0.2))
    norm_first = bool(args.norm_first or checkpoint_config.get("norm_first", False))

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
        else:
            transfer_report = {
                "loaded_transformer_keys": [],
                "skipped_pretrained_keys": [
                    key
                    for key in pretrained_state.keys()
                    if key.startswith("transformer.") and key != "transformer.encoder.embed.embed.weight"
                ],
                "transfer_mode_note": "Only remapped pretrained gene embeddings were used; transformer layers were randomly initialized.",
            }

        transfer_report.update(remap_report)
        transfer_report.update(gene_selection_report)
        transfer_report["transfer_mode"] = args.transfer_mode
        transfer_report["task_genes_before_pretrained_filter"] = int(original_gene_count)
        transfer_report["task_genes_after_pretrained_filter"] = int(len(dataset.gene_names))
        transfer_report["filter_to_pretrained_genes"] = bool(filter_applied)
        save_json(result_dir / "transfer_report.json", transfer_report)

        save_args(
            result_dir,
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
                "gene_selection_report": gene_selection_report,
            },
        )
        save_selected_genes(result_dir, dataset.gene_names)
        save_split_assignments(result_dir, dataset.train_ids, dataset.val_ids, dataset.test_ids)
        if args.plot_finetuning_matrices:
            save_finetuning_input_matrix_plot(
                result_dir=result_dir,
                dataset=dataset,
                max_samples=args.plot_input_max_samples,
            )

        class_weights = compute_balanced_class_weights(dataset.train_y) if args.class_weighting == "on" else None
        history_rows, best_state_dict, training_summary = train_classifier(
            model=model,
            train_gene_x=dataset.train_gene_x,
            train_y=dataset.train_y,
            val_gene_x=dataset.val_gene_x,
            val_y=dataset.val_y,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            class_weights=class_weights,
            early_stopping_patience=args.early_stopping_patience,
            device=device,
            logits_fn=logits_fn,
            max_train_batches=args.max_train_batches,
            max_val_batches=args.max_val_batches,
        )
        save_training_history(result_dir, history_rows)

        best_model_path = result_dir / "best_model.pt"
        torch.save(best_state_dict, best_model_path)
        model.load_state_dict(best_state_dict)
        if args.plot_finetuning_matrices:
            attention_matrices = collect_attention_matrices(
                model=model,
                gene_x=dataset.train_gene_x,
                device=device,
                batch_size=args.batch_size,
                max_samples=args.plot_attention_max_samples,
            )
            save_attention_matrix_plots(result_dir, attention_matrices, dataset.gene_names)
            class_attention = collect_attention_matrices_by_class(
                model=model,
                gene_x=dataset.train_gene_x,
                y=dataset.train_y,
                class_names=dataset.class_names,
                device=device,
                batch_size=args.batch_size,
                max_samples_per_class=args.plot_attention_max_samples,
            )
            save_class_attention_matrix_plots(result_dir, class_attention, dataset.gene_names)

        split_results = {
            "train": evaluate_classifier(
                model,
                dataset.train_gene_x,
                dataset.train_y,
                dataset.train_ids,
                args.batch_size,
                device,
                dataset.class_names,
                logits_fn,
            ),
            "val": evaluate_classifier(
                model,
                dataset.val_gene_x,
                dataset.val_y,
                dataset.val_ids,
                args.batch_size,
                device,
                dataset.class_names,
                logits_fn,
            ),
            "test": evaluate_classifier(
                model,
                dataset.test_gene_x,
                dataset.test_y,
                dataset.test_ids,
                args.batch_size,
                device,
                dataset.class_names,
                logits_fn,
            ),
        }

    for split_name, split_result in split_results.items():
        print_report(split_name, split_result["report_df"])

    save_metrics_summary(result_dir, split_results)
    save_test_artifacts(result_dir, split_results["test"], dataset.class_names)
    save_json(
        result_dir / "model_summary.json",
        {
            "model": "txt_from_pretraining_simple_baseline_loop",
            "num_genes": len(dataset.gene_names),
            "num_classes": len(dataset.class_names),
            "best_model_path": str(best_model_path),
            "training_summary": training_summary,
        },
    )
    print(f"Fine-tuning complete. Results: {result_dir}")


if __name__ == "__main__":
    main()
