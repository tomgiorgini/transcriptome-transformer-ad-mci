#!/usr/bin/env python3
"""Post-training dense TxT attention analysis.

The script reconstructs a saved multitask TxT run, loads its selected
checkpoint and analyses attention on one held-out split without materialising
all sample x head x gene x gene matrices at once.

Dense TxT attention follows ``A[head, query_gene, key_gene]``.  Since every
query row is softmax-normalised, key genes are ranked by incoming attention,
whereas query genes are ranked by row selectivity (one minus normalised
entropy).  Q/K vector norms are exported separately and are never labelled as
attention mass.

For PPI-volumetric checkpoints the script also analyses sparse VMA weights.
Sparse key enrichment is corrected for the opportunity induced by target-node
degree; sparse query selectivity is defined only for targets with at least two
neighbours.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import torch


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.scripts.paper_comparison import train_txt_multitask as worker  # noqa: E402
from experiments.scripts.paper_comparison.txt_volumetric.common import (  # noqa: E402
    ProtocolError,
    resolve_path,
    write_json,
)
from experiments.scripts.paper_comparison.txt_volumetric.export_attention import (  # noqa: E402
    namespace_from_saved_args,
    verify_selected_genes,
)
from source.models.txt_volumetric import load_induced_ppi_graph  # noqa: E402
from source.models.txt_volumetric.layers import VolumetricAttentionAugmentation  # noqa: E402
from source.pipeline.utils import resolve_device, set_seed  # noqa: E402


DEFAULT_CLINICAL_FILE = (
    ROOT / "task_dataset" / "processed" / "alzheimer_multiclass" / "clinical.csv"
)
CLASS_CONTRASTS = (
    ("AD_minus_MCI", "AD", "MCI"),
    ("AD_minus_Control", "AD", "Control"),
    ("MCI_minus_Control", "MCI", "Control"),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Analyse post-training TxT attention by biological class. Dense rows are queries "
            "and dense columns are keys; query rows are ranked by selectivity, not row sum."
        )
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--query-top-k", type=int, default=10)
    parser.add_argument("--top-genes", type=int, default=25)
    parser.add_argument("--matrix-top-genes", type=int, default=48)
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--clinical-file", type=Path, default=DEFAULT_CLINICAL_FILE)
    parser.add_argument("--x-file", type=Path, default=None)
    parser.add_argument("--y-file", type=Path, default=None)
    parser.add_argument("--split-file", type=Path, default=None)
    parser.add_argument("--ppi-edge-file", type=Path, default=None)
    parser.add_argument(
        "--skip-vma",
        action="store_true",
        help="Skip sparse VMA analysis even when the checkpoint contains a VMA branch.",
    )
    return parser


def _resolve_override(path: Path | None) -> Path | None:
    return resolve_path(path) if path is not None else None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reconstruct_model(
    run_dir: Path,
    checkpoint: Path,
    device: torch.device,
    *,
    x_file: Path | None = None,
    y_file: Path | None = None,
    split_file: Path | None = None,
    ppi_edge_file: Path | None = None,
) -> tuple[argparse.Namespace, Any, list[Any], torch.nn.Module]:
    args = namespace_from_saved_args(run_dir / "args.json", run_dir)
    overrides = {
        "x_file": x_file,
        "y_file": y_file,
        "split_file": split_file,
        "ppi_edge_file": ppi_edge_file,
    }
    for name, value in overrides.items():
        if value is not None:
            setattr(args, name, value)

    required_paths = ["x_file", "y_file"]
    if getattr(args, "split_mode", None) == "custom":
        required_paths.append("split_file")
    if getattr(args, "model_variant", "baseline") == "ppi_volumetric":
        required_paths.append("ppi_edge_file")
    missing = [
        name
        for name in required_paths
        if getattr(args, name, None) is None or not Path(getattr(args, name)).exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Saved run paths are unavailable for: "
            + ", ".join(missing)
            + ". Supply the corresponding command-line override(s)."
        )

    set_seed(int(args.seed))
    resolved_split_file = worker.resolve_split_file(args)
    dataset, _, task_gene_indices, _ = worker.prepare_multitask_dataset(
        args, resolved_split_file
    )
    verify_selected_genes(run_dir, dataset.gene_names)
    task_specs = worker.build_task_specs(dataset.class_names)

    embedding_path = run_dir / "gene_embedding.csv"
    if not embedding_path.exists():
        raise FileNotFoundError(f"Missing constructor embedding artifact: {embedding_path}")
    ppi_prior_path = run_dir / "ppi_prior_embedding.csv"
    if not ppi_prior_path.exists():
        ppi_prior_path = None

    graph = None
    if getattr(args, "model_variant", "baseline") == "ppi_volumetric":
        graph = load_induced_ppi_graph(
            Path(args.ppi_edge_file),
            dataset.gene_names,
            score_threshold=float(args.ppi_score_threshold),
        )
    model = worker.build_txt_model(
        args,
        dataset,
        task_specs,
        task_gene_indices,
        embedding_path,
        ppi_prior_path,
        graph,
    ).to(device)
    state = worker.load_checkpoint_state(checkpoint, device)
    worker.validate_checkpoint_graph(state, graph)
    model.load_state_dict(state, strict=True)
    model.eval()
    return args, dataset, task_specs, model


def split_arrays(dataset: Any, split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.asarray(getattr(dataset, f"{split}_gene_x"), dtype=np.float32),
        np.asarray(getattr(dataset, f"{split}_y"), dtype=np.int64),
        np.asarray(getattr(dataset, f"{split}_ids")).astype(str),
    )


def load_cohorts(clinical_file: Path, sample_ids: np.ndarray) -> np.ndarray:
    if not clinical_file.exists():
        return np.asarray(["Unknown"] * len(sample_ids), dtype=str)
    clinical = pd.read_csv(clinical_file, usecols=["sample_id", "dataset_gse"])
    cohort_by_id = dict(
        zip(
            clinical["sample_id"].astype(str),
            clinical["dataset_gse"].fillna("Unknown").astype(str),
        )
    )
    return np.asarray([cohort_by_id.get(str(value), "Unknown") for value in sample_ids])


def _encoder_attention_context(
    model: torch.nn.Module,
    batch_x: torch.Tensor,
) -> tuple[Any, Any, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if getattr(model, "encoder_sharing", None) != "shared":
        raise ProtocolError("This analysis currently requires encoder_sharing='shared'.")
    transformer = model.encoder_modules()[0]
    encoder = transformer.encoder
    if len(encoder.layers) != 1:
        raise ProtocolError(
            "This exact streaming analysis currently supports the one-layer TxT used in the thesis."
        )
    layer = encoder.layers[0]
    attention_layer = layer.multi_head_attention_layer
    batch_size, n_genes = batch_x.shape
    gene_indices = torch.arange(n_genes, device=batch_x.device).expand(batch_size, -1)
    hidden = encoder.embed(gene_indices)
    if encoder.ppi_prior is not None and encoder.ppi_gate is not None:
        hidden = hidden + torch.tanh(encoder.ppi_gate) * encoder.ppi_prior[gene_indices]
    if encoder.expression_projection is not None:
        hidden = hidden + encoder.expression_scale * encoder.expression_projection(
            batch_x.unsqueeze(-1)
        )
    attention_input = layer.layer_norm_1(hidden) if layer.norm_first else hidden

    q = attention_layer.q_linear(attention_input).view(
        batch_size, n_genes, attention_layer.n_heads, attention_layer.d_head
    ).transpose(1, 2)
    k = attention_layer.k_linear(attention_input).view(
        batch_size, n_genes, attention_layer.n_heads, attention_layer.d_head
    ).transpose(1, 2)
    v = attention_layer.v_linear(attention_input).view(
        batch_size, n_genes, attention_layer.n_heads, attention_layer.d_head
    ).transpose(1, 2)

    expression = batch_x.unsqueeze(-1)
    tupe_q = encoder.tupe.q_linear(expression).view(
        batch_size, n_genes, encoder.tupe.n_heads, encoder.tupe.d_head
    ).transpose(1, 2)
    tupe_k = encoder.tupe.k_linear(expression).view(
        batch_size, n_genes, encoder.tupe.n_heads, encoder.tupe.d_head
    ).transpose(1, 2)
    if encoder.tupe_mode == "on":
        tupe = torch.matmul(tupe_q, tupe_k.transpose(-2, -1)) / math.sqrt(
            2 * encoder.tupe.d_head
        )
    else:
        tupe = batch_x.new_zeros(
            (batch_size, attention_layer.n_heads, n_genes, n_genes)
        )
    return encoder, attention_layer, q, k, v, tupe_q, tupe_k, tupe


def _rank_desc(values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    safe = np.where(finite, values, -np.inf)
    order = np.lexsort((np.arange(len(values)), -safe))
    ranks = np.empty(len(values), dtype=np.int64)
    ranks[order] = np.arange(1, len(values) + 1)
    ranks[~finite] = len(values) + 1
    return ranks


def nanmean_silent(values: np.ndarray, axis: int | tuple[int, ...]) -> np.ndarray:
    values = np.asarray(values)
    finite = np.isfinite(values)
    count = finite.sum(axis=axis)
    total = np.where(finite, values, 0.0).sum(axis=axis)
    return np.divide(
        total,
        count,
        out=np.full_like(total, np.nan, dtype=np.result_type(values, np.float64)),
        where=count > 0,
    )


def nanstd_silent(values: np.ndarray, axis: int | tuple[int, ...]) -> np.ndarray:
    values = np.asarray(values)
    mean = nanmean_silent(values, axis=axis)
    if isinstance(axis, tuple):
        expanded = mean
        for item in sorted(axis):
            expanded = np.expand_dims(expanded, item)
    else:
        expanded = np.expand_dims(mean, axis)
    finite = np.isfinite(values)
    count = finite.sum(axis=axis)
    squared = np.where(finite, np.square(values - expanded), 0.0).sum(axis=axis)
    variance = np.divide(
        squared,
        count,
        out=np.full_like(squared, np.nan, dtype=np.result_type(values, np.float64)),
        where=count > 0,
    )
    return np.sqrt(variance)


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=np.float64)
    result = np.full_like(p, np.nan)
    finite_indices = np.flatnonzero(np.isfinite(p))
    if not len(finite_indices):
        return result
    finite_p = p[finite_indices]
    order = np.argsort(finite_p, kind="mergesort")
    ranked = finite_p[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    restored = np.empty_like(adjusted)
    restored[order] = adjusted
    result[finite_indices] = restored
    return result


def adjusted_class_contrast(
    values: np.ndarray,
    labels: np.ndarray,
    cohorts: np.ndarray,
    positive_label: int,
    negative_label: int,
) -> dict[str, np.ndarray | float | int]:
    values = np.asarray(values, dtype=np.float64)
    keep = np.isin(labels, [positive_label, negative_label])
    y = values[keep]
    class_indicator = (labels[keep] == positive_label).astype(np.float64)
    cohort_values = cohorts[keep].astype(str)
    cohort_levels = sorted(set(cohort_values))

    design_columns = [np.ones(len(y), dtype=np.float64), class_indicator]
    for level in cohort_levels[1:]:
        design_columns.append((cohort_values == level).astype(np.float64))
    design = np.column_stack(design_columns)
    design_rank = int(np.linalg.matrix_rank(design))
    degrees_freedom = max(len(y) - design_rank, 1)
    design_full_rank = design_rank == design.shape[1]
    if design_full_rank:
        inverse = np.linalg.inv(design.T @ design)
        coefficients = inverse @ design.T @ y
        fitted = design @ coefficients
        residuals = y - fitted
        residual_variance = np.sum(np.square(residuals), axis=0) / degrees_freedom
        standard_error = np.sqrt(
            np.maximum(inverse[1, 1] * residual_variance, 0.0)
        )
        class_beta = coefficients[1]
        t_statistic = np.divide(
            class_beta,
            standard_error,
            out=np.zeros_like(class_beta),
            where=standard_error > 0,
        )
        p_value = 2.0 * stats.t.sf(np.abs(t_statistic), degrees_freedom)
    else:
        feature_count = values.shape[1]
        class_beta = np.full(feature_count, np.nan)
        standard_error = np.full(feature_count, np.nan)
        t_statistic = np.full(feature_count, np.nan)
        p_value = np.full(feature_count, np.nan)

    positive_values = values[labels == positive_label]
    negative_values = values[labels == negative_label]
    positive_mean = np.nanmean(positive_values, axis=0)
    negative_mean = np.nanmean(negative_values, axis=0)
    positive_var = np.nanvar(positive_values, axis=0, ddof=1)
    negative_var = np.nanvar(negative_values, axis=0, ddof=1)
    n_positive = positive_values.shape[0]
    n_negative = negative_values.shape[0]
    pooled_variance = (
        (n_positive - 1) * positive_var + (n_negative - 1) * negative_var
    ) / max(n_positive + n_negative - 2, 1)
    standardized = np.divide(
        positive_mean - negative_mean,
        np.sqrt(np.maximum(pooled_variance, 0.0)),
        out=np.zeros_like(positive_mean),
        where=pooled_variance > 0,
    )
    correction = 1.0 - 3.0 / max(4.0 * (n_positive + n_negative) - 9.0, 1.0)

    interaction_beta = np.full(values.shape[1], np.nan)
    interaction_p = np.full(values.shape[1], np.nan)
    if len(cohort_levels) == 2:
        cohort_indicator = (cohort_values == cohort_levels[1]).astype(np.float64)
        interaction_design = np.column_stack(
            [
                np.ones(len(y)),
                class_indicator,
                cohort_indicator,
                class_indicator * cohort_indicator,
            ]
        )
        if np.linalg.matrix_rank(interaction_design) == interaction_design.shape[1]:
            inv_interaction = np.linalg.pinv(interaction_design.T @ interaction_design)
            interaction_coefficients = (
                inv_interaction @ interaction_design.T @ y
            )
            interaction_residuals = y - interaction_design @ interaction_coefficients
            interaction_df = max(len(y) - interaction_design.shape[1], 1)
            interaction_variance = (
                np.sum(np.square(interaction_residuals), axis=0) / interaction_df
            )
            interaction_se = np.sqrt(
                np.maximum(inv_interaction[3, 3] * interaction_variance, 0.0)
            )
            interaction_beta = interaction_coefficients[3]
            interaction_t = np.divide(
                interaction_beta,
                interaction_se,
                out=np.zeros_like(interaction_beta),
                where=interaction_se > 0,
            )
            interaction_p = 2.0 * stats.t.sf(np.abs(interaction_t), interaction_df)

    return {
        "n_positive": n_positive,
        "n_negative": n_negative,
        "positive_mean": positive_mean,
        "negative_mean": negative_mean,
        "raw_difference": positive_mean - negative_mean,
        "adjusted_beta": class_beta,
        "standard_error": standard_error,
        "t_statistic": t_statistic,
        "p_value": p_value,
        "q_value": benjamini_hochberg(p_value),
        "hedges_g": correction * standardized,
        "cohort_interaction_beta": interaction_beta,
        "cohort_interaction_p": interaction_p,
        "cohort_levels": cohort_levels,
        "degrees_freedom": degrees_freedom,
        "design_rank": design_rank,
        "design_columns": int(design.shape[1]),
        "design_full_rank": design_full_rank,
    }


def _scatter_sum(
    values: torch.Tensor,
    indices: torch.Tensor,
    n_genes: int,
) -> torch.Tensor:
    output = values.new_zeros((*values.shape[:-1], n_genes))
    expanded = indices.view(*([1] * (values.ndim - 1)), -1).expand_as(values)
    output.scatter_add_(-1, expanded, values)
    return output


def _scatter_max(
    values: torch.Tensor,
    indices: torch.Tensor,
    n_genes: int,
) -> torch.Tensor:
    output = values.new_zeros((*values.shape[:-1], n_genes))
    expanded = indices.view(*([1] * (values.ndim - 1)), -1).expand_as(values)
    output.scatter_reduce_(-1, expanded, values, reduce="amax", include_self=True)
    return output


def _finalize_class_attention_sums(
    class_matrix_sums: torch.Tensor,
    class_counts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-class and all-sample means without mutating shared storage."""
    class_counts = np.asarray(class_counts, dtype=np.int64)
    if class_matrix_sums.shape[0] != len(class_counts):
        raise ValueError("Class counts are not aligned with attention-matrix sums.")
    sample_count = int(class_counts.sum())
    if sample_count <= 0:
        raise ValueError("At least one sample is required to finalize attention matrices.")
    class_sum_array = class_matrix_sums.detach().cpu().numpy()
    overall_mean = class_sum_array.sum(axis=0) / float(sample_count)
    class_means = class_sum_array.copy()
    for class_index, count in enumerate(class_counts):
        if count:
            class_means[class_index] /= float(count)
    return class_means, overall_mean


@torch.inference_mode()
def collect_attention(
    model: torch.nn.Module,
    gene_x: np.ndarray,
    labels: np.ndarray,
    *,
    class_count: int,
    batch_size: int,
    device: torch.device,
    query_top_k: int,
    include_vma: bool,
) -> dict[str, Any]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    n_samples, n_genes = gene_x.shape
    if not 0 < query_top_k <= n_genes:
        raise ValueError("query_top_k must be between one and the gene count.")

    first_batch = torch.as_tensor(gene_x[:1], dtype=torch.float32, device=device)
    _, attention_layer, _, _, _, _, _, _ = _encoder_attention_context(model, first_batch)
    n_heads = attention_layer.n_heads
    class_matrix_sums = torch.zeros(
        (class_count, n_heads, n_genes, n_genes), dtype=torch.float32
    )
    class_counts = np.zeros(class_count, dtype=np.int64)
    metric_parts: dict[str, list[np.ndarray]] = {
        name: []
        for name in (
            "incoming_enrichment",
            "query_specificity",
            "query_max_attention",
            "query_topk_mass",
            "self_attention",
            "q_norm",
            "k_norm",
            "tupe_q_norm",
            "tupe_k_norm",
        )
    }
    qc_rows: list[dict[str, Any]] = []

    vma = attention_layer.attention_augmentation
    if not isinstance(vma, VolumetricAttentionAugmentation) or not include_vma:
        vma = None
    vma_parts: dict[str, list[np.ndarray]] = {
        "key_degree_corrected_enrichment": [],
        "query_specificity": [],
        "query_max_attention": [],
    }
    vma_class_edge_sums: dict[str, torch.Tensor] | None = None
    vma_edge_index: np.ndarray | None = None
    vma_expected_key_mass: np.ndarray | None = None
    vma_degrees: np.ndarray | None = None
    if vma is not None:
        n_edges = int(vma.edge_index.shape[1])
        vma_class_edge_sums = {
            field: torch.zeros((class_count, n_heads, n_edges), dtype=torch.float32)
            for field in ("attention_weights", "volumes", "logits")
        }
        vma_edge_index = vma.edge_index.detach().cpu().numpy().astype(np.int64)
        target_cpu = torch.as_tensor(vma_edge_index[0], dtype=torch.long)
        source_cpu = torch.as_tensor(vma_edge_index[1], dtype=torch.long)
        degrees_cpu = torch.bincount(target_cpu, minlength=n_genes).float()
        null_edge_weight = 1.0 / degrees_cpu[target_cpu].clamp_min(1.0)
        vma_expected_key_mass = (
            _scatter_sum(null_edge_weight.view(1, 1, -1), source_cpu, n_genes)
            .squeeze(0)
            .squeeze(0)
            .numpy()
        )
        vma_degrees = degrees_cpu.numpy().astype(np.int64)

    for batch_number, start in enumerate(range(0, n_samples, batch_size), start=1):
        stop = min(start + batch_size, n_samples)
        batch_x = torch.as_tensor(gene_x[start:stop], dtype=torch.float32, device=device)
        (
            _,
            attention_layer,
            q,
            k,
            v,
            tupe_q,
            tupe_k,
            tupe,
        ) = _encoder_attention_context(model, batch_x)
        logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(
            2 * attention_layer.d_head
        ) + tupe
        attention = torch.softmax(logits, dim=-1)
        row_sums = attention.sum(dim=-1)
        entropy = -(
            attention * attention.clamp_min(torch.finfo(attention.dtype).tiny).log()
        ).sum(dim=-1)
        batch_metrics = {
            "incoming_enrichment": n_genes * attention.mean(dim=-2),
            "query_specificity": 1.0 - entropy / math.log(n_genes),
            "query_max_attention": attention.amax(dim=-1),
            "query_topk_mass": torch.topk(
                attention, k=query_top_k, dim=-1, sorted=False
            ).values.sum(dim=-1),
            "self_attention": attention.diagonal(dim1=-2, dim2=-1),
            "q_norm": q.float().norm(dim=-1),
            "k_norm": k.float().norm(dim=-1),
            "tupe_q_norm": tupe_q.float().norm(dim=-1),
            "tupe_k_norm": tupe_k.float().norm(dim=-1),
        }
        for name, values in batch_metrics.items():
            metric_parts[name].append(values.detach().cpu().numpy().astype(np.float32))

        batch_labels = labels[start:stop]
        attention_cpu = attention.detach().cpu()
        for class_index in range(class_count):
            class_mask = np.flatnonzero(batch_labels == class_index)
            if not len(class_mask):
                continue
            class_matrix_sums[class_index] += attention_cpu[class_mask].sum(dim=0)
            class_counts[class_index] += len(class_mask)

        qc_rows.append(
            {
                "batch": batch_number,
                "start": start,
                "stop": stop,
                "samples": stop - start,
                "row_sum_max_abs_error": float((row_sums - 1.0).abs().max().cpu()),
                "attention_min": float(attention.min().cpu()),
                "attention_max": float(attention.max().cpu()),
                "attention_nonfinite": int((~torch.isfinite(attention)).sum().cpu()),
            }
        )

        if vma is not None:
            _, weights, volumes, sparse_logits = vma.compute_vma_output(
                q,
                k,
                v,
                tupe,
                expression=batch_x,
            )
            target = vma.destination_index
            source = vma.source_index
            assert vma_expected_key_mass is not None and vma_degrees is not None
            observed_key_mass = _scatter_sum(weights, source, n_genes)
            expected = torch.as_tensor(
                vma_expected_key_mass, dtype=weights.dtype, device=weights.device
            )
            key_enrichment = observed_key_mass / expected.view(1, 1, -1).clamp_min(
                torch.finfo(weights.dtype).tiny
            )
            key_enrichment[..., expected == 0] = torch.nan
            sparse_entropy = _scatter_sum(
                -(weights * weights.clamp_min(torch.finfo(weights.dtype).tiny).log()),
                target,
                n_genes,
            )
            degree_tensor = torch.as_tensor(
                vma_degrees, dtype=weights.dtype, device=weights.device
            )
            query_specificity = 1.0 - sparse_entropy / degree_tensor.clamp_min(2).log()
            query_specificity[..., degree_tensor < 2] = torch.nan
            query_max = _scatter_max(weights, target, n_genes)
            query_max[..., degree_tensor == 0] = torch.nan
            for name, values in {
                "key_degree_corrected_enrichment": key_enrichment,
                "query_specificity": query_specificity,
                "query_max_attention": query_max,
            }.items():
                vma_parts[name].append(
                    values.detach().cpu().numpy().astype(np.float32)
                )
            assert vma_class_edge_sums is not None
            for class_index in range(class_count):
                class_mask = np.flatnonzero(batch_labels == class_index)
                if not len(class_mask):
                    continue
                vma_class_edge_sums["attention_weights"][class_index] += (
                    weights.detach().cpu()[class_mask].sum(dim=0)
                )
                vma_class_edge_sums["volumes"][class_index] += (
                    volumes.detach().cpu()[class_mask].sum(dim=0)
                )
                vma_class_edge_sums["logits"][class_index] += (
                    sparse_logits.detach().cpu()[class_mask].sum(dim=0)
                )

    if not int(class_counts.sum()) == n_samples:
        raise RuntimeError("Class-matrix aggregation did not account for every sample.")
    class_mean_attention, overall_mean_attention = _finalize_class_attention_sums(
        class_matrix_sums,
        class_counts,
    )

    result: dict[str, Any] = {
        "metrics": {
            name: np.concatenate(parts, axis=0) for name, parts in metric_parts.items()
        },
        "class_mean_attention": class_mean_attention,
        "overall_mean_attention": overall_mean_attention,
        "class_counts": class_counts,
        "qc": pd.DataFrame(qc_rows),
    }
    if vma is not None:
        assert vma_class_edge_sums is not None
        vma_class_means: dict[str, np.ndarray] = {}
        for field, sums in vma_class_edge_sums.items():
            means = sums.numpy()
            for class_index, count in enumerate(class_counts):
                if count:
                    means[class_index] /= float(count)
            vma_class_means[field] = means
        result["vma"] = {
            "metrics": {
                name: np.concatenate(parts, axis=0) for name, parts in vma_parts.items()
            },
            "class_edge_means": vma_class_means,
            "edge_index": vma_edge_index,
            "expected_key_mass": vma_expected_key_mass,
            "degrees": vma_degrees,
        }
    return result


def build_gene_summary(
    metrics: dict[str, np.ndarray],
    gene_x: np.ndarray,
    labels: np.ndarray,
    class_names: Sequence[str],
    gene_names: Sequence[str],
) -> pd.DataFrame:
    n_samples, n_heads, n_genes = next(iter(metrics.values())).shape
    if n_samples != len(labels) or n_genes != len(gene_names):
        raise ValueError("Metric tensors are not aligned to labels/gene names.")
    class_groups: list[tuple[str, np.ndarray]] = [("ALL", np.ones(n_samples, dtype=bool))]
    class_groups.extend(
        (str(class_name), labels == class_index)
        for class_index, class_name in enumerate(class_names)
    )
    frames: list[pd.DataFrame] = []
    genes = np.asarray(gene_names, dtype=str)
    for class_name, keep in class_groups:
        if not bool(keep.any()):
            continue
        expression_mean = np.nanmean(gene_x[keep], axis=0)
        expression_std = np.nanstd(gene_x[keep], axis=0)
        for head_label, head_values in [
            *[(str(head), {name: values[keep, head] for name, values in metrics.items()}) for head in range(n_heads)],
            (
                "mean_heads",
                {name: np.nanmean(values[keep], axis=1) for name, values in metrics.items()},
            ),
        ]:
            payload: dict[str, Any] = {
                "class_name": class_name,
                "head": head_label,
                "sample_count": int(keep.sum()),
                "gene_index": np.arange(n_genes, dtype=np.int64),
                "gene": genes,
                "expression_mean": expression_mean,
                "expression_std": expression_std,
            }
            for name, values in head_values.items():
                payload[f"{name}_mean"] = np.nanmean(values, axis=0)
                payload[f"{name}_std"] = np.nanstd(values, axis=0)
            frame = pd.DataFrame(payload)
            frame["incoming_log2_enrichment"] = np.log2(
                frame["incoming_enrichment_mean"].clip(lower=np.finfo(float).tiny)
            )
            frame["key_rank"] = _rank_desc(
                frame["incoming_enrichment_mean"].to_numpy()
            )
            frame["query_rank"] = _rank_desc(
                frame["query_specificity_mean"].to_numpy()
            )
            frame["q_norm_rank"] = _rank_desc(frame["q_norm_mean"].to_numpy())
            frame["k_norm_rank"] = _rank_desc(frame["k_norm_mean"].to_numpy())
            frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def build_contrasts(
    metrics: dict[str, np.ndarray],
    labels: np.ndarray,
    cohorts: np.ndarray,
    class_names: Sequence[str],
    gene_names: Sequence[str],
) -> pd.DataFrame:
    class_index = {str(name): idx for idx, name in enumerate(class_names)}
    rows: list[pd.DataFrame] = []
    contrast_metrics = {
        "incoming_enrichment": np.nanmean(metrics["incoming_enrichment"], axis=1),
        "query_specificity": np.nanmean(metrics["query_specificity"], axis=1),
    }
    for contrast_name, positive_name, negative_name in CLASS_CONTRASTS:
        if positive_name not in class_index or negative_name not in class_index:
            continue
        for metric_name, values in contrast_metrics.items():
            result = adjusted_class_contrast(
                values,
                labels,
                cohorts,
                class_index[positive_name],
                class_index[negative_name],
            )
            rows.append(
                pd.DataFrame(
                    {
                        "contrast": contrast_name,
                        "positive_class": positive_name,
                        "negative_class": negative_name,
                        "metric": metric_name,
                        "gene_index": np.arange(len(gene_names), dtype=np.int64),
                        "gene": np.asarray(gene_names, dtype=str),
                        "n_positive": result["n_positive"],
                        "n_negative": result["n_negative"],
                        "positive_mean": result["positive_mean"],
                        "negative_mean": result["negative_mean"],
                        "raw_difference": result["raw_difference"],
                        "cohort_adjusted_beta": result["adjusted_beta"],
                        "standard_error": result["standard_error"],
                        "t_statistic": result["t_statistic"],
                        "p_value": result["p_value"],
                        "q_value": result["q_value"],
                        "hedges_g": result["hedges_g"],
                        "cohort_interaction_beta": result["cohort_interaction_beta"],
                        "cohort_interaction_p": result["cohort_interaction_p"],
                        "degrees_freedom": result["degrees_freedom"],
                        "cohort_levels": ";".join(result["cohort_levels"]),
                        "design_rank": result["design_rank"],
                        "design_columns": result["design_columns"],
                        "design_full_rank": result["design_full_rank"],
                    }
                )
            )
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def build_vma_summary(
    vma_metrics: dict[str, np.ndarray],
    labels: np.ndarray,
    class_names: Sequence[str],
    gene_names: Sequence[str],
    degrees: np.ndarray,
    expected_key_mass: np.ndarray,
) -> pd.DataFrame:
    n_samples, n_heads, n_genes = next(iter(vma_metrics.values())).shape
    class_groups: list[tuple[str, np.ndarray]] = [("ALL", np.ones(n_samples, dtype=bool))]
    class_groups.extend(
        (str(class_name), labels == class_index)
        for class_index, class_name in enumerate(class_names)
    )
    frames: list[pd.DataFrame] = []
    for class_name, keep in class_groups:
        if not bool(keep.any()):
            continue
        for head_label, head_values in [
            *[(str(head), {name: values[keep, head] for name, values in vma_metrics.items()}) for head in range(n_heads)],
            (
                "mean_heads",
                {name: nanmean_silent(values[keep], axis=1) for name, values in vma_metrics.items()},
            ),
        ]:
            payload: dict[str, Any] = {
                "class_name": class_name,
                "head": head_label,
                "sample_count": int(keep.sum()),
                "gene_index": np.arange(n_genes, dtype=np.int64),
                "gene": np.asarray(gene_names, dtype=str),
                "ppi_degree": degrees,
                "expected_key_mass_under_uniform_neighbors": expected_key_mass,
            }
            for name, values in head_values.items():
                payload[f"{name}_mean"] = nanmean_silent(values, axis=0)
                payload[f"{name}_std"] = nanstd_silent(values, axis=0)
            frame = pd.DataFrame(payload)
            frame["key_rank"] = _rank_desc(
                frame["key_degree_corrected_enrichment_mean"].to_numpy()
            )
            frame["query_rank"] = _rank_desc(
                frame["query_specificity_mean"].to_numpy()
            )
            frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def build_vma_edges(
    class_edge_means: dict[str, np.ndarray],
    edge_index: np.ndarray,
    class_names: Sequence[str],
    class_counts: np.ndarray,
    gene_names: Sequence[str],
    top_edges: int = 250,
) -> pd.DataFrame:
    weights = class_edge_means["attention_weights"]
    volumes = class_edge_means["volumes"]
    logits = class_edge_means["logits"]
    target, source = edge_index
    target_degree = np.bincount(target, minlength=len(gene_names)).astype(np.int64)
    edge_target_degree = target_degree[target]
    informative_edges = edge_target_degree >= 2
    class_counts = np.asarray(class_counts, dtype=np.float64)
    if len(class_counts) != len(class_names) or len(class_counts) != weights.shape[0]:
        raise ValueError("Class counts are not aligned with the VMA class-edge means.")
    if float(class_counts.sum()) <= 0:
        raise ValueError("At least one class sample is required to aggregate VMA edges.")
    normalized_counts = class_counts / class_counts.sum()
    frames: list[pd.DataFrame] = []
    groups: Iterable[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = [
        (
            "ALL",
            np.tensordot(normalized_counts, weights, axes=(0, 0)),
            np.tensordot(normalized_counts, volumes, axes=(0, 0)),
            np.tensordot(normalized_counts, logits, axes=(0, 0)),
        )
    ]
    groups = [
        *groups,
        *[
            (str(name), weights[index], volumes[index], logits[index])
            for index, name in enumerate(class_names)
        ],
    ]
    for class_name, group_weights, group_volumes, group_logits in groups:
        for head in range(group_weights.shape[0]):
            degree_corrected = group_weights[head] * edge_target_degree
            valid_offsets = np.flatnonzero(
                informative_edges & np.isfinite(degree_corrected)
            )
            order = valid_offsets[
                np.argsort(degree_corrected[valid_offsets], kind="mergesort")[::-1]
            ][: min(top_edges, len(valid_offsets))]
            if not len(order):
                continue
            frames.append(
                pd.DataFrame(
                    {
                        "class_name": class_name,
                        "head": head,
                        "edge_offset": order,
                        "target_index": target[order],
                        "source_index": source[order],
                        "target_query_gene": [gene_names[index] for index in target[order]],
                        "source_key_gene": [gene_names[index] for index in source[order]],
                        "target_ppi_degree": edge_target_degree[order],
                        "uniform_neighbor_attention": 1.0 / edge_target_degree[order],
                        "attention_mean": group_weights[head, order],
                        "degree_corrected_attention_enrichment": degree_corrected[order],
                        "log2_degree_corrected_attention_enrichment": np.log2(
                            np.maximum(degree_corrected[order], 1e-12)
                        ),
                        "volume_mean": group_volumes[head, order],
                        "logit_mean": group_logits[head, order],
                    }
                )
            )
    if not frames:
        return pd.DataFrame(
            columns=[
                "class_name",
                "head",
                "edge_offset",
                "target_index",
                "source_index",
                "target_query_gene",
                "source_key_gene",
                "target_ppi_degree",
                "uniform_neighbor_attention",
                "attention_mean",
                "degree_corrected_attention_enrichment",
                "log2_degree_corrected_attention_enrichment",
                "volume_mean",
                "logit_mean",
            ]
        )
    return pd.concat(frames, ignore_index=True)


def build_dense_top_edges(
    class_mean: np.ndarray,
    overall_mean: np.ndarray,
    class_names: Sequence[str],
    gene_names: Sequence[str],
    top_edges: int = 250,
) -> pd.DataFrame:
    groups: list[tuple[str, np.ndarray]] = [("ALL", overall_mean)]
    groups.extend(
        (str(class_name), class_mean[class_index])
        for class_index, class_name in enumerate(class_names)
    )
    frames: list[pd.DataFrame] = []
    n_genes = len(gene_names)
    for class_name, head_matrices in groups:
        matrices = [*head_matrices, head_matrices.mean(axis=0)]
        head_names = [*[str(head) for head in range(head_matrices.shape[0])], "mean_heads"]
        for head_name, matrix in zip(head_names, matrices):
            flat = matrix.reshape(-1)
            count = min(top_edges, flat.size)
            offsets = np.argpartition(flat, -count)[-count:]
            offsets = offsets[np.argsort(flat[offsets], kind="mergesort")[::-1]]
            query = offsets // n_genes
            key = offsets % n_genes
            frames.append(
                pd.DataFrame(
                    {
                        "class_name": class_name,
                        "head": head_name,
                        "edge_offset": offsets,
                        "query_index": query,
                        "key_index": key,
                        "query_gene": [gene_names[index] for index in query],
                        "key_gene": [gene_names[index] for index in key],
                        "attention_mean": flat[offsets],
                        "attention_enrichment_over_uniform": n_genes * flat[offsets],
                        "is_self_attention": query == key,
                    }
                )
            )
    return pd.concat(frames, ignore_index=True)


def bootstrap_mean_interval(
    values: np.ndarray,
    iterations: int,
    seed: int,
    transform: Any | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    if iterations <= 0:
        mean = np.nanmean(values, axis=0)
        if transform is not None:
            mean = transform(mean)
        return mean, mean
    rng = np.random.default_rng(seed)
    bootstrap = np.empty((iterations, values.shape[1]), dtype=np.float64)
    for iteration in range(iterations):
        indices = rng.integers(0, values.shape[0], size=values.shape[0])
        bootstrap[iteration] = np.nanmean(values[indices], axis=0)
    if transform is not None:
        bootstrap = transform(bootstrap)
    return np.nanpercentile(bootstrap, 2.5, axis=0), np.nanpercentile(
        bootstrap, 97.5, axis=0
    )


def _save_figure(fig: plt.Figure, output_stem: Path) -> None:
    fig.savefig(output_stem.with_suffix(".png"), dpi=240, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def remove_stale_vma_outputs(output_dir: Path, plots_dir: Path) -> None:
    """Remove only files owned by this script when VMA export is disabled."""

    for filename in (
        "vma_gene_attention_summary.csv",
        "vma_gene_attention_ranking.csv",
        "vma_top_attention_edges.csv",
        "vma_attention_metrics.npz",
    ):
        (output_dir / filename).unlink(missing_ok=True)
    for stem in ("vma_key_gene_ranking", "vma_query_gene_ranking"):
        for suffix in (".png", ".pdf"):
            (plots_dir / f"{stem}{suffix}").unlink(missing_ok=True)


def plot_full_attention(
    overall_mean: np.ndarray,
    n_genes: int,
    output_dir: Path,
) -> None:
    matrices = [*overall_mean, overall_mean.mean(axis=0)]
    titles = [*[f"Head {head}" for head in range(overall_mean.shape[0])], "Mean heads"]
    transformed = [np.log2(np.maximum(matrix * n_genes, 1e-12)) for matrix in matrices]
    limit = max(float(np.nanpercentile(np.abs(matrix), 99.5)) for matrix in transformed)
    limit = max(limit, 1e-6)
    fig, axes = plt.subplots(1, len(matrices), figsize=(6.0 * len(matrices), 5.5))
    axes = np.atleast_1d(axes)
    image = None
    for ax, matrix, title in zip(axes, transformed, titles):
        image = ax.imshow(
            matrix,
            cmap="coolwarm",
            vmin=-limit,
            vmax=limit,
            interpolation="nearest",
            aspect="auto",
            rasterized=True,
        )
        ax.set_title(title)
        ax.set_xlabel("Key gene index")
        ax.set_ylabel("Query gene index")
    assert image is not None
    fig.colorbar(image, ax=axes.tolist(), shrink=0.82, label="log2 attention / uniform")
    fig.suptitle("Post-training dense TxT attention — full selected-gene matrix")
    _save_figure(fig, output_dir / "dense_attention_full_matrix")


def select_matrix_genes(ranking: pd.DataFrame, count: int) -> np.ndarray:
    count = min(count, len(ranking))
    key_count = (count + 1) // 2
    query_count = count - key_count
    chosen: list[int] = []
    seen: set[int] = set()
    for column, quota in [("key_rank", key_count), ("query_rank", query_count)]:
        added = 0
        for index in ranking.sort_values(column)["gene_index"].astype(int):
            if index in seen:
                continue
            chosen.append(index)
            seen.add(index)
            added += 1
            if added >= quota:
                break
    if len(chosen) < count:
        for index in ranking.sort_values(["key_rank", "query_rank"])["gene_index"].astype(int):
            if index not in seen:
                chosen.append(index)
                seen.add(index)
            if len(chosen) >= count:
                break
    return np.asarray(chosen[:count], dtype=np.int64)


def plot_top_submatrix(
    overall_mean: np.ndarray,
    class_mean: np.ndarray,
    class_names: Sequence[str],
    ranking: pd.DataFrame,
    gene_names: Sequence[str],
    matrix_gene_count: int,
    output_dir: Path,
) -> None:
    indices = select_matrix_genes(ranking, matrix_gene_count)
    labels = [gene_names[index] for index in indices]
    n_genes = len(gene_names)
    overall = overall_mean.mean(axis=0)[np.ix_(indices, indices)]
    overall_log = np.log2(np.maximum(overall * n_genes, 1e-12))
    class_index = {str(name): idx for idx, name in enumerate(class_names)}
    difference = None
    if "AD" in class_index and "MCI" in class_index:
        ad = class_mean[class_index["AD"]].mean(axis=0)[np.ix_(indices, indices)]
        mci = class_mean[class_index["MCI"]].mean(axis=0)[np.ix_(indices, indices)]
        difference = n_genes * (ad - mci)
    matrices = [overall_log] + ([difference] if difference is not None else [])
    titles = ["All test subjects: log2 attention / uniform"]
    if difference is not None:
        titles.append("AD − MCI attention enrichment")
    fig, axes = plt.subplots(1, len(matrices), figsize=(12.5 * len(matrices), 11.5))
    axes = np.atleast_1d(axes)
    for ax, matrix, title in zip(axes, matrices, titles):
        limit = max(float(np.nanpercentile(np.abs(matrix), 99.0)), 1e-8)
        image = ax.imshow(matrix, cmap="coolwarm", vmin=-limit, vmax=limit, aspect="auto")
        ax.set_title(title)
        ax.set_xticks(np.arange(len(labels)), labels=labels, rotation=90, fontsize=7)
        ax.set_yticks(np.arange(len(labels)), labels=labels, fontsize=7)
        ax.set_xlabel("Key gene")
        ax.set_ylabel("Query gene")
        fig.colorbar(image, ax=ax, shrink=0.78)
    fig.suptitle("Dense TxT attention among top key/query genes")
    _save_figure(fig, output_dir / "dense_attention_top_gene_submatrix")


def plot_ranked_metric(
    values: np.ndarray,
    ranking_values: np.ndarray,
    gene_names: Sequence[str],
    *,
    top_count: int,
    iterations: int,
    seed: int,
    null_value: float | None,
    xlabel: str,
    title: str,
    output_stem: Path,
    transform: Any | None = None,
) -> None:
    safe_ranking = np.where(np.isfinite(ranking_values), ranking_values, -np.inf)
    order = np.argsort(safe_ranking, kind="mergesort")[::-1][:top_count]
    selected = values[:, order]
    means = np.nanmean(selected, axis=0)
    if transform is not None:
        means = transform(means)
    lower, upper = bootstrap_mean_interval(
        selected,
        iterations,
        seed,
        transform=transform,
    )
    display_order = np.argsort(means)
    means = means[display_order]
    lower = lower[display_order]
    upper = upper[display_order]
    ordered_genes = np.asarray(gene_names)[order][display_order]
    fig, ax = plt.subplots(figsize=(9.0, max(6.0, 0.32 * len(order))))
    y = np.arange(len(order))
    ax.errorbar(
        means,
        y,
        xerr=np.vstack([means - lower, upper - means]),
        fmt="o",
        color="#2f5597",
        ecolor="#9aa8bd",
        elinewidth=1.1,
        capsize=2,
    )
    if null_value is not None:
        ax.axvline(null_value, color="#7f7f7f", linestyle="--", linewidth=1.0)
    ax.set_yticks(y, ordered_genes)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(axis="x", color="#dddddd", linewidth=0.6)
    _save_figure(fig, output_stem)


def plot_class_profiles(
    summary: pd.DataFrame,
    ranking: pd.DataFrame,
    class_names: Sequence[str],
    top_count: int,
    output_dir: Path,
) -> None:
    top = ranking.nsmallest(top_count, "key_rank")["gene"].tolist()
    class_frame = summary[
        (summary["head"] == "mean_heads") & summary["class_name"].isin(class_names)
    ]
    incoming = class_frame.pivot(index="gene", columns="class_name", values="incoming_log2_enrichment")
    query = class_frame.pivot(index="gene", columns="class_name", values="query_specificity_mean")
    incoming = incoming.reindex(index=top, columns=list(class_names))
    query = query.reindex(index=top, columns=list(class_names))
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11.5, max(7.0, 0.34 * len(top))),
        constrained_layout=True,
    )
    for ax, matrix, title, cmap in [
        (axes[0], incoming.to_numpy(), "Key: log2 incoming enrichment", "coolwarm"),
        (axes[1], query.to_numpy(), "Query: row specificity", "viridis"),
    ]:
        if cmap == "coolwarm":
            limit = max(float(np.nanmax(np.abs(matrix))), 1e-8)
            image = ax.imshow(matrix, cmap=cmap, vmin=-limit, vmax=limit, aspect="auto")
        else:
            image = ax.imshow(matrix, cmap=cmap, aspect="auto")
        ax.set_xticks(np.arange(len(class_names)), class_names, rotation=30, ha="right")
        ax.set_yticks(np.arange(len(top)), top, fontsize=8)
        ax.set_title(title)
        fig.colorbar(image, ax=ax, shrink=0.78)
    fig.suptitle("Class profiles for top dense-attention key genes")
    _save_figure(fig, output_dir / "dense_attention_class_profiles")


def plot_ad_mci_volcano(
    contrasts: pd.DataFrame,
    output_dir: Path,
) -> None:
    selected = contrasts[
        (contrasts["contrast"] == "AD_minus_MCI")
        & (contrasts["metric"] == "incoming_enrichment")
    ].copy()
    if selected.empty:
        return
    selected["minus_log10_q"] = -np.log10(selected["q_value"].clip(lower=1e-300))
    fig, ax = plt.subplots(figsize=(9.0, 6.5))
    significant = selected["q_value"] < 0.05
    ax.scatter(
        selected.loc[~significant, "cohort_adjusted_beta"],
        selected.loc[~significant, "minus_log10_q"],
        s=12,
        alpha=0.45,
        color="#8a8a8a",
        linewidths=0,
    )
    ax.scatter(
        selected.loc[significant, "cohort_adjusted_beta"],
        selected.loc[significant, "minus_log10_q"],
        s=19,
        alpha=0.8,
        color="#b23a48",
        linewidths=0,
    )
    labels = selected.sort_values(
        ["q_value", "cohort_adjusted_beta"], ascending=[True, False]
    ).head(12)
    for _, row in labels.iterrows():
        ax.annotate(
            row["gene"],
            (row["cohort_adjusted_beta"], row["minus_log10_q"]),
            fontsize=8,
            xytext=(3, 3),
            textcoords="offset points",
        )
    ax.axhline(-math.log10(0.05), color="#777777", linestyle="--", linewidth=1)
    ax.axvline(0.0, color="#777777", linewidth=0.8)
    ax.set_xlabel("AD − MCI cohort-adjusted incoming enrichment")
    ax.set_ylabel("−log10 BH q-value")
    ax.set_title("Dense-attention key contrast on held-out subjects")
    _save_figure(fig, output_dir / "dense_attention_ad_mci_key_contrast")


def plot_qk_relationships(
    ranking: pd.DataFrame,
    output_dir: Path,
) -> dict[str, float]:
    pairs = [
        ("k_norm_mean", "incoming_enrichment_mean", "K norm", "Key incoming enrichment"),
        ("q_norm_mean", "query_specificity_mean", "Q norm", "Query specificity"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.7))
    correlations: dict[str, float] = {}
    for ax, (x_col, y_col, x_label, y_label) in zip(axes, pairs):
        x = ranking[x_col].to_numpy(dtype=float)
        y = ranking[y_col].to_numpy(dtype=float)
        finite = np.isfinite(x) & np.isfinite(y)
        rho, p_value = stats.spearmanr(x[finite], y[finite])
        correlations[f"{x_col}_vs_{y_col}_spearman_rho"] = float(rho)
        correlations[f"{x_col}_vs_{y_col}_spearman_p"] = float(p_value)
        ax.scatter(x[finite], y[finite], s=11, alpha=0.38, color="#4472c4", linewidths=0)
        annotate = ranking.loc[finite].nlargest(8, y_col)
        for _, row in annotate.iterrows():
            ax.annotate(
                row["gene"],
                (row[x_col], row[y_col]),
                fontsize=7,
                xytext=(3, 3),
                textcoords="offset points",
            )
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(f"Spearman rho={rho:.3f}, p={p_value:.2g}")
        ax.grid(color="#e1e1e1", linewidth=0.5)
    fig.suptitle("Projected Q/K geometry versus attention-derived metrics")
    _save_figure(fig, output_dir / "dense_attention_qk_relationships")
    return correlations


def plot_expression_relationships(
    ranking: pd.DataFrame,
    output_dir: Path,
) -> dict[str, float]:
    pairs = [
        ("expression_mean", "incoming_enrichment_mean", "Mean scaled expression", "Key incoming enrichment"),
        ("expression_std", "incoming_enrichment_mean", "Scaled-expression SD", "Key incoming enrichment"),
        ("expression_mean", "query_specificity_mean", "Mean scaled expression", "Query specificity"),
        ("expression_std", "query_specificity_mean", "Scaled-expression SD", "Query specificity"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 10.0))
    correlations: dict[str, float] = {}
    for ax, (x_col, y_col, x_label, y_label) in zip(axes.flat, pairs):
        x = ranking[x_col].to_numpy(dtype=float)
        y = ranking[y_col].to_numpy(dtype=float)
        finite = np.isfinite(x) & np.isfinite(y)
        rho, p_value = stats.spearmanr(x[finite], y[finite])
        key = f"{x_col}_vs_{y_col}"
        correlations[f"{key}_spearman_rho"] = float(rho)
        correlations[f"{key}_spearman_p"] = float(p_value)
        p_label = "<1e-300" if p_value == 0 else f"{p_value:.2g}"
        ax.scatter(x[finite], y[finite], s=11, alpha=0.35, color="#5b9bd5", linewidths=0)
        annotate = ranking.loc[finite].nlargest(3, y_col)
        for annotation_index, (_, row) in enumerate(annotate.iterrows()):
            ax.annotate(
                row["gene"],
                (row[x_col], row[y_col]),
                fontsize=7,
                xytext=(3, 3 + 8 * annotation_index),
                textcoords="offset points",
            )
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(f"Spearman rho={rho:.3f}, p={p_label}")
        ax.grid(color="#e1e1e1", linewidth=0.5)
    fig.suptitle("Attention-derived metrics versus held-out expression distribution")
    _save_figure(fig, output_dir / "dense_attention_expression_relationships")
    return correlations


def plot_vma_rankings(
    vma_metrics: dict[str, np.ndarray],
    vma_summary: pd.DataFrame,
    gene_names: Sequence[str],
    top_count: int,
    iterations: int,
    seed: int,
    output_dir: Path,
) -> None:
    ranking = vma_summary[
        (vma_summary["class_name"] == "ALL")
        & (vma_summary["head"] == "mean_heads")
    ].sort_values("gene_index")
    key_values = nanmean_silent(vma_metrics["key_degree_corrected_enrichment"], axis=1)
    plot_ranked_metric(
        key_values,
        ranking["key_degree_corrected_enrichment_mean"].to_numpy(),
        gene_names,
        top_count=top_count,
        iterations=iterations,
        seed=seed + 17,
        null_value=1.0,
        xlabel="Degree-corrected incoming VMA enrichment (uniform-neighbour null = 1)",
        title="Top sparse VMA key/source genes",
        output_stem=output_dir / "vma_key_gene_ranking",
    )
    query_values = nanmean_silent(vma_metrics["query_specificity"], axis=1)
    plot_ranked_metric(
        query_values,
        ranking["query_specificity_mean"].to_numpy(),
        gene_names,
        top_count=top_count,
        iterations=iterations,
        seed=seed + 19,
        null_value=0.0,
        xlabel="Sparse VMA query specificity (1 − normalized neighbour entropy)",
        title="Top sparse VMA query/target genes (degree >= 2)",
        output_stem=output_dir / "vma_query_gene_ranking",
    )


def write_markdown_summary(
    output_path: Path,
    *,
    run_dir: Path,
    checkpoint: Path,
    split: str,
    sample_count: int,
    class_names: Sequence[str],
    class_counts: np.ndarray,
    ranking: pd.DataFrame,
    vma_ranking: pd.DataFrame | None,
    caveat: str,
) -> None:
    def markdown_table(frame: pd.DataFrame, columns: Sequence[str], count: int = 15) -> str:
        selected = frame.loc[:, list(columns)].head(count).copy()
        for column in selected.select_dtypes(include=[np.number]).columns:
            selected[column] = selected[column].map(lambda value: f"{value:.6g}")
        header = "| " + " | ".join(selected.columns) + " |"
        separator = "| " + " | ".join(["---"] * len(selected.columns)) + " |"
        rows = [
            "| " + " | ".join(map(str, row)) + " |"
            for row in selected.itertuples(index=False, name=None)
        ]
        return "\n".join([header, separator, *rows])

    key_top = ranking.sort_values("key_rank")
    query_top = ranking.sort_values("query_rank")
    lines = [
        "# Post-training TxT attention analysis",
        "",
        f"- Run: `{run_dir}`",
        f"- Checkpoint: `{checkpoint}`",
        f"- Split: `{split}` ({sample_count} subjects)",
        "- Classes: "
        + ", ".join(
            f"{name}={int(class_counts[index])}" for index, name in enumerate(class_names)
        ),
        "",
        "## Dense attention: top key genes",
        "",
        markdown_table(
            key_top,
            ["gene", "incoming_enrichment_mean", "incoming_log2_enrichment", "key_rank"],
        ),
        "",
        "## Dense attention: top selective query genes",
        "",
        markdown_table(
            query_top,
            ["gene", "query_specificity_mean", "query_topk_mass_mean", "query_rank"],
        ),
    ]
    if vma_ranking is not None:
        lines.extend(
            [
                "",
                "## Sparse VMA: top degree-corrected key/source genes",
                "",
                markdown_table(
                    vma_ranking.sort_values("key_rank"),
                    [
                        "gene",
                        "ppi_degree",
                        "key_degree_corrected_enrichment_mean",
                        "key_rank",
                    ],
                ),
                "",
                "## Sparse VMA: top selective query/target genes",
                "",
                markdown_table(
                    vma_ranking.sort_values("query_rank"),
                    ["gene", "ppi_degree", "query_specificity_mean", "query_rank"],
                ),
            ]
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            caveat,
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    cli = build_parser().parse_args(argv)
    if cli.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if cli.max_samples is not None and cli.max_samples <= 0:
        raise ValueError("--max-samples must be positive when provided.")
    if cli.top_genes <= 0 or cli.matrix_top_genes <= 0:
        raise ValueError("--top-genes and --matrix-top-genes must be positive.")
    if cli.bootstrap_iterations < 0:
        raise ValueError("--bootstrap-iterations cannot be negative.")

    run_dir = resolve_path(cli.run_dir)
    checkpoint = (
        _resolve_override(cli.checkpoint)
        if cli.checkpoint is not None
        else run_dir / "best_model.pt"
    )
    assert checkpoint is not None
    output_dir = (
        _resolve_override(cli.output_dir)
        if cli.output_dir is not None
        else run_dir / "attention_analysis" / cli.split
    )
    assert output_dir is not None
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(cli.device)

    args, dataset, _, model = reconstruct_model(
        run_dir,
        checkpoint,
        device,
        x_file=_resolve_override(cli.x_file),
        y_file=_resolve_override(cli.y_file),
        split_file=_resolve_override(cli.split_file),
        ppi_edge_file=_resolve_override(cli.ppi_edge_file),
    )
    gene_x, labels, sample_ids = split_arrays(dataset, cli.split)
    if cli.max_samples is not None:
        selected = np.linspace(
            0,
            len(sample_ids) - 1,
            num=min(cli.max_samples, len(sample_ids)),
            dtype=np.int64,
        )
        gene_x = gene_x[selected]
        labels = labels[selected]
        sample_ids = sample_ids[selected]
    cohorts = load_cohorts(resolve_path(cli.clinical_file), sample_ids)
    collection = collect_attention(
        model,
        gene_x,
        labels,
        class_count=len(dataset.class_names),
        batch_size=cli.batch_size,
        device=device,
        query_top_k=cli.query_top_k,
        include_vma=not cli.skip_vma,
    )
    if "vma" not in collection:
        remove_stale_vma_outputs(output_dir, plots_dir)
    metrics = collection["metrics"]
    summary = build_gene_summary(
        metrics,
        gene_x,
        labels,
        dataset.class_names,
        dataset.gene_names,
    )
    summary.to_csv(output_dir / "gene_attention_summary.csv", index=False)
    ranking = summary[
        (summary["class_name"] == "ALL") & (summary["head"] == "mean_heads")
    ].copy()
    ranking = ranking.sort_values("key_rank").reset_index(drop=True)
    ranking.to_csv(output_dir / "gene_attention_ranking.csv", index=False)
    contrasts = build_contrasts(
        metrics,
        labels,
        cohorts,
        dataset.class_names,
        dataset.gene_names,
    )
    contrasts.to_csv(output_dir / "gene_attention_contrasts.csv", index=False)
    collection["qc"].to_csv(output_dir / "attention_qc.csv", index=False)
    dense_edges = build_dense_top_edges(
        collection["class_mean_attention"],
        collection["overall_mean_attention"],
        dataset.class_names,
        dataset.gene_names,
    )
    dense_edges.to_csv(output_dir / "dense_top_attention_edges.csv", index=False)

    np.savez_compressed(
        output_dir / "sample_gene_attention_metrics.npz",
        **{name: values.astype(np.float32) for name, values in metrics.items()},
        sample_ids=sample_ids,
        source_labels=labels,
        class_names=np.asarray(dataset.class_names, dtype=str),
        cohorts=cohorts,
        gene_names=np.asarray(dataset.gene_names, dtype=str),
    )
    np.savez_compressed(
        output_dir / "class_mean_attention.npz",
        class_mean_attention=collection["class_mean_attention"].astype(np.float32),
        overall_mean_attention=collection["overall_mean_attention"].astype(np.float32),
        class_counts=collection["class_counts"],
        class_names=np.asarray(dataset.class_names, dtype=str),
        gene_names=np.asarray(dataset.gene_names, dtype=str),
        axis_semantics=np.asarray("class,head,query_gene,key_gene"),
    )

    overall_metric_values = {
        name: np.nanmean(values, axis=1) for name, values in metrics.items()
    }
    plot_full_attention(
        collection["overall_mean_attention"], len(dataset.gene_names), plots_dir
    )
    plot_top_submatrix(
        collection["overall_mean_attention"],
        collection["class_mean_attention"],
        dataset.class_names,
        ranking,
        dataset.gene_names,
        cli.matrix_top_genes,
        plots_dir,
    )
    plot_ranked_metric(
        overall_metric_values["incoming_enrichment"],
        ranking.sort_values("gene_index")["incoming_enrichment_mean"].to_numpy(),
        dataset.gene_names,
        top_count=min(cli.top_genes, len(dataset.gene_names)),
        iterations=cli.bootstrap_iterations,
        seed=int(args.seed),
        null_value=0.0,
        xlabel="log2 incoming attention / uniform (bootstrap 95% CI)",
        title="Top dense-attention key genes",
        output_stem=plots_dir / "dense_attention_key_gene_ranking",
        transform=lambda values: np.log2(np.maximum(values, 1e-12)),
    )
    plot_ranked_metric(
        overall_metric_values["query_specificity"],
        ranking.sort_values("gene_index")["query_specificity_mean"].to_numpy(),
        dataset.gene_names,
        top_count=min(cli.top_genes, len(dataset.gene_names)),
        iterations=cli.bootstrap_iterations,
        seed=int(args.seed) + 1,
        null_value=0.0,
        xlabel="Query specificity = 1 − normalized row entropy (bootstrap 95% CI)",
        title="Top selective dense-attention query genes",
        output_stem=plots_dir / "dense_attention_query_gene_ranking",
    )
    plot_class_profiles(
        summary,
        ranking,
        dataset.class_names,
        min(cli.top_genes, len(dataset.gene_names)),
        plots_dir,
    )
    plot_ad_mci_volcano(contrasts, plots_dir)
    correlations = plot_qk_relationships(ranking, plots_dir)
    correlations.update(plot_expression_relationships(ranking, plots_dir))

    vma_summary: pd.DataFrame | None = None
    vma_ranking: pd.DataFrame | None = None
    if "vma" in collection:
        vma_result = collection["vma"]
        vma_summary = build_vma_summary(
            vma_result["metrics"],
            labels,
            dataset.class_names,
            dataset.gene_names,
            vma_result["degrees"],
            vma_result["expected_key_mass"],
        )
        vma_summary.to_csv(output_dir / "vma_gene_attention_summary.csv", index=False)
        vma_ranking = vma_summary[
            (vma_summary["class_name"] == "ALL")
            & (vma_summary["head"] == "mean_heads")
        ].copy()
        vma_ranking.sort_values("key_rank").to_csv(
            output_dir / "vma_gene_attention_ranking.csv", index=False
        )
        vma_edges = build_vma_edges(
            vma_result["class_edge_means"],
            vma_result["edge_index"],
            dataset.class_names,
            collection["class_counts"],
            dataset.gene_names,
        )
        vma_edges.to_csv(output_dir / "vma_top_attention_edges.csv", index=False)
        np.savez_compressed(
            output_dir / "vma_attention_metrics.npz",
            **{
                name: values.astype(np.float32)
                for name, values in vma_result["metrics"].items()
            },
            **{
                f"class_mean_{name}": values.astype(np.float32)
                for name, values in vma_result["class_edge_means"].items()
            },
            edge_index=vma_result["edge_index"],
            expected_key_mass=vma_result["expected_key_mass"],
            degrees=vma_result["degrees"],
            sample_ids=sample_ids,
            source_labels=labels,
            class_names=np.asarray(dataset.class_names, dtype=str),
            gene_names=np.asarray(dataset.gene_names, dtype=str),
            edge_semantics=np.asarray("row0_target_query__row1_source_key"),
        )
        plot_vma_rankings(
            vma_result["metrics"],
            vma_summary,
            dataset.gene_names,
            min(cli.top_genes, len(dataset.gene_names)),
            cli.bootstrap_iterations,
            int(args.seed),
            plots_dir,
        )

    caveat = (
        "These are attention-derived rankings, not causal feature-importance estimates. "
        "A high weight does not by itself establish predictive necessity, biological causality, "
        "or biomarker validity. The final thesis claim requires stability across the complete "
        "set of trained seeds/checkpoints and perturbation-based validation."
    )
    write_markdown_summary(
        output_dir / "SUMMARY.md",
        run_dir=run_dir,
        checkpoint=checkpoint,
        split=cli.split,
        sample_count=len(sample_ids),
        class_names=dataset.class_names,
        class_counts=collection["class_counts"],
        ranking=ranking,
        vma_ranking=vma_ranking,
        caveat=caveat,
    )

    checkpoint_summary = worker.load_source_model_summary(checkpoint)
    source_metadata = worker.source_checkpoint_member_metadata(
        checkpoint, checkpoint_summary
    )
    output_manifest = {
        "summary": str(output_dir / "SUMMARY.md"),
        "gene_ranking": str(output_dir / "gene_attention_ranking.csv"),
        "gene_summary": str(output_dir / "gene_attention_summary.csv"),
        "contrasts": str(output_dir / "gene_attention_contrasts.csv"),
        "top_dense_edges": str(output_dir / "dense_top_attention_edges.csv"),
        "class_mean_matrices": str(output_dir / "class_mean_attention.npz"),
        "sample_metrics": str(output_dir / "sample_gene_attention_metrics.npz"),
        "plots": str(plots_dir),
    }
    if "vma" in collection:
        output_manifest.update(
            {
                "vma_gene_ranking": str(
                    output_dir / "vma_gene_attention_ranking.csv"
                ),
                "vma_top_edges": str(output_dir / "vma_top_attention_edges.csv"),
                "vma_metrics": str(output_dir / "vma_attention_metrics.npz"),
            }
        )
    manifest = {
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "analysis_script_sha256": sha256_file(Path(__file__).resolve()),
        "source_checkpoint_metadata": source_metadata,
        "model_variant": getattr(args, "model_variant", "baseline"),
        "seed": int(args.seed),
        "split": cli.split,
        "samples": int(len(sample_ids)),
        "class_counts": {
            str(name): int(collection["class_counts"][index])
            for index, name in enumerate(dataset.class_names)
        },
        "cohort_counts": {
            str(name): int(count)
            for name, count in zip(*np.unique(cohorts, return_counts=True))
        },
        "genes": int(len(dataset.gene_names)),
        "heads": int(metrics["incoming_enrichment"].shape[1]),
        "layers": 1,
        "axis_semantics": "dense_attention[head, query_gene, key_gene]",
        "key_metric": "G * mean_over_queries(attention); uniform reference = 1",
        "query_metric": "1 - row_entropy/log(G); row sum is excluded because it is always 1",
        "query_top_k": int(cli.query_top_k),
        "bootstrap_iterations": int(cli.bootstrap_iterations),
        "cohort_adjustment": "OLS class coefficient adjusted for dataset_gse; class-by-cohort interaction exported",
        "dense_attention_dropout": "disabled by model.eval() and post-softmax reconstruction",
        "max_row_sum_abs_error": float(
            collection["qc"]["row_sum_max_abs_error"].max()
        ),
        "qk_correlations": correlations,
        "vma_included": "vma" in collection,
        "vma_key_metric": (
            "observed incoming source mass / expected incoming mass under uniform attention within each target neighbourhood"
            if "vma" in collection
            else None
        ),
        "interpretation_caveat": caveat,
        "single_seed_exploratory": True,
        "outputs": output_manifest,
    }
    write_json(output_dir / "attention_analysis_manifest.json", manifest)
    print(f"Attention analysis written to: {output_dir}")
    print(
        ranking.sort_values("key_rank")
        .loc[:, ["gene", "incoming_enrichment_mean", "key_rank"]]
        .head(15)
        .to_string(index=False)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
