#!/usr/bin/env python3
"""Extract only dense TxT key-side incoming attention from one checkpoint.

The dense tensor has shape ``[sample, head, query_gene, key_gene]``.  This
script intentionally does not compute or export query-side metrics and never
touches the optional sparse VMA branch.  For every held-out subject it stores

    G * mean_over_queries(attention)

so that 1 is the uniform-attention reference.  The primary per-seed ranking is
the equal-weight macro mean across biological classes after averaging heads.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.scripts.paper_comparison.analyze_txt_attention import (
    _encoder_attention_context,
    _rank_desc,
    reconstruct_model,
    sha256_file,
    split_arrays,
)
from experiments.scripts.paper_comparison.txt_volumetric.common import (
    resolve_path,
    write_json,
)
from source.pipeline.utils import resolve_device


PROTOCOL_FIELDS = (
    "model_variant",
    "n_layers",
    "n_heads",
    "d_model",
    "d_ff",
    "d_hidden1",
    "d_hidden2",
    "dropout",
    "batch_size",
    "max_genes",
    "gene_selection",
    "ad_mci_gene_fraction",
    "scaler",
    "feature_selection_fit_scope",
    "scaler_fit_scope",
    "train_sampling",
    "epochs",
    "early_stopping_patience",
    "val_loss_stop_threshold",
    "val_loss_stop_patience",
    "lr_encoder",
    "lr_head",
    "lr_embedding",
    "weight_decay",
    "task_loss_weights",
    "pooling_mode",
    "task_specific_pooling",
    "head_norm",
    "mask_aware_heads",
    "encoder_sharing",
    "gradient_strategy",
    "grad_clip_scope",
    "class_weighting",
    "expression_residual",
    "tupe_mode",
    "post_model_construction_reseed",
    "checkpoint_metric",
    "checkpoint_ensemble_size",
    "augmentation",
    "embedding_gene_policy",
    "embedding_rescale",
    "embedding_init_scale",
    "ppi_integration",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract dense TxT key-side incoming attention only; VMA and query metrics "
            "are intentionally ignored."
        )
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--x-file", type=Path, default=None)
    parser.add_argument("--y-file", type=Path, default=None)
    parser.add_argument("--split-file", type=Path, default=None)
    return parser


@torch.inference_mode()
def collect_key_attention(
    model: torch.nn.Module,
    gene_x: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, pd.DataFrame]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    n_samples, n_genes = gene_x.shape
    parts: list[np.ndarray] = []
    qc_rows: list[dict[str, Any]] = []
    for batch_number, start in enumerate(range(0, n_samples, batch_size), start=1):
        stop = min(start + batch_size, n_samples)
        batch_x = torch.as_tensor(gene_x[start:stop], dtype=torch.float32, device=device)
        _, attention_layer, q, k, _, _, _, tupe = _encoder_attention_context(
            model, batch_x
        )
        logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(
            2 * attention_layer.d_head
        ) + tupe
        attention = torch.softmax(logits, dim=-1)
        incoming = n_genes * attention.mean(dim=-2)
        if not torch.isfinite(attention).all() or not torch.isfinite(incoming).all():
            raise FloatingPointError(
                f"Non-finite attention detected in batch {batch_number}."
            )
        row_sum_error = float((attention.sum(dim=-1) - 1.0).abs().max().cpu())
        incoming_mean_error = float(
            (incoming.mean(dim=-1) - 1.0).abs().max().cpu()
        )
        if row_sum_error > 1e-4 or incoming_mean_error > 1e-4:
            raise RuntimeError(
                "Attention normalization QC failed in batch "
                f"{batch_number}: row error={row_sum_error:.3g}, "
                f"incoming error={incoming_mean_error:.3g}."
            )
        parts.append(incoming.detach().cpu().numpy().astype(np.float32))
        qc_rows.append(
            {
                "batch": batch_number,
                "start": start,
                "stop": stop,
                "samples": stop - start,
                "row_sum_max_abs_error": row_sum_error,
                "incoming_gene_mean_max_abs_error": incoming_mean_error,
                "attention_min": float(attention.min().cpu()),
                "attention_max": float(attention.max().cpu()),
                "attention_nonfinite": int((~torch.isfinite(attention)).sum().cpu()),
            }
        )
    return np.concatenate(parts, axis=0), pd.DataFrame(qc_rows)


def build_seed_ranking(
    incoming: np.ndarray,
    labels: np.ndarray,
    class_names: Sequence[str],
    gene_names: Sequence[str],
) -> tuple[pd.DataFrame, np.ndarray]:
    if incoming.ndim != 3:
        raise ValueError("Expected incoming[sample, head, gene].")
    if not np.isfinite(incoming).all():
        raise FloatingPointError("Incoming attention contains non-finite values.")
    n_samples, n_heads, n_genes = incoming.shape
    if n_samples != len(labels) or n_genes != len(gene_names):
        raise ValueError("Incoming attention is not aligned to labels/gene names.")
    present_classes = [index for index in range(len(class_names)) if (labels == index).any()]
    if len(present_classes) != len(class_names):
        raise ValueError("Every biological class must be represented in the selected split.")

    mean_heads = incoming.mean(axis=1, dtype=np.float64)
    class_means = np.stack(
        [mean_heads[labels == class_index].mean(axis=0) for class_index in present_classes]
    )
    macro_mean = class_means.mean(axis=0)
    pooled_mean = mean_heads.mean(axis=0)
    head_macro = np.stack(
        [
            np.stack(
                [
                    incoming[labels == class_index, head].mean(axis=0)
                    for class_index in present_classes
                ]
            ).mean(axis=0)
            for head in range(n_heads)
        ]
    )
    rank = _rank_desc(macro_mean)
    denominator = max(n_genes - 1, 1)
    payload: dict[str, Any] = {
        "gene_index": np.arange(n_genes, dtype=np.int64),
        "gene": np.asarray(gene_names, dtype=str),
        "incoming_macro_mean": macro_mean,
        "incoming_pooled_mean": pooled_mean,
        "incoming_log2_macro_mean": np.log2(np.maximum(macro_mean, 1e-12)),
        "enrichment_percent_macro_mean": 100.0 * (macro_mean - 1.0),
        "key_rank": rank,
        "key_percentile_score": 1.0 - (rank - 1.0) / denominator,
    }
    for class_index, class_name in enumerate(class_names):
        payload[f"incoming_class_{class_name}"] = class_means[class_index]
    for head in range(n_heads):
        payload[f"incoming_head_{head}_macro_mean"] = head_macro[head]
        payload[f"key_head_{head}_rank"] = _rank_desc(head_macro[head])
    return pd.DataFrame(payload).sort_values("key_rank"), class_means


def main(argv: Sequence[str] | None = None) -> int:
    cli = build_parser().parse_args(argv)
    if cli.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if cli.max_samples is not None and cli.max_samples <= 0:
        raise ValueError("--max-samples must be positive when provided.")

    run_dir = resolve_path(cli.run_dir)
    checkpoint = (
        resolve_path(cli.checkpoint) if cli.checkpoint is not None else run_dir / "best_model.pt"
    )
    output_dir = (
        resolve_path(cli.output_dir)
        if cli.output_dir is not None
        else run_dir / "key_attention" / cli.split
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(cli.device)
    if cli.device != "cpu" and device.type != cli.device:
        raise RuntimeError(
            f"Requested device {cli.device!r}, but PyTorch resolved {device!s}; "
            "refusing a silent CPU fallback."
        )
    args, dataset, _, model = reconstruct_model(
        run_dir,
        checkpoint,
        device,
        x_file=resolve_path(cli.x_file) if cli.x_file is not None else None,
        y_file=resolve_path(cli.y_file) if cli.y_file is not None else None,
        split_file=resolve_path(cli.split_file) if cli.split_file is not None else None,
    )
    if getattr(args, "model_variant", "baseline") != "baseline":
        raise ValueError("This key-only workflow accepts baseline TxT checkpoints only.")
    gene_x, labels, sample_ids = split_arrays(dataset, cli.split)
    full_split_samples = int(len(sample_ids))
    if cli.max_samples is not None:
        if cli.max_samples < len(dataset.class_names):
            raise ValueError(
                "--max-samples must be at least the number of biological classes."
            )
        class_indices = [
            np.flatnonzero(labels == class_index)
            for class_index in range(len(dataset.class_names))
        ]
        keep_values: list[int] = []
        for offset in range(max(map(len, class_indices))):
            for indices in class_indices:
                if offset < len(indices):
                    keep_values.append(int(indices[offset]))
                    if len(keep_values) == min(cli.max_samples, len(sample_ids)):
                        break
            if len(keep_values) == min(cli.max_samples, len(sample_ids)):
                break
        keep = np.asarray(keep_values, dtype=np.int64)
        gene_x, labels, sample_ids = gene_x[keep], labels[keep], sample_ids[keep]

    incoming, qc = collect_key_attention(
        model,
        gene_x,
        batch_size=cli.batch_size,
        device=device,
    )
    ranking, class_means = build_seed_ranking(
        incoming,
        labels,
        dataset.class_names,
        dataset.gene_names,
    )
    ranking.to_csv(output_dir / "key_attention_by_seed.csv", index=False)
    qc.to_csv(output_dir / "key_attention_qc.csv", index=False)
    np.savez_compressed(
        output_dir / "key_attention_by_subject.npz",
        incoming_enrichment=incoming,
        sample_ids=sample_ids,
        labels=labels,
        class_names=np.asarray(dataset.class_names, dtype=str),
        gene_names=np.asarray(dataset.gene_names, dtype=str),
        class_mean_incoming_enrichment=class_means.astype(np.float32),
        axis_semantics=np.asarray("sample,head,key_gene"),
    )
    gene_order_sha256 = hashlib.sha256(
        "\n".join(map(str, dataset.gene_names)).encode("utf-8")
    ).hexdigest()
    protocol = {name: getattr(args, name, None) for name in PROTOCOL_FIELDS}
    protocol_sha256 = hashlib.sha256(
        json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    artifact_paths = {
        "args": run_dir / "args.json",
        "x": Path(args.x_file),
        "y": Path(args.y_file),
        "split": Path(args.split_file),
        "selected_genes": run_dir / "selected_genes.csv",
        "constructor_embedding": run_dir / "gene_embedding.csv",
        "extractor": Path(__file__).resolve(),
    }
    manifest = {
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "artifact_sha256": {
            name: sha256_file(path) for name, path in artifact_paths.items()
        },
        "seed": int(args.seed),
        "model_variant": str(getattr(args, "model_variant", "baseline")),
        "protocol": protocol,
        "protocol_sha256": protocol_sha256,
        "split": cli.split,
        "full_split_samples": full_split_samples,
        "max_samples_requested": cli.max_samples,
        "is_full_split": int(len(sample_ids)) == full_split_samples,
        "device_requested": cli.device,
        "device_resolved": str(device),
        "samples": int(len(sample_ids)),
        "class_counts": {
            str(name): int((labels == index).sum())
            for index, name in enumerate(dataset.class_names)
        },
        "genes": int(len(dataset.gene_names)),
        "gene_order_sha256": gene_order_sha256,
        "heads": int(incoming.shape[1]),
        "key_metric": "G * mean_over_query(attention); uniform reference = 1",
        "seed_aggregation": "mean heads, then equal-weight macro mean across biological classes",
        "vma_included": False,
        "query_metrics_included": False,
        "max_row_sum_abs_error": float(qc["row_sum_max_abs_error"].max()),
        "max_incoming_gene_mean_abs_error": float(
            qc["incoming_gene_mean_max_abs_error"].max()
        ),
    }
    write_json(output_dir / "key_attention_manifest.json", manifest)
    print(f"Key-only attention written to: {output_dir}")
    print(
        ranking[
            ["gene", "incoming_macro_mean", "enrichment_percent_macro_mean", "key_rank"]
        ]
        .head(20)
        .to_string(index=False)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
