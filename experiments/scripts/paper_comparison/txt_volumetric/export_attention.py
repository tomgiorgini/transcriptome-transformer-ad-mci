#!/usr/bin/env python3
"""Post-hoc sparse VMA attention exporter.

No full dense TxT attention matrix is materialized or saved. The exporter
rebuilds the selected run, loads its checkpoint, enables the model's explicit
capture API, and records only HIPPIE-edge VMA quantities.
"""

from __future__ import annotations

import argparse
import math
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.scripts.paper_comparison import train_txt_multitask as worker  # noqa: E402
from experiments.scripts.paper_comparison.txt_volumetric.common import (  # noqa: E402
    ProtocolError,
    read_json,
    resolve_path,
    write_json,
)
from source.models.txt_volumetric import load_induced_ppi_graph  # noqa: E402
from source.pipeline.utils import resolve_device, set_seed  # noqa: E402


PATH_ARGUMENTS = {
    "x_file",
    "y_file",
    "split_file",
    "result_dir",
    "embed_file",
    "candidate_gene_file",
    "ppi_edge_file",
    "evaluation_only_checkpoint",
}
CAPTURE_FIELDS = ("attention_weights", "volumes", "logits")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Load a trained PPI-volumetric checkpoint and export sparse HIPPIE-edge VMA weights. "
            "The aggregate CSV is grouped by biological class, capture layer, head, and directed edge."
        )
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Candidate or selected-test run containing args.json, gene_embedding.csv, and best_model.pt.",
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--batch-size", type=int, default=9)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--aggregate-name", default="vma_attention_aggregate.csv")
    parser.add_argument(
        "--save-per-sample-npz",
        action="store_true",
        help="Also save sample x capture-layer x head x directed-edge arrays in compressed NPZ form.",
    )
    parser.add_argument("--per-sample-name", default="vma_attention_per_sample.npz")
    return parser


def namespace_from_saved_args(path: Path, run_dir: Path) -> Namespace:
    if not path.exists():
        raise FileNotFoundError(f"Missing worker arguments: {path}")
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise ProtocolError(f"Expected a JSON object in {path}.")
    payload.pop("resolved_split_file", None)
    payload.pop("device_resolved", None)
    # Runs produced before the normalized-volume experiment predate these
    # fields.  Their exact historical behavior is raw volume with VMA dropout
    # inherited from the backbone dropout.
    payload.setdefault("volumetric_volume_mode", "raw")
    payload.setdefault("volumetric_dropout", None)
    payload.setdefault("volumetric_message_mode", "legacy")
    payload.setdefault("volumetric_output_norm", "none")
    payload.setdefault("volumetric_gate_mode", "scalar")
    payload.setdefault("volumetric_backbone_gradient_mode", "coupled")
    payload.setdefault("tupe_mode", "on")
    for key in PATH_ARGUMENTS:
        value = payload.get(key)
        if value not in {None, ""}:
            payload[key] = Path(str(value))
    payload["result_dir"] = run_dir
    return Namespace(**payload)


def verify_selected_genes(run_dir: Path, gene_names: Sequence[str]) -> None:
    path = run_dir / "selected_genes.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing selected-gene artifact: {path}")
    frame = pd.read_csv(path)
    if "gene" not in frame.columns:
        raise ProtocolError(f"{path} must contain a gene column.")
    saved = frame["gene"].astype(str).tolist()
    if saved != [str(gene) for gene in gene_names]:
        raise ProtocolError(
            "Reconstructed feature selection does not match selected_genes.csv. "
            "Refusing to export misaligned edge weights."
        )


def reconstruct_model(run_dir: Path, checkpoint: Path, device: torch.device):
    args = namespace_from_saved_args(run_dir / "args.json", run_dir)
    if getattr(args, "model_variant", None) != "ppi_volumetric":
        raise ProtocolError(
            f"{run_dir} is model_variant={getattr(args, 'model_variant', None)!r}; "
            "VMA attention export requires ppi_volumetric."
        )
    if getattr(args, "augmentation", "none") != "none":
        raise ProtocolError(
            "Post-hoc reconstruction currently supports the fixed protocol augmentation='none' only."
        )
    set_seed(int(args.seed))
    split_file = worker.resolve_split_file(args)
    dataset, _, task_gene_indices, _ = worker.prepare_multitask_dataset(args, split_file)
    verify_selected_genes(run_dir, dataset.gene_names)
    task_specs = worker.build_task_specs(dataset.class_names)

    embedding_path = run_dir / "gene_embedding.csv"
    if not embedding_path.exists():
        raise FileNotFoundError(f"Missing constructor embedding artifact: {embedding_path}")
    ppi_prior_path = run_dir / "ppi_prior_embedding.csv"
    if not ppi_prior_path.exists():
        ppi_prior_path = None
    ppi_edge_file = Path(args.ppi_edge_file)
    if not ppi_edge_file.is_absolute():
        ppi_edge_file = resolve_path(ppi_edge_file)
    graph = load_induced_ppi_graph(
        ppi_edge_file,
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
    model.load_state_dict(state)
    requested_volume_mode = getattr(args, "volumetric_volume_mode", "raw")
    effective_volume_mode = getattr(model, "volumetric_volume_mode", None)
    if effective_volume_mode != requested_volume_mode:
        raise ProtocolError(
            "Checkpoint volumetric volume mode does not match args.json: "
            f"checkpoint={effective_volume_mode!r}, args={requested_volume_mode!r}. "
            "Refusing to export attention with mislabeled semantics."
        )
    model.eval()
    return args, dataset, task_specs, graph, model


def split_arrays(dataset: Any, split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.asarray(getattr(dataset, f"{split}_gene_x")),
        np.asarray(getattr(dataset, f"{split}_y")),
        np.asarray(getattr(dataset, f"{split}_ids")).astype(str),
    )


def _capture_to_numpy(capture: dict[str, Any], field: str) -> np.ndarray:
    value = capture.get(field)
    if value is None:
        raise ProtocolError(f"Model attention capture is missing {field!r}.")
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value)
    array = value.detach().float().cpu().numpy()
    if array.ndim != 3:
        raise ProtocolError(
            f"Captured {field} must have shape [batch, heads, edges], got {array.shape}."
        )
    return array


@torch.no_grad()
def collect_sparse_captures(
    model: torch.nn.Module,
    dataset: Any,
    task_specs: Sequence[Any],
    split: str,
    *,
    batch_size: int,
    device: torch.device,
    max_batches: int | None,
    mask_aware_heads: bool,
) -> dict[str, Any]:
    setter = getattr(model, "set_volumetric_attention_capture", None)
    getter = getattr(model, "volumetric_attention_captures", None)
    if not callable(setter) or not callable(getter):
        raise ProtocolError(
            "This TxTVolumetric build does not expose the required post-hoc capture API: "
            "set_volumetric_attention_capture()/volumetric_attention_captures()."
        )
    if batch_size <= 0:
        raise ValueError("batch-size must be positive.")
    if max_batches is not None and max_batches <= 0:
        raise ValueError("max-batches must be positive when provided.")

    gene_x, source_y, sample_ids = split_arrays(dataset, split)
    task_y, task_mask = worker.make_task_targets_for_samples(source_y, list(task_specs), sample_ids)
    setter(True)
    getter(clear=True)
    field_batches: dict[str, list[np.ndarray]] = {field: [] for field in CAPTURE_FIELDS}
    labels: list[np.ndarray] = []
    ids: list[np.ndarray] = []
    edge_index: np.ndarray | None = None
    capture_names: list[str] | None = None

    try:
        for batch_number, start in enumerate(range(0, len(gene_x), batch_size), start=1):
            if max_batches is not None and batch_number > max_batches:
                break
            stop = min(start + batch_size, len(gene_x))
            batch_x = torch.as_tensor(gene_x[start:stop], dtype=torch.float32, device=device)
            batch_mask = torch.as_tensor(task_mask[start:stop], dtype=torch.bool, device=device)
            model(batch_x, task_sample_mask=batch_mask if mask_aware_heads else None)
            captures = getter(clear=True)
            if not isinstance(captures, list) or not captures:
                raise ProtocolError(
                    "Attention capture was enabled, but the model returned no VMA layer captures."
                )
            current_names = [
                str(capture.get("name", capture.get("layer", f"capture_{index}")))
                for index, capture in enumerate(captures)
            ]
            if capture_names is None:
                capture_names = current_names
            elif capture_names != current_names:
                raise ProtocolError("VMA capture-layer order changed between batches.")

            current_edges = []
            for capture in captures:
                capture_edges = torch.as_tensor(capture.get("edge_index"), dtype=torch.long).cpu().numpy()
                if capture_edges.ndim != 2 or capture_edges.shape[0] != 2:
                    raise ProtocolError("Captured edge_index must have shape [2, edges].")
                current_edges.append(capture_edges)
            if any(not np.array_equal(current_edges[0], item) for item in current_edges[1:]):
                raise ProtocolError("Capture layers use different edge ordering; a single aligned export is impossible.")
            if edge_index is None:
                edge_index = current_edges[0]
            elif not np.array_equal(edge_index, current_edges[0]):
                raise ProtocolError("Captured edge ordering changed between batches.")

            for field in CAPTURE_FIELDS:
                per_layer = [_capture_to_numpy(capture, field) for capture in captures]
                if any(array.shape != per_layer[0].shape for array in per_layer[1:]):
                    raise ProtocolError(f"Capture layers returned inconsistent {field} shapes.")
                # [batch, capture_layer, head, edge]
                field_batches[field].append(np.stack(per_layer, axis=1))
            labels.append(source_y[start:stop])
            ids.append(sample_ids[start:stop])
    finally:
        setter(False)
        getter(clear=True)

    if not labels or edge_index is None or capture_names is None:
        raise ProtocolError(f"Split {split!r} produced no capture batches.")
    return {
        **{field: np.concatenate(parts, axis=0) for field, parts in field_batches.items()},
        "edge_index": edge_index,
        "capture_names": np.asarray(capture_names, dtype=str),
        "source_y": np.concatenate(labels).astype(np.int64),
        "sample_ids": np.concatenate(ids).astype(str),
    }


def edge_scores_aligned(graph: Any, edge_index: np.ndarray) -> np.ndarray:
    score_by_edge = {
        (int(destination), int(source)): float(score)
        for (destination, source), score in zip(
            graph.edge_index.transpose(0, 1).tolist(), graph.edge_scores.tolist()
        )
    }
    try:
        return np.asarray(
            [score_by_edge[(int(destination), int(source))] for destination, source in edge_index.T],
            dtype=np.float32,
        )
    except KeyError as exc:
        raise ProtocolError(f"Captured edge {exc.args[0]} is absent from the reconstructed graph.") from exc


def aggregate_capture_frame(
    captures: dict[str, Any],
    *,
    gene_names: Sequence[str],
    class_names: Sequence[str],
    edge_scores: np.ndarray,
) -> pd.DataFrame:
    weights = captures["attention_weights"]
    volumes = captures["volumes"]
    logits = captures["logits"]
    labels = captures["source_y"]
    edge_index = captures["edge_index"]
    capture_names = captures["capture_names"]
    if weights.shape != volumes.shape or weights.shape != logits.shape:
        raise ProtocolError("attention_weights, volumes, and logits must have identical capture shapes.")
    _, n_layers, n_heads, n_edges = weights.shape
    if edge_index.shape[1] != n_edges or edge_scores.shape != (n_edges,):
        raise ProtocolError("Graph metadata is not aligned to captured edges.")

    target = edge_index[0].astype(np.int64)
    source = edge_index[1].astype(np.int64)
    columns = [
        "class_index", "class_name", "capture_index", "capture_name", "head", "edge_offset",
        "target_index", "source_index", "target_gene", "source_gene", "ppi_score", "sample_count",
        "attention_mean", "attention_std", "attention_min", "attention_max", "volume_mean", "volume_std",
        "logit_mean", "logit_std",
    ]
    frames = []
    for class_index, class_name in enumerate(class_names):
        keep = labels == class_index
        if not bool(keep.any()):
            continue
        for layer_index in range(n_layers):
            for head_index in range(n_heads):
                class_weights = weights[keep, layer_index, head_index, :]
                class_volumes = volumes[keep, layer_index, head_index, :]
                class_logits = logits[keep, layer_index, head_index, :]
                frames.append(
                    pd.DataFrame(
                        {
                            "class_index": class_index,
                            "class_name": str(class_name),
                            "capture_index": layer_index,
                            "capture_name": str(capture_names[layer_index]),
                            "head": head_index,
                            "edge_offset": np.arange(n_edges, dtype=np.int64),
                            "target_index": target,
                            "source_index": source,
                            "target_gene": [gene_names[index] for index in target],
                            "source_gene": [gene_names[index] for index in source],
                            "ppi_score": edge_scores,
                            "sample_count": int(keep.sum()),
                            "attention_mean": class_weights.mean(axis=0),
                            "attention_std": class_weights.std(axis=0),
                            "attention_min": class_weights.min(axis=0),
                            "attention_max": class_weights.max(axis=0),
                            "volume_mean": class_volumes.mean(axis=0),
                            "volume_std": class_volumes.std(axis=0),
                            "logit_mean": class_logits.mean(axis=0),
                            "logit_std": class_logits.std(axis=0),
                        }
                    )
                )
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_dir = resolve_path(args.run_dir)
    checkpoint = resolve_path(args.checkpoint) if args.checkpoint is not None else run_dir / "best_model.pt"
    output_dir = resolve_path(args.output_dir) if args.output_dir is not None else run_dir / "attention_export"
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)

    saved_args, dataset, task_specs, graph, model = reconstruct_model(run_dir, checkpoint, device)
    captures = collect_sparse_captures(
        model,
        dataset,
        task_specs,
        args.split,
        batch_size=args.batch_size,
        device=device,
        max_batches=args.max_batches,
        mask_aware_heads=getattr(saved_args, "mask_aware_heads", "off") == "on",
    )
    scores = edge_scores_aligned(graph, captures["edge_index"])
    aggregate = aggregate_capture_frame(
        captures,
        gene_names=dataset.gene_names,
        class_names=dataset.class_names,
        edge_scores=scores,
    )
    aggregate_path = output_dir / args.aggregate_name
    aggregate.to_csv(aggregate_path, index=False)

    npz_path: Path | None = None
    if args.save_per_sample_npz:
        npz_path = output_dir / args.per_sample_name
        np.savez_compressed(
            npz_path,
            attention_weights=captures["attention_weights"].astype(np.float32),
            volumes=captures["volumes"].astype(np.float32),
            logits=captures["logits"].astype(np.float32),
            edge_index=captures["edge_index"].astype(np.int64),
            edge_scores=scores,
            sample_ids=captures["sample_ids"],
            source_labels=captures["source_y"],
            class_names=np.asarray(dataset.class_names, dtype=str),
            gene_names=np.asarray(dataset.gene_names, dtype=str),
            capture_names=captures["capture_names"],
        )
    diagnostics = getattr(model, "volumetric_diagnostics", None)
    write_json(
        output_dir / "attention_export_manifest.json",
        {
            "run_dir": str(run_dir),
            "checkpoint": str(checkpoint),
            "split": args.split,
            "device": str(device),
            "samples": int(len(captures["sample_ids"])),
            "capture_layers": captures["capture_names"].tolist(),
            "heads": int(captures["attention_weights"].shape[2]),
            "directed_edges": int(captures["attention_weights"].shape[3]),
            "edge_index_convention": "row0_target_i__row1_source_j",
            "aggregate_csv": str(aggregate_path),
            "per_sample_npz": str(npz_path) if npz_path is not None else None,
            "dense_attention_saved": False,
            "volumetric_beta": float(saved_args.volumetric_beta),
            "volumetric_eps": float(saved_args.volumetric_eps),
            "volumetric_volume_mode": getattr(saved_args, "volumetric_volume_mode", "raw"),
            "volumetric_dropout": getattr(saved_args, "volumetric_dropout", None),
            "volumetric_dropout_effective": float(
                saved_args.dropout
                if getattr(saved_args, "volumetric_dropout", None) is None
                else saved_args.volumetric_dropout
            ),
            "volumetric_message_mode": getattr(
                saved_args, "volumetric_message_mode", "legacy"
            ),
            "volumetric_output_norm": getattr(saved_args, "volumetric_output_norm", "none"),
            "volumetric_gate_mode": getattr(saved_args, "volumetric_gate_mode", "scalar"),
            "volumetric_backbone_gradient_mode": getattr(
                saved_args, "volumetric_backbone_gradient_mode", "coupled"
            ),
            "volumetric_diagnostics": diagnostics() if callable(diagnostics) else None,
        },
    )
    print(f"Sparse VMA attention aggregate: {aggregate_path}")
    if npz_path is not None:
        print(f"Per-sample sparse VMA arrays: {npz_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
