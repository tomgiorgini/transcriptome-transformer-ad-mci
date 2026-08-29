#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from source.pretraining.txt.data import load_pretraining_matrix
from source.pretraining.txt.modeling import TxTMaskedRestorer
from source.pipeline.utils import resolve_device

from experiments.scripts.finetuning.legacy_finetune_txt_from_pretraining import collect_attention_matrices, save_attention_matrix_plots


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot TxT input and attention matrices from an already computed pretraining checkpoint."
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument(
        "--finetuning-run-dir",
        type=Path,
        default=None,
        help="Completed finetuning run dir. Its args.json is used to find the pretrained checkpoint.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", choices=["cuda", "cpu", "mps"], default="cuda")
    parser.add_argument("--attention-max-samples", type=int, default=32)
    parser.add_argument("--input-max-samples", type=int, default=160)
    parser.add_argument("--attention-scale", choices=["linear", "percentile", "log", "relative"], default="relative")
    parser.add_argument("--attention-vmax-percentile", type=float, default=99.0)
    parser.add_argument(
        "--attention-linear-vmax",
        type=float,
        default=None,
        help="Optional fixed vmax for raw linear attention plots. Values above it are capped only in the visualization.",
    )
    parser.add_argument("--attention-output-suffix", default="")
    parser.add_argument("--save-attention-csv", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_checkpoint(args: argparse.Namespace) -> Path:
    if args.checkpoint is not None:
        return args.checkpoint
    if args.finetuning_run_dir is None:
        raise ValueError("Pass either --checkpoint or --finetuning-run-dir.")
    run_args = load_json(args.finetuning_run_dir / "args.json")
    return ROOT / run_args["pretrained_checkpoint"]


def save_pretraining_input_matrix_plot(
    output_dir: Path,
    values: np.ndarray,
    sample_ids: np.ndarray,
    gene_names: list[str],
    max_samples: int,
) -> None:
    plot_dir = output_dir / "plots" / "matrices"
    plot_dir.mkdir(parents=True, exist_ok=True)
    if max_samples <= 0 or values.shape[0] <= max_samples:
        selected = np.arange(values.shape[0], dtype=np.int64)
    else:
        selected = np.linspace(0, values.shape[0] - 1, num=max_samples, dtype=np.int64)

    matrix = values[selected]
    pd.DataFrame(matrix, columns=gene_names).to_csv(plot_dir / "pretraining_input_matrix_plotted_values.csv", index=False)
    pd.DataFrame({"sample_id": sample_ids[selected]}).to_csv(plot_dir / "pretraining_input_matrix_plotted_rows.csv", index=False)

    width = min(max(9.0, matrix.shape[1] / 55.0), 18.0)
    height = min(max(5.0, matrix.shape[0] / 20.0), 12.0)
    fig, ax = plt.subplots(figsize=(width, height), constrained_layout=True)
    image = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap="viridis")
    ax.set_title("Pretraining input matrix")
    ax.set_xlabel(f"Genes (n={len(gene_names)})")
    ax.set_ylabel(f"Samples (n={matrix.shape[0]})")
    ax.set_xticks([])
    fig.colorbar(image, ax=ax, label="Pretraining expression value")
    fig.savefig(plot_dir / "pretraining_input_matrix.png", dpi=220)
    fig.savefig(plot_dir / "pretraining_input_matrix.pdf")
    plt.close(fig)


def build_pretraining_model(checkpoint: dict[str, Any], checkpoint_path: Path, device: torch.device) -> TxTMaskedRestorer:
    config = checkpoint.get("config", {})
    gene_names = [str(gene) for gene in checkpoint["gene_names"]]
    state = checkpoint["model_state_dict"]
    embedding = state["transformer.encoder.embed.embed.weight"].detach().cpu().numpy()

    with tempfile.TemporaryDirectory() as temp_dir:
        embedding_path = Path(temp_dir) / "pretraining_embedding.csv"
        pd.DataFrame(embedding, index=gene_names).to_csv(embedding_path)
        model = TxTMaskedRestorer(
            embed_file=str(embedding_path),
            gene_list=gene_names,
            n_heads=int(config.get("n_heads", 4)),
            d_model=int(config.get("d_model", 128)),
            dropout=float(config.get("dropout", 0.2)),
            d_ff=int(config.get("d_ff", 512)),
            norm_first=bool(config.get("norm_first", False)),
            n_layers=int(config.get("n_layers", 3)),
        ).to(device)

    model.load_state_dict(state)
    model.eval()
    print(f"Loaded pretraining checkpoint: {checkpoint_path}", flush=True)
    return model


def main() -> None:
    args = parse_args()
    checkpoint_path = resolve_checkpoint(args)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})
    matrix_file = config.get("matrix_file")
    if matrix_file is None:
        raise ValueError("Checkpoint config does not contain matrix_file.")

    matrix = load_pretraining_matrix(ROOT / matrix_file)
    checkpoint_genes = [str(gene) for gene in checkpoint["gene_names"]]
    matrix_gene_to_idx = {gene: idx for idx, gene in enumerate(matrix.gene_names)}
    missing = [gene for gene in checkpoint_genes if gene not in matrix_gene_to_idx]
    if missing:
        preview = ", ".join(missing[:10])
        raise ValueError(f"Checkpoint genes missing from pretraining matrix. Example: {preview}")
    keep = np.asarray([matrix_gene_to_idx[gene] for gene in checkpoint_genes], dtype=np.int64)
    values = matrix.values[:, keep].astype(np.float32)

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = checkpoint_path.parent / "post_pretraining_plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    model = build_pretraining_model(checkpoint, checkpoint_path, device)
    save_pretraining_input_matrix_plot(output_dir, values, matrix.sample_ids, checkpoint_genes, args.input_max_samples)
    attention_matrices = collect_attention_matrices(
        model=model,
        gene_x=values,
        device=device,
        batch_size=int(config.get("batch_size", 8)),
        max_samples=args.attention_max_samples,
    )
    save_attention_matrix_plots(
        output_dir,
        attention_matrices,
        checkpoint_genes,
        save_csv=args.save_attention_csv,
        scale=args.attention_scale,
        vmax_percentile=args.attention_vmax_percentile,
        filename_suffix=args.attention_output_suffix,
        linear_vmax=args.attention_linear_vmax,
    )
    print(f"Pretraining plots written to: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
