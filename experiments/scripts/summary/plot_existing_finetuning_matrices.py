#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from source.models.txt import TxT
from source.pipeline.dataset import PreparedDataset, prepare_dataset
from source.pipeline.utils import resolve_device

from experiments.scripts.finetuning.legacy_finetune_txt_from_pretraining import (
    collect_attention_matrices,
    collect_attention_matrices_by_class,
    save_attention_matrix_plots,
    save_class_attention_matrix_plots,
    save_finetuning_input_matrix_plot,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot finetuning input matrices and TxT attention matrices from already computed finetuning runs."
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=ROOT / "results" / "pretraining" / "finetuning",
        help="Root folder to scan for completed finetuning runs.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        action="append",
        default=[],
        help="Specific completed split/run directory. Can be passed multiple times.",
    )
    parser.add_argument("--device", choices=["cuda", "cpu", "mps"], default="cuda")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size used for attention extraction.")
    parser.add_argument("--attention-max-samples", type=int, default=32)
    parser.add_argument("--input-max-samples", type=int, default=160)
    parser.add_argument("--input-scale", choices=["raw", "gene_zscore", "gene_robust_zscore"], default="raw")
    parser.add_argument("--input-cmap", default="viridis")
    parser.add_argument("--input-clip", type=float, default=None)
    parser.add_argument("--skip-attention", action="store_true")
    parser.add_argument(
        "--attention-scale",
        choices=["linear", "percentile", "log", "relative"],
        default="percentile",
        help="Display transform for attention heatmaps. Numeric NPZ data remains untransformed.",
    )
    parser.add_argument(
        "--attention-vmax-percentile",
        type=float,
        default=99.0,
        help="Upper percentile used when --attention-scale percentile.",
    )
    parser.add_argument(
        "--attention-linear-vmax",
        type=float,
        default=None,
        help="Optional fixed vmax for raw linear attention plots. Values above it are clipped only for visualization.",
    )
    parser.add_argument(
        "--attention-output-suffix",
        default="",
        help="Suffix added before .png/.pdf for attention plots, for example _log or _relative.",
    )
    parser.add_argument(
        "--save-attention-csv",
        action="store_true",
        help="Also write full gene-by-gene attention CSV files. PNG/PDF and compressed NPZ are always saved.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def find_run_dirs(runs_root: Path) -> list[Path]:
    if not runs_root.exists():
        raise FileNotFoundError(f"Runs root not found: {runs_root}")
    return sorted(path.parent for path in runs_root.rglob("best_model.pt") if (path.parent / "args.json").exists())


def read_selected_genes(run_dir: Path) -> list[str]:
    selected_path = run_dir / "selected_genes.csv"
    if not selected_path.exists():
        raise FileNotFoundError(f"Missing selected genes file: {selected_path}")
    df = pd.read_csv(selected_path)
    if "gene" not in df.columns:
        raise ValueError(f"selected_genes.csv must contain a gene column: {selected_path}")
    return df["gene"].astype(str).tolist()


def subset_dataset_to_genes(dataset: PreparedDataset, selected_genes: list[str]) -> PreparedDataset:
    gene_to_idx = {gene: idx for idx, gene in enumerate(dataset.gene_names)}
    missing = [gene for gene in selected_genes if gene not in gene_to_idx]
    if missing:
        preview = ", ".join(missing[:10])
        raise ValueError(f"Selected genes are missing from reconstructed dataset. Example: {preview}")
    keep_indices = np.asarray([gene_to_idx[gene] for gene in selected_genes], dtype=np.int64)
    return PreparedDataset(
        class_names=dataset.class_names,
        gene_names=selected_genes,
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


def reconstruct_dataset(run_dir: Path, run_args: dict[str, Any], selected_genes: list[str]) -> PreparedDataset:
    dataset = prepare_dataset(
        x_file=ROOT / run_args["x_file"],
        y_file=ROOT / run_args["y_file"],
        seed=int(run_args.get("seed", 42)),
        val_ratio=float(run_args.get("val_ratio", 0.1)),
        test_ratio=float(run_args.get("test_ratio", 0.2)),
        max_genes=int(run_args.get("max_genes", 0)),
        scaler=str(run_args.get("scaler", "minmax")),
        scaler_fit_scope=str(run_args.get("scaler_fit_scope", "train")),
        split_file=None,
        split_seed=int(run_args.get("split_seed", run_args.get("seed", 42))),
        split_mode=str(run_args.get("split_mode", "stratified")),
    )
    try:
        return subset_dataset_to_genes(dataset, selected_genes)
    except ValueError as exc:
        raise ValueError(f"Could not reconstruct selected gene matrix for {run_dir}: {exc}") from exc


def build_model(
    run_dir: Path,
    run_args: dict[str, Any],
    selected_genes: list[str],
    n_classes: int,
    device: torch.device,
) -> TxT:
    embedding_path = run_dir / "gene_embedding_from_pretraining.csv"
    best_model_path = run_dir / "best_model.pt"
    if not embedding_path.exists():
        raise FileNotFoundError(f"Missing remapped embedding file: {embedding_path}")
    if not best_model_path.exists():
        raise FileNotFoundError(f"Missing best model file: {best_model_path}")

    resolved = run_args.get("resolved_model_config", {})
    model = TxT(
        embed_file=str(embedding_path),
        gene_list=selected_genes,
        n_heads=int(resolved.get("n_heads", run_args.get("n_heads") or 4)),
        d_model=int(resolved.get("d_model", run_args.get("d_model") or 128)),
        dropout=float(resolved.get("dropout", run_args.get("dropout") or 0.2)),
        d_ff=int(resolved.get("d_ff", run_args.get("d_ff") or 512)),
        norm_first=bool(resolved.get("norm_first", run_args.get("norm_first", False))),
        n_layers=int(resolved.get("n_layers", run_args.get("n_layers") or 3)),
        aggfunc=str(run_args.get("aggfunc", "Avgpool")),
        d_hidden1=int(run_args.get("d_hidden1", 256)),
        d_hidden2=int(run_args.get("d_hidden2", 128)),
        slope=float(run_args.get("slope", 0.2)),
        d_output_dict={"label": n_classes},
    ).to(device)
    state_dict = torch.load(best_model_path, map_location=device, weights_only=False)
    model.load_state_dict(state_dict)
    return model


def plots_exist(run_dir: Path) -> bool:
    return (
        (run_dir / "plots" / "matrices" / "finetuning_input_matrix.png").exists()
        and (run_dir / "plots" / "attention" / "attention_all_layers_mean.png").exists()
    )


def process_run(
    run_dir: Path,
    device: torch.device,
    batch_size_override: int | None,
    attention_max_samples: int,
    input_max_samples: int,
    input_scale: str,
    input_cmap: str,
    input_clip: float | None,
    save_attention_csv: bool,
    attention_scale: str,
    attention_vmax_percentile: float,
    attention_output_suffix: str,
    attention_linear_vmax: float | None,
    skip_attention: bool,
    overwrite: bool,
) -> bool:
    if plots_exist(run_dir) and not overwrite:
        print(f"Skipping existing plots: {run_dir}", flush=True)
        return False

    run_args = load_json(run_dir / "args.json")
    selected_genes = read_selected_genes(run_dir)
    dataset = reconstruct_dataset(run_dir, run_args, selected_genes)

    save_finetuning_input_matrix_plot(run_dir, dataset, input_max_samples, scale=input_scale, cmap=input_cmap, clip=input_clip)
    if skip_attention:
        print(f"Plotted input matrix: {run_dir}", flush=True)
        return True

    model = build_model(run_dir, run_args, selected_genes, len(dataset.class_names), device)
    batch_size = batch_size_override if batch_size_override is not None else int(run_args.get("batch_size", 8))
    attention_matrices = collect_attention_matrices(
        model=model,
        gene_x=dataset.train_gene_x,
        device=device,
        batch_size=batch_size,
        max_samples=attention_max_samples,
    )
    save_attention_matrix_plots(
        run_dir,
        attention_matrices,
        dataset.gene_names,
        save_csv=save_attention_csv,
        scale=attention_scale,
        vmax_percentile=attention_vmax_percentile,
        filename_suffix=attention_output_suffix,
        linear_vmax=attention_linear_vmax,
    )
    class_attention = collect_attention_matrices_by_class(
        model=model,
        gene_x=dataset.train_gene_x,
        y=dataset.train_y,
        class_names=dataset.class_names,
        device=device,
        batch_size=batch_size,
        max_samples_per_class=attention_max_samples,
    )
    save_class_attention_matrix_plots(
        run_dir,
        class_attention,
        dataset.gene_names,
        save_csv=save_attention_csv,
        scale=attention_scale,
        vmax_percentile=attention_vmax_percentile,
        filename_suffix=attention_output_suffix,
        linear_vmax=attention_linear_vmax,
    )
    print(f"Plotted: {run_dir}", flush=True)
    return True


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    run_dirs = [path.resolve() for path in args.run_dir] if args.run_dir else find_run_dirs(args.runs_root)
    if not run_dirs:
        raise ValueError("No completed finetuning runs found.")

    completed = 0
    failed: list[tuple[Path, str]] = []
    for run_dir in run_dirs:
        try:
            completed += int(
                process_run(
                    run_dir=run_dir,
                    device=device,
                    batch_size_override=args.batch_size,
                    attention_max_samples=args.attention_max_samples,
                    input_max_samples=args.input_max_samples,
                    input_scale=args.input_scale,
                    input_cmap=args.input_cmap,
                    input_clip=args.input_clip,
                    save_attention_csv=args.save_attention_csv,
                    attention_scale=args.attention_scale,
                    attention_vmax_percentile=args.attention_vmax_percentile,
                    attention_output_suffix=args.attention_output_suffix,
                    attention_linear_vmax=args.attention_linear_vmax,
                    skip_attention=args.skip_attention,
                    overwrite=args.overwrite,
                )
            )
        except Exception as exc:  # noqa: BLE001
            failed.append((run_dir, str(exc)))
            print(f"Failed: {run_dir} | {exc}", flush=True)

    print(f"Completed plots for {completed} run(s). Failed: {len(failed)}", flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
