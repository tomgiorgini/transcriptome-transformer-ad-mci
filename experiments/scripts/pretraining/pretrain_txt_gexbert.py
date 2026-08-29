#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from source.pipeline.utils import namespace_to_dict, resolve_device, save_json, set_seed
from source.pretraining.txt.data import (
    ExpressionMatrixDataset,
    global_zscore_report,
    load_pretraining_matrix,
    make_masked_collate_fn,
    split_train_val_indices,
)
from source.pretraining.txt.modeling import TxTMaskedRestorer
from source.pretraining.txt.training import run_restoration_training
from source.pretraining.txt.transfer import build_random_embedding_dataframe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Self-supervised TxT/GExBERT-style pretraining.")
    parser.add_argument("--matrix-file", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--device", choices=["cuda", "cpu", "mps"], default="cuda")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--d-ff", type=int, default=512)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--norm-first", action="store_true")
    parser.add_argument("--gene-subset-size", type=int, default=512)
    parser.add_argument("--mask-ratio", type=float, default=0.15)
    parser.add_argument("--keep-top-k-checkpoints", type=int, default=5)
    parser.add_argument("--save-every-epochs", type=int, default=0)
    parser.add_argument("--early-stopping-patience", type=int, default=0)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    return parser.parse_args()


def default_result_dir() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return ROOT / "results" / "pretraining" / "self_supervised" / "txt_gexbert" / f"run_{timestamp}"


def save_checkpoint(
    path: Path,
    model: TxTMaskedRestorer,
    epoch: int,
    val_loss: float,
    config: dict[str, object],
    gene_names: list[str],
    training_summary: dict[str, float | int],
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "val_loss": val_loss,
            "config": config,
            "gene_names": gene_names,
            "training_summary": training_summary,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)

    result_dir = args.result_dir if args.result_dir is not None else default_result_dir()
    result_dir.mkdir(parents=True, exist_ok=True)

    matrix = load_pretraining_matrix(args.matrix_file)
    train_idx, val_idx = split_train_val_indices(len(matrix.sample_ids), args.val_ratio, args.seed)

    embedding_df = build_random_embedding_dataframe(matrix.gene_names, args.embed_dim, args.seed)
    embedding_path = result_dir / "gene_embedding.csv"
    embedding_df.to_csv(embedding_path)

    train_generator = torch.Generator().manual_seed(args.seed)
    val_generator = torch.Generator().manual_seed(args.seed + 1)
    train_loader = DataLoader(
        ExpressionMatrixDataset(matrix.values[train_idx]),
        batch_size=args.batch_size,
        shuffle=True,
        generator=train_generator,
        collate_fn=make_masked_collate_fn(args.gene_subset_size, args.mask_ratio, train_generator),
    )
    val_loader = DataLoader(
        ExpressionMatrixDataset(matrix.values[val_idx]),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=make_masked_collate_fn(args.gene_subset_size, args.mask_ratio, val_generator),
    )

    model = TxTMaskedRestorer(
        embed_file=str(embedding_path),
        gene_list=matrix.gene_names,
        n_heads=args.n_heads,
        d_model=args.d_model,
        dropout=args.dropout,
        d_ff=args.d_ff,
        norm_first=args.norm_first,
        n_layers=args.n_layers,
    ).to(device)

    config = {
        **namespace_to_dict(args),
        "result_dir": str(result_dir),
        "device_resolved": str(device),
        "samples": int(matrix.values.shape[0]),
        "genes": int(matrix.values.shape[1]),
        "effective_gene_subset_size": int(matrix.values.shape[1] if args.gene_subset_size <= 0 else min(args.gene_subset_size, matrix.values.shape[1])),
        "train_samples": int(len(train_idx)),
        "val_samples": int(len(val_idx)),
        "embedding_file": str(embedding_path),
        "zscore_report": global_zscore_report(matrix.values),
    }
    save_json(result_dir / "config.json", config)
    pd.DataFrame({"sample_id": matrix.sample_ids[train_idx], "split": "train"}).to_csv(
        result_dir / "train_samples.csv", index=False
    )
    pd.DataFrame({"sample_id": matrix.sample_ids[val_idx], "split": "val"}).to_csv(
        result_dir / "val_samples.csv", index=False
    )
    (result_dir / "pretrained_genes.txt").write_text("\n".join(matrix.gene_names) + "\n", encoding="utf-8")

    checkpoint_dir = result_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    top_checkpoints: list[tuple[float, Path]] = []

    def save_epoch_checkpoint(epoch: int, model: TxTMaskedRestorer, row: dict[str, float]) -> None:
        val_loss = float(row["val_loss"])
        should_save_periodic = args.save_every_epochs > 0 and epoch % args.save_every_epochs == 0
        should_save_top_k = args.keep_top_k_checkpoints > 0
        if not should_save_periodic and not should_save_top_k:
            return

        path = checkpoint_dir / f"epoch_{epoch:04d}_val_{val_loss:.6f}.pt"
        saved = False
        if should_save_periodic or len(top_checkpoints) < args.keep_top_k_checkpoints or val_loss < max(top_checkpoints, key=lambda item: item[0])[0]:
            save_checkpoint(path, model, epoch, val_loss, config, matrix.gene_names, {})
            saved = True

        if should_save_top_k and saved:
            top_checkpoints.append((val_loss, path))
            top_checkpoints.sort(key=lambda item: item[0])
            while len(top_checkpoints) > args.keep_top_k_checkpoints:
                _, stale_path = top_checkpoints.pop()
                if stale_path.exists():
                    stale_path.unlink()

    history, best_state_dict, training_summary = run_restoration_training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=device,
        max_train_batches=args.max_train_batches,
        max_val_batches=args.max_val_batches,
        early_stopping_patience=args.early_stopping_patience,
        epoch_callback=save_epoch_checkpoint,
    )
    pd.DataFrame(history).to_csv(result_dir / "training_history.csv", index=False)
    save_json(result_dir / "training_summary.json", training_summary)

    last_epoch = int(history[-1]["epoch"]) if history else 0
    last_val_loss = float(history[-1]["val_loss"]) if history else float("inf")
    save_checkpoint(
        result_dir / "last_checkpoint.pt",
        model,
        last_epoch,
        last_val_loss,
        config,
        matrix.gene_names,
        training_summary,
    )

    model.load_state_dict(best_state_dict)
    save_checkpoint(
        result_dir / "best_checkpoint.pt",
        model,
        int(training_summary["best_epoch"]),
        float(training_summary["best_val_loss"]),
        config,
        matrix.gene_names,
        training_summary,
    )
    print(f"Pretraining complete. Results: {result_dir}")


if __name__ == "__main__":
    main()
