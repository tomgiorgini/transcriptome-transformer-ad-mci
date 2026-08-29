#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from source.models.tgem import TGemClassifier
from source.pipeline.dataset import prepare_dataset
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
from source.pipeline.utils import (
    DEFAULT_OFFICIAL_SPLIT_FILE,
    DEFAULT_X_FILE,
    DEFAULT_Y_FILE,
    compute_balanced_class_weights,
    namespace_to_dict,
    resolve_device,
    save_json,
    set_seed,
)


DEFAULT_RESULT_DIR = ROOT / "results" / "baseline" / "tgem" / "default_run"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the T-GEM classifier on the canonical Alzheimer dataset.")
    parser.add_argument("--x-file", type=Path, default=DEFAULT_X_FILE)
    parser.add_argument("--y-file", type=Path, default=DEFAULT_Y_FILE)
    parser.add_argument("--split-mode", choices=["official", "custom", "stratified"], default="official")
    parser.add_argument("--split-file", type=Path, default=None, help="Used only when --split-mode=custom.")
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--split-seed",
        type=int,
        default=None,
        help="Used only with --split-mode=stratified. If omitted, the main --seed is reused for the split.",
    )
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument(
        "--max-genes",
        type=int,
        default=256,
        help="Top genes by train variance. Use 0 or a negative value to disable truncation and keep all provided genes.",
    )
    parser.add_argument("--scaler", choices=["none", "minmax", "standard"], default="minmax")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--class-weighting", choices=["on", "off"], default="on")
    parser.add_argument("--device", choices=["cuda", "cpu", "mps"], default="cuda")
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--activation", choices=["relu", "leakyrelu", "gelu"], default="leakyrelu")
    return parser.parse_args()


def resolve_split_file(args: argparse.Namespace) -> Path | None:
    if args.split_mode == "official":
        return DEFAULT_OFFICIAL_SPLIT_FILE
    if args.split_mode == "custom":
        if args.split_file is None:
            raise ValueError("`--split-file` is required when `--split-mode=custom`.")
        return args.split_file
    return None


def logits_fn(model: TGemClassifier, batch_gene_x: torch.Tensor) -> torch.Tensor:
    return model(batch_gene_x)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    resolved_split_file = resolve_split_file(args)
    args.result_dir.mkdir(parents=True, exist_ok=True)
    save_args(
        args.result_dir,
        {
            **namespace_to_dict(args),
            "resolved_split_file": str(resolved_split_file) if resolved_split_file is not None else None,
        },
    )

    if resolved_split_file is not None and not resolved_split_file.exists():
        raise FileNotFoundError(f"Split file not found: {resolved_split_file}")

    device = resolve_device(args.device)
    dataset = prepare_dataset(
        x_file=args.x_file,
        y_file=args.y_file,
        seed=args.seed,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        max_genes=args.max_genes,
        scaler=args.scaler,
        split_file=resolved_split_file,
        split_seed=args.split_seed,
    )

    save_selected_genes(args.result_dir, dataset.gene_names)
    save_split_assignments(args.result_dir, dataset.train_ids, dataset.val_ids, dataset.test_ids)

    class_weights = compute_balanced_class_weights(dataset.train_y) if args.class_weighting == "on" else None

    model = TGemClassifier(
        n_genes=len(dataset.gene_names),
        n_classes=len(dataset.class_names),
        n_heads=args.n_heads,
        dropout=args.dropout,
        activation=args.activation,
        n_layers=args.n_layers,
    ).to(device)

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
    )
    save_training_history(args.result_dir, history_rows)

    best_model_path = args.result_dir / "best_model.pt"
    torch.save(best_state_dict, best_model_path)
    model.load_state_dict(best_state_dict)

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

    save_metrics_summary(args.result_dir, split_results)
    save_test_artifacts(args.result_dir, split_results["test"], dataset.class_names)
    save_json(
        args.result_dir / "model_summary.json",
        {
            "model": "tgem",
            "device": str(device),
            "num_genes": len(dataset.gene_names),
            "num_classes": len(dataset.class_names),
            "split_mode": args.split_mode,
            "split_seed": args.split_seed,
            "resolved_split_file": str(resolved_split_file) if resolved_split_file is not None else None,
            "class_weighting": args.class_weighting,
            "best_model_path": str(best_model_path),
            "training_summary": training_summary,
        },
    )


if __name__ == "__main__":
    main()
