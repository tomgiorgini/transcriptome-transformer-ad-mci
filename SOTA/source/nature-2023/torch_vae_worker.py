#!/usr/bin/env python3
from __future__ import annotations

# Import torch before numpy/sklearn/TensorFlow. On Windows, loading it later can
# fail with WinError 1114 because of conflicting native runtime DLLs.
import torch  # noqa: F401

import argparse
import json
from pathlib import Path
import sys

import numpy as np


MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from feature_selection import _torch_vae_latents


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Isolated PyTorch worker for Kelly VAE latent features.")
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--outputs", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--patience", type=int, required=True)
    parser.add_argument("--architecture", choices=["basic", "batchnorm", "batchnorm_dropout"], required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--reconstruction-loss", choices=["categorical_crossentropy", "binary_crossentropy"], required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], required=True)
    parser.add_argument("--hidden-dims", type=int, nargs=3, default=[4096, 1024, 512])
    parser.add_argument("--latent-dim", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with np.load(args.inputs) as inputs:
        x_train = inputs["x_train"].astype(np.float32)
        x_val = inputs["x_val"].astype(np.float32)
        x_test = inputs["x_test"].astype(np.float32)
    z_train, z_val, z_test, metadata = _torch_vae_latents(
        x_train,
        x_val,
        x_test,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        architecture=args.architecture,
        learning_rate=args.learning_rate,
        reconstruction_loss=args.reconstruction_loss,
        device_name=args.device,
        hidden_dims=tuple(args.hidden_dims),
        latent_dim=args.latent_dim,
    )
    args.outputs.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.outputs, z_train=z_train, z_val=z_val, z_test=z_test)
    args.metadata.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
