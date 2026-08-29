#!/usr/bin/env python3
"""Isolated CTGAN worker for native Windows PyTorch DLL compatibility."""

from __future__ import annotations

# PyTorch must be the first numerical runtime loaded in this interpreter.
import torch  # noqa: F401
import ctgan  # noqa: F401

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from augmentation import _train_and_sample_ctgan_inprocess


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--outputs", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--n-to-generate", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--latent-dim", type=int, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--sampling-strategy", choices=["proportional", "minority", "balanced"], required=True)
    parser.add_argument("--pac", type=int, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], required=True)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CTGAN CUDA was requested, but torch.cuda.is_available() is false.")
    with np.load(args.inputs, allow_pickle=False) as payload:
        x = payload["x"].astype(np.float32)
        y = payload["y"].astype(np.int32)
    frame = pd.DataFrame(x, columns=[f"feature_{index}" for index in range(x.shape[1])])
    synthetic_x, synthetic_y, metadata = _train_and_sample_ctgan_inprocess(
        x_train=frame,
        y_train=y,
        n_to_generate=args.n_to_generate,
        seed=args.seed,
        latent_dim=args.latent_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        sampling_strategy=args.sampling_strategy,
        pac=args.pac,
        enable_gpu=args.device == "cuda",
    )
    metadata.update(
        {
            "torch_version": torch.__version__,
            "torch_device": args.device,
            "torch_cuda_available": bool(torch.cuda.is_available()),
        }
    )
    np.savez_compressed(args.outputs, synthetic_x=synthetic_x, synthetic_y=synthetic_y)
    args.metadata.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
