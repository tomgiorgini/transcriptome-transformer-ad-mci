#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
PREPROCESSOR = ROOT / "experiments" / "scripts" / "finetuning" / "build_optuna_task_datasets.py"
OPTUNA_CV = ROOT / "experiments" / "scripts" / "finetuning" / "optuna_finetuning_cv.py"
DEFAULT_DATA_DIR = ROOT / "task_dataset" / "processed" / "optuna_current"
DEFAULT_RESULT_ROOT = ROOT / "results" / "pretraining" / "finetuning_optuna" / "current_8combo"
DEFAULT_CHECKPOINT = (
    ROOT
    / "results"
    / "pretraining"
    / "self_supervised"
    / "txt_gexbert"
    / "baseline_1l2h_d256_dff1024_dropout04_deg_with_reference_500ep_mask25"
    / "best_checkpoint.pt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one current 8-combination Optuna 5-CV TxT experiment.")
    parser.add_argument("--dataset", choices=["legacy", "augmented"], required=True)
    parser.add_argument("--task", choices=["ad_mci", "ad_mci_ctl"], required=True)
    parser.add_argument("--arch", choices=["1l2h", "2l2h"], required=True)
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--device", choices=["cuda", "cpu", "mps"], default="cuda")
    parser.add_argument("--python-exe", default=sys.executable)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--pretrained-checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--skip-preprocessing", action="store_true")
    parser.add_argument("--force-preprocessing", action="store_true")
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--early-stopping-patience", type=int, default=45)
    parser.add_argument("--cv-seed", type=int, default=42)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--class-weighting", choices=["on", "off"], default="off")
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--storage", default="", help="Optional Optuna storage URL. Defaults to per-combo SQLite.")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else (ROOT / path).resolve()


def run(cmd: list[str]) -> None:
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    args = parse_args()
    args.data_dir = resolve(args.data_dir)
    args.result_root = resolve(args.result_root)
    args.pretrained_checkpoint = resolve(args.pretrained_checkpoint)
    if not args.pretrained_checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint required by finetune_txt.py not found: {args.pretrained_checkpoint}")

    x_file = args.data_dir / f"X_{args.dataset}_{args.task}.csv"
    y_file = args.data_dir / f"y_{args.dataset}_{args.task}.csv"
    if not args.skip_preprocessing:
        preprocess_cmd = [
            args.python_exe,
            "-u",
            str(PREPROCESSOR),
            "--dataset",
            args.dataset,
            "--task",
            args.task,
            "--output-dir",
            str(args.data_dir),
        ]
        if args.force_preprocessing:
            preprocess_cmd.append("--force")
        run(preprocess_cmd)

    if not x_file.exists() or not y_file.exists():
        raise FileNotFoundError(
            f"Missing processed files for {args.dataset}/{args.task}. "
            f"Expected {x_file} and {y_file}. Run without --skip-preprocessing first."
        )

    n_layers = 1 if args.arch == "1l2h" else 2
    combo_name = f"{args.dataset}_{args.task}_{args.arch}"
    combo_root = args.result_root / combo_name
    combo_root.mkdir(parents=True, exist_ok=True)

    config = {
        "dataset": args.dataset,
        "task": args.task,
        "arch": args.arch,
        "x_file": str(x_file),
        "y_file": str(y_file),
        "result_root": str(combo_root),
        "transfer_mode": "random_init",
        "objective_metric": "val_auc_f1_50_50",
        "n_trials": args.n_trials,
        "n_folds": args.n_folds,
        "model": {
            "n_layers": n_layers,
            "n_heads": 2,
            "d_model": 128,
            "d_head": 64,
            "d_ff": 128,
            "d_embed": 128,
            "aggfunc": "Avgpool",
            "d_hidden1": 128,
            "d_hidden2": 64,
        },
        "search_space": {
            "lr_low": 1e-4,
            "lr_high": 5e-4,
            "dropout_options": [0.2, 0.3, 0.4, 0.5],
            "batch_size": [8, 16, 32],
            "weight_decay": 1e-4,
        },
    }
    (combo_root / "combo_config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    optuna_cmd = [
        args.python_exe,
        "-u",
        str(OPTUNA_CV),
        "--python-exe",
        args.python_exe,
        "--device",
        args.device,
        "--pretrained-checkpoint",
        str(args.pretrained_checkpoint),
        "--x-file",
        str(x_file),
        "--y-file",
        str(y_file),
        "--result-root",
        str(combo_root),
        "--study-name",
        combo_name,
        "--n-trials",
        str(args.n_trials),
        "--n-folds",
        str(args.n_folds),
        "--cv-seed",
        str(args.cv_seed),
        "--seed",
        str(args.seed),
        "--transfer-modes",
        "random_init",
        "--dataset-mode",
        "deg",
        "--max-genes",
        "0",
        "--scaler",
        "minmax",
        "--scaler-fit-scope",
        "train",
        "--epochs",
        str(args.epochs),
        "--early-stopping-patience",
        str(args.early_stopping_patience),
        "--objective-metric",
        "val_auc_f1_50_50",
        "--checkpoint-metric",
        "val_auc_f1_50_50",
        "--class-weighting",
        args.class_weighting,
        "--label-smoothing",
        str(args.label_smoothing),
        "--n-layers",
        str(n_layers),
        "--n-heads",
        "2",
        "--d-model",
        "128",
        "--d-ff",
        "128",
        "--d-hidden1",
        "128",
        "--d-hidden2",
        "64",
        "--aggfunc",
        "Avgpool",
        "--lr-low",
        "0.0001",
        "--lr-high",
        "0.0005",
        "--dropout-options",
        "0.2",
        "0.3",
        "0.4",
        "0.5",
    ]
    if args.storage.strip():
        optuna_cmd.extend(["--storage", args.storage.strip()])
    run(optuna_cmd)
    print(f"Completed combo: {combo_name}", flush=True)
    print(f"Outputs: {combo_root}", flush=True)


if __name__ == "__main__":
    main()
