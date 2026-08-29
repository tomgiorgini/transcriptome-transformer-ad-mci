#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.scripts.paper_comparison.txt_volumetric.common import (  # noqa: E402
    BETA_CANDIDATES,
    CHECKPOINT_METRIC,
    DEFAULT_PPI_EDGE_FILE,
    DEFAULT_RESULT_ROOT,
    DEFAULT_SHARED_DATASET,
    DEFAULT_WORKER,
    PROTOCOL_SEEDS,
    ProtocolError,
    baseline_dir,
    baseline_test_dir,
    beta_run_name,
    build_resume_fingerprint,
    candidate_dir,
    ensemble_checkpoint_paths,
    evaluation_command_from_selection,
    inspect_run_artifacts,
    metrics_has_split,
    read_json,
    resolve_path,
    run_worker,
    save_worker_command,
    seed_dir,
    select_beta_for_seed,
    selection_checkpoint_paths,
    selected_beta_path,
    selected_test_dir,
    sha256_file,
    validate_resume_fingerprint,
    write_beta_selection,
    write_json,
    write_summary_artifacts,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the paired TxT baseline/PPI-volumetric protocol. VMA candidates are trained "
            "without test evaluation; beta is selected from validation only, and only the selected "
            "checkpoint is subsequently evaluated on test."
        )
    )
    parser.add_argument("--python-exe", default=sys.executable)
    parser.add_argument("--worker", type=Path, default=DEFAULT_WORKER)
    parser.add_argument("--x-file", type=Path, default=DEFAULT_SHARED_DATASET / "X.csv")
    parser.add_argument("--y-file", type=Path, default=DEFAULT_SHARED_DATASET / "y.csv")
    parser.add_argument("--ppi-edge-file", type=Path, default=DEFAULT_PPI_EDGE_FILE)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(PROTOCOL_SEEDS))
    parser.add_argument("--betas", nargs="+", type=float, default=list(BETA_CANDIDATES))
    parser.add_argument(
        "--beta-policy",
        choices=["auto", "global_fixed", "per_seed_validation_argmax"],
        default="auto",
        help=(
            "auto uses a singleton beta globally and treats --primary-beta as globally fixed "
            "for the complete VMA v2 profile; historical multi-beta profiles retain per-seed "
            "validation argmax."
        ),
    )
    parser.add_argument("--primary-beta", type=float, default=1.0)
    parser.add_argument("--ppi-score-threshold", type=float, default=0.73)
    parser.add_argument("--volumetric-eps", type=float, default=1e-8)
    parser.add_argument("--volumetric-gate-init", type=float, default=0.0)
    parser.add_argument(
        "--volumetric-volume-mode",
        choices=["raw", "l2"],
        default="raw",
    )
    parser.add_argument(
        "--volumetric-dropout",
        type=float,
        default=None,
        help="VMA-only attention dropout; omit to inherit the model dropout.",
    )
    parser.add_argument(
        "--volumetric-message-mode",
        choices=["legacy", "expression_contrast"],
        default="legacy",
    )
    parser.add_argument(
        "--volumetric-output-norm",
        choices=["none", "rms"],
        default="none",
    )
    parser.add_argument(
        "--volumetric-gate-mode",
        choices=["scalar", "per_head"],
        default="scalar",
    )
    parser.add_argument(
        "--volumetric-backbone-gradient-mode",
        choices=["coupled", "detached"],
        default="coupled",
    )
    parser.add_argument(
        "--tupe-mode",
        choices=["on", "off"],
        default="on",
        help="Enable TUPE or run the paired exact-zero TUPE ablation in both baseline and VMA.",
    )

    parser.add_argument("--device", choices=["cuda", "cpu", "mps"], default="cuda")
    parser.add_argument(
        "--post-model-construction-reseed",
        choices=["on", "off"],
        default="on",
        help="Keep paired baseline/VMA training RNG aligned after variant-specific construction.",
    )
    parser.add_argument(
        "--grad-clip-scope",
        choices=["joint", "separate_volumetric"],
        default="joint",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--early-stopping-patience", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-volumetric", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-genes", type=int, default=2000)
    parser.add_argument("--checkpoint-ensemble-size", type=int, default=1)
    parser.add_argument("--checkpoint-ensemble-min-gap", type=int, default=3)
    parser.add_argument(
        "--baseline-evaluate-test",
        choices=["on", "off"],
        default="off",
        help=(
            "Whether the branch-free baseline may access test during training. The leakage-safe "
            "default is off; the final phase evaluates its frozen checkpoint separately."
        ),
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260718)

    parser.add_argument(
        "--phase",
        choices=["all", "train", "select-evaluate", "summarize"],
        default="all",
        help="Run the whole resumable protocol or only one stage.",
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Validation-only CPU smoke: first seed, one epoch, two train/validation batches, "
            "baseline plus one VMA candidate. It never runs final test evaluation."
        ),
    )
    parser.add_argument("--smoke-beta", type=float, default=1.0)
    parser.add_argument("--smoke-max-genes", type=int, default=64)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("--seeds must contain at least one unique seed and no duplicates.")
    if not args.betas or any(not math.isfinite(beta) or beta < 0 for beta in args.betas):
        raise ValueError("--betas requires finite, non-negative values.")
    if len(set(float(beta) for beta in args.betas)) != len(args.betas):
        raise ValueError("--betas must not contain duplicates.")
    if not math.isfinite(args.primary_beta) or args.primary_beta < 0:
        raise ValueError("--primary-beta must be finite and non-negative.")
    if args.ppi_score_threshold < 0:
        raise ValueError("--ppi-score-threshold must be non-negative.")
    if args.volumetric_eps <= 0 or not math.isfinite(args.volumetric_eps):
        raise ValueError("--volumetric-eps must be finite and positive.")
    if not math.isfinite(args.volumetric_gate_init):
        raise ValueError("--volumetric-gate-init must be finite.")
    volumetric_dropout = getattr(args, "volumetric_dropout", None)
    if volumetric_dropout is not None and (
        not math.isfinite(volumetric_dropout)
        or not 0.0 <= volumetric_dropout <= 1.0
    ):
        raise ValueError("--volumetric-dropout must be finite and in [0, 1].")
    if args.epochs <= 0 or args.early_stopping_patience < 0 or args.max_genes <= 0:
        raise ValueError("epochs/max-genes must be positive and patience must be non-negative.")
    lr_volumetric = getattr(args, "lr_volumetric", None)
    if lr_volumetric is not None and (
        not math.isfinite(lr_volumetric) or lr_volumetric <= 0
    ):
        raise ValueError("--lr-volumetric must be finite and positive when provided.")
    if args.checkpoint_ensemble_size <= 0 or args.checkpoint_ensemble_min_gap < 0:
        raise ValueError("checkpoint ensemble size must be positive and min gap non-negative.")
    if args.smoke_max_genes <= 0:
        raise ValueError("--smoke-max-genes must be positive.")
    if args.smoke and getattr(args, "baseline_evaluate_test", "off") != "off":
        raise ValueError("--smoke requires --baseline-evaluate-test off.")
    if args.smoke and args.phase not in {"all", "train"}:
        raise ValueError("--smoke supports only --phase all or --phase train.")


def is_vma_v2_profile(args: argparse.Namespace) -> bool:
    return bool(
        getattr(args, "volumetric_volume_mode", "raw") == "l2"
        and getattr(args, "volumetric_message_mode", "legacy") == "expression_contrast"
        and getattr(args, "volumetric_output_norm", "none") == "rms"
        and getattr(args, "volumetric_gate_mode", "scalar") == "per_head"
        and getattr(args, "volumetric_backbone_gradient_mode", "coupled") == "detached"
    )


def resolve_beta_policy(
    args: argparse.Namespace,
    protocol: dict[str, Any],
) -> tuple[str, float | None]:
    betas = [float(beta) for beta in protocol["betas"]]
    requested = getattr(args, "beta_policy", "auto")
    if requested == "auto":
        if len(betas) == 1:
            return "global_fixed", betas[0]
        if is_vma_v2_profile(args) and float(args.primary_beta) in betas:
            return "global_fixed", float(args.primary_beta)
        return "per_seed_validation_argmax", None
    if requested == "global_fixed":
        primary_beta = float(args.primary_beta)
        if primary_beta not in betas:
            raise ValueError(
                f"--primary-beta {primary_beta:g} must be present in --betas for global_fixed."
            )
        return "global_fixed", primary_beta
    return "per_seed_validation_argmax", None


def effective_protocol(args: argparse.Namespace) -> dict[str, Any]:
    if args.smoke:
        return {
            "seeds": [int(args.seeds[0])],
            "betas": [float(args.smoke_beta)],
            "device": "cpu",
            "epochs": 1,
            "max_genes": int(min(args.max_genes, args.smoke_max_genes)),
            "max_train_batches": 2,
            "max_val_batches": 2,
            "smoke": True,
        }
    return {
        "seeds": [int(seed) for seed in args.seeds],
        "betas": [float(beta) for beta in args.betas],
        "device": args.device,
        "epochs": int(args.epochs),
        "max_genes": int(args.max_genes),
        "max_train_batches": None,
        "max_val_batches": None,
        "smoke": False,
    }


def load_labels(y_file: Path) -> pd.DataFrame:
    frame = pd.read_csv(y_file)
    required = {"sample_id", "label"}
    if not required.issubset(frame.columns):
        raise ValueError(f"{y_file} must contain columns {sorted(required)}.")
    result = frame[["sample_id", "label"]].copy()
    result["sample_id"] = result["sample_id"].astype(str)
    if result["sample_id"].duplicated().any():
        raise ValueError(f"{y_file} contains duplicate sample_id values.")
    return result


def build_seed_split(y_df: pd.DataFrame, seed: int) -> pd.DataFrame:
    indices = y_df.index.to_numpy()
    labels = y_df["label"].to_numpy()
    train_idx, holdout_idx = train_test_split(
        indices,
        test_size=0.30,
        random_state=int(seed),
        stratify=labels,
    )
    val_idx, test_idx = train_test_split(
        holdout_idx,
        test_size=2.0 / 3.0,
        random_state=int(seed) + 1000,
        stratify=labels[holdout_idx],
    )
    split_by_index = {
        **{int(index): "train" for index in train_idx},
        **{int(index): "val" for index in val_idx},
        **{int(index): "test" for index in test_idx},
    }
    rows = [
        {"sample_id": row.sample_id, "split": split_by_index[int(index)]}
        for index, row in y_df.iterrows()
    ]
    return pd.DataFrame(rows)


def seed_split_manifest(split: pd.DataFrame, path: Path, seed: int) -> dict[str, Any]:
    counts = split["split"].value_counts().to_dict()
    return {
        "seed": int(seed),
        "file": str(path),
        "sha256": sha256_file(path),
        "counts": {name: int(counts.get(name, 0)) for name in ("train", "val", "test")},
        "ratios": {name: float(counts.get(name, 0) / len(split)) for name in ("train", "val", "test")},
    }


def write_seed_split(y_df: pd.DataFrame, path: Path, seed: int) -> dict[str, Any]:
    split = build_seed_split(y_df, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    split.to_csv(path, index=False)
    return seed_split_manifest(split, path, seed)


def ensure_splits(args: argparse.Namespace, protocol: dict[str, Any]) -> dict[int, Path]:
    result_root = resolve_path(args.result_root)
    y_file = resolve_path(args.y_file)
    labels = load_labels(y_file)
    paths: dict[int, Path] = {}
    manifests = []
    for seed in protocol["seeds"]:
        path = result_root / "splits" / f"seed_{seed}.csv"
        expected_split = build_seed_split(labels, seed)
        if args.skip_existing and path.exists():
            observed_split = pd.read_csv(path, dtype={"sample_id": str, "split": str})
            expected_for_compare = expected_split.astype({"sample_id": str, "split": str})
            if not observed_split.equals(expected_for_compare):
                raise ProtocolError(
                    f"Refusing --skip-existing: stored split differs from the deterministic "
                    f"seed={seed} split for the current y.csv: {path}"
                )
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            expected_split.to_csv(path, index=False)
        manifests.append(seed_split_manifest(expected_split, path, seed))
        paths[int(seed)] = path
    split_manifest = {
        "source_y": str(y_file),
        "source_y_sha256": sha256_file(y_file),
        "splits": manifests,
    }
    manifest_path = result_root / "split_manifest.json"
    if args.skip_existing and manifest_path.exists():
        observed_manifest = read_json(manifest_path)
        observed_semantic = dict(observed_manifest) if isinstance(observed_manifest, dict) else {}
        observed_semantic.pop("source_y", None)
        expected_semantic = dict(split_manifest)
        expected_semantic.pop("source_y", None)
        for manifest in (observed_semantic, expected_semantic):
            for entry in manifest.get("splits", []):
                if isinstance(entry, dict):
                    entry["file"] = f"<seed-{entry.get('seed')}-split>"
        if observed_semantic != expected_semantic:
            raise ProtocolError(
                "Refusing --skip-existing: split_manifest.json does not match the current "
                "labels and deterministic splits. Use a new result root."
            )
    else:
        write_json(manifest_path, split_manifest)
    return paths


def build_worker_command(
    args: argparse.Namespace,
    protocol: dict[str, Any],
    *,
    seed: int,
    split_file: Path,
    result_dir: Path,
    model_variant: str,
    beta: float | None,
    evaluate_test: str,
) -> list[str]:
    command = [
        args.python_exe,
        "-u",
        str(resolve_path(args.worker)),
        "--x-file", str(resolve_path(args.x_file)),
        "--y-file", str(resolve_path(args.y_file)),
        "--split-mode", "custom",
        "--split-file", str(split_file),
        "--result-dir", str(result_dir),
        "--seed", str(seed),
        "--post-model-construction-reseed", getattr(
            args, "post_model_construction_reseed", "on"
        ),
        "--max-genes", str(protocol["max_genes"]),
        "--gene-selection", "variance",
        "--feature-selection-fit-scope", "train",
        "--scaler", "minmax",
        "--scaler-fit-scope", "train",
        "--batch-size", "9",
        "--train-sampling", "balanced_classes",
        "--epochs", str(protocol["epochs"]),
        "--early-stopping-patience", str(args.early_stopping_patience),
        "--val-loss-stop-threshold", "1.0",
        "--val-loss-stop-patience", "5",
        "--lr-encoder", str(args.lr),
        "--lr-head", str(args.lr),
        "--lr-embedding", str(args.lr),
        "--freeze-embedding-epochs", "0",
        "--weight-decay", str(args.weight_decay),
        "--class-weighting", "off",
        "--task-loss-weights", "0.5", "0.25", "0.25",
        "--label-smoothing", "0.0",
        "--augmentation", "none",
        "--checkpoint-metric", CHECKPOINT_METRIC,
        "--checkpoint-ensemble-size", str(getattr(args, "checkpoint_ensemble_size", 1)),
        "--checkpoint-ensemble-min-gap", str(
            getattr(args, "checkpoint_ensemble_min_gap", 3)
        ),
        "--evaluate-test", evaluate_test,
        "--evaluate-test-each-epoch", "off",
        "--device", str(protocol["device"]),
        "--n-heads", "2",
        "--n-layers", "1",
        "--d-model", "64",
        "--embed-dim", "64",
        "--d-ff", "256",
        "--dropout", "0.2",
        "--aggfunc", "Avgpool",
        "--pooling-mode", "average",
        "--task-specific-pooling", "off",
        "--head-norm", "batch",
        "--mask-aware-heads", "on",
        "--encoder-sharing", "shared",
        "--gradient-strategy", "weighted_sum",
        "--gradient-diagnostics", "off",
        "--primary-adapter-dim", "0",
        "--expression-residual", "none",
        "--tupe-mode", getattr(args, "tupe_mode", "on"),
        "--d-hidden1", "128",
        "--d-hidden2", "64",
        "--grad-clip-norm", "1.0",
        "--grad-clip-scope", getattr(args, "grad_clip_scope", "joint"),
        "--embedding-gene-policy", "all",
        "--embedding-rescale", "none",
        "--embedding-init-scale", "0.02",
        "--ppi-integration", "direct",
        "--model-variant", model_variant,
    ]
    if model_variant == "ppi_volumetric":
        if beta is None:
            raise ValueError("The PPI-volumetric worker command requires beta.")
        command.extend(
            [
                "--ppi-edge-file", str(resolve_path(args.ppi_edge_file)),
                "--ppi-score-threshold", str(args.ppi_score_threshold),
                "--volumetric-beta", str(float(beta)),
                "--volumetric-eps", str(args.volumetric_eps),
                "--volumetric-gate-init", str(args.volumetric_gate_init),
                "--volumetric-volume-mode", getattr(args, "volumetric_volume_mode", "raw"),
                "--volumetric-message-mode", getattr(args, "volumetric_message_mode", "legacy"),
                "--volumetric-output-norm", getattr(args, "volumetric_output_norm", "none"),
                "--volumetric-gate-mode", getattr(args, "volumetric_gate_mode", "scalar"),
                "--volumetric-backbone-gradient-mode", getattr(
                    args, "volumetric_backbone_gradient_mode", "coupled"
                ),
            ]
        )
        volumetric_dropout = getattr(args, "volumetric_dropout", None)
        if volumetric_dropout is not None:
            command.extend(["--volumetric-dropout", str(volumetric_dropout)])
        if getattr(args, "lr_volumetric", None) is not None:
            command.extend(["--lr-volumetric", str(args.lr_volumetric)])
    if protocol["max_train_batches"] is not None:
        command.extend(["--max-train-batches", str(protocol["max_train_batches"])])
    if protocol["max_val_batches"] is not None:
        command.extend(["--max-val-batches", str(protocol["max_val_batches"])])
    return command


def protocol_manifest(args: argparse.Namespace, protocol: dict[str, Any]) -> dict[str, Any]:
    ppi_file = resolve_path(args.ppi_edge_file)
    beta_policy, global_beta = resolve_beta_policy(args, protocol)
    if is_vma_v2_profile(args):
        protocol_variant = (
            "vma_v2_expression_contrast_no_tupe"
            if getattr(args, "tupe_mode", "on") == "off"
            else "vma_v2_expression_contrast"
        )
    else:
        protocol_variant = "legacy_or_custom"
    fixed = {
        "feature_selection": "top variance, train only",
        "max_genes": protocol["max_genes"],
        "embedding": "random d64",
        "architecture": {
            "layers": 1,
            "heads": 2,
            "d_model": 64,
            "d_ff": 256,
            "head_hidden_dims": [128, 64],
            "head_norm": "batch",
            "dropout": 0.2,
            "tupe_mode": getattr(args, "tupe_mode", "on"),
        },
        "batch": {"size": 9, "sampling": "balanced_classes", "samples_per_biological_class": 3},
        "pooling": "average",
        "mask_aware_class_only_heads": True,
        "task_loss_weights": [0.5, 0.25, 0.25],
        "split": {"train": 0.70, "validation": 0.10, "test": 0.20},
        "checkpoint_metric": CHECKPOINT_METRIC,
        "checkpoint_ensemble": {
            "size": int(getattr(args, "checkpoint_ensemble_size", 1)),
            "min_gap": int(getattr(args, "checkpoint_ensemble_min_gap", 3)),
            "aggregation": "arithmetic_mean_softmax_probabilities",
        },
    }
    volumetric_dropout = getattr(args, "volumetric_dropout", None)
    effective_vma_dropout = float(
        fixed["architecture"]["dropout"]
        if volumetric_dropout is None
        else volumetric_dropout
    )
    reseed_enabled = getattr(args, "post_model_construction_reseed", "on") == "on"
    grad_clip_scope = getattr(args, "grad_clip_scope", "joint")
    gate_init = float(args.volumetric_gate_init)
    exact_zero_gate_first_step_pairing = bool(
        reseed_enabled
        and effective_vma_dropout == 0.0
        and gate_init == 0.0
        and grad_clip_scope == "separate_volumetric"
    )
    pairing_caveats = []
    if not reseed_enabled:
        pairing_caveats.append("post_model_construction_reseed_disabled")
    if effective_vma_dropout != 0.0:
        pairing_caveats.append("volumetric_dropout_consumes_training_rng")
    if gate_init != 0.0:
        pairing_caveats.append("volumetric_gate_not_zero_initialized")
    if grad_clip_scope != "separate_volumetric":
        pairing_caveats.append(
            "joint_gradient_clipping_couples_vma_and_shared_gradient_norms"
        )
    return {
        "protocol_schema_version": 2,
        "protocol": "TxT multitask with PPI Volumetric Attention",
        "protocol_variant": protocol_variant,
        "seeds": protocol["seeds"],
        "beta_candidates": protocol["betas"],
        "beta_policy": {
            "policy": beta_policy,
            "global_beta": global_beta,
            "control_betas": (
                [float(beta) for beta in protocol["betas"] if float(beta) != global_beta]
                if global_beta is not None
                else []
            ),
            "per_seed_selection": beta_policy == "per_seed_validation_argmax",
        },
        "beta_zero_interpretation": "PPI pair-wise control; not the branch-free baseline",
        "ppi": {
            "source_file": str(ppi_file),
            "source_sha256": sha256_file(ppi_file) if ppi_file.exists() else None,
            "inclusive_score_threshold": float(args.ppi_score_threshold),
        },
        "volumetric": {
            "epsilon": float(args.volumetric_eps),
            "gate_init": gate_init,
            "volume_mode": getattr(args, "volumetric_volume_mode", "raw"),
            "dropout": volumetric_dropout,
            "dropout_effective": effective_vma_dropout,
            "dropout_policy": (
                "inherit_model_dropout"
                if volumetric_dropout is None
                else "explicit_vma_dropout"
            ),
            "message_mode": getattr(args, "volumetric_message_mode", "legacy"),
            "output_norm": getattr(args, "volumetric_output_norm", "none"),
            "gate_mode": getattr(args, "volumetric_gate_mode", "scalar"),
            "backbone_gradient_mode": getattr(
                args, "volumetric_backbone_gradient_mode", "coupled"
            ),
        },
        "fixed_configuration": fixed,
        "training": {
            "epochs": protocol["epochs"],
            "early_stopping_patience": int(args.early_stopping_patience),
            "val_loss_stop_threshold": 1.0,
            "val_loss_stop_patience": 5,
            "lr": float(args.lr),
            "lr_volumetric": (
                None if getattr(args, "lr_volumetric", None) is None else float(args.lr_volumetric)
            ),
            "weight_decay": float(args.weight_decay),
            "device": protocol["device"],
            "grad_clip_norm": 1.0,
            "grad_clip_scope": grad_clip_scope,
            "baseline_evaluate_test": getattr(args, "baseline_evaluate_test", "off"),
            "candidate_evaluate_test": "off",
            "rng_pairing_policy": {
                "post_model_construction_reseed": reseed_enabled,
                "reseed_point": (
                    "immediately_after_model_construction_and_device_transfer"
                    if reseed_enabled
                    else None
                ),
                "effective_vma_dropout": effective_vma_dropout,
                "grad_clip_scope": grad_clip_scope,
                "gate_init": gate_init,
                "exact_zero_gate_first_step_pairing": exact_zero_gate_first_step_pairing,
                "purpose": (
                    "exact_zero_gate_first_step_pairing"
                    if exact_zero_gate_first_step_pairing
                    else (
                        "post_construction_rng_reseed"
                        if reseed_enabled
                        else "legacy_rng_sequence_after_model_construction"
                    )
                ),
                "caveats": pairing_caveats,
            },
        },
        "smoke": protocol["smoke"],
        "smoke_limits": {
            "max_train_batches": protocol["max_train_batches"],
            "max_val_batches": protocol["max_val_batches"],
        },
    }


def _protocol_manifest_digest(payload: dict[str, Any]) -> str:
    semantic = json.loads(json.dumps(payload))
    semantic.pop("resume_fingerprint", None)
    ppi = semantic.get("ppi")
    if isinstance(ppi, dict) and "source_file" in ppi:
        ppi["source_file"] = "<ppi-edge-file>"
    encoded = json.dumps(semantic, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_or_validate_protocol_manifest(
    path: Path,
    payload: dict[str, Any],
    *,
    skip_existing: bool,
) -> None:
    expected_digest = _protocol_manifest_digest(payload)
    payload = {
        **payload,
        "resume_fingerprint": {
            "schema_version": 1,
            "semantic_sha256": expected_digest,
        },
    }
    if skip_existing and path.exists():
        observed = read_json(path)
        if not isinstance(observed, dict):
            raise ProtocolError(f"Cannot safely reuse malformed protocol manifest: {path}")
        observed_digest = _protocol_manifest_digest(observed)
        if observed_digest != expected_digest:
            raise ProtocolError(
                "Refusing --skip-existing: protocol_manifest.json does not match the requested "
                f"protocol (stored={observed_digest}, requested={expected_digest}). Use a new "
                "result root."
            )
        return
    write_json(path, payload)


def _fingerprint_inputs(args: argparse.Namespace, split_file: Path) -> dict[str, Path]:
    return {
        "x_file": resolve_path(args.x_file),
        "y_file": resolve_path(args.y_file),
        "split_file": resolve_path(split_file),
        "ppi_edge_file": resolve_path(args.ppi_edge_file),
    }


def _directory_has_content(path: Path) -> bool:
    return path.exists() and any(path.iterdir())


def run_or_reuse(
    args: argparse.Namespace,
    *,
    run_dir: Path,
    command: Sequence[str],
    kind: str,
    seed: int,
    beta: float | None,
    complete: bool,
    split_file: Path,
    checkpoint_files: Sequence[Path] = (),
) -> bool:
    """Run a worker or safely reuse an exactly matching complete run."""

    fingerprint = build_resume_fingerprint(
        command,
        kind="evaluation" if checkpoint_files else "training",
        input_files=_fingerprint_inputs(args, split_file),
        checkpoint_files=checkpoint_files,
    )
    if args.skip_existing and _directory_has_content(run_dir):
        validate_resume_fingerprint(run_dir, fingerprint)
        if complete:
            print(f"Reusing fingerprint-matched run: {run_dir}", flush=True)
            return True
    save_worker_command(
        run_dir,
        command,
        kind=kind,
        seed=seed,
        beta=beta,
        resume_fingerprint=fingerprint,
    )
    run_worker(command, run_dir, dry_run=args.dry_run)
    return False


def train_stage(args: argparse.Namespace, protocol: dict[str, Any], split_files: dict[int, Path]) -> None:
    result_root = resolve_path(args.result_root)
    for seed in protocol["seeds"]:
        base_dir = baseline_dir(result_root, seed)
        base_command = build_worker_command(
            args,
            protocol,
            seed=seed,
            split_file=split_files[seed],
            result_dir=base_dir,
            model_variant="baseline",
            beta=None,
            evaluate_test=getattr(args, "baseline_evaluate_test", "off"),
        )
        baseline_complete = inspect_run_artifacts(
            base_dir,
            variant="baseline",
            require_test=getattr(args, "baseline_evaluate_test", "off") == "on",
        )["ok"]
        run_or_reuse(
            args,
            run_dir=base_dir,
            command=base_command,
            kind="baseline_train",
            seed=seed,
            beta=None,
            complete=baseline_complete,
            split_file=split_files[seed],
        )

        for beta in protocol["betas"]:
            run_dir = candidate_dir(result_root, seed, beta)
            command = build_worker_command(
                args,
                protocol,
                seed=seed,
                split_file=split_files[seed],
                result_dir=run_dir,
                model_variant="ppi_volumetric",
                beta=beta,
                evaluate_test="off",
            )
            candidate_complete = inspect_run_artifacts(
                run_dir,
                variant="ppi_volumetric",
                require_test=False,
            )["ok"]
            run_or_reuse(
                args,
                run_dir=run_dir,
                command=command,
                kind="beta_candidate_train_no_test",
                seed=seed,
                beta=beta,
                complete=candidate_complete,
                split_file=split_files[seed],
            )


def select_evaluate_stage(args: argparse.Namespace, protocol: dict[str, Any]) -> None:
    result_root = resolve_path(args.result_root)
    beta_policy, global_beta = resolve_beta_policy(args, protocol)
    for seed in protocol["seeds"]:
        selection = select_beta_for_seed(
            result_root,
            seed,
            protocol["betas"],
            fixed_beta=global_beta if beta_policy == "global_fixed" else None,
        )
        split_file = result_root / "splits" / f"seed_{seed}.csv"

        base_training_dir = baseline_dir(result_root, seed)
        base_checkpoint = base_training_dir / "best_model.pt"
        if not base_checkpoint.exists():
            raise ProtocolError(f"Missing baseline checkpoint for seed {seed}: {base_checkpoint}")
        base_ensemble = ensemble_checkpoint_paths(base_training_dir)
        base_evaluation_dir = baseline_test_dir(result_root, seed)
        base_command = evaluation_command_from_selection(
            {
                "selected_run_dir": str(base_training_dir),
                "selected_checkpoint": str(base_checkpoint),
                "selected_ensemble_checkpoints": [str(path) for path in base_ensemble],
            },
            base_evaluation_dir,
            python_exe=args.python_exe,
            worker=resolve_path(args.worker),
            device=str(protocol["device"]),
        )
        base_complete = inspect_run_artifacts(
            base_evaluation_dir,
            variant="baseline",
            require_test=True,
        )["ok"]
        run_or_reuse(
            args,
            run_dir=base_evaluation_dir,
            command=base_command,
            kind="baseline_checkpoint_test_evaluation",
            seed=seed,
            beta=None,
            complete=base_complete,
            split_file=split_file,
            checkpoint_files=base_ensemble,
        )

        evaluation_dir = selected_test_dir(result_root, seed)
        command = evaluation_command_from_selection(
            selection,
            evaluation_dir,
            python_exe=args.python_exe,
            worker=resolve_path(args.worker),
            device=str(protocol["device"]),
        )
        selected_checkpoints = selection_checkpoint_paths(selection)
        evaluation_complete = inspect_run_artifacts(
            evaluation_dir,
            variant="ppi_volumetric",
            require_test=True,
        )["ok"]
        run_or_reuse(
            args,
            run_dir=evaluation_dir,
            command=command,
            kind="selected_checkpoint_test_evaluation",
            seed=seed,
            beta=selection.selected_beta,
            complete=evaluation_complete,
            split_file=split_file,
            checkpoint_files=selected_checkpoints,
        )

        selection_file = write_beta_selection(result_root, selection)
        payload = read_json(selection_file)
        payload.update(
            {
                "baseline_evaluation_dir": str(base_evaluation_dir),
                "baseline_evaluation_command_file": str(
                    base_evaluation_dir / "worker_command.json"
                ),
                "baseline_ensemble_checkpoints": [str(path) for path in base_ensemble],
                "baseline_ensemble_size": len(base_ensemble),
                "baseline_ensemble_aggregation": "arithmetic_mean_softmax_probabilities",
                "evaluation_dir": str(evaluation_dir),
                "evaluation_command_file": str(evaluation_dir / "worker_command.json"),
            }
        )
        write_json(selection_file, payload)


def smoke_artifact_report(args: argparse.Namespace, protocol: dict[str, Any]) -> dict[str, Any]:
    result_root = resolve_path(args.result_root)
    checks = []
    for seed in protocol["seeds"]:
        baseline_check = inspect_run_artifacts(
            baseline_dir(result_root, seed),
            variant="baseline",
            require_test=False,
        )
        if baseline_check.get("test_metrics_present"):
            baseline_check["missing"].append("unexpected test rows in validation-only smoke")
            baseline_check["ok"] = False
        checks.append(baseline_check)
        for beta in protocol["betas"]:
            candidate_check = inspect_run_artifacts(
                candidate_dir(result_root, seed, beta),
                variant="ppi_volumetric",
                require_test=False,
            )
            if candidate_check.get("test_metrics_present"):
                candidate_check["missing"].append(
                    "unexpected test rows in validation-only smoke"
                )
                candidate_check["ok"] = False
            checks.append(candidate_check)
    return {
        "ok": all(check["ok"] for check in checks),
        "split_policy": "train_and_validation_only",
        "test_evaluated": False,
        "checks": checks,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    validate_args(args)
    protocol = effective_protocol(args)
    result_root = resolve_path(args.result_root)
    result_root.mkdir(parents=True, exist_ok=True)
    write_or_validate_protocol_manifest(
        result_root / "protocol_manifest.json",
        protocol_manifest(args, protocol),
        skip_existing=args.skip_existing,
    )

    if args.phase in {"all", "train"}:
        if not args.dry_run:
            for path in (resolve_path(args.x_file), resolve_path(args.y_file), resolve_path(args.worker), resolve_path(args.ppi_edge_file)):
                if not path.exists():
                    raise FileNotFoundError(path)
        split_files = ensure_splits(args, protocol)
        train_stage(args, protocol, split_files)
        if args.dry_run:
            print("Dry run complete; selection/evaluation requires completed candidate artifacts.", flush=True)
            return 0

    if not args.smoke and args.phase in {"all", "select-evaluate"}:
        select_evaluate_stage(args, protocol)

    if not args.smoke and args.phase in {"all", "summarize"}:
        paths = write_summary_artifacts(
            result_root,
            protocol["seeds"],
            betas=protocol["betas"],
            replicates=args.bootstrap_replicates,
            bootstrap_seed=args.bootstrap_seed,
        )
        print(json.dumps(paths, indent=2), flush=True)

    if args.smoke and args.phase in {"all", "train"}:
        report = smoke_artifact_report(args, protocol)
        write_json(result_root / "smoke_artifact_report.json", report)
        if not report["ok"]:
            raise ProtocolError(f"Smoke execution completed but artifact checks failed: {result_root / 'smoke_artifact_report.json'}")
    print(f"Paired TxT/PPI-volumetric protocol complete: {result_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
