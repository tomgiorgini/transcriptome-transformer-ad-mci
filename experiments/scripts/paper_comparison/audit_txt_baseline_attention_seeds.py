#!/usr/bin/env python3
"""Audit regenerated baseline TxT seeds and optional key-attention exports.

The audit is intentionally read-only with respect to training and attention
artifacts.  It writes only a compact per-seed CSV and a JSON report to the
requested output directory, and returns a non-zero exit status if any requested
seed fails a check.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split


DEFAULT_SEEDS = list(range(101, 111))
EXPECTED_TASKS = ("AD_vs_MCI", "AD_vs_CTL", "MCI_vs_CTL")
EXPECTED_SPLITS = ("train", "val", "test")
EXPECTED_SPLIT_COUNTS = {"train": 497, "val": 71, "test": 143}
EXPECTED_CLASS_COUNTS = {
    "train": {"Control": 166, "MCI": 132, "AD": 199},
    "val": {"Control": 24, "MCI": 19, "AD": 28},
    "test": {"Control": 48, "MCI": 38, "AD": 57},
}
EXPECTED_LABELS = {0: "Control", 1: "MCI", 2: "AD"}
EXPECTED_STATE_KEYS = 63
EXPECTED_STATE_VALUES = 230_092

REQUIRED_RUN_FILES = (
    "args.json",
    "best_model.pt",
    "ensemble_checkpoint_rank1.pt",
    "training_log.csv",
    "metrics_summary.csv",
    "selected_genes.csv",
    "gene_selection_details.csv",
    "gene_embedding.csv",
    "trained_gene_embedding.csv",
    "model_summary.json",
    "embedding_manifest.json",
    "augmentation_manifest.json",
    "label_mask_summary.csv",
    "train.log",
)

REQUIRED_ATTENTION_FILES = (
    "key_attention_by_seed.csv",
    "key_attention_by_subject.npz",
    "key_attention_qc.csv",
    "key_attention_manifest.json",
)

EXPECTED_PROTOCOL: dict[str, Any] = {
    "model_variant": "baseline",
    "post_model_construction_reseed": "off",
    "tupe_mode": "on",
    "grad_clip_scope": "joint",
    "n_layers": 1,
    "n_heads": 2,
    "d_model": 64,
    "embed_dim": 64,
    "d_ff": 256,
    "d_hidden1": 128,
    "d_hidden2": 64,
    "dropout": 0.2,
    "norm_first": False,
    "aggfunc": "Avgpool",
    "batch_size": 9,
    "train_sampling": "balanced_classes",
    "epochs": 200,
    "early_stopping_patience": 0,
    "val_loss_stop_threshold": 1.0,
    "val_loss_stop_patience": 5,
    "lr_encoder": 0.0001,
    "lr_head": 0.0001,
    "lr_embedding": 0.0001,
    "freeze_embedding_epochs": 0,
    "weight_decay": 0.0001,
    "class_weighting": "off",
    "task_loss_weights": [0.5, 0.25, 0.25],
    "label_smoothing": 0.0,
    "max_genes": 2000,
    "gene_selection": "variance",
    "ad_mci_gene_fraction": 0.5,
    "feature_selection_fit_scope": "train",
    "scaler": "minmax",
    "scaler_fit_scope": "train",
    "pooling_mode": "average",
    "task_specific_pooling": "off",
    "head_norm": "batch",
    "mask_aware_heads": "on",
    "encoder_sharing": "shared",
    "gradient_strategy": "weighted_sum",
    "gradient_diagnostics": "off",
    "gradient_diagnostic_interval": 1,
    "grad_clip_norm": 1.0,
    "primary_adapter_dim": 0,
    "expression_residual": "none",
    "checkpoint_ensemble_size": 1,
    "checkpoint_ensemble_min_gap": 3,
    "checkpoint_metric": "val_weighted_70_15_15_auc_minus_025_loss",
    "augmentation": "none",
    "evaluate_test": "on",
    "evaluate_test_each_epoch": "off",
    "skip_final_test": False,
    "split_mode": "custom",
    "split_seed": None,
    "val_ratio": 0.1,
    "test_ratio": 0.2,
    "max_train_batches": None,
    "max_val_batches": None,
    "embed_file": None,
    "candidate_gene_file": None,
    "embedding_gene_policy": "all",
    "embedding_rescale": "none",
    "embedding_init_scale": 0.02,
    "ppi_integration": "direct",
    "ppi_gate_init": 0.0,
}


@dataclass
class SeedAudit:
    seed: int
    run_dir: str = ""
    status: str = "FAIL"
    attention_status: str = "not_present"
    checks: int = 0
    failed_checks: int = 0
    epochs_ran: int | None = None
    best_epoch: int | None = None
    selected_genes: int | None = None
    test_samples: int | None = None
    attention_samples: int | None = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def expect(self, condition: bool, message: str) -> None:
        self.checks += 1
        if not bool(condition):
            self.failed_checks += 1
            self.errors.append(message)

    def fail(self, message: str) -> None:
        self.expect(False, message)

    def finish(self) -> None:
        self.status = "PASS" if not self.errors else "FAIL"

    def csv_row(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "status": self.status,
            "attention_status": self.attention_status,
            "checks": self.checks,
            "failed_checks": self.failed_checks,
            "epochs_ran": self.epochs_ran,
            "best_epoch": self.best_epoch,
            "selected_genes": self.selected_genes,
            "test_samples": self.test_samples,
            "attention_samples": self.attention_samples,
            "run_dir": self.run_dir,
            "errors": " | ".join(self.errors),
            "warnings": " | ".join(self.warnings),
        }


@dataclass
class SeedContext:
    args: dict[str, Any]
    y: pd.DataFrame
    split: pd.DataFrame
    selected_genes: list[str]
    class_counts: dict[str, dict[str, int]]
    training_log: pd.DataFrame | None = None
    best_epoch: int | None = None
    best_checkpoint_value: float | None = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit baseline TxT seed runs and any key-only test-attention exports. "
            "Returns 1 if one or more requested seeds fail."
        )
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--expected-device", choices=["cpu", "cuda", "mps"], default="mps")
    parser.add_argument("--attention-split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}.")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def gene_order_sha256(genes: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(map(str, genes)).encode("utf-8")).hexdigest()


def values_equal(observed: Any, expected: Any) -> bool:
    if isinstance(expected, float):
        try:
            return math.isclose(float(observed), expected, rel_tol=1e-12, abs_tol=1e-12)
        except (TypeError, ValueError):
            return False
    if isinstance(expected, list):
        if not isinstance(observed, list) or len(observed) != len(expected):
            return False
        return all(values_equal(left, right) for left, right in zip(observed, expected))
    return observed == expected


def resolved_equal(left: str | Path, right: str | Path) -> bool:
    try:
        return Path(left).expanduser().resolve() == Path(right).expanduser().resolve()
    except (OSError, TypeError, ValueError):
        return False


def discover_seed_dir(seed_root: Path, seed: int, audit: SeedAudit) -> Path | None:
    matches = sorted(
        path.resolve()
        for path in seed_root.glob(f"*/seed_{seed}")
        if path.is_dir()
    )
    audit.expect(
        len(matches) == 1,
        f"expected exactly one run directory for seed {seed}, found {len(matches)}: {matches}",
    )
    return matches[0] if len(matches) == 1 else None


def audit_required_files(run_dir: Path, audit: SeedAudit) -> bool:
    complete = True
    for name in REQUIRED_RUN_FILES:
        path = run_dir / name
        ok = path.is_file() and path.stat().st_size > 0
        audit.expect(ok, f"missing or empty run artifact: {path}")
        complete &= ok
    for forbidden in ("ppi_graph_manifest.json", "induced_ppi_edges.csv"):
        path = run_dir / forbidden
        audit.expect(not path.exists(), f"baseline run unexpectedly contains VMA/PPI graph artifact: {path}")
    return complete


def audit_protocol(
    run_dir: Path,
    expected_split_path: Path,
    seed: int,
    expected_device: str,
    audit: SeedAudit,
) -> dict[str, Any] | None:
    try:
        args = read_json(run_dir / "args.json")
    except Exception as exc:
        audit.fail(f"cannot read args.json: {type(exc).__name__}: {exc}")
        return None

    expected = {**EXPECTED_PROTOCOL, "seed": seed, "device": expected_device, "device_resolved": expected_device}
    for key, value in expected.items():
        audit.expect(key in args, f"args.json missing protocol field {key!r}")
        if key in args:
            audit.expect(
                values_equal(args[key], value),
                f"protocol mismatch {key}: expected {value!r}, found {args[key]!r}",
            )

    audit.expect(
        resolved_equal(args.get("result_dir", ""), run_dir),
        f"args.result_dir does not resolve to {run_dir}",
    )
    for key in ("split_file", "resolved_split_file"):
        audit.expect(
            resolved_equal(args.get(key, ""), expected_split_path),
            f"args.{key} does not resolve to {expected_split_path}",
        )
    for key in ("x_file", "y_file", "ppi_edge_file"):
        value = args.get(key)
        audit.expect(value is not None, f"args.{key} is missing")
        if value is not None:
            path = Path(value).expanduser()
            audit.expect(path.is_file() and path.stat().st_size > 0, f"args.{key} is missing/empty: {path}")

    rng = args.get("rng_pairing_policy")
    audit.expect(isinstance(rng, dict), "args.rng_pairing_policy is missing or invalid")
    if isinstance(rng, dict):
        audit.expect(rng.get("seed") == seed, "rng_pairing_policy seed mismatch")
        audit.expect(rng.get("model_variant") == "baseline", "rng_pairing_policy is not baseline")
        audit.expect(rng.get("post_model_construction_reseed") is False, "legacy RNG policy is not active")
        audit.expect(rng.get("grad_clip_scope") == "joint", "legacy joint clipping policy is not active")
    return args


def deterministic_split_map(y: pd.DataFrame, seed: int) -> dict[str, str]:
    labels = y["label"].to_numpy()
    all_idx = np.arange(len(y), dtype=np.int64)
    train_idx, holdout_idx = train_test_split(
        all_idx,
        test_size=0.3,
        random_state=seed,
        stratify=labels,
    )
    val_idx, test_idx = train_test_split(
        holdout_idx,
        test_size=2.0 / 3.0,
        random_state=seed + 1000,
        stratify=labels[holdout_idx],
    )
    sample_ids = y["sample_id"].astype(str).to_numpy()
    mapping: dict[str, str] = {}
    for split_name, indices in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
        mapping.update({str(sample_ids[index]): split_name for index in indices})
    return mapping


def audit_split(
    args: dict[str, Any],
    expected_split_path: Path,
    seed: int,
    audit: SeedAudit,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, int]]] | None:
    try:
        y = pd.read_csv(Path(args["y_file"]), dtype={"sample_id": str})
        split = pd.read_csv(expected_split_path, dtype={"sample_id": str})
    except Exception as exc:
        audit.fail(f"cannot read y/split CSV: {type(exc).__name__}: {exc}")
        return None

    audit.expect({"sample_id", "label", "label_name"}.issubset(y.columns), "y.csv lacks required columns")
    audit.expect({"sample_id", "split"}.issubset(split.columns), "split CSV lacks required columns")
    if not {"sample_id", "label", "label_name"}.issubset(y.columns) or not {"sample_id", "split"}.issubset(split.columns):
        return None

    y["sample_id"] = y["sample_id"].astype(str)
    split["sample_id"] = split["sample_id"].astype(str)
    audit.expect(len(y) == 711, f"expected 711 labels, found {len(y)}")
    audit.expect(y["sample_id"].is_unique, "y.csv sample IDs are not unique")
    audit.expect(len(split) == 711, f"expected 711 split rows, found {len(split)}")
    audit.expect(split["sample_id"].is_unique, "split sample IDs are not unique")
    audit.expect(set(split["split"]) == set(EXPECTED_SPLITS), f"unexpected split labels: {sorted(set(split['split']))}")
    audit.expect(set(split["sample_id"]) == set(y["sample_id"]), "split sample coverage differs from y.csv")

    observed_label_map = (
        y[["label", "label_name"]]
        .drop_duplicates()
        .set_index("label")["label_name"]
        .to_dict()
    )
    audit.expect(observed_label_map == EXPECTED_LABELS, f"unexpected biological label map: {observed_label_map}")
    observed_split_counts = split["split"].value_counts().to_dict()
    audit.expect(observed_split_counts == EXPECTED_SPLIT_COUNTS, f"unexpected split counts: {observed_split_counts}")

    try:
        merged = split.merge(y, on="sample_id", how="left", validate="one_to_one")
    except Exception as exc:
        audit.fail(f"cannot align split with labels: {type(exc).__name__}: {exc}")
        return None
    class_counts: dict[str, dict[str, int]] = {}
    for split_name in EXPECTED_SPLITS:
        counts = (
            merged.loc[merged["split"] == split_name, "label_name"]
            .value_counts()
            .reindex(["Control", "MCI", "AD"], fill_value=0)
            .astype(int)
            .to_dict()
        )
        class_counts[split_name] = counts
        audit.expect(
            counts == EXPECTED_CLASS_COUNTS[split_name],
            f"unexpected {split_name} class counts: {counts}",
        )

    observed_map = split.set_index("sample_id")["split"].to_dict()
    expected_map = deterministic_split_map(y, seed)
    audit.expect(observed_map == expected_map, "split mapping differs from deterministic seed protocol")
    audit.test_samples = observed_split_counts.get("test")
    return y, split, class_counts


def read_gene_embedding(path: Path, genes: Sequence[str], dim: int, label: str, audit: SeedAudit) -> None:
    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        audit.fail(f"cannot read {label}: {type(exc).__name__}: {exc}")
        return
    audit.expect(frame.shape == (len(genes), dim + 1), f"{label} shape is {frame.shape}, expected {(len(genes), dim + 1)}")
    audit.expect("Gene" in frame.columns, f"{label} lacks Gene column")
    if "Gene" not in frame.columns:
        return
    observed_genes = frame["Gene"].astype(str).tolist()
    audit.expect(observed_genes == list(genes), f"{label} gene order differs from selected_genes.csv")
    values = frame.drop(columns=["Gene"]).apply(pd.to_numeric, errors="coerce").to_numpy(float)
    audit.expect(np.isfinite(values).all(), f"{label} contains non-finite/non-numeric values")


def audit_genes_and_embeddings(
    run_dir: Path,
    args: dict[str, Any],
    audit: SeedAudit,
) -> list[str] | None:
    try:
        selected = pd.read_csv(run_dir / "selected_genes.csv")
    except Exception as exc:
        audit.fail(f"cannot read selected_genes.csv: {type(exc).__name__}: {exc}")
        return None
    audit.expect(list(selected.columns) == ["gene"], f"unexpected selected_genes columns: {selected.columns.tolist()}")
    if "gene" not in selected.columns:
        return None
    genes = selected["gene"].astype(str).tolist()
    expected_genes = int(args["max_genes"])
    audit.selected_genes = len(genes)
    audit.expect(len(genes) == expected_genes, f"expected {expected_genes} selected genes, found {len(genes)}")
    audit.expect(len(set(genes)) == len(genes), "selected genes are not unique")
    audit.expect(all(gene and gene.lower() != "nan" for gene in genes), "selected genes contain empty/NaN names")

    try:
        x_columns = pd.read_csv(Path(args["x_file"]), nrows=0).columns.astype(str).tolist()
        x_genes = set(x_columns) - {"sample_id"}
        audit.expect("sample_id" in x_columns, "X.csv lacks sample_id")
        audit.expect(set(genes).issubset(x_genes), "selected genes are not a subset of X.csv columns")
    except Exception as exc:
        audit.fail(f"cannot validate selected genes against X.csv: {type(exc).__name__}: {exc}")

    read_gene_embedding(run_dir / "gene_embedding.csv", genes, int(args["embed_dim"]), "gene_embedding.csv", audit)
    read_gene_embedding(
        run_dir / "trained_gene_embedding.csv",
        genes,
        int(args["embed_dim"]),
        "trained_gene_embedding.csv",
        audit,
    )

    try:
        details = pd.read_csv(run_dir / "gene_selection_details.csv")
        audit.expect(len(details) == len(genes), "gene_selection_details row count differs from selected genes")
        audit.expect("gene" in details.columns, "gene_selection_details lacks gene")
        if "gene" in details.columns:
            audit.expect(details["gene"].astype(str).tolist() == genes, "gene_selection_details gene order differs")
        if "selection_order" in details.columns:
            order = pd.to_numeric(details["selection_order"], errors="coerce").to_numpy(float)
            audit.expect(np.array_equal(order, np.arange(1, len(genes) + 1)), "selection_order is not 1..G")
        audit.expect("fit_scope" in details.columns, "gene_selection_details lacks fit_scope")
        if "fit_scope" in details.columns:
            audit.expect(set(details["fit_scope"].astype(str)) == {"train"}, "gene selection was not fit only on train")
    except Exception as exc:
        audit.fail(f"cannot validate gene_selection_details.csv: {type(exc).__name__}: {exc}")

    try:
        model_summary = read_json(run_dir / "model_summary.json")
        audit.expect(model_summary.get("model") == "txt_multitask", "model_summary model mismatch")
        audit.expect(model_summary.get("model_variant") == "baseline", "model_summary is not baseline")
        audit.expect(model_summary.get("num_genes") == len(genes), "model_summary gene count mismatch")
        audit.expect(model_summary.get("ppi_graph") is None, "baseline model_summary unexpectedly contains PPI graph")
        architecture = model_summary.get("architecture", {})
        audit.expect(isinstance(architecture, dict), "model_summary architecture is invalid")
        if isinstance(architecture, dict):
            for key in ("n_layers", "n_heads", "d_model", "d_ff", "model_variant", "tupe_mode"):
                audit.expect(
                    values_equal(architecture.get(key), args.get(key)),
                    f"model_summary architecture mismatch for {key}",
                )
            counts = architecture.get("parameter_counts", {})
            audit.expect(counts.get("total") == 228_934, f"unexpected trainable parameter total: {counts.get('total')}")
    except Exception as exc:
        audit.fail(f"cannot validate model_summary.json: {type(exc).__name__}: {exc}")
    return genes


def load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, dict) or not value:
        raise TypeError(f"{path} is not a non-empty state dictionary")
    if not all(isinstance(key, str) and torch.is_tensor(tensor) for key, tensor in value.items()):
        raise TypeError(f"{path} contains non-tensor state entries")
    return value


def audit_checkpoints(run_dir: Path, audit: SeedAudit) -> None:
    try:
        best = load_state_dict(run_dir / "best_model.pt")
        rank1 = load_state_dict(run_dir / "ensemble_checkpoint_rank1.pt")
    except Exception as exc:
        audit.fail(f"cannot load checkpoints: {type(exc).__name__}: {exc}")
        return
    audit.expect(len(best) == EXPECTED_STATE_KEYS, f"checkpoint has {len(best)} state keys, expected {EXPECTED_STATE_KEYS}")
    value_count = sum(int(tensor.numel()) for tensor in best.values())
    audit.expect(value_count == EXPECTED_STATE_VALUES, f"checkpoint has {value_count} state values, expected {EXPECTED_STATE_VALUES}")
    audit.expect(all(torch.isfinite(tensor).all().item() for tensor in best.values()), "checkpoint contains non-finite tensors")
    audit.expect(
        not any("volum" in key.lower() or "augmentation" in key.lower() for key in best),
        "baseline checkpoint contains VMA/augmentation parameters",
    )
    expected_core = {
        "transformer.encoder.tupe.q_linear.weight",
        "transformer.encoder.tupe.k_linear.weight",
        "transformer.encoder.layers.0.multi_head_attention_layer.q_linear.weight",
        "transformer.encoder.layers.0.multi_head_attention_layer.k_linear.weight",
    }
    audit.expect(expected_core.issubset(best), "checkpoint lacks expected TxT/TUPE attention parameters")
    same = best.keys() == rank1.keys() and all(torch.equal(best[key], rank1[key]) for key in best)
    audit.expect(same, "ensemble rank-1 checkpoint is not tensor-identical to best_model.pt")


def audit_training_and_metrics(
    run_dir: Path,
    args: dict[str, Any],
    class_counts: dict[str, dict[str, int]],
    audit: SeedAudit,
) -> tuple[pd.DataFrame | None, int | None, float | None]:
    training: pd.DataFrame | None = None
    best_epoch: int | None = None
    best_value: float | None = None
    try:
        training = pd.read_csv(run_dir / "training_log.csv")
        required = {
            "epoch",
            "val_loss",
            "checkpoint_metric",
            "checkpoint_value",
            "val_AD_vs_MCI_roc_auc",
            "val_AD_vs_CTL_roc_auc",
            "val_MCI_vs_CTL_roc_auc",
        }
        audit.expect(required.issubset(training.columns), f"training_log lacks columns: {sorted(required - set(training.columns))}")
        audit.expect(len(training) >= 1, "training_log is empty")
        if len(training):
            epochs = pd.to_numeric(training["epoch"], errors="coerce").to_numpy(float)
            audit.expect(np.array_equal(epochs, np.arange(1, len(training) + 1)), "training epochs are not consecutive 1..N")
            audit.expect(len(training) <= int(args["epochs"]), "training exceeded configured epoch limit")
            audit.epochs_ran = len(training)
        numeric = training.select_dtypes(include=[np.number]).to_numpy(float)
        audit.expect(np.isfinite(numeric).all(), "training_log contains non-finite numeric values")
        if required.issubset(training.columns) and len(training):
            formula = (
                0.70 * training["val_AD_vs_MCI_roc_auc"].to_numpy(float)
                + 0.15 * training["val_AD_vs_CTL_roc_auc"].to_numpy(float)
                + 0.15 * training["val_MCI_vs_CTL_roc_auc"].to_numpy(float)
                - 0.25 * training["val_loss"].to_numpy(float)
            )
            observed = training["checkpoint_value"].to_numpy(float)
            audit.expect(np.allclose(formula, observed, rtol=1e-12, atol=1e-12), "checkpoint score formula mismatch")
            audit.expect(
                set(training["checkpoint_metric"].astype(str)) == {args["checkpoint_metric"]},
                "training_log checkpoint metric mismatch",
            )
            best_index = int(np.argmax(observed))
            best_epoch = int(training.iloc[best_index]["epoch"])
            best_value = float(observed[best_index])
            audit.best_epoch = best_epoch
        if len(training) < int(args["epochs"]):
            patience = int(args["val_loss_stop_patience"])
            threshold = float(args["val_loss_stop_threshold"])
            audit.expect(len(training) >= patience, "early-stopped run is shorter than loss-stop patience")
            if len(training) >= patience:
                audit.expect(
                    bool((training["val_loss"].tail(patience).to_numpy(float) > threshold).all()),
                    "early stop did not end after the configured consecutive val-loss exceedances",
                )
    except Exception as exc:
        audit.fail(f"cannot validate training_log.csv: {type(exc).__name__}: {exc}")

    try:
        log_text = (run_dir / "train.log").read_text(encoding="utf-8", errors="replace")
        audit.expect("Multitask TxT complete." in log_text, "train.log lacks completion marker")
        if training is not None and len(training) < int(args["epochs"]):
            marker = rf"Early stopping at epoch {len(training)}: val_loss_threshold_patience"
            audit.expect(re.search(marker, log_text) is not None, "train.log lacks matching loss-stop marker")
    except Exception as exc:
        audit.fail(f"cannot validate train.log: {type(exc).__name__}: {exc}")

    audit_checkpoints(run_dir, audit)

    try:
        metrics = pd.read_csv(run_dir / "metrics_summary.csv")
        required_columns = {
            "split",
            "task",
            "samples",
            "loss",
            "accuracy",
            "macro_f1",
            "weighted_f1",
            "balanced_accuracy",
            "roc_auc",
        }
        audit.expect(required_columns.issubset(metrics.columns), f"metrics_summary lacks columns: {sorted(required_columns - set(metrics.columns))}")
        expected_pairs = {(split, task) for split in EXPECTED_SPLITS for task in EXPECTED_TASKS}
        observed_pairs = set(zip(metrics.get("split", []), metrics.get("task", [])))
        audit.expect(len(metrics) == 9 and observed_pairs == expected_pairs, "metrics_summary is not exactly 3 tasks x 3 splits")
        audit.expect(not metrics.duplicated(["split", "task"]).any(), "metrics_summary contains duplicate split/task rows")
        numeric_columns = [
            "samples",
            "loss",
            "accuracy",
            "macro_f1",
            "weighted_f1",
            "balanced_accuracy",
            "roc_auc",
        ]
        numeric = metrics[numeric_columns].apply(pd.to_numeric, errors="coerce")
        audit.expect(np.isfinite(numeric.to_numpy(float)).all(), "metrics_summary contains non-finite values")
        audit.expect(bool((numeric["loss"] >= 0).all()), "metrics_summary contains negative losses")
        for key in ("accuracy", "macro_f1", "weighted_f1", "balanced_accuracy", "roc_auc"):
            audit.expect(bool(numeric[key].between(0, 1).all()), f"metrics_summary {key} lies outside [0,1]")

        for _, row in metrics.iterrows():
            split_name = str(row["split"])
            task = str(row["task"])
            if split_name not in class_counts or task not in EXPECTED_TASKS:
                continue
            left, right = task.split("_vs_")
            name_map = {"CTL": "Control", "MCI": "MCI", "AD": "AD"}
            expected_samples = class_counts[split_name][name_map[left]] + class_counts[split_name][name_map[right]]
            audit.expect(
                int(row["samples"]) == expected_samples,
                f"metrics sample count mismatch for {split_name}/{task}: {row['samples']} vs {expected_samples}",
            )
        if training is not None and best_epoch is not None:
            best_row = training.loc[training["epoch"] == best_epoch].iloc[0]
            for task in EXPECTED_TASKS:
                observed_auc = float(
                    metrics.loc[(metrics["split"] == "val") & (metrics["task"] == task), "roc_auc"].iloc[0]
                )
                audit.expect(
                    math.isclose(observed_auc, float(best_row[f"val_{task}_roc_auc"]), rel_tol=1e-10, abs_tol=1e-10),
                    f"final validation AUC for {task} does not match best checkpoint epoch",
                )
    except Exception as exc:
        audit.fail(f"cannot validate metrics_summary.csv: {type(exc).__name__}: {exc}")

    try:
        summary = read_json(run_dir / "model_summary.json")
        training_summary = summary.get("training_summary", {})
        audit.expect(isinstance(training_summary, dict), "model_summary training_summary is invalid")
        if isinstance(training_summary, dict) and best_epoch is not None and best_value is not None:
            audit.expect(training_summary.get("epochs_ran") == len(training), "model_summary epochs_ran mismatch")
            audit.expect(training_summary.get("best_epoch") == best_epoch, "model_summary best_epoch mismatch")
            audit.expect(
                math.isclose(float(training_summary.get("best_checkpoint_value", math.nan)), best_value, rel_tol=1e-12, abs_tol=1e-12),
                "model_summary best checkpoint value mismatch",
            )
            if len(training) < int(args["epochs"]):
                audit.expect(training_summary.get("stopped_early") is True, "model_summary does not mark early stop")
                audit.expect(training_summary.get("stop_reason") == "val_loss_threshold_patience", "model_summary stop reason mismatch")
    except Exception as exc:
        audit.fail(f"cannot validate model_summary training metadata: {type(exc).__name__}: {exc}")
    return training, best_epoch, best_value


def descending_ranks(values: np.ndarray) -> np.ndarray:
    order = np.lexsort((np.arange(len(values)), -np.asarray(values, dtype=np.float64)))
    ranks = np.empty(len(values), dtype=np.int64)
    ranks[order] = np.arange(1, len(values) + 1)
    return ranks


def audit_attention(
    run_dir: Path,
    split_name: str,
    context: SeedContext,
    audit: SeedAudit,
) -> None:
    failures_before_attention = audit.failed_checks
    attention_dir = run_dir / "key_attention" / split_name
    present = [(attention_dir / name).exists() for name in REQUIRED_ATTENTION_FILES]
    if not any(present):
        audit.attention_status = "not_present"
        return
    audit.attention_status = "FAIL"
    for name, exists in zip(REQUIRED_ATTENTION_FILES, present):
        path = attention_dir / name
        audit.expect(exists and path.is_file() and path.stat().st_size > 0, f"partial/empty attention export: {path}")
    if not all(present):
        return

    try:
        manifest = read_json(attention_dir / "key_attention_manifest.json")
        expected_manifest = {
            "seed": audit.seed,
            "model_variant": "baseline",
            "split": split_name,
            "samples": EXPECTED_SPLIT_COUNTS[split_name],
            "genes": len(context.selected_genes),
            "heads": int(context.args["n_heads"]),
            "vma_included": False,
            "query_metrics_included": False,
        }
        for key, value in expected_manifest.items():
            audit.expect(values_equal(manifest.get(key), value), f"attention manifest mismatch {key}: {manifest.get(key)!r}")
        audit.expect(resolved_equal(manifest.get("run_dir", ""), run_dir), "attention manifest run_dir mismatch")
        audit.expect(
            resolved_equal(manifest.get("checkpoint", ""), run_dir / "best_model.pt"),
            "attention manifest checkpoint path mismatch",
        )
        checkpoint_hash = sha256_file(run_dir / "best_model.pt")
        audit.expect(manifest.get("checkpoint_sha256") == checkpoint_hash, "attention checkpoint SHA-256 mismatch")
        audit.expect(
            manifest.get("gene_order_sha256") == gene_order_sha256(context.selected_genes),
            "attention gene-order SHA-256 mismatch",
        )
        expected_class_counts = context.class_counts[split_name]
        audit.expect(manifest.get("class_counts") == expected_class_counts, "attention manifest class counts mismatch")
        audit.expect(float(manifest.get("max_row_sum_abs_error", math.inf)) <= 1e-4, "attention row-normalization QC exceeds 1e-4")
        audit.expect(
            float(manifest.get("max_incoming_gene_mean_abs_error", math.inf)) <= 1e-4,
            "attention incoming-mean QC exceeds 1e-4",
        )
        hashes = manifest.get("artifact_sha256", {})
        audit.expect(isinstance(hashes, dict), "attention artifact_sha256 is invalid")
        if isinstance(hashes, dict):
            artifact_paths = {
                "args": run_dir / "args.json",
                "x": Path(context.args["x_file"]),
                "y": Path(context.args["y_file"]),
                "split": Path(context.args["split_file"]),
                "selected_genes": run_dir / "selected_genes.csv",
                "constructor_embedding": run_dir / "gene_embedding.csv",
            }
            for key, path in artifact_paths.items():
                audit.expect(hashes.get(key) == sha256_file(path), f"attention provenance hash mismatch for {key}")
    except Exception as exc:
        audit.fail(f"cannot validate attention manifest: {type(exc).__name__}: {exc}")
        return

    try:
        with np.load(attention_dir / "key_attention_by_subject.npz", allow_pickle=False) as data:
            required_keys = {
                "incoming_enrichment",
                "sample_ids",
                "labels",
                "class_names",
                "gene_names",
                "class_mean_incoming_enrichment",
                "axis_semantics",
            }
            audit.expect(required_keys.issubset(data.files), f"attention NPZ lacks keys: {sorted(required_keys - set(data.files))}")
            if not required_keys.issubset(data.files):
                return
            incoming_raw = np.asarray(data["incoming_enrichment"])
            incoming = incoming_raw.astype(np.float64, copy=False)
            sample_ids = np.asarray(data["sample_ids"]).astype(str)
            labels = np.asarray(data["labels"], dtype=np.int64)
            class_names = np.asarray(data["class_names"]).astype(str)
            gene_names = np.asarray(data["gene_names"]).astype(str)
            saved_class_means = np.asarray(data["class_mean_incoming_enrichment"], dtype=np.float64)
            axis_semantics = str(np.asarray(data["axis_semantics"]).item())
    except Exception as exc:
        audit.fail(f"cannot read attention NPZ: {type(exc).__name__}: {exc}")
        return

    expected_shape = (
        EXPECTED_SPLIT_COUNTS[split_name],
        int(context.args["n_heads"]),
        len(context.selected_genes),
    )
    audit.attention_samples = int(incoming.shape[0]) if incoming.ndim else None
    audit.expect(incoming.shape == expected_shape, f"incoming attention shape is {incoming.shape}, expected {expected_shape}")
    audit.expect(sample_ids.ndim == 1, f"attention sample_ids must be 1-D, found shape {sample_ids.shape}")
    audit.expect(labels.ndim == 1, f"attention labels must be 1-D, found shape {labels.shape}")
    audit.expect(class_names.ndim == 1, f"attention class_names must be 1-D, found shape {class_names.shape}")
    audit.expect(gene_names.ndim == 1, f"attention gene_names must be 1-D, found shape {gene_names.shape}")
    audit.expect(np.isfinite(incoming).all(), "incoming attention contains non-finite values")
    audit.expect(bool((incoming >= 0).all()), "incoming attention contains negative values")
    if incoming.shape == expected_shape:
        audit.expect(
            np.allclose(incoming.mean(axis=-1), 1.0, rtol=0.0, atol=1e-4),
            "incoming attention does not have mean-gene enrichment 1 per subject/head",
        )
    audit.expect(axis_semantics == "sample,head,key_gene", f"unexpected attention axis semantics: {axis_semantics}")
    if sample_ids.ndim != 1 or labels.ndim != 1 or class_names.ndim != 1 or gene_names.ndim != 1:
        return
    audit.expect(gene_names.tolist() == context.selected_genes, "attention NPZ gene order mismatch")
    audit.expect(len(sample_ids) == len(set(sample_ids.tolist())), "attention NPZ sample IDs are not unique")

    split_ids = set(context.split.loc[context.split["split"] == split_name, "sample_id"].astype(str))
    audit.expect(set(sample_ids.tolist()) == split_ids, "attention NPZ sample IDs differ from target split")
    expected_class_names = [EXPECTED_LABELS[index] for index in sorted(EXPECTED_LABELS)]
    audit.expect(class_names.tolist() == expected_class_names, f"attention class names mismatch: {class_names.tolist()}")
    y_labels = context.y.set_index("sample_id")["label"].astype(int).to_dict()
    expected_labels = np.asarray([y_labels.get(sample_id, -1) for sample_id in sample_ids], dtype=np.int64)
    audit.expect(np.array_equal(labels, expected_labels), "attention NPZ labels are not aligned to sample IDs")
    counts = {
        class_names[index]: int((labels == index).sum())
        for index in range(len(class_names))
    }
    audit.expect(counts == context.class_counts[split_name], f"attention NPZ class counts mismatch: {counts}")

    if incoming.shape != expected_shape or len(class_names) != len(EXPECTED_LABELS):
        return
    mean_heads = incoming.mean(axis=1)
    class_means = np.stack([mean_heads[labels == index].mean(axis=0) for index in range(len(class_names))])
    macro_mean = class_means.mean(axis=0)
    pooled_mean = mean_heads.mean(axis=0)
    audit.expect(saved_class_means.shape == class_means.shape, "saved attention class means shape mismatch")
    if saved_class_means.shape == class_means.shape:
        audit.expect(np.allclose(saved_class_means, class_means, rtol=2e-6, atol=2e-6), "saved attention class means mismatch")

    try:
        ranking = pd.read_csv(attention_dir / "key_attention_by_seed.csv")
        required_columns = {
            "gene",
            "incoming_macro_mean",
            "incoming_pooled_mean",
            "key_rank",
            "key_percentile_score",
            "incoming_head_0_macro_mean",
            "incoming_head_1_macro_mean",
            "key_head_0_rank",
            "key_head_1_rank",
        }
        audit.expect(required_columns.issubset(ranking.columns), f"attention ranking lacks columns: {sorted(required_columns - set(ranking.columns))}")
        audit.expect(len(ranking) == len(context.selected_genes), "attention ranking gene count mismatch")
        audit.expect(ranking["gene"].astype(str).is_unique, "attention ranking genes are not unique")
        forbidden_columns = [column for column in ranking.columns if "query" in column.lower() or "vma" in column.lower()]
        audit.expect(not forbidden_columns, f"key-only ranking contains query/VMA columns: {forbidden_columns}")
        indexed = ranking.assign(gene=ranking["gene"].astype(str)).set_index("gene").reindex(context.selected_genes)
        audit.expect(not indexed.index.has_duplicates and not indexed.isna().all(axis=1).any(), "attention ranking does not cover selected genes")
        if required_columns.issubset(ranking.columns) and not indexed.isna().all(axis=1).any():
            audit.expect(np.allclose(indexed["incoming_macro_mean"].to_numpy(float), macro_mean, rtol=2e-6, atol=2e-6), "ranking macro incoming values mismatch NPZ")
            audit.expect(np.allclose(indexed["incoming_pooled_mean"].to_numpy(float), pooled_mean, rtol=2e-6, atol=2e-6), "ranking pooled incoming values mismatch NPZ")
            expected_ranks = descending_ranks(macro_mean)
            audit.expect(np.array_equal(indexed["key_rank"].to_numpy(int), expected_ranks), "ranking key ranks mismatch NPZ")
            denominator = max(len(macro_mean) - 1, 1)
            expected_percentiles = 1.0 - (expected_ranks - 1.0) / denominator
            audit.expect(np.allclose(indexed["key_percentile_score"].to_numpy(float), expected_percentiles, rtol=1e-12, atol=1e-12), "ranking percentile scores mismatch ranks")
            for head in range(int(context.args["n_heads"])):
                head_macro = np.stack(
                    [
                        incoming_raw[labels == index, head].mean(axis=0)
                        for index in range(len(class_names))
                    ]
                ).mean(axis=0)
                audit.expect(
                    np.allclose(indexed[f"incoming_head_{head}_macro_mean"].to_numpy(float), head_macro, rtol=2e-6, atol=2e-6),
                    f"head {head} macro incoming values mismatch NPZ",
                )
                audit.expect(
                    np.array_equal(indexed[f"key_head_{head}_rank"].to_numpy(int), descending_ranks(head_macro)),
                    f"head {head} ranks mismatch NPZ",
                )
    except Exception as exc:
        audit.fail(f"cannot validate attention ranking: {type(exc).__name__}: {exc}")

    try:
        qc = pd.read_csv(attention_dir / "key_attention_qc.csv")
        required_qc = {
            "samples",
            "row_sum_max_abs_error",
            "incoming_gene_mean_max_abs_error",
            "attention_nonfinite",
        }
        audit.expect(required_qc.issubset(qc.columns), f"attention QC lacks columns: {sorted(required_qc - set(qc.columns))}")
        if required_qc.issubset(qc.columns):
            audit.expect(int(qc["samples"].sum()) == len(sample_ids), "attention QC sample total mismatch")
            audit.expect(bool((qc["row_sum_max_abs_error"] <= 1e-4).all()), "attention QC row errors exceed 1e-4")
            audit.expect(bool((qc["incoming_gene_mean_max_abs_error"] <= 1e-4).all()), "attention QC incoming errors exceed 1e-4")
            audit.expect(int(qc["attention_nonfinite"].sum()) == 0, "attention QC reports non-finite weights")
    except Exception as exc:
        audit.fail(f"cannot validate attention QC: {type(exc).__name__}: {exc}")

    if audit.failed_checks == failures_before_attention:
        audit.attention_status = "PASS"


def audit_seed(
    seed_root: Path,
    seed: int,
    expected_device: str,
    attention_split: str,
) -> SeedAudit:
    audit = SeedAudit(seed=seed)
    run_dir = discover_seed_dir(seed_root, seed, audit)
    if run_dir is None:
        audit.finish()
        return audit
    audit.run_dir = str(run_dir)
    expected_split_path = (seed_root / "splits" / f"seed_{seed}.csv").resolve()
    audit.expect(
        expected_split_path.is_file() and expected_split_path.stat().st_size > 0,
        f"missing or empty split file: {expected_split_path}",
    )
    files_complete = audit_required_files(run_dir, audit)
    args = audit_protocol(run_dir, expected_split_path, seed, expected_device, audit) if (run_dir / "args.json").exists() else None
    split_payload = audit_split(args, expected_split_path, seed, audit) if args is not None and expected_split_path.exists() else None
    genes = audit_genes_and_embeddings(run_dir, args, audit) if args is not None and (run_dir / "selected_genes.csv").exists() else None

    if args is not None and split_payload is not None and files_complete:
        y, split, class_counts = split_payload
        training, best_epoch, best_value = audit_training_and_metrics(run_dir, args, class_counts, audit)
        if genes is not None:
            context = SeedContext(
                args=args,
                y=y,
                split=split,
                selected_genes=genes,
                class_counts=class_counts,
                training_log=training,
                best_epoch=best_epoch,
                best_checkpoint_value=best_value,
            )
            audit_attention(run_dir, attention_split, context, audit)
    audit.finish()
    return audit


def main(argv: Sequence[str] | None = None) -> int:
    cli = build_parser().parse_args(argv)
    seeds = [int(seed) for seed in cli.seeds]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("--seeds must contain one or more unique integers")
    run_root = cli.run_root.expanduser().resolve()
    if not run_root.is_dir():
        raise FileNotFoundError(f"Run root does not exist: {run_root}")
    seed_root = run_root if run_root.name == "seeds10" else run_root / "seeds10"
    if not seed_root.is_dir():
        raise FileNotFoundError(f"Seed root does not exist: {seed_root}")
    output_dir = (
        cli.output_dir.expanduser().resolve()
        if cli.output_dir is not None
        else run_root / "audit_baseline_attention"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    audits = [
        audit_seed(seed_root, seed, cli.expected_device, cli.attention_split)
        for seed in seeds
    ]
    csv_path = output_dir / "seed_audit_status.csv"
    json_path = output_dir / "seed_audit_status.json"
    pd.DataFrame([audit.csv_row() for audit in audits]).to_csv(csv_path, index=False)
    failures = [audit for audit in audits if audit.status != "PASS"]
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_root": str(run_root),
        "seed_root": str(seed_root),
        "expected_device": cli.expected_device,
        "attention_split": cli.attention_split,
        "requested_seeds": seeds,
        "overall_status": "PASS" if not failures else "FAIL",
        "passed_seeds": [audit.seed for audit in audits if audit.status == "PASS"],
        "failed_seeds": [audit.seed for audit in failures],
        "status_csv": str(csv_path),
        "results": [asdict(audit) for audit in audits],
    }
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"TxT baseline seed audit: {report['overall_status']} ({len(audits) - len(failures)}/{len(audits)} passed)")
    print(f"CSV:  {csv_path}")
    print(f"JSON: {json_path}")
    for audit in failures:
        print(f"seed {audit.seed}: FAIL ({audit.failed_checks}/{audit.checks} checks failed)", file=sys.stderr)
        for message in audit.errors:
            print(f"  - {message}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
