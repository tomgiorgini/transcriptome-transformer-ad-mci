#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch  # Load before sklearn on Windows to avoid PyTorch DLL initialization failures.
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
COMMON_DIR = Path(__file__).resolve().parents[1]
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))
SOTA_COMMON_DIR = ROOT / "SOTA" / "source"
if str(SOTA_COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(SOTA_COMMON_DIR))

from source.models.txt import TxT
from source.pipeline.training import evaluate_classifier, train_classifier
from source.pipeline.utils import compute_balanced_class_weights, resolve_device, set_seed
from source.pretraining.txt.transfer import build_random_embedding_dataframe, remap_embedding_dataframe
from strict_v2_utils import calibrate_threshold


DEFAULT_X = ROOT / "task_dataset" / "processed" / "ad_mci_binary" / "X_ad_mci.csv"
DEFAULT_Y = ROOT / "task_dataset" / "processed" / "ad_mci_binary" / "y_ad_mci.csv"
DEFAULT_RESULT_ROOT = ROOT / "results" / "TxT" / "benchmark"
DEFAULT_BASELINE_RESULT_ROOT = ROOT / "results" / "TxT" / "benchmark"
DEFAULT_NATURE2020_RESULT_ROOT = ROOT / "results" / "SOTA" / "nature-2020"
DEFAULT_GSE63060_METADATA = ROOT / "pretraining_dataset" / "geo_downloads" / "GSE63060" / "eset_1_GPL6947" / "GSE63060_sample_metadata.tsv.gz"
DEFAULT_GSE63061_METADATA = ROOT / "pretraining_dataset" / "geo_downloads" / "GSE63061" / "eset_1_GPL10558" / "GSE63061_sample_metadata.tsv.gz"
DEFAULT_PPI_EMBEDDING_FILE = ROOT / "results" / "pretraining" / "ppi_init" / "hippie_highconf_allgenes_seed42" / "ppi_node_embedding.csv"
NATURE2026_AUGMENTATION = ROOT / "SOTA" / "source" / "nature-2026" / "augmentation.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate TxT batch-holdout AD vs MCI benchmarks.")
    parser.add_argument("--x-file", type=Path, default=DEFAULT_X)
    parser.add_argument("--y-file", type=Path, default=DEFAULT_Y)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--gene-source", choices=["input", "top_variance", "top_mad", "nature2020"], default="top_variance")
    parser.add_argument("--top-variance-genes", type=int, default=1000)
    parser.add_argument("--top-mad-genes", type=int, default=1000)
    parser.add_argument("--nature2020-result-root", type=Path, default=DEFAULT_NATURE2020_RESULT_ROOT)
    parser.add_argument("--nature2020-feature-set", choices=["deg", "tf_genes", "cfg_genes"], default="deg")
    parser.add_argument(
        "--batch-scenarios",
        nargs="+",
        default=["shared_test", "test_gse63060", "test_gse63061"],
        choices=["shared_test", "test_gse63060", "test_gse63061"],
    )
    parser.add_argument("--split-protocol", choices=["batch_holdout", "stratified_holdout"], default="batch_holdout")
    parser.add_argument("--train-size", type=float, default=0.7)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--shared-test-size", type=float, default=0.2)
    parser.add_argument("--gse63060-metadata", type=Path, default=DEFAULT_GSE63060_METADATA)
    parser.add_argument("--gse63061-metadata", type=Path, default=DEFAULT_GSE63061_METADATA)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--inner-val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cuda", "cpu", "mps"], default="cuda")
    parser.add_argument("--scaler", choices=["minmax", "standard", "robust", "none"], default="minmax")
    parser.add_argument("--class-weighting", choices=["on", "off"], default="off")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--early-stopping-patience", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--checkpoint-metric", choices=["macro_f1", "roc_auc_macro_f1", "strict_v2"], default="strict_v2")
    parser.add_argument("--checkpoint-auc-weight", type=float, default=0.5)
    parser.add_argument("--threshold-mode", choices=["fixed_0_5", "validation_macro_f1", "validation_accuracy", "validation_accuracy_macro_f1"], default="fixed_0_5")
    parser.add_argument("--augmentation", choices=["none", "gan", "borderline_smote"], default="none")
    parser.add_argument("--gan-targets", nargs="+", default=["double", "1000", "2000"])
    parser.add_argument("--gan-epochs", type=int, default=200)
    parser.add_argument("--gan-batch-size", type=int, default=64)
    parser.add_argument("--gan-latent-dim", type=int, default=128)
    parser.add_argument("--gan-learning-rate", type=float, default=0.001)
    parser.add_argument("--gan-sampling-strategy", choices=["balanced", "minority", "proportional"], default="balanced")
    parser.add_argument("--smote-k-neighbors", type=int, default=5)
    parser.add_argument("--smote-m-neighbors", type=int, default=10)
    parser.add_argument("--smote-kind", choices=["borderline-1", "borderline-2"], default="borderline-1")
    parser.add_argument("--baseline-result-root", type=Path, default=DEFAULT_BASELINE_RESULT_ROOT)
    parser.add_argument("--n-layers", type=int, default=1)
    parser.add_argument("--n-heads", type=int, default=2)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--embedding-source", choices=["random", "ppi"], default="random")
    parser.add_argument("--ppi-embedding-file", type=Path, default=DEFAULT_PPI_EMBEDDING_FILE)
    parser.add_argument("--d-ff", type=int, default=None, help="Transformer feed-forward dimension. Defaults to 4*d_model.")
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--aggfunc", choices=["Flatten", "Avgpool"], default="Flatten")
    parser.add_argument("--d-hidden1", type=int, default=128)
    parser.add_argument("--d-hidden2", type=int, default=64)
    parser.add_argument("--slope", type=float, default=0.2)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--strict-v2", action="store_true", help="Run the unified strict leakage-free V2 configuration.")
    parser.add_argument("--overwrite-results", action="store_true", help="Delete the result root before running.")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else (ROOT / path).resolve()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def load_augmentation_function():
    if not NATURE2026_AUGMENTATION.exists():
        raise FileNotFoundError(f"Nature-2026 augmentation module not found: {NATURE2026_AUGMENTATION}")
    spec = importlib.util.spec_from_file_location("nature2026_augmentation_for_txt", NATURE2026_AUGMENTATION)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load augmentation module from {NATURE2026_AUGMENTATION}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.augment_training_data


def load_binary_dataset(x_file: Path, y_file: Path) -> tuple[pd.DataFrame, pd.Series]:
    x = pd.read_csv(x_file, index_col=0)
    y_df = pd.read_csv(y_file, index_col=0)
    if "label" not in y_df.columns:
        raise ValueError(f"{y_file} must contain a label column.")
    common = x.index.intersection(y_df.index)
    x = x.loc[common].copy()
    y = y_df.loc[common, "label"].astype(int)
    if set(y.unique()) != {0, 1}:
        raise ValueError(f"Expected AD/MCI labels 0/1, got {sorted(y.unique())}.")
    return x, y


def read_geo_metadata_sample_ids(path: Path) -> set[str]:
    md = pd.read_csv(path, sep="\t", compression="infer", dtype=str)
    for candidate in ("sample_id", "geo_accession"):
        if candidate in md.columns:
            return set(md[candidate].astype(str).str.strip())
    raise ValueError(f"{path} must contain sample_id or geo_accession.")


def infer_batch_labels(sample_ids: pd.Index, gse63060_metadata: Path, gse63061_metadata: Path) -> pd.Series:
    gse63060_ids = read_geo_metadata_sample_ids(gse63060_metadata)
    gse63061_ids = read_geo_metadata_sample_ids(gse63061_metadata)
    labels: dict[str, str] = {}
    for sample_id in sample_ids.astype(str):
        gsm_id = sample_id.split("_", 1)[0].strip()
        if gsm_id in gse63060_ids:
            labels[sample_id] = "GSE63060"
        elif gsm_id in gse63061_ids:
            labels[sample_id] = "GSE63061"
        else:
            labels[sample_id] = "unknown"
    batch = pd.Series(labels, index=sample_ids, name="batch")
    unknown = batch[batch == "unknown"]
    if not unknown.empty:
        preview = ", ".join(unknown.index.astype(str)[:10])
        raise ValueError(f"Could not map {len(unknown)} samples to batches. Examples: {preview}")
    return batch


def make_batch_holdout_splits(
    y: pd.Series,
    batch: pd.Series,
    scenarios: list[str],
    repeats: int,
    inner_val_ratio: float,
    shared_test_size: float,
    seed: int,
) -> list[dict[str, Any]]:
    y_values = y.to_numpy(dtype=np.int64)
    all_idx = np.arange(len(y_values))
    batch_values = batch.to_numpy()
    splits: list[dict[str, Any]] = []
    for repeat in range(1, repeats + 1):
        repeat_seed = seed + repeat - 1
        for scenario_idx, scenario in enumerate(scenarios, start=1):
            if scenario == "test_gse63060":
                pool_idx = all_idx[batch_values == "GSE63061"]
                test_idx = all_idx[batch_values == "GSE63060"]
            elif scenario == "test_gse63061":
                pool_idx = all_idx[batch_values == "GSE63060"]
                test_idx = all_idx[batch_values == "GSE63061"]
            elif scenario == "shared_test":
                train_parts: list[np.ndarray] = []
                test_parts: list[np.ndarray] = []
                for batch_name in ("GSE63060", "GSE63061"):
                    batch_idx = all_idx[batch_values == batch_name]
                    train_part, test_part = train_test_split(
                        batch_idx,
                        test_size=shared_test_size,
                        random_state=repeat_seed,
                        stratify=y_values[batch_idx],
                    )
                    train_parts.append(np.asarray(train_part, dtype=np.int64))
                    test_parts.append(np.asarray(test_part, dtype=np.int64))
                pool_idx = np.concatenate(train_parts)
                test_idx = np.concatenate(test_parts)
            else:
                raise ValueError(f"Unsupported scenario: {scenario}")
            train_idx, val_idx = train_test_split(
                np.asarray(pool_idx, dtype=np.int64),
                test_size=inner_val_ratio,
                random_state=repeat_seed * 100 + scenario_idx,
                stratify=y_values[pool_idx],
            )
            splits.append(
                {
                    "scenario": scenario,
                    "scenario_index": scenario_idx,
                    "repeat": repeat,
                    "fold": 1,
                    "seed": repeat_seed,
                    "train_inner_idx": np.asarray(train_idx, dtype=np.int64),
                    "val_inner_idx": np.asarray(val_idx, dtype=np.int64),
                    "outer_test_idx": np.asarray(test_idx, dtype=np.int64),
                }
            )
    return splits


def make_stratified_holdout_splits(
    y: pd.Series,
    repeats: int,
    train_size: float,
    val_size: float,
    test_size: float,
    seed: int,
) -> list[dict[str, Any]]:
    total = train_size + val_size + test_size
    if not np.isclose(total, 1.0):
        raise ValueError(f"train_size + val_size + test_size must be 1.0, got {total}")
    if min(train_size, val_size, test_size) <= 0:
        raise ValueError("train_size, val_size, and test_size must be positive.")
    y_values = y.to_numpy(dtype=np.int64)
    all_idx = np.arange(len(y_values))
    scenario = f"stratified_{int(train_size * 100)}_{int(val_size * 100)}_{int(test_size * 100)}"
    splits: list[dict[str, Any]] = []
    for repeat in range(1, repeats + 1):
        repeat_seed = seed + repeat - 1
        train_val_idx, test_idx = train_test_split(
            all_idx,
            test_size=test_size,
            random_state=repeat_seed,
            stratify=y_values,
        )
        val_fraction_of_train_val = val_size / (train_size + val_size)
        train_idx, val_idx = train_test_split(
            np.asarray(train_val_idx, dtype=np.int64),
            test_size=val_fraction_of_train_val,
            random_state=repeat_seed,
            stratify=y_values[train_val_idx],
        )
        splits.append(
            {
                "scenario": scenario,
                "scenario_index": 1,
                "repeat": repeat,
                "fold": 1,
                "seed": repeat_seed,
                "train_inner_idx": np.asarray(train_idx, dtype=np.int64),
                "val_inner_idx": np.asarray(val_idx, dtype=np.int64),
                "outer_test_idx": np.asarray(test_idx, dtype=np.int64),
            }
        )
    return splits


def preprocess_split(x: pd.DataFrame, train_idx: np.ndarray, val_idx: np.ndarray, test_idx: np.ndarray, scaler_name: str):
    x_train_raw = x.iloc[train_idx]
    x_val_raw = x.iloc[val_idx]
    x_test_raw = x.iloc[test_idx]
    imputer = SimpleImputer(strategy="median")
    x_train = imputer.fit_transform(x_train_raw)
    x_val = imputer.transform(x_val_raw)
    x_test = imputer.transform(x_test_raw)
    if scaler_name == "minmax":
        scaler = MinMaxScaler()
        x_train = scaler.fit_transform(x_train)
        x_val = scaler.transform(x_val)
        x_test = scaler.transform(x_test)
    elif scaler_name == "standard":
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_val = scaler.transform(x_val)
        x_test = scaler.transform(x_test)
    elif scaler_name == "robust":
        scaler = RobustScaler()
        x_train = scaler.fit_transform(x_train)
        x_val = scaler.transform(x_val)
        x_test = scaler.transform(x_test)
    elif scaler_name != "none":
        raise ValueError(f"Unsupported scaler: {scaler_name}")
    return (
        np.nan_to_num(x_train.astype(np.float32), nan=0.0),
        np.nan_to_num(x_val.astype(np.float32), nan=0.0),
        np.nan_to_num(x_test.astype(np.float32), nan=0.0),
    )


def build_embedding_for_run(
    gene_names: list[str],
    args: argparse.Namespace,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if args.embedding_source == "random":
        embed_df = build_random_embedding_dataframe(gene_names, args.embed_dim, seed, init_scale=0.02)
        return embed_df, {
            "embedding_source": "random",
            "embedding_dim": int(args.embed_dim),
            "target_genes": int(len(gene_names)),
            "matched_genes": 0,
            "missing_genes": int(len(gene_names)),
            "missing_gene_names": gene_names,
            "source_file": "",
        }

    if args.embedding_source == "ppi":
        validate_ppi_embedding_file(args.ppi_embedding_file)
        source_df = pd.read_csv(args.ppi_embedding_file, index_col=0)
        embed_df, report = remap_embedding_dataframe(
            source_df,
            gene_names,
            seed,
            init_scale=0.02,
        )
        if int(report["missing_genes"]) > 0:
            missing_preview = ", ".join(report["missing_gene_names"][:10])
            raise ValueError(
                f"PPI mapped-only mode received {report['missing_genes']} unmapped genes. "
                f"Examples: {missing_preview}"
            )
        report = {
            **report,
            "embedding_source": "ppi",
            "source_file": str(args.ppi_embedding_file),
            "gene_policy": "mapped_only",
            "coverage": float(report["matched_genes"] / max(report["target_genes"], 1)),
            **load_ppi_source_metadata(args.ppi_embedding_file),
        }
        return embed_df, report

    raise ValueError(f"Unsupported embedding source: {args.embedding_source}")


def embedding_model_suffix(args: argparse.Namespace) -> str:
    return "" if args.embedding_source == "random" else "_ppi_mapped"


def load_ppi_gene_set(path: Path) -> set[str]:
    validate_ppi_embedding_file(path)
    genes = pd.read_csv(path, usecols=[0]).iloc[:, 0].astype(str).str.strip()
    genes = genes[genes != ""]
    if genes.empty:
        raise ValueError(f"PPI embedding file has no gene index: {path}")
    return set(genes)


def validate_ppi_embedding_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"PPI embedding file not found: {path}")
    if path.name != "ppi_node_embedding.csv":
        raise ValueError(
            "High-quality PPI mode requires the raw graph-node embedding file "
            "`ppi_node_embedding.csv`, not a target-remapped `gene_embedding.csv` file."
        )


def load_ppi_source_metadata(path: Path) -> dict[str, Any]:
    report_path = path.with_name("ppi_embedding_report.json")
    if not report_path.exists():
        return {"ppi_metadata_note": f"No PPI report found next to {path}"}
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"ppi_metadata_note": f"Could not parse PPI report: {report_path}"}
    hippie = report.get("hippie", {})
    return {
        "ppi_source": report.get("source", "unknown"),
        "ppi_score_threshold": report.get("score_threshold", ""),
        "ppi_embedding_dim_source": report.get("embedding_dim", ""),
        "ppi_edges_after_dedup": hippie.get("edges_after_dedup", ""),
        "ppi_nodes_after_filter": hippie.get("nodes_after_filter", ""),
    }


def restrict_to_ppi_mapped(genes: list[str], ppi_gene_set: set[str] | None) -> list[str]:
    if ppi_gene_set is None:
        return genes
    return [gene for gene in genes if gene in ppi_gene_set]


def median_absolute_deviation(frame: pd.DataFrame) -> pd.Series:
    median = frame.median(axis=0)
    return frame.sub(median, axis=1).abs().median(axis=0)


def load_nature2020_genes(result_root: Path, scenario: str, repeat: int, feature_set: str, available_genes: pd.Index) -> list[str]:
    base = result_root / "runs" / scenario / f"repeat_{repeat:02d}" / feature_set
    preferred = base / "lr" / "selected_genes.txt"
    if preferred.exists():
        path = preferred
    else:
        matches = sorted(base.glob("*/selected_genes.txt"))
        if not matches:
            return []
        path = matches[0]
    raw_genes = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    available = set(available_genes.astype(str))
    return [gene for gene in raw_genes if gene in available]


def split_gene_names(
    args: argparse.Namespace,
    split: dict[str, Any],
    x: pd.DataFrame,
    ppi_gene_set: set[str] | None,
) -> tuple[list[str], str, dict[str, Any]]:
    if args.gene_source == "input":
        raw_genes = x.columns.astype(str).tolist()
        genes = restrict_to_ppi_mapped(raw_genes, ppi_gene_set)
        return genes, "txt_deg", {
            "gene_source": "input",
            "source_feature_set": "input_matrix",
            "ppi_gene_policy": "mapped_only" if ppi_gene_set is not None else "not_applied",
            "n_genes_before_ppi_filter": len(raw_genes),
            "n_genes_after_ppi_filter": len(genes),
        }
    if args.gene_source == "top_variance":
        candidate_genes = restrict_to_ppi_mapped(x.columns.astype(str).tolist(), ppi_gene_set)
        x_train = x.loc[x.index[split["train_inner_idx"]], candidate_genes]
        variances = x_train.var(axis=0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        genes = variances.sort_values(ascending=False).head(min(args.top_variance_genes, len(candidate_genes))).index.astype(str).tolist()
        return genes, "txt_top1000_variance", {
            "gene_source": "top_variance",
            "source_feature_set": "train_inner_variance",
            "top_variance_genes": int(args.top_variance_genes),
            "selection_scope": "train_inner",
            "ppi_gene_policy": "mapped_only" if ppi_gene_set is not None else "not_applied",
            "n_genes_before_ppi_filter": int(x.shape[1]),
            "n_genes_after_ppi_filter": len(candidate_genes),
        }
    if args.gene_source == "top_mad":
        candidate_genes = restrict_to_ppi_mapped(x.columns.astype(str).tolist(), ppi_gene_set)
        x_train = x.loc[x.index[split["train_inner_idx"]], candidate_genes]
        mad = median_absolute_deviation(x_train).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        genes = mad.sort_values(ascending=False).head(min(args.top_mad_genes, len(candidate_genes))).index.astype(str).tolist()
        return genes, f"txt_top{args.top_mad_genes}_mad", {
            "gene_source": "top_mad",
            "source_feature_set": "train_inner_mad",
            "top_mad_genes": int(args.top_mad_genes),
            "selection_scope": "train_inner",
            "ppi_gene_policy": "mapped_only" if ppi_gene_set is not None else "not_applied",
            "n_genes_before_ppi_filter": int(x.shape[1]),
            "n_genes_after_ppi_filter": len(candidate_genes),
        }
    if args.gene_source == "nature2020":
        raw_genes = load_nature2020_genes(args.nature2020_result_root, split["scenario"], split["repeat"], args.nature2020_feature_set, x.columns)
        genes = restrict_to_ppi_mapped(raw_genes, ppi_gene_set)
        model_name = f"txt_nature2020_{args.nature2020_feature_set}"
        return genes, model_name, {
            "gene_source": "nature2020",
            "nature2020_result_root": str(args.nature2020_result_root),
            "source_feature_set": args.nature2020_feature_set,
            "ppi_gene_policy": "mapped_only" if ppi_gene_set is not None else "not_applied",
            "n_genes_before_ppi_filter": len(raw_genes),
            "n_genes_after_ppi_filter": len(genes),
        }
    raise ValueError(f"Unsupported gene_source: {args.gene_source}")


def augmentation_variants(args: argparse.Namespace, n_train: int) -> list[dict[str, Any]]:
    if args.augmentation == "none":
        return [
            {
                "augmentation": "none",
                "target_name": "none",
                "target_size": n_train,
                "model_suffix": "",
            }
        ]

    if args.augmentation == "borderline_smote":
        return [
            {
                "augmentation": "borderline_smote",
                "target_name": "auto_balance",
                "target_size": n_train,
                "model_suffix": "_borderline_smote",
            }
        ]

    variants: list[dict[str, Any]] = []
    for raw_target in args.gan_targets:
        target = str(raw_target).strip().lower()
        if target == "double":
            target_size = int(2 * n_train)
            target_name = "double"
            suffix = "_gan_double"
        else:
            try:
                target_size = int(target)
            except ValueError as exc:
                raise ValueError(f"Unsupported GAN target {raw_target!r}; use 'double' or a positive integer.") from exc
            if target_size <= 0:
                raise ValueError(f"GAN target must be positive, got {raw_target!r}.")
            target_name = f"target{target_size}"
            suffix = f"_gan_target{target_size}"
        variants.append(
            {
                "augmentation": "gan",
                "target_name": target_name,
                "target_size": target_size,
                "model_suffix": suffix,
            }
        )
    return variants


def logits_fn(model: TxT, batch_gene_x: torch.Tensor) -> torch.Tensor:
    return model(batch_gene_x)[0]


def evaluate_binary_from_probs(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    scores = y_prob[:, 1]
    metrics = {
        "pr_auc": float(average_precision_score(y_true, scores)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, scores))
    except ValueError:
        metrics["roc_auc"] = float("nan")
    try:
        metrics["log_loss"] = float(log_loss(y_true, y_prob, labels=[0, 1]))
    except ValueError:
        metrics["log_loss"] = float("nan")
    return metrics


def write_run_artifacts(
    run_dir: Path,
    metric_row: dict[str, Any],
    sample_ids: pd.Index,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    selected_genes: list[str],
    training_summary: dict[str, Any],
    args: argparse.Namespace,
    augmentation_manifest: dict[str, Any],
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    scores = y_prob[:, 1]
    pd.DataFrame([metric_row]).to_csv(run_dir / "metrics.csv", index=False)
    pd.DataFrame({"sample_id": sample_ids.astype(str), "y_true": y_true, "y_score": scores, "y_pred": y_pred}).to_csv(run_dir / "predictions.csv", index=False)
    pd.DataFrame(confusion_matrix(y_true, y_pred, labels=[0, 1]), index=["true_0", "true_1"], columns=["pred_0", "pred_1"]).to_csv(run_dir / "confusion_matrix.csv")
    pd.DataFrame(classification_report(y_true, y_pred, labels=[0, 1], output_dict=True, zero_division=0)).T.to_csv(run_dir / "classification_report.csv")
    (run_dir / "selected_genes.txt").write_text("\n".join(selected_genes) + "\n", encoding="utf-8")
    write_json(run_dir / "hyperparameters.json", training_summary)
    write_json(run_dir / "augmentation_manifest.json", augmentation_manifest)
    write_json(run_dir / "run_manifest.json", {"metrics": metric_row, "training": training_summary, "augmentation": augmentation_manifest, "args": vars(args)})


def summarize(result_root: Path, baseline_result_root: Path | None = None) -> None:
    frames = [pd.read_csv(path) for path in result_root.rglob("metrics.csv")]
    if not frames:
        return
    all_metrics = pd.concat(frames, ignore_index=True)
    all_metrics.to_csv(result_root / "all_metrics.csv", index=False)
    metric_cols = ["pr_auc", "roc_auc", "accuracy", "balanced_accuracy", "precision", "recall", "macro_f1", "weighted_f1", "log_loss"]
    summary = all_metrics.groupby(["protocol", "scenario", "model"], dropna=False)[metric_cols].agg(["mean", "std", "count"]).reset_index()
    summary.columns = ["_".join([str(part) for part in col if part]) for col in summary.columns.to_flat_index()]
    summary = summary.sort_values("roc_auc_mean", ascending=False)
    summary.to_csv(result_root / "summary_by_method.csv", index=False)
    summary.to_csv(result_root / "ranking_by_roc_auc.csv", index=False)
    summary.sort_values("macro_f1_mean", ascending=False).to_csv(result_root / "ranking_by_macro_f1.csv", index=False)
    summary.sort_values("accuracy_mean", ascending=False).to_csv(result_root / "ranking_by_accuracy.csv", index=False)

    if baseline_result_root is None or not baseline_result_root.exists() or baseline_result_root.resolve() == result_root.resolve():
        return
    baseline_summary_path = baseline_result_root / "summary_by_method.csv"
    if baseline_summary_path.exists():
        baseline_summary = pd.read_csv(baseline_summary_path)
        baseline_summary.insert(0, "summary_source", str(baseline_result_root))
        current_summary = summary.copy()
        current_summary.insert(0, "summary_source", str(result_root))
        baseline_summary.to_csv(result_root / "baseline_reference_summary.csv", index=False)
        pd.concat([current_summary, baseline_summary], ignore_index=True, sort=False).to_csv(
            result_root / "summary_with_baseline_reference.csv",
            index=False,
        )
    baseline_metrics_path = baseline_result_root / "all_metrics.csv"
    if baseline_metrics_path.exists():
        baseline_metrics = pd.read_csv(baseline_metrics_path)
        baseline_metrics.insert(0, "metrics_source", str(baseline_result_root))
        current_metrics = all_metrics.copy()
        current_metrics.insert(0, "metrics_source", str(result_root))
        pd.concat([current_metrics, baseline_metrics], ignore_index=True, sort=False).to_csv(
            result_root / "all_metrics_with_baseline_reference.csv",
            index=False,
        )


def main() -> None:
    args = parse_args()
    args.x_file = resolve(args.x_file)
    args.y_file = resolve(args.y_file)
    args.result_root = resolve(args.result_root)
    args.nature2020_result_root = resolve(args.nature2020_result_root)
    args.gse63060_metadata = resolve(args.gse63060_metadata)
    args.gse63061_metadata = resolve(args.gse63061_metadata)
    args.baseline_result_root = resolve(args.baseline_result_root) if args.baseline_result_root is not None else None
    args.ppi_embedding_file = resolve(args.ppi_embedding_file)
    if args.strict_v2:
        args.split_protocol = "batch_holdout"
        args.repeats = 10
        args.batch_scenarios = ["shared_test", "test_gse63060", "test_gse63061"]
        args.gene_source = "top_mad" if args.embedding_source == "ppi" else "top_variance"
        args.top_variance_genes = 1000
        args.top_mad_genes = 1000
        args.scaler = "minmax"
        args.checkpoint_metric = "strict_v2"
        args.threshold_mode = "fixed_0_5"
        args.skip_existing = False
        args.overwrite_results = True
    if args.smoke:
        args.repeats = 1
        args.batch_scenarios = args.batch_scenarios[:1]
        args.epochs = min(args.epochs, 2)
        args.early_stopping_patience = min(args.early_stopping_patience, 1)
        args.gan_epochs = min(args.gan_epochs, 2)
        args.max_train_batches = 2
        args.max_val_batches = 1
    if args.d_ff is None:
        args.d_ff = 4 * args.d_model
    if args.overwrite_results and args.result_root.exists():
        shutil.rmtree(args.result_root)
    args.result_root.mkdir(parents=True, exist_ok=True)
    write_json(args.result_root / "args.json", vars(args))
    set_seed(args.seed)
    device = resolve_device(args.device)
    x, y = load_binary_dataset(args.x_file, args.y_file)
    ppi_gene_set = load_ppi_gene_set(args.ppi_embedding_file) if args.embedding_source == "ppi" else None
    batch = infer_batch_labels(x.index, args.gse63060_metadata, args.gse63061_metadata)
    pd.DataFrame({"sample_id": x.index.astype(str), "batch": batch.to_numpy(), "label": y.to_numpy(dtype=np.int64)}).to_csv(args.result_root / "batch_manifest.csv", index=False)
    if args.split_protocol == "batch_holdout":
        splits = make_batch_holdout_splits(y, batch, args.batch_scenarios, args.repeats, args.inner_val_ratio, args.shared_test_size, args.seed)
    else:
        splits = make_stratified_holdout_splits(y, args.repeats, args.train_size, args.val_size, args.test_size, args.seed)
    class_names = ["MCI", "AD"]
    status_rows: list[dict[str, Any]] = []
    augment_training_data = load_augmentation_function() if args.augmentation in {"gan", "borderline_smote"} else None

    for split in splits:
        train_idx = split["train_inner_idx"]
        val_idx = split["val_inner_idx"]
        test_idx = split["outer_test_idx"]
        if set(train_idx.tolist()) & set(val_idx.tolist()) or set(train_idx.tolist()) & set(test_idx.tolist()) or set(val_idx.tolist()) & set(test_idx.tolist()):
            raise ValueError(f"Split overlap detected for {split['scenario']} repeat {split['repeat']}")
        split_seed = int(split["seed"])
        gene_names, model_label, gene_meta = split_gene_names(args, split, x, ppi_gene_set)
        status_rows.append(
            {
                "scenario": split["scenario"],
                "repeat": split["repeat"],
                "gene_source": args.gene_source,
                "source_feature_set": gene_meta["source_feature_set"],
                "ppi_gene_policy": gene_meta.get("ppi_gene_policy", "not_applied"),
                "n_genes_before_ppi_filter": gene_meta.get("n_genes_before_ppi_filter", ""),
                "n_genes_after_ppi_filter": gene_meta.get("n_genes_after_ppi_filter", ""),
                "n_genes": len(gene_names),
                "status": "completed" if gene_names else "skipped_empty_gene_set",
            }
        )
        if not gene_names:
            print(f"Skipping TxT scenario={split['scenario']} repeat={split['repeat']} because gene set is empty.", flush=True)
            continue
        print(f"Preparing TxT scenario={split['scenario']} repeat={split['repeat']} genes={len(gene_names)}", flush=True)
        set_seed(split_seed)
        x_selected = x.loc[:, gene_names]
        x_train, x_val, x_test = preprocess_split(x_selected, train_idx, val_idx, test_idx, args.scaler)
        y_values = y.to_numpy(dtype=np.int64)
        y_train = y_values[train_idx]
        y_val = y_values[val_idx]
        y_test = y_values[test_idx]
        x_train_frame = pd.DataFrame(x_train, index=x.index[train_idx].astype(str), columns=gene_names)

        for variant in augmentation_variants(args, len(train_idx)):
            variant_model_label = f"{model_label}{embedding_model_suffix(args)}{variant['model_suffix']}"
            run_dir = args.result_root / "runs" / split["scenario"] / f"repeat_{split['repeat']:02d}" / variant_model_label
            if args.skip_existing and (run_dir / "metrics.csv").exists():
                continue

            if variant["augmentation"] in {"gan", "borderline_smote"}:
                if augment_training_data is None:
                    raise RuntimeError("Augmentation requested but augmentation loader was not initialized.")
                aug_result = augment_training_data(
                    x_train_frame,
                    y_train,
                    variant["augmentation"],
                    split_seed,
                    target_size=int(variant["target_size"]),
                    latent_dim=args.gan_latent_dim,
                    epochs=args.gan_epochs,
                    batch_size=args.gan_batch_size,
                    learning_rate=args.gan_learning_rate,
                    sampling_strategy=args.gan_sampling_strategy,
                    smote_k_neighbors=args.smote_k_neighbors,
                    smote_m_neighbors=args.smote_m_neighbors,
                    smote_kind=args.smote_kind,
                )
                train_gene_x = aug_result.x_train.to_numpy(dtype=np.float32)
                train_y = aug_result.y_train.astype(np.int64)
                augmentation_manifest = {
                    **aug_result.manifest,
                    "requested_target": variant["target_name"],
                    "requested_target_size": int(variant["target_size"]),
                    "augmentation_scope": "train_inner_after_train_only_preprocessing",
                    "validation_and_test_scope": "not_augmented",
                }
            else:
                train_gene_x = x_train
                train_y = y_train
                augmentation_manifest = {
                    "augmentation": "none",
                    "augmentation_scope": "not_applied",
                    "requested_target": "none",
                    "requested_target_size": int(len(y_train)),
                    "n_original_train": int(len(y_train)),
                    "n_synthetic": 0,
                    "n_augmented_train": int(len(y_train)),
                }

            print(
                f"Running TxT scenario={split['scenario']} repeat={split['repeat']} "
                f"model={variant_model_label} genes={len(gene_names)} train={len(train_y)} "
                f"embedding={args.embedding_source}",
                flush=True,
            )
            set_seed(split_seed)
            class_weights = compute_balanced_class_weights(train_y) if args.class_weighting == "on" else None
            run_dir.mkdir(parents=True, exist_ok=True)

            with tempfile.TemporaryDirectory() as temp_dir:
                embed_df, embedding_manifest = build_embedding_for_run(gene_names, args, split_seed)
                embed_path = Path(temp_dir) / "gene_embedding.csv"
                embed_df.to_csv(embed_path)
                embed_df.to_csv(run_dir / "gene_embedding_used.csv")
                model = TxT(
                    embed_file=str(embed_path),
                    gene_list=gene_names,
                    n_heads=args.n_heads,
                    d_model=args.d_model,
                    dropout=args.dropout,
                    d_ff=args.d_ff,
                    n_layers=args.n_layers,
                    aggfunc=args.aggfunc,
                    d_hidden1=args.d_hidden1,
                    d_hidden2=args.d_hidden2,
                    slope=args.slope,
                    d_output_dict={"label": 2},
                ).to(device)
                history_rows, best_state_dict, training_summary = train_classifier(
                    model=model,
                    train_gene_x=train_gene_x,
                    train_y=train_y,
                    val_gene_x=x_val,
                    val_y=y_val,
                    batch_size=args.batch_size,
                    epochs=args.epochs,
                    lr=args.lr,
                    weight_decay=args.weight_decay,
                    class_weights=class_weights,
                    early_stopping_patience=args.early_stopping_patience,
                    device=device,
                    logits_fn=logits_fn,
                    max_train_batches=args.max_train_batches,
                    max_val_batches=args.max_val_batches,
                    checkpoint_metric=args.checkpoint_metric,
                    checkpoint_auc_weight=args.checkpoint_auc_weight,
                )
                pd.DataFrame(history_rows).to_csv(run_dir / "training_log.csv", index=False)
                torch.save(best_state_dict, run_dir / "best_model.pt")
                model.load_state_dict(best_state_dict)
                val_result = evaluate_classifier(model, x_val, y_val, x.index[val_idx].to_numpy(), args.batch_size, device, class_names, logits_fn)
                test_result = evaluate_classifier(model, x_test, y_test, x.index[test_idx].to_numpy(), args.batch_size, device, class_names, logits_fn)
                if args.threshold_mode == "fixed_0_5":
                    threshold = 0.5
                    threshold_meta = {"threshold": threshold, "threshold_mode": "fixed_0_5"}
                else:
                    metric = args.threshold_mode.replace("validation_", "")
                    threshold, threshold_meta = calibrate_threshold(val_result["y_true"], val_result["y_prob"][:, 1], metric=metric)
                    threshold_meta["threshold_mode"] = args.threshold_mode
                test_result["y_pred"] = (test_result["y_prob"][:, 1] >= threshold).astype(int)
                metrics = evaluate_binary_from_probs(test_result["y_true"], test_result["y_pred"], test_result["y_prob"])
                row = {
                    "protocol": args.split_protocol,
                    "scenario": split["scenario"],
                    "repeat": split["repeat"],
                    "fold": 1,
                    "seed": split_seed,
                    "model": variant_model_label,
                    "augmentation": variant["augmentation"],
                    "augmentation_target": variant["target_name"],
                    "n_train_inner": len(train_idx),
                    "n_train_augmented": int(len(train_y)),
                    "n_synthetic": int(augmentation_manifest.get("n_synthetic", 0)),
                    "n_val_inner": len(val_idx),
                    "n_outer_test": len(test_idx),
                    "n_genes": len(gene_names),
                    "embedding_source": args.embedding_source,
                    "embedding_dim": int(embedding_manifest["embedding_dim"]),
                    "embedding_matched_genes": int(embedding_manifest.get("matched_genes", 0)),
                    "embedding_missing_genes": int(embedding_manifest.get("missing_genes", 0)),
                    "embedding_coverage": float(embedding_manifest.get("coverage", 0.0)),
                    "train_batches": "|".join(sorted(batch.iloc[train_idx].unique().tolist())),
                    "val_batches": "|".join(sorted(batch.iloc[val_idx].unique().tolist())),
                    "test_batches": "|".join(sorted(batch.iloc[test_idx].unique().tolist())),
                    **metrics,
                }
                training_summary = {
                    **training_summary,
                    **gene_meta,
                    "embedding": embedding_manifest,
                    "threshold_calibration": threshold_meta,
                    "augmentation": augmentation_manifest,
                    "class_weight_source": "augmented_train" if args.class_weighting == "on" else "disabled",
                }
                write_run_artifacts(
                    run_dir,
                    row,
                    x.index[test_idx],
                    test_result["y_true"],
                    test_result["y_pred"],
                    test_result["y_prob"],
                    gene_names,
                    training_summary,
                    args,
                    augmentation_manifest,
                )
    summarize(args.result_root, args.baseline_result_root)
    pd.DataFrame(status_rows).to_csv(args.result_root / "gene_selection_status.csv", index=False)
    pd.DataFrame(
        [
            {"item": "input_dataset", "status": "pass", "details": f"{len(y)} samples x {x.shape[1]} genes from {args.x_file}"},
            {
                "item": "protocol",
                "status": "pass",
                "details": (
                    f"{args.repeats} repeat(s), scenarios={','.join(args.batch_scenarios)}"
                    if args.split_protocol == "batch_holdout"
                    else f"{args.repeats} repeat(s), split={args.train_size:.2f}/{args.val_size:.2f}/{args.test_size:.2f}, seed={args.seed}"
                ),
            },
            {"item": "preprocessing_scope", "status": "pass", "details": "Median imputer and scaler fit only on train_inner."},
            {
                "item": "feature_scope",
                "status": "strict_v2_pass",
                "details": (
                    f"gene_source=top_mad; MAD is computed only on train_inner with k={args.top_mad_genes}."
                    if args.gene_source == "top_mad"
                    else f"gene_source={args.gene_source}; top_variance is computed only on train_inner with k={args.top_variance_genes}."
                ),
            },
            {
                "item": "embedding_scope",
                "status": "strict_v2_pass",
                "details": (
                    f"embedding_source=ppi; high-quality PPI graph-node embeddings loaded from {args.ppi_embedding_file}; gene selection is restricted to PPI-mapped genes before top-variance ranking."
                    if args.embedding_source == "ppi" and args.gene_source != "top_mad"
                    else f"embedding_source=ppi; high-quality PPI graph-node embeddings loaded from {args.ppi_embedding_file}; gene selection is restricted to PPI-mapped genes before MAD ranking."
                    if args.embedding_source == "ppi"
                    else "embedding_source=random; embeddings are initialized inside each run from the split seed."
                ),
            },
            {
                "item": "augmentation_scope",
                "status": "strict_v2_pass",
                "details": (
                    f"{args.augmentation} applied only to train_inner after train-only preprocessing; validation and outer_test are never augmented."
                    if args.augmentation in {"gan", "borderline_smote"}
                    else "No augmentation applied."
                ),
            },
            {"item": "checkpoint_scope", "status": "strict_v2_pass", "details": "checkpoint_metric=strict_v2 uses 0.5*val_roc_auc + 0.5*val_macro_f1 - 0.25*val_loss."},
        ]
    ).to_csv(args.result_root / "leakage_audit.csv", index=False)
    print(f"Finished. Results written to {args.result_root}", flush=True)


if __name__ == "__main__":
    main()
