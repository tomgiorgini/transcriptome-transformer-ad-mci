#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

ROOT = Path(__file__).resolve().parents[3]
MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))
COMMON_DIR = Path(__file__).resolve().parents[1]
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

from data import assert_no_split_overlap, load_ad_mci_dataset, repeated_stratified_nested_splits
from feature_selection import SelectedFeatureSet, select_lasso_genes, use_fixed_input_genes
from image_transformer import LDAImageTransformer
from metrics import classification_report_frame, confusion_frame, predictions_frame
from models import train_and_evaluate
from summarize_results import summarize
from shared_test_splits import load_task_splits, split_counts


DEFAULT_X = ROOT / "task_dataset" / "processed" / "ad_mci_binary" / "X_ad_mci.csv"
DEFAULT_Y = ROOT / "task_dataset" / "processed" / "ad_mci_binary" / "y_ad_mci.csv"
DEFAULT_RESULT_ROOT = ROOT / "results" / "SOTA" / "one2mfusion"
DEFAULT_GSE63060_METADATA = ROOT / "pretraining_dataset" / "geo_downloads" / "GSE63060" / "eset_1_GPL6947" / "GSE63060_sample_metadata.tsv.gz"
DEFAULT_GSE63061_METADATA = ROOT / "pretraining_dataset" / "geo_downloads" / "GSE63061" / "eset_1_GPL10558" / "GSE63061_sample_metadata.tsv.gz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Leakage-safe One2MFusion reproduction for a local binary transcriptomic task.")
    parser.add_argument("--x-file", type=Path, default=DEFAULT_X)
    parser.add_argument("--y-file", type=Path, default=DEFAULT_Y)
    parser.add_argument(
        "--split-manifest-dir",
        type=Path,
        default=None,
        help="Directory of TxT seed_<seed>.csv manifests. When set, use their exact shared-test sample membership.",
    )
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument(
        "--implementation-profile",
        choices=["paper", "public_code"],
        default="paper",
        help="Use article defaults or the conflicting defaults in the released AD-vs-NC notebook.",
    )
    parser.add_argument("--models", nargs="+", default=["fnn", "cnn", "one2mfusion"], choices=["fnn", "cnn", "one2mfusion"])
    parser.add_argument("--protocol", choices=["nested_leakage_safe", "batch_holdout"], default="nested_leakage_safe")
    parser.add_argument(
        "--batch-scenarios",
        nargs="+",
        default=["shared_test", "test_gse63060", "test_gse63061"],
        choices=["shared_test", "test_gse63060", "test_gse63061"],
    )
    parser.add_argument("--shared-test-size", type=float, default=0.2)
    parser.add_argument("--gse63060-metadata", type=Path, default=DEFAULT_GSE63060_METADATA)
    parser.add_argument("--gse63061-metadata", type=Path, default=DEFAULT_GSE63061_METADATA)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-val-ratio", type=float, default=0.125)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lasso-iter", type=int, default=25)
    parser.add_argument("--lasso-cv-folds", type=int, default=5)
    parser.add_argument("--hyperparameter-mode", choices=["bayes", "fixed_paper"], default="fixed_paper")
    parser.add_argument("--paper-lasso-alpha", type=float, default=None)
    parser.add_argument("--gene-selection-mode", choices=["nonzero", "fixed_k", "fixed_input"], default="nonzero")
    parser.add_argument(
        "--artifact-scope",
        choices=["train_inner", "full_dataset"],
        default="train_inner",
        help="Use train_inner-fitted artifacts or intentionally leaky full-dataset preprocessing/LASSO/LDA artifacts.",
    )
    parser.add_argument("--target-gene-count", type=int, default=488)
    parser.add_argument("--n-jobs", type=int, default=2)
    parser.add_argument("--pixels", type=int, default=90)
    parser.add_argument("--fisher-groups", type=int, default=15)
    parser.add_argument("--image-gene-order", choices=["fisher", "input"], default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size-fusion", type=int, default=None)
    parser.add_argument("--batch-size-single", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None, help="Override profile learning rates for every model.")
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--early-stopping-monitor", choices=["val_loss", "loss", "strict_v2"], default=None)
    parser.add_argument(
        "--paper-like-preprocessing",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--threshold-mode", choices=["fixed_0_5", "validation_macro_f1", "validation_accuracy", "validation_accuracy_macro_f1"], default="fixed_0_5")
    parser.add_argument("--start-from-epoch", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--strict-v2", action="store_true", help="Run the unified strict leakage-free V2 configuration.")
    parser.add_argument("--overwrite-results", action="store_true", help="Delete the result root before running.")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def apply_implementation_profile(args: argparse.Namespace) -> None:
    if args.implementation_profile == "paper":
        defaults = {
            "paper_lasso_alpha": 1e-6,
            "image_gene_order": "fisher",
            "epochs": 1003,
            "batch_size_fusion": 30,
            "batch_size_single": 30,
            "early_stopping_monitor": "loss",
            "start_from_epoch": 250,
        }
    else:
        defaults = {
            "paper_lasso_alpha": 2e-4,
            "image_gene_order": "input",
            "epochs": 1003,
            "batch_size_fusion": 64,
            "batch_size_single": 32,
            "early_stopping_monitor": "loss",
            "start_from_epoch": 250,
        }
    for name, value in defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, value)


def model_learning_rate(args: argparse.Namespace, model_name: str) -> float:
    if args.learning_rate is not None:
        return float(args.learning_rate)
    if args.implementation_profile == "paper" or model_name == "cnn":
        return 1e-4
    return 1e-3


def model_epochs(args: argparse.Namespace, model_name: str) -> int:
    if args.implementation_profile != "public_code" or args.epochs != 1003:
        return int(args.epochs)
    return {"fnn": 1002, "cnn": 1001, "one2mfusion": 1003}[model_name]


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else (ROOT / path).resolve()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)


def read_geo_metadata_sample_ids(path: Path) -> set[str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing GEO metadata file: {path}")
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
        raise ValueError(f"Could not map {len(unknown)} samples to GSE63060/GSE63061 metadata. Examples: {preview}")
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
                raise ValueError(f"Unsupported batch scenario: {scenario}")
            train_idx, val_idx = train_test_split(
                np.asarray(pool_idx, dtype=np.int64),
                test_size=inner_val_ratio,
                random_state=repeat_seed * 100 + len(splits) + 1,
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


def split_frames(x: pd.DataFrame, y: pd.Series, train_idx: np.ndarray, val_idx: np.ndarray, test_idx: np.ndarray):
    return (
        x.iloc[train_idx].copy(),
        y.iloc[train_idx].to_numpy(dtype=np.int64),
        x.iloc[val_idx].copy(),
        y.iloc[val_idx].to_numpy(dtype=np.int64),
        x.iloc[test_idx].copy(),
        y.iloc[test_idx].to_numpy(dtype=np.int64),
    )


def preprocess_fold(x_train: pd.DataFrame, x_val: pd.DataFrame, x_test: pd.DataFrame):
    imputer = SimpleImputer(strategy="median")
    scaler = MinMaxScaler()
    train_imp = imputer.fit_transform(x_train)
    val_imp = imputer.transform(x_val)
    test_imp = imputer.transform(x_test)
    train_scaled = scaler.fit_transform(train_imp)
    val_scaled = scaler.transform(val_imp)
    test_scaled = scaler.transform(test_imp)
    columns = x_train.columns
    return (
        pd.DataFrame(train_scaled, index=x_train.index, columns=columns),
        pd.DataFrame(val_scaled, index=x_val.index, columns=columns),
        pd.DataFrame(test_scaled, index=x_test.index, columns=columns),
        {
            "imputer": "SimpleImputer(strategy=median) fit on train_inner",
            "scaler": "MinMaxScaler fit on train_inner",
            "n_input_features": int(len(columns)),
        },
    )


def preprocess_fold_paper_like(x_train: pd.DataFrame, x_val: pd.DataFrame, x_test: pd.DataFrame):
    imputer = SimpleImputer(strategy="median")
    train_imp = imputer.fit_transform(x_train)
    val_imp = imputer.transform(x_val)
    test_imp = imputer.transform(x_test)
    columns = x_train.columns
    train_raw = pd.DataFrame(train_imp, index=x_train.index, columns=columns)
    val_raw = pd.DataFrame(val_imp, index=x_val.index, columns=columns)
    test_raw = pd.DataFrame(test_imp, index=x_test.index, columns=columns)

    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_imp)
    val_scaled = scaler.transform(val_imp).clip(0.0, 1.0)
    test_scaled = scaler.transform(test_imp).clip(0.0, 1.0)
    train_image = pd.DataFrame(train_scaled, index=x_train.index, columns=columns)
    val_image = pd.DataFrame(val_scaled, index=x_val.index, columns=columns)
    test_image = pd.DataFrame(test_scaled, index=x_test.index, columns=columns)
    return (
        train_raw,
        val_raw,
        test_raw,
        train_image,
        val_image,
        test_image,
        {
            "imputer": "SimpleImputer(strategy=median) fit on train_inner",
            "gene_branch_scaling": "none; imputed train_inner values used, matching notebook FNN/fusion branch",
            "image_branch_scaling": "MinMaxScaler fit on train_inner, clipped [0,1], matching notebook image branch",
            "n_input_features": int(len(columns)),
        },
    )


def preprocess_full(x: pd.DataFrame):
    imputer = SimpleImputer(strategy="median")
    scaler = MinMaxScaler()
    x_imp = imputer.fit_transform(x)
    x_scaled = scaler.fit_transform(x_imp)
    return (
        pd.DataFrame(x_scaled, index=x.index, columns=x.columns),
        {
            "imputer": "SimpleImputer(strategy=median) fit on full dataset",
            "scaler": "MinMaxScaler fit on full dataset",
            "n_input_features": int(x.shape[1]),
            "artifact_scope": "full_dataset",
        },
    )


def subset_selected_features(selected: Any, train_idx: np.ndarray, val_idx: np.ndarray, test_idx: np.ndarray):
    return type(selected)(
        x_train=selected.x_train.iloc[train_idx].copy(),
        x_val=selected.x_train.iloc[val_idx].copy(),
        x_test=selected.x_train.iloc[test_idx].copy(),
        selected_genes=selected.selected_genes,
        metadata={**selected.metadata, "scope": "fit on full dataset; subset per split"},
    )


def gene_frames_for_model(
    model_name: str,
    all_gene_frames: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    selected: SelectedFeatureSet,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Route the paper's standalone FNN to all genes and fusion to LASSO genes."""

    if model_name == "fnn":
        return all_gene_frames
    if model_name == "one2mfusion":
        return selected.x_train, selected.x_val, selected.x_test
    if model_name == "cnn":
        # CNN consumes images; retaining the selected frames here makes model
        # metadata describe the LASSO source of those images.
        return selected.x_train, selected.x_val, selected.x_test
    raise ValueError(f"Unsupported model_name: {model_name}")


def write_run_artifacts(
    run_dir: Path,
    metric_row: dict[str, Any],
    sample_ids: pd.Index,
    y_test: np.ndarray,
    scores: np.ndarray,
    y_pred: np.ndarray,
    model_metadata: dict[str, Any],
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([metric_row]).to_csv(run_dir / "metrics.csv", index=False)
    predictions_frame(sample_ids, y_test, scores, y_pred).to_csv(run_dir / "predictions.csv", index=False)
    confusion_frame(y_test, y_pred).to_csv(run_dir / "confusion_matrix.csv")
    classification_report_frame(y_test, y_pred).to_csv(run_dir / "classification_report.csv")
    write_json(run_dir / "run_manifest.json", {"metrics": metric_row, "model": model_metadata})


def write_audit(result_root: Path, args: argparse.Namespace, x: pd.DataFrame, y: pd.Series, run_count: int) -> None:
    audit = pd.DataFrame(
        [
            {"item": "input_dataset", "status": "pass", "details": f"{len(y)} samples x {x.shape[1]} genes from {args.x_file}"},
            {
                "item": "configuration_matrix",
                "status": "pass",
                "details": f"models={','.join(args.models)}; FNN=all genes, CNN=LASSO image, One2MFusion=LASSO gene+image.",
            },
            {
                "item": "implementation_profile",
                "status": "documented_conflict",
                "details": (
                    f"profile={args.implementation_profile}; alpha={args.paper_lasso_alpha}; "
                    f"image_order={args.image_gene_order}; batches single/fusion={args.batch_size_single}/{args.batch_size_fusion}. "
                    "The article and released notebook disagree on alpha, learning rates, batches, and Fisher ordering."
                ),
            },
            {
                "item": "protocol",
                "status": "pass",
                "details": (
                    f"{args.repeats} repeats across batch scenarios {', '.join(args.batch_scenarios)}; shared_test_size={args.shared_test_size}; inner val ratio {args.inner_val_ratio}"
                    if args.protocol == "batch_holdout"
                    else f"{args.repeats} repeats x {args.outer_folds} outer folds; inner val ratio {args.inner_val_ratio}"
                ),
            },
            {
                "item": "preprocessing_scope",
                "status": "intentional_leakage" if args.artifact_scope == "full_dataset" else "pass",
                "details": (
                    "Imputer and MinMaxScaler fit on full dataset for paper-like/leaky experiment."
                    if args.artifact_scope == "full_dataset"
                    else "Imputer and MinMaxScaler fit only on train_inner; the same scaled all-gene matrix feeds standalone FNN."
                ),
            },
            {
                "item": "feature_selection_scope",
                "status": "intentional_leakage" if args.artifact_scope == "full_dataset" and args.gene_selection_mode != "fixed_input" else "pass",
                "details": (
                    "Fixed input genes are used without fold-level LASSO."
                    if args.gene_selection_mode == "fixed_input"
                    else (
                        f"LASSO alpha fixed to paper value {args.paper_lasso_alpha}; gene selection uses full dataset."
                        if args.artifact_scope == "full_dataset" and args.hyperparameter_mode == "fixed_paper"
                        else f"LASSO alpha fixed to paper value {args.paper_lasso_alpha}; gene selection uses train_inner only."
                        if args.hyperparameter_mode == "fixed_paper"
                        else (
                            "LASSO alpha tuning and gene selection use full dataset."
                            if args.artifact_scope == "full_dataset"
                            else "LASSO alpha tuning and gene selection use train_inner only."
                        )
                    )
                ),
            },
            {
                "item": "image_mapping_scope",
                "status": "intentional_leakage" if args.artifact_scope == "full_dataset" else "pass",
                "details": (
                    f"Gene groups and LDA image mapping fit on full dataset selected genes; image_gene_order={args.image_gene_order}."
                    if args.artifact_scope == "full_dataset"
                    else f"Gene groups and LDA image mapping fit only on train_inner selected genes; image_gene_order={args.image_gene_order}."
                ),
            },
            {
                "item": "model_input_routing",
                "status": "pass",
                "details": "Standalone FNN receives all genes; CNN and both One2MFusion branches originate from train-only LASSO genes.",
            },
            {
                "item": "outer_test_scope",
                "status": "intentional_leakage" if args.artifact_scope == "full_dataset" else "pass",
                "details": (
                    "Outer test participates in preprocessing, LASSO selection, and LDA mapping for this paper-like/leaky experiment."
                    if args.artifact_scope == "full_dataset"
                    else "Outer test is transformed with fitted train_inner artifacts and evaluated once."
                ),
            },
            {"item": "tensorflow_device", "status": "cpu_expected", "details": "Current Windows TensorFlow build does not detect CUDA; One2 deep models run with TensorFlow/Keras on CPU unless environment changes."},
            {"item": "completed_runs", "status": "pass", "details": str(run_count)},
        ]
    )
    audit.to_csv(result_root / "leakage_audit.csv", index=False)


def main() -> None:
    args = parse_args()
    apply_implementation_profile(args)
    args.x_file = resolve(args.x_file)
    args.y_file = resolve(args.y_file)
    args.split_manifest_dir = resolve(args.split_manifest_dir) if args.split_manifest_dir else None
    args.result_root = resolve(args.result_root)
    args.gse63060_metadata = resolve(args.gse63060_metadata)
    args.gse63061_metadata = resolve(args.gse63061_metadata)
    if args.strict_v2:
        args.protocol = "batch_holdout"
        args.repeats = 10
        args.batch_scenarios = ["shared_test"]
        args.models = ["fnn", "cnn", "one2mfusion"]
        args.artifact_scope = "train_inner"
        args.hyperparameter_mode = "fixed_paper"
        args.gene_selection_mode = "nonzero"
        args.lasso_cv_folds = 5
        args.lasso_iter = 0
        args.implementation_profile = "paper"
        args.paper_lasso_alpha = 1e-6
        args.image_gene_order = "fisher"
        args.paper_like_preprocessing = False
        args.patience = 10
        args.start_from_epoch = 250
        args.early_stopping_monitor = "loss"
        args.batch_size_single = 30
        args.batch_size_fusion = 30
        args.learning_rate = 1e-4
        args.threshold_mode = "fixed_0_5"
        args.skip_existing = False
        args.overwrite_results = True
    if args.smoke:
        args.repeats = 1
        args.outer_folds = 5
        args.lasso_iter = min(args.lasso_iter, 1)
        args.lasso_cv_folds = 2
        args.epochs = min(args.epochs, 2)
        args.patience = min(args.patience, 1)
        args.start_from_epoch = 0
        args.n_jobs = 1

    if args.overwrite_results and args.result_root.exists():
        shutil.rmtree(args.result_root)
    args.result_root.mkdir(parents=True, exist_ok=True)
    write_json(args.result_root / "args.json", vars(args))
    x, y = load_ad_mci_dataset(args.x_file, args.y_file)
    full_selected = None
    full_transformer = None
    full_preprocessing_meta: dict[str, Any] | None = None
    x_full = None
    if args.artifact_scope == "full_dataset":
        print("Fitting full-dataset preprocessing, LASSO selection, and LDA mapping (intentional leakage experiment).", flush=True)
        x_full, full_preprocessing_meta = preprocess_full(x)
        y_full = y.to_numpy(dtype=np.int64)
        if args.gene_selection_mode == "fixed_input":
            full_selected = use_fixed_input_genes(x_full, x_full, x_full)
        else:
            full_selected = select_lasso_genes(
                x_full,
                y_full,
                x_full,
                x_full,
                seed=args.seed,
                n_iter=args.lasso_iter,
                cv_folds=args.lasso_cv_folds,
                n_jobs=args.n_jobs,
                selection_mode=args.gene_selection_mode,
                target_gene_count=args.target_gene_count,
                fixed_alpha=args.paper_lasso_alpha if args.hyperparameter_mode == "fixed_paper" else None,
            )
        full_transformer = LDAImageTransformer(pixels=args.pixels, n_groups=args.fisher_groups, order_by=args.image_gene_order).fit(full_selected.x_train, y_full)
        full_dir = args.result_root / "full_dataset_artifacts"
        full_dir.mkdir(parents=True, exist_ok=True)
        write_json(full_dir / "preprocessing_manifest.json", full_preprocessing_meta)
        write_json(full_dir / "lasso_hyperparameters.json", full_selected.metadata)
        write_json(full_dir / "image_mapping_manifest.json", full_transformer.manifest())
        (full_dir / "selected_genes.txt").write_text("\n".join(full_selected.selected_genes) + "\n", encoding="utf-8")
    if args.protocol == "batch_holdout":
        batch = infer_batch_labels(x.index, args.gse63060_metadata, args.gse63061_metadata)
        pd.DataFrame({"sample_id": x.index.astype(str), "batch": batch.to_numpy(), "label": y.to_numpy(dtype=np.int64)}).to_csv(
            args.result_root / "batch_manifest.csv", index=False
        )
        if args.split_manifest_dir:
            splits = load_task_splits(
                x.index,
                y,
                args.split_manifest_dir,
                repeats=args.repeats,
                expected_seeds=range(args.seed, args.seed + args.repeats),
            )
            split_counts(splits).to_csv(args.result_root / "shared_split_counts.csv", index=False)
        else:
            splits = make_batch_holdout_splits(y, batch, args.batch_scenarios, args.repeats, args.inner_val_ratio, args.shared_test_size, args.seed)
    else:
        batch = None
        max_outer = 1 if args.smoke else None
        splits = repeated_stratified_nested_splits(y, args.repeats, args.outer_folds, args.inner_val_ratio, args.seed, max_outer)
    run_rows: list[dict[str, Any]] = []

    for split in splits:
        if isinstance(split, dict):
            train_inner_idx = split["train_inner_idx"]
            val_inner_idx = split["val_inner_idx"]
            outer_test_idx = split["outer_test_idx"]
            split_repeat = split["repeat"]
            split_fold = split["fold"]
            split_seed = int(split["seed"]) if split.get("split_source") else int(split["seed"] * 1000 + split["scenario_index"] * 100 + split["repeat"])
            scenario = split["scenario"]
            if set(train_inner_idx.tolist()) & set(val_inner_idx.tolist()) or set(train_inner_idx.tolist()) & set(outer_test_idx.tolist()) or set(val_inner_idx.tolist()) & set(outer_test_idx.tolist()):
                raise ValueError(f"Split overlap detected for scenario={scenario} repeat={split_repeat}.")
        else:
            assert_no_split_overlap(split)
            train_inner_idx = split.train_inner_idx
            val_inner_idx = split.val_inner_idx
            outer_test_idx = split.outer_test_idx
            split_repeat = split.repeat
            split_fold = split.fold
            split_seed = split.seed * 1000 + split.fold
            scenario = "shared_test"
        x_train_raw, y_train, x_val_raw, y_val, x_test_raw, y_test = split_frames(
            x, y, train_inner_idx, val_inner_idx, outer_test_idx
        )
        if args.artifact_scope == "full_dataset":
            if full_selected is None or full_transformer is None or full_preprocessing_meta is None:
                raise RuntimeError("Full-dataset artifacts were not initialized.")
            selected = subset_selected_features(full_selected, train_inner_idx, val_inner_idx, outer_test_idx)
            preprocessing_meta = full_preprocessing_meta
            transformer = full_transformer
            if x_full is None:
                raise RuntimeError("Full-dataset preprocessing matrix was not initialized.")
            x_train_all_df = x_full.iloc[train_inner_idx].copy()
            x_val_all_df = x_full.iloc[val_inner_idx].copy()
            x_test_all_df = x_full.iloc[outer_test_idx].copy()
        else:
            if args.paper_like_preprocessing:
                (
                    x_train_gene_df,
                    x_val_gene_df,
                    x_test_gene_df,
                    x_train_image_df,
                    x_val_image_df,
                    x_test_image_df,
                    preprocessing_meta,
                ) = preprocess_fold_paper_like(x_train_raw, x_val_raw, x_test_raw)
                x_train_all_df, x_val_all_df, x_test_all_df = x_train_gene_df, x_val_gene_df, x_test_gene_df
                if args.gene_selection_mode == "fixed_input":
                    selected = use_fixed_input_genes(x_train_gene_df, x_val_gene_df, x_test_gene_df)
                else:
                    selected = select_lasso_genes(
                        x_train_gene_df,
                        y_train,
                        x_val_gene_df,
                        x_test_gene_df,
                        seed=split_seed,
                        n_iter=args.lasso_iter,
                        cv_folds=args.lasso_cv_folds,
                        n_jobs=args.n_jobs,
                        selection_mode=args.gene_selection_mode,
                        target_gene_count=args.target_gene_count,
                        fixed_alpha=args.paper_lasso_alpha if args.hyperparameter_mode == "fixed_paper" else None,
                    )
                selected_image = SelectedFeatureSet(
                    x_train=x_train_image_df.loc[:, selected.selected_genes].copy(),
                    x_val=x_val_image_df.loc[:, selected.selected_genes].copy(),
                    x_test=x_test_image_df.loc[:, selected.selected_genes].copy(),
                    selected_genes=selected.selected_genes,
                    metadata={**selected.metadata, "branch": "image_minmax"},
                )
                transformer = LDAImageTransformer(pixels=args.pixels, n_groups=args.fisher_groups, order_by=args.image_gene_order).fit(selected_image.x_train, y_train)
            else:
                x_train, x_val, x_test, preprocessing_meta = preprocess_fold(x_train_raw, x_val_raw, x_test_raw)
                x_train_all_df, x_val_all_df, x_test_all_df = x_train, x_val, x_test
                if args.gene_selection_mode == "fixed_input":
                    selected = use_fixed_input_genes(x_train, x_val, x_test)
                else:
                    selected = select_lasso_genes(
                        x_train,
                        y_train,
                        x_val,
                        x_test,
                        seed=split_seed,
                        n_iter=args.lasso_iter,
                        cv_folds=args.lasso_cv_folds,
                        n_jobs=args.n_jobs,
                        selection_mode=args.gene_selection_mode,
                        target_gene_count=args.target_gene_count,
                        fixed_alpha=args.paper_lasso_alpha if args.hyperparameter_mode == "fixed_paper" else None,
                    )
                selected_image = selected
                transformer = LDAImageTransformer(pixels=args.pixels, n_groups=args.fisher_groups, order_by=args.image_gene_order).fit(selected_image.x_train, y_train)
        if args.artifact_scope == "full_dataset":
            selected_image = selected
        x_train_image = transformer.transform(selected_image.x_train)
        x_val_image = transformer.transform(selected_image.x_val)
        x_test_image = transformer.transform(selected_image.x_test)

        all_gene_frames = (x_train_all_df, x_val_all_df, x_test_all_df)
        if args.protocol != "batch_holdout":
            fold_dir = args.result_root / "runs" / scenario / f"repeat_{split_repeat:02d}" / f"fold_{split_fold:02d}"
        else:
            fold_dir = args.result_root / "runs" / scenario / f"repeat_{split_repeat:02d}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        split_meta = {
            "protocol": args.protocol,
            "scenario": scenario,
            "repeat": split_repeat,
            "fold": split_fold,
            "seed": split_seed,
            "n_train_inner": len(y_train),
            "n_val_inner": len(y_val),
            "n_outer_test": len(y_test),
            "n_selected_genes": len(selected.selected_genes),
        }
        if batch is not None:
            split_meta.update(
                {
                    "train_batches": "|".join(sorted(batch.iloc[train_inner_idx].unique().tolist())),
                    "val_batches": "|".join(sorted(batch.iloc[val_inner_idx].unique().tolist())),
                    "test_batches": "|".join(sorted(batch.iloc[outer_test_idx].unique().tolist())),
                }
            )
        write_json(fold_dir / "split_manifest.json", {**split_meta, **preprocessing_meta})
        write_json(fold_dir / "lasso_hyperparameters.json", selected.metadata)
        write_json(fold_dir / "image_mapping_manifest.json", transformer.manifest())
        (fold_dir / "selected_genes.txt").write_text("\n".join(selected.selected_genes) + "\n", encoding="utf-8")

        for model_name in args.models:
            run_dir = fold_dir / model_name
            if args.skip_existing and (run_dir / "metrics.csv").exists():
                continue
            print(f"Running repeat={split_repeat} fold={split_fold} scenario={scenario} model={model_name} selected_genes={len(selected.selected_genes)}", flush=True)
            model_gene_frames = gene_frames_for_model(model_name, all_gene_frames, selected)
            x_train_gene = model_gene_frames[0].to_numpy(dtype=np.float32)
            x_val_gene = model_gene_frames[1].to_numpy(dtype=np.float32)
            x_test_gene = model_gene_frames[2].to_numpy(dtype=np.float32)
            batch_size = args.batch_size_fusion if model_name == "one2mfusion" else args.batch_size_single
            scores, y_pred, metrics, model_meta = train_and_evaluate(
                model_name=model_name,
                x_train_gene=x_train_gene,
                x_val_gene=x_val_gene,
                x_test_gene=x_test_gene,
                x_train_image=x_train_image,
                x_val_image=x_val_image,
                x_test_image=x_test_image,
                y_train=y_train,
                y_val=y_val,
                y_test=y_test,
                seed=split_seed,
                epochs=model_epochs(args, model_name),
                batch_size=batch_size,
                patience=args.patience,
                checkpoint_path=run_dir / f"{model_name}.keras",
                early_stopping_monitor=args.early_stopping_monitor,
                start_from_epoch=args.start_from_epoch,
                threshold_mode=args.threshold_mode,
                learning_rate=model_learning_rate(args, model_name),
                implementation_profile=args.implementation_profile,
            )
            row = {**split_meta, "model": model_name, **metrics}
            write_run_artifacts(run_dir, row, x_test_raw.index, y_test, scores, y_pred, model_meta)
            run_rows.append(row)

    all_metrics, summary = summarize(args.result_root)
    all_metrics.to_csv(args.result_root / "all_metrics.csv", index=False)
    summary.to_csv(args.result_root / "summary_by_method.csv", index=False)
    summary.to_csv(args.result_root / "ranking_by_pr_auc.csv", index=False)
    if not summary.empty:
        summary.sort_values("roc_auc_mean", ascending=False).to_csv(args.result_root / "ranking_by_roc_auc.csv", index=False)
        summary.sort_values("macro_f1_mean", ascending=False).to_csv(args.result_root / "ranking_by_macro_f1.csv", index=False)
        summary.sort_values("accuracy_mean", ascending=False).to_csv(args.result_root / "ranking_by_accuracy.csv", index=False)
    write_audit(args.result_root, args, x, y, len(all_metrics))
    print(f"Finished. Results written to {args.result_root}", flush=True)


if __name__ == "__main__":
    main()
