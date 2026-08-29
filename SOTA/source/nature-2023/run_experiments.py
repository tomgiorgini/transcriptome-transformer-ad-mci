#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler

ROOT = Path(__file__).resolve().parents[3]
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
COMMON_DIR = Path(__file__).resolve().parents[1]
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

from data import assert_no_split_overlap, load_ad_mci_dataset, repeated_stratified_nested_splits
from deep_models import train_cnn_classifier, train_vae_classifier
from feature_selection import (
    FeatureSet,
    all_genes,
    knowledge_genes_features,
    lasso_features,
    vae_latent_features,
    vssrfe_lr_features,
)
from metrics import (
    classification_report_frame,
    confusion_frame,
    evaluate_binary,
    positive_scores,
    predictions_frame,
)
from models import available_model_names, fit_tuned_model
from summarize_results import summarize
from strict_v2_utils import calibrate_threshold
from shared_test_splits import load_task_splits, split_counts


DEFAULT_X = ROOT / "task_dataset" / "processed" / "ad_mci_binary" / "X_ad_mci.csv"
DEFAULT_Y = ROOT / "task_dataset" / "processed" / "ad_mci_binary" / "y_ad_mci.csv"
DEFAULT_RESULT_ROOT = ROOT / "results" / "SOTA" / "nature-2023"
DEFAULT_GSE63060_METADATA = ROOT / "pretraining_dataset" / "geo_downloads" / "GSE63060" / "eset_1_GPL6947" / "GSE63060_sample_metadata.tsv.gz"
DEFAULT_GSE63061_METADATA = ROOT / "pretraining_dataset" / "geo_downloads" / "GSE63061" / "eset_1_GPL10558" / "GSE63061_sample_metadata.tsv.gz"
PAPER_FEATURE_SETS = ("all_genes", "knowledge_genes", "vssrfe_lr", "lasso", "vae_latent")
PAPER_STANDALONE_MODELS = ("vae_classifier", "cnn")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproduce Kelly et al. 2023 on a local binary transcriptomic task.")
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
        help="Resolve documented paper defaults or conflicting defaults from the authors' public AD code.",
    )
    parser.add_argument(
        "--feature-sets",
        nargs="+",
        default=list(PAPER_FEATURE_SETS),
        choices=list(PAPER_FEATURE_SETS),
    )
    parser.add_argument("--models", nargs="+", default=available_model_names(), choices=available_model_names())
    parser.add_argument(
        "--protocol",
        choices=["nested_leakage_safe", "initial_fs_5cv", "batch_holdout"],
        default="nested_leakage_safe",
        help=(
            "nested_leakage_safe fits feature selection inside each fold; "
            "initial_fs_5cv fits feature selection once before 5-fold CV; "
            "batch_holdout uses GSE63060/GSE63061 holdout scenarios."
        ),
    )
    parser.add_argument(
        "--batch-scenarios",
        nargs="+",
        default=["shared_test", "test_gse63060", "test_gse63061"],
        choices=["shared_test", "test_gse63060", "test_gse63061"],
    )
    parser.add_argument("--shared-test-size", type=float, default=0.2)
    parser.add_argument("--gse63060-metadata", type=Path, default=DEFAULT_GSE63060_METADATA)
    parser.add_argument("--gse63061-metadata", type=Path, default=DEFAULT_GSE63061_METADATA)
    parser.add_argument("--include-deep", action="store_true", help="Run standalone VAE-classifier and CNN on all genes.")
    parser.add_argument(
        "--artifact-scope",
        choices=["train_inner", "full_dataset"],
        default="train_inner",
        help="For batch_holdout, fit preprocessing/feature representations per train_inner or once on the full dataset with intentional leakage.",
    )
    parser.add_argument(
        "--deep-feature-models",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--knowledge-genes-file", type=Path, default=None)
    parser.add_argument("--knowledge-mad-top-k", type=int, default=3000)
    parser.add_argument(
        "--allow-incomplete-knowledge",
        action="store_true",
        help="Run a labelled top-MAD-only ablation when the unpublished curated list is unavailable; otherwise skip knowledge_genes.",
    )
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-val-ratio", type=float, default=0.125)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bayes-iter", type=int, default=100, help="Bayesian iterations for the five classifiers.")
    parser.add_argument("--feature-bayes-iter", type=int, default=200, help="Bayesian iterations for LASSO alpha and VSSRFE LR C.")
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--n-jobs", type=int, default=2)
    parser.add_argument(
        "--hyperparameter-mode",
        choices=["bayes", "fixed_paper"],
        default="bayes",
        help="Use BayesSearchCV tuning, or fixed AD hyperparameters from the paper's optimized public code.",
    )
    parser.add_argument("--paper-lasso-alpha", type=float, default=0.15135923730480524)
    parser.add_argument("--paper-vssrfe-c", type=float, default=0.012944980118048744)
    parser.add_argument("--paper-vssrfe-n-genes", type=int, default=159)
    parser.add_argument("--disable-fixed-vssrfe-n-genes", action="store_true")
    parser.add_argument(
        "--xgboost-device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device for XGBoost. auto uses the portable CPU path; request cuda explicitly when supported.",
    )
    parser.add_argument("--scaler", choices=["standard", "minmax"], default="standard")
    parser.add_argument("--vssrfe-min-genes", type=int, default=1)
    parser.add_argument("--vssrfe-max-genes", type=int, default=200)
    parser.add_argument("--vssrfe-step-genes", type=int, default=1)
    parser.add_argument("--vssrfe-extra-gene-counts", type=int, nargs="*", default=[])
    parser.add_argument("--deep-epochs", type=int, default=None, help="Legacy override for both standalone deep models.")
    parser.add_argument("--vae-classifier-epochs", type=int, default=1000)
    parser.add_argument("--cnn-epochs", type=int, default=100)
    parser.add_argument("--vae-epochs", type=int, default=None, help="Epoch cap for the VAE latent representation; profile default is 1000.")
    parser.add_argument("--vae-architecture", choices=["basic", "batchnorm", "batchnorm_dropout"], default="basic")
    parser.add_argument("--vae-learning-rate", type=float, default=None, help="VAE feature encoder LR; paper=1e-5, public_code=1e-4.")
    parser.add_argument("--vae-backend", choices=["tensorflow", "torch"], default="tensorflow")
    parser.add_argument("--vae-device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--vae-classifier-learning-rate", type=float, default=0.001)
    parser.add_argument(
        "--vae-reconstruction-loss",
        choices=["categorical_crossentropy", "binary_crossentropy"],
        default="categorical_crossentropy",
    )
    parser.add_argument("--deep-batch-size", type=int, default=32)
    parser.add_argument("--deep-patience", type=int, default=3)
    parser.add_argument("--threshold-mode", choices=["fixed_0_5", "validation_macro_f1", "validation_accuracy", "validation_accuracy_macro_f1"], default="fixed_0_5")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--strict-v2", action="store_true", help="Run the unified strict leakage-free V2 configuration.")
    parser.add_argument("--overwrite-results", action="store_true", help="Delete the result root before running.")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def apply_implementation_profile(args: argparse.Namespace) -> None:
    """Resolve only defaults that conflict between the article and public code."""

    if args.vae_epochs is None:
        args.vae_epochs = 1000
    if args.vae_learning_rate is None:
        args.vae_learning_rate = 1e-5 if args.implementation_profile == "paper" else 1e-4


def standalone_epochs(args: argparse.Namespace, model_name: str) -> int:
    if args.deep_epochs is not None:
        return int(args.deep_epochs)
    if model_name == "vae_classifier":
        return int(args.vae_classifier_epochs)
    if model_name == "cnn":
        return int(args.cnn_epochs)
    raise ValueError(model_name)


def requested_configuration_matrix(args: argparse.Namespace) -> list[tuple[str, str]]:
    matrix = [(feature_set, model_name) for feature_set in args.feature_sets for model_name in args.models]
    if args.include_deep:
        matrix.extend(("all_genes", model_name) for model_name in PAPER_STANDALONE_MODELS)
    return matrix


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


def stratified_inner_split(pool_idx: np.ndarray, y_values: np.ndarray, val_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    train_idx, val_idx = train_test_split(
        pool_idx,
        test_size=val_ratio,
        random_state=seed,
        stratify=y_values[pool_idx],
    )
    return np.asarray(train_idx, dtype=np.int64), np.asarray(val_idx, dtype=np.int64)


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
    splits: list[dict[str, Any]] = []

    for repeat in range(1, repeats + 1):
        repeat_seed = seed + repeat - 1
        for scenario_idx, scenario in enumerate(scenarios, start=1):
            if scenario == "test_gse63060":
                pool_idx = all_idx[batch.to_numpy() == "GSE63061"]
                test_idx = all_idx[batch.to_numpy() == "GSE63060"]
            elif scenario == "test_gse63061":
                pool_idx = all_idx[batch.to_numpy() == "GSE63060"]
                test_idx = all_idx[batch.to_numpy() == "GSE63061"]
            elif scenario == "shared_test":
                train_parts: list[np.ndarray] = []
                test_parts: list[np.ndarray] = []
                for batch_name in ("GSE63060", "GSE63061"):
                    batch_idx = all_idx[batch.to_numpy() == batch_name]
                    batch_train, batch_test = train_test_split(
                        batch_idx,
                        test_size=shared_test_size,
                        random_state=repeat_seed,
                        stratify=y_values[batch_idx],
                    )
                    train_parts.append(np.asarray(batch_train, dtype=np.int64))
                    test_parts.append(np.asarray(batch_test, dtype=np.int64))
                pool_idx = np.concatenate(train_parts)
                test_idx = np.concatenate(test_parts)
            else:
                raise ValueError(f"Unsupported batch scenario: {scenario}")

            train_idx, val_idx = stratified_inner_split(
                np.asarray(pool_idx, dtype=np.int64),
                y_values,
                inner_val_ratio,
                repeat_seed * 100 + len(splits) + 1,
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
                    "pool_idx": np.asarray(pool_idx, dtype=np.int64),
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


def make_scaler(name: str):
    if name == "standard":
        return StandardScaler()
    if name == "minmax":
        return MinMaxScaler()
    raise ValueError(f"Unsupported scaler: {name}")


def preprocess_fold(x_train: pd.DataFrame, x_val: pd.DataFrame, x_test: pd.DataFrame, scaler_name: str):
    imputer = SimpleImputer(strategy="median")
    scaler = make_scaler(scaler_name)
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
            "scaler": f"{scaler.__class__.__name__} fit on train_inner",
            "n_features": int(len(columns)),
        },
    )


def preprocess_full_initial(x: pd.DataFrame, scaler_name: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    imputer = SimpleImputer(strategy="median")
    scaler = make_scaler(scaler_name)
    x_imp = imputer.fit_transform(x)
    x_scaled = scaler.fit_transform(x_imp)
    return (
        pd.DataFrame(x_scaled, index=x.index, columns=x.columns),
        {
            "imputer": "SimpleImputer(strategy=median) fit once on full input before CV",
            "scaler": f"{scaler.__class__.__name__} fit once on full input before CV",
            "n_features": int(len(x.columns)),
        },
    )


def build_feature_set(
    name: str,
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_val: pd.DataFrame,
    x_test: pd.DataFrame,
    args: argparse.Namespace,
    seed: int,
) -> FeatureSet:
    if name == "all_genes":
        return all_genes(x_train, x_val, x_test)
    if name == "lasso":
        fixed_alpha = args.paper_lasso_alpha if args.hyperparameter_mode == "fixed_paper" else None
        return lasso_features(x_train, y_train, x_val, x_test, seed, args.feature_bayes_iter, args.cv_folds, args.n_jobs, fixed_alpha)
    if name == "vssrfe_lr":
        fixed_c = args.paper_vssrfe_c if args.hyperparameter_mode == "fixed_paper" else None
        fixed_n = None if args.disable_fixed_vssrfe_n_genes else (args.paper_vssrfe_n_genes if args.hyperparameter_mode == "fixed_paper" else None)
        return vssrfe_lr_features(
            x_train,
            y_train,
            x_val,
            x_test,
            seed,
            args.feature_bayes_iter,
            args.cv_folds,
            args.n_jobs,
            args.vssrfe_min_genes,
            args.vssrfe_max_genes,
            args.vssrfe_step_genes,
            args.vssrfe_extra_gene_counts,
            fixed_c,
            fixed_n,
        )
    if name == "vae_latent":
        return vae_latent_features(
            x_train,
            x_val,
            x_test,
            seed,
            args.vae_epochs,
            args.deep_batch_size,
            args.deep_patience,
            args.vae_architecture,
            args.vae_learning_rate,
            args.vae_reconstruction_loss,
            args.implementation_profile,
            args.vae_backend,
            args.vae_device,
        )
    if name == "knowledge_genes":
        feature_set = knowledge_genes_features(
            x_train,
            x_val,
            x_test,
            args.knowledge_genes_file,
            args.knowledge_mad_top_k,
        )
        status = str(feature_set.metadata.get("status", "ok"))
        if status != "ok" and not args.allow_incomplete_knowledge:
            feature_set.metadata.update(
                {
                    "status": "skipped_missing_curated_knowledge_list",
                    "partial_feature_set_available": True,
                    "required_opt_in": "--allow-incomplete-knowledge",
                    "ranking_policy": "not evaluated or ranked under the paper knowledge_genes label",
                }
            )
        elif status != "ok":
            feature_set.name = "mad_top3000_only"
            feature_set.metadata.update(
                {
                    "status": "ok_incomplete_ablation",
                    "reported_feature_name": "mad_top3000_only",
                    "ranking_policy": "explicit ablation; not the paper's complete knowledge feature set",
                }
            )
        return feature_set
    raise ValueError(f"Unsupported feature set: {name}")


def build_initial_feature_sets(
    x_full: pd.DataFrame,
    y_full: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, FeatureSet]:
    feature_sets: dict[str, FeatureSet] = {}
    for feature_name in args.feature_sets:
        feature_set = build_feature_set(
            feature_name,
            x_full,
            y_full,
            x_full,
            x_full,
            args,
            args.seed,
        )
        feature_set.metadata["selection_scope"] = "fit once on full dataset before 5-fold CV"
        feature_sets[feature_name] = feature_set
    return feature_sets


def subset_initial_feature_set(feature_set: FeatureSet, train_idx: np.ndarray, test_idx: np.ndarray) -> FeatureSet:
    return FeatureSet(
        name=feature_set.name,
        x_train=feature_set.x_train[train_idx],
        x_val=np.empty((0, feature_set.x_train.shape[1]), dtype=np.float32),
        x_test=feature_set.x_train[test_idx],
        feature_names=feature_set.feature_names,
        metadata=feature_set.metadata,
        selected_genes=feature_set.selected_genes,
    )


def subset_full_dataset_feature_set(feature_set: FeatureSet, train_idx: np.ndarray, val_idx: np.ndarray, test_idx: np.ndarray) -> FeatureSet:
    return FeatureSet(
        name=feature_set.name,
        x_train=feature_set.x_train[train_idx],
        x_val=feature_set.x_train[val_idx],
        x_test=feature_set.x_train[test_idx],
        feature_names=feature_set.feature_names,
        metadata={**feature_set.metadata, "selection_scope": "fit once on full dataset before batch-holdout splits"},
        selected_genes=feature_set.selected_genes,
    )


def run_sklearn_model(
    feature_set: FeatureSet,
    y_train: np.ndarray,
    y_val: np.ndarray,
    x_test_ids: pd.Index,
    y_test: np.ndarray,
    model_name: str,
    run_dir: Path,
    args: argparse.Namespace,
    seed: int,
    split_meta: dict[str, Any],
) -> dict[str, Any]:
    estimator, tuning_meta = fit_tuned_model(
        model_name,
        feature_set.x_train,
        y_train,
        seed,
        args.bayes_iter,
        args.cv_folds,
        args.n_jobs,
        feature_set.name,
        args.hyperparameter_mode,
        args.xgboost_device,
    )
    x_test = np.nan_to_num(np.asarray(feature_set.x_test, dtype=np.float32), nan=0.0, posinf=10.0, neginf=-10.0)
    x_test = np.clip(x_test, -10.0, 10.0)
    scores = positive_scores(estimator, x_test)
    if args.threshold_mode == "fixed_0_5" or len(y_val) == 0:
        threshold = 0.5
        threshold_meta = {"threshold": threshold, "threshold_mode": "fixed_0_5"}
    else:
        val_x = np.nan_to_num(np.asarray(feature_set.x_val, dtype=np.float32), nan=0.0, posinf=10.0, neginf=-10.0)
        val_x = np.clip(val_x, -10.0, 10.0)
        val_scores = positive_scores(estimator, val_x)
        metric = args.threshold_mode.replace("validation_", "")
        threshold, threshold_meta = calibrate_threshold(y_val, val_scores, metric=metric)
        threshold_meta["threshold_mode"] = args.threshold_mode
    y_pred = (scores >= threshold).astype(int)
    metrics = evaluate_binary(y_test, scores, y_pred)
    row = {**split_meta, "feature_set": feature_set.name, "model": model_name, **metrics}
    tuning_meta = {**tuning_meta, "threshold_calibration": threshold_meta}
    write_run_artifacts(run_dir, row, x_test_ids, y_test, scores, y_pred, feature_set, tuning_meta)
    return row


def run_feature_deep_model(
    feature_set: FeatureSet,
    y_train: np.ndarray,
    y_val: np.ndarray,
    x_test_ids: pd.Index,
    y_test: np.ndarray,
    model_name: str,
    run_dir: Path,
    args: argparse.Namespace,
    seed: int,
    split_meta: dict[str, Any],
) -> dict[str, Any]:
    checkpoint = run_dir / f"{model_name}.keras"
    if model_name == "vae_classifier":
        scores, metric_values, meta = train_vae_classifier(
            feature_set.x_train,
            y_train,
            feature_set.x_val,
            y_val,
            feature_set.x_test,
            y_test,
            seed,
            standalone_epochs(args, model_name),
            args.deep_batch_size,
            args.deep_patience,
            checkpoint,
            args.vae_architecture,
            args.vae_classifier_learning_rate,
        )
    elif model_name == "cnn":
        scores, metric_values, meta = train_cnn_classifier(
            feature_set.x_train,
            y_train,
            feature_set.x_val,
            y_val,
            feature_set.x_test,
            y_test,
            seed,
            standalone_epochs(args, model_name),
            args.deep_batch_size,
            args.deep_patience,
            checkpoint,
        )
    else:
        raise ValueError(model_name)
    y_pred = (scores >= 0.5).astype(int)
    row = {**split_meta, "feature_set": feature_set.name, "model": model_name, **metric_values}
    write_run_artifacts(run_dir, row, x_test_ids, y_test, scores, y_pred, feature_set, meta)
    return row


def run_initial_fs_5cv(args: argparse.Namespace, x: pd.DataFrame, y: pd.Series) -> list[dict[str, Any]]:
    x_full, preprocessing_meta = preprocess_full_initial(x, args.scaler)
    y_values = y.to_numpy(dtype=np.int64)
    write_json(
        args.result_root / "initial_feature_selection_manifest.json",
        {
            "protocol": "initial_fs_5cv",
            "scope": "Feature selection/representation fitted once on the full dataset before 5-fold CV.",
            "feature_sets": {},
            **preprocessing_meta,
        },
    )

    cv = StratifiedKFold(n_splits=args.outer_folds, shuffle=True, random_state=args.seed)
    folds = list(cv.split(np.arange(len(y_values)), y_values))
    run_rows: list[dict[str, Any]] = []
    feature_manifest: dict[str, Any] = {}
    for feature_name in args.feature_sets:
        print(f"Selecting feature set={feature_name} on full dataset protocol=initial_fs_5cv", flush=True)
        initial_feature_set = build_feature_set(feature_name, x_full, y_values, x_full, x_full, args, args.seed)
        initial_feature_set.metadata["selection_scope"] = "fit once on full dataset before 5-fold CV"
        feature_manifest[feature_name] = initial_feature_set.metadata
        write_json(
            args.result_root / "initial_feature_selection_manifest.json",
            {
                "protocol": "initial_fs_5cv",
                "scope": "Feature selection/representation fitted once on the full dataset before 5-fold CV.",
                "feature_sets": feature_manifest,
                **preprocessing_meta,
            },
        )
        if initial_feature_set.selected_genes is not None:
            feature_dir = args.result_root / "initial_feature_sets" / feature_name
            feature_dir.mkdir(parents=True, exist_ok=True)
            (feature_dir / "selected_genes.txt").write_text("\n".join(initial_feature_set.selected_genes) + "\n", encoding="utf-8")
            write_json(feature_dir / "feature_manifest.json", initial_feature_set.metadata)

        for fold, (train_idx, test_idx) in enumerate(folds, start=1):
            split_seed = args.seed * 1000 + fold
            y_train = y_values[train_idx]
            y_test = y_values[test_idx]
            split_meta = {
                "repeat": 1,
                "fold": fold,
                "seed": split_seed,
                "n_train_inner": len(y_train),
                "n_val_inner": 0,
                "n_outer_test": len(y_test),
                "protocol": "initial_fs_5cv",
            }
            fold_dir = args.result_root / "runs" / "repeat_01" / f"fold_{fold:02d}"
            write_json(fold_dir / "split_manifest.json", {**split_meta, **preprocessing_meta})
            if initial_feature_set.metadata.get("status", "ok").startswith("skipped"):
                run_dir = fold_dir / feature_name / "_skipped"
                run_dir.mkdir(parents=True, exist_ok=True)
                write_json(run_dir / "run_manifest.json", {"feature_set": initial_feature_set.metadata, **split_meta})
                continue
            fold_feature_set = subset_initial_feature_set(initial_feature_set, train_idx, test_idx)
            for model_name in args.models:
                run_dir = fold_dir / feature_name / model_name
                if args.skip_existing and (run_dir / "metrics.csv").exists():
                    continue
                print(f"Running fold={fold} feature={feature_name} model={model_name} protocol=initial_fs_5cv", flush=True)
                row = run_sklearn_model(
                    fold_feature_set,
                    y_train,
                    np.empty((0,), dtype=np.int64),
                    x.index[test_idx],
                    y_test,
                    model_name,
                    run_dir,
                    args,
                    split_seed,
                    split_meta,
                )
                run_rows.append(row)
    return run_rows


def write_run_artifacts(
    run_dir: Path,
    metric_row: dict[str, Any],
    sample_ids: pd.Index,
    y_test: np.ndarray,
    scores: np.ndarray,
    y_pred: np.ndarray,
    feature_set: FeatureSet,
    hyperparameters: dict[str, Any],
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([metric_row]).to_csv(run_dir / "metrics.csv", index=False)
    predictions_frame(sample_ids, y_test, scores, y_pred).to_csv(run_dir / "predictions.csv", index=False)
    confusion_frame(y_test, y_pred).to_csv(run_dir / "confusion_matrix.csv")
    classification_report_frame(y_test, y_pred).to_csv(run_dir / "classification_report.csv")
    write_json(run_dir / "hyperparameters.json", hyperparameters)
    write_json(run_dir / "run_manifest.json", {"metrics": metric_row, "feature_set": feature_set.metadata})
    if feature_set.selected_genes is not None:
        (run_dir / "selected_genes.txt").write_text("\n".join(feature_set.selected_genes) + "\n", encoding="utf-8")


def run_deep_standalone(
    model_name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    x_test_ids: pd.Index,
    run_dir: Path,
    args: argparse.Namespace,
    seed: int,
    split_meta: dict[str, Any],
) -> dict[str, Any]:
    checkpoint = run_dir / f"{model_name}.keras"
    if model_name == "vae_classifier":
        scores, metric_values, meta = train_vae_classifier(
            x_train,
            y_train,
            x_val,
            y_val,
            x_test,
            y_test,
            seed,
            standalone_epochs(args, model_name),
            args.deep_batch_size,
            args.deep_patience,
            checkpoint,
            args.vae_architecture,
            args.vae_classifier_learning_rate,
        )
    elif model_name == "cnn":
        scores, metric_values, meta = train_cnn_classifier(
            x_train,
            y_train,
            x_val,
            y_val,
            x_test,
            y_test,
            seed,
            standalone_epochs(args, model_name),
            args.deep_batch_size,
            args.deep_patience,
            checkpoint,
        )
    else:
        raise ValueError(model_name)
    y_pred = (scores >= 0.5).astype(int)
    feature_set = FeatureSet(
        name="all_genes",
        x_train=x_train,
        x_val=x_val,
        x_test=x_test,
        feature_names=[],
        metadata={"status": "ok", "method": "standalone_deep_all_genes"},
    )
    row = {**split_meta, "feature_set": "all_genes", "model": model_name, **metric_values}
    write_run_artifacts(run_dir, row, x_test_ids, y_test, scores, y_pred, feature_set, meta)
    return row


def summarize_and_audit(args: argparse.Namespace, x: pd.DataFrame, y: pd.Series, run_rows: list[dict[str, Any]]) -> None:
    all_metrics, summary = summarize(args.result_root)
    all_metrics.to_csv(args.result_root / "all_metrics.csv", index=False)
    summary.to_csv(args.result_root / "summary_by_method.csv", index=False)
    summary.to_csv(args.result_root / "ranking_by_pr_auc.csv", index=False)
    if not summary.empty:
        summary.sort_values("roc_auc_mean", ascending=False).to_csv(args.result_root / "ranking_by_roc_auc.csv", index=False)
        summary.sort_values("macro_f1_mean", ascending=False).to_csv(args.result_root / "ranking_by_macro_f1.csv", index=False)
        summary.sort_values("accuracy_mean", ascending=False).to_csv(args.result_root / "ranking_by_accuracy.csv", index=False)
    write_audit(args.result_root, args, x, y, run_rows)


def run_batch_holdout(args: argparse.Namespace, x: pd.DataFrame, y: pd.Series) -> list[dict[str, Any]]:
    batch = infer_batch_labels(x.index, args.gse63060_metadata, args.gse63061_metadata)
    batch_manifest = (
        pd.DataFrame({"sample_id": x.index.astype(str), "batch": batch.to_numpy(), "label": y.to_numpy(dtype=np.int64)})
        .sort_values(["batch", "label", "sample_id"])
        .reset_index(drop=True)
    )
    batch_manifest.to_csv(args.result_root / "batch_manifest.csv", index=False)

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
        splits = make_batch_holdout_splits(
            y=y,
            batch=batch,
            scenarios=args.batch_scenarios,
            repeats=args.repeats,
            inner_val_ratio=args.inner_val_ratio,
            shared_test_size=args.shared_test_size,
            seed=args.seed,
        )
    run_rows: list[dict[str, Any]] = []
    full_feature_sets: dict[str, FeatureSet] = {}
    full_preprocessing_meta: dict[str, Any] = {}
    if args.artifact_scope == "full_dataset":
        print("Fitting Nature 2023 preprocessing and feature sets on full dataset (intentional leakage).", flush=True)
        x_full, full_preprocessing_meta = preprocess_full_initial(x, args.scaler)
        y_full = y.to_numpy(dtype=np.int64)
        for feature_name in args.feature_sets:
            full_fs = build_feature_set(feature_name, x_full, y_full, x_full, x_full, args, args.seed)
            full_fs.metadata["selection_scope"] = "fit once on full dataset before batch-holdout splits"
            full_fs.metadata["artifact_scope"] = "full_dataset"
            full_feature_sets[feature_name] = full_fs
            feature_dir = args.result_root / "full_dataset_artifacts" / feature_name
            feature_dir.mkdir(parents=True, exist_ok=True)
            write_json(feature_dir / "feature_manifest.json", full_fs.metadata)
            if full_fs.selected_genes is not None:
                (feature_dir / "selected_genes.txt").write_text("\n".join(full_fs.selected_genes) + "\n", encoding="utf-8")

    for split in splits:
        train_idx = split["train_inner_idx"]
        val_idx = split["val_inner_idx"]
        test_idx = split["outer_test_idx"]
        if set(train_idx.tolist()) & set(val_idx.tolist()) or set(train_idx.tolist()) & set(test_idx.tolist()) or set(val_idx.tolist()) & set(test_idx.tolist()):
            raise ValueError(f"Split overlap detected for scenario={split['scenario']} repeat={split['repeat']}.")

        split_seed = int(split["seed"]) if split.get("split_source") else int(split["seed"] * 1000 + split["scenario_index"] * 100 + split["repeat"])
        x_train_raw, y_train, x_val_raw, y_val, x_test_raw, y_test = split_frames(x, y, train_idx, val_idx, test_idx)
        if args.artifact_scope == "full_dataset":
            x_train = x_val = x_test = None
            preprocessing_meta = {**full_preprocessing_meta, "artifact_scope": "full_dataset"}
        else:
            x_train, x_val, x_test, preprocessing_meta = preprocess_fold(x_train_raw, x_val_raw, x_test_raw, args.scaler)
        split_meta = {
            "protocol": "batch_holdout",
            "scenario": split["scenario"],
            "repeat": split["repeat"],
            "fold": 1,
            "seed": split_seed,
            "n_train_inner": len(y_train),
            "n_val_inner": len(y_val),
            "n_outer_test": len(y_test),
            "train_batches": "|".join(sorted(batch.iloc[train_idx].unique().tolist())),
            "val_batches": "|".join(sorted(batch.iloc[val_idx].unique().tolist())),
            "test_batches": "|".join(sorted(batch.iloc[test_idx].unique().tolist())),
        }
        split_dir = args.result_root / "runs" / split["scenario"] / f"repeat_{split['repeat']:02d}"
        write_json(
            split_dir / "split_manifest.json",
            {
                **split_meta,
                **preprocessing_meta,
                "train_inner_ids": x.index[train_idx].astype(str).tolist(),
                "val_inner_ids": x.index[val_idx].astype(str).tolist(),
                "outer_test_ids": x.index[test_idx].astype(str).tolist(),
            },
        )

        for feature_name in args.feature_sets:
            pending_models = [
                model_name
                for model_name in args.models
                if not (args.skip_existing and (split_dir / feature_name / model_name / "metrics.csv").exists())
            ]
            if not pending_models:
                continue
            feature_started = time.perf_counter()
            print(
                f"Preparing scenario={split['scenario']} repeat={split['repeat']} feature={feature_name} "
                f"mode={args.hyperparameter_mode}",
                flush=True,
            )
            if args.artifact_scope == "full_dataset":
                feature_set = subset_full_dataset_feature_set(full_feature_sets[feature_name], train_idx, val_idx, test_idx)
            else:
                feature_set = build_feature_set(feature_name, x_train, y_train, x_val, x_test, args, split_seed)
            print(
                f"Prepared scenario={split['scenario']} repeat={split['repeat']} feature={feature_set.name} "
                f"status={feature_set.metadata.get('status', 'ok')} n_features={feature_set.x_train.shape[1]} "
                f"elapsed_s={time.perf_counter() - feature_started:.1f}",
                flush=True,
            )
            if feature_set.metadata.get("status", "ok").startswith("skipped"):
                run_dir = split_dir / feature_name / "_skipped"
                run_dir.mkdir(parents=True, exist_ok=True)
                write_json(run_dir / "run_manifest.json", {"feature_set": feature_set.metadata, **split_meta})
                continue
            for model_name in pending_models:
                run_dir = split_dir / feature_name / model_name
                print(
                    f"Running scenario={split['scenario']} repeat={split['repeat']} feature={feature_name} model={model_name}",
                    flush=True,
                )
                row = run_sklearn_model(
                    feature_set,
                    y_train,
                    y_val,
                    x_test_raw.index,
                    y_test,
                    model_name,
                    run_dir,
                    args,
                    split_seed,
                    split_meta,
                )
                run_rows.append(row)

        if args.include_deep:
            if args.artifact_scope == "full_dataset":
                raise ValueError("Standalone deep models require --artifact-scope train_inner in the leakage-safe runner.")
            for deep_name in PAPER_STANDALONE_MODELS:
                run_dir = split_dir / "all_genes" / deep_name
                if args.skip_existing and (run_dir / "metrics.csv").exists():
                    continue
                print(f"Running scenario={split['scenario']} repeat={split['repeat']} feature=all_genes model={deep_name}", flush=True)
                row = run_deep_standalone(
                    deep_name,
                    x_train.to_numpy(dtype=np.float32),
                    y_train,
                    x_val.to_numpy(dtype=np.float32),
                    y_val,
                    x_test.to_numpy(dtype=np.float32),
                    y_test,
                    x_test.index,
                    run_dir,
                    args,
                    split_seed,
                    split_meta,
                )
                run_rows.append(row)
    return run_rows


def write_audit(result_root: Path, args: argparse.Namespace, x: pd.DataFrame, y: pd.Series, run_rows: list[dict[str, Any]]) -> None:
    knowledge_status = "not_requested"
    if "knowledge_genes" in args.feature_sets:
        knowledge_status = "complete" if args.knowledge_genes_file and args.knowledge_genes_file.exists() else "incomplete_missing_curated"
    audit = pd.DataFrame(
        [
            {"item": "input_dataset", "status": "pass", "details": f"{len(y)} samples x {x.shape[1]} genes from {args.x_file}"},
            {
                "item": "configuration_matrix",
                "status": "pass",
                "details": (
                    f"{len(requested_configuration_matrix(args))} requested configurations: "
                    f"{len(args.feature_sets)} feature sets x {len(args.models)} classical models"
                    + (f" + {len(PAPER_STANDALONE_MODELS)} standalone all-gene deep models" if args.include_deep else "")
                    + "; deep models are never crossed with feature sets."
                ),
            },
            {
                "item": "implementation_profile",
                "status": "documented_conflict",
                "details": (
                    f"profile={args.implementation_profile}; VAE-feature lr={args.vae_learning_rate}, "
                    f"loss={args.vae_reconstruction_loss}, epochs={args.vae_epochs}. "
                    "Paper reports lr=1e-5; public AD code uses lr=1e-4."
                ),
            },
            {
                "item": "protocol",
                "status": "pass",
                "details": (
                    f"Initial feature selection once, then 1 repeat x {args.outer_folds}-fold CV."
                    if args.protocol == "initial_fs_5cv"
                    else (
                        f"{args.repeats} repeats across batch holdout scenarios {', '.join(args.batch_scenarios)}; "
                        f"shared_test_size={args.shared_test_size}; inner val ratio {args.inner_val_ratio}"
                        if args.protocol == "batch_holdout"
                        else f"{args.repeats} repeats x {args.outer_folds} outer folds; inner val ratio {args.inner_val_ratio}"
                    )
                ),
            },
            {
                "item": "preprocessing_scope",
                "status": "paper_like" if args.protocol == "initial_fs_5cv" else "pass",
                "details": (
                    f"Median imputer and {args.scaler} scaler fit once on the full dataset before CV, matching initial-FS protocol."
                    if args.protocol == "initial_fs_5cv"
                    else f"Median imputer and {args.scaler} scaler fit only on train_inner for every outer fold."
                ),
            },
            {
                "item": "paper_scaler",
                "status": "pass" if args.scaler == "standard" else "deviation",
                "details": f"Using scaler={args.scaler}; Kelly et al. report StandardScaler.",
            },
            {
                "item": "hyperparameter_mode",
                "status": "paper_fixed" if args.hyperparameter_mode == "fixed_paper" else "tuned",
                "details": (
                    "Using fixed AD hyperparameters extracted from the published reference implementation."
                    if args.hyperparameter_mode == "fixed_paper"
                    else (
                        f"Using PR-AUC BayesSearchCV with model n_iter={args.bayes_iter}, "
                        f"feature n_iter={args.feature_bayes_iter}, cv_folds={args.cv_folds}."
                    )
                ),
            },
            {
                "item": "xgboost_device",
                "status": "configured",
                "details": f"xgboost_device={args.xgboost_device}; TensorFlow/Keras GPU usage depends on TensorFlow detecting CUDA GPUs.",
            },
            {
                "item": "feature_selection_scope",
                "status": "paper_like" if args.protocol == "initial_fs_5cv" else "pass",
                "details": (
                    "Feature selectors/representations are fit once on the full dataset before 5-fold CV."
                    if args.protocol == "initial_fs_5cv"
                    else "Feature selectors are fit only on train_inner and applied to val_inner/outer_test."
                ),
            },
            {
                "item": "outer_test_scope",
                "status": "paper_like" if args.protocol == "initial_fs_5cv" else "pass",
                "details": (
                    "CV validation folds are held out for model fitting, but preprocessing and feature selection already saw all samples."
                    if args.protocol == "initial_fs_5cv"
                    else "Outer test is evaluated once per run and not used for tuning or checkpoint selection."
                ),
            },
            {
                "item": "knowledge_genes",
                "status": knowledge_status,
                "details": (
                    f"train-only top-{args.knowledge_mad_top_k} MAD union curated={args.knowledge_genes_file}. "
                    "The exact authors' prev_features.txt is absent from the public repository."
                ),
            },
            {"item": "completed_runs", "status": "pass", "details": str(len(run_rows))},
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
    args.knowledge_genes_file = resolve(args.knowledge_genes_file) if args.knowledge_genes_file else None
    args.gse63060_metadata = resolve(args.gse63060_metadata)
    args.gse63061_metadata = resolve(args.gse63061_metadata)
    if args.deep_feature_models:
        raise ValueError(
            "--deep-feature-models is not a Kelly et al. configuration. "
            "CNN and the paper-named VAE classifier are standalone all-gene models; use --include-deep."
        )
    if args.strict_v2:
        args.protocol = "batch_holdout"
        args.repeats = 10
        args.batch_scenarios = ["shared_test"]
        args.artifact_scope = "train_inner"
        args.scaler = "standard"
        args.feature_sets = list(PAPER_FEATURE_SETS)
        args.models = available_model_names()
        args.hyperparameter_mode = "bayes"
        args.cv_folds = 5
        args.vssrfe_min_genes = 1
        args.vssrfe_max_genes = 200
        args.vssrfe_step_genes = 1
        args.vssrfe_extra_gene_counts = []
        args.disable_fixed_vssrfe_n_genes = True
        args.threshold_mode = "fixed_0_5"
        args.include_deep = True
        args.overwrite_results = not args.skip_existing
    if args.smoke:
        args.repeats = 1
        args.outer_folds = 5
        args.bayes_iter = min(args.bayes_iter, 2)
        args.feature_bayes_iter = min(args.feature_bayes_iter, 2)
        args.cv_folds = 2
        args.vssrfe_min_genes = min(args.vssrfe_min_genes, 1)
        args.vssrfe_max_genes = min(args.vssrfe_max_genes, 3)
        args.vssrfe_step_genes = 1
        args.vssrfe_extra_gene_counts = []
        if args.deep_epochs is not None:
            args.deep_epochs = min(args.deep_epochs, 1)
        args.vae_classifier_epochs = min(args.vae_classifier_epochs, 1)
        args.cnn_epochs = min(args.cnn_epochs, 1)
        args.vae_epochs = min(args.vae_epochs, 1)
        args.deep_patience = min(args.deep_patience, 1)
        args.n_jobs = 1

    if args.overwrite_results and args.result_root.exists():
        shutil.rmtree(args.result_root)
    args.result_root.mkdir(parents=True, exist_ok=True)
    write_json(args.result_root / "args.json", vars(args))
    x, y = load_ad_mci_dataset(args.x_file, args.y_file)
    if args.protocol == "initial_fs_5cv":
        args.repeats = 1
        run_rows = run_initial_fs_5cv(args, x, y)
        summarize_and_audit(args, x, y, run_rows)
        print(f"Finished. Results written to {args.result_root}", flush=True)
        return

    if args.protocol == "batch_holdout":
        run_rows = run_batch_holdout(args, x, y)
        summarize_and_audit(args, x, y, run_rows)
        print(f"Finished. Results written to {args.result_root}", flush=True)
        return

    max_outer = 1 if args.smoke else None
    splits = repeated_stratified_nested_splits(y, args.repeats, args.outer_folds, args.inner_val_ratio, args.seed, max_outer)
    run_rows: list[dict[str, Any]] = []

    for split in splits:
        assert_no_split_overlap(split)
        split_seed = split.seed * 1000 + split.fold
        x_train_raw, y_train, x_val_raw, y_val, x_test_raw, y_test = split_frames(
            x, y, split.train_inner_idx, split.val_inner_idx, split.outer_test_idx
        )
        x_train, x_val, x_test, preprocessing_meta = preprocess_fold(x_train_raw, x_val_raw, x_test_raw, args.scaler)
        split_meta = {
            "scenario": "shared_test",
            "repeat": split.repeat,
            "fold": split.fold,
            "seed": split_seed,
            "n_train_inner": len(y_train),
            "n_val_inner": len(y_val),
            "n_outer_test": len(y_test),
        }
        fold_dir = args.result_root / "runs" / "shared_test" / f"repeat_{split.repeat:02d}" / f"fold_{split.fold:02d}"
        write_json(fold_dir / "split_manifest.json", {**split_meta, **preprocessing_meta})

        for feature_name in args.feature_sets:
            feature_set = build_feature_set(feature_name, x_train, y_train, x_val, x_test, args, split_seed)
            if feature_set.metadata.get("status", "ok").startswith("skipped"):
                run_dir = fold_dir / feature_name / "_skipped"
                run_dir.mkdir(parents=True, exist_ok=True)
                write_json(run_dir / "run_manifest.json", {"feature_set": feature_set.metadata, **split_meta})
                continue
            for model_name in args.models:
                run_dir = fold_dir / feature_name / model_name
                if args.skip_existing and (run_dir / "metrics.csv").exists():
                    continue
                print(f"Running repeat={split.repeat} fold={split.fold} feature={feature_name} model={model_name}", flush=True)
                row = run_sklearn_model(feature_set, y_train, y_val, x_test.index, y_test, model_name, run_dir, args, split_seed, split_meta)
                run_rows.append(row)

        if args.include_deep:
            for deep_name in PAPER_STANDALONE_MODELS:
                run_dir = fold_dir / "all_genes" / deep_name
                if args.skip_existing and (run_dir / "metrics.csv").exists():
                    continue
                print(f"Running repeat={split.repeat} fold={split.fold} feature=all_genes model={deep_name}", flush=True)
                row = run_deep_standalone(
                    deep_name,
                    x_train.to_numpy(dtype=np.float32),
                    y_train,
                    x_val.to_numpy(dtype=np.float32),
                    y_val,
                    x_test.to_numpy(dtype=np.float32),
                    y_test,
                    x_test.index,
                    run_dir,
                    args,
                    split_seed,
                    split_meta,
                )
                run_rows.append(row)

    summarize_and_audit(args, x, y, run_rows)
    print(f"Finished. Results written to {args.result_root}", flush=True)


if __name__ == "__main__":
    main()
