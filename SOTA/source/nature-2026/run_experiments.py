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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import MinMaxScaler

ROOT = Path(__file__).resolve().parents[3]
MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))
COMMON_DIR = Path(__file__).resolve().parents[1]
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

from augmentation import augment_training_data
from balancing import balance_training_data
from data import assert_no_overlap, infer_batch_labels, load_ad_mci_dataset, make_batch_holdout_splits, make_stratified_5cv_splits
from feature_selection import PAPER_INTEGRATED_K, select_features
from metrics import classification_report_frame, confusion_frame, predictions_frame
from models import fit_predict_deep, fit_predict_sklearn
from summarize_results import summarize
from shared_test_splits import load_task_splits, split_counts


DEFAULT_X = ROOT / "task_dataset" / "processed" / "ad_mci_binary" / "X_ad_mci.csv"
DEFAULT_Y = ROOT / "task_dataset" / "processed" / "ad_mci_binary" / "y_ad_mci.csv"
DEFAULT_RESULT_ROOT = ROOT / "results" / "SOTA" / "nature-2026"
DEFAULT_GSE63060_METADATA = ROOT / "pretraining_dataset" / "geo_downloads" / "GSE63060" / "eset_1_GPL6947" / "GSE63060_sample_metadata.tsv.gz"
DEFAULT_GSE63061_METADATA = ROOT / "pretraining_dataset" / "geo_downloads" / "GSE63061" / "eset_1_GPL10558" / "GSE63061_sample_metadata.tsv.gz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Nature 2026 leakage-safe reproduction for local AD vs MCI.")
    parser.add_argument("--x-file", type=Path, default=DEFAULT_X)
    parser.add_argument("--y-file", type=Path, default=DEFAULT_Y)
    parser.add_argument(
        "--split-manifest-dir",
        type=Path,
        default=None,
        help="Directory of TxT seed_<seed>.csv manifests. When set, use their exact shared-test sample membership.",
    )
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--protocol", choices=["batch_holdout", "stratified_5cv"], default="batch_holdout")
    parser.add_argument(
        "--batch-scenarios",
        nargs="+",
        default=["shared_test", "test_gse63060", "test_gse63061"],
        choices=["shared_test", "test_gse63060", "test_gse63061"],
    )
    parser.add_argument("--shared-test-size", type=float, default=0.2)
    parser.add_argument("--inner-val-ratio", type=float, default=0.125, help="Fraction of the post-test pool; 0.125 yields an overall 70/10/20 split when test size is 0.20.")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--training-balance",
        choices=["none", "undersample"],
        default="undersample",
        help="Balance train_inner only. 'undersample' is the paper-aligned default; validation/test are never resampled.",
    )
    parser.add_argument(
        "--artifact-scope",
        choices=["train_inner", "full_dataset"],
        default="train_inner",
        help="Fit preprocessing/feature selection on train_inner or once on the full dataset with intentional leakage.",
    )
    parser.add_argument("--gse63060-metadata", type=Path, default=DEFAULT_GSE63060_METADATA)
    parser.add_argument("--gse63061-metadata", type=Path, default=DEFAULT_GSE63061_METADATA)
    parser.add_argument(
        "--feature-selectors",
        nargs="+",
        choices=["all_genes", "chi2", "anova", "rfe", "elasticnet", "lasso", "rf_importance"],
        default=["all_genes", "chi2", "anova", "rfe", "elasticnet"],
    )
    parser.add_argument(
        "--chi2-bins",
        type=int,
        default=10,
        help="Train-fitted quantile bins used before chi-square. The paper requires binning but does not disclose this count.",
    )
    parser.add_argument("--chi2-k", type=int, default=PAPER_INTEGRATED_K["chi2"])
    parser.add_argument("--anova-k", type=int, default=PAPER_INTEGRATED_K["anova"])
    parser.add_argument("--rfe-k", type=int, default=PAPER_INTEGRATED_K["rfe"])
    parser.add_argument("--elasticnet-k", type=int, default=PAPER_INTEGRATED_K["elasticnet"])
    parser.add_argument("--lasso-k", type=int, default=PAPER_INTEGRATED_K["lasso"])
    parser.add_argument("--rf-importance-k", type=int, default=PAPER_INTEGRATED_K["rf_importance"])
    parser.add_argument("--rfe-step", type=float, default=0.2)
    parser.add_argument("--feature-count-mode", choices=["fixed", "auto", "natural"], default="fixed")
    parser.add_argument("--auto-k-values", type=int, nargs="+", default=[100, 200, 500, 1000, 2000])
    parser.add_argument("--auto-k-scoring", choices=["roc_auc", "accuracy", "f1_macro"], default="roc_auc")
    parser.add_argument("--auto-k-cv-folds", type=int, default=5)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["dnn", "cnn", "svm", "rf", "adaboost", "xgboost"],
        default=["dnn", "cnn"],
    )
    parser.add_argument("--augmentations", nargs="+", choices=["none", "gan", "ctgan"], default=["none"])
    parser.add_argument("--gan-target-size", type=int, default=2000)
    parser.add_argument("--gan-epochs", type=int, default=200)
    parser.add_argument("--gan-batch-size", type=int, default=64)
    parser.add_argument("--gan-latent-dim", type=int, default=128)
    parser.add_argument("--gan-learning-rate", type=float, default=0.001)
    parser.add_argument("--ctgan-pac", type=int, default=1, help="CTGAN PAC value; batch size must be divisible by PAC.")
    parser.add_argument("--ctgan-cuda", action="store_true", help="Enable GPU training in the external ctgan package.")
    parser.add_argument("--deep-epochs", type=int, default=100)
    parser.add_argument("--deep-patience", type=int, default=50)
    parser.add_argument("--deep-batch-size", type=int, default=32)
    parser.add_argument("--deep-learning-rate", type=float, default=0.001)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--threshold-mode", choices=["fixed_0_5", "validation_macro_f1", "validation_accuracy", "validation_accuracy_macro_f1"], default="fixed_0_5")
    parser.add_argument("--n-jobs", type=int, default=2)
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


def preprocess_split(
    x_train: pd.DataFrame,
    x_val: pd.DataFrame,
    x_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    imputer = SimpleImputer(strategy="median")
    scaler = MinMaxScaler()
    train_imp = imputer.fit_transform(x_train)
    val_imp = imputer.transform(x_val)
    test_imp = imputer.transform(x_test)
    train_scaled = np.nan_to_num(scaler.fit_transform(train_imp).astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0)
    val_scaled = np.nan_to_num(scaler.transform(val_imp).astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0)
    test_scaled = np.nan_to_num(scaler.transform(test_imp).astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0)
    return (
        pd.DataFrame(train_scaled, index=x_train.index, columns=x_train.columns),
        pd.DataFrame(val_scaled, index=x_val.index, columns=x_val.columns),
        pd.DataFrame(test_scaled, index=x_test.index, columns=x_test.columns),
        {
            "imputer": "SimpleImputer(strategy=median) fit on train_inner",
            "scaler": "MinMaxScaler fit on train_inner",
            "normalization": "MinMax, paper-compatible and chi-square compatible",
            "n_input_features": int(x_train.shape[1]),
        },
    )


def preprocess_full(x: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    imputer = SimpleImputer(strategy="median")
    scaler = MinMaxScaler()
    full_imp = imputer.fit_transform(x)
    full_scaled = np.nan_to_num(scaler.fit_transform(full_imp).astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0)
    return (
        pd.DataFrame(full_scaled, index=x.index, columns=x.columns),
        {
            "imputer": "SimpleImputer(strategy=median) fit on full dataset",
            "scaler": "MinMaxScaler fit on full dataset",
            "normalization": "MinMax, paper-compatible and chi-square compatible",
            "artifact_scope": "full_dataset",
            "n_input_features": int(x.shape[1]),
        },
    )


def k_overrides(args: argparse.Namespace) -> dict[str, int]:
    return {
        "chi2": args.chi2_k,
        "anova": args.anova_k,
        "rfe": args.rfe_k,
        "elasticnet": args.elasticnet_k,
        "lasso": args.lasso_k,
        "rf_importance": args.rf_importance_k,
    }


def select_features_with_optional_auto_k(
    feature_selector: str,
    x_source: pd.DataFrame,
    y_source: np.ndarray,
    seed: int,
    args: argparse.Namespace,
    override_k: dict[str, int],
) -> tuple[Any, pd.DataFrame | None]:
    if feature_selector == "all_genes":
        return (
            select_features(
                feature_selector,
                x_source,
                y_source,
                seed,
                n_jobs=args.n_jobs,
                chi2_bins=args.chi2_bins,
            ),
            None,
        )
    if args.feature_count_mode in {"fixed", "natural"}:
        return (
            select_features(
                feature_selector,
                x_source,
                y_source,
                seed,
                k_overrides=override_k,
                rfe_step=args.rfe_step,
                n_jobs=args.n_jobs,
                natural_elasticnet=args.feature_count_mode == "natural" and feature_selector == "elasticnet",
                cv_folds=args.auto_k_cv_folds,
                chi2_bins=args.chi2_bins,
            ),
            None,
        )

    candidate_rows: list[dict[str, Any]] = []
    best_result = None
    best_score = -np.inf
    cv = StratifiedKFold(n_splits=args.auto_k_cv_folds, shuffle=True, random_state=seed)
    estimator = LogisticRegression(max_iter=5000, solver="liblinear", random_state=seed)
    valid_ks = sorted({min(k, x_source.shape[1]) for k in args.auto_k_values if k > 0})
    for k in valid_ks:
        result = select_features(
            feature_selector,
            x_source,
            y_source,
            seed,
            k_overrides={feature_selector: k},
            rfe_step=args.rfe_step,
            n_jobs=args.n_jobs,
            chi2_bins=args.chi2_bins,
        )
        scores = cross_val_score(
            estimator,
            x_source.loc[:, result.selected_genes],
            y_source,
            cv=cv,
            scoring=args.auto_k_scoring,
            n_jobs=1,
        )
        score_mean = float(np.mean(scores))
        score_std = float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0
        candidate_rows.append(
            {
                "feature_selector": feature_selector,
                "k": int(k),
                "scoring": args.auto_k_scoring,
                "cv_mean": score_mean,
                "cv_std": score_std,
            }
        )
        if score_mean > best_score:
            best_score = score_mean
            best_result = result
    if best_result is None:
        raise ValueError(f"No valid auto-k candidate for {feature_selector}.")
    trace = pd.DataFrame(candidate_rows)
    best_k = int(trace.sort_values("cv_mean", ascending=False).iloc[0]["k"])
    best_result.metadata = {
        **best_result.metadata,
        "feature_count_mode": "auto",
        "auto_k_selected": best_k,
        "auto_k_values": valid_ks,
        "auto_k_scoring": args.auto_k_scoring,
        "auto_k_cv_folds": args.auto_k_cv_folds,
        "auto_k_best_score": best_score,
    }
    return best_result, trace


def copy_feature_artifacts(feature_dir: Path, run_dir: Path) -> None:
    for name in ("selected_genes.txt", "feature_ranking.csv", "feature_selection_manifest.json", "auto_k_trace.csv"):
        src = feature_dir / name
        if src.exists():
            shutil.copy2(src, run_dir / name)


def write_run_artifacts(
    run_dir: Path,
    metric_row: dict[str, Any],
    sample_ids: pd.Index,
    y_test: np.ndarray,
    scores: np.ndarray,
    y_pred: np.ndarray,
    model_metadata: dict[str, Any],
    augmentation_manifest: dict[str, Any],
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([metric_row]).to_csv(run_dir / "metrics.csv", index=False)
    predictions_frame(sample_ids, y_test, scores, y_pred).to_csv(run_dir / "predictions.csv", index=False)
    confusion_frame(y_test, y_pred).to_csv(run_dir / "confusion_matrix.csv")
    classification_report_frame(y_test, y_pred).to_csv(run_dir / "classification_report.csv")
    write_json(run_dir / "augmentation_manifest.json", augmentation_manifest)
    write_json(run_dir / "hyperparameters.json", model_metadata)
    write_json(run_dir / "run_manifest.json", {"metrics": metric_row, "model": model_metadata, "augmentation": augmentation_manifest})


def write_audit(result_root: Path, args: argparse.Namespace, x: pd.DataFrame, y: pd.Series, run_count: int) -> None:
    expected = len(args.batch_scenarios) * args.repeats * len(args.feature_selectors) * len(args.augmentations) * len(args.models)
    audit = pd.DataFrame(
        [
            {"item": "input_dataset", "status": "pass", "details": f"{len(y)} samples x {x.shape[1]} genes from {args.x_file}"},
            {
                "item": "protocol",
                "status": "pass",
                "details": f"{args.repeats} repeats across {', '.join(args.batch_scenarios)}; inner_val_ratio={args.inner_val_ratio}",
            },
            {"item": "split_overlap", "status": "pass", "details": "Train, validation, and test indices checked for every split."},
            {
                "item": "training_balance",
                "status": "pass",
                "details": f"training_balance={args.training_balance}; when enabled, deterministic undersampling is confined to train_inner and counts are recorded per split.",
            },
            {
                "item": "preprocessing_scope",
                "status": "intentional_leakage" if args.artifact_scope == "full_dataset" else "pass",
                "details": "Median imputer and MinMaxScaler fit on full dataset." if args.artifact_scope == "full_dataset" else "Median imputer and MinMaxScaler fit only on train_inner.",
            },
            {
                "item": "feature_selection_scope",
                "status": "intentional_leakage" if args.artifact_scope == "full_dataset" else "pass",
                "details": "Chi-square, ANOVA, RFE, ElasticNet fit on full dataset." if args.artifact_scope == "full_dataset" else "Chi-square, ANOVA, RFE, ElasticNet fit only on train_inner.",
            },
            {"item": "augmentation_scope", "status": "pass", "details": "GAN/CTGAN augmentation, when enabled, is trained and sampled only from train_inner after feature selection; manifests distinguish the external CTGAN from the local approximation."},
            {"item": "test_scope", "status": "pass", "details": "Outer test is used only for final evaluation in each scenario/repeat."},
            {"item": "completed_runs", "status": "pass", "details": f"{run_count} completed metrics rows; expected full rows={expected}"},
            {
                "item": "paper_adaptation",
                "status": "note",
                "details": "Original paper is AD vs CTL with ADNI integration; this implementation adapts methods to local AD vs MCI GSE63060+GSE63061.",
            },
        ]
    )
    audit.to_csv(result_root / "leakage_audit.csv", index=False)


def apply_smoke_overrides(args: argparse.Namespace) -> None:
    args.repeats = 1
    args.batch_scenarios = args.batch_scenarios[:1]
    args.feature_selectors = [fs for fs in args.feature_selectors if fs in {"anova", "elasticnet"}][:1] or ["anova"]
    args.models = [model for model in args.models if model in {"dnn", "cnn"}] or ["dnn"]
    requested_generators = [mode for mode in args.augmentations if mode in {"gan", "ctgan"}]
    args.augmentations = ["none", *requested_generators]
    args.chi2_k = min(args.chi2_k, 30)
    args.anova_k = min(args.anova_k, 30)
    args.rfe_k = min(args.rfe_k, 30)
    args.elasticnet_k = min(args.elasticnet_k, 30)
    args.lasso_k = min(args.lasso_k, 30)
    args.rf_importance_k = min(args.rf_importance_k, 30)
    args.gan_target_size = 340
    args.gan_epochs = min(args.gan_epochs, 2)
    args.deep_epochs = min(args.deep_epochs, 2)
    args.deep_patience = min(args.deep_patience, 1)


def main() -> None:
    args = parse_args()
    args.x_file = resolve(args.x_file)
    args.y_file = resolve(args.y_file)
    args.split_manifest_dir = resolve(args.split_manifest_dir) if args.split_manifest_dir else None
    args.result_root = resolve(args.result_root)
    args.gse63060_metadata = resolve(args.gse63060_metadata)
    args.gse63061_metadata = resolve(args.gse63061_metadata)
    if args.strict_v2:
        args.repeats = 10
        args.batch_scenarios = ["shared_test", "test_gse63060", "test_gse63061"]
        args.artifact_scope = "train_inner"
        args.training_balance = "undersample"
        args.feature_selectors = ["elasticnet"]
        args.feature_count_mode = "natural"
        args.auto_k_cv_folds = 5
        args.models = ["dnn"]
        args.augmentations = ["ctgan"]
        args.threshold_mode = "fixed_0_5"
        args.skip_existing = False
        args.overwrite_results = True
    if args.smoke and args.strict_v2:
        args.repeats = 1
        args.batch_scenarios = args.batch_scenarios[:1]
        args.feature_count_mode = "fixed"
        args.elasticnet_k = min(args.elasticnet_k, 30)
        args.auto_k_cv_folds = 2
        args.gan_target_size = 340
        args.deep_epochs = min(args.deep_epochs, 2)
        args.deep_patience = min(args.deep_patience, 1)
        args.gan_epochs = min(args.gan_epochs, 2)
    elif args.smoke:
        apply_smoke_overrides(args)

    if args.overwrite_results and args.result_root.exists():
        shutil.rmtree(args.result_root)
    args.result_root.mkdir(parents=True, exist_ok=True)
    write_json(args.result_root / "args.json", vars(args))

    x, y = load_ad_mci_dataset(args.x_file, args.y_file)
    batch = infer_batch_labels(x.index, args.gse63060_metadata, args.gse63061_metadata)
    pd.DataFrame({"sample_id": x.index.astype(str), "batch": batch.to_numpy(), "label": y.to_numpy(dtype=np.int64)}).to_csv(
        args.result_root / "batch_manifest.csv", index=False
    )
    if args.split_manifest_dir:
        if args.protocol != "batch_holdout":
            raise ValueError("--split-manifest-dir requires --protocol batch_holdout.")
        splits = load_task_splits(
            x.index,
            y,
            args.split_manifest_dir,
            repeats=args.repeats,
            expected_seeds=range(args.seed, args.seed + args.repeats),
        )
        split_counts(splits).to_csv(args.result_root / "shared_split_counts.csv", index=False)
    elif args.protocol == "stratified_5cv":
        splits = make_stratified_5cv_splits(y, args.repeats, args.inner_val_ratio, args.seed, outer_folds=5)
    else:
        splits = make_batch_holdout_splits(y, batch, args.batch_scenarios, args.repeats, args.inner_val_ratio, args.shared_test_size, args.seed)
    y_values = y.to_numpy(dtype=np.int64)
    override_k = k_overrides(args)
    run_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    x_full_scaled: pd.DataFrame | None = None
    full_preprocessing_manifest: dict[str, Any] | None = None
    full_feature_artifacts: dict[str, tuple[list[str], dict[str, Any], Path]] = {}
    if args.artifact_scope == "full_dataset":
        print("Selecting Nature 2026 features on full dataset (intentional leakage).", flush=True)
        x_full_scaled, full_preprocessing_manifest = preprocess_full(x)
        for feature_selector in args.feature_selectors:
            feature_dir = args.result_root / "full_dataset_artifacts" / "feature_selection" / feature_selector
            selected_path = feature_dir / "selected_genes.txt"
            if args.skip_existing and selected_path.exists():
                selected_genes = [line.strip() for line in selected_path.read_text(encoding="utf-8").splitlines() if line.strip()]
                fs_manifest = json.loads((feature_dir / "feature_selection_manifest.json").read_text(encoding="utf-8"))
            else:
                fs_result, auto_trace = select_features_with_optional_auto_k(
                    feature_selector,
                    x_full_scaled,
                    y_values,
                    args.seed,
                    args,
                    override_k,
                )
                feature_dir.mkdir(parents=True, exist_ok=True)
                fs_result.ranking.to_csv(feature_dir / "feature_ranking.csv", index=False)
                if auto_trace is not None:
                    auto_trace.to_csv(feature_dir / "auto_k_trace.csv", index=False)
                (feature_dir / "selected_genes.txt").write_text("\n".join(fs_result.selected_genes) + "\n", encoding="utf-8")
                fs_manifest = {**fs_result.metadata, "artifact_scope": "full_dataset", "selection_scope": "full_dataset"}
                write_json(feature_dir / "feature_selection_manifest.json", fs_manifest)
                write_json(feature_dir / "preprocessing_manifest.json", full_preprocessing_manifest)
                selected_genes = fs_result.selected_genes
            full_feature_artifacts[feature_selector] = (selected_genes, fs_manifest, feature_dir)

    for split in splits:
        assert_no_overlap(split)
        train_idx = split["train_inner_idx"]
        val_idx = split["val_inner_idx"]
        test_idx = split["outer_test_idx"]
        scenario = split["scenario"]
        repeat = split["repeat"]
        split_seed = int(split["seed"]) if split.get("split_source") else int(split["seed"] * 1000 + split["scenario_index"] * 100 + repeat)
        split_dir = args.result_root / "runs" / scenario / f"repeat_{repeat:02d}" / f"fold_{split['fold']:02d}"
        split_dir.mkdir(parents=True, exist_ok=True)

        x_train_split_raw = x.iloc[train_idx].copy()
        x_val_raw = x.iloc[val_idx].copy()
        x_test_raw = x.iloc[test_idx].copy()
        y_train_split = y_values[train_idx]
        y_val = y_values[val_idx]
        y_test = y_values[test_idx]
        balance_result = balance_training_data(
            x_train_split_raw,
            y_train_split,
            args.training_balance,
            split_seed,
        )
        x_train_raw = balance_result.x_train
        y_train = balance_result.y_train
        write_json(split_dir / "training_balance_manifest.json", balance_result.manifest)
        if args.artifact_scope == "full_dataset":
            if x_full_scaled is None or full_preprocessing_manifest is None:
                raise RuntimeError("Full-dataset preprocessing was not initialized.")
            x_train_scaled = x_full_scaled.loc[x_train_raw.index].copy()
            x_val_scaled = x_full_scaled.iloc[val_idx].copy()
            x_test_scaled = x_full_scaled.iloc[test_idx].copy()
            preprocessing_manifest = full_preprocessing_manifest
        else:
            x_train_scaled, x_val_scaled, x_test_scaled, preprocessing_manifest = preprocess_split(x_train_raw, x_val_raw, x_test_raw)
        split_meta_base = {
            "protocol": args.protocol,
            "scenario": scenario,
            "repeat": repeat,
            "fold": split["fold"],
            "seed": split_seed,
            "n_train_inner": int(len(train_idx)),
            "training_balance": args.training_balance,
            "n_train_after_balance": int(len(y_train)),
            "train_class_counts_before_balance": json.dumps(balance_result.manifest["class_counts_before"], sort_keys=True),
            "train_class_counts_after_balance": json.dumps(balance_result.manifest["class_counts_after"], sort_keys=True),
            "n_val_inner": int(len(val_idx)),
            "n_outer_test": int(len(test_idx)),
            "train_batches": "|".join(sorted(batch.iloc[train_idx].unique().tolist())),
            "val_batches": "|".join(sorted(batch.iloc[val_idx].unique().tolist())),
            "test_batches": "|".join(sorted(batch.iloc[test_idx].unique().tolist())),
        }
        write_json(
            split_dir / "split_manifest.json",
            {
                **split_meta_base,
                **preprocessing_manifest,
                "train_inner_ids": x.index[train_idx].astype(str).tolist(),
                "model_train_ids_after_balance": x_train_raw.index.astype(str).tolist(),
                "val_inner_ids": x.index[val_idx].astype(str).tolist(),
                "outer_test_ids": x.index[test_idx].astype(str).tolist(),
            },
        )

        for feature_selector in args.feature_selectors:
            feature_dir = split_dir / "feature_selection" / args.training_balance / feature_selector
            selected_path = feature_dir / "selected_genes.txt"
            if args.artifact_scope == "full_dataset":
                selected_genes, fs_manifest, source_feature_dir = full_feature_artifacts[feature_selector]
                feature_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_feature_dir / "selected_genes.txt", feature_dir / "selected_genes.txt")
                shutil.copy2(source_feature_dir / "feature_ranking.csv", feature_dir / "feature_ranking.csv")
                shutil.copy2(source_feature_dir / "feature_selection_manifest.json", feature_dir / "feature_selection_manifest.json")
                auto_trace = source_feature_dir / "auto_k_trace.csv"
                if auto_trace.exists():
                    shutil.copy2(auto_trace, feature_dir / "auto_k_trace.csv")
            elif args.skip_existing and selected_path.exists():
                selected_genes = [line.strip() for line in selected_path.read_text(encoding="utf-8").splitlines() if line.strip()]
                fs_manifest = json.loads((feature_dir / "feature_selection_manifest.json").read_text(encoding="utf-8"))
            else:
                print(f"Selecting scenario={scenario} repeat={repeat} feature={feature_selector}", flush=True)
                fs_result, auto_trace = select_features_with_optional_auto_k(
                    feature_selector,
                    x_train_scaled,
                    y_train,
                    split_seed,
                    args,
                    override_k,
                )
                feature_dir.mkdir(parents=True, exist_ok=True)
                fs_result.ranking.to_csv(feature_dir / "feature_ranking.csv", index=False)
                if auto_trace is not None:
                    auto_trace.to_csv(feature_dir / "auto_k_trace.csv", index=False)
                (feature_dir / "selected_genes.txt").write_text("\n".join(fs_result.selected_genes) + "\n", encoding="utf-8")
                write_json(feature_dir / "feature_selection_manifest.json", fs_result.metadata)
                selected_genes = fs_result.selected_genes
                fs_manifest = fs_result.metadata

            x_train_sel = x_train_scaled.loc[:, selected_genes].copy()
            x_val_sel = x_val_scaled.loc[:, selected_genes].copy()
            x_test_sel = x_test_scaled.loc[:, selected_genes].copy()
            split_meta = {
                **split_meta_base,
                "feature_selector": feature_selector,
                "n_selected_genes": int(len(selected_genes)),
            }

            for augmentation in args.augmentations:
                if args.skip_existing and all(
                    (split_dir / args.training_balance / feature_selector / augmentation / model_name / "metrics.csv").exists()
                    for model_name in args.models
                ):
                    continue
                aug_dir = split_dir / "augmentation" / args.training_balance / feature_selector / augmentation
                aug_result = augment_training_data(
                    x_train_sel,
                    y_train,
                    augmentation,
                    split_seed,
                    target_size=args.gan_target_size,
                    latent_dim=args.gan_latent_dim,
                    epochs=args.gan_epochs,
                    batch_size=args.gan_batch_size,
                    learning_rate=args.gan_learning_rate,
                    ctgan_pac=args.ctgan_pac,
                    ctgan_cuda=args.ctgan_cuda,
                )
                aug_dir.mkdir(parents=True, exist_ok=True)
                write_json(aug_dir / "augmentation_manifest.json", aug_result.manifest)

                for model_name in args.models:
                    run_dir = split_dir / args.training_balance / feature_selector / augmentation / model_name
                    if args.skip_existing and (run_dir / "metrics.csv").exists():
                        continue
                    print(
                        f"Running scenario={scenario} repeat={repeat} feature={feature_selector} augmentation={augmentation} model={model_name} genes={len(selected_genes)}",
                        flush=True,
                    )
                    try:
                        if model_name in {"dnn", "cnn"}:
                            scores, y_pred, metrics, model_meta = fit_predict_deep(
                                model_name,
                                aug_result.x_train,
                                aug_result.y_train,
                                x_val_sel,
                                y_val,
                                x_test_sel,
                                y_test,
                                split_seed,
                                args.deep_epochs,
                                args.deep_patience,
                                args.deep_batch_size,
                                args.deep_learning_rate,
                                args.dropout,
                                run_dir,
                                args.threshold_mode,
                            )
                        else:
                            scores, y_pred, metrics, model_meta = fit_predict_sklearn(
                                model_name,
                                aug_result.x_train,
                                aug_result.y_train,
                                x_test_sel,
                                y_test,
                                split_seed,
                                args.n_jobs,
                                x_val_sel,
                                y_val,
                                args.threshold_mode,
                            )
                        row = {**split_meta, "augmentation": augmentation, "model": model_name, **metrics}
                        run_dir.mkdir(parents=True, exist_ok=True)
                        copy_feature_artifacts(feature_dir, run_dir)
                        shutil.copy2(aug_dir / "augmentation_manifest.json", run_dir / "augmentation_manifest.json")
                        shutil.copy2(split_dir / "training_balance_manifest.json", run_dir / "training_balance_manifest.json")
                        write_run_artifacts(run_dir, row, x.index[test_idx], y_test, scores, y_pred, model_meta, aug_result.manifest)
                        run_rows.append(row)
                    except Exception as exc:
                        run_dir.mkdir(parents=True, exist_ok=True)
                        failure = {**split_meta, "augmentation": augmentation, "model": model_name, "status": "failed", "error": repr(exc)}
                        write_json(run_dir / "failure.json", failure)
                        failure_rows.append(failure)
                        print(f"FAILED scenario={scenario} repeat={repeat} feature={feature_selector} augmentation={augmentation} model={model_name}: {exc}", flush=True)

    all_metrics, summary = summarize(args.result_root)
    all_metrics.to_csv(args.result_root / "all_metrics.csv", index=False)
    summary.to_csv(args.result_root / "summary_by_method.csv", index=False)
    if not summary.empty:
        summary.sort_values("roc_auc_mean", ascending=False).to_csv(args.result_root / "ranking_by_roc_auc.csv", index=False)
        summary.sort_values("macro_f1_mean", ascending=False).to_csv(args.result_root / "ranking_by_macro_f1.csv", index=False)
        summary.sort_values("accuracy_mean", ascending=False).to_csv(args.result_root / "ranking_by_accuracy.csv", index=False)
    write_audit(args.result_root, args, x, y, len(all_metrics))
    if failure_rows:
        pd.DataFrame(failure_rows).to_csv(args.result_root / "failures.csv", index=False)
        raise RuntimeError(
            f"{len(failure_rows)} Hariharan configurations failed; see {args.result_root / 'failures.csv'}. "
            "The benchmark is incomplete and was not reported as successful."
        )
    print(f"Finished. Results written to {args.result_root}", flush=True)


if __name__ == "__main__":
    main()
