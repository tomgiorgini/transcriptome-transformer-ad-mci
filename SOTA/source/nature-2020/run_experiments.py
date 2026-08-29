#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
COMMON_DIR = Path(__file__).resolve().parents[1]
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

from data import BatchSplit, assert_no_split_overlap, infer_batch_labels, load_binary_dataset, make_batch_holdout_splits, make_stratified_5cv_splits
from deep_models import fit_predict_dnn
from feature_selection import (
    FeatureSet,
    empty_feature_set,
    fit_imputer_scaler,
    fit_vae_feature_set,
    load_cfg_genes,
    load_hprd_hub_genes,
    load_tf_genes,
    run_limma_deg,
    subset_feature_set,
    write_json,
)
from metrics import classification_report_frame, confusion_frame, evaluate_binary, predictions_frame
from models import available_classical_models, build_paper_estimator, positive_scores
from summarize_results import summarize
from strict_v2_utils import calibrate_threshold, cv_score_estimator
from shared_test_splits import load_task_splits, split_counts


DEFAULT_X = ROOT / "task_dataset" / "processed" / "ad_mci_binary" / "X_ad_mci.csv"
DEFAULT_Y = ROOT / "task_dataset" / "processed" / "ad_mci_binary" / "y_ad_mci.csv"
DEFAULT_RESULT_ROOT = ROOT / "results" / "SOTA" / "nature-2020"
DEFAULT_SUPPLEMENT_DIR = ROOT / "SOTA" / "Public Code" / "nature-2020"
DEFAULT_GSE63060_METADATA = ROOT / "pretraining_dataset" / "geo_downloads" / "GSE63060" / "eset_1_GPL6947" / "GSE63060_sample_metadata.tsv.gz"
DEFAULT_GSE63061_METADATA = ROOT / "pretraining_dataset" / "geo_downloads" / "GSE63061" / "eset_1_GPL10558" / "GSE63061_sample_metadata.tsv.gz"
SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lee & Lee 2020 full-grid adaptation for a binary TxT task.")
    parser.add_argument("--x-file", type=Path, default=DEFAULT_X)
    parser.add_argument("--y-file", type=Path, default=DEFAULT_Y)
    parser.add_argument(
        "--split-manifest-dir",
        type=Path,
        default=None,
        help="Directory of TxT seed_<seed>.csv manifests. When set, use their exact shared-test sample membership.",
    )
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--supplement-dir", type=Path, default=DEFAULT_SUPPLEMENT_DIR)
    parser.add_argument("--protocol", choices=["batch_holdout", "stratified_5cv"], default="batch_holdout")
    parser.add_argument("--gse63060-metadata", type=Path, default=DEFAULT_GSE63060_METADATA)
    parser.add_argument("--gse63061-metadata", type=Path, default=DEFAULT_GSE63061_METADATA)
    parser.add_argument("--batch-scenarios", nargs="+", default=["shared_test"], choices=["shared_test", "test_gse63060", "test_gse63061"])
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--inner-val-ratio", type=float, default=0.125, help="Fraction of the post-test pool used for validation; 0.125 gives total 70/10/20 when test-size is 0.20.")
    parser.add_argument("--shared-test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fdr-threshold", type=float, default=0.01)
    parser.add_argument("--p-value-threshold", type=float, default=None)
    parser.add_argument("--deg-fallback-top-k", type=int, default=0)
    parser.add_argument("--scaler", choices=["minmax", "none"], default="minmax")
    parser.add_argument(
        "--artifact-scope",
        choices=["train_inner", "full_dataset"],
        default="train_inner",
        help="Fit preprocessing and DEG selection on train_inner or on the full dataset with intentional leakage. VAE weights always fit train_inner.",
    )
    parser.add_argument("--feature-sets", nargs="+", default=["deg", "vae", "tf_genes", "cfg_genes", "hub_genes"], choices=["deg", "vae", "tf_genes", "cfg_genes", "hub_genes"])
    parser.add_argument("--models", nargs="+", default=["lr", "l1_lr", "svm", "rf", "dnn"], choices=["lr", "l1_lr", "svm", "rf", "dnn"])
    parser.add_argument("--class-weighting", choices=["off", "balanced"], default="off", help="Class weighting is off in the paper profile; balanced is an explicit benchmark ablation.")
    parser.add_argument("--allow-cfg-supplement-proxy", action="store_true", help="Opt in to the incomplete MOESM3-5 CFG proxy, which is outcome-derived on the original datasets and is not leakage-independent.")
    parser.add_argument("--hprd-network-file", type=Path, default=None, help="Optional local HPRD edge file. No network is downloaded automatically.")
    parser.add_argument("--hprd-gene-columns", nargs=2, type=int, default=None, metavar=("LEFT", "RIGHT"), help="Zero-based gene-symbol columns; defaults to 0,3 for HPRD flat files and 0,1 for two-column edge lists.")
    parser.add_argument("--hprd-degree-threshold", type=int, default=10, help="Select HPRD nodes with degree strictly greater than this value (paper: >10).")
    parser.add_argument("--vae-epochs", type=int, default=3000, help="Full-batch VAE pretraining iterations in the paper profile.")
    parser.add_argument("--vae-fine-tune-epochs", type=int, default=None, help="Supervised fine-tuning iterations; defaults to --vae-epochs.")
    parser.add_argument("--vae-batch-size", type=int, default=None, help="Defaults to the complete train_inner set so one epoch corresponds to one optimizer update.")
    parser.add_argument("--vae-latent-policy", choices=["paper_reconstruction", "fixed"], default="paper_reconstruction")
    parser.add_argument("--vae-latent-dim", type=int, default=None, help="Required with --vae-latent-policy fixed.")
    parser.add_argument("--dnn-max-epochs", type=int, default=None)
    parser.add_argument("--dnn-batch-size", type=int, default=32)
    parser.add_argument("--dnn-patience", type=int, default=30)
    parser.add_argument("--threshold-mode", choices=["fixed_0_5", "validation_macro_f1", "validation_accuracy", "validation_accuracy_macro_f1"], default="fixed_0_5")
    parser.add_argument("--class-names", nargs=2, default=["class_0", "class_1"], metavar=("NEGATIVE", "POSITIVE"), help="Human-readable names for labels 0 and 1 in output artifacts.")
    parser.add_argument("--deg-threshold-cv", action="store_true", help="Select DEG threshold with train-only CV instead of using --fdr-threshold directly.")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--paper-profile", action="store_true", help="Force paper-like method defaults (full grid, FDR 0.01, no DEG fallback, no class weighting).")
    parser.add_argument("--strict-v2", action="store_true", help="Compatibility profile: train-only shared test, 10 repeats, full 5x5 grid and paper-like method defaults.")
    parser.add_argument("--overwrite-results", action="store_true", help="Delete the result root before running.")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else (ROOT / path).resolve()


def apply_paper_method_profile(args: argparse.Namespace) -> None:
    args.artifact_scope = "train_inner"
    args.feature_sets = ["deg", "vae", "tf_genes", "cfg_genes", "hub_genes"]
    args.models = ["lr", "l1_lr", "svm", "rf", "dnn"]
    args.fdr_threshold = 0.01
    args.p_value_threshold = None
    args.deg_fallback_top_k = 0
    args.deg_threshold_cv = False
    args.class_weighting = "off"
    args.threshold_mode = "fixed_0_5"


def keras_class_weights(y_train: np.ndarray, weighting: str) -> dict[int, float] | None:
    if weighting == "off":
        return None
    if weighting != "balanced":
        raise ValueError(f"Unsupported class weighting: {weighting}")
    counts = np.bincount(np.asarray(y_train, dtype=np.int64), minlength=2)
    if np.any(counts == 0):
        raise ValueError("Balanced class weighting requires both classes in train_inner.")
    n_samples = int(counts.sum())
    return {label: n_samples / (2.0 * int(count)) for label, count in enumerate(counts)}


def safe_name(value: str) -> str:
    return value.replace("/", "_").replace("\\", "_").replace(" ", "_")


def remove_result_root(path: Path, retries: int = 8, delay_seconds: float = 1.0) -> None:
    last_error: Exception | None = None
    for _ in range(retries):
        try:
            shutil.rmtree(path)
            return
        except PermissionError as exc:
            last_error = exc
            time.sleep(delay_seconds)
    raise RuntimeError(
        f"Could not delete result root because a file is locked: {path}. "
        "Close any process using files in that folder, or use a different --result-root."
    ) from last_error


def fit_full_imputer_scaler(x_df: pd.DataFrame, scaler_name: str) -> tuple[np.ndarray, dict[str, object]]:
    imputer = SimpleImputer(strategy="median")
    x_all = imputer.fit_transform(x_df)
    meta: dict[str, object] = {
        "imputer": "SimpleImputer(strategy=median)",
        "scaler": scaler_name,
        "fit_scope": "full_dataset",
        "artifact_scope": "full_dataset",
    }
    if scaler_name == "minmax":
        scaler = MinMaxScaler()
        x_all = scaler.fit_transform(x_all)
    elif scaler_name != "none":
        raise ValueError(f"Unsupported scaler: {scaler_name}")
    return np.nan_to_num(x_all.astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0), meta


def write_run_artifacts(
    run_dir: Path,
    metric_row: dict[str, Any],
    sample_ids: pd.Index,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scores: np.ndarray,
    feature_set: FeatureSet,
    hyperparameters: dict[str, object],
    manifest: dict[str, object],
    class_names: tuple[str, str],
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([metric_row]).to_csv(run_dir / "metrics.csv", index=False)
    predictions_frame(sample_ids, y_true, scores, y_pred, class_names).to_csv(run_dir / "predictions.csv", index=False)
    confusion_frame(y_true, y_pred, class_names).to_csv(run_dir / "confusion_matrix.csv")
    classification_report_frame(y_true, y_pred, class_names).to_csv(run_dir / "classification_report.csv")
    (run_dir / "selected_genes.txt").write_text("\n".join(feature_set.feature_names) + ("\n" if feature_set.feature_names else ""), encoding="utf-8")
    write_json(run_dir / "hyperparameters.json", hyperparameters)
    write_json(run_dir / "run_manifest.json", manifest)


def append_status(status_rows: list[dict[str, object]], split, feature_set: str, status: str, details: dict[str, object]) -> None:
    status_rows.append(
        {
            "scenario": split.scenario,
            "repeat": split.repeat,
            "fold": split.fold,
            "feature_set": feature_set,
            "status": status,
            **details,
        }
    )


def build_feature_sets(
    args: argparse.Namespace,
    x: pd.DataFrame,
    y: pd.Series,
    split,
    batch: pd.Series,
    status_rows: list[dict[str, object]],
) -> dict[str, FeatureSet]:
    train_idx = split.train_inner_idx
    val_idx = split.val_inner_idx
    test_idx = split.outer_test_idx
    y_values = y.to_numpy(dtype=np.int64)
    all_gene_names = x.columns.astype(str).tolist()
    if args.artifact_scope == "full_dataset":
        x_all_scaled, preprocessing_meta = fit_full_imputer_scaler(x, args.scaler)
        x_train_all = x_all_scaled[train_idx]
        x_val_all = x_all_scaled[val_idx]
        x_test_all = x_all_scaled[test_idx]
        deg_source_x = x
        deg_source_y = y_values
        deg_scope = "full_dataset"
    else:
        x_train_all, x_val_all, x_test_all, preprocessing_meta = fit_imputer_scaler(x, train_idx, val_idx, test_idx, args.scaler)
        deg_source_x = x.iloc[train_idx]
        deg_source_y = y_values[train_idx]
        deg_scope = "train_inner"

    split_root = args.result_root / "runs" / split.scenario / f"repeat_{split.repeat:02d}" / f"fold_{split.fold:02d}"
    deg_dir = split_root / "_feature_selection"
    deg_table, deg_genes = run_limma_deg(
        x_train_df=deg_source_x,
        y_train=deg_source_y,
        script_path=SCRIPT_DIR / "deg_limma.R",
        fdr_threshold=args.fdr_threshold,
        work_dir=deg_dir,
        p_value_threshold=args.p_value_threshold,
    )
    deg_selection_rule = f"p_value < {args.p_value_threshold}" if args.p_value_threshold is not None else f"adjusted_p < {args.fdr_threshold}"
    deg_status = "completed" if deg_genes else "skipped_empty_feature_set"
    if args.deg_threshold_cv:
        candidate_rows: list[dict[str, object]] = []
        best_score = -np.inf
        best_genes: list[str] = []
        adjusted = pd.to_numeric(deg_table["adjusted_p"], errors="coerce")
        p_values = pd.to_numeric(deg_table["p_value"], errors="coerce")
        for threshold in (0.01, 0.05, 0.1, 0.2):
            genes = [g for g in deg_table.loc[adjusted < threshold, "gene"].astype(str).tolist() if g in x.columns]
            if not genes:
                candidate_rows.append({"threshold_type": "adjusted_p", "threshold": threshold, "n_genes": 0, "strict_v2_cv_mean": np.nan})
                continue
            estimator = make_pipeline(
                SimpleImputer(strategy="median"),
                MinMaxScaler(),
                LogisticRegression(max_iter=5000, solver="liblinear", class_weight=None if args.class_weighting == "off" else "balanced", random_state=split.seed),
            )
            score, fold_scores = cv_score_estimator(estimator, deg_source_x.loc[:, genes], deg_source_y, folds=5, seed=split.seed + int(threshold * 1000))
            candidate_rows.append({"threshold_type": "adjusted_p", "threshold": threshold, "n_genes": len(genes), "strict_v2_cv_mean": score, "strict_v2_cv_folds": fold_scores})
            if score > best_score:
                best_score = score
                best_genes = genes
                deg_selection_rule = f"strict_v2_cv_selected_adjusted_p_lt_{threshold}"
        for threshold in (0.001, 0.005, 0.01, 0.05):
            genes = [g for g in deg_table.loc[p_values < threshold, "gene"].astype(str).tolist() if g in x.columns]
            if not genes:
                candidate_rows.append({"threshold_type": "p_value", "threshold": threshold, "n_genes": 0, "strict_v2_cv_mean": np.nan})
                continue
            estimator = make_pipeline(
                SimpleImputer(strategy="median"),
                MinMaxScaler(),
                LogisticRegression(max_iter=5000, solver="liblinear", class_weight=None if args.class_weighting == "off" else "balanced", random_state=split.seed),
            )
            score, fold_scores = cv_score_estimator(estimator, deg_source_x.loc[:, genes], deg_source_y, folds=5, seed=split.seed + 2000 + int(threshold * 100000))
            candidate_rows.append({"threshold_type": "p_value", "threshold": threshold, "n_genes": len(genes), "strict_v2_cv_mean": score, "strict_v2_cv_folds": fold_scores})
            if score > best_score:
                best_score = score
                best_genes = genes
                deg_selection_rule = f"strict_v2_cv_selected_p_value_lt_{threshold}"
        pd.DataFrame(candidate_rows).to_csv(deg_dir / "deg_threshold_cv_trace.csv", index=False)
        deg_genes = best_genes
        deg_status = "completed" if deg_genes else "skipped_empty_feature_set"
    if args.deg_fallback_top_k > 0 and len(deg_genes) < args.deg_fallback_top_k:
        ranked = deg_table.sort_values("adjusted_p", ascending=True)
        deg_genes = [g for g in ranked["gene"].astype(str).tolist() if g in x.columns][: args.deg_fallback_top_k]
        deg_selection_rule = f"fallback_top_{args.deg_fallback_top_k}_by_adjusted_p"
        deg_status = "completed_fallback_top_k" if deg_genes else "skipped_empty_feature_set"
    write_json(
        deg_dir / "feature_selection_manifest.json",
        {
            "method": "limma",
            "scope": deg_scope,
            "artifact_scope": args.artifact_scope,
            "fdr_threshold": args.fdr_threshold,
            "p_value_threshold": args.p_value_threshold,
            "fallback_top_k": args.deg_fallback_top_k,
            "threshold_cv": bool(args.deg_threshold_cv),
            "selection_rule": deg_selection_rule,
            "n_train_inner": len(train_idx),
            "n_selection_samples": len(deg_source_y),
            "n_deg": len(deg_genes),
            "train_batches": sorted(batch.iloc[train_idx].unique().tolist()),
        },
    )
    append_status(status_rows, split, "deg", deg_status, {"n_features": len(deg_genes), "scope": deg_scope, "selection_rule": deg_selection_rule})

    feature_sets: dict[str, FeatureSet] = {}
    deg_fs = subset_feature_set("deg", x_train_all, x_val_all, x_test_all, all_gene_names, deg_genes, {"selection": "limma", "fdr_threshold": args.fdr_threshold, "fallback_top_k": args.deg_fallback_top_k, "selection_rule": deg_selection_rule, **preprocessing_meta})
    feature_sets["deg"] = deg_fs

    if "tf_genes" in args.feature_sets:
        tf_genes = load_tf_genes(args.supplement_dir)
        selected = [g for g in deg_genes if g in tf_genes]
        feature_sets["tf_genes"] = subset_feature_set("tf_genes", x_train_all, x_val_all, x_test_all, all_gene_names, selected, {"selection": "DEG intersect TRANSFAC TF list", "external_gene_list": "MOESM2", **preprocessing_meta})
        append_status(status_rows, split, "tf_genes", feature_sets["tf_genes"].status, {"n_features": len(selected), "external_genes": len(tf_genes)})

    if "cfg_genes" in args.feature_sets:
        if not args.allow_cfg_supplement_proxy:
            cfg_metadata = {
                "reason": "MOESM3-5 contain CFG scores only for outcome-derived DEG lists from the complete original datasets, not an all-gene external CFG catalogue.",
                "required_flag": "--allow-cfg-supplement-proxy",
                "leakage_status": "not_independent_for_original_GSE_holdouts",
            }
            feature_sets["cfg_genes"] = empty_feature_set(
                "cfg_genes",
                len(train_idx),
                len(val_idx),
                len(test_idx),
                "skipped_cfg_proxy_requires_opt_in",
                cfg_metadata,
            )
            append_status(status_rows, split, "cfg_genes", "skipped_cfg_proxy_requires_opt_in", {"n_features": 0, **cfg_metadata})
        else:
            cfg_genes = load_cfg_genes(args.supplement_dir, min_score=3)
            selected = [g for g in deg_genes if g in cfg_genes]
            cfg_metadata = {
                "selection": "DEG intersect Final_CFG>=3 supplement proxy",
                "external_gene_list": "MOESM3-5",
                "proxy_opt_in": True,
                "leakage_status": "not_independent_for_original_GSE_holdouts",
                **preprocessing_meta,
            }
            feature_sets["cfg_genes"] = subset_feature_set("cfg_genes", x_train_all, x_val_all, x_test_all, all_gene_names, selected, cfg_metadata)
            append_status(status_rows, split, "cfg_genes", feature_sets["cfg_genes"].status, {"n_features": len(selected), "external_genes": len(cfg_genes), "proxy_opt_in": True})

    if "vae" in args.feature_sets:
        vae_fs = fit_vae_feature_set(
            deg_fs,
            y_train=y_values[train_idx],
            y_val=y_values[val_idx],
            seed=split.seed,
            epochs=args.vae_epochs,
            batch_size=args.vae_batch_size,
            latent_policy=args.vae_latent_policy,
            fixed_latent_dim=args.vae_latent_dim,
            fine_tune_epochs=args.vae_fine_tune_epochs,
        )
        vae_fs.metadata["upstream_artifact_scope"] = args.artifact_scope
        feature_sets["vae"] = vae_fs
        append_status(status_rows, split, "vae", vae_fs.status, {"n_features": len(vae_fs.feature_names), **vae_fs.metadata})

    if "hub_genes" in args.feature_sets:
        if args.hprd_network_file is None:
            metadata = {
                "reason": "The official supplements do not contain the HPRD network. Supply a local file explicitly with --hprd-network-file.",
                "download_policy": "no_automatic_download",
            }
            feature_sets["hub_genes"] = empty_feature_set(
                "hub_genes",
                len(train_idx),
                len(val_idx),
                len(test_idx),
                "skipped_missing_hprd_network",
                metadata,
            )
            append_status(status_rows, split, "hub_genes", "skipped_missing_hprd_network", {"n_features": 0, **metadata})
        else:
            hprd_genes, hprd_provenance = load_hprd_hub_genes(
                args.hprd_network_file,
                degree_threshold=args.hprd_degree_threshold,
                gene_columns=tuple(args.hprd_gene_columns) if args.hprd_gene_columns else None,
            )
            selected = [gene for gene in deg_genes if gene in hprd_genes]
            metadata = {
                "selection": f"DEG intersect HPRD degree > {args.hprd_degree_threshold}",
                "network_provenance": hprd_provenance,
                "paper_faithfulness": "conditional reconstruction; exact processed Lee & Lee graph was not published",
                **preprocessing_meta,
            }
            feature_sets["hub_genes"] = subset_feature_set(
                "hub_genes",
                x_train_all,
                x_val_all,
                x_test_all,
                all_gene_names,
                selected,
                metadata,
            )
            append_status(
                status_rows,
                split,
                "hub_genes",
                feature_sets["hub_genes"].status,
                {
                    "n_features": len(selected),
                    "external_genes": len(hprd_genes),
                    "network_sha256": hprd_provenance["network_sha256"],
                    "n_unique_edges": hprd_provenance["n_unique_edges"],
                },
            )

    return {name: fs for name, fs in feature_sets.items() if name in args.feature_sets}


def run_model(args: argparse.Namespace, split, feature_set: FeatureSet, y_train: np.ndarray, y_val: np.ndarray, y_test: np.ndarray, sample_ids: pd.Index, model_name: str, run_dir: Path, batch: pd.Series) -> dict[str, object] | None:
    if feature_set.status != "completed" or feature_set.x_train.shape[1] == 0:
        return None
    print(f"Running scenario={split.scenario} repeat={split.repeat} feature={feature_set.name} model={model_name}", flush=True)
    sklearn_class_weight = None if args.class_weighting == "off" else "balanced"
    if model_name == "dnn":
        scores, y_pred, params = fit_predict_dnn(
            feature_set.x_train,
            y_train,
            feature_set.x_val,
            y_val,
            feature_set.x_test,
            seed=split.seed,
            max_epochs=args.dnn_max_epochs,
            batch_size=args.dnn_batch_size,
            patience=args.dnn_patience,
            class_weight=keras_class_weights(y_train, args.class_weighting),
        )
        hyperparameters = {"model": "dnn", **params}
        threshold_meta = {"threshold": 0.5, "threshold_mode": "fixed_0_5", "reason": "DNN helper returns fixed-threshold predictions."}
    else:
        estimator, params = build_paper_estimator(
            model_name,
            feature_set.x_train.shape[1],
            split.seed,
            class_weight=sklearn_class_weight,
        )
        estimator.fit(feature_set.x_train, y_train)
        val_scores = positive_scores(estimator, feature_set.x_val)
        scores = positive_scores(estimator, feature_set.x_test)
        if args.threshold_mode == "fixed_0_5":
            threshold = 0.5
            threshold_meta = {"threshold": threshold, "threshold_mode": "fixed_0_5"}
        else:
            metric = args.threshold_mode.replace("validation_", "")
            threshold, threshold_meta = calibrate_threshold(y_val, val_scores, metric=metric)
            threshold_meta["threshold_mode"] = args.threshold_mode
        y_pred = (scores >= threshold).astype(int)
        hyperparameters = {"model": model_name, **params, "threshold": threshold_meta["threshold"], "threshold_calibration": threshold_meta}

    metrics = evaluate_binary(y_test, scores, y_pred)
    row = {
        "protocol": args.protocol,
        "scenario": split.scenario,
        "repeat": split.repeat,
        "fold": split.fold,
        "seed": split.seed,
        "feature_set": feature_set.name,
        "model": model_name,
        "n_train_inner": len(y_train),
        "n_val_inner": len(y_val),
        "n_outer_test": len(y_test),
        "n_features": feature_set.x_train.shape[1],
        "negative_class": args.class_names[0],
        "positive_class": args.class_names[1],
        "train_batches": "|".join(sorted(batch.iloc[split.train_inner_idx].unique().tolist())),
        "val_batches": "|".join(sorted(batch.iloc[split.val_inner_idx].unique().tolist())),
        "test_batches": "|".join(sorted(batch.iloc[split.outer_test_idx].unique().tolist())),
        **metrics,
    }
    manifest = {
        "input_files": {"x_file": str(args.x_file), "y_file": str(args.y_file)},
        "paper": "Lee & Lee 2020 Scientific Reports",
        "adaptation": "binary pairwise TxT benchmark; Lee & Lee trained classifiers only for AD vs cognitively normal",
        "class_encoding": {"0": args.class_names[0], "1": args.class_names[1]},
        "leakage_policy": (
            "Intentional leakage experiment: preprocessing and DEG-derived intersections use the full dataset; VAE weights and model weights still use train_inner/val_inner only."
            if args.artifact_scope == "full_dataset"
            else "preprocessing, DEG, VAE, and model checkpoints use train_inner/val_inner only; outer_test is transform/evaluation only"
        ),
        "feature_set": feature_set.name,
        "feature_status": feature_set.status,
        "feature_metadata": feature_set.metadata,
        "metrics": row,
        "hyperparameters": hyperparameters,
        "threshold_calibration": threshold_meta,
    }
    write_run_artifacts(
        run_dir,
        row,
        sample_ids,
        y_test,
        y_pred,
        scores,
        feature_set,
        hyperparameters,
        manifest,
        tuple(args.class_names),
    )
    return row


def main() -> None:
    args = parse_args()
    args.x_file = resolve(args.x_file)
    args.y_file = resolve(args.y_file)
    args.split_manifest_dir = resolve(args.split_manifest_dir) if args.split_manifest_dir else None
    args.result_root = resolve(args.result_root)
    args.supplement_dir = resolve(args.supplement_dir)
    args.gse63060_metadata = resolve(args.gse63060_metadata)
    args.gse63061_metadata = resolve(args.gse63061_metadata)
    args.hprd_network_file = resolve(args.hprd_network_file) if args.hprd_network_file else None
    if args.paper_profile:
        apply_paper_method_profile(args)
    if args.strict_v2:
        apply_paper_method_profile(args)
        args.repeats = 10
        args.batch_scenarios = ["shared_test"]
        args.inner_val_ratio = 0.125
        args.shared_test_size = 0.2
        args.artifact_scope = "train_inner"
        args.scaler = "minmax"
        args.skip_existing = False
        args.overwrite_results = True
    if not all(str(name).strip() for name in args.class_names) or args.class_names[0] == args.class_names[1]:
        raise ValueError("--class-names must contain two distinct non-empty names.")
    if args.vae_latent_policy == "fixed" and args.vae_latent_dim is None:
        raise ValueError("--vae-latent-dim is required with --vae-latent-policy fixed.")
    if args.smoke:
        args.repeats = 1
        args.batch_scenarios = args.batch_scenarios[:1]
        if not args.strict_v2:
            args.models = [m for m in args.models if m in {"lr", "svm", "dnn"}]
            args.feature_sets = [f for f in args.feature_sets if f in {"deg", "vae", "hub_genes"}]
        args.dnn_max_epochs = 2
        args.vae_epochs = 2
        args.dnn_patience = 1
    if args.overwrite_results and args.result_root.exists():
        remove_result_root(args.result_root)
    args.result_root.mkdir(parents=True, exist_ok=True)
    write_json(args.result_root / "args.json", vars(args))

    x, y = load_binary_dataset(args.x_file, args.y_file)
    batch = infer_batch_labels(x.index, args.gse63060_metadata, args.gse63061_metadata)
    label_values = y.to_numpy(dtype=np.int64)
    pd.DataFrame(
        {
            "sample_id": x.index.astype(str),
            "batch": batch.to_numpy(),
            "label": label_values,
            "class_name": [args.class_names[label] for label in label_values],
        }
    ).to_csv(args.result_root / "batch_manifest.csv", index=False)
    if args.split_manifest_dir:
        if args.protocol != "batch_holdout":
            raise ValueError("--split-manifest-dir requires --protocol batch_holdout.")
        raw_splits = load_task_splits(
            x.index,
            y,
            args.split_manifest_dir,
            repeats=args.repeats,
            expected_seeds=range(args.seed, args.seed + args.repeats),
        )
        split_counts(raw_splits).to_csv(args.result_root / "shared_split_counts.csv", index=False)
        splits = [
            BatchSplit(
                scenario="shared_test",
                repeat=int(item["repeat"]),
                fold=1,
                seed=int(item["seed"]),
                train_inner_idx=item["train_inner_idx"],
                val_inner_idx=item["val_inner_idx"],
                outer_test_idx=item["outer_test_idx"],
            )
            for item in raw_splits
        ]
    elif args.protocol == "stratified_5cv":
        splits = make_stratified_5cv_splits(y, args.repeats, args.inner_val_ratio, args.seed, outer_folds=5)
    else:
        splits = make_batch_holdout_splits(y, batch, args.batch_scenarios, args.repeats, args.inner_val_ratio, args.shared_test_size, args.seed)
    status_rows: list[dict[str, object]] = []
    y_values = y.to_numpy(dtype=np.int64)

    for split in splits:
        assert_no_split_overlap(split)
        y_train = y_values[split.train_inner_idx]
        y_val = y_values[split.val_inner_idx]
        y_test = y_values[split.outer_test_idx]
        feature_sets = build_feature_sets(args, x, y, split, batch, status_rows)
        for feature_name, feature_set in feature_sets.items():
            for model_name in args.models:
                if model_name not in available_classical_models() and model_name != "dnn":
                    continue
                run_dir = args.result_root / "runs" / split.scenario / f"repeat_{split.repeat:02d}" / f"fold_{split.fold:02d}" / safe_name(feature_name) / safe_name(model_name)
                if args.skip_existing and (run_dir / "metrics.csv").exists():
                    continue
                run_model(args, split, feature_set, y_train, y_val, y_test, x.index[split.outer_test_idx], model_name, run_dir, batch)

    pd.DataFrame(status_rows).to_csv(args.result_root / "feature_selection_status.csv", index=False)
    leakage_rows = [
        {"item": "input_dataset", "status": "pass", "details": f"{len(y)} samples x {x.shape[1]} genes from {args.x_file}"},
        {"item": "protocol", "status": "pass", "details": f"{args.protocol} repeats={args.repeats}"},
        {"item": "split_overlap", "status": "pass", "details": "train_inner, val_inner, and outer_test are disjoint for every split."},
        {
            "item": "preprocessing_scope",
            "status": "intentional_leakage" if args.artifact_scope == "full_dataset" else "pass",
            "details": (
                f"Median imputer and {args.scaler} scaler fit on the full dataset."
                if args.artifact_scope == "full_dataset"
                else f"Median imputer and {args.scaler} scaler fit only on train_inner."
            ),
        },
        {
            "item": "feature_selection_scope",
            "status": "intentional_leakage" if args.artifact_scope == "full_dataset" else "pass",
            "details": (
                "limma DEG uses the full dataset; TF/Hub/CFG intersections inherit that scope. The VAE weights still fit train_inner only."
                if args.artifact_scope == "full_dataset"
                else "limma DEG and VAE weights fit train_inner only; TF and optional Hub are external lists intersected with train_inner DEG."
            ),
        },
        {
            "item": "cfg_supplement_proxy",
            "status": "opt_in_not_independent" if args.allow_cfg_supplement_proxy else "safely_skipped",
            "details": (
                "MOESM3-5 proxy explicitly enabled; its original full-dataset DEG membership is not independent of GSE holdouts."
                if args.allow_cfg_supplement_proxy
                else "MOESM3-5 proxy disabled; enable only with --allow-cfg-supplement-proxy."
            ),
        },
        {
            "item": "hub_genes",
            "status": "conditional_reconstruction" if args.hprd_network_file else "skipped_missing_hprd_network",
            "details": (
                f"User-supplied network with stored provenance: {args.hprd_network_file}"
                if args.hprd_network_file
                else "HPRD is absent from the official supplements; no file was downloaded or substituted."
            ),
        },
    ]
    pd.DataFrame(leakage_rows).to_csv(args.result_root / "leakage_audit.csv", index=False)
    summarize(args.result_root)
    print(f"Finished. Results written to {args.result_root}", flush=True)


if __name__ == "__main__":
    main()
