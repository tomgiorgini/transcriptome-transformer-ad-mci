#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import torch  # Load before sklearn on Windows to avoid PyTorch DLL initialization failures.
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler

ROOT = Path(__file__).resolve().parents[3]
MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))
COMMON_DIR = Path(__file__).resolve().parents[1]
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

from data import assert_no_overlap, infer_batch_labels, load_ad_mci_dataset, make_batch_holdout_splits, make_stratified_5cv_splits
from feature_selection import load_selected_genes, run_limma_dgs, write_dgs_inputs, write_manifest
from metrics import classification_report_frame, confusion_frame, predictions_frame
from models import train_dgs_tabnet
from summarize_results import summarize
from strict_v2_utils import cv_score_estimator


DEFAULT_X = ROOT / "task_dataset" / "processed" / "ad_mci_binary" / "X_ad_mci.csv"
DEFAULT_Y = ROOT / "task_dataset" / "processed" / "ad_mci_binary" / "y_ad_mci.csv"
DEFAULT_RESULT_ROOT = ROOT / "results" / "SOTA" / "tabnet"
DEFAULT_GSE63060_METADATA = ROOT / "pretraining_dataset" / "geo_downloads" / "GSE63060" / "eset_1_GPL6947" / "GSE63060_sample_metadata.tsv.gz"
DEFAULT_GSE63061_METADATA = ROOT / "pretraining_dataset" / "geo_downloads" / "GSE63061" / "eset_1_GPL10558" / "GSE63061_sample_metadata.tsv.gz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paper-like DGS-TabNet reproduction for local AD vs MCI.")
    parser.add_argument("--x-file", type=Path, default=DEFAULT_X)
    parser.add_argument("--y-file", type=Path, default=DEFAULT_Y)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--protocol", choices=["batch_holdout", "stratified_5cv"], default="batch_holdout")
    parser.add_argument(
        "--batch-scenarios",
        nargs="+",
        default=["shared_test", "test_gse63060", "test_gse63061"],
        choices=["shared_test", "test_gse63060", "test_gse63061"],
    )
    parser.add_argument("--shared-test-size", type=float, default=0.2)
    parser.add_argument("--gse63060-metadata", type=Path, default=DEFAULT_GSE63060_METADATA)
    parser.add_argument("--gse63061-metadata", type=Path, default=DEFAULT_GSE63061_METADATA)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--inner-val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--artifact-scope", choices=["full_dataset", "train_inner"], default="full_dataset")
    parser.add_argument("--dgs-method", choices=["limma"], default="limma")
    parser.add_argument("--dgs-adj-p-threshold", type=float, default=0.01)
    parser.add_argument(
        "--dgs-min-genes",
        type=int,
        default=0,
        help="Minimum acceptable DGS gene count. Threshold fallbacks continue until this count is reached; 0 disables this rule.",
    )
    parser.add_argument(
        "--dgs-fallback-thresholds",
        nargs="*",
        type=float,
        default=[],
        help="If limma selects zero genes at the primary FDR threshold, retry these relaxed FDR thresholds in order.",
    )
    parser.add_argument(
        "--dgs-fallback-p-thresholds",
        nargs="*",
        type=float,
        default=[],
        help="If all FDR thresholds select zero genes, retry using nominal p-value thresholds in order.",
    )
    parser.add_argument(
        "--dgs-threshold-cv",
        action="store_true",
        help="Choose among DGS threshold candidates with 5-fold CV on train_inner. Disabled by default for paper-like strict-v2.",
    )
    parser.add_argument(
        "--skip-below-min-genes",
        action="store_true",
        help="Mark a split as skipped if no DGS threshold reaches --dgs-min-genes instead of training on the largest too-small set.",
    )
    parser.add_argument("--deg-rule", choices=["no_cap"], default="no_cap")
    parser.add_argument("--models", nargs="+", default=["dgs_tabnet"], choices=["dgs_tabnet"])
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--threshold-mode", choices=["fixed_0_5", "validation_macro_f1", "validation_accuracy", "validation_accuracy_macro_f1"], default="fixed_0_5")
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


def preprocess_selected_split(
    x_train: pd.DataFrame,
    x_val: pd.DataFrame,
    x_test: pd.DataFrame,
    selected_genes: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    imputer = SimpleImputer(strategy="median")
    scaler = MinMaxScaler()
    x_train_sel = x_train.loc[:, selected_genes]
    x_val_sel = x_val.loc[:, selected_genes]
    x_test_sel = x_test.loc[:, selected_genes]
    train_imp = imputer.fit_transform(x_train_sel)
    val_imp = imputer.transform(x_val_sel)
    test_imp = imputer.transform(x_test_sel)
    train_scaled = np.nan_to_num(scaler.fit_transform(train_imp).astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0)
    val_scaled = np.nan_to_num(scaler.transform(val_imp).astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0)
    test_scaled = np.nan_to_num(scaler.transform(test_imp).astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0)
    return (
        pd.DataFrame(train_scaled, index=x_train.index, columns=selected_genes),
        pd.DataFrame(val_scaled, index=x_val.index, columns=selected_genes),
        pd.DataFrame(test_scaled, index=x_test.index, columns=selected_genes),
        {
            "imputer": "SimpleImputer(strategy=median) fit on train_inner selected genes",
            "scaler": "MinMaxScaler fit on train_inner selected genes",
            "n_input_features": int(len(selected_genes)),
        },
    )


def copy_dgs_outputs(global_dgs_dir: Path, run_dir: Path) -> None:
    for name in ("dgs_table.csv", "selected_genes.txt"):
        shutil.copy2(global_dgs_dir / name, run_dir / name)


def run_dgs_with_threshold_fallback(
    args: argparse.Namespace,
    x_dgs_file: Path,
    y_dgs_file: Path,
    out_dir: Path,
    available_genes: pd.Index,
) -> tuple[list[str], dict[str, Any]]:
    thresholds = [args.dgs_adj_p_threshold, *args.dgs_fallback_thresholds]
    errors: list[str] = []
    min_genes = max(1, int(args.dgs_min_genes)) if args.dgs_min_genes else 1

    def below_min_manifest(selected: list[str], manifest: dict[str, Any], errors: list[str]) -> dict[str, Any]:
        return {
            **manifest,
            "status": "skipped_below_min_genes",
            "threshold_selection": "largest_available_below_minimum",
            "dgs_min_genes": int(args.dgs_min_genes),
            "minimum_gene_count_met": False,
            "threshold_attempt_errors": errors,
            "n_selected_genes": int(len(selected)),
        }

    def remember_best(
        current: tuple[int, list[str], dict[str, Any], Path] | None,
        selected: list[str],
        manifest: dict[str, Any],
        attempt_dir: Path,
    ) -> tuple[int, list[str], dict[str, Any], Path]:
        candidate = (len(selected), selected, manifest, attempt_dir)
        if current is None or candidate[0] > current[0]:
            return candidate
        return current

    def promote_attempt(attempt_dir: Path) -> None:
        if attempt_dir != out_dir:
            copy_dgs_outputs(attempt_dir, out_dir)
            shutil.copy2(attempt_dir / "dgs_manifest.csv", out_dir / "dgs_manifest.csv")

    best_by_count: tuple[int, list[str], dict[str, Any], Path] | None = None
    if args.dgs_threshold_cv:
        candidate_rows: list[dict[str, Any]] = []
        best: tuple[float, list[str], dict[str, Any], Path] | None = None
        for attempt, threshold in enumerate(thresholds, start=1):
            attempt_dir = out_dir if attempt == 1 else out_dir / f"fdr_{str(threshold).replace('.', 'p')}"
            try:
                manifest = run_limma_dgs(MODULE_DIR / "dgs_limma.R", x_dgs_file, y_dgs_file, attempt_dir, threshold)
                selected_genes = load_selected_genes(attempt_dir / "selected_genes.txt", available_genes)
            except ValueError as exc:
                errors.append(f"FDR {threshold}: {exc}")
                continue
            best_by_count = remember_best(best_by_count, selected_genes, manifest, attempt_dir)
            if len(selected_genes) < min_genes:
                errors.append(f"FDR {threshold}: selected {len(selected_genes)} genes, below minimum {min_genes}")
                continue
            x_dgs = pd.read_csv(x_dgs_file, index_col=0).loc[:, selected_genes]
            y_dgs = pd.read_csv(y_dgs_file, index_col=0).iloc[:, 0].to_numpy(dtype=np.int64)
            estimator = make_pipeline(
                SimpleImputer(strategy="median"),
                MinMaxScaler(),
                LogisticRegression(max_iter=5000, solver="liblinear", class_weight="balanced", random_state=args.seed),
            )
            score, fold_scores = cv_score_estimator(estimator, x_dgs, y_dgs, folds=5, seed=args.seed + attempt)
            candidate_rows.append({"threshold_type": "adj.P.Val", "threshold": threshold, "n_genes": len(selected_genes), "strict_v2_cv_mean": score, "strict_v2_cv_folds": fold_scores})
            if best is None or score > best[0]:
                best = (score, selected_genes, manifest, attempt_dir)
        for attempt, threshold in enumerate(args.dgs_fallback_p_thresholds, start=1):
            attempt_dir = out_dir / f"p_{str(threshold).replace('.', 'p')}"
            try:
                manifest = run_limma_dgs(
                    MODULE_DIR / "dgs_limma.R",
                    x_dgs_file,
                    y_dgs_file,
                    attempt_dir,
                    args.dgs_adj_p_threshold,
                    p_value_threshold=threshold,
                )
                selected_genes = load_selected_genes(attempt_dir / "selected_genes.txt", available_genes)
            except ValueError as exc:
                errors.append(f"P.Value {threshold}: {exc}")
                continue
            best_by_count = remember_best(best_by_count, selected_genes, manifest, attempt_dir)
            if len(selected_genes) < min_genes:
                errors.append(f"P.Value {threshold}: selected {len(selected_genes)} genes, below minimum {min_genes}")
                continue
            x_dgs = pd.read_csv(x_dgs_file, index_col=0).loc[:, selected_genes]
            y_dgs = pd.read_csv(y_dgs_file, index_col=0).iloc[:, 0].to_numpy(dtype=np.int64)
            estimator = make_pipeline(
                SimpleImputer(strategy="median"),
                MinMaxScaler(),
                LogisticRegression(max_iter=5000, solver="liblinear", class_weight="balanced", random_state=args.seed),
            )
            score, fold_scores = cv_score_estimator(estimator, x_dgs, y_dgs, folds=5, seed=args.seed + 100 + attempt)
            candidate_rows.append({"threshold_type": "P.Value", "threshold": threshold, "n_genes": len(selected_genes), "strict_v2_cv_mean": score, "strict_v2_cv_folds": fold_scores})
            if best is None or score > best[0]:
                best = (score, selected_genes, manifest, attempt_dir)
        if best is None:
            if best_by_count is None:
                raise ValueError("DGS selected zero genes for all configured thresholds. " + " | ".join(errors))
            _, selected_genes, manifest, attempt_dir = best_by_count
            promote_attempt(attempt_dir)
            manifest = below_min_manifest(selected_genes, manifest, errors) if args.skip_below_min_genes else {
                **manifest,
                "threshold_selection": "largest_available_below_minimum",
                "dgs_min_genes": int(args.dgs_min_genes),
                "minimum_gene_count_met": False,
                "threshold_attempt_errors": errors,
            }
            return selected_genes, manifest
        best_score, selected_genes, manifest, attempt_dir = best
        promote_attempt(attempt_dir)
        pd.DataFrame(candidate_rows).to_csv(out_dir / "dgs_threshold_cv_trace.csv", index=False)
        manifest = {
            **manifest,
            "threshold_selection": "5fold_cv_on_train_inner",
            "threshold_cv_score": best_score,
            "threshold_candidates": candidate_rows,
            "dgs_min_genes": int(args.dgs_min_genes),
            "minimum_gene_count_met": bool(len(selected_genes) >= min_genes),
        }
        return selected_genes, manifest

    for attempt, threshold in enumerate(thresholds, start=1):
        attempt_dir = out_dir if attempt == 1 else out_dir / f"fdr_{str(threshold).replace('.', 'p')}"
        try:
            manifest = run_limma_dgs(MODULE_DIR / "dgs_limma.R", x_dgs_file, y_dgs_file, attempt_dir, threshold)
            selected_genes = load_selected_genes(attempt_dir / "selected_genes.txt", available_genes)
        except ValueError as exc:
            errors.append(f"FDR {threshold}: {exc}")
            continue
        best_by_count = remember_best(best_by_count, selected_genes, manifest, attempt_dir)
        if len(selected_genes) < min_genes:
            errors.append(f"FDR {threshold}: selected {len(selected_genes)} genes, below minimum {min_genes}")
            continue
        promote_attempt(attempt_dir)
        manifest = {
            **manifest,
            "primary_adj_p_threshold": args.dgs_adj_p_threshold,
            "effective_adj_p_threshold": threshold,
            "threshold_fallback_used": bool(threshold != args.dgs_adj_p_threshold),
            "threshold_attempts": thresholds,
            "dgs_min_genes": int(args.dgs_min_genes),
            "minimum_gene_count_met": bool(len(selected_genes) >= min_genes),
        }
        return selected_genes, manifest
    for attempt, threshold in enumerate(args.dgs_fallback_p_thresholds, start=1):
        attempt_dir = out_dir / f"p_{str(threshold).replace('.', 'p')}"
        try:
            manifest = run_limma_dgs(
                MODULE_DIR / "dgs_limma.R",
                x_dgs_file,
                y_dgs_file,
                attempt_dir,
                args.dgs_adj_p_threshold,
                p_value_threshold=threshold,
            )
            selected_genes = load_selected_genes(attempt_dir / "selected_genes.txt", available_genes)
        except ValueError as exc:
            errors.append(f"P.Value {threshold}: {exc}")
            continue
        best_by_count = remember_best(best_by_count, selected_genes, manifest, attempt_dir)
        if len(selected_genes) < min_genes:
            errors.append(f"P.Value {threshold}: selected {len(selected_genes)} genes, below minimum {min_genes}")
            continue
        promote_attempt(attempt_dir)
        manifest = {
            **manifest,
            "primary_adj_p_threshold": args.dgs_adj_p_threshold,
            "effective_p_value_threshold": threshold,
            "threshold_fallback_used": True,
            "threshold_attempts": thresholds,
            "p_value_threshold_attempts": args.dgs_fallback_p_thresholds,
            "dgs_min_genes": int(args.dgs_min_genes),
            "minimum_gene_count_met": bool(len(selected_genes) >= min_genes),
        }
        return selected_genes, manifest
    if best_by_count is not None:
        _, selected_genes, manifest, attempt_dir = best_by_count
        promote_attempt(attempt_dir)
        manifest = below_min_manifest(selected_genes, manifest, errors) if args.skip_below_min_genes else {
            **manifest,
            "threshold_selection": "largest_available_below_minimum",
            "dgs_min_genes": int(args.dgs_min_genes),
            "minimum_gene_count_met": False,
            "threshold_attempt_errors": errors,
        }
        return selected_genes, manifest
    raise ValueError("DGS selected zero genes for all configured thresholds. " + " | ".join(errors))


def write_run_artifacts(
    run_dir: Path,
    metric_row: dict[str, Any],
    sample_ids: pd.Index,
    y_test: np.ndarray,
    scores: np.ndarray,
    y_pred: np.ndarray,
    feature_importances: pd.DataFrame,
    model_metadata: dict[str, Any],
    selected_genes: list[str],
    dgs_manifest: dict[str, Any],
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([metric_row]).to_csv(run_dir / "metrics.csv", index=False)
    predictions_frame(sample_ids, y_test, scores, y_pred).to_csv(run_dir / "predictions.csv", index=False)
    confusion_frame(y_test, y_pred).to_csv(run_dir / "confusion_matrix.csv")
    classification_report_frame(y_test, y_pred).to_csv(run_dir / "classification_report.csv")
    feature_importances.to_csv(run_dir / "feature_importances.csv", index=False)
    important = feature_importances[feature_importances["importance"] > 0]["gene"].astype(str).tolist()
    (run_dir / "global_important_genes.txt").write_text("\n".join(important) + ("\n" if important else ""), encoding="utf-8")
    (run_dir / "selected_genes.txt").write_text("\n".join(selected_genes) + "\n", encoding="utf-8")
    write_json(run_dir / "hyperparameters.json", model_metadata)
    write_json(run_dir / "run_manifest.json", {"metrics": metric_row, "model": model_metadata, "dgs": dgs_manifest})


def write_audit(result_root: Path, args: argparse.Namespace, x: pd.DataFrame, y: pd.Series, run_count: int, n_selected: int | str) -> None:
    fs_status = "intentional_leakage" if args.artifact_scope == "full_dataset" else "pass"
    fs_details = (
        f"limma DGS fit on full unscaled expression dataset; primary adj.P.Value < {args.dgs_adj_p_threshold}; FDR fallback {args.dgs_fallback_thresholds}; nominal p-value fallback {args.dgs_fallback_p_thresholds} if zero genes."
        if args.artifact_scope == "full_dataset"
        else f"limma DGS fit separately on each train_inner unscaled expression matrix; primary adj.P.Value < {args.dgs_adj_p_threshold}; FDR fallback {args.dgs_fallback_thresholds}; nominal p-value fallback {args.dgs_fallback_p_thresholds} if zero genes."
    )
    audit = pd.DataFrame(
        [
            {"item": "input_dataset", "status": "pass", "details": f"{len(y)} samples x {x.shape[1]} genes from {args.x_file}"},
            {
                "item": "protocol",
                "status": "pass",
                "details": f"{args.repeats} repeat(s) across batch scenarios {', '.join(args.batch_scenarios)}; shared_test_size={args.shared_test_size}; inner_val_ratio={args.inner_val_ratio}",
            },
            {"item": "preprocessing_scope", "status": "pass", "details": "Median imputer and MinMaxScaler fit on train_inner after DGS gene selection."},
            {"item": "feature_selection_scope", "status": fs_status, "details": fs_details},
            {"item": "dgs_scope", "status": fs_status, "details": f"{n_selected} selected genes."},
            {"item": "split_overlap", "status": "pass", "details": "Train, validation, and test indices checked for every scenario."},
            {"item": "completed_runs", "status": "pass", "details": str(run_count)},
        ]
    )
    audit.to_csv(result_root / "leakage_audit.csv", index=False)


def main() -> None:
    args = parse_args()
    args.x_file = resolve(args.x_file)
    args.y_file = resolve(args.y_file)
    args.result_root = resolve(args.result_root)
    args.gse63060_metadata = resolve(args.gse63060_metadata)
    args.gse63061_metadata = resolve(args.gse63061_metadata)
    if args.strict_v2:
        args.repeats = 10
        args.batch_scenarios = ["shared_test", "test_gse63060", "test_gse63061"]
        args.artifact_scope = "train_inner"
        args.dgs_adj_p_threshold = 0.01
        args.dgs_fallback_thresholds = [0.05, 0.1, 0.2]
        args.dgs_fallback_p_thresholds = [0.001, 0.005, 0.01, 0.05, 0.1]
        if args.dgs_min_genes <= 0:
            args.dgs_min_genes = 100
        args.skip_below_min_genes = True
        args.models = ["dgs_tabnet"]
        args.threshold_mode = "fixed_0_5"
        args.skip_existing = False
        args.overwrite_results = True
    if args.smoke:
        args.repeats = 1
        args.batch_scenarios = args.batch_scenarios[:1]
        args.max_epochs = min(args.max_epochs, 1)
        args.patience = min(args.patience, 1)

    if args.overwrite_results and args.result_root.exists():
        shutil.rmtree(args.result_root)
    args.result_root.mkdir(parents=True, exist_ok=True)
    write_json(args.result_root / "args.json", vars(args))

    x, y = load_ad_mci_dataset(args.x_file, args.y_file)
    batch = infer_batch_labels(x.index, args.gse63060_metadata, args.gse63061_metadata)
    pd.DataFrame({"sample_id": x.index.astype(str), "batch": batch.to_numpy(), "label": y.to_numpy(dtype=np.int64)}).to_csv(
        args.result_root / "batch_manifest.csv", index=False
    )

    global_selected_genes: list[str] | None = None
    global_dgs_manifest: dict[str, Any] | None = None
    dgs_dir: Path | None = None
    if args.artifact_scope == "full_dataset":
        print("Fitting full-dataset limma DGS on unscaled expression matrix (intentional leakage experiment).", flush=True)
        dgs_dir = args.result_root / "full_dataset_artifacts"
        x_dgs_file, y_dgs_file = write_dgs_inputs(x, y, dgs_dir)
        global_selected_genes, global_dgs_manifest = run_dgs_with_threshold_fallback(args, x_dgs_file, y_dgs_file, dgs_dir, x.columns)
        global_dgs_manifest = {
            **global_dgs_manifest,
            "dgs_expression_source": "unscaled_input_expression",
            "deg_rule": args.deg_rule,
            "artifact_scope": args.artifact_scope,
        }
        write_json(args.result_root / "dgs_manifest.json", global_dgs_manifest)
        write_json(dgs_dir / "dgs_manifest.json", global_dgs_manifest)

    if args.protocol == "stratified_5cv":
        splits = make_stratified_5cv_splits(y, args.repeats, args.inner_val_ratio, args.seed, outer_folds=5)
    else:
        splits = make_batch_holdout_splits(y, batch, args.batch_scenarios, args.repeats, args.inner_val_ratio, args.shared_test_size, args.seed)
    run_rows: list[dict[str, Any]] = []

    for split in splits:
        assert_no_overlap(split)
        train_idx = split["train_inner_idx"]
        val_idx = split["val_inner_idx"]
        test_idx = split["outer_test_idx"]
        y_values = y.to_numpy(dtype=np.int64)
        x_train_raw = x.iloc[train_idx].copy()
        x_val_raw = x.iloc[val_idx].copy()
        x_test_raw = x.iloc[test_idx].copy()
        y_train = y_values[train_idx]
        y_val = y_values[val_idx]
        y_test = y_values[test_idx]
        split_seed = int(split["seed"] * 1000 + split["scenario_index"] * 100 + split["repeat"])
        split_dir = args.result_root / "runs" / split["scenario"] / f"repeat_{split['repeat']:02d}" / f"fold_{split['fold']:02d}"
        split_dir.mkdir(parents=True, exist_ok=True)
        if args.artifact_scope == "train_inner":
            split_dgs_dir = split_dir / "dgs_artifacts"
            x_dgs_file, y_dgs_file = write_dgs_inputs(x_train_raw, y.iloc[train_idx], split_dgs_dir)
            selected_genes, dgs_manifest = run_dgs_with_threshold_fallback(args, x_dgs_file, y_dgs_file, split_dgs_dir, x.columns)
            dgs_manifest = {
                **dgs_manifest,
                "dgs_expression_source": "train_inner_unscaled_input_expression",
                "deg_rule": args.deg_rule,
                "artifact_scope": args.artifact_scope,
            }
            dgs_source_dir = split_dgs_dir
        else:
            selected_genes = list(global_selected_genes or [])
            dgs_manifest = dict(global_dgs_manifest or {})
            dgs_source_dir = dgs_dir

        if str(dgs_manifest.get("status", "")).startswith("skipped"):
            for model_name in args.models:
                run_dir = split_dir / model_name
                run_dir.mkdir(parents=True, exist_ok=True)
                (run_dir / "selected_genes.txt").write_text("\n".join(selected_genes) + ("\n" if selected_genes else ""), encoding="utf-8")
                write_json(
                    run_dir / "run_manifest.json",
                    {
                        "status": dgs_manifest.get("status"),
                        "reason": "DGS did not reach the configured minimum gene count; TabNet training skipped for credibility.",
                        "dgs": dgs_manifest,
                        "scenario": split["scenario"],
                        "repeat": split["repeat"],
                        "n_selected_genes": int(len(selected_genes)),
                    },
                )
            continue

        x_train, x_val, x_test, preprocessing_manifest = preprocess_selected_split(x_train_raw, x_val_raw, x_test_raw, selected_genes)

        split_meta = {
            "protocol": args.protocol,
            "scenario": split["scenario"],
            "repeat": split["repeat"],
            "fold": split["fold"],
            "seed": split_seed,
            "n_train_inner": int(len(train_idx)),
            "n_val_inner": int(len(val_idx)),
            "n_outer_test": int(len(test_idx)),
            "n_selected_genes": int(len(selected_genes)),
            "train_batches": "|".join(sorted(batch.iloc[train_idx].unique().tolist())),
            "val_batches": "|".join(sorted(batch.iloc[val_idx].unique().tolist())),
            "test_batches": "|".join(sorted(batch.iloc[test_idx].unique().tolist())),
        }
        write_json(
            split_dir / "split_manifest.json",
            {
                **split_meta,
                **preprocessing_manifest,
                "train_inner_ids": x.index[train_idx].astype(str).tolist(),
                "val_inner_ids": x.index[val_idx].astype(str).tolist(),
                "outer_test_ids": x.index[test_idx].astype(str).tolist(),
            },
        )
        for model_name in args.models:
            run_dir = split_dir / model_name
            if args.skip_existing and (run_dir / "metrics.csv").exists():
                continue
            print(f"Running scenario={split['scenario']} repeat={split['repeat']} model={model_name} genes={len(selected_genes)}", flush=True)
            scores, y_pred, metrics, feature_importances, model_meta = train_dgs_tabnet(
                x_train,
                y_train,
                x_val,
                y_val,
                x_test,
                y_test,
                split_seed,
                args.max_epochs,
                args.patience,
                args.batch_size,
                run_dir,
                args.threshold_mode,
            )
            row = {**split_meta, "model": model_name, **metrics}
            if dgs_source_dir is not None:
                copy_dgs_outputs(dgs_source_dir, run_dir)
            write_run_artifacts(run_dir, row, x.index[test_idx], y_test, scores, y_pred, feature_importances, model_meta, selected_genes, dgs_manifest)
            run_rows.append(row)

    all_metrics, summary = summarize(args.result_root)
    all_metrics.to_csv(args.result_root / "all_metrics.csv", index=False)
    summary.to_csv(args.result_root / "summary_by_method.csv", index=False)
    summary.to_csv(args.result_root / "ranking_by_pr_auc.csv", index=False)
    if not summary.empty:
        summary.sort_values("roc_auc_mean", ascending=False).to_csv(args.result_root / "ranking_by_roc_auc.csv", index=False)
        summary.sort_values("macro_f1_mean", ascending=False).to_csv(args.result_root / "ranking_by_macro_f1.csv", index=False)
        summary.sort_values("accuracy_mean", ascending=False).to_csv(args.result_root / "ranking_by_accuracy.csv", index=False)
    selected_count: int | str = len(global_selected_genes) if global_selected_genes is not None else "per-split train_inner DGS"
    write_audit(args.result_root, args, x, y, len(all_metrics), selected_count)
    print(f"Finished. Results written to {args.result_root}", flush=True)


if __name__ == "__main__":
    main()
