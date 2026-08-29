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
from sklearn.preprocessing import MinMaxScaler

ROOT = Path(__file__).resolve().parents[3]
MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))
COMMON_DIR = Path(__file__).resolve().parents[1]
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

from data import assert_no_overlap, infer_batch_labels, load_ad_mci_dataset, make_batch_holdout_splits, make_stratified_5cv_splits
from feature_selection import PAPER_SFBS_TARGET_GENES, select_xgboost_sfbs_genes
from metrics import classification_report_frame, confusion_frame, predictions_frame
from models import fit_predict_dl, fit_predict_sklearn
from sampling import apply_sampling
from summarize_results import summarize
from shared_test_splits import load_task_splits, split_counts


DEFAULT_X = ROOT / "task_dataset" / "processed" / "ad_mci_binary" / "X_ad_mci.csv"
DEFAULT_Y = ROOT / "task_dataset" / "processed" / "ad_mci_binary" / "y_ad_mci.csv"
DEFAULT_RESULT_ROOT = ROOT / "results" / "SOTA" / "diagnostics-2025"
DEFAULT_GSE63060_METADATA = ROOT / "pretraining_dataset" / "geo_downloads" / "GSE63060" / "eset_1_GPL6947" / "GSE63060_sample_metadata.tsv.gz"
DEFAULT_GSE63061_METADATA = ROOT / "pretraining_dataset" / "geo_downloads" / "GSE63061" / "eset_1_GPL10558" / "GSE63061_sample_metadata.tsv.gz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnostics 2025 leakage-safe reproduction for local AD vs MCI.")
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
    parser.add_argument("--gse63060-metadata", type=Path, default=DEFAULT_GSE63060_METADATA)
    parser.add_argument("--gse63061-metadata", type=Path, default=DEFAULT_GSE63061_METADATA)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--inner-val-ratio", type=float, default=0.125, help="Fraction of the post-test pool; 0.125 yields an overall 70/10/20 split when test size is 0.20.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--models", nargs="+", choices=["dl", "svm", "gbm", "rf"], default=["dl", "svm", "gbm", "rf"])
    parser.add_argument("--sampling", nargs="+", choices=["no_smote", "borderline_smote"], default=["no_smote", "borderline_smote"])
    parser.add_argument(
        "--artifact-scope",
        choices=["train_inner", "full_dataset"],
        default="train_inner",
        help="Fit preprocessing/feature selection on train_inner or once on the full dataset with intentional leakage.",
    )
    parser.add_argument("--xgb-top-k", type=int, default=300)
    parser.add_argument("--sfbs-min-genes", type=int, default=20)
    parser.add_argument("--sfbs-max-genes", type=int, default=95)
    parser.add_argument("--sfbs-step", type=int, default=5)
    parser.add_argument("--sfbs-cv-folds", type=int, default=5)
    parser.add_argument("--sfbs-mode", choices=["true", "approximate_lr"], default="true")
    parser.add_argument(
        "--sfbs-target-genes",
        type=int,
        default=PAPER_SFBS_TARGET_GENES,
        help="Fixed number of genes retained by true SFBS (95 in the paper).",
    )
    parser.add_argument(
        "--allow-sfbs-fallback",
        action="store_true",
        help="If true SFBS fails, explicitly permit a labelled XGBoost-ranking-only fallback. Disabled by default.",
    )
    parser.add_argument("--n-jobs", type=int, default=2)
    parser.add_argument("--dl-epochs", type=int, default=4000)
    parser.add_argument("--dl-patience", type=int, default=100)
    parser.add_argument("--dl-batch-size", type=int, default=5)
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
            "artifact_scope": "full_dataset",
            "n_input_features": int(x.shape[1]),
        },
    )


def copy_feature_selection_artifacts(feature_dir: Path, run_dir: Path) -> None:
    for name in ("selected_genes.txt", "xgboost_feature_ranking.csv", "sfbs_trace.csv", "feature_selection_manifest.json"):
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
    sampling_manifest: dict[str, Any],
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([metric_row]).to_csv(run_dir / "metrics.csv", index=False)
    predictions_frame(sample_ids, y_test, scores, y_pred).to_csv(run_dir / "predictions.csv", index=False)
    confusion_frame(y_test, y_pred).to_csv(run_dir / "confusion_matrix.csv")
    classification_report_frame(y_test, y_pred).to_csv(run_dir / "classification_report.csv")
    write_json(run_dir / "sampling_manifest.json", sampling_manifest)
    write_json(run_dir / "hyperparameters.json", model_metadata)
    write_json(run_dir / "run_manifest.json", {"metrics": metric_row, "model": model_metadata, "sampling": sampling_manifest})


def write_audit(result_root: Path, args: argparse.Namespace, x: pd.DataFrame, y: pd.Series, run_count: int) -> None:
    audit = pd.DataFrame(
        [
            {"item": "input_dataset", "status": "pass", "details": f"{len(y)} samples x {x.shape[1]} genes from {args.x_file}"},
            {
                "item": "protocol",
                "status": "pass",
                "details": f"{args.repeats} repeats across {', '.join(args.batch_scenarios)}; shared_test_size={args.shared_test_size}; inner_val_ratio={args.inner_val_ratio}",
            },
            {"item": "split_overlap", "status": "pass", "details": "Train, validation, and test indices checked for every split."},
            {
                "item": "preprocessing_scope",
                "status": "intentional_leakage" if args.artifact_scope == "full_dataset" else "pass",
                "details": "Median imputer and MinMaxScaler fit on full dataset." if args.artifact_scope == "full_dataset" else "Median imputer and MinMaxScaler fit only on train_inner.",
            },
            {
                "item": "xgboost_scope",
                "status": "intentional_leakage" if args.artifact_scope == "full_dataset" else "pass",
                "details": "XGBoost feature ranking fit on full dataset." if args.artifact_scope == "full_dataset" else "XGBoost feature ranking fit only on train_inner.",
            },
            {
                "item": "sfbs_scope",
                "status": "intentional_leakage" if args.artifact_scope == "full_dataset" else "pass",
                "details": (
                    f"{args.sfbs_mode} feature selection fit on full dataset."
                    if args.artifact_scope == "full_dataset"
                    else f"{args.sfbs_mode} feature selection fit only on train_inner; true mode uses fixed k={args.sfbs_target_genes}."
                ),
            },
            {
                "item": "paper_reproducibility",
                "status": "note",
                "details": "Best-effort adaptation: the paper omits XGBoost parameters and the SFBS estimator, scorer, stopping/tie-breaking rules, genes, and seeds; manifests record the implementation assumptions.",
            },
            {"item": "smote_scope", "status": "pass", "details": "BorderlineSMOTE applied only to train_inner after feature selection."},
            {"item": "completed_runs", "status": "pass", "details": str(run_count)},
        ]
    )
    audit.to_csv(result_root / "leakage_audit.csv", index=False)


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
        args.models = ["dl"]
        args.sampling = ["borderline_smote"]
        args.threshold_mode = "fixed_0_5"
        args.xgb_top_k = 300
        args.sfbs_min_genes = 20
        args.sfbs_max_genes = 95
        args.sfbs_step = 5
        args.sfbs_cv_folds = 5
        args.sfbs_mode = "true"
        args.sfbs_target_genes = PAPER_SFBS_TARGET_GENES
        args.allow_sfbs_fallback = False
        args.skip_existing = False
        args.overwrite_results = True
    if args.smoke:
        args.repeats = 1
        args.batch_scenarios = args.batch_scenarios[:1]
        args.models = [model for model in args.models if model in {"dl", "svm"}]
        args.xgb_top_k = min(args.xgb_top_k, 50)
        args.sfbs_min_genes = min(args.sfbs_min_genes, 5)
        args.sfbs_max_genes = min(args.sfbs_max_genes, 15)
        args.sfbs_target_genes = min(args.sfbs_target_genes, 15)
        args.sfbs_step = max(args.sfbs_step, 5)
        args.dl_epochs = min(args.dl_epochs, 2)
        args.dl_patience = min(args.dl_patience, 1)

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
    run_rows: list[dict[str, Any]] = []
    x_full_scaled: pd.DataFrame | None = None
    full_preprocessing_manifest: dict[str, Any] | None = None
    full_selected_genes: list[str] | None = None
    full_fs_manifest: dict[str, Any] | None = None
    full_feature_dir = args.result_root / "full_dataset_artifacts" / "feature_selection"
    if args.artifact_scope == "full_dataset":
        print("Selecting Diagnostics 2025 features on full dataset (intentional leakage).", flush=True)
        x_full_scaled, full_preprocessing_manifest = preprocess_full(x)
        selected_path = full_feature_dir / "selected_genes.txt"
        if args.skip_existing and selected_path.exists():
            full_selected_genes = [line.strip() for line in selected_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            full_fs_manifest = json.loads((full_feature_dir / "feature_selection_manifest.json").read_text(encoding="utf-8"))
        else:
            fs = select_xgboost_sfbs_genes(
                x_full_scaled,
                y_values,
                args.seed,
                top_k=args.xgb_top_k,
                min_genes=args.sfbs_min_genes,
                max_genes=args.sfbs_max_genes,
                step=args.sfbs_step,
                cv_folds=args.sfbs_cv_folds,
                n_jobs=args.n_jobs,
                sfbs_mode=args.sfbs_mode,
                target_genes=args.sfbs_target_genes,
                allow_sfbs_fallback=args.allow_sfbs_fallback,
            )
            full_feature_dir.mkdir(parents=True, exist_ok=True)
            fs.ranking.to_csv(full_feature_dir / "xgboost_feature_ranking.csv", index=False)
            fs.trace.to_csv(full_feature_dir / "sfbs_trace.csv", index=False)
            (full_feature_dir / "selected_genes.txt").write_text("\n".join(fs.selected_genes) + "\n", encoding="utf-8")
            full_fs_manifest = {**fs.metadata, "artifact_scope": "full_dataset", "selection_scope": "full_dataset"}
            write_json(full_feature_dir / "feature_selection_manifest.json", full_fs_manifest)
            write_json(full_feature_dir / "preprocessing_manifest.json", full_preprocessing_manifest)
            full_selected_genes = fs.selected_genes

    for split in splits:
        assert_no_overlap(split)
        train_idx = split["train_inner_idx"]
        val_idx = split["val_inner_idx"]
        test_idx = split["outer_test_idx"]
        scenario = split["scenario"]
        repeat = split["repeat"]
        split_seed = int(split["seed"]) if split.get("split_source") else int(split["seed"] * 1000 + split["scenario_index"] * 100 + repeat)
        split_dir = args.result_root / "runs" / scenario / f"repeat_{repeat:02d}" / f"fold_{split['fold']:02d}"
        feature_dir = split_dir / "feature_selection"
        split_dir.mkdir(parents=True, exist_ok=True)

        x_train_raw = x.iloc[train_idx].copy()
        x_val_raw = x.iloc[val_idx].copy()
        x_test_raw = x.iloc[test_idx].copy()
        y_train = y_values[train_idx]
        y_val = y_values[val_idx]
        y_test = y_values[test_idx]
        if args.artifact_scope == "full_dataset":
            if x_full_scaled is None or full_preprocessing_manifest is None:
                raise RuntimeError("Full-dataset preprocessing was not initialized.")
            x_train_scaled = x_full_scaled.iloc[train_idx].copy()
            x_val_scaled = x_full_scaled.iloc[val_idx].copy()
            x_test_scaled = x_full_scaled.iloc[test_idx].copy()
            preprocessing_manifest = full_preprocessing_manifest
        else:
            x_train_scaled, x_val_scaled, x_test_scaled, preprocessing_manifest = preprocess_split(x_train_raw, x_val_raw, x_test_raw)

        selected_path = feature_dir / "selected_genes.txt"
        if args.artifact_scope == "full_dataset":
            if full_selected_genes is None or full_fs_manifest is None:
                raise RuntimeError("Full-dataset feature selection was not initialized.")
            selected_genes = full_selected_genes
            fs_manifest = full_fs_manifest
            feature_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(full_feature_dir / "selected_genes.txt", feature_dir / "selected_genes.txt")
            shutil.copy2(full_feature_dir / "xgboost_feature_ranking.csv", feature_dir / "xgboost_feature_ranking.csv")
            shutil.copy2(full_feature_dir / "sfbs_trace.csv", feature_dir / "sfbs_trace.csv")
            shutil.copy2(full_feature_dir / "feature_selection_manifest.json", feature_dir / "feature_selection_manifest.json")
        elif args.skip_existing and selected_path.exists():
            selected_genes = [line.strip() for line in selected_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            fs_manifest = json.loads((feature_dir / "feature_selection_manifest.json").read_text(encoding="utf-8"))
        else:
            print(f"Selecting features scenario={scenario} repeat={repeat}", flush=True)
            fs = select_xgboost_sfbs_genes(
                x_train_scaled,
                y_train,
                split_seed,
                top_k=args.xgb_top_k,
                min_genes=args.sfbs_min_genes,
                max_genes=args.sfbs_max_genes,
                step=args.sfbs_step,
                cv_folds=args.sfbs_cv_folds,
                n_jobs=args.n_jobs,
                sfbs_mode=args.sfbs_mode,
                target_genes=args.sfbs_target_genes,
                allow_sfbs_fallback=args.allow_sfbs_fallback,
            )
            feature_dir.mkdir(parents=True, exist_ok=True)
            fs.ranking.to_csv(feature_dir / "xgboost_feature_ranking.csv", index=False)
            fs.trace.to_csv(feature_dir / "sfbs_trace.csv", index=False)
            (feature_dir / "selected_genes.txt").write_text("\n".join(fs.selected_genes) + "\n", encoding="utf-8")
            fs_manifest = fs.metadata
            write_json(feature_dir / "feature_selection_manifest.json", fs_manifest)
            selected_genes = fs.selected_genes

        x_train_sel = x_train_scaled.loc[:, selected_genes].copy()
        x_val_sel = x_val_scaled.loc[:, selected_genes].copy()
        x_test_sel = x_test_scaled.loc[:, selected_genes].copy()
        split_meta = {
            "protocol": args.protocol,
            "scenario": scenario,
            "repeat": repeat,
            "fold": split["fold"],
            "seed": split_seed,
            "n_train_inner": int(len(train_idx)),
            "n_val_inner": int(len(val_idx)),
            "n_outer_test": int(len(test_idx)),
            "train_batches": "|".join(sorted(batch.iloc[train_idx].unique().tolist())),
            "val_batches": "|".join(sorted(batch.iloc[val_idx].unique().tolist())),
            "test_batches": "|".join(sorted(batch.iloc[test_idx].unique().tolist())),
            "feature_selector": fs_manifest.get("feature_selector", "unknown"),
            "n_selected_genes": int(len(selected_genes)),
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

        for sampling_mode in args.sampling:
            x_sampled, y_sampled, sampling_manifest = apply_sampling(x_train_sel, y_train, sampling_mode, split_seed)
            for model_name in args.models:
                run_dir = split_dir / sampling_mode / model_name
                if args.skip_existing and (run_dir / "metrics.csv").exists():
                    continue
                print(f"Running scenario={scenario} repeat={repeat} sampling={sampling_mode} model={model_name} genes={len(selected_genes)}", flush=True)
                if model_name == "dl":
                    scores, y_pred, metrics, model_meta = fit_predict_dl(
                        x_sampled,
                        y_sampled,
                        x_val_sel,
                        y_val,
                        x_test_sel,
                        y_test,
                        split_seed,
                        args.dl_epochs,
                        args.dl_patience,
                        args.dl_batch_size,
                        run_dir,
                        args.threshold_mode,
                    )
                else:
                    scores, y_pred, metrics, model_meta = fit_predict_sklearn(
                        model_name,
                        x_sampled,
                        y_sampled,
                        x_test_sel,
                        y_test,
                        split_seed,
                        x_val_sel,
                        y_val,
                        args.threshold_mode,
                    )
                row = {**split_meta, "sampling": sampling_mode, "model": model_name, **metrics}
                run_dir.mkdir(parents=True, exist_ok=True)
                copy_feature_selection_artifacts(feature_dir, run_dir)
                write_run_artifacts(run_dir, row, x.index[test_idx], y_test, scores, y_pred, model_meta, sampling_manifest)
                run_rows.append(row)

    all_metrics, summary = summarize(args.result_root)
    all_metrics.to_csv(args.result_root / "all_metrics.csv", index=False)
    summary.to_csv(args.result_root / "summary_by_method.csv", index=False)
    summary.sort_values("roc_auc_mean", ascending=False).to_csv(args.result_root / "ranking_by_roc_auc.csv", index=False)
    summary.sort_values("macro_f1_mean", ascending=False).to_csv(args.result_root / "ranking_by_macro_f1.csv", index=False)
    summary.sort_values("accuracy_mean", ascending=False).to_csv(args.result_root / "ranking_by_accuracy.csv", index=False)
    write_audit(args.result_root, args, x, y, len(all_metrics))
    print(f"Finished. Results written to {args.result_root}", flush=True)


if __name__ == "__main__":
    main()
