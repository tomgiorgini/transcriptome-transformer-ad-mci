#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
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
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover - optional runtime dependency
    XGBClassifier = None


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_X = ROOT / "task_dataset" / "processed" / "ad_mci_binary" / "X_ad_mci.csv"
DEFAULT_Y = ROOT / "task_dataset" / "processed" / "ad_mci_binary" / "y_ad_mci.csv"
DEFAULT_RESULT_ROOT = ROOT / "results" / "SOTA" / "simple-baselines-top1000"
DEFAULT_GSE63060_METADATA = ROOT / "pretraining_dataset" / "geo_downloads" / "GSE63060" / "eset_1_GPL6947" / "GSE63060_sample_metadata.tsv.gz"
DEFAULT_GSE63061_METADATA = ROOT / "pretraining_dataset" / "geo_downloads" / "GSE63061" / "eset_1_GPL10558" / "GSE63061_sample_metadata.tsv.gz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Leakage-free simple baselines on top variance genes.")
    parser.add_argument("--x-file", type=Path, default=DEFAULT_X)
    parser.add_argument("--y-file", type=Path, default=DEFAULT_Y)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--gse63060-metadata", type=Path, default=DEFAULT_GSE63060_METADATA)
    parser.add_argument("--gse63061-metadata", type=Path, default=DEFAULT_GSE63061_METADATA)
    parser.add_argument("--batch-scenarios", nargs="+", default=["shared_test", "test_gse63060", "test_gse63061"], choices=["shared_test", "test_gse63060", "test_gse63061"])
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--inner-val-ratio", type=float, default=0.15)
    parser.add_argument("--shared-test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-variance-genes", type=int, default=1000)
    parser.add_argument("--scaler", choices=["minmax", "standard", "none"], default="minmax")
    parser.add_argument("--models", nargs="+", default=["lr", "l1_lr", "svm_rbf", "rf", "xgboost", "gnb"], choices=["lr", "l1_lr", "svm_rbf", "rf", "xgboost", "gnb"])
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--overwrite-results", action="store_true")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else (ROOT / path).resolve()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def load_binary_dataset(x_file: Path, y_file: Path) -> tuple[pd.DataFrame, pd.Series]:
    x = pd.read_csv(x_file, index_col=0)
    y_df = pd.read_csv(y_file, index_col=0)
    if "label" not in y_df.columns:
        raise ValueError(f"{y_file} must contain a label column.")
    common = x.index.intersection(y_df.index)
    x = x.loc[common].apply(pd.to_numeric, errors="coerce")
    y = y_df.loc[common, "label"].astype(int)
    if sorted(y.unique().tolist()) != [0, 1]:
        raise ValueError(f"Expected binary labels 0/1, got {sorted(y.unique().tolist())}.")
    return x, y


def read_geo_metadata_sample_ids(path: Path) -> set[str]:
    md = pd.read_csv(path, sep="\t", compression="infer", dtype=str)
    for candidate in ("sample_id", "geo_accession"):
        if candidate in md.columns:
            return set(md[candidate].astype(str).str.strip())
    raise ValueError(f"{path} must contain sample_id or geo_accession.")


def infer_batch_labels(sample_ids: pd.Index, gse63060_metadata: Path, gse63061_metadata: Path) -> pd.Series:
    ids_63060 = read_geo_metadata_sample_ids(gse63060_metadata)
    ids_63061 = read_geo_metadata_sample_ids(gse63061_metadata)
    labels: dict[str, str] = {}
    for sample_id in sample_ids.astype(str):
        gsm_id = sample_id.split("_", 1)[0]
        if gsm_id in ids_63060:
            labels[sample_id] = "GSE63060"
        elif gsm_id in ids_63061:
            labels[sample_id] = "GSE63061"
        else:
            labels[sample_id] = "unknown"
    batch = pd.Series(labels, index=sample_ids, name="batch")
    if (batch == "unknown").any():
        raise ValueError(f"Could not map samples to batches: {batch[batch == 'unknown'].index[:5].tolist()}")
    return batch


def make_splits(y: pd.Series, batch: pd.Series, args: argparse.Namespace) -> list[dict[str, Any]]:
    y_values = y.to_numpy(dtype=np.int64)
    batch_values = batch.to_numpy()
    all_idx = np.arange(len(y_values))
    splits: list[dict[str, Any]] = []
    for repeat in range(1, args.repeats + 1):
        repeat_seed = args.seed + repeat - 1
        for scenario_idx, scenario in enumerate(args.batch_scenarios, start=1):
            if scenario == "test_gse63060":
                pool_idx = all_idx[batch_values == "GSE63061"]
                test_idx = all_idx[batch_values == "GSE63060"]
            elif scenario == "test_gse63061":
                pool_idx = all_idx[batch_values == "GSE63060"]
                test_idx = all_idx[batch_values == "GSE63061"]
            else:
                train_parts: list[np.ndarray] = []
                test_parts: list[np.ndarray] = []
                for batch_name in ("GSE63060", "GSE63061"):
                    batch_idx = all_idx[batch_values == batch_name]
                    train_part, test_part = train_test_split(batch_idx, test_size=args.shared_test_size, random_state=repeat_seed, stratify=y_values[batch_idx])
                    train_parts.append(np.asarray(train_part, dtype=np.int64))
                    test_parts.append(np.asarray(test_part, dtype=np.int64))
                pool_idx = np.concatenate(train_parts)
                test_idx = np.concatenate(test_parts)
            train_idx, val_idx = train_test_split(np.asarray(pool_idx, dtype=np.int64), test_size=args.inner_val_ratio, random_state=repeat_seed * 100 + scenario_idx, stratify=y_values[pool_idx])
            splits.append({"scenario": scenario, "repeat": repeat, "fold": 1, "seed": repeat_seed * 1000 + scenario_idx, "train_idx": train_idx, "val_idx": val_idx, "test_idx": test_idx})
    return splits


def select_top_variance_genes(x_train_raw: pd.DataFrame, k: int) -> list[str]:
    variances = x_train_raw.var(axis=0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return variances.sort_values(ascending=False).head(min(k, x_train_raw.shape[1])).index.astype(str).tolist()


def preprocess_selected(x: pd.DataFrame, train_idx: np.ndarray, val_idx: np.ndarray, test_idx: np.ndarray, genes: list[str], scaler_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_train_raw = x.iloc[train_idx].loc[:, genes]
    x_val_raw = x.iloc[val_idx].loc[:, genes]
    x_test_raw = x.iloc[test_idx].loc[:, genes]
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
    return (
        np.nan_to_num(x_train.astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0),
        np.nan_to_num(x_val.astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0),
        np.nan_to_num(x_test.astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0),
    )


def build_model(name: str, seed: int) -> tuple[Any, dict[str, Any]]:
    if name == "lr":
        return LogisticRegression(max_iter=5000, solver="liblinear", class_weight="balanced", random_state=seed), {"model": name, "class_weight": "balanced"}
    if name == "l1_lr":
        return LogisticRegression(max_iter=5000, solver="liblinear", penalty="l1", C=1.0, class_weight="balanced", random_state=seed), {"model": name, "penalty": "l1", "C": 1.0, "class_weight": "balanced"}
    if name == "svm_rbf":
        return SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, class_weight="balanced", random_state=seed), {"model": name, "kernel": "rbf", "C": 1.0, "gamma": "scale", "class_weight": "balanced"}
    if name == "rf":
        return RandomForestClassifier(n_estimators=500, max_features="sqrt", class_weight="balanced", n_jobs=-1, random_state=seed), {"model": name, "n_estimators": 500, "max_features": "sqrt", "class_weight": "balanced"}
    if name == "xgboost":
        if XGBClassifier is None:
            raise RuntimeError("xgboost is not installed.")
        return XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, eval_metric="logloss", random_state=seed, n_jobs=-1), {"model": name, "n_estimators": 300, "max_depth": 3, "learning_rate": 0.03}
    if name == "gnb":
        return GaussianNB(), {"model": name}
    raise ValueError(f"Unsupported model: {name}")


def positive_scores(model: Any, x_test: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x_test)[:, 1]
    if hasattr(model, "decision_function"):
        raw = model.decision_function(x_test)
        return 1.0 / (1.0 + np.exp(-raw))
    return model.predict(x_test).astype(float)


def evaluate(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> tuple[dict[str, float], np.ndarray]:
    y_pred = (scores >= threshold).astype(int)
    metrics = {
        "pr_auc": average_precision_score(y_true, scores),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "roc_auc": roc_auc_score(y_true, scores),
        "log_loss": log_loss(y_true, np.clip(scores, 1e-6, 1 - 1e-6), labels=[0, 1]),
    }
    return metrics, y_pred


def summarize(result_root: Path) -> None:
    metric_files = list((result_root / "runs").glob("*/*/*/metrics.csv"))
    if not metric_files:
        return
    all_metrics = pd.concat([pd.read_csv(path) for path in metric_files], ignore_index=True)
    all_metrics.to_csv(result_root / "all_metrics.csv", index=False)
    metric_cols = ["accuracy", "macro_f1", "roc_auc", "pr_auc", "balanced_accuracy", "precision", "recall"]
    summary = all_metrics.groupby(["scenario", "model"])[metric_cols].agg(["mean", "std"]).reset_index()
    summary.to_csv(result_root / "summary_by_method.csv", index=False)
    for metric in ("accuracy", "macro_f1", "roc_auc"):
        ranking = all_metrics.groupby(["scenario", "model"])[metric].mean().reset_index().sort_values(["scenario", metric], ascending=[True, False])
        ranking.to_csv(result_root / f"ranking_by_{metric}.csv", index=False)


def main() -> None:
    args = parse_args()
    args.x_file = resolve(args.x_file)
    args.y_file = resolve(args.y_file)
    args.result_root = resolve(args.result_root)
    args.gse63060_metadata = resolve(args.gse63060_metadata)
    args.gse63061_metadata = resolve(args.gse63061_metadata)
    if args.overwrite_results and args.result_root.exists():
        shutil.rmtree(args.result_root)
    args.result_root.mkdir(parents=True, exist_ok=True)
    write_json(args.result_root / "args.json", vars(args))

    x, y = load_binary_dataset(args.x_file, args.y_file)
    batch = infer_batch_labels(x.index, args.gse63060_metadata, args.gse63061_metadata)
    splits = make_splits(y, batch, args)
    y_values = y.to_numpy(dtype=np.int64)

    for split in splits:
        train_idx = split["train_idx"]
        val_idx = split["val_idx"]
        test_idx = split["test_idx"]
        genes = select_top_variance_genes(x.iloc[train_idx], args.top_variance_genes)
        x_train, _x_val, x_test = preprocess_selected(x, train_idx, val_idx, test_idx, genes, args.scaler)
        y_train = y_values[train_idx]
        y_test = y_values[test_idx]
        for model_name in args.models:
            run_dir = args.result_root / "runs" / split["scenario"] / f"repeat_{split['repeat']:02d}" / model_name
            if args.skip_existing and (run_dir / "metrics.csv").exists():
                continue
            print(f"Running scenario={split['scenario']} repeat={split['repeat']} model={model_name} genes={len(genes)}", flush=True)
            model, hyperparameters = build_model(model_name, split["seed"])
            model.fit(x_train, y_train)
            scores = positive_scores(model, x_test)
            metrics, y_pred = evaluate(y_test, scores, args.threshold)
            row = {
                "protocol": "batch_holdout",
                "scenario": split["scenario"],
                "repeat": split["repeat"],
                "fold": split["fold"],
                "seed": split["seed"],
                "feature_set": f"top{len(genes)}_variance",
                "model": model_name,
                "n_train_inner": len(train_idx),
                "n_val_inner": len(val_idx),
                "n_outer_test": len(test_idx),
                "n_genes": len(genes),
                "train_batches": "|".join(sorted(batch.iloc[train_idx].unique().tolist())),
                "val_batches": "|".join(sorted(batch.iloc[val_idx].unique().tolist())),
                "test_batches": "|".join(sorted(batch.iloc[test_idx].unique().tolist())),
                **metrics,
            }
            run_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame([row]).to_csv(run_dir / "metrics.csv", index=False)
            pd.DataFrame({"sample_id": x.index[test_idx].astype(str), "y_true": y_test, "y_pred": y_pred, "score_ad": scores}).to_csv(run_dir / "predictions.csv", index=False)
            pd.DataFrame(confusion_matrix(y_test, y_pred, labels=[0, 1]), index=["true_0", "true_1"], columns=["pred_0", "pred_1"]).to_csv(run_dir / "confusion_matrix.csv")
            pd.DataFrame(classification_report(y_test, y_pred, labels=[0, 1], output_dict=True, zero_division=0)).T.to_csv(run_dir / "classification_report.csv")
            (run_dir / "selected_genes.txt").write_text("\n".join(genes) + "\n", encoding="utf-8")
            write_json(run_dir / "hyperparameters.json", hyperparameters)
            write_json(run_dir / "run_manifest.json", {"split": split, "gene_selection": {"method": "top_variance", "scope": "train_inner", "k": args.top_variance_genes}, "preprocessing": {"imputer": "median", "scaler": args.scaler, "scope": "train_inner"}, "threshold": args.threshold, "metrics": row})

    pd.DataFrame(
        [
            {"item": "split_overlap", "status": "pass", "details": "train_inner, val_inner, and outer_test are disjoint by construction."},
            {"item": "feature_scope", "status": "pass", "details": f"Top {args.top_variance_genes} variance genes computed only on train_inner."},
            {"item": "preprocessing_scope", "status": "pass", "details": "Median imputer and scaler fit only on train_inner."},
        ]
    ).to_csv(args.result_root / "leakage_audit.csv", index=False)
    summarize(args.result_root)
    print(f"Finished. Results written to {args.result_root}", flush=True)


if __name__ == "__main__":
    main()
