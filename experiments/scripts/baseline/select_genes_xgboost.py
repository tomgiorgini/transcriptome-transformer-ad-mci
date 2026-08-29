#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

try:
    from xgboost import XGBClassifier
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing dependency: xgboost. Install it in this Python environment with:\n"
        "  python -m pip install xgboost\n"
        "Then rerun this script."
    ) from exc

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_X_FILE = ROOT / "task_dataset" / "processed" / "ad_mci_binary" / "X_ad_mci.csv"
DEFAULT_Y_FILE = ROOT / "task_dataset" / "processed" / "ad_mci_binary" / "y_ad_mci.csv"
DEFAULT_SPLIT_FILE = ROOT / "task_dataset" / "processed" / "ad_mci_binary" / "splits" / "official_seed42.csv"
DEFAULT_OUTPUT_DIR = ROOT / "task_dataset" / "processed" / "ad_mci_xgboost_top_genes"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select AD/MCI genes from the full gene matrix using train-only XGBoost feature importance."
    )
    parser.add_argument("--x-file", type=Path, default=DEFAULT_X_FILE)
    parser.add_argument("--y-file", type=Path, default=DEFAULT_Y_FILE)
    parser.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--top-k",
        type=int,
        default=0,
        help="Fixed number of genes to keep. Use 0 with --selection-mode auto to choose K from CV metrics.",
    )
    parser.add_argument("--selection-mode", choices=["auto", "fixed"], default="auto")
    parser.add_argument(
        "--candidate-k",
        type=str,
        default="25,50,100,200,300,500,788,1000,1500,2000,3000,5000",
        help="Comma-separated candidate panel sizes used when --selection-mode auto.",
    )
    parser.add_argument(
        "--auto-select-by",
        choices=["cv_roc_auc", "cv_macro_f1", "cv_balanced_accuracy", "importance_elbow", "cumulative_importance"],
        default="cv_roc_auc",
        help="Criterion used to choose K automatically.",
    )
    parser.add_argument(
        "--selection-tolerance",
        choices=["one_se", "none"],
        default="one_se",
        help="With one_se, choose the smallest K within one standard error of the best CV score.",
    )
    parser.add_argument(
        "--cumulative-importance-threshold",
        type=float,
        default=0.95,
        help="Importance mass threshold used when --auto-select-by cumulative_importance.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--importance-type", choices=["gain", "weight", "cover", "total_gain", "total_cover"], default="gain")
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample-bytree", type=float, default=0.5)
    parser.add_argument("--reg-alpha", type=float, default=0.0)
    parser.add_argument("--reg-lambda", type=float, default=1.0)
    parser.add_argument("--early-stopping-rounds", type=int, default=30)
    parser.add_argument("--n-jobs", type=int, default=1)
    return parser.parse_args()


def load_binary_dataset(x_file: Path, y_file: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[str]]:
    x_df = pd.read_csv(x_file)
    y_df = pd.read_csv(y_file)
    if "sample_id" not in x_df.columns:
        raise ValueError(f"{x_file} must contain sample_id.")
    if "sample_id" not in y_df.columns:
        raise ValueError(f"{y_file} must contain sample_id.")
    label_col = "label" if "label" in y_df.columns else "label_name"
    if label_col not in y_df.columns:
        raise ValueError(f"{y_file} must contain label or label_name.")

    x_df["sample_id"] = x_df["sample_id"].astype(str).str.strip()
    y_df["sample_id"] = y_df["sample_id"].astype(str).str.strip()
    merged = x_df.merge(y_df[["sample_id", label_col]], on="sample_id", how="inner", validate="one_to_one")
    if merged.empty:
        raise ValueError("X and y have no overlapping sample_id values.")

    gene_names = [column for column in merged.columns if column not in {"sample_id", label_col}]
    x = merged[gene_names].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    labels = merged[label_col].astype(str)
    unique_labels = sorted(labels.unique().tolist())
    if len(unique_labels) != 2:
        raise ValueError(f"Expected binary labels, found {len(unique_labels)}: {unique_labels}")
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    y = labels.map(label_to_idx).to_numpy(dtype=np.int64)
    sample_ids = merged["sample_id"].to_numpy()
    return sample_ids, x, y, gene_names, unique_labels


def parse_candidate_k(value: str, max_genes: int) -> list[int]:
    candidates: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        k = int(part)
        if 1 <= k <= max_genes:
            candidates.add(k)
    if not candidates:
        raise ValueError("No valid --candidate-k values after parsing.")
    return sorted(candidates)


def load_split_file(path: Path, sample_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    split_df = pd.read_csv(path)
    if "sample_id" not in split_df.columns or "split" not in split_df.columns:
        raise ValueError(f"{path} must contain sample_id and split columns.")
    split_df["sample_id"] = split_df["sample_id"].astype(str).str.strip()
    split_df["split"] = split_df["split"].astype(str).str.lower().str.strip()
    sample_to_idx = {sample_id: idx for idx, sample_id in enumerate(sample_ids.astype(str))}
    train_idx = [sample_to_idx[sample_id] for sample_id in split_df.loc[split_df["split"] == "train", "sample_id"] if sample_id in sample_to_idx]
    non_train_idx = [
        sample_to_idx[sample_id]
        for sample_id in split_df.loc[split_df["split"].isin(["val", "test"]), "sample_id"]
        if sample_id in sample_to_idx
    ]
    if not train_idx:
        raise ValueError(f"No train samples found in split file: {path}")
    return np.asarray(train_idx, dtype=np.int64), np.asarray(non_train_idx, dtype=np.int64)


def make_model(args: argparse.Namespace, scale_pos_weight: float) -> XGBClassifier:
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        reg_alpha=args.reg_alpha,
        reg_lambda=args.reg_lambda,
        scale_pos_weight=scale_pos_weight,
        random_state=args.seed,
        n_jobs=args.n_jobs,
    )


def class_weight_ratio(y: np.ndarray) -> float:
    positives = float((y == 1).sum())
    negatives = float((y == 0).sum())
    if positives == 0:
        return 1.0
    return negatives / positives


def fit_model(
    args: argparse.Namespace,
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    seed_offset: int,
) -> XGBClassifier:
    model_args = argparse.Namespace(**vars(args))
    model_args.seed = int(args.seed + seed_offset)
    model = make_model(model_args, class_weight_ratio(train_y))
    try:
        model.fit(
            train_x,
            train_y,
            eval_set=[(val_x, val_y)],
            verbose=False,
            early_stopping_rounds=args.early_stopping_rounds,
        )
    except TypeError:
        model.fit(train_x, train_y, eval_set=[(val_x, val_y)], verbose=False)
    return model


def booster_importance(model: XGBClassifier, gene_names: list[str], importance_type: str) -> np.ndarray:
    score = model.get_booster().get_score(importance_type=importance_type)
    values = np.zeros(len(gene_names), dtype=np.float64)
    for key, value in score.items():
        if not key.startswith("f"):
            continue
        idx = int(key[1:])
        if 0 <= idx < len(values):
            values[idx] = float(value)
    return values


def evaluate_model(model: XGBClassifier, x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    probability = model.predict_proba(x)[:, 1]
    prediction = (probability >= 0.5).astype(np.int64)
    try:
        auc = float(roc_auc_score(y, probability))
    except ValueError:
        auc = float("nan")
    return {
        "accuracy": float(accuracy_score(y, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "macro_f1": float(f1_score(y, prediction, average="macro")),
        "roc_auc": auc,
    }


def select_k_by_importance_elbow(ranking: pd.DataFrame) -> int:
    values = ranking["mean_importance"].to_numpy(dtype=np.float64)
    positive = values[values > 0]
    if positive.size <= 2:
        return int(max(positive.size, 1))

    y = positive / max(float(positive.max()), 1e-12)
    x = np.linspace(0.0, 1.0, num=len(y))
    start = np.array([x[0], y[0]])
    end = np.array([x[-1], y[-1]])
    line = end - start
    denom = np.linalg.norm(line)
    if denom == 0:
        return int(len(positive))
    points = np.column_stack([x, y])
    distances = np.abs(np.cross(line, start - points)) / denom
    return int(np.argmax(distances) + 1)


def select_k_by_cumulative_importance(ranking: pd.DataFrame, threshold: float) -> int:
    if not 0.0 < threshold <= 1.0:
        raise ValueError("--cumulative-importance-threshold must be in (0, 1].")
    values = ranking["mean_importance"].clip(lower=0.0).to_numpy(dtype=np.float64)
    total = float(values.sum())
    if total <= 0:
        return 1
    cumulative = np.cumsum(values) / total
    return int(np.searchsorted(cumulative, threshold, side="left") + 1)


def evaluate_candidate_k_cv(
    args: argparse.Namespace,
    x: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    gene_names: list[str],
    ranking: pd.DataFrame,
    candidate_k: list[int],
) -> pd.DataFrame:
    train_x = x[train_idx]
    train_y = y[train_idx]
    gene_to_idx = {gene: idx for idx, gene in enumerate(gene_names)}
    ranked_indices = np.asarray([gene_to_idx[gene] for gene in ranking["gene"].tolist()], dtype=np.int64)
    splitter = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed + 10_000)
    rows: list[dict[str, Any]] = []

    for k in candidate_k:
        selected_idx = ranked_indices[:k]
        for fold, (inner_train, inner_val) in enumerate(splitter.split(train_x, train_y), start=1):
            model = fit_model(
                args,
                train_x[inner_train][:, selected_idx],
                train_y[inner_train],
                train_x[inner_val][:, selected_idx],
                train_y[inner_val],
                seed_offset=20_000 + fold + k,
            )
            metrics = evaluate_model(model, train_x[inner_val][:, selected_idx], train_y[inner_val])
            rows.append({"candidate_k": k, "fold": fold, **metrics})

    fold_df = pd.DataFrame(rows)
    summary = (
        fold_df.groupby("candidate_k", as_index=False)
        .agg(
            folds=("fold", "nunique"),
            cv_roc_auc_mean=("roc_auc", "mean"),
            cv_roc_auc_std=("roc_auc", "std"),
            cv_macro_f1_mean=("macro_f1", "mean"),
            cv_macro_f1_std=("macro_f1", "std"),
            cv_balanced_accuracy_mean=("balanced_accuracy", "mean"),
            cv_balanced_accuracy_std=("balanced_accuracy", "std"),
        )
        .sort_values("candidate_k")
    )
    return summary


def choose_k_from_candidate_metrics(metrics_df: pd.DataFrame, criterion: str, tolerance: str) -> tuple[int, dict[str, Any]]:
    mean_col = f"{criterion}_mean"
    std_col = f"{criterion}_std"
    if mean_col not in metrics_df.columns:
        raise ValueError(f"Candidate metrics do not contain {mean_col}.")
    best_idx = metrics_df[mean_col].idxmax()
    best_row = metrics_df.loc[best_idx]
    threshold = float(best_row[mean_col])
    if tolerance == "one_se":
        folds = max(float(best_row.get("folds", 1.0)), 1.0)
        std = float(best_row.get(std_col, 0.0))
        if np.isfinite(std):
            threshold -= std / np.sqrt(folds)
    eligible = metrics_df.loc[metrics_df[mean_col] >= threshold].sort_values("candidate_k")
    chosen = eligible.iloc[0]
    details = {
        "criterion": criterion,
        "selection_tolerance": tolerance,
        "best_candidate_k": int(best_row["candidate_k"]),
        "best_score": float(best_row[mean_col]),
        "chosen_threshold": threshold,
        "chosen_score": float(chosen[mean_col]),
    }
    return int(chosen["candidate_k"]), details


def rank_genes_cv(args: argparse.Namespace, x: np.ndarray, y: np.ndarray, gene_names: list[str], train_idx: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_x = x[train_idx]
    train_y = y[train_idx]
    splitter = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    importances: list[np.ndarray] = []
    fold_rows: list[dict[str, Any]] = []

    for fold, (inner_train, inner_val) in enumerate(splitter.split(train_x, train_y), start=1):
        model = fit_model(
            args,
            train_x[inner_train],
            train_y[inner_train],
            train_x[inner_val],
            train_y[inner_val],
            seed_offset=fold,
        )
        importance = booster_importance(model, gene_names, args.importance_type)
        importances.append(importance)
        metrics = evaluate_model(model, train_x[inner_val], train_y[inner_val])
        fold_rows.append(
            {
                "fold": fold,
                "selected_nonzero_importance": int((importance > 0).sum()),
                "best_iteration": int(getattr(model, "best_iteration", -1) or -1),
                **metrics,
            }
        )

    stacked = np.vstack(importances)
    mean_importance = stacked.mean(axis=0)
    std_importance = stacked.std(axis=0, ddof=1) if stacked.shape[0] > 1 else np.zeros_like(mean_importance)
    nonzero_folds = (stacked > 0).sum(axis=0)
    rank_order = np.lexsort((-nonzero_folds, -mean_importance))

    ranking = pd.DataFrame(
        {
            "rank": np.arange(1, len(gene_names) + 1),
            "gene": [gene_names[idx] for idx in rank_order],
            "mean_importance": mean_importance[rank_order],
            "std_importance": std_importance[rank_order],
            "nonzero_folds": nonzero_folds[rank_order],
        }
    )
    return ranking, pd.DataFrame(fold_rows)


def write_subset(
    args: argparse.Namespace,
    x_file: Path,
    y_file: Path,
    split_file: Path,
    selected_genes: list[str],
) -> None:
    x_df = pd.read_csv(x_file)
    y_df = pd.read_csv(y_file)
    split_df = pd.read_csv(split_file)
    missing = [gene for gene in selected_genes if gene not in x_df.columns]
    if missing:
        preview = ", ".join(missing[:10])
        raise ValueError(f"Selected genes missing from X file: {preview}")
    subset_x = x_df[["sample_id", *selected_genes]]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    split_dir = args.output_dir / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    subset_x.to_csv(args.output_dir / f"X_xgboost_top{len(selected_genes)}_ad_mci.csv", index=False)
    y_df.to_csv(args.output_dir / "y_ad_mci.csv", index=False)
    split_df.to_csv(split_dir / split_file.name, index=False)
    pd.DataFrame({"gene": selected_genes, "rank": range(1, len(selected_genes) + 1)}).to_csv(
        args.output_dir / f"selected_genes_top{len(selected_genes)}.csv",
        index=False,
    )


def main() -> None:
    args = parse_args()
    sample_ids, x, y, gene_names, class_names = load_binary_dataset(args.x_file, args.y_file)
    train_idx, non_train_idx = load_split_file(args.split_file, sample_ids)
    if args.selection_mode == "fixed" and (args.top_k <= 0 or args.top_k > len(gene_names)):
        raise ValueError(f"With --selection-mode fixed, --top-k must be between 1 and {len(gene_names)}.")

    ranking, fold_metrics = rank_genes_cv(args, x, y, gene_names, train_idx)

    candidate_metrics = pd.DataFrame()
    k_selection_report: dict[str, Any]
    if args.selection_mode == "fixed":
        selected_k = int(args.top_k)
        k_selection_report = {"selection_mode": "fixed", "selected_k": selected_k}
    elif args.auto_select_by == "importance_elbow":
        selected_k = select_k_by_importance_elbow(ranking)
        k_selection_report = {
            "selection_mode": "auto",
            "auto_select_by": args.auto_select_by,
            "selected_k": selected_k,
        }
    elif args.auto_select_by == "cumulative_importance":
        selected_k = select_k_by_cumulative_importance(ranking, args.cumulative_importance_threshold)
        k_selection_report = {
            "selection_mode": "auto",
            "auto_select_by": args.auto_select_by,
            "cumulative_importance_threshold": args.cumulative_importance_threshold,
            "selected_k": selected_k,
        }
    else:
        candidate_k = parse_candidate_k(args.candidate_k, len(gene_names))
        candidate_metrics = evaluate_candidate_k_cv(args, x, y, train_idx, gene_names, ranking, candidate_k)
        selected_k, details = choose_k_from_candidate_metrics(candidate_metrics, args.auto_select_by, args.selection_tolerance)
        k_selection_report = {
            "selection_mode": "auto",
            "auto_select_by": args.auto_select_by,
            "candidate_k": candidate_k,
            "selected_k": selected_k,
            **details,
        }

    selected_k = min(max(int(selected_k), 1), len(gene_names))
    selected_genes = ranking.head(selected_k)["gene"].tolist()

    final_train_idx, final_val_idx = train_test_split(
        train_idx,
        test_size=0.2,
        stratify=y[train_idx],
        random_state=args.seed,
    )
    final_model = fit_model(args, x[final_train_idx], y[final_train_idx], x[final_val_idx], y[final_val_idx], seed_offset=999)
    final_metrics = {
        "train_holdout_val": evaluate_model(final_model, x[final_val_idx], y[final_val_idx]),
    }
    if len(non_train_idx) > 0:
        final_metrics["official_non_train"] = evaluate_model(final_model, x[non_train_idx], y[non_train_idx])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(args.output_dir / "xgboost_gene_ranking.csv", index=False)
    fold_metrics.to_csv(args.output_dir / "xgboost_cv_metrics.csv", index=False)
    if not candidate_metrics.empty:
        candidate_metrics.to_csv(args.output_dir / "xgboost_candidate_k_metrics.csv", index=False)
    write_subset(args, args.x_file, args.y_file, args.split_file, selected_genes)

    report = {
        "x_file": str(args.x_file),
        "y_file": str(args.y_file),
        "split_file": str(args.split_file),
        "output_dir": str(args.output_dir),
        "classes": class_names,
        "samples": int(x.shape[0]),
        "candidate_genes": len(gene_names),
        "requested_top_k": args.top_k,
        "k_selection": k_selection_report,
        "selected_genes": len(selected_genes),
        "train_samples_for_selection": int(len(train_idx)),
        "importance_type": args.importance_type,
        "model_params": {
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "learning_rate": args.learning_rate,
            "subsample": args.subsample,
            "colsample_bytree": args.colsample_bytree,
            "reg_alpha": args.reg_alpha,
            "reg_lambda": args.reg_lambda,
            "early_stopping_rounds": args.early_stopping_rounds,
        },
        "final_model_metrics": final_metrics,
    }
    with (args.output_dir / "xgboost_selection_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(f"Candidate genes: {len(gene_names)}")
    print(f"Selected genes: {len(selected_genes)}")
    print(f"Selection: {k_selection_report}")
    print(f"Output directory: {args.output_dir}")
    print(f"Top genes: {', '.join(selected_genes[:20])}")


if __name__ == "__main__":
    main()
