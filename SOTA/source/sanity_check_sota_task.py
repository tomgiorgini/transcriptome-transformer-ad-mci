#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_selection import f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_X = ROOT / "task_dataset" / "processed" / "ad_mci_binary" / "X_ad_mci.csv"
DEFAULT_Y = ROOT / "task_dataset" / "processed" / "ad_mci_binary" / "y_ad_mci.csv"
DEFAULT_GSE63060_METADATA = ROOT / "pretraining_dataset" / "geo_downloads" / "GSE63060" / "eset_1_GPL6947" / "GSE63060_sample_metadata.tsv.gz"
DEFAULT_GSE63061_METADATA = ROOT / "pretraining_dataset" / "geo_downloads" / "GSE63061" / "eset_1_GPL10558" / "GSE63061_sample_metadata.tsv.gz"
DEFAULT_OUTPUT = ROOT / "results" / "SOTA" / "audit" / "sanity_checks.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run lightweight sanity checks for the AD/MCI SOTA benchmark.")
    parser.add_argument("--x-file", type=Path, default=DEFAULT_X)
    parser.add_argument("--y-file", type=Path, default=DEFAULT_Y)
    parser.add_argument("--gse63060-metadata", type=Path, default=DEFAULT_GSE63060_METADATA)
    parser.add_argument("--gse63061-metadata", type=Path, default=DEFAULT_GSE63061_METADATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=1000)
    parser.add_argument("--shared-test-size", type=float, default=0.2)
    parser.add_argument("--scenarios", nargs="+", default=["shared_test", "test_gse63060", "test_gse63061"], choices=["shared_test", "test_gse63060", "test_gse63061"])
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else (ROOT / path).resolve()


def load_dataset(x_file: Path, y_file: Path) -> tuple[pd.DataFrame, pd.Series]:
    x = pd.read_csv(x_file, index_col=0)
    y_df = pd.read_csv(y_file, index_col=0)
    if "label" not in y_df.columns:
        raise ValueError(f"{y_file} must contain a label column.")
    common = x.index.intersection(y_df.index)
    return x.loc[common].copy(), y_df.loc[common, "label"].astype(int)


def read_geo_ids(path: Path) -> set[str]:
    md = pd.read_csv(path, sep="\t", compression="infer", dtype=str)
    for col in ("sample_id", "geo_accession"):
        if col in md.columns:
            return set(md[col].astype(str).str.strip())
    raise ValueError(f"{path} must contain sample_id or geo_accession.")


def infer_batch(sample_ids: pd.Index, gse63060_metadata: Path, gse63061_metadata: Path) -> pd.Series:
    ids60 = read_geo_ids(gse63060_metadata)
    ids61 = read_geo_ids(gse63061_metadata)
    labels: dict[str, str] = {}
    for sample_id in sample_ids.astype(str):
        gsm = sample_id.split("_", 1)[0]
        if gsm in ids60:
            labels[sample_id] = "GSE63060"
        elif gsm in ids61:
            labels[sample_id] = "GSE63061"
        else:
            labels[sample_id] = "unknown"
    batch = pd.Series(labels, index=sample_ids, name="batch")
    if (batch == "unknown").any():
        raise ValueError(f"Could not infer batch for {(batch == 'unknown').sum()} samples.")
    return batch


def make_splits(y: pd.Series, batch: pd.Series, scenarios: list[str], repeats: int, shared_test_size: float, seed: int) -> list[dict[str, Any]]:
    y_values = y.to_numpy(dtype=int)
    batch_values = batch.to_numpy()
    all_idx = np.arange(len(y_values))
    splits: list[dict[str, Any]] = []
    for repeat in range(1, repeats + 1):
        repeat_seed = seed + repeat - 1
        for scenario in scenarios:
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
                    train_part, test_part = train_test_split(
                        batch_idx,
                        test_size=shared_test_size,
                        random_state=repeat_seed,
                        stratify=y_values[batch_idx],
                    )
                    train_parts.append(np.asarray(train_part, dtype=int))
                    test_parts.append(np.asarray(test_part, dtype=int))
                pool_idx = np.concatenate(train_parts)
                test_idx = np.concatenate(test_parts)
            train_idx, val_idx = train_test_split(
                np.asarray(pool_idx, dtype=int),
                test_size=0.15,
                random_state=repeat_seed,
                stratify=y_values[pool_idx],
            )
            splits.append(
                {
                    "scenario": scenario,
                    "repeat": repeat,
                    "seed": repeat_seed,
                    "train_idx": np.asarray(train_idx, dtype=int),
                    "val_idx": np.asarray(val_idx, dtype=int),
                    "test_idx": np.asarray(test_idx, dtype=int),
                }
            )
    return splits


def evaluate(y_true: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    scores = np.asarray(scores, dtype=float)
    scores = np.nan_to_num(scores, nan=0.5, posinf=1.0, neginf=0.0)
    scores = np.clip(scores, 0.0, 1.0)
    pred = (scores >= 0.5).astype(int)
    try:
        roc = float(roc_auc_score(y_true, scores))
    except ValueError:
        roc = float("nan")
    try:
        pr = float(average_precision_score(y_true, scores))
    except ValueError:
        pr = float("nan")
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "roc_auc": roc,
        "pr_auc": pr,
        "pred_class0": int((pred == 0).sum()),
        "pred_class1": int((pred == 1).sum()),
    }


def select_top_variance(x_train: pd.DataFrame, top_k: int) -> list[str]:
    scores = x_train.var(axis=0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return scores.sort_values(ascending=False).head(min(top_k, x_train.shape[1])).index.astype(str).tolist()


def select_full_anova(x: pd.DataFrame, y: pd.Series, top_k: int) -> list[str]:
    imputed = SimpleImputer(strategy="median").fit_transform(x)
    scores, _ = f_classif(imputed, y.to_numpy(dtype=int))
    scores = pd.Series(np.nan_to_num(scores, nan=0.0), index=x.columns)
    return scores.sort_values(ascending=False).head(min(top_k, x.shape[1])).index.astype(str).tolist()


def fit_lr_scores(x_train: pd.DataFrame, y_train: np.ndarray, x_test: pd.DataFrame) -> np.ndarray:
    estimator = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=5000, solver="liblinear", class_weight="balanced", random_state=42),
    )
    estimator.fit(x_train, y_train)
    return estimator.predict_proba(x_test)[:, 1]


def batch_only_scores(train_batch: pd.Series, y_train: np.ndarray, test_batch: pd.Series) -> np.ndarray:
    if len(np.unique(y_train)) < 2:
        dummy = DummyClassifier(strategy="prior")
        dummy.fit(np.zeros((len(y_train), 1)), y_train)
        return dummy.predict_proba(np.zeros((len(test_batch), 1)))[:, 1]
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    x_train = encoder.fit_transform(train_batch.to_numpy().reshape(-1, 1))
    x_test = encoder.transform(test_batch.to_numpy().reshape(-1, 1))
    if np.all(x_train == x_train[0]):
        dummy = DummyClassifier(strategy="prior")
        dummy.fit(np.zeros((len(y_train), 1)), y_train)
        return dummy.predict_proba(np.zeros((len(test_batch), 1)))[:, 1]
    model = LogisticRegression(max_iter=1000, solver="liblinear", class_weight="balanced", random_state=42)
    model.fit(x_train, y_train)
    return model.predict_proba(x_test)[:, 1]


def main() -> None:
    args = parse_args()
    args.x_file = resolve(args.x_file)
    args.y_file = resolve(args.y_file)
    args.gse63060_metadata = resolve(args.gse63060_metadata)
    args.gse63061_metadata = resolve(args.gse63061_metadata)
    args.output = resolve(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    x, y = load_dataset(args.x_file, args.y_file)
    batch = infer_batch(x.index, args.gse63060_metadata, args.gse63061_metadata)
    splits = make_splits(y, batch, args.scenarios, args.repeats, args.shared_test_size, args.seed)
    full_anova_genes = select_full_anova(x, y, args.top_k)
    rows: list[dict[str, Any]] = []
    y_values = y.to_numpy(dtype=int)
    rng = np.random.default_rng(args.seed)
    for split in splits:
        train_idx = split["train_idx"]
        test_idx = split["test_idx"]
        y_train = y_values[train_idx]
        y_test = y_values[test_idx]
        x_train = x.iloc[train_idx]
        x_test = x.iloc[test_idx]
        top_var_genes = select_top_variance(x_train, args.top_k)
        checks = []
        checks.append(("train_only_top_variance_lr", top_var_genes, y_train, "train_only"))
        checks.append(("label_permutation_top_variance_lr", top_var_genes, rng.permutation(y_train), "negative_control"))
        checks.append(("leaky_full_dataset_anova_lr", full_anova_genes, y_train, "intentional_leakage_upper_bound"))
        for check_name, genes, labels_for_fit, scope in checks:
            scores = fit_lr_scores(x_train.loc[:, genes], labels_for_fit, x_test.loc[:, genes])
            rows.append(
                {
                    "check": check_name,
                    "scope": scope,
                    "scenario": split["scenario"],
                    "repeat": split["repeat"],
                    "n_train": len(train_idx),
                    "n_test": len(test_idx),
                    "n_genes": len(genes),
                    **evaluate(y_test, scores),
                }
            )
        batch_scores = batch_only_scores(batch.iloc[train_idx], y_train, batch.iloc[test_idx])
        rows.append(
            {
                "check": "batch_only_lr_or_prior",
                "scope": "batch_signal_control",
                "scenario": split["scenario"],
                "repeat": split["repeat"],
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "n_genes": 1,
                **evaluate(y_test, batch_scores),
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(args.output, index=False)
    summary = result.groupby(["check", "scope", "scenario"], dropna=False)[["accuracy", "macro_f1", "roc_auc", "pr_auc"]].agg(["mean", "std", "count"]).reset_index()
    summary.columns = ["_".join([str(part) for part in col if part]) for col in summary.columns.to_flat_index()]
    summary.to_csv(args.output.with_name("sanity_checks_summary.csv"), index=False)
    print(f"Sanity checks written to {args.output}", flush=True)


if __name__ == "__main__":
    main()
