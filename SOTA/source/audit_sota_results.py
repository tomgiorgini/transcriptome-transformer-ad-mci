#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULT_ROOT = ROOT / "results" / "SOTA"
DEFAULT_OUTPUT_DIR = DEFAULT_RESULT_ROOT / "audit"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit SOTA result artifacts and selected-gene stability.")
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--low-roc-threshold", type=float, default=0.55)
    parser.add_argument("--low-macro-f1-threshold", type=float, default=0.45)
    parser.add_argument("--low-gene-threshold", type=int, default=10)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else (ROOT / path).resolve()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"_json_error": f"Could not parse {path}"}


def first_present(row: pd.Series | dict[str, Any], names: list[str], default: Any = np.nan) -> Any:
    for name in names:
        if name in row and not pd.isna(row[name]):
            return row[name]
    return default


def infer_experiment(result_root: Path, metrics_path: Path) -> str:
    rel = metrics_path.relative_to(result_root)
    return rel.parts[0] if rel.parts else "unknown"


def normalize_feature_set(value: Any, metrics_path: Path) -> str:
    if value is not None and not pd.isna(value) and str(value):
        return str(value)
    parts = metrics_path.parts
    for idx, part in enumerate(parts):
        if part.startswith("repeat_") and idx + 1 < len(parts):
            return parts[idx + 1]
        if part.startswith("fold_") and idx + 1 < len(parts):
            return parts[idx + 1]
    return "unknown"


def normalize_model(value: Any, metrics_path: Path) -> str:
    if value is not None and not pd.isna(value) and str(value):
        return str(value)
    parent = metrics_path.parent.name
    return parent if parent else "unknown"


def read_selected_genes(run_dir: Path) -> list[str]:
    path = run_dir / "selected_genes.txt"
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def confusion_from_predictions(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "predictions.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if "y_true" not in df.columns or "y_pred" not in df.columns:
        return {}
    y_true = df["y_true"].astype(int).to_numpy()
    y_pred = df["y_pred"].astype(int).to_numpy()
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    return {
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "test_class0": int((y_true == 0).sum()),
        "test_class1": int((y_true == 1).sum()),
        "pred_class0": int((y_pred == 0).sum()),
        "pred_class1": int((y_pred == 1).sum()),
    }


def confusion_from_matrix(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "confusion_matrix.csv"
    if not path.exists():
        return {}
    try:
        df = pd.read_csv(path, index_col=0)
    except Exception:
        return {}
    values = df.to_numpy()
    if values.shape[0] < 2 or values.shape[1] < 2:
        return {}
    tn, fp, fn, tp = int(values[0, 0]), int(values[0, 1]), int(values[1, 0]), int(values[1, 1])
    return {
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "test_class0": tn + fp,
        "test_class1": fn + tp,
        "pred_class0": tn + fn,
        "pred_class1": fp + tp,
    }


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else float("nan")


def class_metrics_from_confusion(conf: dict[str, Any]) -> dict[str, float]:
    if not conf:
        return {}
    tn, fp, fn, tp = conf["tn"], conf["fp"], conf["fn"], conf["tp"]
    precision_0 = safe_div(tn, tn + fn)
    recall_0 = safe_div(tn, tn + fp)
    precision_1 = safe_div(tp, tp + fp)
    recall_1 = safe_div(tp, tp + fn)
    return {
        "precision_class0": precision_0,
        "recall_class0": recall_0,
        "precision_class1": precision_1,
        "recall_class1": recall_1,
        "specificity_class1": recall_0,
        "sensitivity_class1": recall_1,
    }


def manifest_text(manifest: dict[str, Any]) -> str:
    try:
        return json.dumps(manifest, default=str).lower()
    except TypeError:
        return str(manifest).lower()


def fallback_used(manifest: Any) -> bool:
    if isinstance(manifest, dict):
        for key, value in manifest.items():
            key_text = str(key).lower()
            if "fallback" in key_text and isinstance(value, bool) and value:
                return True
            if "fallback" in key_text and isinstance(value, (int, float)) and value > 0 and key_text not in {"fallback_top_k"}:
                return True
            if fallback_used(value):
                return True
        return False
    if isinstance(manifest, list):
        return any(fallback_used(value) for value in manifest)
    if isinstance(manifest, str):
        value = manifest.lower()
        return (
            "completed_fallback" in value
            or value.startswith("fallback_top_")
            or "fallback used" in value
            or "fallback_used=true" in value
        )
    return False


def audit_flags(row: dict[str, Any], args: argparse.Namespace, manifest: dict[str, Any]) -> str:
    flags: list[str] = []
    roc = row.get("roc_auc", np.nan)
    macro = row.get("macro_f1", np.nan)
    n_features = row.get("n_features", np.nan)
    pred0 = row.get("pred_class0", np.nan)
    pred1 = row.get("pred_class1", np.nan)
    if pd.isna(roc):
        flags.append("missing_roc_auc")
    elif float(roc) < args.low_roc_threshold:
        flags.append("low_roc_auc")
    if pd.isna(macro):
        flags.append("missing_macro_f1")
    elif float(macro) < args.low_macro_f1_threshold:
        flags.append("low_macro_f1")
    if not pd.isna(n_features) and int(n_features) < args.low_gene_threshold:
        flags.append("very_low_gene_count")
    if not pd.isna(pred0) and not pd.isna(pred1) and (int(pred0) == 0 or int(pred1) == 0):
        flags.append("single_class_predictions")
    text = manifest_text(manifest)
    if "full_dataset" in text or "intentional_leakage" in text:
        flags.append("leakage_marked")
    if fallback_used(manifest):
        flags.append("fallback_used")
    return "|".join(sorted(set(flags)))


def collect_runs(args: argparse.Namespace) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    result_root = resolve(args.result_root)
    for metrics_path in sorted(result_root.rglob("metrics.csv")):
        if "audit" in metrics_path.parts:
            continue
        try:
            metrics = pd.read_csv(metrics_path)
        except Exception:
            continue
        if metrics.empty:
            continue
        run_dir = metrics_path.parent
        manifest = read_json(run_dir / "run_manifest.json")
        selected_genes = read_selected_genes(run_dir)
        conf = confusion_from_predictions(run_dir) or confusion_from_matrix(run_dir)
        class_metrics = class_metrics_from_confusion(conf)
        for _, metric_row in metrics.iterrows():
            experiment = infer_experiment(result_root, metrics_path)
            feature_set = normalize_feature_set(metric_row.get("feature_set", None), metrics_path)
            model = normalize_model(metric_row.get("model", None), metrics_path)
            n_features = first_present(
                metric_row,
                ["n_features", "n_genes", "n_selected", "embedding_dim"],
                len(selected_genes) if selected_genes else np.nan,
            )
            row = {
                "experiment": experiment,
                "run_dir": str(run_dir),
                "metrics_path": str(metrics_path),
                "protocol": first_present(metric_row, ["protocol"], "unknown"),
                "scenario": first_present(metric_row, ["scenario"], "unknown"),
                "repeat": first_present(metric_row, ["repeat"], np.nan),
                "fold": first_present(metric_row, ["fold"], np.nan),
                "feature_set": feature_set,
                "model": model,
                "n_train_inner": first_present(metric_row, ["n_train_inner", "n_train"], np.nan),
                "n_val_inner": first_present(metric_row, ["n_val_inner", "n_validation", "n_val"], np.nan),
                "n_outer_test": first_present(metric_row, ["n_outer_test", "n_test"], np.nan),
                "n_features": n_features,
                "n_selected_genes_file": len(selected_genes),
                "accuracy": first_present(metric_row, ["accuracy"], np.nan),
                "macro_f1": first_present(metric_row, ["macro_f1"], np.nan),
                "weighted_f1": first_present(metric_row, ["weighted_f1"], np.nan),
                "roc_auc": first_present(metric_row, ["roc_auc", "roc_auc_ovr_macro"], np.nan),
                "pr_auc": first_present(metric_row, ["pr_auc", "average_precision"], np.nan),
                "balanced_accuracy": first_present(metric_row, ["balanced_accuracy"], np.nan),
                **conf,
                **class_metrics,
            }
            row["audit_flags"] = audit_flags(row, args, manifest)
            rows.append(row)
    return pd.DataFrame(rows)


def collect_skipped_runs(args: argparse.Namespace) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    result_root = resolve(args.result_root)
    for manifest_path in sorted(result_root.rglob("run_manifest.json")):
        if "audit" in manifest_path.parts or (manifest_path.parent / "metrics.csv").exists():
            continue
        manifest = read_json(manifest_path)
        status = str(manifest.get("status", ""))
        if not status.startswith("skipped"):
            nested = manifest.get("feature_set", {}) if isinstance(manifest.get("feature_set", {}), dict) else {}
            status = str(nested.get("status", status))
        if not status.startswith("skipped"):
            dgs = manifest.get("dgs", {}) if isinstance(manifest.get("dgs", {}), dict) else {}
            status = str(dgs.get("status", status))
        if not status.startswith("skipped"):
            continue
        rows.append(
            {
                "experiment": infer_experiment(result_root, manifest_path),
                "run_dir": str(manifest_path.parent),
                "status": status,
                "reason": manifest.get("reason", ""),
                "scenario": manifest.get("scenario", "unknown"),
                "repeat": manifest.get("repeat", np.nan),
                "feature_set": manifest.get("feature_set", {}).get("name", "") if isinstance(manifest.get("feature_set", {}), dict) else "",
                "model": manifest_path.parent.name,
                "n_selected_genes": manifest.get("n_selected_genes", manifest.get("dgs", {}).get("n_selected_genes", np.nan) if isinstance(manifest.get("dgs", {}), dict) else np.nan),
                "manifest_path": str(manifest_path),
            }
        )
    return pd.DataFrame(rows)


def summarize_runs(run_df: pd.DataFrame) -> pd.DataFrame:
    if run_df.empty:
        return pd.DataFrame()
    metric_cols = [
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "roc_auc",
        "pr_auc",
        "balanced_accuracy",
        "n_features",
        "n_outer_test",
        "recall_class0",
        "recall_class1",
        "precision_class0",
        "precision_class1",
    ]
    available = [col for col in metric_cols if col in run_df.columns]
    summary = (
        run_df.groupby(["experiment", "scenario", "feature_set", "model"], dropna=False)[available]
        .agg(["mean", "std", "min", "max", "count"])
        .reset_index()
    )
    summary.columns = ["_".join([str(part) for part in col if part]) for col in summary.columns.to_flat_index()]
    return summary


def best_tables(summary: pd.DataFrame, output_dir: Path) -> None:
    if summary.empty:
        return
    for metric in ("accuracy", "macro_f1", "roc_auc"):
        mean_col = f"{metric}_mean"
        if mean_col not in summary.columns:
            continue
        ranked = summary.sort_values(["experiment", "scenario", mean_col], ascending=[True, True, False])
        ranked.to_csv(output_dir / f"ranking_by_{metric}.csv", index=False)
        best = ranked.groupby(["experiment", "scenario"], dropna=False).head(1)
        best.to_csv(output_dir / f"best_by_experiment_scenario_{metric}.csv", index=False)


def selected_gene_stability(run_df: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    if run_df.empty:
        return pd.DataFrame()
    gene_sets: dict[tuple[Any, ...], dict[str, set[str]]] = {}
    for _, row in run_df.iterrows():
        run_dir = Path(str(row["run_dir"]))
        genes = set(read_selected_genes(run_dir))
        if not genes:
            continue
        split_key = f"repeat={row.get('repeat')}|fold={row.get('fold')}|run={run_dir.parent.name}"
        group_key = (row.get("experiment"), row.get("scenario"), row.get("feature_set"))
        gene_sets.setdefault(group_key, {})[split_key] = genes
    for (experiment, scenario, feature_set), split_sets in sorted(gene_sets.items()):
        sets = list(split_sets.values())
        counts = np.asarray([len(s) for s in sets], dtype=float)
        jaccards: list[float] = []
        for a, b in itertools.combinations(sets, 2):
            union = len(a | b)
            jaccards.append(float(len(a & b) / union) if union else float("nan"))
        finite_j = np.asarray([j for j in jaccards if not math.isnan(j)], dtype=float)
        records.append(
            {
                "experiment": experiment,
                "scenario": scenario,
                "feature_set": feature_set,
                "n_gene_sets": len(sets),
                "n_genes_mean": float(np.mean(counts)),
                "n_genes_std": float(np.std(counts, ddof=1)) if len(counts) > 1 else 0.0,
                "n_genes_min": int(np.min(counts)),
                "n_genes_max": int(np.max(counts)),
                "pairwise_jaccard_mean": float(np.mean(finite_j)) if finite_j.size else np.nan,
                "pairwise_jaccard_std": float(np.std(finite_j, ddof=1)) if finite_j.size > 1 else 0.0 if finite_j.size else np.nan,
                "pairwise_jaccard_min": float(np.min(finite_j)) if finite_j.size else np.nan,
                "pairwise_jaccard_max": float(np.max(finite_j)) if finite_j.size else np.nan,
            }
        )
    return pd.DataFrame(records)


def main() -> None:
    args = parse_args()
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_df = collect_runs(args)
    run_df.to_csv(output_dir / "run_level_audit.csv", index=False)
    skipped = collect_skipped_runs(args)
    skipped.to_csv(output_dir / "skipped_runs.csv", index=False)
    summary = summarize_runs(run_df)
    summary.to_csv(output_dir / "summary_by_experiment_scenario_method.csv", index=False)
    best_tables(summary, output_dir)
    stability = selected_gene_stability(run_df)
    stability.to_csv(output_dir / "selected_gene_stability.csv", index=False)
    flags = (
        run_df.assign(audit_flag=run_df["audit_flags"].fillna("").str.split("|"))
        .explode("audit_flag")
        .query("audit_flag != ''")
        .groupby(["experiment", "scenario", "audit_flag"], dropna=False)
        .size()
        .reset_index(name="count")
        if not run_df.empty
        else pd.DataFrame(columns=["experiment", "scenario", "audit_flag", "count"])
    )
    flags.to_csv(output_dir / "audit_flags_summary.csv", index=False)
    print(f"Audited {len(run_df)} runs. Outputs written to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
