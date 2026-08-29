#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = ROOT / "results" / "meeting_figures" / f"pretraining_finetuning_{datetime.now():%Y%m%d_%H%M%S}"


def ensure_out() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def savefig(name: str) -> None:
    path = OUT_DIR / name
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


def as_float(value) -> float:
    if pd.isna(value):
        return np.nan
    if isinstance(value, str):
        value = value.replace(",", ".")
    return float(value)


def parse_float_series(text: str, pattern: str) -> list[tuple[int, float, float]]:
    regex = re.compile(pattern)
    rows: list[tuple[int, float, float]] = []
    for line in text.splitlines():
        match = regex.search(line)
        if match:
            rows.append((int(match.group(1)), float(match.group(2)), float(match.group(3))))
    return rows


def read_text_auto(path: Path) -> str:
    raw = path.read_bytes()
    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        return raw.decode("utf-16", errors="ignore")
    return raw.decode("utf-8", errors="ignore")


def label_pretraining_run(path: Path) -> str:
    name = path.name
    arch = "4L/4H" if "larger_4layer" in name else "2L/2H" if "two_layer_2head" in name else "1L/2H" if "one_layer_2head" in name else "baseline"
    genes = "DEG-only" if "deg_only" in name or "deg_overlap" in name else "all genes"
    ref = "task dataset" if "with_reference" in name or "one_layer" in name else "no task dataset"
    return f"{arch} | {genes} | {ref}"


def label_arch(name: str) -> str:
    if "larger_4layer" in name:
        return "4L/4H"
    if "two_layer_2head" in name:
        return "2L/2H"
    if "one_layer_2head" in name:
        return "1L/2H"
    if "random_init" in name:
        return "random init"
    return name


def strip_reference_suffix(name: str) -> str:
    return name.replace("_with_reference", "").replace("_no_reference", "")


def collect_pretraining_logs() -> pd.DataFrame:
    rows = []
    for log_path in (ROOT / "results" / "pretraining" / "self_supervised" / "txt_gexbert").glob("*/pretraining.log"):
        run_dir = log_path.parent
        text = read_text_auto(log_path)
        loss_rows = parse_float_series(text, r"Epoch\s+(\d+)\s+\|\s+train_loss=([0-9.]+)\s+\|\s+val_loss=([0-9.]+)")
        if not loss_rows:
            continue
        summary_path = run_dir / "training_summary.json"
        summary = {}
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        for epoch, train_loss, val_loss in loss_rows:
            rows.append(
                {
                    "run": run_dir.name,
                    "label": label_pretraining_run(run_dir),
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "best_epoch": summary.get("best_epoch"),
                    "best_val_loss": summary.get("best_val_loss"),
                }
            )
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df[
            df["label"].str.contains("task dataset")
            & ~df["label"].str.contains("no task dataset")
        ].copy()
    if not df.empty:
        df.to_csv(OUT_DIR / "pretraining_curves.csv", index=False)
    return df


def plot_pretraining(df: pd.DataFrame) -> None:
    if df.empty:
        return
    final = df.sort_values("epoch").groupby("label", as_index=False).tail(1)
    order = (
        final.assign(is_current=final["label"].str.contains("task dataset"))
        .sort_values(["is_current", "val_loss"], ascending=[False, True])["label"]
        .tolist()
    )

    current_labels = [label for label in order if "task dataset" in label]
    if current_labels:
        plt.figure(figsize=(11, 6))
        for label in current_labels:
            part = df[df["label"] == label].sort_values("epoch")
            plt.plot(part["epoch"], part["val_loss"], label=label, linewidth=1.8)
        plt.xlabel("Epoch")
        plt.ylabel("Validation restoration loss")
        plt.title("Pretraining TxT: validation loss")
        plt.legend(fontsize=8)
        plt.grid(alpha=0.25)
        savefig("01_pretraining_validation_loss_with_task_dataset.png")

    plt.figure(figsize=(11, 6))
    for label in current_labels[:6]:
        part = df[df["label"] == label].sort_values("epoch")
        plt.plot(part["epoch"], part["train_loss"], linestyle="--", alpha=0.8, linewidth=1.2)
        plt.plot(part["epoch"], part["val_loss"], label=label, linewidth=1.7)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Pretraining TxT: train vs validation loss")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.25)
    savefig("02_pretraining_train_vs_validation_loss.png")

    summary = (
        df.groupby("label", as_index=False)
        .agg(best_val_loss=("val_loss", "min"), epochs=("epoch", "max"))
        .sort_values("best_val_loss")
    )
    summary.to_csv(OUT_DIR / "pretraining_summary.csv", index=False)
    plt.figure(figsize=(10, 5.5))
    labels = summary["label"].tolist()
    y = np.arange(len(labels))
    plt.barh(y, summary["best_val_loss"], color="#4C78A8")
    plt.yticks(y, labels, fontsize=8)
    plt.xlabel("Best validation loss")
    plt.title("Pretraining: best restoration loss")
    plt.gca().invert_yaxis()
    plt.grid(axis="x", alpha=0.25)
    savefig("03_pretraining_best_validation_loss.png")


def read_csv_optional(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def collect_standard_finetuning() -> pd.DataFrame:
    frames = []
    for path in (ROOT / "results" / "pretraining" / "finetuning").glob("*/final_tables/pretraining_finetune_summary.csv"):
        df = pd.read_csv(path)
        df["source_run"] = path.parents[1].name
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    if "reference_policy" in df.columns:
        df = df[df["reference_policy"].eq("with_reference")].copy()
    if "runs" in df.columns:
        df = df[pd.to_numeric(df["runs"], errors="coerce").fillna(0) >= 5].copy()
    if not df.empty and {"pretraining_arch", "transfer_mode", "test_macro_f1_mean"}.issubset(df.columns):
        df = (
            df.sort_values("test_macro_f1_mean", ascending=False)
            .groupby(["pretraining_arch", "transfer_mode"], as_index=False)
            .head(1)
            .copy()
        )
    df.to_csv(OUT_DIR / "standard_finetuning_summary_all.csv", index=False)
    return df


def collect_standard_finetuning_runs() -> pd.DataFrame:
    rows = []
    for path in (ROOT / "results" / "pretraining" / "finetuning").glob("*/**/metrics_summary.csv"):
        if "with_reference" not in str(path):
            continue
        parts = path.parts
        split_seed = path.parent.name.replace("split_seed_", "")
        transfer = path.parent.parent.name
        model = path.parent.parent.parent.name
        source_run = path.parents[3].name
        try:
            metrics = pd.read_csv(path)
        except Exception:
            continue
        test = metrics[metrics["split"].eq("test")]
        if test.empty:
            continue
        row = test.iloc[0].to_dict()
        rows.append(
            {
                "family": "standard",
                "source_run": source_run,
                "model": model,
                "arch": label_arch(model),
                "transfer_mode": transfer,
                "split_seed": split_seed,
                "label": f"standard {label_arch(model)} {transfer}",
                **row,
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        # Keep only 5-split configurations for robust IQR/boxplots.
        counts = df.groupby(["source_run", "model", "transfer_mode"])["split_seed"].transform("nunique")
        df = df[counts >= 5].copy()
        df.to_csv(OUT_DIR / "standard_finetuning_test_runs.csv", index=False)
    return df


def collect_finetuning_loss_curves() -> pd.DataFrame:
    rows = []

    def add_log(path: Path, family: str, source_run: str, model: str, transfer: str, split_seed: str) -> None:
        try:
            df = pd.read_csv(path)
        except Exception:
            return
        if "epoch" not in df.columns or "train_loss" not in df.columns:
            return
        for _, row in df.iterrows():
            rows.append(
                {
                    "family": family,
                    "source_run": source_run,
                    "model": model,
                    "arch": label_arch(model),
                    "transfer_mode": transfer,
                    "split_seed": split_seed,
                    "epoch": int(row["epoch"]),
                    "train_loss": as_float(row["train_loss"]),
                    "val_macro_f1": as_float(row["val_macro_f1"]) if "val_macro_f1" in row else np.nan,
                    "val_accuracy": as_float(row["val_accuracy"]) if "val_accuracy" in row else np.nan,
                }
            )

    for path in (ROOT / "results" / "pretraining" / "finetuning").glob("*/**/training_log.csv"):
        if "with_reference" not in str(path):
            continue
        source_run = path.parents[3].name
        model = path.parent.parent.parent.name
        transfer = path.parent.parent.name
        split_seed = path.parent.name.replace("split_seed_", "")
        add_log(path, "standard", source_run, model, transfer, split_seed)

    allowed_run_roots = {
        "stable_deg_overlap_2l2h_seed101_more_learning",
        "stable_deg_overlap_2l2h_random_init_5split_more_learning",
    }
    for path in (ROOT / "results" / "pretraining" / "finetuning_stable_deg").glob("*/**/training_log.csv"):
        run_root = next((part for part in path.parts if part.startswith("stable_deg_overlap_")), "")
        if run_root not in allowed_run_roots:
            continue
        model = path.parent.parent.parent.name
        transfer = path.parent.parent.name
        split_seed = path.parent.name.replace("split_seed_", "")
        add_log(path, "stable fixed", run_root, model, transfer, split_seed)

    # Representative 1L/2H tuned run from the first Optuna pass.
    tuned_root = ROOT / "results" / "pretraining" / "finetuning_stable_deg_optuna" / "one_layer_2head_deg_overlap_optuna_finetune_only"
    for path in tuned_root.glob("trial_0022/split_seed_*/training_log.csv"):
        split_seed = path.parent.name.replace("split_seed_", "")
        add_log(path, "stable tuned", "trial_0022", "one_layer_2head_deg_only_with_reference", "full", split_seed)

    df = pd.DataFrame(rows)
    if not df.empty:
        counts = df.groupby(["family", "source_run", "model", "transfer_mode"])["split_seed"].transform("nunique")
        df = df[(counts >= 5) | df["family"].eq("stable tuned")].copy()
        df.to_csv(OUT_DIR / "finetuning_loss_curves.csv", index=False)
    return df


def collect_stable_finetuning() -> pd.DataFrame:
    rows = []
    allowed_run_roots = {
        "stable_deg_overlap_2l2h_seed101_more_learning",
        "stable_deg_overlap_2l2h_random_init_5split_more_learning",
    }
    # Stable fixed 2L/2H runs.
    for path in (ROOT / "results" / "pretraining" / "finetuning_stable_deg").glob("*/**/metrics_summary.csv"):
        parts = path.parts
        if "split_seed_" not in str(path):
            continue
        run_root = next((part for part in parts if part.startswith("stable_deg_overlap_")), "")
        if run_root not in allowed_run_roots:
            continue
        split_dir = path.parent.name
        transfer = path.parent.parent.name
        arch = path.parent.parent.parent.name if len(path.parts) > 4 else "unknown"
        test = pd.read_csv(path).query("split == 'test'")
        if test.empty:
            continue
        row = test.iloc[0].to_dict()
        rows.append(
            {
                "family": "stable fixed",
                "model": f"{arch} | {transfer}",
                "split_seed": split_dir.replace("split_seed_", ""),
                **row,
            }
        )
    # Stable Optuna/refinement summaries.
    refinement = ROOT / "results" / "pretraining" / "finetuning_stable_deg_optuna" / "one_layer_2head_refine_top3_10each_3seed" / "refinement_summary.csv"
    if refinement.exists():
        df = pd.read_csv(refinement).sort_values("test_macro_f1_mean", ascending=False).head(1)
        for _, row in df.iterrows():
            top_dir = f"top_{int(row['top_index'])}_source_{int(row['source_trial'])}"
            trial_dir = f"trial_{int(row['local_trial']):04d}"
            split_paths = list((refinement.parent / top_dir / trial_dir).glob("split_seed_*/metrics_summary.csv"))
            if split_paths:
                for split_path in split_paths:
                    metrics = pd.read_csv(split_path)
                    test = metrics[metrics["split"].eq("test")]
                    if test.empty:
                        continue
                    test_row = test.iloc[0].to_dict()
                    rows.append(
                        {
                            "family": "stable tuned",
                            "model": f"1L/2H tuned top{int(row['top_index'])}/trial{int(row['local_trial'])}",
                            "split_seed": split_path.parent.name.replace("split_seed_", ""),
                            **test_row,
                        }
                    )
                continue
            rows.append(
                {
                    "family": "stable tuned",
                    "model": f"1L/2H tuned top{int(row['top_index'])}/trial{int(row['local_trial'])}",
                    "split_seed": "101,102,103",
                    "accuracy": row["test_accuracy_mean"],
                    "macro_f1": row["test_macro_f1_mean"],
                    "balanced_accuracy": row["test_balanced_accuracy_mean"],
                    "roc_auc": row["test_roc_auc_mean"],
                    "pr_auc": row["test_pr_auc_mean"],
                    "recall_mci": row["test_recall_mci_mean"],
                    "recall_ad": row["test_recall_ad_mean"],
                }
            )
    df = pd.DataFrame(rows)
    if not df.empty:
        df.to_csv(OUT_DIR / "stable_finetuning_runs.csv", index=False)
    return df


def summarize_stable(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    summary = (
        df.groupby(["family", "model"], as_index=False)
        .agg(
            runs=("macro_f1", "count"),
            test_accuracy_mean=("accuracy", "mean"),
            test_accuracy_std=("accuracy", "std"),
            test_macro_f1_mean=("macro_f1", "mean"),
            test_macro_f1_std=("macro_f1", "std"),
            test_balanced_accuracy_mean=("balanced_accuracy", "mean"),
            test_roc_auc_mean=("roc_auc", "mean"),
            recall_mci_mean=("recall_mci", "mean"),
            recall_ad_mean=("recall_ad", "mean"),
        )
        .sort_values("test_macro_f1_mean", ascending=False)
    )
    summary.to_csv(OUT_DIR / "stable_finetuning_summary.csv", index=False)
    return summary


def plot_iqr_candles(standard_runs: pd.DataFrame, stable_runs: pd.DataFrame, baselines: pd.DataFrame) -> None:
    rows = []
    if not standard_runs.empty:
        for _, row in standard_runs.iterrows():
            rows.append(
                {
                    "group": "standard",
                    "method": f"standard {row['arch']} {row['transfer_mode']}",
                    "macro_f1": as_float(row["macro_f1"]),
                }
            )
    if not stable_runs.empty:
        for _, row in stable_runs.iterrows():
            rows.append(
                {
                    "group": row["family"],
                    "method": row["model"],
                    "macro_f1": as_float(row["macro_f1"]),
                }
            )
    if not baselines.empty:
        # Baseline summaries only provide mean/std here; keep them out of the IQR plot.
        pass
    df = pd.DataFrame(rows)
    if df.empty:
        return
    order = df.groupby("method")["macro_f1"].median().sort_values(ascending=False).index.tolist()
    df.to_csv(OUT_DIR / "iqr_macro_f1_runs.csv", index=False)

    plt.figure(figsize=(12, 6))
    data = [df[df["method"].eq(method)]["macro_f1"].to_numpy() for method in order]
    box = plt.boxplot(
        data,
        tick_labels=order,
        patch_artist=True,
        showmeans=True,
        meanline=True,
        whis=(0, 100),
    )
    palette = {
        "standard": "#E45756",
        "stable fixed": "#F58518",
        "stable tuned": "#B279A2",
    }
    method_group = df.groupby("method")["group"].first().to_dict()
    for patch, method in zip(box["boxes"], order):
        patch.set_facecolor(palette.get(method_group.get(method, ""), "#4C78A8"))
        patch.set_alpha(0.55)
    plt.ylabel("Test macro F1")
    plt.title("Test macro F1 distribution across split seeds (IQR candles)")
    plt.xticks(rotation=30, ha="right", fontsize=8)
    plt.grid(axis="y", alpha=0.25)
    savefig("04_iqr_candles_test_macro_f1.png")


def plot_finetuning_loss_curves(curves: pd.DataFrame) -> None:
    if curves.empty:
        return
    # Average curves by family/architecture/transfer. This compares 4L, 2L, 1L,
    # stable and non-stable without overplotting every seed.
    curves = curves.copy()
    curves["curve_label"] = curves.apply(
        lambda r: f"{r['family']} | {r['arch']} | {r['transfer_mode']}", axis=1
    )
    grouped = (
        curves.groupby(["curve_label", "epoch"], as_index=False)
        .agg(train_loss=("train_loss", "mean"), val_macro_f1=("val_macro_f1", "mean"))
    )
    final_rank = (
        grouped.groupby("curve_label", as_index=False)
        .agg(best_val_macro_f1=("val_macro_f1", "max"), min_train_loss=("train_loss", "min"))
        .sort_values("best_val_macro_f1", ascending=False)
    )
    labels = final_rank["curve_label"].head(10).tolist()

    plt.figure(figsize=(11, 6))
    for label in labels:
        part = grouped[grouped["curve_label"].eq(label)].sort_values("epoch")
        plt.plot(part["epoch"], part["train_loss"], label=label, linewidth=1.6)
    plt.xlabel("Epoch")
    plt.ylabel("Train loss")
    plt.title("Fine-tuning train loss: standard vs stable")
    plt.legend(fontsize=7)
    plt.grid(alpha=0.25)
    savefig("08_finetuning_train_loss_comparison.png")

    val_part = grouped.dropna(subset=["val_macro_f1"])
    if not val_part.empty:
        plt.figure(figsize=(11, 6))
        for label in labels:
            part = val_part[val_part["curve_label"].eq(label)].sort_values("epoch")
            if not part.empty:
                plt.plot(part["epoch"], part["val_macro_f1"], label=label, linewidth=1.6)
        plt.xlabel("Epoch")
        plt.ylabel("Validation macro F1")
        plt.title("Fine-tuning validation macro F1: standard vs stable")
        plt.legend(fontsize=7)
        plt.grid(alpha=0.25)
        savefig("09_finetuning_validation_macro_f1_comparison.png")

    # Small multiples: standard-only and stable-only train loss.
    for family, filename, title in [
        ("standard", "10_standard_finetuning_train_loss.png", "Standard fine-tuning train loss"),
        ("stable fixed", "11_stable_finetuning_train_loss.png", "Stable fine-tuning train loss"),
    ]:
        sub = grouped[grouped["curve_label"].str.startswith(family)]
        if sub.empty:
            continue
        plt.figure(figsize=(10, 5.5))
        for label in sub["curve_label"].unique():
            part = sub[sub["curve_label"].eq(label)].sort_values("epoch")
            plt.plot(part["epoch"], part["train_loss"], label=label, linewidth=1.6)
        plt.xlabel("Epoch")
        plt.ylabel("Train loss")
        plt.title(title)
        plt.legend(fontsize=8)
        plt.grid(alpha=0.25)
        savefig(filename)


def collect_baselines() -> pd.DataFrame:
    rows = []
    pca = read_csv_optional(ROOT / "results" / "baseline" / "pca_logreg_sweep" / "deg_all_pca_logreg_sweep_5split" / "final_tables" / "pca_logreg_sweep_summary.csv")
    if not pca.empty:
        best = pca.sort_values("test_macro_f1_mean", ascending=False).iloc[0]
        rows.append(
            {
                "method": f"PCA{int(best['pca_components'])}+LogReg",
                "group": "simple baseline",
                "test_macro_f1_mean": best["test_macro_f1_mean"],
                "test_macro_f1_std": best["test_macro_f1_std"],
                "test_accuracy_mean": best["test_accuracy_mean"],
                "test_roc_auc_mean": best["test_roc_auc_mean"],
                "recall_mci_mean": best["recall_mci_mean"],
                "recall_ad_mean": best["recall_ad_mean"],
            }
        )
    pca_nn = read_csv_optional(ROOT / "results" / "baseline" / "pca50_nn" / "deg_all_pca50_vs_nn_5split" / "final_tables" / "pca50_nn_summary.csv")
    if not pca_nn.empty:
        for _, row in pca_nn.iterrows():
            name = "NN standard" if row["model"] == "nn" else f"PCA{int(row['pca_components'])}+LogReg"
            rows.append(
                {
                    "method": name,
                    "group": "simple baseline",
                    "test_macro_f1_mean": row["test_macro_f1_mean"],
                    "test_macro_f1_std": row["test_macro_f1_std"],
                    "test_accuracy_mean": row["test_accuracy_mean"],
                    "test_roc_auc_mean": row["test_roc_auc_mean"],
                    "recall_mci_mean": row["recall_mci_mean"],
                    "recall_ad_mean": row["recall_ad_mean"],
                }
            )
    return pd.DataFrame(rows)


def plot_finetuning(standard: pd.DataFrame, stable_summary: pd.DataFrame, baselines: pd.DataFrame) -> None:
    plot_rows = []
    if not standard.empty:
        best_std = standard.sort_values("test_macro_f1_mean", ascending=False).head(8).copy()
        for _, row in best_std.iterrows():
            plot_rows.append(
                {
                    "method": (
                        f"standard | {row['pretraining_arch']} | "
                        f"{row['reference_policy']} | {row['transfer_mode']}"
                    ),
                    "group": "TxT pretrained standard",
                    "test_macro_f1_mean": row["test_macro_f1_mean"],
                    "test_macro_f1_std": row.get("test_macro_f1_std", np.nan),
                    "val_macro_f1_mean": row.get("val_macro_f1_mean", np.nan),
                    "test_roc_auc_mean": np.nan,
                    "recall_mci_mean": np.nan,
                    "recall_ad_mean": np.nan,
                }
            )
    if not stable_summary.empty:
        for _, row in stable_summary.head(8).iterrows():
            plot_rows.append(
                {
                    "method": f"{row['model']}",
                    "group": row["family"],
                    "test_macro_f1_mean": row["test_macro_f1_mean"],
                    "test_macro_f1_std": row.get("test_macro_f1_std", np.nan),
                    "val_macro_f1_mean": np.nan,
                    "test_roc_auc_mean": row.get("test_roc_auc_mean", np.nan),
                    "recall_mci_mean": row.get("recall_mci_mean", np.nan),
                    "recall_ad_mean": row.get("recall_ad_mean", np.nan),
                }
            )
    if not baselines.empty:
        for _, row in baselines.drop_duplicates("method").iterrows():
            plot_rows.append(
                {
                    "method": row["method"],
                    "group": row["group"],
                    "test_macro_f1_mean": row["test_macro_f1_mean"],
                    "test_macro_f1_std": row.get("test_macro_f1_std", np.nan),
                    "val_macro_f1_mean": np.nan,
                    "test_roc_auc_mean": row.get("test_roc_auc_mean", np.nan),
                    "recall_mci_mean": row.get("recall_mci_mean", np.nan),
                    "recall_ad_mean": row.get("recall_ad_mean", np.nan),
                }
            )

    df = pd.DataFrame(plot_rows)
    if df.empty:
        return
    for column in [
        "test_macro_f1_mean",
        "test_macro_f1_std",
        "val_macro_f1_mean",
        "test_roc_auc_mean",
        "recall_mci_mean",
        "recall_ad_mean",
    ]:
        if column in df.columns:
            df[column] = df[column].apply(as_float)
    df.to_csv(OUT_DIR / "combined_method_summary.csv", index=False)
    df_sorted = df.sort_values("test_macro_f1_mean", ascending=True)

    colors = {
        "simple baseline": "#54A24B",
        "TxT pretrained standard": "#E45756",
        "stable fixed": "#F58518",
        "stable tuned": "#B279A2",
    }
    plt.figure(figsize=(12, 7))
    y = np.arange(len(df_sorted))
    bar_colors = [colors.get(group, "#4C78A8") for group in df_sorted["group"]]
    xerr = df_sorted["test_macro_f1_std"].fillna(0).to_numpy()
    plt.barh(y, df_sorted["test_macro_f1_mean"], xerr=xerr, color=bar_colors, alpha=0.9)
    plt.yticks(y, df_sorted["method"], fontsize=8)
    plt.xlabel("Test macro F1 mean")
    plt.title("AD vs MCI: model comparison")
    plt.xlim(0.45, max(0.76, float(df_sorted["test_macro_f1_mean"].max()) + 0.04))
    plt.grid(axis="x", alpha=0.25)
    legend_handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in colors.values()]
    plt.legend(legend_handles, colors.keys(), fontsize=8, loc="lower right")
    savefig("04_model_comparison_test_macro_f1.png")

    # Validation vs test for standard and tuned stable methods.
    vt = []
    if not standard.empty:
        for _, row in standard.sort_values("test_macro_f1_mean", ascending=False).head(12).iterrows():
            vt.append(
                {
                    "method": f"standard {row['pretraining_arch']} {row['transfer_mode']}",
                    "val_macro_f1": row["val_macro_f1_mean"],
                    "test_macro_f1": row["test_macro_f1_mean"],
                }
            )
    refinement = ROOT / "results" / "pretraining" / "finetuning_stable_deg_optuna" / "one_layer_2head_refine_top3_10each_3seed" / "refinement_summary.csv"
    if refinement.exists():
        r = pd.read_csv(refinement).sort_values("val_macro_f1_mean", ascending=False).head(10)
        for _, row in r.iterrows():
            vt.append(
                {
                    "method": f"stable tuned top{int(row['top_index'])}-{int(row['local_trial'])}",
                    "val_macro_f1": row["val_macro_f1_mean"],
                    "test_macro_f1": row["test_macro_f1_mean"],
                }
            )
    vt_df = pd.DataFrame(vt)
    if not vt_df.empty:
        vt_df.to_csv(OUT_DIR / "validation_vs_test_macro_f1.csv", index=False)
        plt.figure(figsize=(7, 6))
        plt.scatter(vt_df["val_macro_f1"], vt_df["test_macro_f1"], color="#4C78A8", alpha=0.85)
        for _, row in vt_df.iterrows():
            if row["val_macro_f1"] - row["test_macro_f1"] > 0.09:
                plt.annotate(row["method"], (row["val_macro_f1"], row["test_macro_f1"]), fontsize=6, alpha=0.8)
        lim_min = min(vt_df["val_macro_f1"].min(), vt_df["test_macro_f1"].min()) - 0.02
        lim_max = max(vt_df["val_macro_f1"].max(), vt_df["test_macro_f1"].max()) + 0.02
        plt.plot([lim_min, lim_max], [lim_min, lim_max], linestyle="--", color="black", linewidth=1)
        plt.xlabel("Validation macro F1")
        plt.ylabel("Test macro F1")
        plt.title("Validation-test gap during fine-tuning")
        plt.grid(alpha=0.25)
        savefig("05_validation_vs_test_macro_f1_gap.png")

    # Recall balance plot.
    recall_df = df.dropna(subset=["recall_mci_mean", "recall_ad_mean"]).copy()
    if not recall_df.empty:
        recall_df = recall_df.sort_values("test_macro_f1_mean", ascending=False).head(10)
        x = np.arange(len(recall_df))
        width = 0.38
        plt.figure(figsize=(11, 5.5))
        plt.bar(x - width / 2, recall_df["recall_mci_mean"], width, label="MCI recall", color="#4C78A8")
        plt.bar(x + width / 2, recall_df["recall_ad_mean"], width, label="AD recall", color="#F58518")
        plt.xticks(x, recall_df["method"], rotation=35, ha="right", fontsize=8)
        plt.ylabel("Mean recall")
        plt.title("Class recall balance")
        plt.ylim(0, 1)
        plt.legend()
        plt.grid(axis="y", alpha=0.25)
        savefig("06_recall_balance.png")


def plot_stability_examples() -> None:
    # Best available stable runs: plot validation macro F1 curves for representative splits.
    paths = [
        ROOT / "results" / "pretraining" / "finetuning_stable_deg" / "stable_deg_overlap_2l2h_seed101_more_learning" / "two_layer_2head_deg_only_with_reference" / "full" / "split_seed_101" / "training_log.csv",
        ROOT / "results" / "pretraining" / "finetuning_stable_deg_optuna" / "one_layer_2head_deg_overlap_optuna_finetune_only" / "trial_0022" / "split_seed_101" / "training_log.csv",
    ]
    labels = ["stable 2L/2H full seed101", "stable tuned 1L/2H trial22 seed101"]
    plt.figure(figsize=(10, 5.5))
    any_plot = False
    for path, label in zip(paths, labels):
        if path.exists():
            df = pd.read_csv(path)
            if "val_macro_f1" in df.columns:
                plt.plot(df["epoch"], df["val_macro_f1"], label=label, linewidth=1.7)
                any_plot = True
    if any_plot:
        plt.xlabel("Epoch")
        plt.ylabel("Validation macro F1")
        plt.title("Stable fine-tuning: validation macro F1 curves")
        plt.legend(fontsize=8)
        plt.grid(alpha=0.25)
        savefig("07_stable_finetuning_validation_curves.png")
    else:
        plt.close()


def main() -> None:
    ensure_out()
    pretraining = collect_pretraining_logs()
    plot_pretraining(pretraining)

    standard = collect_standard_finetuning()
    standard_runs = collect_standard_finetuning_runs()
    finetuning_curves = collect_finetuning_loss_curves()
    if not standard.empty and not standard_runs.empty:
        allowed = set(zip(standard["source_run"], standard["pretraining_arch"], standard["transfer_mode"]))
        standard_runs = standard_runs[
            standard_runs.apply(
                lambda r: (r["source_run"], strip_reference_suffix(r["model"]), r["transfer_mode"]) in allowed,
                axis=1,
            )
        ].copy()
        standard_runs.to_csv(OUT_DIR / "standard_finetuning_test_runs.csv", index=False)
    if not standard.empty and not finetuning_curves.empty:
        allowed = set(zip(standard["source_run"], standard["pretraining_arch"], standard["transfer_mode"]))
        finetuning_curves = finetuning_curves[
            finetuning_curves.apply(
                lambda r: r["family"] != "standard"
                or (r["source_run"], strip_reference_suffix(r["model"]), r["transfer_mode"]) in allowed,
                axis=1,
            )
        ].copy()
        finetuning_curves.to_csv(OUT_DIR / "finetuning_loss_curves.csv", index=False)
    stable_runs = collect_stable_finetuning()
    stable_summary = summarize_stable(stable_runs)
    baselines = collect_baselines()
    if not baselines.empty:
        baselines.to_csv(OUT_DIR / "baseline_summary.csv", index=False)
    plot_finetuning(standard, stable_summary, baselines)
    plot_iqr_candles(standard_runs, stable_runs, baselines)
    plot_finetuning_loss_curves(finetuning_curves)
    plot_stability_examples()

    manifest = {
        "output_dir": str(OUT_DIR),
        "figures": sorted(path.name for path in OUT_DIR.glob("*.png")),
        "tables": sorted(path.name for path in OUT_DIR.glob("*.csv")),
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
