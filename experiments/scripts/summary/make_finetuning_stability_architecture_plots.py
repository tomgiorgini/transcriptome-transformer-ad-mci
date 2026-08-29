#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = ROOT / "results" / "meeting_figures" / f"finetuning_arch_stability_{datetime.now():%Y%m%d_%H%M%S}"


ARCH_ORDER = ["1L/2H", "2L/2H", "4L/4H"]
TRANSFER_ORDER = ["full", "embedding_only"]


@dataclass(frozen=True)
class RunSpec:
    key: str
    arch: str
    family: str
    transfer_mode: str
    label: str
    log_paths: tuple[Path, ...]
    metrics_paths: tuple[Path, ...]


def arch_from_name(name: str) -> str | None:
    lowered = name.lower()
    if "one_layer_2head" in lowered or "1l" in lowered:
        return "1L/2H"
    if "two_layer_2head" in lowered or "2l" in lowered:
        return "2L/2H"
    if "larger_4layer" in lowered or "4layer" in lowered or "4l" in lowered:
        return "4L/4H"
    return None


def read_metrics_test(path: Path) -> dict[str, float] | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    test = df[df["split"].eq("test")]
    if test.empty:
        return None
    row = test.iloc[0]
    return {
        "test_macro_f1": float(row["macro_f1"]),
        "test_accuracy": float(row.get("accuracy", np.nan)),
        "test_roc_auc": float(row.get("roc_auc", np.nan)) if "roc_auc" in row else np.nan,
    }


def summarize_metrics(paths: tuple[Path, ...]) -> dict[str, float]:
    rows = [read_metrics_test(path) for path in paths]
    rows = [row for row in rows if row is not None]
    if not rows:
        return {"runs": 0, "test_macro_f1_mean": np.nan, "test_macro_f1_std": np.nan}
    values = np.asarray([row["test_macro_f1"] for row in rows], dtype=float)
    return {
        "runs": int(len(values)),
        "test_macro_f1_mean": float(np.mean(values)),
        "test_macro_f1_std": float(np.std(values, ddof=1)) if len(values) > 1 else np.nan,
    }


def average_curves(paths: tuple[Path, ...]) -> pd.DataFrame:
    frames = []
    for path in paths:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if "epoch" not in df.columns or "train_loss" not in df.columns:
            continue
        keep = ["epoch", "train_loss"]
        if "val_macro_f1" in df.columns:
            keep.append("val_macro_f1")
        elif "val_score" in df.columns:
            keep.append("val_score")
            df = df.rename(columns={"val_score": "val_macro_f1"})
            keep[-1] = "val_macro_f1"
        else:
            df["val_macro_f1"] = np.nan
            keep.append("val_macro_f1")
        frames.append(df[keep].copy())
    if not frames:
        return pd.DataFrame(columns=["epoch", "train_loss", "val_macro_f1"])
    df = pd.concat(frames, ignore_index=True)
    return (
        df.groupby("epoch", as_index=False)
        .agg(train_loss=("train_loss", "mean"), val_macro_f1=("val_macro_f1", "mean"))
        .sort_values("epoch")
    )


def standard_specs() -> list[RunSpec]:
    specs: list[RunSpec] = []
    root = ROOT / "results" / "pretraining" / "finetuning" / "finetune_degonly"
    for model_dir in root.glob("*_with_reference"):
        arch = arch_from_name(model_dir.name)
        if arch is None:
            continue
        for transfer in TRANSFER_ORDER:
            tdir = model_dir / transfer
            logs = tuple(sorted(tdir.glob("split_seed_*/training_log.csv")))
            metrics = tuple(sorted(tdir.glob("split_seed_*/metrics_summary.csv")))
            if logs:
                specs.append(
                    RunSpec(
                        key=f"standard:{arch}:{transfer}:{model_dir.name}",
                        arch=arch,
                        family="non-stable",
                        transfer_mode=transfer,
                        label=f"{arch} non-stable {transfer}",
                        log_paths=logs,
                        metrics_paths=metrics,
                    )
                )
    return specs


def stable_fixed_specs() -> list[RunSpec]:
    specs: list[RunSpec] = []
    allowed_roots = [
        ROOT / "results" / "pretraining" / "finetuning_stable_deg" / "stable_deg_overlap_2l2h_seed101_more_learning",
        ROOT / "results" / "pretraining" / "finetuning_stable_deg" / "stable_deg_overlap_2l2h_random_init_5split_more_learning",
    ]
    for run_root in allowed_roots:
        if not run_root.exists():
            continue
        for model_dir in run_root.glob("*_with_reference"):
            arch = arch_from_name(model_dir.name)
            if arch is None:
                continue
            for transfer_dir in model_dir.iterdir():
                if not transfer_dir.is_dir():
                    continue
                transfer = transfer_dir.name
                if transfer not in {"full", "embedding_only", "random_init"}:
                    continue
                logs = tuple(sorted(transfer_dir.glob("split_seed_*/training_log.csv")))
                metrics = tuple(sorted(transfer_dir.glob("split_seed_*/metrics_summary.csv")))
                if logs:
                    specs.append(
                        RunSpec(
                            key=f"stable_fixed:{arch}:{transfer}:{run_root.name}",
                            arch=arch,
                            family="stable",
                            transfer_mode=transfer,
                            label=f"{arch} stable {transfer}",
                            log_paths=logs,
                            metrics_paths=metrics,
                        )
                    )
    return specs


def stable_tuned_specs() -> list[RunSpec]:
    specs: list[RunSpec] = []
    refine = ROOT / "results" / "pretraining" / "finetuning_stable_deg_optuna" / "one_layer_2head_refine_top3_10each_3seed"
    summary = refine / "refinement_summary.csv"
    if summary.exists():
        df = pd.read_csv(summary).sort_values("test_macro_f1_mean", ascending=False)
        if not df.empty:
            row = df.iloc[0]
            top_dir = refine / f"top_{int(row['top_index'])}_source_{int(row['source_trial'])}" / f"trial_{int(row['local_trial']):04d}"
            logs = tuple(sorted(top_dir.glob("split_seed_*/training_log.csv")))
            metrics = tuple(sorted(top_dir.glob("split_seed_*/metrics_summary.csv")))
            if logs:
                specs.append(
                    RunSpec(
                        key="stable_tuned:1L/2H:full:best_refinement",
                        arch="1L/2H",
                        family="stable",
                        transfer_mode="full",
                        label="1L/2H stable tuned full",
                        log_paths=logs,
                        metrics_paths=metrics,
                    )
                )
    return specs


def select_best(specs: list[RunSpec], arch: str, family: str, transfer: str | None = None) -> RunSpec | None:
    candidates = [s for s in specs if s.arch == arch and s.family == family]
    if transfer is not None:
        candidates = [s for s in candidates if s.transfer_mode == transfer]
    if not candidates:
        return None
    ranked = sorted(candidates, key=lambda s: summarize_metrics(s.metrics_paths)["test_macro_f1_mean"], reverse=True)
    return ranked[0]


def plot_arch_panels_nonstable_vs_stable(specs: list[RunSpec]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.8), sharey=False)
    curve_rows = []
    for ax, arch in zip(axes, ARCH_ORDER):
        ax2 = ax.twinx()
        selected = [
            select_best(specs, arch, "non-stable", "full"),
            select_best(specs, arch, "stable", "full"),
        ]
        colors = {"non-stable": "#4C78A8", "stable": "#E45756"}
        any_curve = False
        for spec in selected:
            if spec is None:
                continue
            curves = average_curves(spec.log_paths)
            metrics = summarize_metrics(spec.metrics_paths)
            if curves.empty:
                continue
            any_curve = True
            color = colors[spec.family]
            ax.plot(curves["epoch"], curves["train_loss"], color=color, linestyle="-", linewidth=1.9, label=f"{spec.family} train loss")
            ax2.plot(curves["epoch"], curves["val_macro_f1"], color=color, linestyle="--", linewidth=1.9, label=f"{spec.family} val macro F1")
            tmp = curves.copy()
            tmp["arch"] = arch
            tmp["family"] = spec.family
            tmp["transfer_mode"] = spec.transfer_mode
            tmp["test_macro_f1_mean"] = metrics["test_macro_f1_mean"]
            curve_rows.append(tmp)
        ax.set_title(arch)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Train loss")
        ax2.set_ylabel("Validation macro F1")
        ax.grid(alpha=0.25)
        if not any_curve:
            ax.text(0.5, 0.5, "missing", transform=ax.transAxes, ha="center", va="center", fontsize=14)
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, fontsize=8, loc="lower right")
    fig.suptitle("Fine-tuning: non-stable vs stable full transfer")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "01_nonstable_vs_stable_by_arch_train_loss_val_macro_f1.png", dpi=240, bbox_inches="tight")
    plt.close(fig)
    if curve_rows:
        pd.concat(curve_rows, ignore_index=True).to_csv(OUT_DIR / "nonstable_vs_stable_curves.csv", index=False)


def plot_arch_panels_stable_full_vs_embedding(specs: list[RunSpec]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.8), sharey=False)
    curve_rows = []
    for ax, arch in zip(axes, ARCH_ORDER):
        ax2 = ax.twinx()
        selected = [
            select_best(specs, arch, "stable", "full"),
            select_best(specs, arch, "stable", "embedding_only"),
        ]
        colors = {"full": "#54A24B", "embedding_only": "#F58518"}
        any_curve = False
        for spec in selected:
            if spec is None:
                continue
            curves = average_curves(spec.log_paths)
            metrics = summarize_metrics(spec.metrics_paths)
            if curves.empty:
                continue
            any_curve = True
            color = colors[spec.transfer_mode]
            ax.plot(curves["epoch"], curves["train_loss"], color=color, linestyle="-", linewidth=1.9, label=f"{spec.transfer_mode} train loss")
            ax2.plot(curves["epoch"], curves["val_macro_f1"], color=color, linestyle="--", linewidth=1.9, label=f"{spec.transfer_mode} val macro F1")
            tmp = curves.copy()
            tmp["arch"] = arch
            tmp["family"] = spec.family
            tmp["transfer_mode"] = spec.transfer_mode
            tmp["test_macro_f1_mean"] = metrics["test_macro_f1_mean"]
            curve_rows.append(tmp)
        ax.set_title(arch)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Train loss")
        ax2.set_ylabel("Validation macro F1")
        ax.grid(alpha=0.25)
        if not any_curve:
            ax.text(0.5, 0.5, "missing", transform=ax.transAxes, ha="center", va="center", fontsize=14)
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, fontsize=8, loc="lower right")
    fig.suptitle("Stable fine-tuning: full vs embedding-only transfer")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "02_stable_full_vs_embedding_by_arch_train_loss_val_macro_f1.png", dpi=240, bbox_inches="tight")
    plt.close(fig)
    if curve_rows:
        pd.concat(curve_rows, ignore_index=True).to_csv(OUT_DIR / "stable_full_vs_embedding_curves.csv", index=False)


def plot_best_val_macro_f1_by_arch(specs: list[RunSpec]) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    rows = []
    colors = {"1L/2H": "#4C78A8", "2L/2H": "#F58518", "4L/4H": "#54A24B"}
    for arch in ARCH_ORDER:
        candidates = [s for s in specs if s.arch == arch]
        if not candidates:
            rows.append({"arch": arch, "status": "missing"})
            continue
        best = max(candidates, key=lambda s: summarize_metrics(s.metrics_paths)["test_macro_f1_mean"])
        curves = average_curves(best.log_paths)
        metrics = summarize_metrics(best.metrics_paths)
        if curves.empty:
            rows.append({"arch": arch, "status": "missing_curve", "spec": best.label})
            continue
        label = f"{arch} {best.family} {best.transfer_mode} | test macro F1={metrics['test_macro_f1_mean']:.3f}"
        ax.plot(curves["epoch"], curves["val_macro_f1"], linewidth=2.2, color=colors[arch], label=label)
        rows.append(
            {
                "arch": arch,
                "status": "available",
                "selected_label": best.label,
                "family": best.family,
                "transfer_mode": best.transfer_mode,
                **metrics,
            }
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation macro F1")
    ax.set_title("Best available validation macro F1 trajectory per architecture")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "03_best_val_macro_f1_by_arch_with_test_macro_f1_legend.png", dpi=240, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(rows).to_csv(OUT_DIR / "best_curve_selection_by_arch.csv", index=False)


def missing_tests(specs: list[RunSpec]) -> pd.DataFrame:
    rows = []
    for arch in ARCH_ORDER:
        for family in ["non-stable", "stable"]:
            for transfer in ["full", "embedding_only"]:
                present = select_best(specs, arch, family, transfer) is not None
                rows.append(
                    {
                        "arch": arch,
                        "family": family,
                        "transfer_mode": transfer,
                        "available": present,
                        "needed_for_requested_graphs": True,
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    specs = standard_specs() + stable_fixed_specs() + stable_tuned_specs()
    summary_rows = []
    for spec in specs:
        metrics = summarize_metrics(spec.metrics_paths)
        summary_rows.append(
            {
                "key": spec.key,
                "arch": spec.arch,
                "family": spec.family,
                "transfer_mode": spec.transfer_mode,
                "label": spec.label,
                "log_count": len(spec.log_paths),
                "metrics_count": len(spec.metrics_paths),
                **metrics,
            }
        )
    pd.DataFrame(summary_rows).sort_values(["arch", "family", "transfer_mode"]).to_csv(OUT_DIR / "available_finetuning_runs.csv", index=False)
    missing = missing_tests(specs)
    missing.to_csv(OUT_DIR / "missing_tests_for_requested_graphs.csv", index=False)
    plot_arch_panels_nonstable_vs_stable(specs)
    plot_arch_panels_stable_full_vs_embedding(specs)
    plot_best_val_macro_f1_by_arch(specs)
    print(f"Output: {OUT_DIR}")
    print("Missing tests:")
    print(missing[~missing["available"]].to_string(index=False))


if __name__ == "__main__":
    main()
