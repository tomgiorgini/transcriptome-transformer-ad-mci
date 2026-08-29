#!/usr/bin/env python3
"""Rank TxT key attention by a single arithmetic mean across selected seeds.

This is intentionally separate from the Borda/bootstrap consensus analysis.  It
uses only genes present in every included seed and averages
``incoming_macro_mean`` with equal seed weight.  The default output remains the
mean-only top-10/top-20 analysis; an explicitly separate mode can also write
white-background mean +/- one-standard-deviation plots and sample-variance
tables without replacing any mean-only artifact.  A second, plot-only mode
recreates the same mean/variance summaries as compact portrait forest plots.
A third separate mode writes compact portrait IQR candles (min, Q1, median,
Q3, max) while preserving the original mean-based ranking.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator
import numpy as np
import pandas as pd


DEFAULT_INCLUDED_SEEDS = [101, 102, 103, 105, 106, 107, 109, 110]
DEFAULT_EXCLUDED_SEEDS = [104, 108]
DEFAULT_OUTPUT_DIRNAME = "key_attention_mean_only_excluding_104_108"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Mean-only TxT baseline key-attention ranking; VMA/query excluded."
        )
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument(
        "--included-seeds",
        type=int,
        nargs="+",
        default=DEFAULT_INCLUDED_SEEDS,
    )
    parser.add_argument(
        "--excluded-seeds",
        type=int,
        nargs="+",
        default=DEFAULT_EXCLUDED_SEEDS,
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--dot-white-only",
        action="store_true",
        help=(
            "Read the existing top-10/top-20 CSV files and create only the "
            "opaque white-background dot plots. Existing mean-only outputs "
            "are left untouched."
        ),
    )
    parser.add_argument(
        "--mean-variance-white-only",
        action="store_true",
        help=(
            "Recompute the common-gene ranking from the seed exports and write "
            "only new white-background plots with mean points and +/-1 sample "
            "standard-deviation whiskers, dedicated variance CSV files, and a "
            "dedicated manifest. Existing mean-only outputs are left untouched."
        ),
    )
    parser.add_argument(
        "--mean-variance-forest-white-portrait-only",
        action="store_true",
        help=(
            "Recompute the common-gene mean/variance ranking and write only "
            "compact portrait forest plots (top 10 and top 20) plus their "
            "dedicated manifest. No existing CSV or plot is replaced."
        ),
    )
    parser.add_argument(
        "--iqr-candle-white-portrait-only",
        action="store_true",
        help=(
            "Recompute the common-gene mean ranking and write only a new "
            "IQR-candle artifact family: all/top-10/top-20 statistics CSVs, "
            "compact white portrait PNG/PDF plots, and a dedicated manifest. "
            "Candles show min, linearly interpolated Q1/median/Q3, and max "
            "across seeds. No existing artifact is replaced."
        ),
    )
    return parser


def discover_analysis_dirs(run_root: Path, seeds: Sequence[int]) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for seed in seeds:
        matches = sorted(
            path
            for path in (run_root / "seeds10").glob(
                f"*/seed_{seed}/key_attention/test"
            )
            if (path / "key_attention_by_seed.csv").exists()
        )
        if len(matches) != 1:
            raise FileNotFoundError(
                f"Expected exactly one key-attention export for seed {seed}; "
                f"found {matches}."
            )
        result[int(seed)] = matches[0]
    return result


def load_seed_frames(paths: dict[int, Path]) -> dict[int, pd.DataFrame]:
    frames: dict[int, pd.DataFrame] = {}
    protocol_hash: str | None = None
    dataset_hashes: tuple[str, str] | None = None
    checkpoint_hashes: set[str] = set()

    for seed, analysis_dir in sorted(paths.items()):
        manifest_path = analysis_dir / "key_attention_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing extraction manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        expected_manifest_values = {
            "seed": seed,
            "model_variant": "baseline",
            "split": "test",
            "vma_included": False,
            "query_metrics_included": False,
            "is_full_split": True,
        }
        for field, expected in expected_manifest_values.items():
            if manifest.get(field) != expected:
                raise ValueError(
                    f"Unexpected {field}={manifest.get(field)!r} in {manifest_path}; "
                    f"expected {expected!r}."
                )

        current_protocol_hash = str(manifest.get("protocol_sha256", ""))
        if not current_protocol_hash:
            raise ValueError(f"Missing protocol fingerprint in {manifest_path}.")
        if protocol_hash is None:
            protocol_hash = current_protocol_hash
        elif current_protocol_hash != protocol_hash:
            raise ValueError(f"Protocol mismatch in {manifest_path}.")

        artifact_hashes = manifest.get("artifact_sha256", {})
        current_dataset_hashes = (
            str(artifact_hashes.get("x", "")),
            str(artifact_hashes.get("y", "")),
        )
        if not all(current_dataset_hashes):
            raise ValueError(f"Missing dataset fingerprints in {manifest_path}.")
        if dataset_hashes is None:
            dataset_hashes = current_dataset_hashes
        elif current_dataset_hashes != dataset_hashes:
            raise ValueError(f"Dataset mismatch in {manifest_path}.")

        checkpoint_hash = str(manifest.get("checkpoint_sha256", ""))
        if not checkpoint_hash:
            raise ValueError(f"Missing checkpoint fingerprint in {manifest_path}.")
        if checkpoint_hash in checkpoint_hashes:
            raise ValueError(f"Duplicate checkpoint detected at seed {seed}.")
        checkpoint_hashes.add(checkpoint_hash)

        frame = pd.read_csv(analysis_dir / "key_attention_by_seed.csv")
        required_columns = {"gene", "incoming_macro_mean"}
        missing = required_columns.difference(frame.columns)
        if missing:
            raise ValueError(
                f"{analysis_dir} lacks required columns: {sorted(missing)}."
            )
        if frame["gene"].duplicated().any():
            raise ValueError(f"Duplicate genes in {analysis_dir}.")
        if int(manifest.get("genes", -1)) != len(frame):
            raise ValueError(f"Gene-count mismatch in {analysis_dir}.")

        frame = frame.loc[:, ["gene", "incoming_macro_mean"]].copy()
        frame["gene"] = frame["gene"].astype(str)
        frame["incoming_macro_mean"] = pd.to_numeric(
            frame["incoming_macro_mean"], errors="raise"
        )
        values = frame["incoming_macro_mean"].to_numpy(dtype=float)
        if not np.isfinite(values).all() or np.any(values <= 0.0):
            raise ValueError(f"Invalid incoming attention values in {analysis_dir}.")
        frames[int(seed)] = frame.set_index("gene", verify_integrity=True)

    return frames


def build_mean_ranking(frames: dict[int, pd.DataFrame]) -> tuple[pd.DataFrame, int]:
    common_genes = set.intersection(*(set(frame.index) for frame in frames.values()))
    if not common_genes:
        raise ValueError("No gene is present in every included seed.")
    ordered_genes = sorted(common_genes)
    matrix = np.vstack(
        [
            frames[seed].loc[ordered_genes, "incoming_macro_mean"].to_numpy(float)
            for seed in sorted(frames)
        ]
    )
    mean_incoming = matrix.mean(axis=0)
    ranking = pd.DataFrame(
        {
            "gene": ordered_genes,
            "mean_incoming_attention": mean_incoming,
            "mean_enrichment_percent_vs_uniform": 100.0 * (mean_incoming - 1.0),
        }
    )
    ranking = ranking.sort_values(
        ["mean_incoming_attention", "gene"],
        ascending=[False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    ranking.insert(0, "mean_attention_rank", np.arange(1, len(ranking) + 1))
    return ranking, len(common_genes)


def build_mean_variance_ranking(
    frames: dict[int, pd.DataFrame],
) -> tuple[pd.DataFrame, int]:
    """Rank common genes by mean and retain sample variability across seeds.

    The variance estimator is the sample variance (``ddof=1``).  Attention is
    dimensionless; enrichment is expressed in percentage points relative to
    the uniform baseline, so its variance is in squared percentage points.
    """

    if len(frames) < 2:
        raise ValueError("At least two seeds are required to estimate variance.")

    common_genes = set.intersection(*(set(frame.index) for frame in frames.values()))
    if not common_genes:
        raise ValueError("No gene is present in every included seed.")
    ordered_genes = sorted(common_genes)
    matrix = np.vstack(
        [
            frames[seed].loc[ordered_genes, "incoming_macro_mean"].to_numpy(float)
            for seed in sorted(frames)
        ]
    )

    mean_incoming = matrix.mean(axis=0)
    sample_variance_incoming = matrix.var(axis=0, ddof=1)
    sample_sd_incoming = np.sqrt(sample_variance_incoming)
    ranking = pd.DataFrame(
        {
            "gene": ordered_genes,
            "seed_count": matrix.shape[0],
            "mean_incoming_attention": mean_incoming,
            "sample_variance_incoming_attention": sample_variance_incoming,
            "sample_standard_deviation_incoming_attention": sample_sd_incoming,
            "mean_enrichment_percent_vs_uniform": 100.0 * (mean_incoming - 1.0),
            "sample_variance_enrichment_percentage_points_squared": (
                10000.0 * sample_variance_incoming
            ),
            "sample_standard_deviation_enrichment_percentage_points": (
                100.0 * sample_sd_incoming
            ),
        }
    )
    ranking = ranking.sort_values(
        ["mean_incoming_attention", "gene"],
        ascending=[False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    ranking.insert(0, "mean_attention_rank", np.arange(1, len(ranking) + 1))
    return ranking, len(common_genes)


def build_iqr_candle_ranking(
    frames: dict[int, pd.DataFrame],
) -> tuple[pd.DataFrame, int]:
    """Rank common genes by mean and summarize their across-seed distribution.

    The ranking intentionally remains the descending arithmetic mean used by
    all preceding artifact families.  Quartiles use NumPy's explicit
    ``method="linear"`` estimator (Hyndman--Fan type 7).  Min/max are the
    observed extrema across the included seeds, not Tukey fences.
    """

    if len(frames) < 2:
        raise ValueError("At least two seeds are required for IQR summaries.")

    common_genes = set.intersection(*(set(frame.index) for frame in frames.values()))
    if not common_genes:
        raise ValueError("No gene is present in every included seed.")
    ordered_genes = sorted(common_genes)
    matrix = np.vstack(
        [
            frames[seed].loc[ordered_genes, "incoming_macro_mean"].to_numpy(float)
            for seed in sorted(frames)
        ]
    )
    enrichment = 100.0 * (matrix - 1.0)
    quantile_probabilities = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    incoming_quantiles = np.quantile(
        matrix,
        quantile_probabilities,
        axis=0,
        method="linear",
    )
    enrichment_quantiles = np.quantile(
        enrichment,
        quantile_probabilities,
        axis=0,
        method="linear",
    )
    mean_incoming = matrix.mean(axis=0)

    ranking = pd.DataFrame(
        {
            "gene": ordered_genes,
            "seed_count": matrix.shape[0],
            "mean_incoming_attention": mean_incoming,
            "mean_enrichment_percent_vs_uniform": 100.0 * (mean_incoming - 1.0),
            "minimum_incoming_attention": incoming_quantiles[0],
            "first_quartile_incoming_attention": incoming_quantiles[1],
            "median_incoming_attention": incoming_quantiles[2],
            "third_quartile_incoming_attention": incoming_quantiles[3],
            "maximum_incoming_attention": incoming_quantiles[4],
            "iqr_incoming_attention": incoming_quantiles[3] - incoming_quantiles[1],
            "minimum_enrichment_percentage_points": enrichment_quantiles[0],
            "first_quartile_enrichment_percentage_points": enrichment_quantiles[1],
            "median_enrichment_percentage_points": enrichment_quantiles[2],
            "third_quartile_enrichment_percentage_points": enrichment_quantiles[3],
            "maximum_enrichment_percentage_points": enrichment_quantiles[4],
            "iqr_enrichment_percentage_points": (
                enrichment_quantiles[3] - enrichment_quantiles[1]
            ),
        }
    )
    for row_index, seed in enumerate(sorted(frames)):
        ranking[f"seed_{seed}_enrichment_percentage_points"] = enrichment[
            row_index
        ]
    ranking = ranking.sort_values(
        ["mean_incoming_attention", "gene"],
        ascending=[False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    ranking.insert(0, "mean_attention_rank", np.arange(1, len(ranking) + 1))
    return ranking, len(common_genes)


def save_figure(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".png"), dpi=240, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_mean_only(
    ranking: pd.DataFrame,
    top_count: int,
    seeds: Sequence[int],
    excluded_seeds: Sequence[int],
    output_dir: Path,
) -> None:
    top = ranking.head(top_count).sort_values(
        "mean_attention_rank", ascending=False
    )
    values = top["mean_enrichment_percent_vs_uniform"].to_numpy(float)
    labels = top["gene"].astype(str).to_numpy()
    y = np.arange(len(top))

    fig, ax = plt.subplots(figsize=(10.4, max(5.5, 0.40 * top_count + 2.1)))
    fig.subplots_adjust(left=0.19, right=0.96, bottom=0.12, top=0.84)
    bars = ax.barh(y, values, color="#3f6fa8", height=0.66)
    ax.set_yticks(y, labels)
    ax.set_xlabel("Arricchimento medio di attenzione entrante vs uniforme (%)")
    ax.set_ylabel("Key gene")
    fig.suptitle(
        f"TxT baseline — top {top_count} key per attenzione media",
        fontsize=15,
        y=0.97,
    )
    fig.text(
        0.5,
        0.915,
        "Media aritmetica su seed "
        + ", ".join(map(str, seeds))
        + "; esclusi "
        + ", ".join(map(str, excluded_seeds))
        + "; soli geni comuni",
        ha="center",
        va="center",
        fontsize=9,
        color="#555555",
    )
    ax.axvline(0.0, color="#777777", linewidth=0.9)
    ax.grid(axis="x", color="#dddddd", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    minimum = min(float(values.min()), 0.0)
    maximum = max(float(values.max()), 0.0)
    span = max(maximum - minimum, 0.1)
    ax.set_xlim(minimum - 0.03 * span, maximum + 0.20 * span)
    for bar, value in zip(bars, values):
        horizontal_alignment = "left" if value >= 0.0 else "right"
        offset = 0.012 * span if value >= 0.0 else -0.012 * span
        ax.text(
            value + offset,
            bar.get_y() + bar.get_height() / 2.0,
            f"{value:+.3f}%",
            va="center",
            ha=horizontal_alignment,
            fontsize=9,
            color="#333333",
        )

    save_figure(fig, output_dir / f"baseline_key_attention_mean_top{top_count}")


def _validate_dot_plot_frame(frame: pd.DataFrame, top_count: int, path: Path) -> None:
    required_columns = {
        "mean_attention_rank",
        "gene",
        "mean_enrichment_percent_vs_uniform",
    }
    missing = required_columns.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} lacks required columns: {sorted(missing)}.")
    if len(frame) != top_count:
        raise ValueError(
            f"Expected {top_count} rows in {path}; found {len(frame)}."
        )
    if frame["gene"].astype(str).duplicated().any():
        raise ValueError(f"Duplicate genes in {path}.")
    ranks = pd.to_numeric(frame["mean_attention_rank"], errors="raise").to_numpy()
    if not np.array_equal(np.sort(ranks.astype(int)), np.arange(1, top_count + 1)):
        raise ValueError(f"Unexpected ranks in {path}.")
    values = pd.to_numeric(
        frame["mean_enrichment_percent_vs_uniform"], errors="raise"
    ).to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError(f"Non-finite enrichment values in {path}.")


def save_opaque_figure(fig: plt.Figure, stem: Path) -> None:
    """Save a new PNG/PDF pair with an explicitly opaque white canvas."""

    fig.savefig(
        stem.with_suffix(".png"),
        dpi=240,
        bbox_inches="tight",
        facecolor="white",
        transparent=False,
    )
    fig.savefig(
        stem.with_suffix(".pdf"),
        bbox_inches="tight",
        facecolor="white",
        transparent=False,
    )
    plt.close(fig)


def plot_mean_dot_white(
    top_frame: pd.DataFrame,
    top_count: int,
    seeds: Sequence[int],
    excluded_seeds: Sequence[int],
    output_dir: Path,
) -> None:
    """Plot one mean point per gene, without uncertainty or seed-level marks."""

    top = top_frame.copy()
    source_path = output_dir / f"key_attention_mean_top{top_count}.csv"
    _validate_dot_plot_frame(top, top_count, source_path)
    top["mean_attention_rank"] = pd.to_numeric(
        top["mean_attention_rank"], errors="raise"
    ).astype(int)
    top["mean_enrichment_percent_vs_uniform"] = pd.to_numeric(
        top["mean_enrichment_percent_vs_uniform"], errors="raise"
    )
    top = top.sort_values("mean_attention_rank", ascending=False, kind="mergesort")

    values = top["mean_enrichment_percent_vs_uniform"].to_numpy(float)
    labels = top["gene"].astype(str).to_numpy()
    y = np.arange(len(top))

    fig_height = max(5.8, 0.43 * top_count + 2.0)
    fig, ax = plt.subplots(figsize=(10.6, fig_height), facecolor="white")
    ax.set_facecolor("white")
    fig.subplots_adjust(left=0.20, right=0.95, bottom=0.12, top=0.84)

    ax.scatter(
        values,
        y,
        s=92,
        color="#67b3ef",
        edgecolors="#2f75a8",
        linewidths=0.8,
        zorder=3,
    )
    ax.set_yticks(y, labels)
    ax.set_xlabel("Arricchimento medio di attenzione entrante vs uniforme (%)")
    ax.set_ylabel("Key gene")
    fig.suptitle(
        f"TxT baseline — top {top_count} key per attenzione media",
        fontsize=15,
        color="#202020",
        y=0.97,
    )
    fig.text(
        0.5,
        0.915,
        "Media aritmetica su seed "
        + ", ".join(map(str, seeds))
        + "; esclusi "
        + ", ".join(map(str, excluded_seeds))
        + "; un punto per gene, senza intervalli",
        ha="center",
        va="center",
        fontsize=9,
        color="#4f4f4f",
    )

    ax.axvline(
        0.0,
        color="#777777",
        linewidth=1.0,
        linestyle=(0, (5, 5)),
        zorder=1,
    )
    ax.grid(axis="x", color="#e3e3e3", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", colors="#333333", labelsize=10)
    for spine in ax.spines.values():
        spine.set_color("#b7b7b7")
        spine.set_linewidth(0.8)

    minimum = min(float(values.min()), 0.0)
    maximum = max(float(values.max()), 0.0)
    span = max(maximum - minimum, 0.1)
    label_offset = 0.020 * span
    ax.set_xlim(minimum - 0.035 * span, maximum + 0.18 * span)
    for value, y_position in zip(values, y):
        ax.text(
            value + label_offset,
            y_position,
            f"{value:+.2f}%",
            va="center",
            ha="left",
            fontsize=10,
            color="#202020",
        )

    save_opaque_figure(
        fig,
        output_dir / f"baseline_key_attention_mean_dot_white_top{top_count}",
    )


def generate_dot_white_from_existing(
    output_dir: Path,
    included_seeds: Sequence[int],
    excluded_seeds: Sequence[int],
) -> None:
    """Generate only the new plot variants from existing ranking CSVs."""

    manifest_path = output_dir / "key_attention_mean_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing mean-only manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("included_seeds") != list(map(int, included_seeds)):
        raise ValueError(
            f"Included seeds in {manifest_path} do not match the requested seeds."
        )
    if manifest.get("excluded_seeds") != list(map(int, excluded_seeds)):
        raise ValueError(
            f"Excluded seeds in {manifest_path} do not match the requested seeds."
        )

    for top_count in (10, 20):
        csv_path = output_dir / f"key_attention_mean_top{top_count}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing mean-only ranking: {csv_path}")
        top_frame = pd.read_csv(csv_path)
        plot_mean_dot_white(
            top_frame,
            top_count,
            included_seeds,
            excluded_seeds,
            output_dir,
        )


def _validate_mean_variance_plot_frame(
    frame: pd.DataFrame, top_count: int, path: Path
) -> None:
    required_columns = {
        "mean_attention_rank",
        "gene",
        "seed_count",
        "mean_enrichment_percent_vs_uniform",
        "sample_variance_enrichment_percentage_points_squared",
        "sample_standard_deviation_enrichment_percentage_points",
    }
    missing = required_columns.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} lacks required columns: {sorted(missing)}.")
    if len(frame) != top_count:
        raise ValueError(
            f"Expected {top_count} rows in {path}; found {len(frame)}."
        )
    if frame["gene"].astype(str).duplicated().any():
        raise ValueError(f"Duplicate genes in {path}.")

    ranks = pd.to_numeric(frame["mean_attention_rank"], errors="raise").to_numpy()
    if not np.array_equal(np.sort(ranks.astype(int)), np.arange(1, top_count + 1)):
        raise ValueError(f"Unexpected ranks in {path}.")

    means = pd.to_numeric(
        frame["mean_enrichment_percent_vs_uniform"], errors="raise"
    ).to_numpy(float)
    variances = pd.to_numeric(
        frame["sample_variance_enrichment_percentage_points_squared"],
        errors="raise",
    ).to_numpy(float)
    standard_deviations = pd.to_numeric(
        frame["sample_standard_deviation_enrichment_percentage_points"],
        errors="raise",
    ).to_numpy(float)
    if not (
        np.isfinite(means).all()
        and np.isfinite(variances).all()
        and np.isfinite(standard_deviations).all()
    ):
        raise ValueError(f"Non-finite summary statistics in {path}.")
    if np.any(variances < 0.0) or np.any(standard_deviations < 0.0):
        raise ValueError(f"Negative variance or standard deviation in {path}.")
    if not np.allclose(
        np.square(standard_deviations), variances, rtol=1e-12, atol=1e-14
    ):
        raise ValueError(f"Variance and standard deviation disagree in {path}.")


def plot_mean_variance_dot_white(
    top_frame: pd.DataFrame,
    top_count: int,
    seeds: Sequence[int],
    excluded_seeds: Sequence[int],
    output_dir: Path,
) -> None:
    """Plot mean enrichment with +/-1 sample-SD whiskers on white."""

    source_path = output_dir / f"key_attention_mean_variance_top{top_count}.csv"
    _validate_mean_variance_plot_frame(top_frame, top_count, source_path)
    top = top_frame.copy()
    top["mean_attention_rank"] = pd.to_numeric(
        top["mean_attention_rank"], errors="raise"
    ).astype(int)
    top = top.sort_values("mean_attention_rank", ascending=False, kind="mergesort")

    values = pd.to_numeric(
        top["mean_enrichment_percent_vs_uniform"], errors="raise"
    ).to_numpy(float)
    standard_deviations = pd.to_numeric(
        top["sample_standard_deviation_enrichment_percentage_points"],
        errors="raise",
    ).to_numpy(float)
    labels = top["gene"].astype(str).to_numpy()
    y = np.arange(len(top))

    fig_height = max(5.8, 0.43 * top_count + 2.0)
    fig, ax = plt.subplots(figsize=(11.0, fig_height), facecolor="white")
    ax.set_facecolor("white")
    fig.subplots_adjust(left=0.20, right=0.95, bottom=0.12, top=0.84)

    ax.errorbar(
        values,
        y,
        xerr=standard_deviations,
        fmt="o",
        markersize=8.5,
        markerfacecolor="#67b3ef",
        markeredgecolor="#2f75a8",
        markeredgewidth=0.8,
        ecolor="#4f82ad",
        elinewidth=1.8,
        capsize=5.0,
        capthick=1.4,
        zorder=3,
    )
    ax.set_yticks(y, labels)
    ax.set_xlabel("Arricchimento di attenzione entrante vs uniforme (%)")
    ax.set_ylabel("Key gene")
    fig.suptitle(
        f"TxT baseline — top {top_count} key per attenzione media",
        fontsize=15,
        color="#202020",
        y=0.97,
    )
    fig.text(
        0.5,
        0.915,
        "Punto = media; baffi = ±1 deviazione standard campionaria su seed "
        + ", ".join(map(str, seeds))
        + "; esclusi "
        + ", ".join(map(str, excluded_seeds)),
        ha="center",
        va="center",
        fontsize=9,
        color="#4f4f4f",
    )

    ax.axvline(
        0.0,
        color="#777777",
        linewidth=1.0,
        linestyle=(0, (5, 5)),
        zorder=1,
    )
    ax.grid(axis="x", color="#e3e3e3", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", colors="#333333", labelsize=10)
    for spine in ax.spines.values():
        spine.set_color("#b7b7b7")
        spine.set_linewidth(0.8)

    lower = values - standard_deviations
    upper = values + standard_deviations
    minimum = min(float(lower.min()), 0.0)
    maximum = max(float(upper.max()), 0.0)
    span = max(maximum - minimum, 0.1)
    label_offset = 0.018 * span
    ax.set_xlim(minimum - 0.035 * span, maximum + 0.17 * span)
    for value, upper_endpoint, y_position in zip(values, upper, y):
        ax.text(
            upper_endpoint + label_offset,
            y_position,
            f"{value:+.2f}%",
            va="center",
            ha="left",
            fontsize=10,
            color="#202020",
        )

    save_opaque_figure(
        fig,
        output_dir
        / f"baseline_key_attention_mean_variance_dot_white_top{top_count}",
    )


def _nice_major_tick_step(data_span: float, target_intervals: int = 4) -> float:
    """Return a compact, publication-friendly major-tick interval."""

    if not np.isfinite(data_span) or data_span <= 0.0:
        return 1.0
    rough_step = data_span / float(target_intervals)
    magnitude = 10.0 ** np.floor(np.log10(rough_step))
    normalized = rough_step / magnitude
    for candidate in (1.0, 2.0, 2.5, 5.0, 10.0):
        if normalized <= candidate:
            return candidate * magnitude
    return 10.0 * magnitude


def plot_mean_variance_forest_white_portrait(
    top_frame: pd.DataFrame,
    top_count: int,
    output_dir: Path,
) -> None:
    """Render the reference-style compact portrait forest plot on white.

    The point is the across-seed arithmetic mean; horizontal whiskers are
    +/- one sample standard deviation.  Text labels report the mean itself,
    matching the visual grammar of the supplied reference image.
    """

    source_path = output_dir / f"key_attention_mean_variance_top{top_count}.csv"
    _validate_mean_variance_plot_frame(top_frame, top_count, source_path)
    top = top_frame.copy()
    top["mean_attention_rank"] = pd.to_numeric(
        top["mean_attention_rank"], errors="raise"
    ).astype(int)
    top = top.sort_values("mean_attention_rank", ascending=False, kind="mergesort")

    values = pd.to_numeric(
        top["mean_enrichment_percent_vs_uniform"], errors="raise"
    ).to_numpy(float)
    standard_deviations = pd.to_numeric(
        top["sample_standard_deviation_enrichment_percentage_points"],
        errors="raise",
    ).to_numpy(float)
    labels = top["gene"].astype(str).to_numpy()
    y = np.arange(len(top))

    fig_height = 7.3 if top_count == 10 else 10.6
    fig, ax = plt.subplots(figsize=(5.9, fig_height), facecolor="white")
    ax.set_facecolor("white")
    fig.subplots_adjust(
        left=0.32,
        right=0.985,
        bottom=0.085 if top_count == 20 else 0.105,
        top=0.985,
    )

    ax.errorbar(
        values,
        y,
        xerr=standard_deviations,
        fmt="o",
        markersize=7.6,
        markerfacecolor="#245f8f",
        markeredgecolor="#174b72",
        markeredgewidth=0.75,
        ecolor="#557f9f",
        elinewidth=1.65,
        capsize=4.0,
        capthick=1.25,
        zorder=3,
    )
    ax.set_yticks(y, labels)
    ax.set_ylabel("Key gene", fontsize=11.5, color="#262626", labelpad=28)
    ax.set_xlabel("")
    ax.grid(False)
    ax.axvline(
        0.0,
        color="#7b7b7b",
        linewidth=0.9,
        linestyle=(0, (5, 5)),
        zorder=1,
    )

    lower = values - standard_deviations
    upper = values + standard_deviations
    minimum = min(float(lower.min()), 0.0)
    maximum = max(float(upper.max()), 0.0)
    span = max(maximum - minimum, 0.1)
    label_offset = 0.018 * span
    left_limit = minimum - 0.05 * span
    right_limit = maximum + 0.19 * span
    ax.set_xlim(left_limit, right_limit)
    ax.set_ylim(-0.65, len(top) - 0.35)

    tick_step = _nice_major_tick_step(right_limit - left_limit)
    ax.xaxis.set_major_locator(MultipleLocator(tick_step))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.1f}%"))
    ax.tick_params(
        axis="x",
        colors="#5b5b5b",
        labelsize=10.0,
        length=4.0,
        width=0.8,
        direction="out",
        pad=5.0,
    )
    ax.tick_params(
        axis="y",
        colors="#555555",
        labelsize=10.5,
        length=0.0,
        pad=8.0,
    )
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#b8b8b8")
        spine.set_linewidth(0.8)

    for value, upper_endpoint, y_position in zip(values, upper, y):
        ax.text(
            upper_endpoint + label_offset,
            y_position,
            f"{value:+.2f}%",
            va="center",
            ha="left",
            fontsize=9.5,
            color="#222222",
        )

    save_opaque_figure(
        fig,
        output_dir
        / f"baseline_key_attention_mean_variance_forest_white_portrait_top{top_count}",
    )


def _check_against_existing_mean_only(
    ranking: pd.DataFrame, output_dir: Path
) -> bool:
    """Check that the new ranking exactly reuses the prior mean-only ranking."""

    prior_path = output_dir / "key_attention_mean_all_common_genes.csv"
    if not prior_path.exists():
        return False
    prior = pd.read_csv(prior_path)
    required = {"mean_attention_rank", "gene", "mean_incoming_attention"}
    missing = required.difference(prior.columns)
    if missing:
        raise ValueError(f"{prior_path} lacks required columns: {sorted(missing)}.")
    if len(prior) != len(ranking):
        raise ValueError(
            f"Common-gene count changed: {len(prior)} in {prior_path}, "
            f"{len(ranking)} after recomputation."
        )
    if not np.array_equal(
        prior["gene"].astype(str).to_numpy(), ranking["gene"].to_numpy()
    ):
        raise ValueError("The recomputed gene ranking differs from mean-only output.")
    if not np.array_equal(
        pd.to_numeric(prior["mean_attention_rank"], errors="raise").to_numpy(int),
        ranking["mean_attention_rank"].to_numpy(int),
    ):
        raise ValueError("The recomputed ranks differ from mean-only output.")
    if not np.allclose(
        pd.to_numeric(
            prior["mean_incoming_attention"], errors="raise"
        ).to_numpy(float),
        ranking["mean_incoming_attention"].to_numpy(float),
        rtol=1e-12,
        atol=1e-14,
    ):
        raise ValueError("The recomputed means differ from mean-only output.")
    return True


def write_mean_variance_manifest(
    path: Path,
    run_root: Path,
    output_dir: Path,
    paths: dict[int, Path],
    included_seeds: Sequence[int],
    excluded_seeds: Sequence[int],
    common_gene_count: int,
    per_seed_gene_counts: dict[int, int],
    checked_against_mean_only: bool,
) -> None:
    outputs = {
        "all_common_genes_csv": "key_attention_mean_variance_all_common_genes.csv",
        "top10_csv": "key_attention_mean_variance_top10.csv",
        "top20_csv": "key_attention_mean_variance_top20.csv",
        "top10_png": (
            "baseline_key_attention_mean_variance_dot_white_top10.png"
        ),
        "top10_pdf": (
            "baseline_key_attention_mean_variance_dot_white_top10.pdf"
        ),
        "top20_png": (
            "baseline_key_attention_mean_variance_dot_white_top20.png"
        ),
        "top20_pdf": (
            "baseline_key_attention_mean_variance_dot_white_top20.pdf"
        ),
    }
    manifest = {
        "run_root": str(run_root),
        "output_directory": str(output_dir),
        "included_seeds": list(map(int, included_seeds)),
        "excluded_seeds": list(map(int, excluded_seeds)),
        "included_seed_count": len(included_seeds),
        "analysis_directories": {
            str(seed): str(analysis_dir)
            for seed, analysis_dir in sorted(paths.items())
        },
        "per_seed_gene_counts": {
            str(seed): int(count)
            for seed, count in sorted(per_seed_gene_counts.items())
        },
        "common_gene_count": int(common_gene_count),
        "gene_eligibility": "intersection: gene must be present in every included seed",
        "ranking_metric": (
            "descending arithmetic mean of incoming_macro_mean across included seeds"
        ),
        "seed_weighting": "equal",
        "variance_estimator": "sample variance across seeds; ddof=1; denominator=n-1",
        "plot_point": "mean enrichment in percentage points versus uniform",
        "plot_whiskers": "±1 sample standard deviation across seeds",
        "column_units": {
            "mean_incoming_attention": "dimensionless fold versus uniform",
            "sample_variance_incoming_attention": "dimensionless squared",
            "sample_standard_deviation_incoming_attention": "dimensionless",
            "mean_enrichment_percent_vs_uniform": "percentage points",
            "sample_variance_enrichment_percentage_points_squared": (
                "percentage points squared"
            ),
            "sample_standard_deviation_enrichment_percentage_points": (
                "percentage points"
            ),
        },
        "ranking_checked_against_existing_mean_only": checked_against_mean_only,
        "existing_mean_only_outputs_overwritten": False,
        "plot_background": "opaque white",
        "individual_seed_values_displayed": False,
        "model_variant": "baseline",
        "split": "test",
        "vma_included": False,
        "query_metrics_included": False,
        "outputs": outputs,
    }
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def generate_mean_variance_white(
    run_root: Path,
    output_dir: Path,
    included_seeds: Sequence[int],
    excluded_seeds: Sequence[int],
) -> tuple[pd.DataFrame, int]:
    """Write a separate mean/variance artifact family from raw seed exports."""

    paths = discover_analysis_dirs(run_root, included_seeds)
    frames = load_seed_frames(paths)
    ranking, common_gene_count = build_mean_variance_ranking(frames)
    checked_against_mean_only = _check_against_existing_mean_only(
        ranking, output_dir
    )

    ranking.to_csv(
        output_dir / "key_attention_mean_variance_all_common_genes.csv",
        index=False,
    )
    for top_count in (10, 20):
        top = ranking.head(top_count).copy()
        top.to_csv(
            output_dir / f"key_attention_mean_variance_top{top_count}.csv",
            index=False,
        )
        plot_mean_variance_dot_white(
            top,
            top_count,
            included_seeds,
            excluded_seeds,
            output_dir,
        )

    write_mean_variance_manifest(
        output_dir / "key_attention_mean_variance_manifest.json",
        run_root,
        output_dir,
        paths,
        included_seeds,
        excluded_seeds,
        common_gene_count,
        {seed: len(frame) for seed, frame in frames.items()},
        checked_against_mean_only,
    )
    return ranking, common_gene_count


def _check_against_existing_mean_variance(
    ranking: pd.DataFrame, output_dir: Path
) -> bool:
    """Verify that portrait plots use the already established summaries."""

    prior_path = output_dir / "key_attention_mean_variance_all_common_genes.csv"
    if not prior_path.exists():
        return False
    prior = pd.read_csv(prior_path)
    required_columns = {
        "mean_attention_rank",
        "gene",
        "mean_enrichment_percent_vs_uniform",
        "sample_variance_enrichment_percentage_points_squared",
        "sample_standard_deviation_enrichment_percentage_points",
    }
    missing = required_columns.difference(prior.columns)
    if missing:
        raise ValueError(f"{prior_path} lacks required columns: {sorted(missing)}.")
    if len(prior) != len(ranking):
        raise ValueError(
            f"Common-gene count changed: {len(prior)} in {prior_path}, "
            f"{len(ranking)} after recomputation."
        )
    if not np.array_equal(
        prior["gene"].astype(str).to_numpy(), ranking["gene"].to_numpy()
    ):
        raise ValueError("The recomputed gene order differs from variance output.")
    if not np.array_equal(
        pd.to_numeric(prior["mean_attention_rank"], errors="raise").to_numpy(int),
        ranking["mean_attention_rank"].to_numpy(int),
    ):
        raise ValueError("The recomputed ranks differ from variance output.")

    numeric_columns = [
        "mean_enrichment_percent_vs_uniform",
        "sample_variance_enrichment_percentage_points_squared",
        "sample_standard_deviation_enrichment_percentage_points",
    ]
    for column in numeric_columns:
        if not np.allclose(
            pd.to_numeric(prior[column], errors="raise").to_numpy(float),
            ranking[column].to_numpy(float),
            rtol=1e-12,
            atol=1e-14,
        ):
            raise ValueError(
                f"The recomputed {column} values differ from variance output."
            )
    return True


def write_mean_variance_forest_white_portrait_manifest(
    path: Path,
    run_root: Path,
    output_dir: Path,
    paths: dict[int, Path],
    included_seeds: Sequence[int],
    excluded_seeds: Sequence[int],
    common_gene_count: int,
    per_seed_gene_counts: dict[int, int],
    checked_against_existing_variance: bool,
) -> None:
    """Document the dedicated portrait artifact family."""

    outputs = {
        "top10_png": (
            "baseline_key_attention_mean_variance_forest_white_portrait_top10.png"
        ),
        "top10_pdf": (
            "baseline_key_attention_mean_variance_forest_white_portrait_top10.pdf"
        ),
        "top20_png": (
            "baseline_key_attention_mean_variance_forest_white_portrait_top20.png"
        ),
        "top20_pdf": (
            "baseline_key_attention_mean_variance_forest_white_portrait_top20.pdf"
        ),
    }
    manifest = {
        "run_root": str(run_root),
        "output_directory": str(output_dir),
        "included_seeds": list(map(int, included_seeds)),
        "excluded_seeds": list(map(int, excluded_seeds)),
        "included_seed_count": len(included_seeds),
        "analysis_directories": {
            str(seed): str(analysis_dir)
            for seed, analysis_dir in sorted(paths.items())
        },
        "per_seed_gene_counts": {
            str(seed): int(count)
            for seed, count in sorted(per_seed_gene_counts.items())
        },
        "common_gene_count": int(common_gene_count),
        "gene_eligibility": "intersection: gene must be present in every included seed",
        "ranking_metric": (
            "descending arithmetic mean of incoming_macro_mean across included seeds"
        ),
        "seed_weighting": "equal",
        "variance_estimator": "sample variance across seeds; ddof=1; denominator=n-1",
        "plot_point": "mean enrichment in percentage points versus uniform",
        "plot_whiskers": "±1 sample standard deviation across seeds",
        "plot_value_labels": "mean enrichment in percentage points versus uniform",
        "plot_style": {
            "layout": "portrait; compact horizontal axis",
            "background": "opaque white",
            "marker_fill": "#245f8f",
            "title": False,
            "subtitle": False,
            "grid": False,
            "complete_axis_box": True,
            "zero_reference": "vertical dashed line",
            "x_tick_format": "one decimal place plus percent sign",
        },
        "ranking_checked_against_existing_mean_variance": (
            checked_against_existing_variance
        ),
        "existing_csv_outputs_overwritten": False,
        "existing_plot_outputs_overwritten": False,
        "model_variant": "baseline",
        "split": "test",
        "vma_included": False,
        "query_metrics_included": False,
        "outputs": outputs,
    }
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def generate_mean_variance_forest_white_portrait(
    run_root: Path,
    output_dir: Path,
    included_seeds: Sequence[int],
    excluded_seeds: Sequence[int],
) -> tuple[pd.DataFrame, int]:
    """Recompute summaries and write only compact portrait plot variants."""

    paths = discover_analysis_dirs(run_root, included_seeds)
    frames = load_seed_frames(paths)
    ranking, common_gene_count = build_mean_variance_ranking(frames)
    checked_against_existing_variance = _check_against_existing_mean_variance(
        ranking, output_dir
    )

    for top_count in (10, 20):
        plot_mean_variance_forest_white_portrait(
            ranking.head(top_count).copy(),
            top_count,
            output_dir,
        )

    write_mean_variance_forest_white_portrait_manifest(
        output_dir / "key_attention_mean_variance_forest_white_portrait_manifest.json",
        run_root,
        output_dir,
        paths,
        included_seeds,
        excluded_seeds,
        common_gene_count,
        {seed: len(frame) for seed, frame in frames.items()},
        checked_against_existing_variance,
    )
    return ranking, common_gene_count


def _validate_iqr_candle_frame(
    frame: pd.DataFrame,
    expected_row_count: int,
    path: Path,
) -> None:
    """Validate the five-number summaries used by an IQR candle plot/table."""

    statistic_columns = [
        "minimum_enrichment_percentage_points",
        "first_quartile_enrichment_percentage_points",
        "median_enrichment_percentage_points",
        "third_quartile_enrichment_percentage_points",
        "maximum_enrichment_percentage_points",
    ]
    required_columns = {
        "mean_attention_rank",
        "gene",
        "seed_count",
        "mean_incoming_attention",
        "mean_enrichment_percent_vs_uniform",
        "iqr_enrichment_percentage_points",
        *statistic_columns,
    }
    missing = required_columns.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} lacks required columns: {sorted(missing)}.")
    if len(frame) != expected_row_count:
        raise ValueError(
            f"Expected {expected_row_count} rows in {path}; found {len(frame)}."
        )
    if frame["gene"].astype(str).duplicated().any():
        raise ValueError(f"Duplicate genes in {path}.")

    raw_seed_columns = [
        column
        for column in frame.columns
        if column.startswith("seed_")
        and column.endswith("_enrichment_percentage_points")
    ]
    if not raw_seed_columns:
        raise ValueError(f"No raw seed enrichment columns in {path}.")
    seed_counts = pd.to_numeric(frame["seed_count"], errors="raise").to_numpy(int)
    if not np.all(seed_counts == len(raw_seed_columns)):
        raise ValueError(
            f"Raw seed-column count disagrees with seed_count in {path}."
        )
    raw_seed_values = frame.loc[:, raw_seed_columns].apply(
        pd.to_numeric, errors="raise"
    ).to_numpy(float)
    if not np.isfinite(raw_seed_values).all():
        raise ValueError(f"Non-finite raw seed values in {path}.")

    ranks = pd.to_numeric(frame["mean_attention_rank"], errors="raise").to_numpy()
    if not np.array_equal(
        np.sort(ranks.astype(int)), np.arange(1, expected_row_count + 1)
    ):
        raise ValueError(f"Unexpected ranks in {path}.")

    statistics = np.column_stack(
        [
            pd.to_numeric(frame[column], errors="raise").to_numpy(float)
            for column in statistic_columns
        ]
    )
    if not np.isfinite(statistics).all():
        raise ValueError(f"Non-finite five-number summaries in {path}.")
    if np.any(np.diff(statistics, axis=1) < -1e-12):
        raise ValueError(f"Five-number summaries are not ordered in {path}.")
    expected_statistics = np.quantile(
        raw_seed_values,
        np.array([0.0, 0.25, 0.5, 0.75, 1.0]),
        axis=1,
        method="linear",
    ).T
    if not np.allclose(
        statistics, expected_statistics, rtol=1e-12, atol=1e-12
    ):
        raise ValueError(
            f"Five-number summaries disagree with raw seeds in {path}."
        )

    means = pd.to_numeric(
        frame["mean_enrichment_percent_vs_uniform"], errors="raise"
    ).to_numpy(float)
    if not np.allclose(
        means, raw_seed_values.mean(axis=1), rtol=1e-12, atol=1e-12
    ):
        raise ValueError(f"Mean values disagree with raw seeds in {path}.")

    iqr = pd.to_numeric(
        frame["iqr_enrichment_percentage_points"], errors="raise"
    ).to_numpy(float)
    expected_iqr = statistics[:, 3] - statistics[:, 1]
    if np.any(iqr < -1e-12) or not np.allclose(
        iqr, expected_iqr, rtol=1e-12, atol=1e-12
    ):
        raise ValueError(f"IQR values disagree with Q3 - Q1 in {path}.")


def plot_iqr_candle_white_portrait(
    top_frame: pd.DataFrame,
    top_count: int,
    output_dir: Path,
) -> None:
    """Render compact candles plus raw seeds and mean on opaque white."""

    source_path = output_dir / f"key_attention_iqr_candle_top{top_count}.csv"
    _validate_iqr_candle_frame(top_frame, top_count, source_path)
    top = top_frame.copy()
    top["mean_attention_rank"] = pd.to_numeric(
        top["mean_attention_rank"], errors="raise"
    ).astype(int)
    top = top.sort_values("mean_attention_rank", ascending=False, kind="mergesort")

    minimum = pd.to_numeric(
        top["minimum_enrichment_percentage_points"], errors="raise"
    ).to_numpy(float)
    q1 = pd.to_numeric(
        top["first_quartile_enrichment_percentage_points"], errors="raise"
    ).to_numpy(float)
    median = pd.to_numeric(
        top["median_enrichment_percentage_points"], errors="raise"
    ).to_numpy(float)
    q3 = pd.to_numeric(
        top["third_quartile_enrichment_percentage_points"], errors="raise"
    ).to_numpy(float)
    maximum = pd.to_numeric(
        top["maximum_enrichment_percentage_points"], errors="raise"
    ).to_numpy(float)
    mean = pd.to_numeric(
        top["mean_enrichment_percent_vs_uniform"], errors="raise"
    ).to_numpy(float)
    raw_seed_columns = sorted(
        (
            column
            for column in top.columns
            if column.startswith("seed_")
            and column.endswith("_enrichment_percentage_points")
        ),
        key=lambda column: int(column.split("_")[1]),
    )
    raw_seed_values = top.loc[:, raw_seed_columns].apply(
        pd.to_numeric, errors="raise"
    ).to_numpy(float)
    labels = top["gene"].astype(str).to_numpy()
    y = np.arange(len(top), dtype=float)

    fig_height = 7.3 if top_count == 10 else 10.6
    fig, ax = plt.subplots(figsize=(5.9, fig_height), facecolor="white")
    ax.set_facecolor("white")
    fig.subplots_adjust(
        left=0.32,
        right=0.985,
        bottom=0.085 if top_count == 20 else 0.105,
        top=0.985,
    )

    whisker_color = "#557f9f"
    box_face_color = "#8eb5cf"
    box_edge_color = "#426f91"
    median_color = "#173f5c"
    mean_color = "#0f3552"
    raw_seed_color = "#2f678f"
    box_height = 0.34
    cap_half_height = 0.13

    ax.hlines(
        y,
        minimum,
        maximum,
        color=whisker_color,
        linewidth=1.45,
        zorder=2,
    )
    ax.vlines(
        minimum,
        y - cap_half_height,
        y + cap_half_height,
        color=whisker_color,
        linewidth=1.15,
        zorder=2,
    )
    ax.vlines(
        maximum,
        y - cap_half_height,
        y + cap_half_height,
        color=whisker_color,
        linewidth=1.15,
        zorder=2,
    )
    ax.barh(
        y,
        q3 - q1,
        left=q1,
        height=box_height,
        color=box_face_color,
        edgecolor=box_edge_color,
        linewidth=1.0,
        zorder=3,
    )
    seed_jitter = np.linspace(-0.115, 0.115, raw_seed_values.shape[1])
    for seed_index, jitter in enumerate(seed_jitter):
        ax.scatter(
            raw_seed_values[:, seed_index],
            y + jitter,
            s=15,
            color=raw_seed_color,
            edgecolors="none",
            alpha=0.48,
            zorder=4,
        )
    ax.vlines(
        median,
        y - box_height / 2.0,
        y + box_height / 2.0,
        color=median_color,
        linewidth=1.5,
        zorder=5,
    )
    ax.scatter(
        mean,
        y,
        s=38,
        marker="D",
        color=mean_color,
        edgecolors="white",
        linewidths=0.55,
        zorder=6,
    )

    ax.set_yticks(y, labels)
    ax.set_ylabel("Key gene", fontsize=11.5, color="#262626", labelpad=28)
    ax.set_xlabel("")
    ax.grid(False)
    ax.axvline(
        0.0,
        color="#7b7b7b",
        linewidth=0.9,
        linestyle=(0, (5, 5)),
        zorder=1,
    )

    data_minimum = min(float(minimum.min()), 0.0)
    data_maximum = max(float(maximum.max()), 0.0)
    span = max(data_maximum - data_minimum, 0.1)
    label_offset = 0.018 * span
    left_limit = data_minimum - 0.05 * span
    right_limit = data_maximum + 0.19 * span
    ax.set_xlim(left_limit, right_limit)
    ax.set_ylim(-0.65, len(top) - 0.35)

    tick_step = _nice_major_tick_step(right_limit - left_limit)
    ax.xaxis.set_major_locator(MultipleLocator(tick_step))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.1f}%"))
    ax.tick_params(
        axis="x",
        colors="#5b5b5b",
        labelsize=10.0,
        length=4.0,
        width=0.8,
        direction="out",
        pad=5.0,
    )
    ax.tick_params(
        axis="y",
        colors="#555555",
        labelsize=10.5,
        length=0.0,
        pad=8.0,
    )
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#b8b8b8")
        spine.set_linewidth(0.8)

    for mean_value, maximum_value, y_position in zip(mean, maximum, y):
        ax.text(
            maximum_value + label_offset,
            y_position,
            f"{mean_value:+.2f}%",
            va="center",
            ha="left",
            fontsize=9.5,
            color="#222222",
        )

    save_opaque_figure(
        fig,
        output_dir
        / f"baseline_key_attention_iqr_candle_white_portrait_top{top_count}",
    )


def write_iqr_candle_white_portrait_manifest(
    path: Path,
    run_root: Path,
    output_dir: Path,
    paths: dict[int, Path],
    included_seeds: Sequence[int],
    excluded_seeds: Sequence[int],
    common_gene_count: int,
    per_seed_gene_counts: dict[int, int],
    checked_against_mean_only: bool,
) -> None:
    """Document the separate IQR-candle artifact family."""

    outputs = {
        "all_common_genes_csv": "key_attention_iqr_candle_all_common_genes.csv",
        "top10_csv": "key_attention_iqr_candle_top10.csv",
        "top20_csv": "key_attention_iqr_candle_top20.csv",
        "top10_png": "baseline_key_attention_iqr_candle_white_portrait_top10.png",
        "top10_pdf": "baseline_key_attention_iqr_candle_white_portrait_top10.pdf",
        "top20_png": "baseline_key_attention_iqr_candle_white_portrait_top20.png",
        "top20_pdf": "baseline_key_attention_iqr_candle_white_portrait_top20.pdf",
    }
    manifest = {
        "run_root": str(run_root),
        "output_directory": str(output_dir),
        "included_seeds": list(map(int, included_seeds)),
        "excluded_seeds": list(map(int, excluded_seeds)),
        "included_seed_count": len(included_seeds),
        "analysis_directories": {
            str(seed): str(analysis_dir)
            for seed, analysis_dir in sorted(paths.items())
        },
        "per_seed_gene_counts": {
            str(seed): int(count)
            for seed, count in sorted(per_seed_gene_counts.items())
        },
        "common_gene_count": int(common_gene_count),
        "gene_eligibility": "intersection: gene must be present in every included seed",
        "ranking_metric": (
            "descending arithmetic mean of incoming_macro_mean across included seeds"
        ),
        "seed_weighting": "equal",
        "quantile_method": (
            "linear interpolation; NumPy quantile method='linear' "
            "(Hyndman-Fan type 7)"
        ),
        "candle_box": "Q1 to Q3; width equals the interquartile range",
        "candle_median": "vertical line within the Q1-Q3 box",
        "mean_marker": "dark diamond; ranking and numeric labels use this mean",
        "candle_whiskers": "observed minimum to observed maximum across seeds",
        "raw_seed_points": (
            "all included seed values; small semi-transparent points with "
            "deterministic vertical jitter in ascending seed order"
        ),
        "plot_value_labels": "mean enrichment in percentage points versus uniform",
        "caption_it": (
            "Distribuzione dell'arricchimento di attenzione key tra gli 8 seed: "
            "box = Q1-Q3 (quantili lineari/type 7), linea = mediana, baffi = "
            "minimo-massimo osservati, punti semitrasparenti = singoli seed, "
            "rombo scuro ed etichetta = media; geni ordinati per media."
        ),
        "raw_seed_csv_columns": (
            "seed_<id>_enrichment_percentage_points for every included seed"
        ),
        "column_units": {
            "incoming_attention_columns": "dimensionless fold versus uniform",
            "enrichment_columns": "percentage points versus uniform",
        },
        "plot_style": {
            "layout": "portrait; compact horizontal axis",
            "background": "opaque white",
            "box_fill": "#8eb5cf",
            "median_line": "#173f5c",
            "mean_diamond_fill": "#0f3552",
            "raw_seed_point_fill": "#2f678f at alpha 0.48",
            "title": False,
            "subtitle": False,
            "grid": False,
            "complete_axis_box": True,
            "zero_reference": "vertical dashed line",
            "x_tick_format": "one decimal place plus percent sign",
        },
        "ranking_checked_against_existing_mean_only": checked_against_mean_only,
        "individual_seed_values_displayed": True,
        "existing_csv_outputs_overwritten": False,
        "existing_plot_outputs_overwritten": False,
        "model_variant": "baseline",
        "split": "test",
        "vma_included": False,
        "query_metrics_included": False,
        "outputs": outputs,
    }
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def generate_iqr_candle_white_portrait(
    run_root: Path,
    output_dir: Path,
    included_seeds: Sequence[int],
    excluded_seeds: Sequence[int],
) -> tuple[pd.DataFrame, int]:
    """Write a new IQR-candle artifact family without replacing prior outputs."""

    paths = discover_analysis_dirs(run_root, included_seeds)
    frames = load_seed_frames(paths)
    ranking, common_gene_count = build_iqr_candle_ranking(frames)
    checked_against_mean_only = _check_against_existing_mean_only(ranking, output_dir)

    all_csv_path = output_dir / "key_attention_iqr_candle_all_common_genes.csv"
    _validate_iqr_candle_frame(ranking, common_gene_count, all_csv_path)
    ranking.to_csv(all_csv_path, index=False)
    for top_count in (10, 20):
        top = ranking.head(top_count).copy()
        top.to_csv(
            output_dir / f"key_attention_iqr_candle_top{top_count}.csv",
            index=False,
        )
        plot_iqr_candle_white_portrait(top, top_count, output_dir)

    write_iqr_candle_white_portrait_manifest(
        output_dir / "key_attention_iqr_candle_white_portrait_manifest.json",
        run_root,
        output_dir,
        paths,
        included_seeds,
        excluded_seeds,
        common_gene_count,
        {seed: len(frame) for seed, frame in frames.items()},
        checked_against_mean_only,
    )
    return ranking, common_gene_count


def write_manifest(
    path: Path,
    run_root: Path,
    output_dir: Path,
    paths: dict[int, Path],
    included_seeds: Sequence[int],
    excluded_seeds: Sequence[int],
    common_gene_count: int,
    per_seed_gene_counts: dict[int, int],
) -> None:
    manifest = {
        "run_root": str(run_root),
        "output_directory": str(output_dir),
        "included_seeds": list(map(int, included_seeds)),
        "excluded_seeds": list(map(int, excluded_seeds)),
        "included_seed_count": len(included_seeds),
        "analysis_directories": {
            str(seed): str(path) for seed, path in sorted(paths.items())
        },
        "per_seed_gene_counts": {
            str(seed): int(count) for seed, count in sorted(per_seed_gene_counts.items())
        },
        "common_gene_count": int(common_gene_count),
        "gene_eligibility": "intersection: gene must be present in every included seed",
        "ranking_metric": (
            "descending arithmetic mean of incoming_macro_mean across included seeds"
        ),
        "display_metric": "100 * (mean_incoming_attention - 1); uniform reference = 0%",
        "seed_weighting": "equal",
        "uncertainty_displayed": False,
        "individual_seed_values_displayed": False,
        "frequency_displayed": False,
        "model_variant": "baseline",
        "split": "test",
        "vma_included": False,
        "query_metrics_included": False,
    }
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    cli = build_parser().parse_args(argv)
    selected_plot_only_modes = sum(
        bool(mode)
        for mode in (
            cli.dot_white_only,
            cli.mean_variance_white_only,
            cli.mean_variance_forest_white_portrait_only,
            cli.iqr_candle_white_portrait_only,
        )
    )
    if selected_plot_only_modes > 1:
        raise ValueError(
            "--dot-white-only, --mean-variance-white-only, and "
            "--mean-variance-forest-white-portrait-only, and "
            "--iqr-candle-white-portrait-only are mutually exclusive."
        )
    included_seeds = list(map(int, cli.included_seeds))
    excluded_seeds = list(map(int, cli.excluded_seeds))
    if len(set(included_seeds)) != len(included_seeds):
        raise ValueError("--included-seeds must not contain duplicates.")
    if len(set(excluded_seeds)) != len(excluded_seeds):
        raise ValueError("--excluded-seeds must not contain duplicates.")
    overlap = sorted(set(included_seeds).intersection(excluded_seeds))
    if overlap:
        raise ValueError(f"Seeds cannot be both included and excluded: {overlap}.")

    run_root = cli.run_root.resolve()
    output_dir = (
        cli.output_dir.resolve()
        if cli.output_dir is not None
        else run_root / DEFAULT_OUTPUT_DIRNAME
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    if cli.dot_white_only:
        generate_dot_white_from_existing(
            output_dir,
            included_seeds,
            excluded_seeds,
        )
        print(f"White mean-dot plots written to: {output_dir}")
        return 0

    if cli.mean_variance_white_only:
        ranking, common_gene_count = generate_mean_variance_white(
            run_root,
            output_dir,
            included_seeds,
            excluded_seeds,
        )
        print(f"White mean/variance plots and tables written to: {output_dir}")
        print(f"Common genes ranked: {common_gene_count}")
        columns = [
            "mean_attention_rank",
            "gene",
            "mean_enrichment_percent_vs_uniform",
            "sample_standard_deviation_enrichment_percentage_points",
            "sample_variance_enrichment_percentage_points_squared",
        ]
        print(ranking.loc[:, columns].head(20).to_string(index=False))
        return 0

    if cli.mean_variance_forest_white_portrait_only:
        ranking, common_gene_count = generate_mean_variance_forest_white_portrait(
            run_root,
            output_dir,
            included_seeds,
            excluded_seeds,
        )
        print(f"Portrait mean/variance forest plots written to: {output_dir}")
        print(f"Common genes ranked: {common_gene_count}")
        columns = [
            "mean_attention_rank",
            "gene",
            "mean_enrichment_percent_vs_uniform",
            "sample_standard_deviation_enrichment_percentage_points",
        ]
        print(ranking.loc[:, columns].head(20).to_string(index=False))
        return 0

    if cli.iqr_candle_white_portrait_only:
        ranking, common_gene_count = generate_iqr_candle_white_portrait(
            run_root,
            output_dir,
            included_seeds,
            excluded_seeds,
        )
        print(f"Portrait IQR-candle plots and tables written to: {output_dir}")
        print(f"Common genes ranked: {common_gene_count}")
        columns = [
            "mean_attention_rank",
            "gene",
            "mean_enrichment_percent_vs_uniform",
            "minimum_enrichment_percentage_points",
            "first_quartile_enrichment_percentage_points",
            "median_enrichment_percentage_points",
            "third_quartile_enrichment_percentage_points",
            "maximum_enrichment_percentage_points",
            "iqr_enrichment_percentage_points",
        ]
        print(ranking.loc[:, columns].head(20).to_string(index=False))
        return 0

    paths = discover_analysis_dirs(run_root, included_seeds)
    frames = load_seed_frames(paths)
    ranking, common_gene_count = build_mean_ranking(frames)

    ranking.to_csv(output_dir / "key_attention_mean_all_common_genes.csv", index=False)
    ranking.head(10).to_csv(output_dir / "key_attention_mean_top10.csv", index=False)
    ranking.head(20).to_csv(output_dir / "key_attention_mean_top20.csv", index=False)
    plot_mean_only(ranking, 10, included_seeds, excluded_seeds, output_dir)
    plot_mean_only(ranking, 20, included_seeds, excluded_seeds, output_dir)
    plot_mean_dot_white(
        ranking.head(10), 10, included_seeds, excluded_seeds, output_dir
    )
    plot_mean_dot_white(
        ranking.head(20), 20, included_seeds, excluded_seeds, output_dir
    )
    write_manifest(
        output_dir / "key_attention_mean_manifest.json",
        run_root,
        output_dir,
        paths,
        included_seeds,
        excluded_seeds,
        common_gene_count,
        {seed: len(frame) for seed, frame in frames.items()},
    )

    print(f"Mean-only key-attention outputs written to: {output_dir}")
    print(f"Common genes ranked: {common_gene_count}")
    print(ranking.head(20).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
