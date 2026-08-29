#!/usr/bin/env python3
"""Plot baseline TxT key-gene incoming-attention enrichment.

This post-processing step intentionally ignores the VMA branch and query-side
metrics.  Genes are ranked by the baseline dense-attention metric

    G * mean_over_queries(A[query, key])

where 1 is uniform attention.  Figures display the equivalent percentage
change from uniform with subject-level bootstrap confidence intervals.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ANALYSIS_DIR = (
    ROOT
    / "results"
    / "paper_comparison"
    / "txt_volumetric_mps_seed101_beta1p5_lossstop"
    / "seed_101"
    / "baseline"
    / "attention_analysis"
    / "test"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot top baseline TxT key genes by incoming-attention enrichment."
    )
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=DEFAULT_ANALYSIS_DIR,
        help="Baseline attention-analysis directory.",
    )
    parser.add_argument("--top-counts", type=int, nargs="+", default=[10, 20])
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=101)
    return parser


def bootstrap_intervals(
    values: np.ndarray,
    iterations: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if iterations <= 0:
        means = np.nanmean(values, axis=0)
        return means, means
    rng = np.random.default_rng(seed)
    bootstrap_means = np.empty((iterations, values.shape[1]), dtype=np.float64)
    for iteration in range(iterations):
        sample_indices = rng.integers(0, values.shape[0], size=values.shape[0])
        bootstrap_means[iteration] = np.nanmean(values[sample_indices], axis=0)
    return (
        np.nanpercentile(bootstrap_means, 2.5, axis=0),
        np.nanpercentile(bootstrap_means, 97.5, axis=0),
    )


def build_plot_table(
    ranking: pd.DataFrame,
    incoming_by_subject: np.ndarray,
    gene_names: np.ndarray,
    top_count: int,
    bootstrap_iterations: int,
    seed: int,
) -> pd.DataFrame:
    top = ranking.nsmallest(top_count, "key_rank").sort_values("key_rank")
    indices = top["gene_index"].to_numpy(dtype=np.int64)
    if not np.array_equal(top["gene"].astype(str).to_numpy(), gene_names[indices]):
        raise ValueError("Ranking genes are not aligned with the saved metric gene order.")
    selected = incoming_by_subject[:, indices]
    means = np.nanmean(selected, axis=0)
    lower, upper = bootstrap_intervals(selected, bootstrap_iterations, seed)
    return pd.DataFrame(
        {
            "rank": top["key_rank"].to_numpy(dtype=np.int64),
            "gene": gene_names[indices],
            "incoming_enrichment": means,
            "enrichment_percent": 100.0 * (means - 1.0),
            "ci95_lower_percent": 100.0 * (lower - 1.0),
            "ci95_upper_percent": 100.0 * (upper - 1.0),
            "subjects": incoming_by_subject.shape[0],
        }
    )


def plot_table(table: pd.DataFrame, output_stem: Path) -> None:
    display = table.sort_values("rank", ascending=False).reset_index(drop=True)
    values = display["enrichment_percent"].to_numpy()
    lower = display["ci95_lower_percent"].to_numpy()
    upper = display["ci95_upper_percent"].to_numpy()
    y = np.arange(len(display))

    fig, ax = plt.subplots(figsize=(9.4, max(5.4, 0.39 * len(display) + 1.8)))
    ax.errorbar(
        values,
        y,
        xerr=np.vstack([values - lower, upper - values]),
        fmt="o",
        markersize=6.5,
        color="#2f5597",
        ecolor="#9aabc2",
        elinewidth=1.4,
        capsize=2.5,
        zorder=3,
    )
    ax.axvline(0.0, color="#6f6f6f", linestyle="--", linewidth=1.1, zorder=1)
    for row_index, value in enumerate(values):
        ax.annotate(
            f"{value:+.2f}%",
            (upper[row_index], row_index),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
            color="#333333",
        )

    ax.set_yticks(y, display["gene"])
    ax.set_xlabel("Arricchimento di attenzione entrante rispetto all’uniforme (%)")
    ax.set_ylabel("Key gene")
    ax.set_title(
        f"TxT baseline — top {len(display)} key per attenzione entrante",
        pad=18,
    )
    ax.text(
        0.0,
        1.01,
        "Media sui 2 head; IC bootstrap 95% sui 143 soggetti del test set",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        color="#555555",
    )
    ax.grid(axis="x", color="#dddddd", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.margins(y=0.06)
    fig.text(
        0.01,
        0.01,
        "IC descrittivi: i geni sono stati selezionati sullo stesso test set.",
        fontsize=8.5,
        color="#555555",
    )
    fig.savefig(output_stem.with_suffix(".png"), dpi=240, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> int:
    cli = build_parser().parse_args(argv)
    if any(count <= 0 for count in cli.top_counts):
        raise ValueError("--top-counts values must be positive.")
    if cli.bootstrap_iterations < 0:
        raise ValueError("--bootstrap-iterations cannot be negative.")

    analysis_dir = cli.analysis_dir.resolve()
    ranking = pd.read_csv(analysis_dir / "gene_attention_ranking.csv")
    with np.load(analysis_dir / "sample_gene_attention_metrics.npz") as metrics:
        incoming = metrics["incoming_enrichment"].astype(np.float64)
        gene_names = metrics["gene_names"].astype(str)
    if incoming.ndim != 3:
        raise ValueError("Expected incoming_enrichment[sample, head, gene].")
    incoming_by_subject = np.nanmean(incoming, axis=1)

    plots_dir = analysis_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    for top_count in sorted(set(cli.top_counts)):
        table = build_plot_table(
            ranking,
            incoming_by_subject,
            gene_names,
            top_count,
            cli.bootstrap_iterations,
            cli.seed,
        )
        stem = plots_dir / f"baseline_key_attention_enrichment_top{top_count}"
        table.to_csv(stem.with_suffix(".csv"), index=False)
        plot_table(table, stem)
        print(f"Written: {stem}.png/.pdf/.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
