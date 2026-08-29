#!/usr/bin/env python3
"""Build a consensus and stability audit from TxT attention-ranking exports."""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.scripts.paper_comparison.txt_volumetric.common import write_json  # noqa: E402


TOP_K_VALUES = (10, 20, 50, 100, 200)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize multiple gene_attention_ranking.csv exports using percentile ranks, "
            "top-k frequency, Spearman concordance and Jaccard overlap."
        )
    )
    parser.add_argument("--analysis-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--labels", nargs="+", default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def load_ranking(directory: Path, label: str) -> pd.DataFrame:
    path = directory / "gene_attention_ranking.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing attention ranking: {path}")
    frame = pd.read_csv(path)
    required = {
        "gene",
        "key_rank",
        "query_rank",
        "incoming_enrichment_mean",
        "query_specificity_mean",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    if frame["gene"].astype(str).duplicated().any():
        raise ValueError(f"{path} contains duplicate genes.")
    n_genes = len(frame)
    denominator = max(n_genes - 1, 1)
    result = frame[
        [
            "gene",
            "key_rank",
            "query_rank",
            "incoming_enrichment_mean",
            "query_specificity_mean",
        ]
    ].copy()
    result["run_label"] = label
    result["n_genes_in_run"] = n_genes
    result["key_percentile_score"] = 1.0 - (result["key_rank"] - 1.0) / denominator
    result["query_percentile_score"] = 1.0 - (result["query_rank"] - 1.0) / denominator
    for top_k in TOP_K_VALUES:
        result[f"key_top{top_k}"] = result["key_rank"] <= top_k
        result[f"query_top{top_k}"] = result["query_rank"] <= top_k
    return result


def build_consensus(rankings: Sequence[pd.DataFrame]) -> pd.DataFrame:
    combined = pd.concat(rankings, ignore_index=True)
    run_count = combined["run_label"].nunique()
    rows: list[dict[str, object]] = []
    for gene, group in combined.groupby("gene", sort=True):
        row: dict[str, object] = {
            "gene": str(gene),
            "runs_included": int(group["run_label"].nunique()),
            "run_count": int(run_count),
            "inclusion_frequency": float(group["run_label"].nunique() / run_count),
            "key_percentile_mean": float(group["key_percentile_score"].mean()),
            "key_percentile_median": float(group["key_percentile_score"].median()),
            "query_percentile_mean": float(group["query_percentile_score"].mean()),
            "query_percentile_median": float(group["query_percentile_score"].median()),
            "key_rank_mean": float(group["key_rank"].mean()),
            "key_rank_median": float(group["key_rank"].median()),
            "key_rank_iqr": float(group["key_rank"].quantile(0.75) - group["key_rank"].quantile(0.25)),
            "query_rank_mean": float(group["query_rank"].mean()),
            "query_rank_median": float(group["query_rank"].median()),
            "query_rank_iqr": float(
                group["query_rank"].quantile(0.75) - group["query_rank"].quantile(0.25)
            ),
            "incoming_enrichment_mean_across_runs": float(
                group["incoming_enrichment_mean"].mean()
            ),
            "query_specificity_mean_across_runs": float(
                group["query_specificity_mean"].mean()
            ),
        }
        for top_k in TOP_K_VALUES:
            row[f"key_top{top_k}_frequency"] = float(group[f"key_top{top_k}"].sum() / run_count)
            row[f"query_top{top_k}_frequency"] = float(
                group[f"query_top{top_k}"].sum() / run_count
            )
        rows.append(row)
    consensus = pd.DataFrame(rows)
    # Treat absence from a run's selected-gene universe as zero percentile
    # support.  Otherwise a gene selected in only one seed at rank 1 could
    # outrank a consistently high gene seen in every seed.
    consensus["key_consensus_score"] = (
        consensus["key_percentile_mean"] * consensus["inclusion_frequency"]
    )
    consensus["query_consensus_score"] = (
        consensus["query_percentile_mean"] * consensus["inclusion_frequency"]
    )
    consensus = consensus.sort_values(
        ["key_consensus_score", "key_top50_frequency", "key_percentile_mean", "gene"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    consensus["consensus_key_rank"] = np.arange(1, len(consensus) + 1)
    query_order = consensus.sort_values(
        ["query_consensus_score", "query_top50_frequency", "query_percentile_mean", "gene"],
        ascending=[False, False, False, True],
    ).index
    consensus["consensus_query_rank"] = 0
    consensus.loc[query_order, "consensus_query_rank"] = np.arange(1, len(consensus) + 1)
    return consensus


def build_stability(rankings: Sequence[pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    by_label = {str(frame["run_label"].iloc[0]): frame.set_index("gene") for frame in rankings}
    for left_label, right_label in itertools.combinations(by_label, 2):
        left = by_label[left_label]
        right = by_label[right_label]
        common = left.index.intersection(right.index)
        for metric, rank_column in [("key", "key_rank"), ("query", "query_rank")]:
            rho, p_value = stats.spearmanr(
                left.loc[common, rank_column], right.loc[common, rank_column]
            )
            rows.append(
                {
                    "left_run": left_label,
                    "right_run": right_label,
                    "metric": metric,
                    "statistic": "spearman",
                    "top_k": np.nan,
                    "common_genes": int(len(common)),
                    "value": float(rho),
                    "p_value": float(p_value),
                }
            )
            for top_k in TOP_K_VALUES:
                left_top = set(left.nsmallest(top_k, rank_column).index)
                right_top = set(right.nsmallest(top_k, rank_column).index)
                union = left_top | right_top
                jaccard = len(left_top & right_top) / len(union) if union else np.nan
                rows.append(
                    {
                        "left_run": left_label,
                        "right_run": right_label,
                        "metric": metric,
                        "statistic": "jaccard",
                        "top_k": top_k,
                        "common_genes": int(len(common)),
                        "value": float(jaccard),
                        "p_value": np.nan,
                    }
                )
    return pd.DataFrame(rows)


def _save_figure(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".png"), dpi=240, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_pairwise_percentiles(
    rankings: Sequence[pd.DataFrame],
    consensus: pd.DataFrame,
    output_dir: Path,
) -> None:
    if len(rankings) != 2:
        return
    left = rankings[0].set_index("gene")
    right = rankings[1].set_index("gene")
    left_label = str(rankings[0]["run_label"].iloc[0])
    right_label = str(rankings[1]["run_label"].iloc[0])
    common = left.index.intersection(right.index)
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 6.0))
    for ax, metric in zip(axes, ["key", "query"]):
        column = f"{metric}_percentile_score"
        rho, _ = stats.spearmanr(left.loc[common, column], right.loc[common, column])
        ax.scatter(
            left.loc[common, column],
            right.loc[common, column],
            s=12,
            alpha=0.35,
            color="#4472c4",
            linewidths=0,
        )
        ax.plot([0, 1], [0, 1], color="#777777", linewidth=0.8, linestyle="--")
        ax.set_xlabel(f"{left_label} percentile score")
        ax.set_ylabel(f"{right_label} percentile score")
        ax.set_title(f"{metric.capitalize()} ranking — Spearman rho={rho:.3f}")
        ax.set_xlim(0, 1.01)
        ax.set_ylim(0, 1.01)
        ax.grid(color="#e1e1e1", linewidth=0.5)
    fig.suptitle("Post-training attention-rank concordance")
    _save_figure(fig, output_dir / "attention_rank_pairwise_concordance")


def plot_overlap(stability: pd.DataFrame, output_dir: Path) -> None:
    selected = stability[stability["statistic"] == "jaccard"]
    if selected.empty:
        return
    fig, ax = plt.subplots(figsize=(8.5, 5.8))
    for (left, right, metric), group in selected.groupby(
        ["left_run", "right_run", "metric"], sort=True
    ):
        group = group.sort_values("top_k")
        ax.plot(
            group["top_k"],
            group["value"],
            marker="o",
            label=f"{left} vs {right} — {metric}",
        )
    ax.set_xlabel("Top-k genes")
    ax.set_ylabel("Jaccard overlap")
    ax.set_ylim(0, 1.02)
    ax.set_title("Attention-ranking top-k stability")
    ax.grid(color="#e1e1e1", linewidth=0.5)
    ax.legend(frameon=False, fontsize=8)
    _save_figure(fig, output_dir / "attention_topk_stability")


def main(argv: Sequence[str] | None = None) -> int:
    cli = build_parser().parse_args(argv)
    directories = [path.resolve() for path in cli.analysis_dirs]
    if cli.labels is not None and len(cli.labels) != len(directories):
        raise ValueError("--labels must contain exactly one value per --analysis-dirs entry.")
    labels = cli.labels or [directory.parent.parent.name for directory in directories]
    if len(set(labels)) != len(labels):
        raise ValueError("Run labels must be unique.")
    output_dir = cli.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rankings = [load_ranking(directory, label) for directory, label in zip(directories, labels)]
    consensus = build_consensus(rankings)
    stability = build_stability(rankings)
    consensus.to_csv(output_dir / "gene_attention_consensus.csv", index=False)
    stability.to_csv(output_dir / "attention_stability.csv", index=False)
    plot_pairwise_percentiles(rankings, consensus, output_dir)
    plot_overlap(stability, output_dir)
    write_json(
        output_dir / "attention_consensus_manifest.json",
        {
            "analysis_directories": [str(path) for path in directories],
            "labels": labels,
            "run_count": len(rankings),
            "union_gene_count": int(len(consensus)),
            "complete_inclusion_count": int(
                (consensus["inclusion_frequency"] == 1.0).sum()
            ),
            "top_k_values": list(TOP_K_VALUES),
            "interpretation": (
                "Percentile scores and inclusion-adjusted consensus scores make runs with different "
                "selected-gene sets comparable. A two-run comparison is a paired-variant audit, not "
                "a substitute for stability across independent training seeds."
            ),
        },
    )
    print(f"Attention consensus written to: {output_dir}")
    print(
        consensus[
            [
                "gene",
                "consensus_key_rank",
                "key_percentile_mean",
                "key_top50_frequency",
                "key_rank_iqr",
            ]
        ]
        .head(20)
        .to_string(index=False)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
