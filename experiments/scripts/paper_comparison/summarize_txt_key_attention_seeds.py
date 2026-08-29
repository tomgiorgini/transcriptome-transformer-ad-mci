#!/usr/bin/env python3
"""Build a key-only TxT attention consensus across independent training seeds."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.scripts.paper_comparison.txt_volumetric.common import write_json


DEFAULT_SEEDS = list(range(101, 111))
REGIME_B_SEEDS = {104, 110}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize baseline TxT key attention across seeds; VMA/query are excluded."
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260812)
    parser.add_argument("--stable-core-min-top20-frequency", type=float, default=0.7)
    return parser


def discover_analysis_dirs(run_root: Path, seeds: Sequence[int]) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for seed in seeds:
        matches = sorted(
            path
            for path in (run_root / "seeds10").glob(f"*/seed_{seed}/key_attention/test")
            if (path / "key_attention_by_seed.csv").exists()
        )
        if len(matches) != 1:
            raise FileNotFoundError(
                f"Expected exactly one key-attention export for seed {seed}; found {matches}."
            )
        result[int(seed)] = matches[0]
    return result


def load_seed_frames(paths: dict[int, Path]) -> list[pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    checkpoint_hashes: dict[str, int] = {}
    protocol_hash: str | None = None
    dataset_hashes: tuple[str, str] | None = None
    required = {
        "gene",
        "incoming_macro_mean",
        "key_rank",
        "key_percentile_score",
        "incoming_head_0_macro_mean",
        "incoming_head_1_macro_mean",
        "key_head_0_rank",
        "key_head_1_rank",
    }
    for seed, path in sorted(paths.items()):
        manifest_path = path / "key_attention_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing extraction manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if int(manifest.get("seed", -1)) != int(seed):
            raise ValueError(f"Seed mismatch in {manifest_path}.")
        if manifest.get("model_variant") != "baseline":
            raise ValueError(f"Non-baseline model in {manifest_path}.")
        if manifest.get("split") != "test":
            raise ValueError(f"Non-test extraction in {manifest_path}.")
        if manifest.get("vma_included") is not False:
            raise ValueError(f"VMA must be excluded in {manifest_path}.")
        if manifest.get("query_metrics_included") is not False:
            raise ValueError(f"Query metrics must be excluded in {manifest_path}.")
        if manifest.get("is_full_split") is not True:
            raise ValueError(f"Partial/smoke extraction cannot enter consensus: {manifest_path}.")
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
            raise ValueError(f"Missing X/y hashes in {manifest_path}.")
        if dataset_hashes is None:
            dataset_hashes = current_dataset_hashes
        elif current_dataset_hashes != dataset_hashes:
            raise ValueError(f"Dataset mismatch in {manifest_path}.")
        checkpoint_hash = str(manifest.get("checkpoint_sha256", ""))
        if not checkpoint_hash:
            raise ValueError(f"Missing checkpoint hash in {manifest_path}.")
        if checkpoint_hash in checkpoint_hashes:
            raise ValueError(
                f"Seeds {checkpoint_hashes[checkpoint_hash]} and {seed} use the same checkpoint."
            )
        checkpoint_hashes[checkpoint_hash] = int(seed)
        frame = pd.read_csv(path / "key_attention_by_seed.csv")
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{path} lacks columns: {sorted(missing)}")
        if frame["gene"].duplicated().any():
            raise ValueError(f"Duplicate genes in {path}.")
        if int(manifest.get("genes", -1)) != len(frame):
            raise ValueError(f"Gene-count mismatch in {path}.")
        frame = frame.copy()
        frame["gene"] = frame["gene"].astype(str)
        frame["seed"] = int(seed)
        frames.append(frame)
    return frames


def build_seed_matrices(
    frames: Sequence[pd.DataFrame],
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    genes = sorted(set().union(*(set(frame["gene"]) for frame in frames)))
    gene_to_index = {gene: index for index, gene in enumerate(genes)}
    n_seeds, n_genes = len(frames), len(genes)
    percentile = np.zeros((n_seeds, n_genes), dtype=np.float64)
    incoming = np.full((n_seeds, n_genes), np.nan, dtype=np.float64)
    ranks = np.full((n_seeds, n_genes), np.nan, dtype=np.float64)
    head0_ranks = np.full((n_seeds, n_genes), np.nan, dtype=np.float64)
    head1_ranks = np.full((n_seeds, n_genes), np.nan, dtype=np.float64)
    for seed_index, frame in enumerate(frames):
        indices = np.asarray([gene_to_index[gene] for gene in frame["gene"]], dtype=int)
        percentile[seed_index, indices] = frame["key_percentile_score"].to_numpy(float)
        incoming[seed_index, indices] = frame["incoming_macro_mean"].to_numpy(float)
        ranks[seed_index, indices] = frame["key_rank"].to_numpy(float)
        head0_ranks[seed_index, indices] = frame["key_head_0_rank"].to_numpy(float)
        head1_ranks[seed_index, indices] = frame["key_head_1_rank"].to_numpy(float)
    return genes, percentile, incoming, ranks, head0_ranks, head1_ranks


def seed_bootstrap(
    percentile: np.ndarray,
    incoming: np.ndarray,
    iterations: int,
    seed: int,
) -> dict[str, np.ndarray]:
    n_seeds, n_genes = percentile.shape
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n_seeds, size=(iterations, n_seeds))
    counts = np.stack(
        [(draws == index).sum(axis=1) for index in range(n_seeds)], axis=1
    ).astype(np.int16)
    boot_score = np.empty((iterations, n_genes), dtype=np.float32)
    boot_incoming = np.empty((iterations, n_genes), dtype=np.float32)
    rank_dtype = np.int16 if n_genes <= np.iinfo(np.int16).max else np.int32
    boot_rank = np.empty((iterations, n_genes), dtype=rank_dtype)
    finite = np.isfinite(incoming).astype(np.float32)
    incoming_zeroed = np.nan_to_num(incoming, nan=0.0).astype(np.float32)
    percentile32 = percentile.astype(np.float32)
    base_tie_break = percentile32.mean(axis=0)
    chunk_size = 128
    for start in range(0, iterations, chunk_size):
        stop = min(start + chunk_size, iterations)
        chunk_counts = counts[start:stop].astype(np.float32)
        score_chunk = 100.0 * (chunk_counts @ percentile32) / float(n_seeds)
        boot_score[start:stop] = score_chunk
        numerator = chunk_counts @ incoming_zeroed
        denominator = chunk_counts @ finite
        boot_incoming[start:stop] = np.divide(
            numerator,
            denominator,
            out=np.full_like(numerator, np.nan, dtype=np.float32),
            where=denominator > 0,
        )
        # Break resampling ties with the full-sample Borda score. This avoids
        # making top-k probability depend on the alphabetical union-gene order.
        adjusted_score = score_chunk.astype(np.float64) + 1e-9 * base_tie_break[None, :]
        order = np.argsort(-adjusted_score, axis=1, kind="stable")
        rank_chunk = np.empty(order.shape, dtype=rank_dtype)
        rank_chunk[np.arange(len(order))[:, None], order] = np.arange(
            1, n_genes + 1, dtype=rank_dtype
        )
        boot_rank[start:stop] = rank_chunk
    return {
        "score_lower": np.percentile(boot_score, 2.5, axis=0),
        "score_upper": np.percentile(boot_score, 97.5, axis=0),
        "incoming_lower": np.nanpercentile(boot_incoming, 2.5, axis=0),
        "incoming_upper": np.nanpercentile(boot_incoming, 97.5, axis=0),
        "rank_lower": np.percentile(boot_rank, 2.5, axis=0),
        "rank_upper": np.percentile(boot_rank, 97.5, axis=0),
        "probability_top10": (boot_rank <= 10).mean(axis=0),
        "probability_top20": (boot_rank <= 20).mean(axis=0),
    }


def build_consensus(
    genes: Sequence[str],
    percentile: np.ndarray,
    incoming: np.ndarray,
    ranks: np.ndarray,
    head0_ranks: np.ndarray,
    head1_ranks: np.ndarray,
    bootstrap: dict[str, np.ndarray],
    stable_core_threshold: float,
) -> pd.DataFrame:
    n_seeds = percentile.shape[0]
    included = np.isfinite(ranks)
    runs_included = included.sum(axis=0)
    incoming_count = np.isfinite(incoming).sum(axis=0)
    incoming_mean = np.divide(
        np.nansum(incoming, axis=0),
        incoming_count,
        out=np.full(incoming.shape[1], np.nan),
        where=incoming_count > 0,
    )
    rank_median = np.nanmedian(ranks, axis=0)
    rank_q25 = np.nanpercentile(ranks, 25, axis=0)
    rank_q75 = np.nanpercentile(ranks, 75, axis=0)
    frame = pd.DataFrame(
        {
            "gene": np.asarray(genes, dtype=str),
            "runs_included": runs_included,
            "run_count": n_seeds,
            "inclusion_frequency": runs_included / n_seeds,
            "consensus_score": 100.0 * percentile.mean(axis=0),
            "incoming_mean_across_included_seeds": incoming_mean,
            "incoming_log2_mean_across_included_seeds": np.log2(
                np.maximum(incoming_mean, 1e-12)
            ),
            "enrichment_percent_mean_across_included_seeds": 100.0
            * (incoming_mean - 1.0),
            "mean_rank_when_included": np.nanmean(ranks, axis=0),
            "median_rank_when_included": rank_median,
            "rank_iqr_when_included": rank_q75 - rank_q25,
            "min_rank_when_included": np.nanmin(ranks, axis=0),
            "max_rank_when_included": np.nanmax(ranks, axis=0),
            "top10_count": np.nansum(ranks <= 10, axis=0).astype(int),
            "top10_frequency": np.nansum(ranks <= 10, axis=0) / n_seeds,
            "top20_count": np.nansum(ranks <= 20, axis=0).astype(int),
            "top20_frequency": np.nansum(ranks <= 20, axis=0) / n_seeds,
            "head_both_top20_count": np.nansum(
                (head0_ranks <= 20) & (head1_ranks <= 20), axis=0
            ).astype(int),
            "head_both_top20_frequency": np.nansum(
                (head0_ranks <= 20) & (head1_ranks <= 20), axis=0
            )
            / n_seeds,
            "bootstrap_consensus_score_ci95_lower": bootstrap["score_lower"],
            "bootstrap_consensus_score_ci95_upper": bootstrap["score_upper"],
            "bootstrap_incoming_ci95_lower": bootstrap["incoming_lower"],
            "bootstrap_incoming_ci95_upper": bootstrap["incoming_upper"],
            "bootstrap_consensus_rank_ci95_lower": bootstrap["rank_lower"],
            "bootstrap_consensus_rank_ci95_upper": bootstrap["rank_upper"],
            "bootstrap_probability_top10": bootstrap["probability_top10"],
            "bootstrap_probability_top20": bootstrap["probability_top20"],
        }
    )
    frame = frame.sort_values(
        [
            "consensus_score",
            "top20_frequency",
            "median_rank_when_included",
            "incoming_mean_across_included_seeds",
            "gene",
        ],
        ascending=[False, False, True, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    frame.insert(0, "consensus_rank", np.arange(1, len(frame) + 1))
    frame["stable_among_borda_top20"] = (
        (frame["consensus_rank"] <= 20)
        & (frame["top20_frequency"] >= stable_core_threshold)
    )
    # Compatibility alias for previously generated tables. The new name makes
    # explicit that stability is evaluated only inside the Borda top 20.
    frame["stable_core_top20"] = frame["stable_among_borda_top20"]
    return frame


def build_stability(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    indexed = {int(frame["seed"].iloc[0]): frame.set_index("gene") for frame in frames}
    for left_seed, right_seed in itertools.combinations(sorted(indexed), 2):
        left, right = indexed[left_seed], indexed[right_seed]
        common = left.index.intersection(right.index)
        rho = stats.spearmanr(
            left.loc[common, "key_rank"], right.loc[common, "key_rank"]
        ).statistic
        row: dict[str, float | int] = {
            "left_seed": left_seed,
            "right_seed": right_seed,
            "common_genes": len(common),
            "key_rank_spearman": float(rho),
        }
        for top_k in (10, 20):
            left_top = set(left.nsmallest(top_k, "key_rank").index)
            right_top = set(right.nsmallest(top_k, "key_rank").index)
            union = left_top | right_top
            row[f"jaccard_top{top_k}"] = len(left_top & right_top) / len(union)
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_regime_stability(
    stability: pd.DataFrame,
    seeds: Sequence[int],
) -> dict[str, dict[str, float | int | list[int]]]:
    regime_a, regime_b = _regime_seed_groups(seeds)
    regime_a_set, regime_b_set = set(regime_a), set(regime_b)
    left = stability["left_seed"].astype(int)
    right = stability["right_seed"].astype(int)
    masks = {
        "within_regime_a": left.isin(regime_a_set) & right.isin(regime_a_set),
        "within_regime_b": left.isin(regime_b_set) & right.isin(regime_b_set),
        "between_regimes": (
            (left.isin(regime_a_set) & right.isin(regime_b_set))
            | (left.isin(regime_b_set) & right.isin(regime_a_set))
        ),
    }
    result: dict[str, dict[str, float | int | list[int]]] = {}
    for name, mask in masks.items():
        subset = stability.loc[mask]
        result[name] = {
            "pair_count": int(len(subset)),
            "spearman_mean": float(subset["key_rank_spearman"].mean()),
            "spearman_median": float(subset["key_rank_spearman"].median()),
            "jaccard_top10_mean": float(subset["jaccard_top10"].mean()),
            "jaccard_top20_mean": float(subset["jaccard_top20"].mean()),
        }
    result["within_regime_a"]["seeds"] = regime_a
    result["within_regime_b"]["seeds"] = regime_b
    return result


def save_figure(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".png"), dpi=240, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _regime_seed_groups(seeds: Sequence[int]) -> tuple[list[int], list[int]]:
    regime_b = [int(seed) for seed in seeds if int(seed) in REGIME_B_SEEDS]
    regime_a = [int(seed) for seed in seeds if int(seed) not in REGIME_B_SEEDS]
    return regime_a, regime_b


def plot_forest(
    consensus: pd.DataFrame,
    frames: Sequence[pd.DataFrame],
    top_count: int,
    output_dir: Path,
) -> None:
    frame = consensus.nsmallest(top_count, "consensus_rank").sort_values(
        "consensus_rank", ascending=False
    )
    mean = frame["enrichment_percent_mean_across_included_seeds"].to_numpy()
    lower = 100.0 * (frame["bootstrap_incoming_ci95_lower"].to_numpy() - 1.0)
    upper = 100.0 * (frame["bootstrap_incoming_ci95_upper"].to_numpy() - 1.0)
    y = np.arange(len(frame))
    top10 = frame["consensus_rank"].to_numpy() <= 10
    indexed_frames = {
        int(seed_frame["seed"].iloc[0]): seed_frame.set_index("gene")
        for seed_frame in frames
    }
    seeds = sorted(indexed_frames)
    regime_a, regime_b = _regime_seed_groups(seeds)
    regime_a_offsets = {
        seed: offset
        for seed, offset in zip(
            regime_a,
            np.linspace(-0.12, 0.12, max(len(regime_a), 1)),
        )
    }
    regime_b_offsets = {
        seed: offset
        for seed, offset in zip(
            regime_b,
            np.linspace(-0.18, 0.18, max(len(regime_b), 1)),
        )
    }

    fig, ax = plt.subplots(figsize=(11.2, max(5.8, 0.42 * top_count + 2.0)))
    fig.subplots_adjust(top=0.89, bottom=0.13)
    for index in range(len(frame)):
        gene = str(frame.iloc[index]["gene"])
        for seed in regime_a:
            seed_frame = indexed_frames[seed]
            if gene not in seed_frame.index:
                continue
            raw_enrichment = 100.0 * (
                float(seed_frame.loc[gene, "incoming_macro_mean"]) - 1.0
            )
            ax.scatter(
                raw_enrichment,
                y[index] + regime_a_offsets[seed],
                s=18,
                marker="o",
                facecolor="#74879a",
                edgecolor="white",
                linewidth=0.35,
                alpha=0.72,
                zorder=3,
            )
        for seed, marker, color in (
            (104, "^", "#b23a48"),
            (110, "s", "#d17a22"),
        ):
            if seed not in indexed_frames or seed not in regime_b:
                continue
            seed_frame = indexed_frames[seed]
            if gene not in seed_frame.index:
                continue
            raw_enrichment = 100.0 * (
                float(seed_frame.loc[gene, "incoming_macro_mean"]) - 1.0
            )
            ax.scatter(
                raw_enrichment,
                y[index] + regime_b_offsets[seed],
                s=34,
                marker=marker,
                facecolor=color,
                edgecolor="white",
                linewidth=0.5,
                alpha=0.95,
                zorder=4,
            )
        left_error = max(float(mean[index] - lower[index]), 0.0)
        right_error = max(float(upper[index] - mean[index]), 0.0)
        ax.errorbar(
            mean[index],
            y[index],
            xerr=np.asarray([[left_error], [right_error]]),
            fmt="o",
            markerfacecolor="#2f5597" if top10[index] else "white",
            markeredgecolor="#2f5597",
            markersize=7,
            ecolor="#96a9c3",
            elinewidth=1.4,
            capsize=2.5,
            zorder=5,
        )
        ax.text(
            1.01,
            y[index],
            f"top{top_count}: "
            f"{int(frame.iloc[index][f'top{top_count}_count'])}/"
            f"{int(frame.iloc[index]['run_count'])}",
            transform=ax.get_yaxis_transform(),
            va="center",
            ha="left",
            fontsize=8.5,
            color="#444444",
        )
    ax.axvline(0.0, color="#777777", linestyle="--", linewidth=1.0)
    ax.set_yticks(y, frame["gene"])
    ax.set_xlabel("Arricchimento medio di attenzione entrante vs uniforme (%)")
    ax.set_ylabel("Key gene")
    run_count = int(frame["run_count"].iloc[0])
    fig.suptitle(
        f"TxT baseline — top {top_count} per consenso Borda su {run_count} seed",
        fontsize=15,
        y=0.985,
    )
    ax.text(
        0,
        1.015,
        "Selezione = percentile-rank Borda; asse = arricchimento medio; "
        "barre = intervallo bootstrap descrittivo 95%",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        color="#555555",
    )
    ax.grid(axis="x", color="#dddddd", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#74879a",
            markeredgecolor="white",
            markersize=5.5,
            label="Seed regime A",
        ),
        Line2D(
            [0],
            [0],
            marker="^",
            color="none",
            markerfacecolor="#b23a48",
            markeredgecolor="white",
            markersize=6.5,
            label="Seed 104 (regime B)",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            color="none",
            markerfacecolor="#d17a22",
            markeredgecolor="white",
            markersize=6.0,
            label="Seed 110 (regime B)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="#96a9c3",
            markerfacecolor="#2f5597",
            markeredgecolor="#2f5597",
            markersize=6.5,
            label=f"Media {run_count} seed (pieno = Borda top 10)",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=4,
        frameon=False,
        fontsize=8.2,
        handletextpad=0.4,
        columnspacing=1.2,
    )
    save_figure(fig, output_dir / f"baseline_key_attention_consensus_top{top_count}")


def plot_spearman_heatmap(
    stability: pd.DataFrame,
    seeds: Sequence[int],
    output_dir: Path,
) -> None:
    ordered_seeds = sorted(map(int, seeds))
    seed_to_index = {seed: index for index, seed in enumerate(ordered_seeds)}
    matrix = np.eye(len(ordered_seeds), dtype=np.float64)
    for row in stability.itertuples(index=False):
        left = seed_to_index[int(row.left_seed)]
        right = seed_to_index[int(row.right_seed)]
        matrix[left, right] = float(row.key_rank_spearman)
        matrix[right, left] = float(row.key_rank_spearman)

    fig, ax = plt.subplots(figsize=(9.2, 8.1))
    image = ax.imshow(matrix, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    labels = [str(seed) for seed in ordered_seeds]
    ax.set_xticks(np.arange(len(ordered_seeds)), labels)
    ax.set_yticks(np.arange(len(ordered_seeds)), labels)
    ax.set_xlabel("Seed")
    ax.set_ylabel("Seed")
    ax.set_title("Correlazione Spearman dei ranking key-attention fra seed")
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
    for tick, seed in zip(ax.get_xticklabels(), ordered_seeds):
        if seed in REGIME_B_SEEDS:
            tick.set_color("#9f2d3a")
            tick.set_fontweight("bold")
    for tick, seed in zip(ax.get_yticklabels(), ordered_seeds):
        if seed in REGIME_B_SEEDS:
            tick.set_color("#9f2d3a")
            tick.set_fontweight("bold")
    for row in range(len(ordered_seeds)):
        for column in range(len(ordered_seeds)):
            value = matrix[row, column]
            color = "white" if abs(value) >= 0.65 else "black"
            ax.text(
                column,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color=color,
            )
    colorbar = fig.colorbar(image, ax=ax, shrink=0.84)
    colorbar.set_label("Spearman rho sui geni comuni")
    fig.text(
        0.01,
        0.01,
        "Etichette rosse = regime B (seed 104 e 110); gli altri seed formano il regime A.",
        fontsize=8.5,
        color="#555555",
    )
    save_figure(fig, output_dir / "baseline_key_attention_seed_spearman_heatmap")


def plot_seed_heatmap(
    consensus: pd.DataFrame,
    frames: Sequence[pd.DataFrame],
    output_dir: Path,
) -> None:
    top = consensus.nsmallest(20, "consensus_rank")
    genes = top["gene"].tolist()
    seeds = [int(frame["seed"].iloc[0]) for frame in frames]
    matrix = np.full((len(genes), len(frames)), np.nan)
    rank_labels = np.full((len(genes), len(frames)), "—", dtype=object)
    for column, frame in enumerate(frames):
        indexed = frame.set_index("gene")
        for row, gene in enumerate(genes):
            if gene in indexed.index:
                matrix[row, column] = indexed.loc[gene, "key_percentile_score"]
                rank_labels[row, column] = str(int(indexed.loc[gene, "key_rank"]))
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#d8d8d8")
    fig, ax = plt.subplots(figsize=(12.5, 9.0))
    image = ax.imshow(matrix, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(np.arange(len(seeds)), [str(seed) for seed in seeds])
    ax.set_yticks(np.arange(len(genes)), genes)
    ax.set_xlabel("Seed")
    ax.set_ylabel("Consensus key gene")
    ax.set_title("Stabilità del top 20 Borda: percentile rank per seed")
    for tick, seed in zip(ax.get_xticklabels(), seeds):
        if seed in REGIME_B_SEEDS:
            tick.set_color("#9f2d3a")
            tick.set_fontweight("bold")
    for row in range(len(genes)):
        for column in range(len(seeds)):
            value = matrix[row, column]
            color = "white" if np.isfinite(value) and value < 0.5 else "black"
            ax.text(
                column,
                row,
                rank_labels[row, column],
                ha="center",
                va="center",
                fontsize=7,
                color=color,
            )
    colorbar = fig.colorbar(image, ax=ax, shrink=0.85)
    colorbar.set_label("Percentile rank (1 = migliore)")
    fig.text(
        0.01,
        0.01,
        "Numero nella cella = rank nel seed; grigio/— = gene non selezionato tra i top-2000.",
        fontsize=8.5,
        color="#555555",
    )
    save_figure(fig, output_dir / "baseline_key_attention_top20_seed_heatmap")


def write_summary(
    path: Path,
    consensus: pd.DataFrame,
    stability: pd.DataFrame,
    seeds: Sequence[int],
) -> None:
    top = consensus.head(20).copy()
    regime_stats = summarize_regime_stability(stability, seeds)
    regime_a = regime_stats["within_regime_a"]
    regime_b = regime_stats["within_regime_b"]
    between = regime_stats["between_regimes"]
    regime_a_seeds = ", ".join(map(str, regime_a["seeds"]))
    regime_b_seeds = ", ".join(map(str, regime_b["seeds"]))
    lines = [
        f"# TxT baseline key-attention consensus across {len(seeds)} seeds",
        "",
        f"Seeds: {', '.join(map(str, seeds))}.",
        "",
        "Only branch-free baseline TxT dense key-side incoming attention is included. "
        "VMA and query-side metrics are excluded.",
        "",
        "The primary per-seed value averages the two heads, macro-averages Control/MCI/AD "
        "within that seed's held-out test split, and ranks genes inside the seed. The consensus "
        "is the equal-seed mean percentile rank (Borda score, 0–100); absence from a seed's "
        "train-selected top-2000 set contributes zero percentile support.",
        "The plotted mean enrichment is conditional on a gene being included in that seed's "
        "train-selected top-2000 set; inclusion frequency is therefore reported alongside it.",
        "",
        "| Rank | Gene | Score | Inclusion | Top-10 freq. | Top-20 freq. | Mean enrichment (%) |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in top.itertuples(index=False):
        lines.append(
            f"| {row.consensus_rank} | {row.gene} | {row.consensus_score:.2f} | "
            f"{row.inclusion_frequency:.1f} | {row.top10_frequency:.1f} | "
            f"{row.top20_frequency:.1f} | "
            f"{row.enrichment_percent_mean_across_included_seeds:+.3f} |"
        )
    lines.extend(
        [
            "",
            "## Stability",
            "",
            f"Mean pairwise key-rank Spearman rho on common genes: "
            f"{stability['key_rank_spearman'].mean():.3f}.",
            "",
            f"Mean pairwise Jaccard top 10: {stability['jaccard_top10'].mean():.3f}; "
            f"top 20: {stability['jaccard_top20'].mean():.3f}.",
            "",
            "The global means mask two sharply separated ranking regimes:",
            "",
            f"- Within regime A (seeds {regime_a_seeds}; "
            f"{int(regime_a['pair_count'])} pairs): mean/median Spearman rho "
            f"{float(regime_a['spearman_mean']):.3f}/"
            f"{float(regime_a['spearman_median']):.3f}; mean Jaccard top 10/top 20 "
            f"{float(regime_a['jaccard_top10_mean']):.3f}/"
            f"{float(regime_a['jaccard_top20_mean']):.3f}.",
            f"- Within regime B (seeds {regime_b_seeds}; "
            f"{int(regime_b['pair_count'])} pair): Spearman rho "
            f"{float(regime_b['spearman_mean']):.3f}; Jaccard top 10/top 20 "
            f"{float(regime_b['jaccard_top10_mean']):.3f}/"
            f"{float(regime_b['jaccard_top20_mean']):.3f}.",
            f"- Between regimes ({int(between['pair_count'])} pairs): mean/median "
            f"Spearman rho {float(between['spearman_mean']):.3f}/"
            f"{float(between['spearman_median']):.3f}; mean Jaccard top 10/top 20 "
            f"{float(between['jaccard_top10_mean']):.3f}/"
            f"{float(between['jaccard_top20_mean']):.3f}.",
            "",
            "The Borda list is therefore an equal-seed rank aggregation across both regimes, "
            "not evidence of one homogeneous attention ordering.",
            "",
            "The seed bootstrap is descriptive because the runs reuse much of the same cohort "
            "and mix training initialization with split variability. Attention remains an internal "
            "relational model quantity, not causal gene importance.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    cli = build_parser().parse_args(argv)
    if len(set(cli.seeds)) != len(cli.seeds):
        raise ValueError("--seeds must not contain duplicates.")
    if cli.bootstrap_iterations <= 0:
        raise ValueError("--bootstrap-iterations must be positive.")
    if not 0.0 <= cli.stable_core_min_top20_frequency <= 1.0:
        raise ValueError("Stable-core frequency must be in [0, 1].")
    run_root = cli.run_root.resolve()
    output_dir = (
        cli.output_dir.resolve()
        if cli.output_dir is not None
        else run_root / "key_attention_consensus"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = discover_analysis_dirs(run_root, cli.seeds)
    frames = load_seed_frames(paths)
    genes, percentile, incoming, ranks, head0_ranks, head1_ranks = build_seed_matrices(
        frames
    )
    bootstrap = seed_bootstrap(
        percentile,
        incoming,
        cli.bootstrap_iterations,
        cli.bootstrap_seed,
    )
    consensus = build_consensus(
        genes,
        percentile,
        incoming,
        ranks,
        head0_ranks,
        head1_ranks,
        bootstrap,
        cli.stable_core_min_top20_frequency,
    )
    stability = build_stability(frames)
    consensus.to_csv(output_dir / "key_attention_consensus.csv", index=False)
    consensus.head(10).to_csv(output_dir / "key_attention_consensus_top10.csv", index=False)
    consensus.head(20).to_csv(output_dir / "key_attention_consensus_top20.csv", index=False)
    pd.concat(frames, ignore_index=True).to_csv(
        output_dir / "key_attention_by_seed_long.csv", index=False
    )
    stability.to_csv(output_dir / "key_attention_pairwise_stability.csv", index=False)
    plot_forest(consensus, frames, 10, output_dir)
    plot_forest(consensus, frames, 20, output_dir)
    plot_seed_heatmap(consensus, frames, output_dir)
    plot_spearman_heatmap(stability, cli.seeds, output_dir)
    write_summary(output_dir / "SUMMARY.md", consensus, stability, cli.seeds)
    regime_stability = summarize_regime_stability(stability, cli.seeds)
    write_json(
        output_dir / "key_attention_consensus_manifest.json",
        {
            "run_root": str(run_root),
            "seeds": list(map(int, cli.seeds)),
            "run_count": len(cli.seeds),
            "analysis_directories": {str(seed): str(path) for seed, path in paths.items()},
            "union_gene_count": len(genes),
            "bootstrap_iterations": cli.bootstrap_iterations,
            "bootstrap_seed": cli.bootstrap_seed,
            "consensus_metric": (
                "100 * mean seed percentile rank; absent genes receive zero support"
            ),
            "incoming_aggregation": (
                "mean heads, macro mean biological classes within seed, equal mean across included seeds"
            ),
            "ranking_regimes": regime_stability,
            "vma_included": False,
            "query_metrics_included": False,
        },
    )
    print(f"Key-attention consensus written to: {output_dir}")
    print(
        consensus[
            [
                "consensus_rank",
                "gene",
                "consensus_score",
                "inclusion_frequency",
                "top10_frequency",
                "top20_frequency",
                "enrichment_percent_mean_across_included_seeds",
            ]
        ]
        .head(20)
        .to_string(index=False)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
