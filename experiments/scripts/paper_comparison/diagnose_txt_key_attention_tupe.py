#!/usr/bin/env python3
"""Diagnose TUPE dominance and seed-dependent key-attention orientation.

This is a read-only post-training diagnostic for one-layer baseline TxT runs.
It analyses the held-out test split and the already exported key-side incoming
attention.  It deliberately does not compute query metrics and never accesses
the optional VMA branch.

For the thesis configuration (``expression_residual='none'``), TUPE factorises
for each head as

    TUPE_h[i, j] = c_h * x[i] * x[j]
    c_h = <w_q_h, w_k_h> / sqrt(2 * d_head).

The script compares the scale and ranking induced by this term with the dense
embedding-derived QK term, and records whether each seed favours high- or
low-expression keys.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.scripts.paper_comparison.analyze_txt_attention import sha256_file  # noqa: E402
from experiments.scripts.paper_comparison.txt_volumetric.common import (  # noqa: E402
    resolve_path,
    write_json,
)


DEFAULT_SEEDS = tuple(range(101, 111))
ATTENTION_RELATIVE_PATH = Path("key_attention/test/key_attention_by_seed.csv")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose TUPE dominance in baseline TxT test-set key attention; "
            "VMA and query metrics are excluded."
        )
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        required=True,
        help="Directory containing the seed_<N> run directories (recursively).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for diagnostic CSV, JSON, PNG, PDF, and Markdown outputs.",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--checkpoint-name", default="best_model.pt")
    parser.add_argument(
        "--attention-relative-path",
        type=Path,
        default=ATTENTION_RELATIVE_PATH,
    )
    return parser


def _resolve_seed_run(runs_root: Path, seed: int, checkpoint_name: str) -> Path:
    candidates = sorted(
        path
        for path in runs_root.rglob(f"seed_{seed}")
        if path.is_dir()
        and (path / "args.json").is_file()
        and (path / checkpoint_name).is_file()
    )
    if len(candidates) != 1:
        rendered = ", ".join(str(path) for path in candidates) or "none"
        raise RuntimeError(
            f"Expected exactly one complete run directory for seed {seed}; found "
            f"{len(candidates)}: {rendered}."
        )
    return candidates[0]


def _spearman(left: np.ndarray, right: np.ndarray, *, label: str) -> float:
    coefficient = float(stats.spearmanr(left, right).statistic)
    if not np.isfinite(coefficient):
        raise FloatingPointError(f"Non-finite Spearman coefficient for {label}.")
    return coefficient


def _macro_expression(
    gene_x: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    class_means = []
    for class_value in sorted(np.unique(labels).tolist()):
        keep = labels == class_value
        if not keep.any():
            raise ValueError(f"Test split has no samples for class {class_value!r}.")
        class_means.append(gene_x[keep].mean(axis=0, dtype=np.float64))
    return np.stack(class_means).mean(axis=0)


def _tupe_logit_sd(gene_x: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    """Exact population SD over test sample x query x key TUPE logits."""

    expression = np.asarray(gene_x, dtype=np.float64)
    sample_mean = expression.mean(axis=1)
    sample_second_moment = np.square(expression).mean(axis=1)
    product_mean = float(np.square(sample_mean).mean())
    product_second_moment = float(np.square(sample_second_moment).mean())
    product_variance = max(product_second_moment - product_mean**2, 0.0)
    return np.abs(coefficients) * math.sqrt(product_variance)


def _resolve_saved_input(value: Any, run_dir: Path, *, field: str) -> Path:
    if value is None:
        raise ValueError(f"Saved run has no {field!r} path.")
    path = Path(str(value)).expanduser()
    candidates = [path] if path.is_absolute() else [run_dir / path, ROOT / path, path]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        f"Cannot resolve saved {field!r} path {value!r} for {run_dir}."
    )


def _load_indexed_csv(
    path: Path,
    cache: dict[Path, pd.DataFrame],
    *,
    index_column: str = "sample_id",
) -> pd.DataFrame:
    if path not in cache:
        frame = pd.read_csv(path)
        if index_column not in frame.columns:
            raise KeyError(f"Missing {index_column!r} in {path}.")
        frame[index_column] = frame[index_column].astype(str)
        if frame[index_column].duplicated().any():
            raise ValueError(f"Duplicate {index_column!r} values in {path}.")
        cache[path] = frame.set_index(index_column)
    return cache[path]


def _load_checkpoint_state(checkpoint: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and payload and all(
        isinstance(value, torch.Tensor) for value in payload.values()
    ):
        return payload
    if isinstance(payload, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            state = payload.get(key)
            if isinstance(state, dict) and state and all(
                isinstance(value, torch.Tensor) for value in state.values()
            ):
                return state
    raise TypeError(f"Unsupported checkpoint payload in {checkpoint}.")


def diagnose_seed(
    run_dir: Path,
    *,
    seed: int,
    checkpoint_name: str,
    attention_relative_path: Path,
    table_cache: dict[Path, pd.DataFrame],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    checkpoint = run_dir / checkpoint_name
    attention_file = run_dir / attention_relative_path
    if not attention_file.is_file():
        raise FileNotFoundError(
            f"Missing exported test key attention for seed {seed}: {attention_file}"
        )

    with (run_dir / "args.json").open(encoding="utf-8") as handle:
        args = json.load(handle)
    if args.get("model_variant", "baseline") != "baseline":
        raise ValueError(f"Seed {seed} is not a baseline TxT run.")
    if args.get("tupe_mode", "on") != "on":
        raise ValueError(f"Seed {seed} has TUPE disabled; this diagnostic requires TUPE on.")
    if args.get("expression_residual", "none") != "none":
        raise ValueError(
            f"Seed {seed} uses expression_residual={args.get('expression_residual')!r}; "
            "the factorised baseline diagnostic requires 'none'."
        )
    if int(args.get("n_layers", 1)) != 1:
        raise ValueError(f"Seed {seed} is not a one-layer TxT run.")
    if args.get("scaler") != "minmax":
        raise ValueError(
            f"Seed {seed} uses scaler={args.get('scaler')!r}; this exact diagnostic "
            "currently requires the saved min-max protocol."
        )

    x_file = _resolve_saved_input(args.get("x_file"), run_dir, field="x_file")
    y_file = _resolve_saved_input(args.get("y_file"), run_dir, field="y_file")
    split_file = _resolve_saved_input(
        args.get("split_file"), run_dir, field="split_file"
    )
    expression_frame = _load_indexed_csv(x_file, table_cache)
    label_frame = _load_indexed_csv(y_file, table_cache)
    split_frame = _load_indexed_csv(split_file, table_cache)
    if "split" not in split_frame.columns:
        raise KeyError(f"Missing 'split' column in {split_file}.")
    selected_gene_file = run_dir / "selected_genes.csv"
    selected = pd.read_csv(selected_gene_file)
    if "gene" not in selected.columns or selected["gene"].duplicated().any():
        raise ValueError(f"Invalid selected-gene artifact: {selected_gene_file}")
    gene_names = selected["gene"].astype(str).to_numpy()

    train_ids = split_frame.index[split_frame["split"].astype(str) == "train"]
    test_ids = split_frame.index[split_frame["split"].astype(str) == "test"]
    missing_ids = sorted(
        (set(train_ids) | set(test_ids))
        - set(expression_frame.index.astype(str))
    )
    if missing_ids:
        raise ValueError(f"Split for seed {seed} references missing expression samples.")
    raw_selected = expression_frame.loc[:, gene_names]
    train_min = raw_selected.loc[train_ids].min(axis=0)
    train_range = raw_selected.loc[train_ids].max(axis=0) - train_min
    if (train_range <= 0).any():
        raise ValueError(f"Seed {seed} selected a zero-range gene under min-max scaling.")
    gene_x = (
        (raw_selected.loc[test_ids] - train_min) / train_range
    ).to_numpy(dtype=np.float32)
    if "label" not in label_frame.columns:
        raise KeyError(f"Missing 'label' column in {y_file}.")
    labels = label_frame.loc[test_ids, "label"].to_numpy(dtype=np.int64)
    sample_ids = test_ids.astype(str).to_numpy()
    if len(sample_ids) == 0 or len(gene_names) == 0:
        raise ValueError(f"Seed {seed} has an empty test split or gene set.")
    macro_expression = _macro_expression(gene_x, labels)

    ranking = pd.read_csv(attention_file)
    if ranking["gene"].duplicated().any():
        raise ValueError(f"Duplicate genes in {attention_file}.")
    ranking = ranking.set_index("gene")
    missing_genes = sorted(set(gene_names) - set(ranking.index.astype(str)))
    if missing_genes:
        raise ValueError(
            f"Attention export for seed {seed} is missing {len(missing_genes)} genes."
        )
    ranking = ranking.loc[gene_names]

    state = _load_checkpoint_state(checkpoint)
    n_heads = int(args["n_heads"])
    d_model = int(args["d_model"])
    if d_model % n_heads:
        raise ValueError(f"Seed {seed}: d_model is not divisible by n_heads.")
    d_head = d_model // n_heads
    embedding = state["transformer.encoder.embed.embed.weight"].float()
    if embedding.shape != (len(gene_names), d_model):
        raise ValueError(
            f"Seed {seed}: checkpoint embedding shape {tuple(embedding.shape)} is "
            f"not {(len(gene_names), d_model)}."
        )
    attention_input = embedding
    if bool(args.get("norm_first", False)):
        attention_input = F.layer_norm(
            attention_input,
            (d_model,),
            state["transformer.encoder.layers.0.layer_norm_1.weight"].float(),
            state["transformer.encoder.layers.0.layer_norm_1.bias"].float(),
            1e-5,
        )
    q = torch.matmul(
        attention_input,
        state[
            "transformer.encoder.layers.0.multi_head_attention_layer.q_linear.weight"
        ].float().transpose(0, 1),
    ).reshape(len(gene_names), n_heads, d_head).transpose(0, 1)
    k = torch.matmul(
        attention_input,
        state[
            "transformer.encoder.layers.0.multi_head_attention_layer.k_linear.weight"
        ].float().transpose(0, 1),
    ).reshape(len(gene_names), n_heads, d_head).transpose(0, 1)
    base_logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(2 * d_head)
    base_attention = torch.softmax(base_logits, dim=-1)
    base_incoming = (
        len(gene_names) * base_attention.mean(dim=-2)
    ).detach().cpu().numpy().astype(np.float64)
    base_logit_sd = (
        base_logits.float()
        .std(dim=(-2, -1), unbiased=False)
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64)
    )

    q_weight = state["transformer.encoder.tupe.q_linear.weight"].numpy().reshape(
        n_heads, d_head
    )
    k_weight = state["transformer.encoder.tupe.k_linear.weight"].numpy().reshape(
        n_heads, d_head
    )
    coefficients = np.sum(q_weight * k_weight, axis=1) / math.sqrt(
        2 * d_head
    )
    tupe_logit_sd = _tupe_logit_sd(gene_x, coefficients)

    head_rows: list[dict[str, Any]] = []
    full_heads: list[np.ndarray] = []
    for head in range(n_heads):
        column = f"incoming_head_{head}_macro_mean"
        if column not in ranking.columns:
            raise KeyError(f"Missing column {column!r} in {attention_file}.")
        full_incoming = ranking[column].to_numpy(dtype=np.float64)
        full_heads.append(full_incoming)
        head_rows.append(
            {
                "seed": seed,
                "head": head,
                "tupe_coefficient_c_h": float(coefficients[head]),
                "tupe_coefficient_sign": (
                    "positive" if coefficients[head] > 0 else "negative"
                ),
                "base_qk_logit_sd": float(base_logit_sd[head]),
                "tupe_logit_sd": float(tupe_logit_sd[head]),
                "tupe_to_base_logit_sd_ratio": float(
                    tupe_logit_sd[head] / base_logit_sd[head]
                ),
                "spearman_full_vs_macro_expression": _spearman(
                    full_incoming,
                    macro_expression,
                    label=f"seed {seed} head {head}: full vs expression",
                ),
                "spearman_base_vs_macro_expression": _spearman(
                    base_incoming[head],
                    macro_expression,
                    label=f"seed {seed} head {head}: base vs expression",
                ),
                "spearman_full_vs_base": _spearman(
                    full_incoming,
                    base_incoming[head],
                    label=f"seed {seed} head {head}: full vs base",
                ),
                "test_samples": int(len(sample_ids)),
                "genes": int(len(gene_names)),
                "checkpoint": str(checkpoint.resolve()),
                "attention_file": str(attention_file.resolve()),
            }
        )

    full_macro = ranking["incoming_macro_mean"].to_numpy(dtype=np.float64)
    full_macro_rho = _spearman(
        full_macro,
        macro_expression,
        label=f"seed {seed}: mean-head full vs expression",
    )
    coefficient_sum = float(coefficients.sum())
    mode = "high_expression_keys" if full_macro_rho > 0 else "low_expression_keys"
    predicted_mode = (
        "high_expression_keys" if coefficient_sum > 0 else "low_expression_keys"
    )
    head_rank_spearman = (
        _spearman(
            full_heads[0],
            full_heads[1],
            label=f"seed {seed}: head 0 vs head 1",
        )
        if len(full_heads) == 2
        else float("nan")
    )
    seed_row = {
        "seed": seed,
        "tupe_coefficient_sum": coefficient_sum,
        "tupe_coefficient_mean": float(coefficients.mean()),
        "full_mean_head_vs_macro_expression_spearman": full_macro_rho,
        "observed_key_mode": mode,
        "coefficient_predicted_key_mode": predicted_mode,
        "coefficient_sign_matches_observed_mode": bool(mode == predicted_mode),
        "head_0_vs_head_1_full_spearman": head_rank_spearman,
        "test_samples": int(len(sample_ids)),
        "genes": int(len(gene_names)),
    }
    provenance = {
        "seed": seed,
        "run_dir": str(run_dir.resolve()),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "attention_file": str(attention_file.resolve()),
        "attention_file_sha256": sha256_file(attention_file),
        "model_variant": str(args.get("model_variant", "baseline")),
        "tupe_mode": str(args.get("tupe_mode", "on")),
        "expression_residual": str(args.get("expression_residual", "none")),
        "x_file": str(x_file),
        "y_file": str(y_file),
        "split_file": str(split_file),
    }
    return head_rows, seed_row, provenance


def plot_modes(head_df: pd.DataFrame, seed_df: pd.DataFrame, output_dir: Path) -> None:
    seeds = seed_df["seed"].astype(int).tolist()
    x = np.arange(len(seeds), dtype=float)
    colors = {
        "low_expression_keys": "#2F6B9A",
        "high_expression_keys": "#D9792B",
    }

    fig, axes = plt.subplots(2, 1, figsize=(12.5, 8.5), sharex=True)
    ax = axes[0]
    offsets = np.linspace(-0.18, 0.18, head_df["head"].nunique())
    head_colors = ["#6A3D9A", "#1B9E77", "#E6AB02", "#A6761D"]
    for offset, (head, group) in zip(offsets, head_df.groupby("head", sort=True)):
        ordered = group.set_index("seed").loc[seeds]
        ax.bar(
            x + offset,
            ordered["tupe_coefficient_c_h"],
            width=0.34,
            color=head_colors[int(head) % len(head_colors)],
            label=f"Head {int(head)}",
            alpha=0.92,
        )
    ax.axhline(0.0, color="#303030", linewidth=1.0)
    ax.set_ylabel(r"TUPE coefficient $c_h$")
    ax.set_title(r"Learned TUPE orientation: $TUPE_h(i,j)=c_h\,x_i x_j$")
    ax.legend(frameon=False, ncol=max(1, head_df["head"].nunique()))
    ax.grid(axis="y", alpha=0.18)

    ax = axes[1]
    rho = seed_df["full_mean_head_vs_macro_expression_spearman"].to_numpy()
    bar_colors = [colors[value] for value in seed_df["observed_key_mode"]]
    bars = ax.bar(x, rho, color=bar_colors, width=0.68)
    ax.axhline(0.0, color="#303030", linewidth=1.0)
    ax.set_ylim(-1.10, 1.10)
    ax.set_ylabel("Spearman ρ\nfull key attention vs expression")
    ax.set_xlabel("Training seed")
    ax.set_xticks(x, [str(seed) for seed in seeds])
    ax.grid(axis="y", alpha=0.18)
    low_count = int((seed_df["observed_key_mode"] == "low_expression_keys").sum())
    high_count = int((seed_df["observed_key_mode"] == "high_expression_keys").sum())
    ax.set_title(
        f"Observed modes on the held-out test split: {low_count} low-expression, "
        f"{high_count} high-expression"
    )
    for bar, value in zip(bars, rho):
        vertical_alignment = "bottom" if value >= 0 else "top"
        offset = 0.035 if value >= 0 else -0.035
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + offset,
            f"{value:+.3f}",
            ha="center",
            va=vertical_alignment,
            fontsize=8,
        )
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=colors["low_expression_keys"]),
        plt.Rectangle((0, 0), 1, 1, color=colors["high_expression_keys"]),
    ]
    ax.legend(
        legend_handles,
        ["Low-expression keys", "High-expression keys"],
        frameon=False,
        loc="center right",
    )

    fig.suptitle(
        "Baseline TxT key-attention diagnostic (10 seeds, test split)",
        fontsize=15,
        y=0.995,
    )
    fig.text(
        0.5,
        0.008,
        "Key-side incoming attention only; query metrics and VMA excluded.",
        ha="center",
        fontsize=9,
        color="#4B4B4B",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.98))
    for suffix in ("png", "pdf"):
        fig.savefig(
            output_dir / f"txt_key_attention_tupe_mode_diagnostics.{suffix}",
            dpi=300 if suffix == "png" else None,
            bbox_inches="tight",
        )
    plt.close(fig)


def write_summary(head_df: pd.DataFrame, seed_df: pd.DataFrame, output_dir: Path) -> None:
    low = seed_df.loc[
        seed_df["observed_key_mode"] == "low_expression_keys", "seed"
    ].astype(int).tolist()
    high = seed_df.loc[
        seed_df["observed_key_mode"] == "high_expression_keys", "seed"
    ].astype(int).tolist()
    ratio = head_df["tupe_to_base_logit_sd_ratio"]
    abs_rho = head_df["spearman_full_vs_macro_expression"].abs()
    lines = [
        "# TxT key-attention TUPE diagnostic",
        "",
        "- Scope: baseline TxT, held-out test split, key-side incoming attention only.",
        "- Query metrics and VMA are excluded.",
        f"- Low-expression-key mode: seeds {', '.join(map(str, low))}.",
        f"- High-expression-key mode: seeds {', '.join(map(str, high))}.",
        (
            "- TUPE/base logit-SD ratio: "
            f"{ratio.min():.1f}× to {ratio.max():.1f}× "
            f"(median {ratio.median():.1f}×)."
        ),
        (
            "- |Spearman ρ|, full head incoming attention vs class-macro "
            f"expression: {abs_rho.min():.4f} to {abs_rho.max():.4f}."
        ),
        "",
        (
            "Interpretation: in this configuration, the key ranking is dominated by "
            "the factorised TUPE expression term. Its learned sign determines whether "
            "high- or low-expression genes receive more incoming attention."
        ),
        "",
    ]
    (output_dir / "TXT_KEY_ATTENTION_TUPE_DIAGNOSTIC.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main(argv: Sequence[str] | None = None) -> int:
    cli = build_parser().parse_args(argv)
    runs_root = resolve_path(cli.runs_root)
    output_dir = resolve_path(cli.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seeds = sorted(set(int(seed) for seed in cli.seeds))
    if not seeds:
        raise ValueError("At least one seed is required.")

    all_head_rows: list[dict[str, Any]] = []
    all_seed_rows: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    table_cache: dict[Path, pd.DataFrame] = {}
    for seed in seeds:
        run_dir = _resolve_seed_run(runs_root, seed, cli.checkpoint_name)
        head_rows, seed_row, seed_provenance = diagnose_seed(
            run_dir,
            seed=seed,
            checkpoint_name=cli.checkpoint_name,
            attention_relative_path=cli.attention_relative_path,
            table_cache=table_cache,
        )
        all_head_rows.extend(head_rows)
        all_seed_rows.append(seed_row)
        provenance.append(seed_provenance)
        print(
            f"seed {seed}: {seed_row['observed_key_mode']}, "
            f"rho={seed_row['full_mean_head_vs_macro_expression_spearman']:+.4f}, "
            f"sum(c_h)={seed_row['tupe_coefficient_sum']:+.4f}",
            flush=True,
        )

    head_df = pd.DataFrame(all_head_rows).sort_values(["seed", "head"])
    seed_df = pd.DataFrame(all_seed_rows).sort_values("seed")
    if not seed_df["coefficient_sign_matches_observed_mode"].all():
        mismatches = seed_df.loc[
            ~seed_df["coefficient_sign_matches_observed_mode"], "seed"
        ].tolist()
        raise RuntimeError(
            "TUPE coefficient sign does not match observed full-attention mode for "
            f"seeds {mismatches}."
        )

    head_csv = output_dir / "txt_key_attention_tupe_diagnostics_by_seed_head.csv"
    seed_csv = output_dir / "txt_key_attention_tupe_modes_by_seed.csv"
    head_df.to_csv(head_csv, index=False)
    seed_df.to_csv(seed_csv, index=False)
    plot_modes(head_df, seed_df, output_dir)
    write_summary(head_df, seed_df, output_dir)

    manifest = {
        "analysis": "baseline_txt_key_attention_tupe_diagnostic",
        "split": "test",
        "seeds": seeds,
        "seed_count": len(seeds),
        "scope": {
            "model_variant": "baseline",
            "attention_side": "key_incoming_only",
            "query_metrics": False,
            "vma": False,
            "checkpoint_mutation": False,
        },
        "formula": {
            "dense_logits": "QK^T / sqrt(2*d_head) + TUPE",
            "tupe_factorization": "TUPE_h[i,j] = c_h*x[i]*x[j]",
            "coefficient": "c_h = dot(w_q_h,w_k_h)/sqrt(2*d_head)",
            "incoming": "G*mean_over_queries(softmax(logits, key_dimension))",
        },
        "outputs": {
            "by_seed_head_csv": str(head_csv.resolve()),
            "by_seed_csv": str(seed_csv.resolve()),
            "plot_png": str(
                (output_dir / "txt_key_attention_tupe_mode_diagnostics.png").resolve()
            ),
            "plot_pdf": str(
                (output_dir / "txt_key_attention_tupe_mode_diagnostics.pdf").resolve()
            ),
            "summary_markdown": str(
                (output_dir / "TXT_KEY_ATTENTION_TUPE_DIAGNOSTIC.md").resolve()
            ),
        },
        "provenance": provenance,
    }
    write_json(output_dir / "txt_key_attention_tupe_diagnostic_manifest.json", manifest)
    print(f"wrote diagnostics to {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
