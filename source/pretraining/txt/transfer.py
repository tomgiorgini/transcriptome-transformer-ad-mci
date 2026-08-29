from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def build_random_embedding_dataframe(
    gene_names: list[str],
    embed_dim: int,
    seed: int,
    init_scale: float = 0.02,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    values = rng.normal(0.0, init_scale, size=(len(gene_names), embed_dim)).astype(np.float32)
    embed_df = pd.DataFrame(values, index=gene_names)
    embed_df.index.name = "Gene"
    return embed_df


def remap_embedding_dataframe(
    source_df: pd.DataFrame,
    target_genes: list[str],
    seed: int,
    init_scale: float = 0.02,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if source_df.empty:
        raise ValueError("Cannot remap an empty embedding dataframe.")

    source_df = source_df.copy()
    source_df.index = source_df.index.astype(str).str.strip()
    source_df = source_df.loc[~source_df.index.duplicated(keep="first")]
    source_df = source_df.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    rng = np.random.default_rng(seed)
    rows: list[np.ndarray] = []
    missing: list[str] = []
    for gene in target_genes:
        if gene in source_df.index:
            rows.append(source_df.loc[gene].to_numpy(dtype=np.float32, copy=True))
        else:
            missing.append(gene)
            rows.append(rng.normal(0.0, init_scale, size=source_df.shape[1]).astype(np.float32))

    remapped = pd.DataFrame(np.vstack(rows), index=target_genes)
    remapped.index.name = "Gene"
    report: dict[str, Any] = {
        "target_genes": len(target_genes),
        "source_genes": int(source_df.shape[0]),
        "embedding_dim": int(source_df.shape[1]),
        "matched_genes": len(target_genes) - len(missing),
        "missing_genes": len(missing),
        "missing_gene_names": missing,
    }
    return remapped, report
