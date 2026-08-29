from __future__ import annotations

import csv
import hashlib
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd
import torch


def normalize_gene_symbol(value: object) -> str:
    """Return the case-insensitive symbol used to align HIPPIE and TxT genes."""

    if value is None:
        return ""
    return str(value).strip().strip('"').strip("'").upper()


def _canonical_column_name(value: str) -> str:
    return "".join(character for character in value.lower() if character.isalnum())


def _resolve_column(fieldnames: Sequence[str], candidates: Sequence[str], label: str) -> str:
    available = {_canonical_column_name(name): name for name in fieldnames}
    for candidate in candidates:
        match = available.get(_canonical_column_name(candidate))
        if match is not None:
            return match
    raise ValueError(
        f"PPI edge file is missing the {label} column. "
        f"Available columns: {', '.join(fieldnames)}"
    )


def validate_edge_index(edge_index: torch.Tensor, n_nodes: int) -> torch.Tensor:
    """Validate and normalize a directed ``[destination, source]`` edge index."""

    edge_index = torch.as_tensor(edge_index, dtype=torch.long)
    if edge_index.ndim != 2 or edge_index.size(0) != 2:
        raise ValueError(
            "edge_index must have shape (2, n_directed_edges), with row 0 "
            "containing destinations and row 1 containing sources."
        )
    if n_nodes < 0:
        raise ValueError("n_nodes must be non-negative.")
    if edge_index.numel() > 0:
        minimum = int(edge_index.min().item())
        maximum = int(edge_index.max().item())
        if minimum < 0 or maximum >= n_nodes:
            raise ValueError(
                f"edge_index contains node indices outside [0, {n_nodes}): "
                f"min={minimum}, max={maximum}."
            )
        self_loops = edge_index[0] == edge_index[1]
        if bool(self_loops.any()):
            raise ValueError("edge_index must not contain self-loops.")
    return edge_index.contiguous()


@dataclass(frozen=True)
class InducedPPIGraph:
    """HIPPIE subgraph aligned to the exact order of a selected TxT gene list.

    ``edge_index`` is directed and uses the message-passing convention
    ``edge_index[0] = destination i`` and ``edge_index[1] = source j``.  A
    canonical graph loaded by :func:`load_induced_ppi_graph` contains both
    directions for every undirected HIPPIE interaction.
    """

    genes: tuple[str, ...]
    edge_index: torch.Tensor
    edge_scores: torch.Tensor
    degrees: torch.Tensor
    source_path: str
    source_sha256: str
    score_threshold: float
    rows_total: int = 0
    rows_passing_threshold: int = 0
    rows_induced: int = 0
    duplicate_rows_removed: int = 0
    self_loops_removed: int = 0

    def __post_init__(self) -> None:
        edge_index = validate_edge_index(self.edge_index, len(self.genes))
        edge_scores = torch.as_tensor(self.edge_scores, dtype=torch.float32).flatten().contiguous()
        degrees = torch.as_tensor(self.degrees, dtype=torch.long).flatten().contiguous()
        if edge_scores.numel() != edge_index.size(1):
            raise ValueError("edge_scores must contain one score per directed edge.")
        if degrees.numel() != len(self.genes):
            raise ValueError("degrees must contain one entry per selected gene.")
        object.__setattr__(self, "edge_index", edge_index)
        object.__setattr__(self, "edge_scores", edge_scores)
        object.__setattr__(self, "degrees", degrees)

    @property
    def n_nodes(self) -> int:
        return len(self.genes)

    @property
    def n_directed_edges(self) -> int:
        return int(self.edge_index.size(1))

    @property
    def n_undirected_edges(self) -> int:
        if self.n_directed_edges == 0:
            return 0
        pairs = {
            tuple(sorted((int(destination), int(source))))
            for destination, source in self.edge_index.transpose(0, 1).tolist()
        }
        return len(pairs)

    @property
    def covered_node_count(self) -> int:
        return int((self.degrees > 0).sum().item())

    @property
    def isolated_node_count(self) -> int:
        return self.n_nodes - self.covered_node_count

    @property
    def isolated_indices(self) -> list[int]:
        return (self.degrees == 0).nonzero(as_tuple=False).flatten().tolist()

    @property
    def isolated_genes(self) -> list[str]:
        return [self.genes[index] for index in self.isolated_indices]

    def to_edge_frame(self, *, directed: bool = True) -> pd.DataFrame:
        """Return the induced graph in a stable, artifact-friendly table."""

        rows: list[dict[str, object]] = []
        seen: set[tuple[int, int]] = set()
        for edge_offset, (destination, source) in enumerate(self.edge_index.transpose(0, 1).tolist()):
            destination = int(destination)
            source = int(source)
            if not directed:
                pair = tuple(sorted((destination, source)))
                if pair in seen:
                    continue
                seen.add(pair)
            rows.append(
                {
                    "target_index": destination,
                    "source_index": source,
                    "target_gene": self.genes[destination],
                    "source_gene": self.genes[source],
                    "score": float(self.edge_scores[edge_offset].item()),
                }
            )
        return pd.DataFrame(
            rows,
            columns=["target_index", "source_index", "target_gene", "source_gene", "score"],
        )

    def to_manifest(self) -> dict[str, object]:
        degree_distribution = Counter(int(degree) for degree in self.degrees.tolist())
        scores = self.edge_scores
        return {
            "source_path": self.source_path,
            "source_sha256": self.source_sha256,
            "score_threshold": float(self.score_threshold),
            "threshold_is_inclusive": True,
            "n_nodes": self.n_nodes,
            "n_directed_edges": self.n_directed_edges,
            "n_undirected_edges": self.n_undirected_edges,
            "covered_node_count": self.covered_node_count,
            "isolated_node_count": self.isolated_node_count,
            "isolated_indices": self.isolated_indices,
            "isolated_genes": self.isolated_genes,
            "degree_distribution": {str(key): degree_distribution[key] for key in sorted(degree_distribution)},
            "score_min": float(scores.min().item()) if scores.numel() else None,
            "score_max": float(scores.max().item()) if scores.numel() else None,
            "rows_total": self.rows_total,
            "rows_passing_threshold": self.rows_passing_threshold,
            "rows_induced": self.rows_induced,
            "duplicate_rows_removed": self.duplicate_rows_removed,
            "self_loops_removed": self.self_loops_removed,
            "edge_index_convention": "row0_destination_i__row1_source_j",
        }


def induced_graph_from_edge_index(
    gene_list: Sequence[str],
    edge_index: torch.Tensor,
    edge_scores: torch.Tensor | None = None,
    *,
    source_path: str = "",
    source_sha256: str = "",
    score_threshold: float = 0.73,
) -> InducedPPIGraph:
    """Build graph metadata around an already directed, aligned edge index."""

    genes = tuple(normalize_gene_symbol(gene) for gene in gene_list)
    if any(not gene for gene in genes):
        raise ValueError("gene_list contains an empty gene symbol.")
    if len(set(genes)) != len(genes):
        raise ValueError("gene_list contains duplicate symbols after uppercase normalization.")
    edge_index = validate_edge_index(edge_index, len(genes))
    if edge_scores is None:
        scores = torch.ones(edge_index.size(1), dtype=torch.float32)
    else:
        scores = torch.as_tensor(edge_scores, dtype=torch.float32).flatten()
    degrees = torch.bincount(edge_index[0], minlength=len(genes)) if edge_index.numel() else torch.zeros(len(genes), dtype=torch.long)
    return InducedPPIGraph(
        genes=genes,
        edge_index=edge_index,
        edge_scores=scores,
        degrees=degrees,
        source_path=source_path,
        source_sha256=source_sha256,
        score_threshold=score_threshold,
    )


def load_induced_ppi_graph(
    edge_file: str | Path,
    gene_list: Sequence[str],
    score_threshold: float = 0.73,
) -> InducedPPIGraph:
    """Parse, filter, deduplicate, symmetrize and align a HIPPIE edge CSV."""

    path = Path(edge_file).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"PPI edge file not found: {path}")
    if not math.isfinite(score_threshold):
        raise ValueError("score_threshold must be finite.")

    genes = tuple(normalize_gene_symbol(gene) for gene in gene_list)
    if any(not gene for gene in genes):
        raise ValueError("gene_list contains an empty gene symbol.")
    if len(set(genes)) != len(genes):
        raise ValueError("gene_list contains duplicate symbols after uppercase normalization.")
    gene_to_index = {gene: index for index, gene in enumerate(genes)}

    source_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    rows_total = 0
    rows_passing_threshold = 0
    rows_induced = 0
    duplicate_rows_removed = 0
    self_loops_removed = 0
    # The maximum retained confidence is only metadata: VMA uses binary edges.
    undirected_scores: dict[tuple[int, int], float] = {}

    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"PPI edge file has no header: {path}")
        gene_a_column = _resolve_column(
            reader.fieldnames,
            ("protein1", "gene1", "gene_a", "source", "Gene Name Interactor A"),
            "first protein",
        )
        gene_b_column = _resolve_column(
            reader.fieldnames,
            ("protein2", "gene2", "gene_b", "target", "Gene Name Interactor B"),
            "second protein",
        )
        score_column = _resolve_column(
            reader.fieldnames,
            ("score", "confidence", "confidence_value", "Confidence Value"),
            "confidence score",
        )

        for row in reader:
            rows_total += 1
            try:
                score = float(str(row.get(score_column, "")).strip())
            except (TypeError, ValueError):
                continue
            if not math.isfinite(score) or score < score_threshold:
                continue
            rows_passing_threshold += 1
            gene_a = normalize_gene_symbol(row.get(gene_a_column))
            gene_b = normalize_gene_symbol(row.get(gene_b_column))
            if not gene_a or not gene_b:
                continue
            if gene_a == gene_b:
                self_loops_removed += 1
                continue
            index_a = gene_to_index.get(gene_a)
            index_b = gene_to_index.get(gene_b)
            if index_a is None or index_b is None:
                continue
            rows_induced += 1
            pair = tuple(sorted((index_a, index_b)))
            if pair in undirected_scores:
                duplicate_rows_removed += 1
                undirected_scores[pair] = max(undirected_scores[pair], score)
            else:
                undirected_scores[pair] = score

    directed_edges: list[tuple[int, int, float]] = []
    for (index_a, index_b), score in undirected_scores.items():
        directed_edges.append((index_a, index_b, score))
        directed_edges.append((index_b, index_a, score))
    directed_edges.sort(key=lambda edge: (edge[0], edge[1]))

    if directed_edges:
        edge_index = torch.tensor(
            [[edge[0] for edge in directed_edges], [edge[1] for edge in directed_edges]],
            dtype=torch.long,
        )
        edge_scores = torch.tensor([edge[2] for edge in directed_edges], dtype=torch.float32)
        degrees = torch.bincount(edge_index[0], minlength=len(genes))
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_scores = torch.empty((0,), dtype=torch.float32)
        degrees = torch.zeros(len(genes), dtype=torch.long)

    return InducedPPIGraph(
        genes=genes,
        edge_index=edge_index,
        edge_scores=edge_scores,
        degrees=degrees,
        source_path=str(path),
        source_sha256=source_sha256,
        score_threshold=float(score_threshold),
        rows_total=rows_total,
        rows_passing_threshold=rows_passing_threshold,
        rows_induced=rows_induced,
        duplicate_rows_removed=duplicate_rows_removed,
        self_loops_removed=self_loops_removed,
    )


# Descriptive aliases retained for launchers/tests written during development.
build_induced_ppi_graph = load_induced_ppi_graph
load_hippie_induced_graph = load_induced_ppi_graph


__all__ = [
    "InducedPPIGraph",
    "build_induced_ppi_graph",
    "induced_graph_from_edge_index",
    "load_hippie_induced_graph",
    "load_induced_ppi_graph",
    "normalize_gene_symbol",
    "validate_edge_index",
]
