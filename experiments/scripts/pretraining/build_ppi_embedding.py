#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
import ssl
import sys
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from source.pipeline.utils import save_json, set_seed


DEFAULT_HIPPIE_URL = "https://cbdm-01.zdv.uni-mainz.de/~mschaefer/hippie/HIPPIE-current.mitab.txt"
DEFAULT_RESULT_ROOT = ROOT / "results" / "pretraining" / "ppi_init"
DEFAULT_EDGE_FILE = ROOT / "pretraining_dataset" / "ppi_networks" / "hippie_highconf_edges.csv"
DEFAULT_RAW_FILE = ROOT / "pretraining_dataset" / "ppi_networks" / "raw" / "HIPPIE-current.mitab.txt"
DEFAULT_TARGET_GENE_FILE = ROOT / "task_dataset" / "processed" / "txt_pairwise_multitask" / "shared_ad_mci_ctl" / "X.csv"


@dataclass(frozen=True)
class EdgeBuildResult:
    edges: list[tuple[str, str, float]]
    report: dict[str, object]


class SkipGramNode2Vec(nn.Module):
    def __init__(self, num_nodes: int, embedding_dim: int):
        super().__init__()
        init_bound = 0.5 / embedding_dim
        self.target = nn.Embedding(num_nodes, embedding_dim)
        self.context = nn.Embedding(num_nodes, embedding_dim)
        nn.init.uniform_(self.target.weight, -init_bound, init_bound)
        nn.init.zeros_(self.context.weight)

    def forward(self, src: torch.Tensor, dst: torch.Tensor, neg: torch.Tensor) -> torch.Tensor:
        src_emb = self.target(src)
        dst_emb = self.context(dst)
        pos_score = (src_emb * dst_emb).sum(dim=1)
        pos_loss = nn.functional.logsigmoid(pos_score)

        neg_emb = self.context(neg)
        neg_score = torch.bmm(neg_emb.neg(), src_emb.unsqueeze(2)).squeeze(2)
        neg_loss = nn.functional.logsigmoid(neg_score).sum(dim=1)
        return -(pos_loss + neg_loss).mean()

    def embeddings(self) -> np.ndarray:
        return self.target.weight.detach().cpu().numpy().astype(np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build TxT gene embeddings from a HIPPIE PPI network using node2vec.")
    parser.add_argument("--source", choices=["hippie"], default="hippie")
    parser.add_argument("--hippie-file", type=Path, default=None, help="Local HIPPIE TAB/MITAB file. Downloads current MITAB if omitted.")
    parser.add_argument("--hippie-url", default=DEFAULT_HIPPIE_URL)
    parser.add_argument("--gene-list-file", type=Path, default=DEFAULT_TARGET_GENE_FILE)
    parser.add_argument("--result-dir", type=Path, default=None)
    parser.add_argument("--edge-output-file", type=Path, default=DEFAULT_EDGE_FILE)
    parser.add_argument("--score-threshold", type=float, default=0.73)
    parser.add_argument("--component-policy", choices=["largest", "all"], default="largest")
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--test-ratio", type=float, default=0.10)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--walk-length", type=int, default=20)
    parser.add_argument("--context-size", type=int, default=10)
    parser.add_argument("--walks-per-node", type=int, default=10)
    parser.add_argument("--negative-samples", type=int, default=5)
    parser.add_argument("--p", type=float, default=1.0)
    parser.add_argument("--q", type=float, default=1.0)
    parser.add_argument("--max-pairs-per-epoch", type=int, default=1_000_000)
    parser.add_argument("--device", choices=["cuda", "cpu", "mps"], default="cuda")
    parser.add_argument("--skip-node2vec", action="store_true", help="Only parse/filter HIPPIE and write reports/edge list.")
    return parser.parse_args()


def default_result_dir(seed: int, embedding_dim: int, score_threshold: float) -> Path:
    score_tag = str(score_threshold).replace(".", "p")
    return DEFAULT_RESULT_ROOT / f"hippie_highconf_dim{embedding_dim}_score{score_tag}_seed{seed}"


def resolve_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    if name == "mps" and not torch.backends.mps.is_available():
        return torch.device("cpu")
    return torch.device(name)


def build_download_ssl_context() -> ssl.SSLContext | None:
    try:
        import certifi
    except ImportError:
        return None
    return ssl.create_default_context(cafile=certifi.where())


def download_hippie(url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        return destination
    kwargs: dict[str, object] = {"timeout": 60}
    ssl_context = build_download_ssl_context()
    if ssl_context is not None:
        kwargs["context"] = ssl_context
    try:
        with urllib.request.urlopen(url, **kwargs) as response:
            with destination.open("wb") as handle:
                shutil.copyfileobj(response, handle)
    except urllib.error.URLError as exc:
        if isinstance(exc.reason, ssl.SSLError):
            raise RuntimeError(
                "Failed to download HIPPIE because Python could not verify the HTTPS certificate. "
                "Install/update certifi or pass a local MITAB file with --hippie-file. "
                f"URL: {url}"
            ) from exc
        raise
    return destination


def normalize_gene_symbol(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip().strip('"').strip("'")
    if not text or text in {"-", "NA", "nan", "None"}:
        return None
    if "|" in text:
        text = text.split("|", 1)[0]
    if ":" in text and not text.upper().startswith("LOC"):
        text = text.rsplit(":", 1)[-1]
    if text.endswith("_HUMAN"):
        text = text[: -len("_HUMAN")]
    text = text.upper()
    return text or None


def parse_score(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if ":" in text:
        text = text.rsplit(":", 1)[-1]
    try:
        return float(text)
    except ValueError:
        return None


def is_human_taxid(value: object) -> bool:
    if value is None:
        return True
    text = str(value).lower()
    return not text or text == "-" or "9606" in text or "homo sapiens" in text


def looks_like_header(path: Path) -> bool:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        first = handle.readline()
    lowered = first.lower()
    return "interactor" in lowered or "confidence" in lowered or "gene name" in lowered


def build_edges_from_mitab(path: Path, score_threshold: float) -> EdgeBuildResult:
    df = pd.read_csv(path, sep="\t", dtype=str)
    required = {
        "Gene Name Interactor A",
        "Gene Name Interactor B",
        "Confidence Value",
        "Taxid Interactor A",
        "Taxid Interactor B",
    }
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"HIPPIE MITAB file is missing columns: {', '.join(missing)}")

    counters = Counter()
    by_pair: dict[tuple[str, str], float] = {}
    for record in df.to_dict(orient="records"):
        counters["rows_total"] += 1
        if not is_human_taxid(record["Taxid Interactor A"]) or not is_human_taxid(record["Taxid Interactor B"]):
            counters["rows_non_human"] += 1
            continue
        score = parse_score(record["Confidence Value"])
        if score is None:
            counters["rows_missing_score"] += 1
            continue
        if score < score_threshold:
            counters["rows_below_threshold"] += 1
            continue
        gene_a = normalize_gene_symbol(record["Gene Name Interactor A"])
        gene_b = normalize_gene_symbol(record["Gene Name Interactor B"])
        if gene_a is None or gene_b is None:
            counters["rows_unmapped_gene"] += 1
            continue
        if gene_a == gene_b:
            counters["rows_self_loop"] += 1
            continue
        pair = tuple(sorted((gene_a, gene_b)))
        by_pair[pair] = max(score, by_pair.get(pair, float("-inf")))

    edges = [(a, b, score) for (a, b), score in sorted(by_pair.items())]
    report = dict(counters)
    report.update({"edges_after_dedup": len(edges), "nodes_after_filter": len({g for e in edges for g in e[:2]})})
    return EdgeBuildResult(edges=edges, report=report)


def build_edges_from_hippie_tab(path: Path, score_threshold: float) -> EdgeBuildResult:
    counters = Counter()
    by_pair: dict[tuple[str, str], float] = {}
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row in reader:
            counters["rows_total"] += 1
            if len(row) < 5:
                counters["rows_too_short"] += 1
                continue
            score = parse_score(row[4])
            if score is None:
                counters["rows_missing_score"] += 1
                continue
            if score < score_threshold:
                counters["rows_below_threshold"] += 1
                continue
            gene_a = normalize_gene_symbol(row[0])
            gene_b = normalize_gene_symbol(row[2])
            if gene_a is None or gene_b is None:
                counters["rows_unmapped_gene"] += 1
                continue
            if gene_a == gene_b:
                counters["rows_self_loop"] += 1
                continue
            pair = tuple(sorted((gene_a, gene_b)))
            by_pair[pair] = max(score, by_pair.get(pair, float("-inf")))

    edges = [(a, b, score) for (a, b), score in sorted(by_pair.items())]
    report = dict(counters)
    report.update(
        {
            "format_note": "HIPPIE compact TAB has UniProt-like names, so MITAB is preferred for gene symbols.",
            "edges_after_dedup": len(edges),
            "nodes_after_filter": len({g for e in edges for g in e[:2]}),
        }
    )
    return EdgeBuildResult(edges=edges, report=report)


def build_hippie_edges(path: Path, score_threshold: float) -> EdgeBuildResult:
    if looks_like_header(path):
        return build_edges_from_mitab(path, score_threshold)
    return build_edges_from_hippie_tab(path, score_threshold)


def apply_component_policy(edges_result: EdgeBuildResult, component_policy: str) -> EdgeBuildResult:
    if component_policy == "all":
        report = {
            **edges_result.report,
            "component_policy": "all",
        }
        return EdgeBuildResult(edges=edges_result.edges, report=report)

    if component_policy != "largest":
        raise ValueError(f"Unsupported component policy: {component_policy}")

    graph = nx.Graph()
    graph.add_weighted_edges_from(edges_result.edges)
    components = sorted(nx.connected_components(graph), key=len, reverse=True)
    if not components:
        raise ValueError("No connected components found in PPI graph.")
    largest_nodes = set(components[0])
    filtered_edges = [(a, b, score) for a, b, score in edges_result.edges if a in largest_nodes and b in largest_nodes]
    report = {
        **edges_result.report,
        "component_policy": "largest",
        "connected_components_before_policy": len(components),
        "largest_component_nodes": len(largest_nodes),
        "largest_component_edges": len(filtered_edges),
        "small_component_count": max(len(components) - 1, 0),
        "small_component_max_nodes": max((len(component) for component in components[1:]), default=0),
    }
    return EdgeBuildResult(edges=filtered_edges, report=report)


def write_edges(edges: list[tuple[str, str, float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(edges, columns=["protein1", "protein2", "score"]).to_csv(path, index=False)


def read_target_genes(path: Path | None) -> list[str]:
    if path is None:
        return []
    if not path.exists():
        raise FileNotFoundError(f"Gene list file not found: {path}")
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path, nrows=1)
        genes = []
        for column in df.columns:
            gene = normalize_gene_symbol(column)
            if gene and gene != "SAMPLE_ID":
                genes.append(gene)
        return genes
    genes: list[str] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            for part in line.replace(",", "\t").split("\t"):
                gene = normalize_gene_symbol(part)
                if gene and gene not in {"GENE", "GENES", "SAMPLE_ID"}:
                    genes.append(gene)
    return list(dict.fromkeys(genes))


def split_edges(
    edges: list[tuple[str, str, float]],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]], list[tuple[str, str]]]:
    if val_ratio < 0 or test_ratio < 0 or val_ratio + test_ratio >= 1:
        raise ValueError("val_ratio and test_ratio must be non-negative and sum to less than 1.")
    pairs = [(a, b) for a, b, _ in edges]
    rng = random.Random(seed)
    rng.shuffle(pairs)
    n_total = len(pairs)
    n_val = int(round(n_total * val_ratio))
    n_test = int(round(n_total * test_ratio))
    val = pairs[:n_val]
    test = pairs[n_val : n_val + n_test]
    train = pairs[n_val + n_test :]
    if not train:
        raise ValueError("No training edges available after split.")
    return train, val, test


def build_graph(nodes: Iterable[str], edges: Iterable[tuple[str, str]]) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)
    graph.remove_edges_from(nx.selfloop_edges(graph))
    return graph


def node2vec_walk(graph: nx.Graph, start: str, walk_length: int, p: float, q: float, rng: random.Random) -> list[str]:
    walk = [start]
    while len(walk) < walk_length:
        current = walk[-1]
        neighbors = list(graph.neighbors(current))
        if not neighbors:
            break
        if len(walk) == 1:
            walk.append(rng.choice(neighbors))
            continue
        previous = walk[-2]
        weights = []
        for neighbor in neighbors:
            if neighbor == previous:
                weights.append(1.0 / p)
            elif graph.has_edge(neighbor, previous):
                weights.append(1.0)
            else:
                weights.append(1.0 / q)
        total = sum(weights)
        threshold = rng.random() * total
        cumulative = 0.0
        for neighbor, weight in zip(neighbors, weights):
            cumulative += weight
            if cumulative >= threshold:
                walk.append(neighbor)
                break
    return walk


def iter_walk_pairs(
    graph: nx.Graph,
    node_to_idx: dict[str, int],
    walk_length: int,
    walks_per_node: int,
    context_size: int,
    p: float,
    q: float,
    seed: int,
    max_pairs: int,
) -> Iterable[tuple[int, int]]:
    rng = random.Random(seed)
    nodes = list(graph.nodes())
    emitted = 0
    for _ in range(walks_per_node):
        rng.shuffle(nodes)
        for node in nodes:
            walk = node2vec_walk(graph, node, walk_length, p, q, rng)
            encoded = [node_to_idx[item] for item in walk]
            for center_idx, center in enumerate(encoded):
                left = max(0, center_idx - context_size)
                right = min(len(encoded), center_idx + context_size + 1)
                for context_idx in range(left, right):
                    if context_idx == center_idx:
                        continue
                    yield center, encoded[context_idx]
                    emitted += 1
                    if max_pairs > 0 and emitted >= max_pairs:
                        return


def make_negative_distribution(graph: nx.Graph, node_to_idx: dict[str, int]) -> torch.Tensor:
    degrees = np.zeros(len(node_to_idx), dtype=np.float64)
    for node, degree in graph.degree():
        degrees[node_to_idx[node]] = max(float(degree), 1.0)
    weights = np.power(degrees, 0.75)
    weights /= weights.sum()
    return torch.tensor(weights, dtype=torch.float32)


def train_node2vec(
    graph: nx.Graph,
    embedding_dim: int,
    seed: int,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    walk_length: int,
    walks_per_node: int,
    context_size: int,
    negative_samples: int,
    p: float,
    q: float,
    max_pairs_per_epoch: int,
) -> tuple[pd.DataFrame, list[dict[str, float]], dict[str, int]]:
    nodes = sorted(graph.nodes())
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    model = SkipGramNode2Vec(len(nodes), embedding_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    neg_distribution = make_negative_distribution(graph, node_to_idx).to(device)
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        total_pairs = 0
        batch_src: list[int] = []
        batch_dst: list[int] = []
        pair_iter = iter_walk_pairs(
            graph=graph,
            node_to_idx=node_to_idx,
            walk_length=walk_length,
            walks_per_node=walks_per_node,
            context_size=context_size,
            p=p,
            q=q,
            seed=seed + epoch,
            max_pairs=max_pairs_per_epoch,
        )
        for src, dst in pair_iter:
            batch_src.append(src)
            batch_dst.append(dst)
            if len(batch_src) < batch_size:
                continue
            loss_value = train_pair_batch(model, optimizer, batch_src, batch_dst, neg_distribution, negative_samples, device)
            total_loss += loss_value * len(batch_src)
            total_pairs += len(batch_src)
            batch_src = []
            batch_dst = []
        if batch_src:
            loss_value = train_pair_batch(model, optimizer, batch_src, batch_dst, neg_distribution, negative_samples, device)
            total_loss += loss_value * len(batch_src)
            total_pairs += len(batch_src)
        mean_loss = total_loss / max(total_pairs, 1)
        history.append({"epoch": float(epoch), "training_loss": float(mean_loss), "pairs": float(total_pairs)})
        print(f"Epoch {epoch:03d} | node2vec_loss={mean_loss:.6f} | pairs={total_pairs}", flush=True)

    embedding_df = pd.DataFrame(model.embeddings(), index=nodes)
    embedding_df.index.name = "Gene"
    return embedding_df, history, {"nodes": len(nodes), "node2vec_pairs_last_epoch": int(history[-1]["pairs"]) if history else 0}


def train_pair_batch(
    model: SkipGramNode2Vec,
    optimizer: torch.optim.Optimizer,
    src: list[int],
    dst: list[int],
    neg_distribution: torch.Tensor,
    negative_samples: int,
    device: torch.device,
) -> float:
    src_tensor = torch.tensor(src, dtype=torch.long, device=device)
    dst_tensor = torch.tensor(dst, dtype=torch.long, device=device)
    neg_tensor = torch.multinomial(neg_distribution, len(src) * negative_samples, replacement=True).view(len(src), negative_samples)
    optimizer.zero_grad()
    loss = model(src_tensor, dst_tensor, neg_tensor)
    loss.backward()
    optimizer.step()
    return float(loss.item())


def sample_negative_edges(nodes: list[str], positive_edges: set[tuple[str, str]], count: int, seed: int) -> list[tuple[str, str]]:
    rng = random.Random(seed)
    negatives: set[tuple[str, str]] = set()
    while len(negatives) < count and len(nodes) > 1:
        a, b = rng.sample(nodes, 2)
        pair = tuple(sorted((a, b)))
        if pair not in positive_edges:
            negatives.add(pair)
    return sorted(negatives)


def evaluate_link_prediction(
    embedding_df: pd.DataFrame,
    train_edges: list[tuple[str, str]],
    val_edges: list[tuple[str, str]],
    test_edges: list[tuple[str, str]],
    seed: int,
) -> dict[str, object]:
    if not val_edges or not test_edges:
        return {"link_prediction_note": "Skipped because validation or test split is empty."}
    nodes = embedding_df.index.tolist()
    all_positive = {tuple(sorted(edge)) for edge in train_edges + val_edges + test_edges}
    train_neg = sample_negative_edges(nodes, all_positive, len(train_edges), seed + 10)
    val_neg = sample_negative_edges(nodes, all_positive, len(val_edges), seed + 20)
    test_neg = sample_negative_edges(nodes, all_positive, len(test_edges), seed + 30)
    if not train_neg or not val_neg or not test_neg:
        return {"link_prediction_note": "Skipped because negative sampling failed."}

    x_train, y_train = edge_features(embedding_df, train_edges, train_neg)
    classifier = LogisticRegression(max_iter=1000, random_state=seed).fit(x_train, y_train)
    report: dict[str, object] = {}
    for name, pos, neg in [("val", val_edges, val_neg), ("test", test_edges, test_neg)]:
        x_eval, y_eval = edge_features(embedding_df, pos, neg)
        probabilities = classifier.predict_proba(x_eval)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)
        report[f"{name}_accuracy"] = float(accuracy_score(y_eval, predictions))
        report[f"{name}_auroc"] = float(roc_auc_score(y_eval, probabilities))
    return report


def edge_features(
    embedding_df: pd.DataFrame,
    positive_edges: list[tuple[str, str]],
    negative_edges: list[tuple[str, str]],
) -> tuple[np.ndarray, np.ndarray]:
    rows: list[np.ndarray] = []
    labels: list[int] = []
    for label, edges in [(1, positive_edges), (0, negative_edges)]:
        for a, b in edges:
            rows.append(embedding_df.loc[a].to_numpy(dtype=np.float32) * embedding_df.loc[b].to_numpy(dtype=np.float32))
            labels.append(label)
    return np.vstack(rows), np.asarray(labels, dtype=np.int64)


def remap_embedding_to_target_genes(
    ppi_embedding_df: pd.DataFrame,
    target_genes: list[str],
    embedding_dim: int,
    seed: int,
    init_scale: float = 0.02,
) -> tuple[pd.DataFrame, dict[str, object]]:
    if not target_genes:
        return ppi_embedding_df, {
            "target_gene_source": "graph_nodes",
            "target_genes": int(ppi_embedding_df.shape[0]),
            "matched_genes": int(ppi_embedding_df.shape[0]),
            "missing_genes": 0,
            "missing_gene_names": [],
        }
    rng = np.random.default_rng(seed)
    rows: list[np.ndarray] = []
    missing: list[str] = []
    for gene in target_genes:
        if gene in ppi_embedding_df.index:
            rows.append(ppi_embedding_df.loc[gene].to_numpy(dtype=np.float32, copy=True))
        else:
            missing.append(gene)
            rows.append(rng.normal(0.0, init_scale, size=embedding_dim).astype(np.float32))
    remapped = pd.DataFrame(np.vstack(rows), index=target_genes)
    remapped.index.name = "Gene"
    return remapped, {
        "target_gene_source": "gene_list_file",
        "target_genes": len(target_genes),
        "matched_genes": len(target_genes) - len(missing),
        "missing_genes": len(missing),
        "coverage": float((len(target_genes) - len(missing)) / max(len(target_genes), 1)),
        "missing_gene_names": missing,
    }


def compact_report_for_console(report: dict[str, object], missing_gene_preview: int = 25) -> dict[str, object]:
    compact = json.loads(json.dumps(report))
    target_coverage = compact.get("target_gene_coverage")
    if isinstance(target_coverage, dict):
        missing_names = target_coverage.get("missing_gene_names")
        if isinstance(missing_names, list) and len(missing_names) > missing_gene_preview:
            target_coverage["missing_gene_names_preview"] = missing_names[:missing_gene_preview]
            target_coverage["missing_gene_names_omitted_from_console"] = len(missing_names) - missing_gene_preview
            target_coverage["missing_gene_names"] = "<see ppi_embedding_report.json>"
    return compact


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    result_dir = (
        args.result_dir
        if args.result_dir is not None
        else default_result_dir(args.seed, args.embedding_dim, args.score_threshold)
    )
    result_dir.mkdir(parents=True, exist_ok=True)

    hippie_file = args.hippie_file if args.hippie_file is not None else download_hippie(args.hippie_url, DEFAULT_RAW_FILE)
    edges_result = build_hippie_edges(hippie_file, args.score_threshold)
    edges_result = apply_component_policy(edges_result, args.component_policy)
    if not edges_result.edges:
        raise ValueError("HIPPIE filtering produced zero edges.")
    write_edges(edges_result.edges, args.edge_output_file)

    target_genes = read_target_genes(args.gene_list_file)
    config = {
        **vars(args),
        "hippie_file": str(hippie_file),
        "result_dir": str(result_dir),
        "edge_output_file": str(args.edge_output_file),
        "device_resolved": str(device),
    }
    config = {key: str(value) if isinstance(value, Path) else value for key, value in config.items()}
    save_json(result_dir / "config.json", config)

    all_nodes = sorted({gene for edge in edges_result.edges for gene in edge[:2]})
    train_edges, val_edges, test_edges = split_edges(edges_result.edges, args.val_ratio, args.test_ratio, args.seed)
    graph = build_graph(all_nodes, train_edges)
    split_report = {
        "train_edges": len(train_edges),
        "val_edges": len(val_edges),
        "test_edges": len(test_edges),
        "train_graph_nodes": graph.number_of_nodes(),
        "train_graph_edges": graph.number_of_edges(),
        "connected_components": nx.number_connected_components(graph),
    }

    if args.skip_node2vec:
        rng = np.random.default_rng(args.seed)
        ppi_embedding_df = pd.DataFrame(
            rng.normal(0.0, 0.02, size=(len(all_nodes), args.embedding_dim)).astype(np.float32),
            index=all_nodes,
        )
        ppi_embedding_df.index.name = "Gene"
        history: list[dict[str, float]] = []
        link_prediction_report = {"link_prediction_note": "Skipped because --skip-node2vec was set."}
    else:
        ppi_embedding_df, history, train_report = train_node2vec(
            graph=graph,
            embedding_dim=args.embedding_dim,
            seed=args.seed,
            device=device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            walk_length=args.walk_length,
            walks_per_node=args.walks_per_node,
            context_size=args.context_size,
            negative_samples=args.negative_samples,
            p=args.p,
            q=args.q,
            max_pairs_per_epoch=args.max_pairs_per_epoch,
        )
        split_report.update(train_report)
        link_prediction_report = evaluate_link_prediction(ppi_embedding_df, train_edges, val_edges, test_edges, args.seed)

    ppi_embedding_df.to_csv(result_dir / "ppi_node_embedding.csv")
    pd.DataFrame(history).to_csv(result_dir / "loss.csv", index=False)
    gene_embedding_df, coverage_report = remap_embedding_to_target_genes(
        ppi_embedding_df=ppi_embedding_df,
        target_genes=target_genes,
        embedding_dim=args.embedding_dim,
        seed=args.seed,
    )
    gene_embedding_df.to_csv(result_dir / "gene_embedding.csv")

    report = {
        "source": args.source,
        "score_threshold": args.score_threshold,
        "embedding_dim": args.embedding_dim,
        "hippie": edges_result.report,
        "split": split_report,
        "target_gene_coverage": coverage_report,
        "link_prediction": link_prediction_report,
        "outputs": {
            "edge_file": str(args.edge_output_file),
            "ppi_node_embedding": str(result_dir / "ppi_node_embedding.csv"),
            "gene_embedding": str(result_dir / "gene_embedding.csv"),
            "loss": str(result_dir / "loss.csv"),
        },
    }
    save_json(result_dir / "ppi_embedding_report.json", report)
    print(json.dumps(compact_report_for_console(report), indent=2))


if __name__ == "__main__":
    main()
