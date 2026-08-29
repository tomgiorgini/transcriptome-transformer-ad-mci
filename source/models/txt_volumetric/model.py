from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

import torch

from source.models.txt.model import TxT

from .graph import (
    InducedPPIGraph,
    induced_graph_from_edge_index,
    load_induced_ppi_graph,
    normalize_gene_symbol,
)
from .layers import VolumetricAttentionAugmentation


class TxTVolumetric(TxT):
    """TxT with a zero-initialized residual PPI volumetric attention branch.

    The complete baseline ``TxT`` is initialized by ``super().__init__`` before
    any augmentation parameter is created.  Resetting the same RNG seed before
    constructing a baseline and this class therefore produces identical
    baseline parameters; with ``volumetric_gate_init=0`` their evaluation
    logits are exactly identical as well.
    """

    def __init__(
        self,
        embed_file: str,
        gene_list: list[str],
        n_heads: int = 8,
        d_model: int = 512,
        dropout: float = 0.1,
        d_ff: int = 2048,
        norm_first: bool = False,
        n_layers: int = 6,
        aggfunc: str = "Flatten",
        d_hidden1: int = 128,
        d_hidden2: int = 64,
        slope: float = 0.2,
        d_output_dict: dict[str, int] | None = None,
        task_gene_indices: dict[str, list[int]] | None = None,
        head_norm: str = "batch",
        encoder_sharing: str = "shared",
        pooling_mode: str = "average",
        attention_pooling_hidden_dim: int = 16,
        attention_pooling_dropout: float = 0.1,
        primary_adapter_dim: int = 0,
        expression_residual: str = "none",
        ppi_prior_file: str | None = None,
        ppi_gate_init: float = 0.0,
        tupe_mode: str = "on",
        *,
        edge_index: torch.Tensor | None = None,
        ppi_edge_scores: torch.Tensor | None = None,
        ppi_graph: InducedPPIGraph | None = None,
        ppi_edge_file: str | Path | None = None,
        ppi_score_threshold: float = 0.73,
        volumetric_beta: float = 1.0,
        volumetric_eps: float = 1e-8,
        volumetric_gate_init: float = 0.0,
        volumetric_dropout: float | None = None,
        volumetric_volume_mode: str = "raw",
        volumetric_message_mode: str = "legacy",
        volumetric_output_norm: str = "none",
        volumetric_gate_mode: str = "scalar",
        volumetric_backbone_gradient_mode: str = "coupled",
    ):
        # Do not move graph parsing or augmentation construction above this
        # call: paired initialization relies on the complete baseline consuming
        # precisely the same random-number sequence as TxT.
        super().__init__(
            embed_file=embed_file,
            gene_list=gene_list,
            n_heads=n_heads,
            d_model=d_model,
            dropout=dropout,
            d_ff=d_ff,
            norm_first=norm_first,
            n_layers=n_layers,
            aggfunc=aggfunc,
            d_hidden1=d_hidden1,
            d_hidden2=d_hidden2,
            slope=slope,
            d_output_dict=d_output_dict,
            task_gene_indices=task_gene_indices,
            head_norm=head_norm,
            encoder_sharing=encoder_sharing,
            pooling_mode=pooling_mode,
            attention_pooling_hidden_dim=attention_pooling_hidden_dim,
            attention_pooling_dropout=attention_pooling_dropout,
            primary_adapter_dim=primary_adapter_dim,
            expression_residual=expression_residual,
            ppi_prior_file=ppi_prior_file,
            ppi_gate_init=ppi_gate_init,
            tupe_mode=tupe_mode,
        )

        supplied_graph_sources = sum(
            source is not None for source in (edge_index, ppi_graph, ppi_edge_file)
        )
        if supplied_graph_sources != 1:
            raise ValueError(
                "Provide exactly one of edge_index, ppi_graph or ppi_edge_file "
                "when constructing TxTVolumetric."
            )
        if ppi_graph is not None:
            if tuple(normalize_gene_symbol(gene) for gene in gene_list) != ppi_graph.genes:
                raise ValueError("ppi_graph genes are not aligned to gene_list in the same order.")
            graph = ppi_graph
        elif ppi_edge_file is not None:
            graph = load_induced_ppi_graph(ppi_edge_file, gene_list, ppi_score_threshold)
        else:
            assert edge_index is not None
            graph = induced_graph_from_edge_index(
                gene_list,
                edge_index,
                ppi_edge_scores,
                score_threshold=ppi_score_threshold,
            )

        self.register_buffer("ppi_edge_index", graph.edge_index.clone(), persistent=True)
        self.register_buffer("ppi_edge_scores", graph.edge_scores.clone(), persistent=True)
        self.register_buffer("ppi_degrees", graph.degrees.clone(), persistent=True)
        self.ppi_graph_metadata = graph.to_manifest()
        self.ppi_source_path = graph.source_path
        self.ppi_source_sha256 = graph.source_sha256
        self.ppi_score_threshold = float(graph.score_threshold)
        self.volumetric_beta = float(volumetric_beta)
        self.volumetric_eps = float(volumetric_eps)
        self.volumetric_gate_init = float(volumetric_gate_init)
        self.volumetric_dropout = float(dropout if volumetric_dropout is None else volumetric_dropout)
        self.volumetric_volume_mode = volumetric_volume_mode
        self.volumetric_message_mode = volumetric_message_mode
        self.volumetric_output_norm = volumetric_output_norm
        self.volumetric_gate_mode = volumetric_gate_mode
        self.volumetric_backbone_gradient_mode = volumetric_backbone_gradient_mode

        self._volumetric_layer_names: list[str] = []
        encoder_names = ["shared"] if self.encoder_sharing == "shared" else list(self.task_names)
        for encoder_name, transformer in zip(encoder_names, self.encoder_modules()):
            for layer_index, encoder_layer in enumerate(transformer.encoder.layers):
                attention_layer = encoder_layer.multi_head_attention_layer
                augmentation = VolumetricAttentionAugmentation(
                    n_heads=attention_layer.n_heads,
                    d_head=attention_layer.d_head,
                    edge_index=self.ppi_edge_index,
                    n_nodes=len(gene_list),
                    beta=volumetric_beta,
                    eps=volumetric_eps,
                    dropout=self.volumetric_dropout,
                    gate_init=volumetric_gate_init,
                    volumetric_volume_mode=volumetric_volume_mode,
                    volumetric_message_mode=volumetric_message_mode,
                    volumetric_output_norm=volumetric_output_norm,
                    volumetric_gate_mode=volumetric_gate_mode,
                    volumetric_backbone_gradient_mode=volumetric_backbone_gradient_mode,
                )
                attention_layer.attention_augmentation = augmentation
                self._volumetric_layer_names.append(f"{encoder_name}.layer_{layer_index}")

    def iter_volumetric_augmentations(
        self,
    ) -> Iterator[tuple[str, VolumetricAttentionAugmentation]]:
        encoder_names = ["shared"] if self.encoder_sharing == "shared" else list(self.task_names)
        for encoder_name, transformer in zip(encoder_names, self.encoder_modules()):
            for layer_index, encoder_layer in enumerate(transformer.encoder.layers):
                augmentation = encoder_layer.multi_head_attention_layer.attention_augmentation
                if not isinstance(augmentation, VolumetricAttentionAugmentation):
                    raise RuntimeError("TxTVolumetric attention augmentation was detached or replaced.")
                yield f"{encoder_name}.layer_{layer_index}", augmentation

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        gene_indices: torch.Tensor | None = None,
        task_sample_mask: torch.Tensor | None = None,
        expression_valid_mask: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        if self.volumetric_message_mode != "expression_contrast":
            return super().forward(x, mask, gene_indices, task_sample_mask)

        finite_mask = torch.isfinite(x)
        valid_mask = finite_mask
        if expression_valid_mask is not None:
            supplied = torch.as_tensor(
                expression_valid_mask,
                device=x.device,
                dtype=torch.bool,
            )
            try:
                supplied = torch.broadcast_to(supplied, x.shape)
            except RuntimeError as exc:
                raise ValueError(
                    "expression_valid_mask is not broadcastable to the model input."
                ) from exc
            valid_mask = valid_mask & supplied

        # Keep legacy behavior untouched, but prevent non-finite expression
        # from contaminating TUPE/dense attention in the V2 path.  The original
        # finite mask remains authoritative for VMA standardization.
        safe_x = torch.where(finite_mask, x, torch.zeros_like(x))

        augmentations = [augmentation for _, augmentation in self.iter_volumetric_augmentations()]
        for augmentation in augmentations:
            augmentation.set_expression_context(safe_x, valid_mask)
        try:
            return super().forward(safe_x, mask, gene_indices, task_sample_mask)
        finally:
            for augmentation in augmentations:
                augmentation.clear_expression_context()

    def volumetric_gate_values(self) -> dict[str, float]:
        """Return effective residual gates, i.e. ``tanh(gamma_l)``."""

        values: dict[str, float] = {}
        for name, augmentation in self.iter_volumetric_augmentations():
            gate = torch.tanh(augmentation.gamma.detach()).flatten().cpu()
            if augmentation.gate_mode == "scalar":
                values[name] = float(gate.item())
            else:
                for head_index, value in enumerate(gate):
                    values[f"{name}.head_{head_index}"] = float(value.item())
        return values

    def volumetric_raw_gate_values(self) -> dict[str, float]:
        values: dict[str, float] = {}
        for name, augmentation in self.iter_volumetric_augmentations():
            gate = augmentation.gamma.detach().flatten().cpu()
            if augmentation.gate_mode == "scalar":
                values[name] = float(gate.item())
            else:
                for head_index, value in enumerate(gate):
                    values[f"{name}.head_{head_index}"] = float(value.item())
        return values

    def volumetric_diagnostics(self) -> dict[str, dict[str, float | int | str]]:
        return {
            name: augmentation.diagnostics()
            for name, augmentation in self.iter_volumetric_augmentations()
        }

    def set_volumetric_attention_capture(self, enabled: bool, *, clear: bool = True) -> None:
        """Enable per-sample sparse weight capture for post-hoc evaluation only."""

        for _, augmentation in self.iter_volumetric_augmentations():
            augmentation.set_attention_capture(enabled, clear=clear)

    def volumetric_attention_captures(self, *, clear: bool = False) -> list[dict[str, Any]]:
        captures: list[dict[str, Any]] = []
        for name, augmentation in self.iter_volumetric_augmentations():
            capture = augmentation.attention_capture(clear=clear)
            if capture is not None:
                captures.append({"name": name, **capture})
        return captures

    def clear_volumetric_attention_captures(self) -> None:
        for _, augmentation in self.iter_volumetric_augmentations():
            augmentation.attention_capture(clear=True)

    def induced_graph_manifest(self) -> dict[str, object]:
        """Return checkpoint-aligned graph statistics for run artifacts."""

        manifest = dict(self.ppi_graph_metadata)
        manifest.update(
            {
                "source_path": self.ppi_source_path,
                "source_sha256": self.ppi_source_sha256,
                "score_threshold": self.ppi_score_threshold,
                "n_directed_edges": int(self.ppi_edge_index.size(1)),
                "covered_node_count": int((self.ppi_degrees > 0).sum().item()),
                "isolated_node_count": int((self.ppi_degrees == 0).sum().item()),
            }
        )
        return manifest

    # Preserve non-tensor provenance in ordinary state_dict checkpoints while
    # graph topology/scores/degrees remain regular strict-loadable buffers.
    def get_extra_state(self) -> dict[str, object]:
        return {
            "ppi_graph_metadata": self.ppi_graph_metadata,
            "ppi_source_path": self.ppi_source_path,
            "ppi_source_sha256": self.ppi_source_sha256,
            "ppi_score_threshold": self.ppi_score_threshold,
            "tupe_mode": self.encoder_modules()[0].encoder.tupe_mode,
            "volumetric_volume_mode": self.volumetric_volume_mode,
            "volumetric_message_mode": self.volumetric_message_mode,
            "volumetric_output_norm": self.volumetric_output_norm,
            "volumetric_gate_mode": self.volumetric_gate_mode,
            "volumetric_backbone_gradient_mode": self.volumetric_backbone_gradient_mode,
        }

    def set_extra_state(self, state: dict[str, object]) -> None:
        self.ppi_graph_metadata = dict(state.get("ppi_graph_metadata", {}))
        self.ppi_source_path = str(state.get("ppi_source_path", ""))
        self.ppi_source_sha256 = str(state.get("ppi_source_sha256", ""))
        self.ppi_score_threshold = float(state.get("ppi_score_threshold", 0.73))
        volume_mode = str(state.get("volumetric_volume_mode", "raw"))
        loaded_modes = {
            "tupe_mode": str(state.get("tupe_mode", "on")),
            "volumetric_volume_mode": volume_mode,
            "volumetric_message_mode": str(state.get("volumetric_message_mode", "legacy")),
            "volumetric_output_norm": str(state.get("volumetric_output_norm", "none")),
            "volumetric_gate_mode": str(state.get("volumetric_gate_mode", "scalar")),
            "volumetric_backbone_gradient_mode": str(
                state.get("volumetric_backbone_gradient_mode", "coupled")
            ),
        }
        for attribute, loaded_value in loaded_modes.items():
            configured_value = str(
                self.encoder_modules()[0].encoder.tupe_mode
                if attribute == "tupe_mode"
                else getattr(self, attribute)
            )
            if loaded_value != configured_value:
                raise RuntimeError(
                    f"Checkpoint {attribute}={loaded_value!r} does not match the "
                    f"constructed model value {configured_value!r}."
                )
        for _, augmentation in self.iter_volumetric_augmentations():
            augmentation.set_volume_mode(volume_mode)
        self.volumetric_volume_mode = volume_mode


__all__ = ["TxTVolumetric"]
