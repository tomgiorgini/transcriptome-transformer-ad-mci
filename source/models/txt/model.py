from __future__ import annotations

import re

import pandas as pd
import torch
import torch.nn as nn

from source.pipeline.utils import get_clones

from .embed import Embedder, TUPE_A
from .layers import EncoderLayer, TaskSpecificLayer


class TaskAttentionPooling(nn.Module):
    def __init__(self, d_embed: int, hidden_dim: int = 16, dropout: float = 0.1):
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("attention pooling hidden_dim must be positive.")
        self.projection = nn.Linear(d_embed, hidden_dim)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(dropout)
        self.score = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, encoded: torch.Tensor) -> torch.Tensor:
        scores = self.score(self.dropout(self.activation(self.projection(encoded)))).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        return torch.sum(encoded * weights.unsqueeze(-1), dim=1)


class ResidualAdapter(nn.Module):
    def __init__(self, d_embed: int, bottleneck_dim: int, dropout: float = 0.1):
        super().__init__()
        if bottleneck_dim <= 0:
            raise ValueError("adapter bottleneck_dim must be positive.")
        self.down = nn.Linear(d_embed, bottleneck_dim, bias=False)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.up = nn.Linear(bottleneck_dim, d_embed, bias=False)
        # The adapter starts as an exact identity residual branch.
        nn.init.zeros_(self.up.weight)

    def forward(self, encoded: torch.Tensor) -> torch.Tensor:
        return encoded + self.up(self.dropout(self.activation(self.down(encoded))))


class Encoder(nn.Module):
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
        expression_residual: str = "none",
        ppi_prior_file: str | None = None,
        ppi_gate_init: float = 0.0,
        tupe_mode: str = "on",
    ):
        super().__init__()
        if expression_residual not in {"none", "additive_zero_init"}:
            raise ValueError(f"Unsupported expression_residual: {expression_residual}")
        if tupe_mode not in {"on", "off"}:
            raise ValueError(f"Unsupported tupe_mode: {tupe_mode}")
        self.embed = Embedder(embed_file, gene_list)
        self.d_embed = self.embed.embed.embedding_dim
        if ppi_prior_file is not None:
            ppi_df = pd.read_csv(ppi_prior_file, index_col=0).loc[gene_list]
            ppi_prior = torch.tensor(ppi_df.values, dtype=torch.float32)
            if ppi_prior.shape[1] != self.d_embed:
                raise ValueError(
                    f"PPI prior dimension {ppi_prior.shape[1]} does not match embedding dimension {self.d_embed}."
                )
            self.register_buffer("ppi_prior", ppi_prior, persistent=True)
            self.ppi_gate = nn.Parameter(torch.tensor(float(ppi_gate_init)))
        else:
            self.register_buffer("ppi_prior", None, persistent=False)
            self.register_parameter("ppi_gate", None)
        self.expression_residual = expression_residual
        if expression_residual == "additive_zero_init":
            self.expression_projection = nn.Linear(1, self.d_embed)
            self.expression_scale = nn.Parameter(torch.tensor(0.0))
        else:
            self.expression_projection = None
            self.register_parameter("expression_scale", None)
        self.tupe = TUPE_A(n_heads, d_model)
        # TUPE remains constructed even when disabled so state-dict structure,
        # parameter initialization and historical checkpoint loading stay
        # unchanged.  The ablation replaces only its forward contribution.
        self.tupe_mode = tupe_mode
        self.layers = get_clones(EncoderLayer(n_heads, d_model, self.d_embed, dropout, d_ff, norm_first), n_layers)
        self.layer_norm = nn.LayerNorm(self.d_embed)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        gene_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, n_genes = x.size()
        if gene_indices is None:
            gene_indices = torch.arange(n_genes, device=x.device).repeat(batch_size).view(batch_size, -1)
        embeddings = self.embed(gene_indices)
        if self.ppi_prior is not None and self.ppi_gate is not None:
            embeddings = embeddings + torch.tanh(self.ppi_gate) * self.ppi_prior[gene_indices]
        if self.expression_projection is not None:
            embeddings = embeddings + self.expression_scale * self.expression_projection(x.unsqueeze(-1))
        if self.tupe_mode == "on":
            tupe = self.tupe(x)
        else:
            tupe = x.new_zeros((batch_size, self.tupe.n_heads, n_genes, n_genes))
        for layer in self.layers:
            embeddings = layer(embeddings, tupe, mask, expression=x)
        return self.layer_norm(embeddings)


class Transformer(nn.Module):
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
        expression_residual: str = "none",
        ppi_prior_file: str | None = None,
        ppi_gate_init: float = 0.0,
        tupe_mode: str = "on",
    ):
        super().__init__()
        self.encoder = Encoder(
            embed_file,
            gene_list,
            n_heads,
            d_model,
            dropout,
            d_ff,
            norm_first,
            n_layers,
            expression_residual,
            ppi_prior_file,
            ppi_gate_init,
            tupe_mode,
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        gene_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.encoder(x, mask, gene_indices)


class TxT(nn.Module):
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
    ):
        super().__init__()
        if d_output_dict is None:
            raise ValueError("d_output_dict is required.")
        if encoder_sharing not in {"shared", "separate"}:
            raise ValueError(f"Unsupported encoder_sharing: {encoder_sharing}")
        if pooling_mode not in {"average", "task_attention"}:
            raise ValueError(f"Unsupported pooling_mode: {pooling_mode}")
        if pooling_mode == "task_attention" and aggfunc != "Avgpool":
            raise ValueError("task_attention pooling requires aggfunc='Avgpool'.")
        if task_gene_indices is not None and aggfunc != "Avgpool":
            raise ValueError("task_gene_indices is currently supported only with aggfunc='Avgpool'.")

        self.encoder_sharing = encoder_sharing
        self.task_names = list(d_output_dict.keys())
        if encoder_sharing == "shared":
            self.transformer = Transformer(
                embed_file,
                gene_list,
                n_heads,
                d_model,
                dropout,
                d_ff,
                norm_first,
                n_layers,
                expression_residual,
                ppi_prior_file,
                ppi_gate_init,
                tupe_mode,
            )
            self.task_transformers = None
            self.d_embed = self.transformer.encoder.d_embed
        else:
            self.transformer = None
            self.task_transformers = nn.ModuleList(
                [
                    Transformer(
                        embed_file,
                        gene_list,
                        n_heads,
                        d_model,
                        dropout,
                        d_ff,
                        norm_first,
                        n_layers,
                        expression_residual,
                        ppi_prior_file,
                        ppi_gate_init,
                        tupe_mode,
                    )
                    for _ in self.task_names
                ]
            )
            self.d_embed = self.task_transformers[0].encoder.d_embed
        self.aggfunc = aggfunc
        self.pooling_mode = pooling_mode
        self.task_gene_index_buffer_names: list[str | None] = []

        if aggfunc == "Flatten":
            self.flatten = nn.Flatten(start_dim=1)
            self.dropout = nn.Dropout(dropout)
            task_input_dim = len(gene_list) * self.d_embed
        else:
            self.pooling = nn.AdaptiveAvgPool1d(1)
            task_input_dim = self.d_embed
        if pooling_mode == "task_attention":
            self.task_attention_pooling = nn.ModuleList(
                [
                    TaskAttentionPooling(self.d_embed, attention_pooling_hidden_dim, attention_pooling_dropout)
                    for _ in self.task_names
                ]
            )
        else:
            self.task_attention_pooling = None
        self.primary_adapter = (
            ResidualAdapter(self.d_embed, primary_adapter_dim, dropout) if primary_adapter_dim > 0 else None
        )

        self.task_specific_layers = nn.ModuleList(
            [
                TaskSpecificLayer(
                    len(gene_list),
                    self.d_embed,
                    dropout,
                    aggfunc,
                    d_hidden1,
                    d_hidden2,
                    slope,
                    d_output,
                    input_dim=task_input_dim,
                    head_norm=head_norm,
                )
                for d_output in d_output_dict.values()
            ]
        )
        if task_gene_indices is None:
            self.task_gene_index_buffer_names = [None for _ in self.task_names]
        else:
            missing = [task_name for task_name in self.task_names if task_name not in task_gene_indices]
            if missing:
                raise ValueError(f"Missing task-specific gene indices for tasks: {missing}")
            for task_name in self.task_names:
                indices = task_gene_indices[task_name]
                if not indices:
                    raise ValueError(f"Task {task_name} has an empty task-specific gene index list.")
                safe_name = re.sub(r"[^0-9A-Za-z_]", "_", task_name)
                buffer_name = f"task_gene_indices_{safe_name}"
                self.register_buffer(buffer_name, torch.tensor(indices, dtype=torch.long), persistent=True)
                self.task_gene_index_buffer_names.append(buffer_name)

    def encoder_modules(self) -> list[Transformer]:
        if self.encoder_sharing == "shared":
            assert self.transformer is not None
            return [self.transformer]
        assert self.task_transformers is not None
        return list(self.task_transformers)

    def embedding_parameters(self):
        for transformer in self.encoder_modules():
            yield from transformer.encoder.embed.parameters()

    def encoder_parameters_without_embeddings(self):
        embedding_ids = {id(parameter) for parameter in self.embedding_parameters()}
        for transformer in self.encoder_modules():
            for parameter in transformer.parameters():
                if id(parameter) not in embedding_ids:
                    yield parameter

    def ppi_gate_values(self) -> dict[str, float]:
        values: dict[str, float] = {}
        names = ["shared"] if self.encoder_sharing == "shared" else list(self.task_names)
        for name, transformer in zip(names, self.encoder_modules()):
            gate = transformer.encoder.ppi_gate
            if gate is not None:
                values[name] = float(torch.tanh(gate.detach()).cpu().item())
        return values

    def _pool_for_task(self, encoded: torch.Tensor, task_idx: int) -> torch.Tensor:
        if task_idx == 0 and self.primary_adapter is not None:
            encoded = self.primary_adapter(encoded)
        buffer_name = self.task_gene_index_buffer_names[task_idx]
        if buffer_name is not None:
            indices = getattr(self, buffer_name)
            encoded = encoded.index_select(1, indices)
        if self.aggfunc == "Flatten":
            return self.dropout(self.flatten(encoded))
        if self.task_attention_pooling is not None:
            return self.task_attention_pooling[task_idx](encoded)
        return self.pooling(encoded.permute(0, 2, 1)).squeeze(-1)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        gene_indices: torch.Tensor | None = None,
        task_sample_mask: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        if self.encoder_sharing == "shared":
            assert self.transformer is not None
            encoded = self.transformer(x, mask, gene_indices)
            pooled_by_task = [self._pool_for_task(encoded, task_idx) for task_idx in range(len(self.task_names))]
        else:
            assert self.task_transformers is not None
            pooled_by_task = [
                self._pool_for_task(transformer(x, mask, gene_indices), task_idx)
                for task_idx, transformer in enumerate(self.task_transformers)
            ]
        if task_sample_mask is None:
            return [layer(task_x) for layer, task_x in zip(self.task_specific_layers, pooled_by_task)]
        if task_sample_mask.ndim != 2 or task_sample_mask.shape != (x.size(0), len(self.task_names)):
            raise ValueError(
                "task_sample_mask must have shape (batch_size, n_tasks); "
                f"received {tuple(task_sample_mask.shape)}."
            )
        outputs: list[torch.Tensor] = []
        for task_idx, (layer, task_x) in enumerate(zip(self.task_specific_layers, pooled_by_task)):
            valid = task_sample_mask[:, task_idx].to(device=task_x.device, dtype=torch.bool)
            output_dim = layer.linear_3.out_features
            if bool(valid.any()):
                valid_logits = layer(task_x[valid])
                logits = task_x.new_zeros((task_x.size(0), output_dim)).index_copy(
                    0, valid.nonzero(as_tuple=False).squeeze(1), valid_logits
                )
            else:
                logits = task_x.new_zeros((task_x.size(0), output_dim))
            outputs.append(logits)
        return outputs
