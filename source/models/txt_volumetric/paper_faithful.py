"""Paper-faithful Volumetric Multi-head Attention.

This module implements Equations 7--10 of the GRAMformer paper without the
PPI-specific substitutions used by :mod:`source.models.txt_volumetric.layers`.
Keys and values are supplied as separate, position-aligned modality streams.

TUPE is an explicit, optional extension.  With ``tupe_mode="off"`` (the
default), no TUPE tensor is accepted and the score is exactly Equation 9.  With
``tupe_mode="on"``, the supplied TUPE tensor is added to that score immediately
before the softmax.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn as nn


PAPER_FAITHFUL_TUPE_MODES = frozenset({"off", "on"})


def _validate_projected_modalities(
    query: torch.Tensor,
    keys: Sequence[torch.Tensor],
    values: Sequence[torch.Tensor] | None = None,
) -> tuple[int, int, int, int, int]:
    if query.ndim != 4:
        raise ValueError("query must have shape [batch, heads, queries, d_head].")
    if not keys:
        raise ValueError("At least one modality is required.")

    batch_size, n_heads, n_queries, d_head = query.shape
    if min(batch_size, n_heads, n_queries, d_head) <= 0:
        raise ValueError("query dimensions must all be positive.")
    if len(keys) + 1 > d_head:
        raise ValueError(
            "The paper requires num_modalities + 1 <= d_head so that the "
            "Gram volume is not structurally degenerate."
        )

    n_keys: int | None = None
    for index, key in enumerate(keys):
        if key.ndim != 4:
            raise ValueError(
                f"keys[{index}] must have shape [batch, heads, keys, d_head]."
            )
        if key.shape[:2] != (batch_size, n_heads) or key.size(-1) != d_head:
            raise ValueError(f"keys[{index}] is incompatible with query.")
        if n_keys is None:
            n_keys = key.size(-2)
            if n_keys <= 0:
                raise ValueError("The key sequence must be non-empty.")
        elif key.size(-2) != n_keys:
            raise ValueError("All modality streams must be aligned to the same key length.")

    if values is not None:
        if len(values) != len(keys):
            raise ValueError("keys and values must contain the same number of modalities.")
        expected = (batch_size, n_heads, int(n_keys), d_head)
        for index, value in enumerate(values):
            if tuple(value.shape) != expected:
                raise ValueError(
                    f"values[{index}] must have shape [batch, heads, keys, d_head]."
                )

    return batch_size, n_heads, n_queries, int(n_keys), d_head


def paper_multimodal_volume(
    query: torch.Tensor,
    keys: Sequence[torch.Tensor],
    *,
    eps: float = 1e-8,
    query_chunk_size: int | None = 64,
) -> torch.Tensor:
    """Compute the Equation 7 Gram volume for every query/key position pair.

    ``query`` has shape ``[B, H, Nq, Dh]``.  Every item in ``keys`` has shape
    ``[B, H, Nk, Dh]`` and represents one modality at the same ``Nk`` aligned
    positions.  The result has shape ``[B, H, Nq, Nk]``.

    Query chunking only bounds temporary Gram-matrix memory; it does not alter
    the equation.  Determinants are evaluated in FP32, matching the numerical
    safeguard already used by the repository's volumetric primitive.
    """

    if eps < 0 or not math.isfinite(eps):
        raise ValueError("eps must be finite and non-negative.")
    if query_chunk_size is not None and query_chunk_size <= 0:
        raise ValueError("query_chunk_size must be positive or None.")

    batch_size, n_heads, n_queries, n_keys, _ = _validate_projected_modalities(
        query, keys
    )
    chunk_size = n_queries if query_chunk_size is None else query_chunk_size

    with torch.autocast(device_type=query.device.type, enabled=False):
        q = query.float()
        modality_keys = [key.float() for key in keys]
        # Key-key inner products do not depend on the query index.
        key_grams = [
            [(left * right).sum(dim=-1) for right in modality_keys]
            for left in modality_keys
        ]
        chunks: list[torch.Tensor] = []
        for start in range(0, n_queries, chunk_size):
            q_chunk = q[:, :, start : start + chunk_size]
            current_queries = q_chunk.size(-2)
            q_norm = (q_chunk * q_chunk).sum(dim=-1, keepdim=True).expand(
                batch_size, n_heads, current_queries, n_keys
            )
            query_key = [
                torch.einsum("bhid,bhjd->bhij", q_chunk, key)
                for key in modality_keys
            ]

            first_row = torch.stack([q_norm, *query_key], dim=-1)
            rows = [first_row]
            for modality_index in range(len(modality_keys)):
                expanded_key_grams = [
                    entry.unsqueeze(-2).expand(
                        batch_size, n_heads, current_queries, n_keys
                    )
                    for entry in key_grams[modality_index]
                ]
                rows.append(
                    torch.stack(
                        [query_key[modality_index], *expanded_key_grams], dim=-1
                    )
                )
            gram = torch.stack(rows, dim=-2)
            determinant = torch.linalg.det(gram)
            chunks.append(torch.sqrt(torch.clamp_min(determinant, 0.0) + eps))

    return torch.cat(chunks, dim=-2)


class PaperFaithfulVMA(nn.Module):
    """Equations 7--10 for already projected multi-head Q/K/V tensors.

    This is a replacement attention operator, not a residual branch beside
    dense dot-product attention.  It owns one paper gate ``W_gate,m`` for each
    modality and averages the gated value aggregations by ``1 / M``.
    """

    def __init__(
        self,
        d_head: int,
        num_modalities: int,
        *,
        beta: float = 1.0,
        eps: float = 1e-8,
        dropout: float = 0.0,
        tupe_mode: str = "off",
        query_chunk_size: int | None = 64,
    ) -> None:
        super().__init__()
        if d_head <= 0 or num_modalities <= 0:
            raise ValueError("d_head and num_modalities must be positive.")
        if num_modalities + 1 > d_head:
            raise ValueError("num_modalities + 1 must be <= d_head.")
        if not math.isfinite(beta):
            raise ValueError("beta must be finite.")
        if eps < 0 or not math.isfinite(eps):
            raise ValueError("eps must be finite and non-negative.")
        if not 0 <= dropout <= 1:
            raise ValueError("dropout must be in [0, 1].")
        if tupe_mode not in PAPER_FAITHFUL_TUPE_MODES:
            choices = ", ".join(sorted(PAPER_FAITHFUL_TUPE_MODES))
            raise ValueError(f"tupe_mode must be one of {{{choices}}}.")
        if query_chunk_size is not None and query_chunk_size <= 0:
            raise ValueError("query_chunk_size must be positive or None.")

        self.d_head = int(d_head)
        self.num_modalities = int(num_modalities)
        self.tupe_mode = tupe_mode
        self.query_chunk_size = query_chunk_size
        self.register_buffer("beta", torch.tensor(float(beta), dtype=torch.float32))
        self.register_buffer("eps", torch.tensor(float(eps), dtype=torch.float32))
        self.modality_gates = nn.ModuleList(
            nn.Linear(d_head, d_head, bias=False) for _ in range(num_modalities)
        )
        self.attention_dropout = nn.Dropout(dropout)

    @staticmethod
    def _broadcast_mask(
        mask: torch.Tensor, shape: torch.Size, device: torch.device
    ) -> torch.Tensor:
        valid = torch.as_tensor(mask, dtype=torch.bool, device=device)
        if valid.ndim == 3:
            valid = valid.unsqueeze(1)
        try:
            return torch.broadcast_to(valid, shape)
        except RuntimeError as exc:
            raise ValueError("mask is not broadcastable to [batch, heads, queries, keys].") from exc

    @staticmethod
    def _validate_tupe(
        tupe: torch.Tensor, shape: torch.Size, device: torch.device
    ) -> torch.Tensor:
        tupe = torch.as_tensor(tupe, dtype=torch.float32, device=device)
        if tupe.ndim == 3:
            tupe = tupe.unsqueeze(1)
        try:
            return torch.broadcast_to(tupe, shape)
        except RuntimeError as exc:
            raise ValueError("tupe is not broadcastable to [batch, heads, queries, keys].") from exc

    def compute_vma_output(
        self,
        query: torch.Tensor,
        keys: Sequence[torch.Tensor],
        values: Sequence[torch.Tensor],
        *,
        tupe: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(output, attention, volumes, logits)`` for auditability."""

        _, _, _, _, d_head = _validate_projected_modalities(query, keys, values)
        if d_head != self.d_head:
            raise ValueError(f"Expected d_head={self.d_head}, received {d_head}.")
        if len(keys) != self.num_modalities:
            raise ValueError(
                f"Expected {self.num_modalities} modalities, received {len(keys)}."
            )
        if self.tupe_mode == "off" and tupe is not None:
            raise ValueError('tupe must be None when tupe_mode="off".')
        if self.tupe_mode == "on" and tupe is None:
            raise ValueError('tupe is required when tupe_mode="on".')

        volumes = paper_multimodal_volume(
            query,
            keys,
            eps=float(self.eps.item()),
            query_chunk_size=self.query_chunk_size,
        )
        with torch.autocast(device_type=query.device.type, enabled=False):
            q = query.float()
            projected_keys = [key.float() for key in keys]
            dot_product_sum = sum(
                torch.einsum("bhid,bhjd->bhij", q, key)
                for key in projected_keys
            )
            logits = (
                -self.beta.float() * volumes + dot_product_sum
            ) / math.sqrt(self.d_head)

            if tupe is not None:
                logits = logits + self._validate_tupe(
                    tupe, logits.shape, logits.device
                )

            valid: torch.Tensor | None = None
            if mask is not None:
                valid = self._broadcast_mask(mask, logits.shape, logits.device)
                logits_for_softmax = torch.where(
                    valid, logits, torch.full_like(logits, float("-inf"))
                )
            else:
                logits_for_softmax = logits
            attention = torch.softmax(logits_for_softmax, dim=-1)
            # Fully masked rows are defined as zero instead of NaN.
            attention = torch.nan_to_num(attention, nan=0.0)
            if valid is not None:
                attention = torch.where(valid, attention, torch.zeros_like(attention))
            dropped_attention = self.attention_dropout(attention)

            modality_outputs = []
            for gate, value in zip(self.modality_gates, values):
                attended_value = torch.matmul(dropped_attention, value.float())
                modality_gate = torch.sigmoid(gate(q))
                modality_outputs.append(attended_value * modality_gate)
            output = torch.stack(modality_outputs, dim=0).mean(dim=0)

        return output.to(query.dtype), attention, volumes, logits

    def forward(
        self,
        query: torch.Tensor,
        keys: Sequence[torch.Tensor],
        values: Sequence[torch.Tensor],
        *,
        tupe: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        result = self.compute_vma_output(
            query, keys, values, tupe=tupe, mask=mask
        )
        return result if return_attention else result[0]


class PaperFaithfulMultiHeadVMA(nn.Module):
    """Projection wrapper around :class:`PaperFaithfulVMA`.

    ``query`` and every modality tensor use shape ``[B, N, d_embed]``.  Each
    modality has independent key and value projections, as required by the
    multimodal operator.  This layer returns VMA directly and deliberately has
    no parallel conventional-attention branch or outer ``gamma`` gate.
    """

    def __init__(
        self,
        n_heads: int,
        d_model: int,
        d_embed: int,
        num_modalities: int,
        *,
        beta: float = 1.0,
        eps: float = 1e-8,
        dropout: float = 0.0,
        tupe_mode: str = "off",
        query_chunk_size: int | None = 64,
    ) -> None:
        super().__init__()
        if n_heads <= 0 or d_model <= 0 or d_embed <= 0:
            raise ValueError("n_heads, d_model and d_embed must be positive.")
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")
        self.n_heads = int(n_heads)
        self.d_model = int(d_model)
        self.d_embed = int(d_embed)
        self.num_modalities = int(num_modalities)
        self.d_head = d_model // n_heads

        self.query_projection = nn.Linear(d_embed, d_model)
        self.key_projections = nn.ModuleList(
            nn.Linear(d_embed, d_model) for _ in range(num_modalities)
        )
        self.value_projections = nn.ModuleList(
            nn.Linear(d_embed, d_model) for _ in range(num_modalities)
        )
        self.vma = PaperFaithfulVMA(
            self.d_head,
            num_modalities,
            beta=beta,
            eps=eps,
            dropout=dropout,
            tupe_mode=tupe_mode,
            query_chunk_size=query_chunk_size,
        )
        self.output_projection = nn.Linear(d_model, d_embed)

    def _split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, _ = tensor.shape
        return tensor.view(
            batch_size, sequence_length, self.n_heads, self.d_head
        ).permute(0, 2, 1, 3)

    def forward(
        self,
        query: torch.Tensor,
        modalities: Sequence[torch.Tensor],
        *,
        tupe: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if query.ndim != 3 or query.size(-1) != self.d_embed:
            raise ValueError("query must have shape [batch, queries, d_embed].")
        if len(modalities) != self.num_modalities:
            raise ValueError(
                f"Expected {self.num_modalities} modalities, received {len(modalities)}."
            )
        for index, modality in enumerate(modalities):
            if modality.ndim != 3 or modality.size(0) != query.size(0):
                raise ValueError(f"modalities[{index}] has an invalid batch or rank.")
            if modality.size(-1) != self.d_embed:
                raise ValueError(f"modalities[{index}] must end in d_embed.")

        projected_query = self._split_heads(self.query_projection(query))
        projected_keys = [
            self._split_heads(projection(modality))
            for projection, modality in zip(self.key_projections, modalities)
        ]
        projected_values = [
            self._split_heads(projection(modality))
            for projection, modality in zip(self.value_projections, modalities)
        ]
        vma_result = self.vma.compute_vma_output(
            projected_query,
            projected_keys,
            projected_values,
            tupe=tupe,
            mask=mask,
        )
        per_head_output, attention, volumes, logits = vma_result
        concatenated = per_head_output.permute(0, 2, 1, 3).contiguous().view(
            query.size(0), query.size(1), self.d_model
        )
        output = self.output_projection(concatenated)
        if return_attention:
            return output, attention, volumes, logits
        return output
