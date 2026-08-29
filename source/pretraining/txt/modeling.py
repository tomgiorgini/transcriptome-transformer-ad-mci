from __future__ import annotations

import torch
import torch.nn as nn

from source.models.txt.model import Transformer


class TxTMaskedRestorer(nn.Module):
    def __init__(
        self,
        embed_file: str,
        gene_list: list[str],
        n_heads: int = 4,
        d_model: int = 128,
        dropout: float = 0.2,
        d_ff: int = 512,
        norm_first: bool = False,
        n_layers: int = 3,
    ):
        super().__init__()
        self.transformer = Transformer(
            embed_file=embed_file,
            gene_list=gene_list,
            n_heads=n_heads,
            d_model=d_model,
            dropout=dropout,
            d_ff=d_ff,
            norm_first=norm_first,
            n_layers=n_layers,
        )
        self.d_embed = self.transformer.encoder.d_embed
        self.mask_value = nn.Parameter(torch.zeros(1))
        self.restoration_head = nn.Linear(self.d_embed, 1)

    def forward(
        self,
        x: torch.Tensor,
        gene_indices: torch.Tensor,
        mask: torch.Tensor | None = None,
        masked_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if masked_positions is None and mask is not None:
            masked_positions = ~mask.squeeze(1).squeeze(1).bool()
        if masked_positions is not None:
            x = x.clone()
            x[masked_positions] = self.mask_value.to(dtype=x.dtype)
        encoded = self.transformer(x, mask=mask, gene_indices=gene_indices)
        return self.restoration_head(encoded).squeeze(-1)
