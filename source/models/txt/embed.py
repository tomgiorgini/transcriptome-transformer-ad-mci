from __future__ import annotations

import math

import pandas as pd
import torch
import torch.nn as nn


class Embedder(nn.Module):
    def __init__(self, embed_file: str, gene_list: list[str]):
        super().__init__()
        embed_df = pd.read_csv(embed_file, index_col=0)
        embed_df = embed_df.loc[gene_list]
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embed_df.values, dtype=torch.float32), freeze=False)

    def forward(self, gene_indices: torch.Tensor) -> torch.Tensor:
        return self.embed(gene_indices)


class TUPE_A(nn.Module):
    def __init__(self, n_heads: int, d_model: int):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        if self.d_head * n_heads != d_model:
            raise ValueError('"d_model" must be divisible by "n_heads".')

        self.q_linear = nn.Linear(1, d_model, bias=False)
        self.k_linear = nn.Linear(1, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        x = x.unsqueeze(-1)
        q = self.q_linear(x).view(batch_size, -1, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_linear(x).view(batch_size, -1, self.n_heads, self.d_head).transpose(1, 2)
        return torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(2 * self.d_head)
