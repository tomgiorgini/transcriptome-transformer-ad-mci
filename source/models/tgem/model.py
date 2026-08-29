from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiAttentionLayer(nn.Module):
    def __init__(self, n_genes: int, n_heads: int):
        super().__init__()
        self.n_genes = n_genes
        self.n_heads = n_heads

        self.wq = nn.Parameter(torch.empty(n_heads, n_genes, 1))
        self.wk = nn.Parameter(torch.empty(n_heads, n_genes, 1))
        self.wv = nn.Parameter(torch.empty(n_heads, n_genes, 1))
        self.output_weights = nn.Parameter(torch.full((n_heads,), 0.001))

        nn.init.xavier_normal_(self.wq, gain=1.0)
        nn.init.xavier_normal_(self.wk, gain=1.0)
        nn.init.xavier_normal_(self.wv, gain=1.0)

    def _mask_self_attention(self, attention_scores: torch.Tensor) -> torch.Tensor:
        eye = torch.eye(attention_scores.shape[1], device=attention_scores.device, dtype=attention_scores.dtype)
        return attention_scores * (1 - eye)

    def _attention(self, x: torch.Tensor, q_seq: torch.Tensor, wk: torch.Tensor, wv: torch.Tensor) -> torch.Tensor:
        k_seq = (x * wk).expand(x.shape[0], x.shape[1], self.n_genes).permute(0, 2, 1)
        v_seq = x * wv
        attention_scores = torch.softmax(q_seq * k_seq, dim=2)
        attention_scores = self._mask_self_attention(attention_scores)
        return torch.matmul(attention_scores, v_seq)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(x.shape[0], x.shape[1], 1)
        head_outputs = []
        for head_idx in range(self.n_heads):
            q_seq = (x * self.wq[head_idx]).expand(x.shape[0], x.shape[1], self.n_genes)
            head_outputs.append(self._attention(x, q_seq, self.wk[head_idx], self.wv[head_idx]))
        stacked = torch.cat(head_outputs, dim=2)
        return torch.matmul(stacked, self.output_weights)


class LayerNorm1D(nn.Module):
    def __init__(self, features: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.weight * (x - mean) / (std + self.eps) + self.bias


class ResidualLayer(nn.Module):
    def __init__(self, size: int, dropout: float):
        super().__init__()
        self.norm = LayerNorm1D(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        return x + self.norm(self.dropout(out))


class TGemClassifier(nn.Module):
    def __init__(
        self,
        n_genes: int,
        n_classes: int,
        n_heads: int = 4,
        dropout: float = 0.2,
        activation: str = "leakyrelu",
        n_layers: int = 3,
    ):
        super().__init__()
        if activation not in {"relu", "leakyrelu", "gelu"}:
            raise ValueError("activation must be one of: relu, leakyrelu, gelu")

        self.activation_name = activation
        self.attention_layers = nn.ModuleList([MultiAttentionLayer(n_genes, n_heads) for _ in range(n_layers)])
        self.residual_layers = nn.ModuleList([ResidualLayer(n_genes, dropout) for _ in range(n_layers)])

        self.classifier = nn.Linear(n_genes, n_classes)
        nn.init.xavier_uniform_(self.classifier.weight, gain=1.0)

    def _apply_activation(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation_name == "relu":
            return F.relu(x)
        if self.activation_name == "gelu":
            return F.gelu(x)
        return F.leaky_relu(x, negative_slope=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = x
        for attention_layer, residual_layer in zip(self.attention_layers, self.residual_layers):
            hidden = residual_layer(hidden, attention_layer(hidden))
        hidden = self._apply_activation(hidden)
        return self.classifier(hidden)
