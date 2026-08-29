from __future__ import annotations

import inspect
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _accepts_expression(attention_augmentation: nn.Module) -> bool:
    """Return whether an attention extension accepts the optional expression keyword.

    Attention augmentations predate expression-aware attention.  Inspecting the
    forward signature lets newer extensions opt in without breaking existing
    modules whose forward method has the historical signature.
    """
    try:
        parameters = inspect.signature(attention_augmentation.forward).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        or (
            parameter.name == "expression"
            and parameter.kind != inspect.Parameter.POSITIONAL_ONLY
        )
        for parameter in parameters
    )


def calculate_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tupe: torch.Tensor,
    mask: torch.Tensor | None = None,
    dropout: nn.Dropout | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    d_head = q.size(-1)
    attention_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(2 * d_head) + tupe

    if mask is not None:
        attention_scores = attention_scores.masked_fill(mask == 0, -1e9)

    attention_scores = F.softmax(attention_scores, dim=-1)
    if dropout is not None:
        attention_scores = dropout(attention_scores)

    output = torch.matmul(attention_scores, v)
    return output, attention_scores


class MultiHeadAttentionLayer(nn.Module):
    def __init__(
        self,
        n_heads: int = 8,
        d_model: int = 512,
        d_embed: int = 512,
        dropout: float = 0.1,
        attention_augmentation: nn.Module | None = None,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError('"d_model" must be divisible by "n_heads".')

        self.n_heads = n_heads
        self.d_model = d_model
        self.d_head = d_model // n_heads

        self.q_linear = nn.Linear(d_embed, d_model, bias=False)
        self.k_linear = nn.Linear(d_embed, d_model, bias=False)
        self.v_linear = nn.Linear(d_embed, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(d_model, d_embed, bias=False)
        # Optional extensions receive the already projected per-head Q/K/V and
        # must return the per-head output to feed to the unchanged projection.
        # Keeping the default as ``None`` preserves the historical TxT module
        # hierarchy, state dict, RNG consumption and numerical path.
        self.attention_augmentation = attention_augmentation

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        tupe: torch.Tensor,
        mask: torch.Tensor | None = None,
        expression: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size = q.size(0)

        q = self.q_linear(q).view(batch_size, -1, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_linear(k).view(batch_size, -1, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_linear(v).view(batch_size, -1, self.n_heads, self.d_head).transpose(1, 2)

        attention_output, _ = calculate_attention(q, k, v, tupe, mask, self.dropout)
        if self.attention_augmentation is not None:
            if _accepts_expression(self.attention_augmentation):
                attention_output = self.attention_augmentation(
                    q,
                    k,
                    v,
                    tupe,
                    attention_output=attention_output,
                    mask=mask,
                    expression=expression,
                )
            else:
                attention_output = self.attention_augmentation(
                    q,
                    k,
                    v,
                    tupe,
                    attention_output=attention_output,
                    mask=mask,
                )
        concat = attention_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.linear(concat)


class PositionWiseFeedForwardLayer(nn.Module):
    def __init__(self, d_embed: int = 512, d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.linear_1 = nn.Linear(d_embed, d_ff, bias=False)
        self.linear_2 = nn.Linear(d_ff, d_embed, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        return self.linear_2(self.dropout(F.relu(self.linear_1(e))))


class EncoderLayer(nn.Module):
    def __init__(
        self,
        n_heads: int = 8,
        d_model: int = 512,
        d_embed: int = 512,
        dropout: float = 0.1,
        d_ff: int = 2048,
        norm_first: bool = False,
    ):
        super().__init__()
        self.multi_head_attention_layer = MultiHeadAttentionLayer(n_heads, d_model, d_embed, dropout)
        self.position_wise_feed_forward_layer = PositionWiseFeedForwardLayer(d_embed, d_ff, dropout)
        self.layer_norm_1 = nn.LayerNorm(d_embed)
        self.layer_norm_2 = nn.LayerNorm(d_embed)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.norm_first = norm_first

    def forward(
        self,
        e: torch.Tensor,
        tupe: torch.Tensor,
        mask: torch.Tensor | None = None,
        expression: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.norm_first:
            e2 = self.layer_norm_1(e)
            e = e + self.dropout_1(
                self.multi_head_attention_layer(e2, e2, e2, tupe, mask, expression=expression)
            )
            e2 = self.layer_norm_2(e)
            e = e + self.dropout_2(self.position_wise_feed_forward_layer(e2))
            return e

        e = self.layer_norm_1(
            e + self.dropout_1(self.multi_head_attention_layer(e, e, e, tupe, mask, expression=expression))
        )
        e = self.layer_norm_2(e + self.dropout_2(self.position_wise_feed_forward_layer(e)))
        return e


class TaskSpecificLayer(nn.Module):
    def __init__(
        self,
        n_genes: int,
        d_embed: int,
        dropout: float = 0.1,
        aggfunc: str = "Flatten",
        d_hidden1: int = 128,
        d_hidden2: int = 64,
        slope: float = 0.2,
        d_output: int = 1,
        input_dim: int | None = None,
        head_norm: str = "batch",
    ):
        super().__init__()
        if head_norm not in {"batch", "layer", "none"}:
            raise ValueError(f"Unsupported head_norm: {head_norm}")
        if input_dim is None:
            input_dim = n_genes * d_embed if aggfunc == "Flatten" else d_embed
        norm_factory = {
            "batch": nn.BatchNorm1d,
            "layer": nn.LayerNorm,
            "none": lambda _: nn.Identity(),
        }[head_norm]
        self.head_norm = head_norm
        self.linear_1 = nn.Linear(input_dim, d_hidden1)
        # Keep the historical attribute names so old BatchNorm checkpoints remain loadable.
        self.batch_norm_1 = norm_factory(d_hidden1)
        self.activation_1 = nn.LeakyReLU(slope, inplace=True)
        self.dropout_1 = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_hidden1, d_hidden2)
        self.batch_norm_2 = norm_factory(d_hidden2)
        self.activation_2 = nn.LeakyReLU(slope, inplace=True)
        self.dropout_2 = nn.Dropout(dropout)
        self.linear_3 = nn.Linear(d_hidden2, d_output)

    @staticmethod
    def _apply_norm(norm: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """Apply head normalization without letting singleton task batches corrupt BatchNorm.

        Mask-aware heads occasionally receive one valid sample even when the full
        minibatch is larger. BatchNorm cannot estimate a variance from one value;
        in that rare case we use its existing running statistics while retaining
        gradients through the affine parameters and surrounding linear layers.
        """
        if isinstance(norm, nn.BatchNorm1d) and norm.training and x.size(0) < 2:
            return F.batch_norm(
                x,
                norm.running_mean,
                norm.running_var,
                norm.weight,
                norm.bias,
                training=False,
                momentum=0.0,
                eps=norm.eps,
            )
        return norm(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear_1(x)
        x = self.dropout_1(self.activation_1(self._apply_norm(self.batch_norm_1, x)))
        x = self.linear_2(x)
        x = self.dropout_2(self.activation_2(self._apply_norm(self.batch_norm_2, x)))
        return self.linear_3(x)
