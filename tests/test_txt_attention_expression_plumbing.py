from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn

from source.models.txt.layers import MultiHeadAttentionLayer
from source.models.txt.model import Encoder


class _ExpressionAwareAugmentation(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.expression: torch.Tensor | None = None

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        tupe: torch.Tensor,
        *,
        attention_output: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        expression: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del q, k, v, tupe, mask
        self.expression = expression
        assert attention_output is not None
        return attention_output


class _LegacyAugmentation(nn.Module):
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        tupe: torch.Tensor,
        *,
        attention_output: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del q, k, v, tupe, mask
        assert attention_output is not None
        return attention_output


class AttentionExpressionPlumbingTests(unittest.TestCase):
    @staticmethod
    def _embedding_file(directory: str) -> Path:
        embedding_file = Path(directory) / "embedding.csv"
        pd.DataFrame(
            torch.arange(16, dtype=torch.float32).view(4, 4).numpy(),
            index=pd.Index([f"G{index}" for index in range(4)], name="Gene"),
        ).to_csv(embedding_file)
        return embedding_file

    @staticmethod
    def _attention_inputs() -> tuple[torch.Tensor, ...]:
        torch.manual_seed(31)
        q = torch.randn(2, 3, 4)
        k = torch.randn(2, 3, 4)
        v = torch.randn(2, 3, 4)
        tupe = torch.randn(2, 2, 3, 3)
        mask = torch.ones(2, 2, 3, 3, dtype=torch.bool)
        expression = torch.randn(2, 3)
        return q, k, v, tupe, mask, expression

    def test_expression_is_exact_noop_without_augmentation(self) -> None:
        torch.manual_seed(17)
        layer = MultiHeadAttentionLayer(n_heads=2, d_model=4, d_embed=4, dropout=0.3)
        q, k, v, tupe, mask, expression = self._attention_inputs()

        torch.manual_seed(101)
        expected = layer(q, k, v, tupe, mask)
        torch.manual_seed(101)
        observed = layer(q, k, v, tupe, mask, expression=expression)

        self.assertTrue(torch.equal(expected, observed))

    def test_legacy_augmentation_remains_callable(self) -> None:
        torch.manual_seed(19)
        layer = MultiHeadAttentionLayer(
            n_heads=2,
            d_model=4,
            d_embed=4,
            dropout=0.0,
            attention_augmentation=_LegacyAugmentation(),
        ).eval()
        q, k, v, tupe, mask, expression = self._attention_inputs()

        observed = layer(q, k, v, tupe, mask, expression=expression)
        layer.attention_augmentation = None
        expected = layer(q, k, v, tupe, mask)

        self.assertTrue(torch.equal(expected, observed))

    def test_encoder_passes_original_expression_to_every_augmentation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            encoder = Encoder(
                embed_file=str(self._embedding_file(tmpdir)),
                gene_list=[f"G{index}" for index in range(4)],
                n_heads=2,
                d_model=4,
                dropout=0.0,
                d_ff=8,
                n_layers=2,
            ).eval()

            augmentations = []
            for layer in encoder.layers:
                augmentation = _ExpressionAwareAugmentation()
                layer.multi_head_attention_layer.attention_augmentation = augmentation
                augmentations.append(augmentation)
            expression = torch.randn(2, 4)
            encoder(expression)

        for augmentation in augmentations:
            self.assertIs(augmentation.expression, expression)

    def test_encoder_baseline_matches_legacy_layer_calls_exactly(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            encoder = Encoder(
                embed_file=str(self._embedding_file(tmpdir)),
                gene_list=[f"G{index}" for index in range(4)],
                n_heads=2,
                d_model=4,
                dropout=0.0,
                d_ff=8,
                n_layers=2,
            ).eval()
            expression = torch.randn(2, 4)
            gene_indices = torch.arange(4).repeat(2, 1)

            embeddings = encoder.embed(gene_indices)
            tupe = encoder.tupe(expression)
            for layer in encoder.layers:
                embeddings = layer(embeddings, tupe)
            expected = encoder.layer_norm(embeddings)
            observed = encoder(expression, gene_indices=gene_indices)

        self.assertTrue(torch.equal(expected, observed))


if __name__ == "__main__":
    unittest.main()
