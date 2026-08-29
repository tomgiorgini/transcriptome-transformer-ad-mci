from __future__ import annotations

import math

import pytest
import torch

from source.models.txt_volumetric import (
    PaperFaithfulMultiHeadVMA,
    PaperFaithfulVMA,
    paper_multimodal_volume,
)


def _projected_inputs() -> tuple[
    torch.Tensor, list[torch.Tensor], list[torch.Tensor]
]:
    generator = torch.Generator().manual_seed(260606249)
    query = torch.randn(1, 2, 3, 4, generator=generator)
    keys = [
        torch.randn(1, 2, 5, 4, generator=generator),
        torch.randn(1, 2, 5, 4, generator=generator),
    ]
    values = [
        torch.randn(1, 2, 5, 4, generator=generator),
        torch.randn(1, 2, 5, 4, generator=generator),
    ]
    return query, keys, values


def test_volume_is_exact_equation_7_for_aligned_modalities() -> None:
    query, keys, _ = _projected_inputs()

    observed = paper_multimodal_volume(
        query, keys, eps=1e-8, query_chunk_size=2
    )

    q = query.unsqueeze(-2).expand(-1, -1, -1, keys[0].size(-2), -1)
    expanded_keys = [
        key.unsqueeze(-3).expand(-1, -1, query.size(-2), -1, -1)
        for key in keys
    ]
    z = torch.stack([q, *expanded_keys], dim=-1)
    gram = z.transpose(-2, -1) @ z
    expected = torch.sqrt(torch.linalg.det(gram).clamp_min(0.0) + 1e-8)

    torch.testing.assert_close(observed, expected, rtol=1e-5, atol=1e-6)


def test_operator_matches_equations_8_to_10_without_tupe() -> None:
    query, keys, values = _projected_inputs()
    beta = 0.75
    operator = PaperFaithfulVMA(
        d_head=4,
        num_modalities=2,
        beta=beta,
        eps=1e-8,
        dropout=0.0,
        tupe_mode="off",
        query_chunk_size=2,
    )
    with torch.no_grad():
        operator.modality_gates[0].weight.copy_(torch.eye(4))
        operator.modality_gates[1].weight.copy_(0.5 * torch.eye(4))

    output, attention, volumes, logits = operator.compute_vma_output(
        query, keys, values
    )

    dot_sum = sum(
        torch.einsum("bhid,bhjd->bhij", query, key) for key in keys
    )
    expected_logits = (-beta * volumes + dot_sum) / math.sqrt(query.size(-1))
    expected_attention = torch.softmax(expected_logits, dim=-1)
    expected_modalities = [
        torch.matmul(expected_attention, value)
        * torch.sigmoid(gate(query))
        for gate, value in zip(operator.modality_gates, values)
    ]
    expected_output = torch.stack(expected_modalities).mean(dim=0)

    torch.testing.assert_close(logits, expected_logits)
    torch.testing.assert_close(attention, expected_attention)
    torch.testing.assert_close(output, expected_output)


def test_tupe_is_only_an_optional_additive_bias_before_softmax() -> None:
    query, keys, values = _projected_inputs()
    without_tupe = PaperFaithfulVMA(4, 2, beta=1.25, tupe_mode="off")
    with_tupe = PaperFaithfulVMA(4, 2, beta=1.25, tupe_mode="on")
    with_tupe.load_state_dict(without_tupe.state_dict())
    tupe = torch.randn(1, 2, 3, 5, generator=torch.Generator().manual_seed(7))

    _, _, _, paper_logits = without_tupe.compute_vma_output(query, keys, values)
    _, observed_attention, _, extended_logits = with_tupe.compute_vma_output(
        query, keys, values, tupe=tupe
    )

    torch.testing.assert_close(extended_logits, paper_logits + tupe)
    torch.testing.assert_close(
        observed_attention, torch.softmax(paper_logits + tupe, dim=-1)
    )
    with pytest.raises(ValueError, match="tupe must be None"):
        without_tupe(query, keys, values, tupe=tupe)
    with pytest.raises(ValueError, match="tupe is required"):
        with_tupe(query, keys, values)


def test_masked_rows_are_zero_and_gradients_are_finite() -> None:
    query, keys, values = _projected_inputs()
    query.requires_grad_(True)
    for tensor in [*keys, *values]:
        tensor.requires_grad_(True)
    operator = PaperFaithfulVMA(4, 2, tupe_mode="off")
    mask = torch.ones(1, 3, 5, dtype=torch.bool)
    mask[:, 1] = False

    output, attention, _, _ = operator(
        query, keys, values, mask=mask, return_attention=True
    )
    output.square().mean().backward()

    assert torch.equal(attention[:, :, 1], torch.zeros_like(attention[:, :, 1]))
    assert torch.equal(output[:, :, 1], torch.zeros_like(output[:, :, 1]))
    for tensor in [query, *keys, *values]:
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()


def test_multihead_wrapper_uses_separate_aligned_modalities() -> None:
    layer = PaperFaithfulMultiHeadVMA(
        n_heads=2,
        d_model=8,
        d_embed=6,
        num_modalities=2,
        beta=1.0,
        tupe_mode="on",
    )
    generator = torch.Generator().manual_seed(11)
    query = torch.randn(2, 3, 6, generator=generator, requires_grad=True)
    modalities = [
        torch.randn(2, 5, 6, generator=generator, requires_grad=True),
        torch.randn(2, 5, 6, generator=generator, requires_grad=True),
    ]
    tupe = torch.randn(2, 2, 3, 5, generator=generator)

    output, attention, volumes, logits = layer(
        query, modalities, tupe=tupe, return_attention=True
    )

    assert output.shape == (2, 3, 6)
    assert attention.shape == volumes.shape == logits.shape == (2, 2, 3, 5)
    output.sum().backward()
    assert query.grad is not None and torch.isfinite(query.grad).all()
    for modality in modalities:
        assert modality.grad is not None and torch.isfinite(modality.grad).all()


def test_paper_dimensional_constraint_is_enforced() -> None:
    with pytest.raises(ValueError, match=r"num_modalities \+ 1"):
        PaperFaithfulVMA(d_head=2, num_modalities=2)
