from __future__ import annotations

import math

import pandas as pd
import pytest
import torch

from source.models.txt_volumetric.layers import VolumetricAttentionAugmentation
from source.models.txt_volumetric.model import TxTVolumetric


def _edge_index() -> torch.Tensor:
    return torch.tensor(
        [[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]],
        dtype=torch.long,
    )


def _inputs(batch_size: int = 2) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(41)
    q = torch.randn(batch_size, 2, 3, 4, generator=generator)
    k = torch.randn(batch_size, 2, 3, 4, generator=generator)
    v = torch.randn(batch_size, 2, 3, 4, generator=generator)
    tupe = torch.randn(batch_size, 2, 3, 3, generator=generator) * 0.1
    return q, k, v, tupe


def _v2_operator(**overrides: object) -> VolumetricAttentionAugmentation:
    kwargs: dict[str, object] = {
        "n_heads": 2,
        "d_head": 4,
        "edge_index": _edge_index(),
        "n_nodes": 3,
        "beta": 1.0,
        "dropout": 0.0,
        "gate_init": 0.4,
        "volumetric_volume_mode": "l2",
        "volumetric_message_mode": "expression_contrast",
        "volumetric_output_norm": "none",
        "volumetric_gate_mode": "scalar",
        "volumetric_backbone_gradient_mode": "coupled",
    }
    kwargs.update(overrides)
    return VolumetricAttentionAugmentation(**kwargs)


def test_legacy_defaults_are_exactly_the_explicit_legacy_path() -> None:
    torch.manual_seed(17)
    implicit = VolumetricAttentionAugmentation(
        2,
        4,
        _edge_index(),
        n_nodes=3,
        dropout=0.0,
    )
    next_after_implicit = torch.rand(5)

    torch.manual_seed(17)
    explicit = VolumetricAttentionAugmentation(
        2,
        4,
        _edge_index(),
        n_nodes=3,
        dropout=0.0,
        volumetric_message_mode="legacy",
        volumetric_output_norm="none",
        volumetric_gate_mode="scalar",
        volumetric_backbone_gradient_mode="coupled",
    )
    next_after_explicit = torch.rand(5)

    assert list(implicit.state_dict()) == list(explicit.state_dict())
    torch.testing.assert_close(next_after_implicit, next_after_explicit, rtol=0, atol=0)
    for left, right in zip(
        implicit.compute_vma_output(*_inputs()),
        explicit.compute_vma_output(*_inputs()),
    ):
        torch.testing.assert_close(left, right, rtol=0, atol=0)


def test_expression_standardization_is_affine_invariant_over_valid_genes() -> None:
    operator = _v2_operator()
    q, k, v, tupe = _inputs()
    expression = torch.tensor([[1.0, 4.0, float("nan")], [2.0, 3.0, 8.0]])
    transformed = expression * torch.tensor([[2.0], [5.0]]) + torch.tensor([[7.0], [-3.0]])

    original = operator.compute_vma_output(q, k, v, tupe, expression=expression)
    affine = operator.compute_vma_output(q, k, v, tupe, expression=transformed)
    for left, right in zip(original, affine):
        torch.testing.assert_close(left, right, rtol=1e-5, atol=1e-6)


def test_expression_contrast_changes_vma_between_non_affine_patient_profiles() -> None:
    operator = _v2_operator()
    q, k, v, tupe = _inputs(batch_size=1)
    q = q.expand(2, -1, -1, -1).clone()
    k = k.expand(2, -1, -1, -1).clone()
    v = v.expand(2, -1, -1, -1).clone()
    tupe = torch.zeros(2, 2, 3, 3)
    expression = torch.tensor([[0.0, 1.0, 4.0], [0.0, 4.0, 1.0]])

    output, weights, _, _ = operator.compute_vma_output(
        q,
        k,
        v,
        tupe,
        expression=expression,
    )
    assert not torch.allclose(output[0], output[1])
    assert not torch.allclose(weights[0], weights[1])


def test_expression_contrast_is_neighbor_minus_self_without_legacy_offset() -> None:
    operator = VolumetricAttentionAugmentation(
        1,
        2,
        torch.tensor([[0], [1]]),
        n_nodes=2,
        beta=0.0,
        dropout=0.0,
        volumetric_message_mode="expression_contrast",
    )
    with torch.no_grad():
        operator.expression_q_projection.weight.zero_()
        operator.expression_k_projection.weight.zero_()
        operator.expression_v_projection.weight.zero_()
        operator.self_gate.weight.zero_()
        operator.neighbor_gate.weight.zero_()

    q = torch.zeros(1, 1, 2, 2)
    k = torch.zeros_like(q)
    v = torch.tensor([[[[1.0, 2.0], [5.0, 8.0]]]])
    tupe = torch.zeros(1, 1, 2, 2)
    output, weights, _, _ = operator.compute_vma_output(
        q,
        k,
        v,
        tupe,
        expression=torch.tensor([[-1.0, 1.0]]),
    )

    torch.testing.assert_close(weights, torch.ones_like(weights))
    torch.testing.assert_close(output[0, 0, 0], torch.tensor([2.0, 3.0]))
    torch.testing.assert_close(output[0, 0, 1], torch.zeros(2))


def test_rms_mode_matches_vma_scale_to_dense_attention() -> None:
    operator = _v2_operator(volumetric_output_norm="rms")
    q, k, v, tupe = _inputs()
    dense = torch.randn_like(q)
    expression = torch.tensor([[0.0, 1.0, 4.0], [4.0, 1.0, 0.0]])

    operator(
        q,
        k,
        v,
        tupe,
        attention_output=dense,
        expression=expression,
    )
    diagnostics = operator.diagnostics()
    assert diagnostics["output_norm"] == "rms"
    assert diagnostics["output_norm_scale_factor"] > 0
    assert diagnostics["vma_output_norm"] == pytest.approx(
        diagnostics["baseline_output_norm"],
        rel=1e-5,
        abs=1e-6,
    )


def test_rms_matching_is_invariant_to_batch_composition_per_patient() -> None:
    operator = _v2_operator(volumetric_output_norm="rms").eval()
    q, k, v, tupe = _inputs()
    dense = torch.randn_like(q)
    expression = torch.tensor([[0.0, 1.0, 4.0], [8.0, -3.0, 2.0]])

    batched = operator(
        q,
        k,
        v,
        tupe,
        attention_output=dense,
        expression=expression,
    )
    alone = operator(
        q[:1],
        k[:1],
        v[:1],
        tupe[:1],
        attention_output=dense[:1],
        expression=expression[:1],
    )
    torch.testing.assert_close(batched[:1], alone, rtol=1e-6, atol=1e-7)


def test_explicit_expression_inherits_context_valid_mask() -> None:
    operator = _v2_operator().eval()
    q, k, v, tupe = _inputs(batch_size=1)
    tupe.zero_()
    valid_mask = torch.tensor([[True, True, False]])
    first = torch.tensor([[0.0, 2.0, 10.0]])
    changed_only_where_invalid = torch.tensor([[0.0, 2.0, -1000.0]])
    operator.set_expression_context(first, valid_mask)
    try:
        inherited = operator.compute_vma_output(
            q,
            k,
            v,
            tupe,
            expression=first,
        )[0]
        inherited_after_invalid_change = operator.compute_vma_output(
            q,
            k,
            v,
            tupe,
            expression=changed_only_where_invalid,
        )[0]
        explicit = operator.compute_vma_output(
            q,
            k,
            v,
            tupe,
            expression=changed_only_where_invalid,
            expression_valid_mask=valid_mask,
        )[0]
    finally:
        operator.clear_expression_context()

    torch.testing.assert_close(inherited, inherited_after_invalid_change, rtol=0, atol=0)
    torch.testing.assert_close(inherited, explicit, rtol=0, atol=0)
    unmasked = operator.compute_vma_output(
        q,
        k,
        v,
        tupe,
        expression=changed_only_where_invalid,
    )[0]
    assert not torch.allclose(inherited, unmasked)


def test_per_head_gate_broadcast_and_diagnostics() -> None:
    operator = _v2_operator(volumetric_gate_mode="per_head", gate_init=0.0)
    with torch.no_grad():
        operator.gamma.copy_(torch.tensor([0.0, math.atanh(0.5)]))
    q, k, v, tupe = _inputs()
    expression = torch.tensor([[0.0, 1.0, 4.0], [4.0, 1.0, 0.0]])
    vma, _, _, _ = operator.compute_vma_output(
        q,
        k,
        v,
        tupe,
        expression=expression,
    )
    fused = operator(
        q,
        k,
        v,
        tupe,
        attention_output=torch.zeros_like(vma),
        expression=expression,
    )

    torch.testing.assert_close(fused[:, 0], torch.zeros_like(fused[:, 0]), atol=1e-7, rtol=0)
    torch.testing.assert_close(fused[:, 1], 0.5 * vma[:, 1], atol=1e-6, rtol=1e-6)
    diagnostics = operator.diagnostics()
    assert diagnostics["gate_mode"] == "per_head"
    assert diagnostics["gamma_effective_head_0"] == pytest.approx(0.0)
    assert diagnostics["gamma_effective_head_1"] == pytest.approx(0.5)


@pytest.mark.parametrize("gradient_mode,expect_backbone_grad", [("coupled", True), ("detached", False)])
def test_backbone_gradient_mode_controls_qkv_and_tupe_gradients(
    gradient_mode: str,
    expect_backbone_grad: bool,
) -> None:
    operator = _v2_operator(volumetric_backbone_gradient_mode=gradient_mode)
    q, k, v, tupe = (tensor.detach().requires_grad_(True) for tensor in _inputs())
    expression = torch.tensor([[0.0, 1.0, 4.0], [4.0, 1.0, 0.0]])
    output = operator(
        q,
        k,
        v,
        tupe,
        attention_output=torch.zeros_like(q),
        expression=expression,
    )
    output.square().sum().backward()

    for tensor in (q, k, v, tupe):
        if expect_backbone_grad:
            assert tensor.grad is not None
            assert torch.count_nonzero(tensor.grad).item() > 0
        else:
            assert tensor.grad is None
    assert operator.expression_q_projection.weight.grad is not None
    assert operator.expression_v_projection.weight.grad is not None


def test_txt_volumetric_v2_supplies_and_clears_expression_context(tmp_path) -> None:
    genes = ["A", "B", "C"]
    embedding_file = tmp_path / "embedding.csv"
    pd.DataFrame(
        torch.arange(12, dtype=torch.float32).view(3, 4).numpy() / 20.0,
        index=genes,
        columns=["e0", "e1", "e2", "e3"],
    ).to_csv(embedding_file)
    kwargs = {
        "embed_file": str(embedding_file),
        "gene_list": genes,
        "n_heads": 2,
        "d_model": 4,
        "dropout": 0.0,
        "d_ff": 8,
        "n_layers": 1,
        "aggfunc": "Avgpool",
        "d_hidden1": 4,
        "d_hidden2": 2,
        "d_output_dict": {"task": 2},
        "head_norm": "none",
        "edge_index": _edge_index(),
        "volumetric_volume_mode": "l2",
        "volumetric_message_mode": "expression_contrast",
        "volumetric_output_norm": "rms",
        "volumetric_gate_mode": "per_head",
        "volumetric_backbone_gradient_mode": "detached",
        "volumetric_gate_init": 0.1,
        "volumetric_dropout": 0.0,
    }
    torch.manual_seed(9)
    model = TxTVolumetric(**kwargs).eval()
    x = torch.tensor([[0.0, float("nan"), 4.0], [4.0, 1.0, 0.0]])
    with torch.no_grad():
        expected = model(x)
    assert all(torch.isfinite(output).all() for output in expected)

    augmentation = next(model.iter_volumetric_augmentations())[1]
    assert augmentation._expression_context is None
    diagnostics = model.volumetric_diagnostics()["shared.layer_0"]
    assert diagnostics["message_mode"] == "expression_contrast"
    assert diagnostics["output_norm"] == "rms"
    assert diagnostics["backbone_gradient_mode"] == "detached"
    assert 0.0 <= diagnostics["patient_specific_fraction"] <= 1.0
    assert set(model.volumetric_gate_values()) == {
        "shared.layer_0.head_0",
        "shared.layer_0.head_1",
    }

    restored = TxTVolumetric(**kwargs).eval()
    restored.load_state_dict(model.state_dict(), strict=True)
    with torch.no_grad():
        observed = restored(x)
    for left, right in zip(expected, observed):
        torch.testing.assert_close(left, right, rtol=0, atol=0)


def test_no_tupe_v2_remains_patient_specific_through_direct_expression_qkv(tmp_path) -> None:
    genes = ["A", "B", "C"]
    embedding_file = tmp_path / "embedding.csv"
    pd.DataFrame(torch.eye(3, 4).numpy(), index=genes).to_csv(embedding_file)
    model = TxTVolumetric(
        embed_file=str(embedding_file),
        gene_list=genes,
        n_heads=2,
        d_model=4,
        dropout=0.0,
        d_ff=8,
        n_layers=1,
        aggfunc="Avgpool",
        d_hidden1=4,
        d_hidden2=2,
        d_output_dict={"task": 2},
        head_norm="none",
        edge_index=_edge_index(),
        tupe_mode="off",
        volumetric_volume_mode="l2",
        volumetric_message_mode="expression_contrast",
        volumetric_output_norm="rms",
        volumetric_gate_mode="per_head",
        volumetric_backbone_gradient_mode="detached",
        volumetric_gate_init=0.4,
        volumetric_dropout=0.0,
    ).eval()
    encoder = model.encoder_modules()[0].encoder
    left = torch.tensor([[0.0, 1.0, 4.0]])
    right = torch.tensor([[0.0, 4.0, 1.0]])
    with torch.no_grad():
        left_output = model(left)[0]
        right_output = model(right)[0]

    assert encoder.tupe_mode == "off"
    assert not torch.allclose(left_output, right_output)


def test_old_extra_state_defaults_to_legacy_v2_modes(tmp_path) -> None:
    genes = ["A", "B", "C"]
    embedding_file = tmp_path / "embedding.csv"
    pd.DataFrame(torch.eye(3, 4).numpy(), index=genes).to_csv(embedding_file)
    kwargs = {
        "embed_file": str(embedding_file),
        "gene_list": genes,
        "n_heads": 2,
        "d_model": 4,
        "dropout": 0.0,
        "d_ff": 8,
        "n_layers": 1,
        "aggfunc": "Avgpool",
        "d_hidden1": 4,
        "d_hidden2": 2,
        "d_output_dict": {"task": 2},
        "head_norm": "none",
        "edge_index": _edge_index(),
        "volumetric_dropout": 0.0,
    }
    source = TxTVolumetric(**kwargs)
    state = source.state_dict()
    old_extra_state = dict(state["_extra_state"])
    for key in (
        "tupe_mode",
        "volumetric_message_mode",
        "volumetric_output_norm",
        "volumetric_gate_mode",
        "volumetric_backbone_gradient_mode",
    ):
        old_extra_state.pop(key)
    state["_extra_state"] = old_extra_state

    restored = TxTVolumetric(**kwargs)
    restored.load_state_dict(state, strict=True)
    assert restored.volumetric_message_mode == "legacy"
    assert restored.volumetric_output_norm == "none"
    assert restored.volumetric_gate_mode == "scalar"
    assert restored.volumetric_backbone_gradient_mode == "coupled"
    assert restored.encoder_modules()[0].encoder.tupe_mode == "on"


def test_checkpoint_tupe_mode_mismatch_is_rejected_clearly(tmp_path) -> None:
    genes = ["A", "B", "C"]
    embedding_file = tmp_path / "embedding.csv"
    pd.DataFrame(torch.eye(3, 4).numpy(), index=genes).to_csv(embedding_file)
    kwargs = {
        "embed_file": str(embedding_file),
        "gene_list": genes,
        "n_heads": 2,
        "d_model": 4,
        "dropout": 0.0,
        "d_ff": 8,
        "n_layers": 1,
        "aggfunc": "Avgpool",
        "d_hidden1": 4,
        "d_hidden2": 2,
        "d_output_dict": {"task": 2},
        "head_norm": "none",
        "edge_index": _edge_index(),
        "volumetric_dropout": 0.0,
    }
    enabled = TxTVolumetric(**kwargs, tupe_mode="on")
    disabled = TxTVolumetric(**kwargs, tupe_mode="off")
    with pytest.raises(
        RuntimeError,
        match=r"Checkpoint tupe_mode='on'.*constructed model value 'off'",
    ):
        disabled.load_state_dict(enabled.state_dict(), strict=True)


def test_checkpoint_volume_mode_mismatch_is_rejected_clearly(tmp_path) -> None:
    genes = ["A", "B", "C"]
    embedding_file = tmp_path / "embedding.csv"
    pd.DataFrame(torch.eye(3, 4).numpy(), index=genes).to_csv(embedding_file)
    kwargs = {
        "embed_file": str(embedding_file),
        "gene_list": genes,
        "n_heads": 2,
        "d_model": 4,
        "dropout": 0.0,
        "d_ff": 8,
        "n_layers": 1,
        "aggfunc": "Avgpool",
        "d_hidden1": 4,
        "d_hidden2": 2,
        "d_output_dict": {"task": 2},
        "head_norm": "none",
        "edge_index": _edge_index(),
        "volumetric_dropout": 0.0,
    }
    raw = TxTVolumetric(**kwargs, volumetric_volume_mode="raw")
    l2 = TxTVolumetric(**kwargs, volumetric_volume_mode="l2")
    with pytest.raises(
        RuntimeError,
        match=r"Checkpoint volumetric_volume_mode='raw'.*constructed model value 'l2'",
    ):
        l2.load_state_dict(raw.state_dict(), strict=True)
