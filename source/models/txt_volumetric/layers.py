from __future__ import annotations

import math

import torch
import torch.nn as nn

from .graph import validate_edge_index


VOLUMETRIC_VOLUME_MODES = frozenset({"raw", "l2"})
VOLUMETRIC_MESSAGE_MODES = frozenset({"legacy", "expression_contrast"})
VOLUMETRIC_OUTPUT_NORMS = frozenset({"none", "rms"})
VOLUMETRIC_GATE_MODES = frozenset({"scalar", "per_head"})
VOLUMETRIC_BACKBONE_GRADIENT_MODES = frozenset({"coupled", "detached"})


def validate_volumetric_volume_mode(volume_mode: str) -> str:
    """Validate and return the volume scaling mode used by VMA logits."""

    if not isinstance(volume_mode, str) or volume_mode not in VOLUMETRIC_VOLUME_MODES:
        choices = ", ".join(sorted(VOLUMETRIC_VOLUME_MODES))
        raise ValueError(f"volume_mode must be one of {{{choices}}}; received {volume_mode!r}.")
    return volume_mode


def _validate_choice(value: str, choices: frozenset[str], name: str) -> str:
    if not isinstance(value, str) or value not in choices:
        rendered = ", ".join(sorted(choices))
        raise ValueError(f"{name} must be one of {{{rendered}}}; received {value!r}.")
    return value


def volumetric_volume(
    q_i: torch.Tensor,
    k_i: torch.Tensor,
    k_j: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute ``sqrt(clamp(det(Z^T Z), 0) + eps)`` in float32.

    The three inputs contain raw (not normalized) per-head vectors and may have
    any common leading shape.  The last dimension is the head dimension.
    """

    if q_i.shape != k_i.shape or q_i.shape != k_j.shape:
        raise ValueError("q_i, k_i and k_j must have identical shapes.")
    if q_i.ndim == 0:
        raise ValueError("Volumetric vectors must have a final feature dimension.")
    if eps < 0 or not math.isfinite(eps):
        raise ValueError("eps must be finite and non-negative.")
    # For Z=[q_i,k_i,k_j], compute det(Z^T Z) analytically. This is the same
    # 3x3 Gram determinant as torch.linalg.det, but avoids materializing Z and
    # Gram tensors and avoids the batched-linalg resize warning on Apple MPS.
    # The disabled autocast context guarantees an FP32 determinant.
    with torch.autocast(device_type=q_i.device.type, enabled=False):
        q = q_i.float()
        anchor = k_i.float()
        neighbor = k_j.float()
        q_q = (q * q).sum(dim=-1)
        q_anchor = (q * anchor).sum(dim=-1)
        q_neighbor = (q * neighbor).sum(dim=-1)
        anchor_anchor = (anchor * anchor).sum(dim=-1)
        anchor_neighbor = (anchor * neighbor).sum(dim=-1)
        neighbor_neighbor = (neighbor * neighbor).sum(dim=-1)
        determinant = (
            q_q * (anchor_anchor * neighbor_neighbor - anchor_neighbor.square())
            - q_anchor
            * (q_anchor * neighbor_neighbor - anchor_neighbor * q_neighbor)
            + q_neighbor
            * (q_anchor * anchor_neighbor - anchor_anchor * q_neighbor)
        )
    return torch.sqrt(torch.clamp_min(determinant, 0.0) + eps)


def segment_softmax(
    logits: torch.Tensor,
    segment_index: torch.Tensor,
    num_segments: int,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Softmax over sparse edges grouped by destination.

    ``logits`` has shape ``[..., E]`` and ``segment_index`` has shape ``[E]``.
    The implementation allocates only ``[..., num_segments]`` scratch tensors
    and uses native ``scatter_reduce_``/``scatter_add_`` operations.
    """

    if logits.ndim < 1:
        raise ValueError("logits must have an edge dimension.")
    if num_segments < 0:
        raise ValueError("num_segments must be non-negative.")
    segment_index = torch.as_tensor(segment_index, dtype=torch.long, device=logits.device).flatten()
    if segment_index.numel() != logits.size(-1):
        raise ValueError("segment_index must contain one destination per logit.")
    if segment_index.numel() == 0:
        return logits.clone()
    if int(segment_index.min().item()) < 0 or int(segment_index.max().item()) >= num_segments:
        raise ValueError("segment_index contains an out-of-range destination.")

    expanded_index = segment_index.view(*([1] * (logits.ndim - 1)), -1).expand_as(logits)
    if mask is None:
        valid = torch.ones_like(logits, dtype=torch.bool)
    else:
        valid = torch.as_tensor(mask, device=logits.device, dtype=torch.bool)
        try:
            valid = torch.broadcast_to(valid, logits.shape)
        except RuntimeError as exc:
            raise ValueError("mask is not broadcastable to logits.") from exc

    negative_infinity = torch.tensor(float("-inf"), device=logits.device, dtype=logits.dtype)
    masked_logits = torch.where(valid, logits, negative_infinity)
    segment_max = torch.full(
        (*logits.shape[:-1], num_segments),
        float("-inf"),
        device=logits.device,
        dtype=logits.dtype,
    )
    segment_max.scatter_reduce_(-1, expanded_index, masked_logits, reduce="amax", include_self=True)
    edge_max = segment_max.gather(-1, expanded_index)
    # A fully masked segment has max=-inf.  Replacing it before subtraction
    # prevents -inf - -inf from creating NaNs; its exponential remains zero.
    safe_edge_max = torch.where(torch.isfinite(edge_max), edge_max, torch.zeros_like(edge_max))
    exponentials = torch.where(valid, torch.exp(logits - safe_edge_max), torch.zeros_like(logits))
    segment_sum = torch.zeros(
        (*logits.shape[:-1], num_segments),
        device=logits.device,
        dtype=logits.dtype,
    )
    segment_sum.scatter_add_(-1, expanded_index, exponentials)
    edge_sum = segment_sum.gather(-1, expanded_index)
    return torch.where(
        edge_sum > 0,
        exponentials / edge_sum.clamp_min(torch.finfo(logits.dtype).tiny),
        torch.zeros_like(logits),
    )


class VolumetricAttentionAugmentation(nn.Module):
    """Sparse PPI volumetric attention fused into a TxT attention layer."""

    def __init__(
        self,
        n_heads: int,
        d_head: int,
        edge_index: torch.Tensor,
        *,
        n_nodes: int,
        beta: float = 1.0,
        eps: float = 1e-8,
        dropout: float = 0.2,
        gate_init: float = 0.0,
        volumetric_volume_mode: str = "raw",
        volumetric_message_mode: str = "legacy",
        volumetric_output_norm: str = "none",
        volumetric_gate_mode: str = "scalar",
        volumetric_backbone_gradient_mode: str = "coupled",
    ):
        super().__init__()
        if n_heads <= 0 or d_head <= 0:
            raise ValueError("n_heads and d_head must be positive.")
        if not math.isfinite(beta):
            raise ValueError("beta must be finite.")
        if eps < 0 or not math.isfinite(eps):
            raise ValueError("eps must be finite and non-negative.")
        if not 0 <= dropout <= 1:
            raise ValueError("dropout must be in [0, 1].")
        if not math.isfinite(gate_init):
            raise ValueError("gate_init must be finite.")
        volume_mode = validate_volumetric_volume_mode(volumetric_volume_mode)
        message_mode = _validate_choice(
            volumetric_message_mode,
            VOLUMETRIC_MESSAGE_MODES,
            "volumetric_message_mode",
        )
        output_norm = _validate_choice(
            volumetric_output_norm,
            VOLUMETRIC_OUTPUT_NORMS,
            "volumetric_output_norm",
        )
        gate_mode = _validate_choice(
            volumetric_gate_mode,
            VOLUMETRIC_GATE_MODES,
            "volumetric_gate_mode",
        )
        backbone_gradient_mode = _validate_choice(
            volumetric_backbone_gradient_mode,
            VOLUMETRIC_BACKBONE_GRADIENT_MODES,
            "volumetric_backbone_gradient_mode",
        )
        edge_index = validate_edge_index(edge_index, n_nodes)

        self.n_heads = int(n_heads)
        self.d_head = int(d_head)
        self.n_nodes = int(n_nodes)
        self.volume_mode = volume_mode
        self.message_mode = message_mode
        self.output_norm = output_norm
        self.gate_mode = gate_mode
        self.backbone_gradient_mode = backbone_gradient_mode
        self.register_buffer("edge_index", edge_index.clone(), persistent=True)
        degrees = (
            torch.bincount(edge_index[0], minlength=n_nodes)
            if edge_index.numel()
            else torch.zeros(n_nodes, dtype=torch.long)
        )
        self.register_buffer("has_neighbors", degrees > 0, persistent=True)
        self.register_buffer("beta", torch.tensor(float(beta), dtype=torch.float32), persistent=True)
        self.register_buffer("eps", torch.tensor(float(eps), dtype=torch.float32), persistent=True)

        self.self_gate = nn.Linear(d_head, d_head, bias=False)
        self.neighbor_gate = nn.Linear(d_head, d_head, bias=False)
        self.attention_dropout = nn.Dropout(dropout)
        if gate_mode == "scalar":
            gamma = torch.tensor(float(gate_init), dtype=torch.float32)
        else:
            gamma = torch.full((n_heads,), float(gate_init), dtype=torch.float32)
        self.gamma = nn.Parameter(gamma)

        # The legacy path deliberately owns no additional parameters.  This
        # preserves its state-dict keys, initialization RNG and numerical path.
        # V2 injects a small, head-specific projection of standardized gene
        # expression into the already projected backbone Q/K/V tensors.
        if message_mode == "expression_contrast":
            projection_dim = n_heads * d_head
            self.expression_q_projection = nn.Linear(1, projection_dim, bias=False)
            self.expression_k_projection = nn.Linear(1, projection_dim, bias=False)
            self.expression_v_projection = nn.Linear(1, projection_dim, bias=False)
            for projection in (
                self.expression_q_projection,
                self.expression_k_projection,
                self.expression_v_projection,
            ):
                nn.init.normal_(projection.weight, mean=0.0, std=1.0 / d_head)
        else:
            self.expression_q_projection = None
            self.expression_k_projection = None
            self.expression_v_projection = None

        self.capture_attention = False
        self._last_capture: dict[str, torch.Tensor] | None = None
        self._expression_context: torch.Tensor | None = None
        self._expression_valid_mask_context: torch.Tensor | None = None
        for name in (
            "q_norm",
            "k_norm",
            "v_norm",
            "baseline_output_norm",
            "vma_output_norm",
            "gated_vma_norm",
            "attention_entropy",
            "attention_entropy_normalized",
            "volume_mean",
            "volume_std",
            "volume_floor_fraction",
            "pre_norm_vma_output_norm",
            "neighbor_message_norm",
            "self_message_norm",
            "patient_specific_rms",
            "patient_specific_fraction",
            "gated_patient_specific_norm",
            "expression_valid_fraction",
            "standardized_expression_mean",
            "standardized_expression_std",
            "output_norm_scale_factor",
        ):
            self.register_buffer(f"_last_{name}", torch.tensor(float("nan")), persistent=False)

    @property
    def destination_index(self) -> torch.Tensor:
        return self.edge_index[0]

    @property
    def source_index(self) -> torch.Tensor:
        return self.edge_index[1]

    @property
    def last_attention_weights(self) -> torch.Tensor | None:
        if self._last_capture is None:
            return None
        return self._last_capture["attention_weights"]

    def set_beta(self, beta: float) -> None:
        if not math.isfinite(beta):
            raise ValueError("beta must be finite.")
        self.beta.fill_(float(beta))

    def set_volume_mode(self, volume_mode: str) -> None:
        self.volume_mode = validate_volumetric_volume_mode(volume_mode)

    def set_expression_context(
        self,
        expression: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> None:
        """Provide the sample expression consumed by the next enclosing forward.

        ``TxTVolumetric`` uses this narrow context bridge because the unchanged
        baseline attention API exposes Q/K/V and TUPE, but not the original
        expression tensor. Direct operator users should prefer the explicit
        ``expression=`` argument accepted by ``forward``/``compute_vma_output``.
        """

        self._expression_context = expression
        self._expression_valid_mask_context = valid_mask

    def clear_expression_context(self) -> None:
        self._expression_context = None
        self._expression_valid_mask_context = None

    def effective_gate(self, *, reference: torch.Tensor | None = None) -> torch.Tensor:
        gate = torch.tanh(self.gamma)
        if self.gate_mode == "per_head":
            gate = gate.view(1, self.n_heads, 1, 1)
        if reference is not None:
            gate = gate.to(device=reference.device, dtype=reference.dtype)
        return gate

    def _standardize_expression(
        self,
        expression: torch.Tensor,
        valid_mask: torch.Tensor | None,
        *,
        device: torch.device,
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        expression = torch.as_tensor(expression, device=device, dtype=torch.float32)
        if expression.ndim != 2 or expression.shape != (batch_size, self.n_nodes):
            raise ValueError(
                "expression must have shape [batch, nodes]; received "
                f"{tuple(expression.shape)}, expected {(batch_size, self.n_nodes)}."
            )
        valid = torch.isfinite(expression)
        if valid_mask is not None:
            supplied_mask = torch.as_tensor(valid_mask, device=device, dtype=torch.bool)
            try:
                supplied_mask = torch.broadcast_to(supplied_mask, expression.shape)
            except RuntimeError as exc:
                raise ValueError(
                    "expression_valid_mask is not broadcastable to [batch, nodes]."
                ) from exc
            valid = valid & supplied_mask

        safe_expression = torch.where(valid, expression, torch.zeros_like(expression))
        counts = valid.sum(dim=-1, keepdim=True)
        safe_counts = counts.clamp_min(1).to(dtype=expression.dtype)
        means = safe_expression.sum(dim=-1, keepdim=True) / safe_counts
        centered = torch.where(valid, expression - means, torch.zeros_like(expression))
        variances = centered.square().sum(dim=-1, keepdim=True) / safe_counts
        scale = torch.sqrt(variances + max(float(self.eps.item()), 1e-12))
        standardized = torch.where(valid, centered / scale, torch.zeros_like(centered))
        return standardized, valid

    def _expression_aware_qkv(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        standardized_expression: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if (
            self.expression_q_projection is None
            or self.expression_k_projection is None
            or self.expression_v_projection is None
        ):
            raise RuntimeError("expression_contrast projections were not constructed.")
        batch_size = standardized_expression.size(0)

        def project(projection: nn.Linear) -> torch.Tensor:
            projected = projection(standardized_expression.unsqueeze(-1))
            return projected.view(
                batch_size,
                self.n_nodes,
                self.n_heads,
                self.d_head,
            ).permute(0, 2, 1, 3)

        return (
            q + project(self.expression_q_projection).to(dtype=q.dtype),
            k + project(self.expression_k_projection).to(dtype=k.dtype),
            v + project(self.expression_v_projection).to(dtype=v.dtype),
        )

    def _normalize_vma_output(
        self,
        output: torch.Tensor,
        reference: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.output_norm == "none":
            return output, output.new_ones((), dtype=torch.float32)
        floor = max(float(self.eps.item()), 1e-12)
        # Match scale independently for every patient.  A global batch RMS
        # would make one sample's representation depend on the other samples
        # present in the evaluation/training batch.
        output_rms = output.float().square().mean(dim=(1, 2, 3), keepdim=True).add(
            floor
        ).sqrt()
        if reference is None:
            target_rms = torch.ones_like(output_rms)
        else:
            if reference.shape != output.shape:
                raise ValueError("RMS normalization reference must match the VMA output shape.")
            target_rms = reference.float().square().mean(
                dim=(1, 2, 3),
                keepdim=True,
            ).add(floor).sqrt()
        scale = (target_rms / output_rms).detach()
        return (output.float() * scale).to(dtype=output.dtype), scale

    def set_attention_capture(self, enabled: bool, *, clear: bool = True) -> None:
        self.capture_attention = bool(enabled)
        if clear:
            self._last_capture = None

    def attention_capture(self, *, clear: bool = False) -> dict[str, torch.Tensor] | None:
        capture = self._last_capture
        if clear:
            self._last_capture = None
        return capture

    def diagnostics(self) -> dict[str, float | int | str]:
        raw_gate = self.gamma.detach().float().flatten()
        effective_gate = torch.tanh(raw_gate)
        result: dict[str, float | int | str] = {
            "n_nodes": self.n_nodes,
            "n_directed_edges": int(self.edge_index.size(1)),
            "covered_node_count": int(self.has_neighbors.sum().item()),
            "gamma_raw": float(raw_gate.mean().cpu().item()),
            "gamma_effective": float(effective_gate.mean().cpu().item()),
            "gamma_effective_abs_mean": float(effective_gate.abs().mean().cpu().item()),
            "gamma_effective_min": float(effective_gate.min().cpu().item()),
            "gamma_effective_max": float(effective_gate.max().cpu().item()),
            "self_gate_weight_norm": float(self.self_gate.weight.detach().norm().cpu().item()),
            "neighbor_gate_weight_norm": float(self.neighbor_gate.weight.detach().norm().cpu().item()),
            "beta": float(self.beta.detach().cpu().item()),
            "eps": float(self.eps.detach().cpu().item()),
            "volume_mode": self.volume_mode,
            "message_mode": self.message_mode,
            "output_norm": self.output_norm,
            "gate_mode": self.gate_mode,
            "backbone_gradient_mode": self.backbone_gradient_mode,
        }
        if self.gate_mode == "per_head":
            for head_index, (raw_value, effective_value) in enumerate(
                zip(raw_gate, effective_gate)
            ):
                result[f"gamma_raw_head_{head_index}"] = float(raw_value.cpu().item())
                result[f"gamma_effective_head_{head_index}"] = float(
                    effective_value.cpu().item()
                )
        if self.expression_q_projection is not None:
            result["expression_q_projection_norm"] = float(
                self.expression_q_projection.weight.detach().norm().cpu().item()
            )
            result["expression_k_projection_norm"] = float(
                self.expression_k_projection.weight.detach().norm().cpu().item()
            )
            result["expression_v_projection_norm"] = float(
                self.expression_v_projection.weight.detach().norm().cpu().item()
            )
        for name in (
            "q_norm",
            "k_norm",
            "v_norm",
            "baseline_output_norm",
            "vma_output_norm",
            "gated_vma_norm",
            "attention_entropy",
            "attention_entropy_normalized",
            "volume_mean",
            "volume_std",
            "volume_floor_fraction",
            "pre_norm_vma_output_norm",
            "neighbor_message_norm",
            "self_message_norm",
            "patient_specific_rms",
            "patient_specific_fraction",
            "gated_patient_specific_norm",
            "expression_valid_fraction",
            "standardized_expression_mean",
            "standardized_expression_std",
            "output_norm_scale_factor",
        ):
            result[name] = float(getattr(self, f"_last_{name}").detach().cpu().item())
        return result

    def _sparse_mask(self, mask: torch.Tensor | None, batch_size: int) -> torch.Tensor | None:
        if mask is None:
            return None
        mask = torch.as_tensor(mask, device=self.edge_index.device)
        if mask.ndim == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)
        elif mask.ndim == 3:
            mask = mask.unsqueeze(1)
        elif mask.ndim != 4:
            raise ValueError("attention mask must have 2, 3 or 4 dimensions.")
        try:
            mask = torch.broadcast_to(mask, (batch_size, self.n_heads, self.n_nodes, self.n_nodes))
        except RuntimeError as exc:
            raise ValueError("attention mask is not broadcastable to [batch, heads, nodes, nodes].") from exc
        return mask[:, :, self.destination_index, self.source_index] != 0

    def _record_diagnostics(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        baseline_output: torch.Tensor | None,
        vma_output: torch.Tensor,
        attention_weights: torch.Tensor,
        volumes: torch.Tensor,
        sparse_mask: torch.Tensor | None,
        *,
        pre_norm_output: torch.Tensor,
        neighbor_message: torch.Tensor,
        self_message: torch.Tensor,
        standardized_expression: torch.Tensor | None,
        expression_valid: torch.Tensor | None,
        output_norm_scale: torch.Tensor,
    ) -> None:
        with torch.no_grad():
            if attention_weights.numel():
                destination = self.destination_index
                expanded_destination = destination.view(1, 1, -1).expand_as(attention_weights)
                valid_edges = (
                    torch.ones_like(attention_weights, dtype=torch.bool)
                    if sparse_mask is None
                    else sparse_mask
                )
                edge_entropy = torch.where(
                    valid_edges,
                    -attention_weights * attention_weights.clamp_min(1e-12).log(),
                    torch.zeros_like(attention_weights),
                )
                segment_entropy = torch.zeros(
                    (*attention_weights.shape[:-1], self.n_nodes),
                    device=attention_weights.device,
                    dtype=attention_weights.dtype,
                )
                segment_entropy.scatter_add_(-1, expanded_destination, edge_entropy)
                segment_counts = torch.zeros_like(segment_entropy, dtype=torch.long)
                segment_counts.scatter_add_(
                    -1,
                    expanded_destination,
                    valid_edges.to(dtype=torch.long),
                )
                normalizable_segments = segment_counts > 1
                normalized_segment_entropy = segment_entropy / segment_counts.clamp_min(2).log()
                normalizable_count = normalizable_segments.sum()
                normalizable_mean = (
                    torch.where(
                        normalizable_segments,
                        normalized_segment_entropy,
                        torch.zeros_like(normalized_segment_entropy),
                    ).sum()
                    / normalizable_count.clamp_min(1)
                )
                # A collection containing only singleton segments is
                # necessarily at its segmented maximum entropy.  Tensor-only
                # selection avoids a device synchronization on every batch.
                singleton_fallback = (segment_counts == 1).any().to(
                    dtype=attention_weights.dtype
                )
                attention_entropy_normalized = torch.where(
                    normalizable_count > 0,
                    normalizable_mean.clamp(0.0, 1.0),
                    singleton_fallback,
                )
                active_volumes = volumes[valid_edges]
            else:
                attention_entropy_normalized = torch.zeros((), device=q.device)
                active_volumes = volumes.flatten()

            if active_volumes.numel():
                volume_floor = self.eps.sqrt().to(device=active_volumes.device)
                volume_mean = active_volumes.mean()
                volume_std = active_volumes.std(unbiased=False)
                volume_floor_fraction = (active_volumes <= volume_floor).float().mean()
            else:
                volume_mean = torch.tensor(float("nan"), device=q.device)
                volume_std = torch.tensor(float("nan"), device=q.device)
                volume_floor_fraction = torch.tensor(float("nan"), device=q.device)

            patient_centered = vma_output.float() - vma_output.float().mean(
                dim=0,
                keepdim=True,
            )
            patient_specific_rms = patient_centered.square().mean().sqrt()
            vma_rms = vma_output.float().square().mean().sqrt()
            patient_specific_fraction = torch.where(
                vma_rms > 0,
                patient_specific_rms / vma_rms.clamp_min(1e-12),
                torch.zeros_like(patient_specific_rms),
            )
            gate = self.effective_gate(reference=vma_output)
            gated_patient_specific_norm = (gate * patient_centered).float().norm()

            if standardized_expression is not None and expression_valid is not None:
                expression_valid_fraction = expression_valid.float().mean()
                valid_expression_values = standardized_expression[expression_valid]
                if valid_expression_values.numel():
                    standardized_expression_mean = valid_expression_values.mean()
                    standardized_expression_std = valid_expression_values.std(unbiased=False)
                else:
                    standardized_expression_mean = torch.tensor(float("nan"), device=q.device)
                    standardized_expression_std = torch.tensor(float("nan"), device=q.device)
            else:
                expression_valid_fraction = torch.tensor(float("nan"), device=q.device)
                standardized_expression_mean = torch.tensor(float("nan"), device=q.device)
                standardized_expression_std = torch.tensor(float("nan"), device=q.device)

            values = {
                "q_norm": q.float().norm(),
                "k_norm": k.float().norm(),
                "v_norm": v.float().norm(),
                "baseline_output_norm": (
                    baseline_output.float().norm()
                    if baseline_output is not None
                    else torch.tensor(float("nan"), device=q.device)
                ),
                "vma_output_norm": vma_output.float().norm(),
                "gated_vma_norm": (gate * vma_output).float().norm(),
                "attention_entropy": (
                    -(attention_weights * attention_weights.clamp_min(1e-12).log()).sum(dim=-1).mean()
                    if attention_weights.numel()
                    else torch.tensor(0.0, device=q.device)
                ),
                "attention_entropy_normalized": attention_entropy_normalized,
                "volume_mean": volume_mean,
                "volume_std": volume_std,
                "volume_floor_fraction": volume_floor_fraction,
                "pre_norm_vma_output_norm": pre_norm_output.float().norm(),
                "neighbor_message_norm": neighbor_message.float().norm(),
                "self_message_norm": self_message.float().norm(),
                "patient_specific_rms": patient_specific_rms,
                "patient_specific_fraction": patient_specific_fraction,
                "gated_patient_specific_norm": gated_patient_specific_norm,
                "expression_valid_fraction": expression_valid_fraction,
                "standardized_expression_mean": standardized_expression_mean,
                "standardized_expression_std": standardized_expression_std,
                "output_norm_scale_factor": output_norm_scale.float().mean(),
            }
            for name, value in values.items():
                target = getattr(self, f"_last_{name}")
                target.copy_(value.detach().to(target.device))

    def compute_vma_output(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        tupe: torch.Tensor,
        mask: torch.Tensor | None = None,
        *,
        expression: torch.Tensor | None = None,
        expression_valid_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return VMA output, normalized edge weights, volumes and edge logits."""

        output, weights, volumes, logits, _, _ = self._compute_vma_output(
            q,
            k,
            v,
            tupe,
            mask,
            expression=expression,
            expression_valid_mask=expression_valid_mask,
            normalization_reference=None,
        )
        return output, weights, volumes, logits

    def _compute_vma_output(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        tupe: torch.Tensor,
        mask: torch.Tensor | None = None,
        *,
        expression: torch.Tensor | None = None,
        expression_valid_mask: torch.Tensor | None = None,
        normalization_reference: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        dict[str, torch.Tensor | None],
    ]:
        """Internal compute path with sparse mask and V2 diagnostic intermediates."""

        if q.shape != k.shape or q.shape != v.shape:
            raise ValueError("Projected q, k and v must have identical shapes.")
        if q.ndim != 4:
            raise ValueError("Projected q, k and v must have shape [batch, heads, nodes, d_head].")
        batch_size, n_heads, n_nodes, d_head = q.shape
        if n_heads != self.n_heads or n_nodes != self.n_nodes or d_head != self.d_head:
            raise ValueError(
                "Projected attention shape does not match the VMA graph/operator: "
                f"received {(n_heads, n_nodes, d_head)}, expected "
                f"{(self.n_heads, self.n_nodes, self.d_head)}."
            )
        if tupe.ndim != 4 or tupe.size(0) != batch_size or tupe.size(1) not in {1, n_heads}:
            raise ValueError("tupe must have shape [batch, heads, nodes, nodes].")
        if tupe.size(-2) != n_nodes or tupe.size(-1) != n_nodes:
            raise ValueError("tupe node dimensions do not match the VMA graph.")

        if self.backbone_gradient_mode == "detached":
            vma_q = q.detach()
            vma_k = k.detach()
            vma_v = v.detach()
            vma_tupe = tupe.detach()
        else:
            vma_q = q
            vma_k = k
            vma_v = v
            vma_tupe = tupe

        standardized_expression: torch.Tensor | None = None
        expression_valid: torch.Tensor | None = None
        if self.message_mode == "expression_contrast":
            resolved_expression = expression
            resolved_mask = expression_valid_mask
            if resolved_expression is None:
                resolved_expression = self._expression_context
            if resolved_mask is None:
                # An enclosing TxTVolumetric forward owns the authoritative
                # validity context even when a caller overrides only the
                # expression tensor explicitly.
                resolved_mask = self._expression_valid_mask_context
            if resolved_expression is None:
                raise ValueError(
                    "volumetric_message_mode='expression_contrast' requires expression."
                )
            standardized_expression, expression_valid = self._standardize_expression(
                resolved_expression,
                resolved_mask,
                device=q.device,
                batch_size=batch_size,
            )
            vma_q, vma_k, vma_v = self._expression_aware_qkv(
                vma_q,
                vma_k,
                vma_v,
                standardized_expression,
            )

        edge_count = self.edge_index.size(1)
        if edge_count == 0:
            empty = q.new_empty((batch_size, n_heads, 0), dtype=torch.float32)
            pre_norm_output = torch.zeros_like(vma_v)
            output, scale = self._normalize_vma_output(
                pre_norm_output,
                normalization_reference,
            )
            metadata: dict[str, torch.Tensor | None] = {
                "q": vma_q,
                "k": vma_k,
                "v": vma_v,
                "pre_norm_output": pre_norm_output,
                "neighbor_message": pre_norm_output,
                "self_message": pre_norm_output,
                "standardized_expression": standardized_expression,
                "expression_valid": expression_valid,
                "output_norm_scale": scale,
            }
            return output, empty, empty, empty, None, metadata

        destination = self.destination_index
        source = self.source_index
        q_i = vma_q.index_select(2, destination)
        k_i = vma_k.index_select(2, destination)
        k_j = vma_k.index_select(2, source)
        v_j = vma_v.index_select(2, source)

        if self.volume_mode == "raw":
            volumes = volumetric_volume(q_i, k_i, k_j, float(self.eps.item()))
        else:
            normalization_eps = max(float(self.eps.item()), torch.finfo(torch.float32).tiny)
            q_unit = q_i.float() / q_i.float().norm(dim=-1, keepdim=True).clamp_min(
                normalization_eps
            )
            k_i_unit = k_i.float() / k_i.float().norm(dim=-1, keepdim=True).clamp_min(
                normalization_eps
            )
            k_j_unit = k_j.float() / k_j.float().norm(dim=-1, keepdim=True).clamp_min(
                normalization_eps
            )
            volumes = volumetric_volume(q_unit, k_i_unit, k_j_unit, float(self.eps.item()))

        dot_neighbor = (q_i * k_j).sum(dim=-1).float()
        tupe_edges = vma_tupe[:, :, destination, source].float()
        if self.volume_mode == "raw":
            # Preserve the legacy formula exactly for checkpoint/reproduction
            # compatibility.  dot_anchor is constant within each destination
            # segment, but remains part of the historical raw path.
            dot_anchor = (q_i * k_i).sum(dim=-1).float()
            logits = (
                -self.beta.float() * volumes + dot_anchor + dot_neighbor
            ) / math.sqrt(self.d_head) + tupe_edges
        else:
            # The dimensionless unit volume is already on an O(1) scale and
            # must not be divided by sqrt(d_head).  The content dot product
            # remains raw and receives standard scaled-dot-product scaling.
            logits = (
                -self.beta.float() * volumes
                + dot_neighbor / math.sqrt(self.d_head)
                + tupe_edges
            )

        sparse_mask = self._sparse_mask(mask, batch_size)
        attention_weights = segment_softmax(logits, destination, n_nodes, sparse_mask)
        dropped_weights = self.attention_dropout(attention_weights).to(dtype=vma_v.dtype)

        weighted_values = dropped_weights.unsqueeze(-1) * v_j
        expanded_destination = destination.view(1, 1, -1, 1).expand(
            batch_size, n_heads, edge_count, d_head
        )
        neighbor_values = torch.zeros_like(vma_v)
        neighbor_values.scatter_add_(2, expanded_destination, weighted_values)

        self_values = vma_v * torch.sigmoid(self.self_gate(vma_q))
        neighbor_values = neighbor_values * torch.sigmoid(self.neighbor_gate(vma_q))
        if self.message_mode == "legacy":
            pre_norm_output = 0.5 * (self_values + neighbor_values)
        else:
            # V2 is a graph contrast, not a positive static self offset.
            pre_norm_output = neighbor_values - self_values

        if sparse_mask is None:
            active_nodes = self.has_neighbors.view(1, 1, n_nodes, 1)
        else:
            active_counts = torch.zeros(
                (batch_size, n_heads, n_nodes),
                device=vma_q.device,
                dtype=torch.long,
            )
            expanded_sparse_destination = destination.view(1, 1, -1).expand(batch_size, n_heads, -1)
            active_counts.scatter_add_(2, expanded_sparse_destination, sparse_mask.to(dtype=torch.long))
            active_nodes = (active_counts > 0).unsqueeze(-1)
        active_nodes = active_nodes.to(dtype=pre_norm_output.dtype)
        pre_norm_output = pre_norm_output * active_nodes
        self_values = self_values * active_nodes
        neighbor_values = neighbor_values * active_nodes
        vma_output, output_norm_scale = self._normalize_vma_output(
            pre_norm_output,
            normalization_reference,
        )
        metadata = {
            "q": vma_q,
            "k": vma_k,
            "v": vma_v,
            "pre_norm_output": pre_norm_output,
            "neighbor_message": neighbor_values,
            "self_message": self_values,
            "standardized_expression": standardized_expression,
            "expression_valid": expression_valid,
            "output_norm_scale": output_norm_scale,
        }
        return vma_output, attention_weights, volumes, logits, sparse_mask, metadata

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
        expression_valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        vma_output, attention_weights, volumes, logits, sparse_mask, metadata = (
            self._compute_vma_output(
                q,
                k,
                v,
                tupe,
                mask,
                expression=expression,
                expression_valid_mask=expression_valid_mask,
                normalization_reference=attention_output,
            )
        )
        self._record_diagnostics(
            metadata["q"],
            metadata["k"],
            metadata["v"],
            attention_output,
            vma_output,
            attention_weights,
            volumes,
            sparse_mask,
            pre_norm_output=metadata["pre_norm_output"],
            neighbor_message=metadata["neighbor_message"],
            self_message=metadata["self_message"],
            standardized_expression=metadata["standardized_expression"],
            expression_valid=metadata["expression_valid"],
            output_norm_scale=metadata["output_norm_scale"],
        )
        if self.capture_attention:
            self._last_capture = {
                "attention_weights": attention_weights.detach().cpu(),
                "volumes": volumes.detach().cpu(),
                "logits": logits.detach().cpu(),
                "edge_index": self.edge_index.detach().cpu(),
            }
            if metadata["standardized_expression"] is not None:
                self._last_capture["standardized_expression"] = metadata[
                    "standardized_expression"
                ].detach().cpu()
        if attention_output is None:
            return vma_output
        return attention_output + self.effective_gate(reference=vma_output) * vma_output


# Short aliases make the pure primitives/operator convenient in tests and tools.
compute_volumetric_volume = volumetric_volume
sparse_segment_softmax = segment_softmax
VolumetricAttention = VolumetricAttentionAugmentation


__all__ = [
    "VolumetricAttention",
    "VolumetricAttentionAugmentation",
    "compute_volumetric_volume",
    "segment_softmax",
    "sparse_segment_softmax",
    "volumetric_volume",
]
