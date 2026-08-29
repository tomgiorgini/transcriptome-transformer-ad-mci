#!/usr/bin/env python3
"""Validation-only mechanistic audit for VMA v2 paired runs.

This script intentionally has no split selector: it reconstructs each primary
VMA candidate and evaluates only the validation partition saved by the worker.
It compares the trained ensemble with two inference-only ablations (gate=0 and
beta=0) and measures rank-1 sparse-attention variation on PPI destinations with
at least two neighbours.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.scripts.paper_comparison import train_txt_multitask as worker  # noqa: E402
from experiments.scripts.paper_comparison.txt_volumetric.common import (  # noqa: E402
    ProtocolError,
    baseline_dir,
    candidate_dir,
    read_json,
    resolve_path,
    validation_checkpoint_score,
    write_json,
)
from experiments.scripts.paper_comparison.txt_volumetric.export_attention import (  # noqa: E402
    collect_sparse_captures,
    namespace_from_saved_args,
    reconstruct_model,
    split_arrays,
)
from source.pipeline.utils import resolve_device  # noqa: E402


ENSEMBLE_AGGREGATION = "arithmetic_mean_softmax_probabilities"
GATE_MAE_THRESHOLD = 1e-4
GATE_MAGNITUDE_THRESHOLD = 1e-3
BETA_ATTENTION_TV_THRESHOLD = 0.01


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit paired VMA v2 runs using validation data only. The script has no test "
            "split option and never reads selected-test artifacts."
        )
    )
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--primary-beta", type=float, default=1.0)
    parser.add_argument("--control-beta", type=float, default=0.0)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--tupe-mode", choices=["on", "off"], default="on")
    parser.add_argument("--batch-size", type=int, default=9)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser


def _checkpoint_rank(path: Path) -> int:
    match = re.search(r"rank(\d+)$", path.stem)
    return int(match.group(1)) if match else 10**9


def ensemble_checkpoint_paths(run_dir: Path) -> list[Path]:
    """Return the effective saved ensemble in deterministic rank order."""

    summary_path = run_dir / "model_summary.json"
    training: dict[str, Any] = {}
    if summary_path.exists():
        summary = read_json(summary_path)
        raw_training = summary.get("training_summary", {}) if isinstance(summary, dict) else {}
        if isinstance(raw_training, dict):
            training = raw_training

    members = training.get("checkpoint_ensemble_members")
    ensemble_payload = training.get("ensemble")
    if not isinstance(members, list) and isinstance(ensemble_payload, dict):
        members = ensemble_payload.get("members")
    if isinstance(members, list) and members:
        ranks = []
        for member in members:
            if not isinstance(member, dict) or member.get("rank") is None:
                raise ProtocolError(f"Malformed ensemble member metadata in {summary_path}.")
            ranks.append(int(member["rank"]))
        if sorted(ranks) != list(range(1, len(ranks) + 1)) or len(set(ranks)) != len(ranks):
            raise ProtocolError(f"Non-contiguous ensemble ranks in {summary_path}: {ranks}.")
        declared_paths = []
        for member in sorted(members, key=lambda item: int(item["rank"])):
            rank = int(member["rank"])
            local_path = run_dir / f"ensemble_checkpoint_rank{rank}.pt"
            if not local_path.exists():
                raise ProtocolError(f"Missing declared rank-{rank} checkpoint: {local_path}")
            declared_names = {
                Path(str(member[key])).name
                for key in ("copied_path", "path")
                if member.get(key) not in {None, ""}
            }
            if declared_names and local_path.name not in declared_names:
                raise ProtocolError(
                    f"Ensemble rank-{rank} metadata points to {sorted(declared_names)!r}, "
                    f"expected {local_path.name!r}."
                )
            declared_paths.append(local_path)
        return declared_paths

    ranked = sorted(run_dir.glob("ensemble_checkpoint_rank*.pt"), key=_checkpoint_rank)
    if ranked:
        effective = training.get("checkpoint_ensemble_effective_k")
        if effective is None and isinstance(ensemble_payload, dict):
            effective = ensemble_payload.get("effective_k")
        if effective is None:
            legacy_metadata = training.get("checkpoint_ensemble")
            if isinstance(legacy_metadata, list):
                effective = len(legacy_metadata)
        effective = len(ranked) if effective is None else int(effective)
        expected_names = [f"ensemble_checkpoint_rank{rank}.pt" for rank in range(1, effective + 1)]
        by_name = {path.name: path for path in ranked}
        missing = [name for name in expected_names if name not in by_name]
        if effective <= 0 or missing:
            raise ProtocolError(
                f"Incomplete checkpoint ensemble in {run_dir}: effective_k={effective}, "
                f"missing={missing}."
            )
        return [by_name[name] for name in expected_names]
    best = run_dir / "best_model.pt"
    if not best.exists():
        raise ProtocolError(f"Missing trained checkpoint in {run_dir}.")
    return [best]


def _load_states(
    model: torch.nn.Module,
    checkpoint_paths: Sequence[Path],
    device: torch.device,
) -> list[dict[str, Any]]:
    states: list[dict[str, Any]] = []
    for checkpoint in checkpoint_paths:
        state = worker.load_checkpoint_state(checkpoint, device)
        model.load_state_dict(state, strict=True)
        states.append(copy.deepcopy(model.state_dict()))
    return states


def _ablated_states(
    model: torch.nn.Module,
    states: Sequence[dict[str, Any]],
    *,
    ablation: str,
) -> list[dict[str, Any]]:
    if ablation not in {"gate0", "beta0"}:
        raise ValueError(f"Unsupported VMA ablation: {ablation}")
    result: list[dict[str, Any]] = []
    iterator = getattr(model, "iter_volumetric_augmentations", None)
    if not callable(iterator):
        raise ProtocolError("The reconstructed model does not expose VMA augmentations.")
    for state in states:
        model.load_state_dict(state, strict=True)
        with torch.no_grad():
            for _, augmentation in iterator():
                if ablation == "gate0":
                    augmentation.gamma.zero_()
                else:
                    augmentation.beta.zero_()
        result.append(copy.deepcopy(model.state_dict()))
    return result


def _evaluate_states(
    model: torch.nn.Module,
    states: Sequence[dict[str, Any]],
    dataset: Any,
    task_specs: Sequence[Any],
    saved_args: argparse.Namespace,
    device: torch.device,
    *,
    batch_size: int,
    max_batches: int | None,
) -> tuple[dict[str, dict[str, Any]], float, float, dict[str, Any]]:
    gene_x, source_y, sample_ids = split_arrays(dataset, "val")
    criteria = worker.build_evaluation_criteria(dataset, list(task_specs), saved_args, device)
    results, loss = worker.evaluate_multitask(
        model,
        dataset,
        "val",
        gene_x,
        source_y,
        sample_ids,
        list(task_specs),
        batch_size,
        device,
        criteria,
        saved_args.task_loss_weights,
        max_batches=max_batches,
        state_dicts=list(states),
        mask_aware_heads=getattr(saved_args, "mask_aware_heads", "off") == "on",
    )
    score, details = worker.compute_checkpoint_value(
        saved_args.checkpoint_metric,
        results,
        loss,
    )
    return results, float(loss), float(score), details


def probability_difference(
    reference: dict[str, dict[str, Any]],
    ablated: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    absolute_total = 0.0
    probability_count = 0
    flips = 0
    sample_count = 0
    for task_name in worker.TASK_NAMES:
        left = np.asarray(reference[task_name]["y_prob"], dtype=np.float64)
        right = np.asarray(ablated[task_name]["y_prob"], dtype=np.float64)
        if left.shape != right.shape:
            raise ProtocolError(f"Probability shape mismatch for {task_name}: {left.shape} vs {right.shape}.")
        left_pred = left.argmax(axis=1) if len(left) else np.asarray([], dtype=np.int64)
        right_pred = right.argmax(axis=1) if len(right) else np.asarray([], dtype=np.int64)
        task_flips = int(np.count_nonzero(left_pred != right_pred))
        task_mae = float(np.mean(np.abs(left - right))) if left.size else math.nan
        rows.append(
            {
                "task": task_name,
                "samples": int(len(left)),
                "probability_mae": task_mae,
                "prediction_flips": task_flips,
            }
        )
        absolute_total += float(np.abs(left - right).sum())
        probability_count += int(left.size)
        flips += task_flips
        sample_count += int(len(left))
    return {
        "pooled_probability_mae": (
            absolute_total / probability_count if probability_count else math.nan
        ),
        "prediction_flips": flips,
        "task_sample_count": sample_count,
        "by_task": rows,
    }


def attention_total_variation(
    left: np.ndarray,
    right: np.ndarray,
    edge_index: np.ndarray,
    *,
    minimum_degree: int = 2,
) -> dict[str, float | int]:
    """Average TV between segmented sparse distributions.

    Arrays must have shape ``[sample, layer, head, edge]``. Only destinations
    with at least ``minimum_degree`` captured neighbours are included.
    """

    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    edge_index = np.asarray(edge_index, dtype=np.int64)
    if left.shape != right.shape or left.ndim != 4:
        raise ValueError("Attention arrays must have one identical [sample, layer, head, edge] shape.")
    if edge_index.shape != (2, left.shape[-1]):
        raise ValueError("edge_index is not aligned to the attention edge dimension.")
    destinations = edge_index[0]
    values: list[np.ndarray] = []
    eligible = 0
    for destination in np.unique(destinations):
        offsets = np.flatnonzero(destinations == destination)
        if len(offsets) < minimum_degree:
            continue
        eligible += 1
        values.append(0.5 * np.abs(left[..., offsets] - right[..., offsets]).sum(axis=-1))
    if not values:
        return {
            "eligible_destination_count": 0,
            "mean_total_variation": math.nan,
            "median_total_variation": math.nan,
            "max_total_variation": math.nan,
        }
    flattened = np.concatenate([value.reshape(-1) for value in values])
    return {
        "eligible_destination_count": eligible,
        "mean_total_variation": float(np.mean(flattened)),
        "median_total_variation": float(np.median(flattened)),
        "max_total_variation": float(np.max(flattened)),
    }


def patient_specific_attention_variation(
    attention: np.ndarray,
    edge_index: np.ndarray,
    *,
    minimum_degree: int = 2,
) -> dict[str, float | int]:
    attention = np.asarray(attention, dtype=np.float64)
    if attention.ndim != 4:
        raise ValueError("Attention must have shape [sample, layer, head, edge].")
    across_patient_mean = np.mean(attention, axis=0, keepdims=True)
    reference = np.broadcast_to(across_patient_mean, attention.shape)
    return attention_total_variation(
        attention,
        reference,
        edge_index,
        minimum_degree=minimum_degree,
    )


def _rank1_attention_audit(
    model: torch.nn.Module,
    rank1_state: dict[str, Any],
    dataset: Any,
    task_specs: Sequence[Any],
    saved_args: argparse.Namespace,
    device: torch.device,
    *,
    batch_size: int,
    max_batches: int | None,
) -> dict[str, Any]:
    model.load_state_dict(rank1_state, strict=True)
    full = collect_sparse_captures(
        model,
        dataset,
        task_specs,
        "val",
        batch_size=batch_size,
        device=device,
        max_batches=max_batches,
        mask_aware_heads=getattr(saved_args, "mask_aware_heads", "off") == "on",
    )
    full_diagnostics = getattr(model, "volumetric_diagnostics")()
    model.load_state_dict(rank1_state, strict=True)
    with torch.no_grad():
        for _, augmentation in model.iter_volumetric_augmentations():
            augmentation.beta.zero_()
    beta0 = collect_sparse_captures(
        model,
        dataset,
        task_specs,
        "val",
        batch_size=batch_size,
        device=device,
        max_batches=max_batches,
        mask_aware_heads=getattr(saved_args, "mask_aware_heads", "off") == "on",
    )
    if not np.array_equal(full["edge_index"], beta0["edge_index"]):
        raise ProtocolError("Rank-1 beta ablation changed sparse edge ordering.")
    return {
        "scope": "rank1_validation",
        "beta1_vs_beta0_inference": attention_total_variation(
            full["attention_weights"],
            beta0["attention_weights"],
            full["edge_index"],
        ),
        "patient_specific_variation": patient_specific_attention_variation(
            full["attention_weights"],
            full["edge_index"],
        ),
        "rank1_full_diagnostics": full_diagnostics,
    }


def _finite_run(payload: dict[str, Any]) -> bool:
    values = [
        payload["primary_validation_score"],
        payload["gate0_validation_score"],
        payload["beta0_inference_validation_score"],
        payload["gate0_difference"]["pooled_probability_mae"],
        payload["attention"]["beta1_vs_beta0_inference"]["mean_total_variation"],
    ]
    return all(math.isfinite(float(value)) for value in values)


def _validated_training_score(
    run_dir: Path,
    *,
    expected_variant: str,
    expected_beta: float | None = None,
    expected_tupe_mode: str = "on",
) -> tuple[float, str]:
    """Read a comparable validation score from a train-only run."""

    saved_args = namespace_from_saved_args(run_dir / "args.json", run_dir)
    if getattr(saved_args, "model_variant", None) != expected_variant:
        raise ProtocolError(
            f"Expected {expected_variant!r} in {run_dir}, found "
            f"{getattr(saved_args, 'model_variant', None)!r}."
        )
    if getattr(saved_args, "feature_selection_fit_scope", "train") != "train":
        raise ProtocolError(f"Validation audit refuses non-train feature selection in {run_dir}.")
    if getattr(saved_args, "scaler_fit_scope", "train") != "train":
        raise ProtocolError(f"Validation audit refuses non-train scaling in {run_dir}.")
    if getattr(saved_args, "evaluate_test", "off") != "off":
        raise ProtocolError(
            f"Validation audit requires --evaluate-test off in source run {run_dir}."
        )
    observed_tupe_mode = getattr(saved_args, "tupe_mode", "on")
    if observed_tupe_mode != expected_tupe_mode:
        raise ProtocolError(
            f"Expected tupe_mode={expected_tupe_mode!r} in {run_dir}, found "
            f"{observed_tupe_mode!r}."
        )
    if int(getattr(saved_args, "checkpoint_ensemble_size", 1)) != 3:
        raise ProtocolError(f"Validation audit requires requested ensemble K=3 in {run_dir}.")
    if expected_beta is not None and not math.isclose(
        float(getattr(saved_args, "volumetric_beta", math.nan)),
        float(expected_beta),
    ):
        raise ProtocolError(
            f"Expected beta={expected_beta:g} in {run_dir}, found "
            f"{getattr(saved_args, 'volumetric_beta', None)!r}."
        )
    if expected_variant == "ppi_volumetric":
        expected_modes = {
            "volumetric_volume_mode": "l2",
            "volumetric_message_mode": "expression_contrast",
            "volumetric_output_norm": "rms",
            "volumetric_gate_mode": "per_head",
            "volumetric_backbone_gradient_mode": "detached",
        }
        mismatches = {
            key: getattr(saved_args, key, None)
            for key, expected in expected_modes.items()
            if getattr(saved_args, key, None) != expected
        }
        if mismatches:
            raise ProtocolError(
                f"Control run is not the preregistered VMA v2 profile in {run_dir}: "
                f"{mismatches}."
            )
    checkpoints = ensemble_checkpoint_paths(run_dir)
    if len(checkpoints) != 3:
        raise ProtocolError(
            f"Validation audit requires effective ensemble K=3 in {run_dir}; "
            f"found K={len(checkpoints)}."
        )
    summary_path = run_dir / "model_summary.json"
    summary = read_json(summary_path) if summary_path.exists() else {}
    training = summary.get("training_summary", {}) if isinstance(summary, dict) else {}
    if len(checkpoints) > 1 and (
        not isinstance(training, dict)
        or training.get("ensemble_validation_score") is None
    ):
        raise ProtocolError(
            f"Cannot compare {run_dir}: it has a {len(checkpoints)}-member ensemble but no "
            "ensemble_validation_score. Re-evaluate validation with the current worker."
        )
    return validation_checkpoint_score(run_dir)


def audit_seed(
    result_root: Path,
    seed: int,
    primary_beta: float,
    control_beta: float,
    device: torch.device,
    *,
    batch_size: int,
    max_batches: int | None,
    tupe_mode: str = "on",
) -> dict[str, Any]:
    primary_dir = candidate_dir(result_root, seed, primary_beta)
    control_dir = candidate_dir(result_root, seed, control_beta)
    base_dir = baseline_dir(result_root, seed)
    checkpoints = ensemble_checkpoint_paths(primary_dir)
    saved_args, dataset, task_specs, _, model = reconstruct_model(
        primary_dir,
        checkpoints[0],
        device,
    )
    if getattr(saved_args, "feature_selection_fit_scope", "train") != "train":
        raise ProtocolError("Validation audit refuses feature selection fitted outside train.")
    if getattr(saved_args, "scaler_fit_scope", "train") != "train":
        raise ProtocolError("Validation audit refuses scaling fitted outside train.")
    if getattr(saved_args, "evaluate_test", "off") != "off":
        raise ProtocolError(
            "Validation audit requires a candidate trained with --evaluate-test off."
        )
    expected_v2 = {
        "volumetric_beta": float(primary_beta),
        "volumetric_volume_mode": "l2",
        "volumetric_message_mode": "expression_contrast",
        "volumetric_output_norm": "rms",
        "volumetric_gate_mode": "per_head",
        "volumetric_backbone_gradient_mode": "detached",
        "volumetric_gate_init": 0.0,
        "volumetric_dropout": 0.0,
        "lr_volumetric": 5e-4,
        "checkpoint_ensemble_size": 3,
        "checkpoint_ensemble_min_gap": 3,
        "tupe_mode": tupe_mode,
    }
    for key, expected in expected_v2.items():
        observed = getattr(saved_args, key, None)
        if isinstance(expected, float):
            matches = observed is not None and math.isclose(
                float(observed), expected, rel_tol=1e-9, abs_tol=1e-12
            )
        else:
            matches = observed == expected
        if not matches:
            raise ProtocolError(
                f"Primary run is not the preregistered VMA v2 profile: "
                f"{key}={observed!r}, expected {expected!r}."
            )
    if len(checkpoints) != 3:
        raise ProtocolError(
            f"Primary run requires effective ensemble K=3; found K={len(checkpoints)}."
        )
    states = _load_states(model, checkpoints, device)
    gate0_states = _ablated_states(model, states, ablation="gate0")
    beta0_states = _ablated_states(model, states, ablation="beta0")

    full_results, full_loss, full_score, full_details = _evaluate_states(
        model,
        states,
        dataset,
        task_specs,
        saved_args,
        device,
        batch_size=batch_size,
        max_batches=max_batches,
    )
    gate0_results, gate0_loss, gate0_score, _ = _evaluate_states(
        model,
        gate0_states,
        dataset,
        task_specs,
        saved_args,
        device,
        batch_size=batch_size,
        max_batches=max_batches,
    )
    beta0_results, beta0_loss, beta0_score, _ = _evaluate_states(
        model,
        beta0_states,
        dataset,
        task_specs,
        saved_args,
        device,
        batch_size=batch_size,
        max_batches=max_batches,
    )
    baseline_score, baseline_score_source = _validated_training_score(
        base_dir,
        expected_variant="baseline",
        expected_tupe_mode=tupe_mode,
    )
    trained_control_score, control_score_source = _validated_training_score(
        control_dir,
        expected_variant="ppi_volumetric",
        expected_beta=control_beta,
        expected_tupe_mode=tupe_mode,
    )
    stored_primary_score, primary_score_source = validation_checkpoint_score(primary_dir)
    score_recalculation_delta = float(full_score - stored_primary_score)
    if not math.isclose(full_score, stored_primary_score, rel_tol=1e-6, abs_tol=1e-6):
        raise ProtocolError(
            "Recomputed primary ensemble validation score does not match its stored score: "
            f"recomputed={full_score:.12g}, stored={stored_primary_score:.12g}, "
            f"source={primary_score_source}."
        )
    model.load_state_dict(states[0], strict=True)
    gates = getattr(model, "volumetric_gate_values")()
    max_abs_gate = max((abs(float(value)) for value in gates.values()), default=0.0)
    attention = _rank1_attention_audit(
        model,
        states[0],
        dataset,
        task_specs,
        saved_args,
        device,
        batch_size=batch_size,
        max_batches=max_batches,
    )
    payload: dict[str, Any] = {
        "seed": int(seed),
        "split": "val",
        "test_evaluated": False,
        "decision_data": "validation_only",
        "primary_beta": float(primary_beta),
        "control_beta": float(control_beta),
        "primary_run_dir": str(primary_dir),
        "ensemble": {
            "size": len(checkpoints),
            "aggregation": ENSEMBLE_AGGREGATION,
            "checkpoints": [str(path) for path in checkpoints],
        },
        "primary_validation_loss": full_loss,
        "primary_validation_score": full_score,
        "stored_primary_validation_score": float(stored_primary_score),
        "primary_score_recalculation_delta": score_recalculation_delta,
        "primary_score_source": primary_score_source,
        "primary_checkpoint_details": full_details,
        "baseline_validation_score": float(baseline_score),
        "baseline_score_source": baseline_score_source,
        "trained_beta0_validation_score": float(trained_control_score),
        "trained_beta0_score_source": control_score_source,
        "delta_primary_minus_baseline": float(full_score - baseline_score),
        "delta_primary_minus_trained_beta0": float(full_score - trained_control_score),
        "gate0_validation_loss": gate0_loss,
        "gate0_validation_score": gate0_score,
        "gate0_difference": probability_difference(full_results, gate0_results),
        "beta0_inference_validation_loss": beta0_loss,
        "beta0_inference_validation_score": beta0_score,
        "beta0_inference_difference": probability_difference(full_results, beta0_results),
        "gate_values": gates,
        "max_abs_effective_gate": max_abs_gate,
        "attention": attention,
    }
    payload["checks"] = {
        "finite": _finite_run(payload),
        "gate_magnitude_gt_1e_3": max_abs_gate > GATE_MAGNITUDE_THRESHOLD,
        "gate0_probability_mae_gt_1e_4": (
            payload["gate0_difference"]["pooled_probability_mae"] > GATE_MAE_THRESHOLD
        ),
        "beta_attention_tv_gt_1_percent": (
            attention["beta1_vs_beta0_inference"]["mean_total_variation"]
            > BETA_ATTENTION_TV_THRESHOLD
        ),
        "primary_beats_baseline": payload["delta_primary_minus_baseline"] > 0,
        "primary_beats_trained_beta0": payload["delta_primary_minus_trained_beta0"] > 0,
    }
    return payload


def summarize_pilot(seed_payloads: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not seed_payloads:
        raise ValueError("At least one seed audit is required.")
    count = len(seed_payloads)
    required_positive = math.ceil(2 * count / 3)
    baseline_deltas = np.asarray(
        [item["delta_primary_minus_baseline"] for item in seed_payloads], dtype=np.float64
    )
    beta0_deltas = np.asarray(
        [item["delta_primary_minus_trained_beta0"] for item in seed_payloads], dtype=np.float64
    )
    mechanical = {
        "all_finite": all(item["checks"]["finite"] for item in seed_payloads),
        "all_gate_magnitude_gt_1e_3": all(
            item["checks"]["gate_magnitude_gt_1e_3"] for item in seed_payloads
        ),
        "all_gate0_probability_mae_gt_1e_4": all(
            item["checks"]["gate0_probability_mae_gt_1e_4"] for item in seed_payloads
        ),
        "all_beta_attention_tv_gt_1_percent": all(
            item["checks"]["beta_attention_tv_gt_1_percent"] for item in seed_payloads
        ),
    }
    predictive = {
        "required_positive_seed_count": required_positive,
        "primary_minus_baseline_mean": float(np.mean(baseline_deltas)),
        "primary_minus_baseline_positive_seed_count": int(np.sum(baseline_deltas > 0)),
        "primary_minus_beta0_mean": float(np.mean(beta0_deltas)),
        "primary_minus_beta0_positive_seed_count": int(np.sum(beta0_deltas > 0)),
    }
    predictive["passes"] = bool(
        predictive["primary_minus_baseline_mean"] > 0
        and predictive["primary_minus_baseline_positive_seed_count"] >= required_positive
        and predictive["primary_minus_beta0_mean"] > 0
        and predictive["primary_minus_beta0_positive_seed_count"] >= required_positive
    )
    is_preregistered_pilot = count == 3
    return {
        "split": "val",
        "test_evaluated": False,
        "decision_data": "validation_only",
        "audit_scope": "preregistered_pilot" if is_preregistered_pilot else "multi_seed_diagnostic",
        "seed_count": count,
        "mechanical": mechanical,
        "predictive": predictive,
        # The original preregistered decision rule applies only to a three-seed
        # pilot.  Larger audits report the same transparent 2/3 diagnostic but
        # must not relabel it as the preregistered pilot decision.
        "passes_diagnostic_criteria": bool(all(mechanical.values()) and predictive["passes"]),
        "passes_preregistered_pilot_criteria": (
            bool(all(mechanical.values()) and predictive["passes"])
            if is_preregistered_pilot
            else None
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("--seeds must be non-empty and unique.")
    if not math.isfinite(args.primary_beta) or not math.isfinite(args.control_beta):
        raise ValueError("Primary/control beta must be finite.")
    if not math.isclose(args.primary_beta, 1.0) or not math.isclose(args.control_beta, 0.0):
        raise ValueError(
            "This preregistered audit requires --primary-beta 1 and --control-beta 0."
        )
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    result_root = resolve_path(args.result_root)
    output_dir = (
        resolve_path(args.output_dir)
        if args.output_dir is not None
        else result_root / "validation_audit"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    seed_payloads: list[dict[str, Any]] = []
    for seed in args.seeds:
        payload = audit_seed(
            result_root,
            int(seed),
            float(args.primary_beta),
            float(args.control_beta),
            device,
            batch_size=args.batch_size,
            max_batches=None,
            tupe_mode=args.tupe_mode,
        )
        seed_payloads.append(payload)
        write_json(output_dir / f"seed_{int(seed)}_validation_audit.json", payload)

    summary = summarize_pilot(seed_payloads)
    summary.update(
        {
            "result_root": str(result_root),
            "seeds": [int(seed) for seed in args.seeds],
            "primary_beta": float(args.primary_beta),
            "control_beta": float(args.control_beta),
            "tupe_mode": args.tupe_mode,
            "thresholds": {
                "gate_magnitude": GATE_MAGNITUDE_THRESHOLD,
                "gate0_probability_mae": GATE_MAE_THRESHOLD,
                "beta_attention_total_variation": BETA_ATTENTION_TV_THRESHOLD,
            },
        }
    )
    write_json(output_dir / "validation_audit_summary.json", summary)
    rows = []
    for item in seed_payloads:
        rows.append(
            {
                "seed": item["seed"],
                "delta_primary_minus_baseline": item["delta_primary_minus_baseline"],
                "delta_primary_minus_trained_beta0": item[
                    "delta_primary_minus_trained_beta0"
                ],
                "max_abs_effective_gate": item["max_abs_effective_gate"],
                "gate0_probability_mae": item["gate0_difference"][
                    "pooled_probability_mae"
                ],
                "beta_attention_mean_total_variation": item["attention"][
                    "beta1_vs_beta0_inference"
                ]["mean_total_variation"],
                "patient_specific_attention_mean_total_variation": item["attention"][
                    "patient_specific_variation"
                ]["mean_total_variation"],
                **{f"check_{key}": value for key, value in item["checks"].items()},
            }
        )
    pd.DataFrame(rows).to_csv(output_dir / "validation_audit_by_seed.csv", index=False)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Validation-only VMA audit complete: {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
