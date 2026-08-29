from __future__ import annotations

import json

import numpy as np

from experiments.scripts.paper_comparison.txt_volumetric.audit_validation import (
    attention_total_variation,
    build_parser,
    ensemble_checkpoint_paths,
    patient_specific_attention_variation,
    summarize_pilot,
)


def test_attention_total_variation_is_segmented_and_ignores_singletons() -> None:
    edge_index = np.asarray([[0, 0, 1], [1, 2, 0]], dtype=np.int64)
    left = np.asarray([[[[1.0, 0.0, 1.0]]]])
    right = np.asarray([[[[0.0, 1.0, 0.0]]]])

    result = attention_total_variation(left, right, edge_index)

    assert result["eligible_destination_count"] == 1
    assert result["mean_total_variation"] == 1.0
    assert result["median_total_variation"] == 1.0


def test_patient_specific_attention_variation_uses_patient_mean() -> None:
    edge_index = np.asarray([[0, 0], [1, 2]], dtype=np.int64)
    attention = np.asarray(
        [
            [[[1.0, 0.0]]],
            [[[0.0, 1.0]]],
        ]
    )

    result = patient_specific_attention_variation(attention, edge_index)

    assert result["eligible_destination_count"] == 1
    assert result["mean_total_variation"] == 0.5


def test_ensemble_checkpoint_paths_honours_effective_size(tmp_path) -> None:
    for rank in (1, 2, 3, 4):
        (tmp_path / f"ensemble_checkpoint_rank{rank}.pt").write_bytes(b"checkpoint")
    (tmp_path / "model_summary.json").write_text(
        json.dumps(
            {
                "training_summary": {
                    "checkpoint_ensemble_effective_k": 3,
                }
            }
        ),
        encoding="utf-8",
    )

    paths = ensemble_checkpoint_paths(tmp_path)

    assert [path.name for path in paths] == [
        "ensemble_checkpoint_rank1.pt",
        "ensemble_checkpoint_rank2.pt",
        "ensemble_checkpoint_rank3.pt",
    ]


def test_pilot_summary_applies_preregistered_two_of_three_rule() -> None:
    payloads = []
    for seed, baseline_delta, beta0_delta in (
        (201, 0.02, 0.01),
        (202, 0.01, 0.02),
        (203, -0.005, -0.002),
    ):
        payloads.append(
            {
                "seed": seed,
                "delta_primary_minus_baseline": baseline_delta,
                "delta_primary_minus_trained_beta0": beta0_delta,
                "checks": {
                    "finite": True,
                    "gate_magnitude_gt_1e_3": True,
                    "gate0_probability_mae_gt_1e_4": True,
                    "beta_attention_tv_gt_1_percent": True,
                },
            }
        )

    summary = summarize_pilot(payloads)

    assert summary["predictive"]["required_positive_seed_count"] == 2
    assert summary["predictive"]["passes"] is True
    assert summary["passes_preregistered_pilot_criteria"] is True
    assert summary["passes_diagnostic_criteria"] is True
    assert summary["audit_scope"] == "preregistered_pilot"


def test_ten_seed_summary_is_diagnostic_not_preregistered_pilot() -> None:
    payloads = []
    for seed in range(101, 111):
        payloads.append(
            {
                "seed": seed,
                "delta_primary_minus_baseline": 0.01,
                "delta_primary_minus_trained_beta0": 0.01,
                "checks": {
                    "finite": True,
                    "gate_magnitude_gt_1e_3": True,
                    "gate0_probability_mae_gt_1e_4": True,
                    "beta_attention_tv_gt_1_percent": True,
                },
            }
        )

    summary = summarize_pilot(payloads)

    assert summary["seed_count"] == 10
    assert summary["predictive"]["required_positive_seed_count"] == 7
    assert summary["passes_diagnostic_criteria"] is True
    assert summary["passes_preregistered_pilot_criteria"] is None
    assert summary["audit_scope"] == "multi_seed_diagnostic"


def test_validation_audit_cli_exposes_no_split_selector() -> None:
    option_strings = {
        option
        for action in build_parser()._actions
        for option in action.option_strings
    }
    assert "--split" not in option_strings
