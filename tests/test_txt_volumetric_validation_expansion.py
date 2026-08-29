from __future__ import annotations

from experiments.scripts.paper_comparison.txt_volumetric.summarize_validation_expansion import (
    build_parser,
    summarize_rows,
)


def test_expansion_requires_seven_positive_seeds_and_positive_mean() -> None:
    rows = [
        {"delta_primary_minus_baseline": value}
        for value in (0.02, 0.01, 0.01, 0.01, 0.005, 0.005, 0.001, -0.001, -0.002, -0.003)
    ]

    summary = summarize_rows(rows)

    assert summary["positive_seed_count"] == 7
    assert summary["mean_delta_primary_minus_baseline"] > 0
    assert summary["passes_preregistered_expansion_criteria"] is True


def test_expansion_fails_with_only_six_positive_seeds() -> None:
    rows = [
        {"delta_primary_minus_baseline": value}
        for value in (0.02, 0.01, 0.01, 0.01, 0.005, 0.005, -0.001, -0.001, -0.001, -0.001)
    ]

    summary = summarize_rows(rows)

    assert summary["positive_seed_count"] == 6
    assert summary["passes_preregistered_expansion_criteria"] is False


def test_expansion_cli_has_no_split_or_test_option() -> None:
    options = {
        option
        for action in build_parser()._actions
        for option in action.option_strings
    }
    assert "--split" not in options
    assert all("test" not in option for option in options)
    assert "--tupe-mode" in options


def test_expansion_cli_accepts_no_tupe() -> None:
    args = build_parser().parse_args(
        [
            "--result-root",
            "results/example",
            "--seeds",
            *[str(seed) for seed in range(101, 111)],
            "--tupe-mode",
            "off",
        ]
    )

    assert args.tupe_mode == "off"
