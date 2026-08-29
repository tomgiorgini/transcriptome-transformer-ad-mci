#!/usr/bin/env python3
"""Summarize the preregistered ten-seed VMA v2 validation expansion."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.scripts.paper_comparison.txt_volumetric.audit_validation import (  # noqa: E402
    _validated_training_score,
)
from experiments.scripts.paper_comparison.txt_volumetric.common import (  # noqa: E402
    baseline_dir,
    candidate_dir,
    resolve_path,
    write_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize beta=1 versus baseline on exactly ten validation-only seeds. "
            "This command has no test split or evaluation option."
        )
    )
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--primary-beta", type=float, default=1.0)
    parser.add_argument("--tupe-mode", choices=["on", "off"], default="on")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser


def summarize_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if len(rows) != 10:
        raise ValueError("The preregistered expansion requires exactly ten seed rows.")
    deltas = np.asarray([float(row["delta_primary_minus_baseline"]) for row in rows])
    if not np.isfinite(deltas).all():
        raise ValueError("All validation deltas must be finite.")
    mean_delta = float(np.mean(deltas))
    positive_count = int(np.sum(deltas > 0))
    return {
        "split": "val",
        "decision_data": "validation_only",
        "test_evaluated": False,
        "seed_count": 10,
        "required_positive_seed_count": 7,
        "positive_seed_count": positive_count,
        "mean_delta_primary_minus_baseline": mean_delta,
        "passes_preregistered_expansion_criteria": bool(
            positive_count >= 7 and mean_delta > 0
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if len(args.seeds) != 10 or len(set(args.seeds)) != 10:
        raise ValueError("--seeds must contain exactly ten unique seeds.")
    if not math.isclose(float(args.primary_beta), 1.0):
        raise ValueError("The preregistered expansion requires --primary-beta 1.")
    result_root = resolve_path(args.result_root)
    output_dir = (
        resolve_path(args.output_dir)
        if args.output_dir is not None
        else result_root / "validation_expansion_summary"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for seed in args.seeds:
        base_dir = baseline_dir(result_root, int(seed))
        primary_dir = candidate_dir(result_root, int(seed), float(args.primary_beta))
        baseline_score, baseline_source = _validated_training_score(
            base_dir,
            expected_variant="baseline",
            expected_tupe_mode=args.tupe_mode,
        )
        primary_score, primary_source = _validated_training_score(
            primary_dir,
            expected_variant="ppi_volumetric",
            expected_beta=float(args.primary_beta),
            expected_tupe_mode=args.tupe_mode,
        )
        rows.append(
            {
                "seed": int(seed),
                "baseline_validation_score": float(baseline_score),
                "primary_validation_score": float(primary_score),
                "delta_primary_minus_baseline": float(primary_score - baseline_score),
                "baseline_score_source": baseline_source,
                "primary_score_source": primary_source,
                "test_evaluated": False,
            }
        )

    summary = summarize_rows(rows)
    summary["passes_expansion_threshold"] = summary[
        "passes_preregistered_expansion_criteria"
    ]
    if args.tupe_mode == "off":
        # The no-TUPE experiment is an exploratory ablation, not the original
        # preregistered TUPE-on expansion. Preserve the numerical threshold but
        # do not mislabel it as a preregistered decision.
        summary["passes_preregistered_expansion_criteria"] = None
        summary["analysis_scope"] = "exploratory_no_tupe_ablation"
    else:
        summary["analysis_scope"] = "preregistered_tupe_on_expansion"
    summary.update(
        {
            "result_root": str(result_root),
            "seeds": [int(seed) for seed in args.seeds],
            "primary_beta": 1.0,
            "tupe_mode": args.tupe_mode,
        }
    )
    pd.DataFrame(rows).to_csv(
        output_dir / "validation_expansion_by_seed.csv",
        index=False,
    )
    write_json(output_dir / "validation_expansion_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Validation-only expansion summary complete: {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
