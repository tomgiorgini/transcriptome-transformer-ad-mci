#!/usr/bin/env python3
"""Select beta from validation-only candidate artifacts; never reads test metrics."""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.scripts.paper_comparison.txt_volumetric.common import (  # noqa: E402
    BETA_CANDIDATES,
    DEFAULT_RESULT_ROOT,
    PROTOCOL_SEEDS,
    choose_beta,
    resolve_path,
    select_beta_for_seed,
    write_beta_selection,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Select one PPI-volumetric beta per seed using the saved validation checkpoint score. "
            "Exact ties select the smaller beta. This command does not inspect or evaluate test data."
        )
    )
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(PROTOCOL_SEEDS))
    parser.add_argument("--betas", nargs="+", type=float, default=list(BETA_CANDIDATES))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result_root = resolve_path(args.result_root)
    rows = []
    for seed in args.seeds:
        selection = select_beta_for_seed(result_root, seed, args.betas)
        path = write_beta_selection(result_root, selection)
        row = asdict(selection)
        row.pop("candidates", None)
        row["selection_file"] = str(path)
        rows.append(row)
        print(f"seed={seed}: beta={selection.selected_beta:g}, validation_score={selection.selected_score:.8g}")
    pd.DataFrame(rows).to_csv(result_root / "beta_selections.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
