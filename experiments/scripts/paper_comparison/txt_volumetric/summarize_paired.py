#!/usr/bin/env python3
"""Create seed-level paired deltas, bootstrap summaries, and beta frequencies."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.scripts.paper_comparison.txt_volumetric.common import (  # noqa: E402
    BETA_CANDIDATES,
    DEFAULT_RESULT_ROOT,
    PROTOCOL_SEEDS,
    resolve_path,
    write_summary_artifacts,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare selected PPI-volumetric test metrics against the paired baseline for each seed/task/metric."
        )
    )
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(PROTOCOL_SEEDS))
    parser.add_argument("--betas", nargs="+", type=float, default=list(BETA_CANDIDATES))
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260718)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = write_summary_artifacts(
        resolve_path(args.result_root),
        args.seeds,
        betas=args.betas,
        replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
    )
    print(json.dumps(paths, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
