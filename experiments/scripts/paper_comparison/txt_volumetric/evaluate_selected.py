#!/usr/bin/env python3
"""Evaluate only validation-selected PPI-volumetric checkpoints on the held-out test split."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.scripts.paper_comparison.txt_volumetric.common import (  # noqa: E402
    DEFAULT_RESULT_ROOT,
    DEFAULT_WORKER,
    PROTOCOL_SEEDS,
    evaluation_command_from_selection,
    metrics_has_split,
    read_json,
    resolve_path,
    run_worker,
    save_worker_command,
    selected_beta_path,
    selected_test_dir,
    write_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run evaluation-only from each selected_beta.json. The saved candidate command is reused "
            "to reconstruct exactly the selected model and preprocessing configuration."
        )
    )
    parser.add_argument("--python-exe", default=sys.executable)
    parser.add_argument("--worker", type=Path, default=DEFAULT_WORKER)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(PROTOCOL_SEEDS))
    parser.add_argument("--device", choices=["cuda", "cpu", "mps"], default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result_root = resolve_path(args.result_root)
    for seed in args.seeds:
        selection_file = selected_beta_path(result_root, seed)
        if not selection_file.exists():
            raise FileNotFoundError(
                f"Missing {selection_file}; run select_beta.py before test evaluation."
            )
        selection = read_json(selection_file)
        evaluation_dir = selected_test_dir(result_root, seed)
        command = evaluation_command_from_selection(
            selection,
            evaluation_dir,
            python_exe=args.python_exe,
            worker=resolve_path(args.worker),
            device=args.device,
        )
        save_worker_command(
            evaluation_dir,
            command,
            kind="selected_checkpoint_test_evaluation",
            seed=seed,
            beta=float(selection["selected_beta"]),
        )
        if not (args.skip_existing and metrics_has_split(evaluation_dir / "metrics_summary.csv", "test")):
            run_worker(command, evaluation_dir, dry_run=args.dry_run)
        selection.update(
            {
                "evaluation_dir": str(evaluation_dir),
                "evaluation_command_file": str(evaluation_dir / "worker_command.json"),
            }
        )
        write_json(selection_file, selection)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
