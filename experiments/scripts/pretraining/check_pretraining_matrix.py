#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from source.pretraining.txt.data import global_zscore_report, load_pretraining_matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a TxT pretraining matrix before a long GPU run.")
    parser.add_argument("--matrix-file", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument(
        "--max-abs-mean",
        type=float,
        default=1e-3,
        help="Maximum accepted absolute per-gene mean after global z-score.",
    )
    parser.add_argument(
        "--std-tolerance",
        type=float,
        default=1e-3,
        help="Maximum accepted distance of min/max gene std from 1 after global z-score.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    matrix_path = args.matrix_file.resolve()
    if not matrix_path.exists():
        raise FileNotFoundError(f"Matrix file not found: {matrix_path}")

    matrix = load_pretraining_matrix(matrix_path)
    report = global_zscore_report(matrix.values)
    payload = {
        "matrix_file": str(matrix_path),
        "samples": int(matrix.values.shape[0]),
        "genes": int(matrix.values.shape[1]),
        "sample_id_unique": int(len(set(matrix.sample_ids.tolist()))),
        "gene_name_unique": int(len(set(matrix.gene_names))),
        "zscore_report": report,
    }

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Matrix file: {matrix_path}")
    print(f"Samples: {payload['samples']}")
    print(f"Genes: {payload['genes']}")
    print(f"Unique sample_id: {payload['sample_id_unique']}")
    print(f"Unique genes: {payload['gene_name_unique']}")
    print(
        "Z-score: "
        f"max_abs_gene_mean={report['max_abs_gene_mean']:.8g}, "
        f"mean_gene_std={report['mean_gene_std']:.8g}, "
        f"min_gene_std={report['min_gene_std']:.8g}, "
        f"max_gene_std={report['max_gene_std']:.8g}"
    )

    if report["max_abs_gene_mean"] > args.max_abs_mean:
        raise SystemExit(
            f"Matrix z-score check failed: max_abs_gene_mean={report['max_abs_gene_mean']:.8g} "
            f"> {args.max_abs_mean:.8g}"
        )
    min_std_distance = abs(report["min_gene_std"] - 1.0)
    max_std_distance = abs(report["max_gene_std"] - 1.0)
    if min_std_distance > args.std_tolerance or max_std_distance > args.std_tolerance:
        raise SystemExit(
            "Matrix z-score check failed: gene std is outside tolerance "
            f"(min distance={min_std_distance:.8g}, max distance={max_std_distance:.8g}, "
            f"tolerance={args.std_tolerance:.8g})"
        )

    print("Pretraining matrix check: PASS")


if __name__ == "__main__":
    main()
