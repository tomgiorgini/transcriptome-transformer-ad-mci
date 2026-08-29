from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd


def write_dgs_inputs(x: pd.DataFrame, y: pd.Series, artifact_dir: Path) -> tuple[Path, Path]:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    x_file = artifact_dir / "dgs_input_expression.csv"
    y_file = artifact_dir / "dgs_labels.csv"
    x.to_csv(x_file)
    pd.DataFrame({"label": y.astype(int)}, index=y.index).to_csv(y_file)
    return x_file, y_file


def run_limma_dgs(
    r_script: Path,
    x_file: Path,
    y_file: Path,
    out_dir: Path,
    adj_p_threshold: float,
    p_value_threshold: float | None = None,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "Rscript",
        str(r_script),
        "--x-file",
        str(x_file),
        "--y-file",
        str(y_file),
        "--out-dir",
        str(out_dir),
        "--adj-p-threshold",
        str(adj_p_threshold),
    ]
    if p_value_threshold is not None:
        cmd.extend(["--p-value-threshold", str(p_value_threshold)])
    completed = subprocess.run(cmd, check=True, capture_output=True, text=True)
    selected_file = out_dir / "selected_genes.txt"
    table_file = out_dir / "dgs_table.csv"
    manifest_file = out_dir / "dgs_manifest.csv"
    if not selected_file.exists() or not table_file.exists() or not manifest_file.exists():
        raise FileNotFoundError("limma DGS did not produce all expected outputs.")
    selected = [line.strip() for line in selected_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    manifest = pd.read_csv(manifest_file).iloc[0].to_dict()
    manifest.update(
        {
            "selected_genes_file": str(selected_file),
            "dgs_table_file": str(table_file),
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
    )
    if not selected:
        raise ValueError("DGS selected zero genes with the configured adj.P.Value threshold.")
    return manifest


def load_selected_genes(selected_file: Path, available_genes: pd.Index) -> list[str]:
    genes = [line.strip() for line in selected_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    available = set(available_genes.astype(str))
    missing = [gene for gene in genes if gene not in available]
    if missing:
        preview = ", ".join(missing[:10])
        raise ValueError(f"{len(missing)} selected genes are absent from X. Examples: {preview}")
    if not genes:
        raise ValueError("No selected genes found.")
    return genes


def write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
