#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.scripts.paper_comparison import run_txt_multitask_final_grid as final_grid


DATASET_BUILDER = ROOT / "experiments" / "scripts" / "paper_comparison" / "build_txt_pairwise_datasets.py"
PPI_BUILDER = ROOT / "experiments" / "scripts" / "pretraining" / "build_ppi_embedding.py"
GRID_RUNNER = ROOT / "experiments" / "scripts" / "paper_comparison" / "run_txt_multitask_final_grid.py"

DEFAULT_SOURCE_DIR = ROOT / "task_dataset" / "processed" / "alzheimer_multiclass"
DEFAULT_DATASET_OUTPUT_DIR = ROOT / "task_dataset" / "processed" / "txt_pairwise_multitask"
DEFAULT_PPI_RESULT_ROOT = ROOT / "results" / "pretraining" / "ppi_init"
DEFAULT_GRID_RESULT_ROOT = ROOT / "results" / "paper_comparison" / "txt_multitask_final_grid"


@dataclass(frozen=True)
class PipelineStep:
    name: str
    command: list[str]
    log_file: Path


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=(
            "Run the final TxT multitask thesis pipeline: build shared AD/MCI/CTL datasets, "
            "build HIPPIE/PPI node2vec initializations, then launch the final multitask grid. "
            "Unknown arguments are passed through to run_txt_multitask_final_grid.py."
        )
    )
    parser.add_argument("--python-exe", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument("--source-x-file", type=Path, default=DEFAULT_SOURCE_DIR / "X.csv")
    parser.add_argument("--source-y-file", type=Path, default=DEFAULT_SOURCE_DIR / "y.csv")
    parser.add_argument("--source-split-file", type=Path, default=DEFAULT_SOURCE_DIR / "splits" / "official_seed42.csv")
    parser.add_argument("--dataset-output-dir", type=Path, default=DEFAULT_DATASET_OUTPUT_DIR)
    parser.add_argument(
        "--shared-dataset-dir",
        type=Path,
        default=None,
        help="Directory containing the final shared X.csv/y.csv. Defaults to dataset-output-dir/shared_ad_mci_ctl.",
    )
    parser.add_argument("--skip-dataset-build", action="store_true")

    parser.add_argument("--skip-ppi-build", action="store_true")
    parser.add_argument("--skip-existing-ppi", action="store_true")
    parser.add_argument("--hippie-file", type=Path, default=None)
    parser.add_argument("--ppi-result-root", type=Path, default=DEFAULT_PPI_RESULT_ROOT)
    parser.add_argument("--ppi-score-threshold", type=float, default=0.73)
    parser.add_argument("--ppi-seed", type=int, default=42)
    parser.add_argument("--ppi-dims", nargs="+", type=int, default=None)
    parser.add_argument("--ppi-component-policy", choices=["largest", "all"], default="largest")
    parser.add_argument("--ppi-device", choices=["cuda", "cpu", "mps"], default="cuda")
    parser.add_argument("--ppi-epochs", type=int, default=20)
    parser.add_argument("--ppi-batch-size", type=int, default=1024)
    parser.add_argument("--ppi-lr", type=float, default=0.01)
    parser.add_argument("--ppi-walk-length", type=int, default=20)
    parser.add_argument("--ppi-context-size", type=int, default=10)
    parser.add_argument("--ppi-walks-per-node", type=int, default=10)
    parser.add_argument("--ppi-negative-samples", type=int, default=5)
    parser.add_argument("--ppi-max-pairs-per-epoch", type=int, default=1_000_000)
    parser.add_argument("--ppi-skip-node2vec", action="store_true")

    parser.add_argument("--grid-result-root", type=Path, default=DEFAULT_GRID_RESULT_ROOT)
    parser.add_argument("--preset", choices=["pilot", "balanced", "wide", "random_sanity", "custom"], default="balanced")
    parser.add_argument("--run-mode", choices=["cv", "seeds", "both"], default="cv")
    parser.add_argument("--device", choices=["cuda", "cpu", "mps"], default="cuda")
    parser.add_argument("--architectures", nargs="+", default=["1l2h"])
    parser.add_argument("--embedding-sources", nargs="+", choices=["random", "ppi"], default=None)
    parser.add_argument("--d-models", nargs="+", type=int, default=None)
    parser.add_argument("--dropouts", nargs="+", type=float, default=None)
    parser.add_argument("--max-genes-list", nargs="+", type=int, default=None)
    parser.add_argument(
        "--gene-selections",
        nargs="+",
        choices=["mad", "variance", "pairwise_anova_union", "ad_mci_priority_anova_union", "ad_mci_vs_ctl_anova_50_50"],
        default=None,
    )
    parser.add_argument("--ad-mci-gene-fractions", nargs="+", type=float, default=None)
    parser.add_argument("--augmentations", nargs="+", choices=["none", "smote", "borderline_smote", "ctgan"], default=None)
    parser.add_argument("--ppi-gene-policy", choices=["all", "mapped_only"], default="mapped_only")
    parser.add_argument("--grid-skip-existing", action="store_true")
    parser.add_argument("--grid-skip-missing-ppi", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_known_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else (ROOT / path).resolve()


def score_tag(score_threshold: float) -> str:
    return f"{score_threshold:g}".replace(".", "p")


def dedupe_preserving_order(values: list[int]) -> list[int]:
    seen: set[int] = set()
    output: list[int] = []
    for value in values:
        if value <= 0:
            raise ValueError(f"Embedding/model dimensions must be positive, got {value}.")
        if value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output


def shared_dataset_dir(args: argparse.Namespace) -> Path:
    if args.shared_dataset_dir is not None:
        return resolve(args.shared_dataset_dir)
    return resolve(args.dataset_output_dir) / "shared_ad_mci_ctl"


def shared_x_file(args: argparse.Namespace) -> Path:
    return shared_dataset_dir(args) / "X.csv"


def shared_y_file(args: argparse.Namespace) -> Path:
    return shared_dataset_dir(args) / "y.csv"


def grid_uses_ppi(args: argparse.Namespace) -> bool:
    if args.embedding_sources is not None:
        return "ppi" in args.embedding_sources
    if args.preset == "custom":
        return True
    return "ppi" in final_grid.GRID_PRESETS[args.preset]["embedding_sources"]


def infer_ppi_dims(args: argparse.Namespace) -> list[int]:
    if args.ppi_dims is not None:
        return dedupe_preserving_order(args.ppi_dims)
    if args.d_models is not None:
        return dedupe_preserving_order(args.d_models)
    if args.preset != "custom":
        return dedupe_preserving_order([int(value) for value in final_grid.GRID_PRESETS[args.preset]["d_models"]])
    raise ValueError("--preset custom requires --ppi-dims or --d-models so PPI initialization dimensions are explicit.")


def ppi_result_dir(args: argparse.Namespace, dim: int) -> Path:
    run_name = f"hippie_highconf_dim{dim}_score{score_tag(args.ppi_score_threshold)}_seed{args.ppi_seed}"
    return resolve(args.ppi_result_root) / run_name


def ppi_embedding_template(args: argparse.Namespace) -> Path:
    run_name = f"hippie_highconf_dim{{dim}}_score{score_tag(args.ppi_score_threshold)}_seed{args.ppi_seed}"
    return resolve(args.ppi_result_root) / run_name / "ppi_node_embedding.csv"


def build_dataset_command(args: argparse.Namespace) -> list[str]:
    return [
        args.python_exe,
        "-u",
        str(DATASET_BUILDER),
        "--x-file",
        str(resolve(args.source_x_file)),
        "--y-file",
        str(resolve(args.source_y_file)),
        "--split-file",
        str(resolve(args.source_split_file)),
        "--output-dir",
        str(resolve(args.dataset_output_dir)),
    ]


def build_ppi_command(args: argparse.Namespace, dim: int) -> list[str]:
    cmd = [
        args.python_exe,
        "-u",
        str(PPI_BUILDER),
        "--source",
        "hippie",
        "--gene-list-file",
        str(shared_x_file(args)),
        "--result-dir",
        str(ppi_result_dir(args, dim)),
        "--edge-output-file",
        str(ROOT / "pretraining_dataset" / "ppi_networks" / "hippie_highconf_edges.csv"),
        "--score-threshold",
        str(args.ppi_score_threshold),
        "--component-policy",
        args.ppi_component_policy,
        "--embedding-dim",
        str(dim),
        "--seed",
        str(args.ppi_seed),
        "--epochs",
        str(args.ppi_epochs),
        "--batch-size",
        str(args.ppi_batch_size),
        "--lr",
        str(args.ppi_lr),
        "--walk-length",
        str(args.ppi_walk_length),
        "--context-size",
        str(args.ppi_context_size),
        "--walks-per-node",
        str(args.ppi_walks_per_node),
        "--negative-samples",
        str(args.ppi_negative_samples),
        "--max-pairs-per-epoch",
        str(args.ppi_max_pairs_per_epoch),
        "--device",
        args.ppi_device,
    ]
    if args.hippie_file is not None:
        cmd.extend(["--hippie-file", str(resolve(args.hippie_file))])
    if args.ppi_skip_node2vec:
        cmd.append("--skip-node2vec")
    return cmd


def append_list_arg(cmd: list[str], flag: str, values: list[Any] | None) -> None:
    if values is not None:
        cmd.extend([flag, *[str(value) for value in values]])


def build_grid_command(args: argparse.Namespace, grid_passthrough: list[str]) -> list[str]:
    cmd = [
        args.python_exe,
        "-u",
        str(GRID_RUNNER),
        "--x-file",
        str(shared_x_file(args)),
        "--y-file",
        str(shared_y_file(args)),
        "--result-root",
        str(resolve(args.grid_result_root)),
        "--preset",
        args.preset,
        "--run-mode",
        args.run_mode,
        "--device",
        args.device,
        "--architectures",
        *args.architectures,
        "--ppi-embedding-template",
        str(ppi_embedding_template(args)),
        "--ppi-gene-policy",
        args.ppi_gene_policy,
    ]
    append_list_arg(cmd, "--embedding-sources", args.embedding_sources)
    append_list_arg(cmd, "--d-models", args.d_models)
    append_list_arg(cmd, "--dropouts", args.dropouts)
    append_list_arg(cmd, "--max-genes-list", args.max_genes_list)
    append_list_arg(cmd, "--gene-selections", args.gene_selections)
    append_list_arg(cmd, "--ad-mci-gene-fractions", args.ad_mci_gene_fractions)
    append_list_arg(cmd, "--augmentations", args.augmentations)
    if args.grid_skip_existing:
        cmd.append("--skip-existing")
    if args.grid_skip_missing_ppi:
        cmd.append("--skip-missing-ppi")
    if args.smoke:
        cmd.append("--smoke")
    cmd.extend(grid_passthrough)
    return cmd


def build_pipeline_steps(args: argparse.Namespace, grid_passthrough: list[str] | None = None) -> list[PipelineStep]:
    grid_passthrough = grid_passthrough or []
    steps: list[PipelineStep] = []

    if not args.skip_dataset_build:
        steps.append(
            PipelineStep(
                name="build_dataset",
                command=build_dataset_command(args),
                log_file=resolve(args.dataset_output_dir) / "final_pipeline_dataset.log",
            )
        )

    if not args.skip_ppi_build and grid_uses_ppi(args):
        for dim in infer_ppi_dims(args):
            result_dir = ppi_result_dir(args, dim)
            if args.skip_existing_ppi and (result_dir / "ppi_node_embedding.csv").exists():
                continue
            steps.append(
                PipelineStep(
                    name=f"build_ppi_dim{dim}",
                    command=build_ppi_command(args, dim),
                    log_file=result_dir / "final_pipeline_ppi.log",
                )
            )

    steps.append(
        PipelineStep(
            name="run_grid",
            command=build_grid_command(args, grid_passthrough),
            log_file=resolve(args.grid_result_root) / "final_pipeline_grid.log",
        )
    )
    return steps


def validate_pipeline_inputs(args: argparse.Namespace) -> None:
    if not args.skip_dataset_build:
        missing_sources = [
            path
            for path in [resolve(args.source_x_file), resolve(args.source_y_file), resolve(args.source_split_file)]
            if not path.exists()
        ]
        if missing_sources:
            raise FileNotFoundError(
                "Cannot build the shared multitask dataset because source files are missing: "
                + ", ".join(str(path) for path in missing_sources)
            )
    elif not shared_x_file(args).exists() or not shared_y_file(args).exists():
        raise FileNotFoundError(
            "Shared multitask X.csv/y.csv are missing. Remove --skip-dataset-build or pass --shared-dataset-dir."
        )


def run_command(cmd: list[str], log_file: Path) -> int:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            cmd,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            handle.write(line)
        return int(process.wait())


def write_pipeline_manifest(
    args: argparse.Namespace,
    steps: list[PipelineStep],
    grid_passthrough: list[str],
) -> Path:
    output_dir = resolve(args.grid_result_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "shared_x_file": str(shared_x_file(args)),
        "shared_y_file": str(shared_y_file(args)),
        "ppi_embedding_template": str(ppi_embedding_template(args)),
        "grid_passthrough": grid_passthrough,
        "steps": [
            {
                **asdict(step),
                "log_file": str(step.log_file),
                "command": step.command,
            }
            for step in steps
        ],
    }
    path = output_dir / "final_pipeline_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def main() -> None:
    args, grid_passthrough = parse_args()
    steps = build_pipeline_steps(args, grid_passthrough)
    manifest_path = write_pipeline_manifest(args, steps, grid_passthrough)

    if args.dry_run:
        print(f"Dry run complete. Planned steps: {len(steps)}. Manifest: {manifest_path}")
        for step in steps:
            print(f"[{step.name}] {' '.join(step.command)}")
        return

    validate_pipeline_inputs(args)
    for index, step in enumerate(steps, start=1):
        print("\n" + "=" * 80, flush=True)
        print(f"Final TxT multitask pipeline step {index}/{len(steps)}: {step.name}", flush=True)
        print("=" * 80, flush=True)
        return_code = run_command(step.command, step.log_file)
        if return_code != 0:
            raise RuntimeError(f"Pipeline step failed: {step.name}. See {step.log_file}")

    print(f"Final TxT multitask pipeline complete. Manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
