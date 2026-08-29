#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from shared_test_splits import TXT_SHARED_SEEDS, write_txt_shared_splits


ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = Path(__file__).resolve().parent
PAIRWISE_ROOT = ROOT / "task_dataset" / "processed" / "txt_pairwise_multitask"
SHARED_Y = PAIRWISE_ROOT / "shared_ad_mci_ctl" / "y.csv"
DEFAULT_RESULT_ROOT = ROOT / "results" / "SOTA" / "shared_test_70_10_20_r10"

TASKS: dict[str, tuple[Path, Path]] = {
    task: (PAIRWISE_ROOT / task / "X.csv", PAIRWISE_ROOT / task / "y.csv")
    for task in ("ad_vs_mci", "ad_vs_ctl", "mci_vs_ctl")
}
TASK_CLASS_NAMES: dict[str, tuple[str, str]] = {
    "ad_vs_mci": ("MCI", "AD"),
    "ad_vs_ctl": ("CTL", "AD"),
    "mci_vs_ctl": ("CTL", "MCI"),
}

PAPERS: dict[str, Path] = {
    "lee-2020": SOURCE_DIR / "nature-2020" / "run_experiments.py",
    "kelly-2023": SOURCE_DIR / "nature-2023" / "run_experiments.py",
    "one2mfusion-2023": SOURCE_DIR / "one2mfusion-2023" / "run_experiments.py",
    "diagnostics-2025": SOURCE_DIR / "diagnostics-2025" / "run_experiments.py",
    "hariharan-2026": SOURCE_DIR / "nature-2026" / "run_experiments.py",
}


@dataclass(frozen=True)
class CommandSpec:
    paper: str
    task: str
    phase: str
    result_dir: str
    command: tuple[str, ...]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one paper's selected paper-aligned method matrix on all three TxT pairwise tasks, "
            "excluding direct all-gene inputs and using the exact shared 70/10/20 seed manifests."
        )
    )
    parser.add_argument("--paper", required=True, choices=sorted(PAPERS))
    parser.add_argument("--tasks", nargs="+", choices=sorted(TASKS), default=list(TASKS))
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--python-exe", default=sys.executable)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--split-manifest-dir", type=Path, default=None)
    parser.add_argument("--n-jobs", type=int, default=2)
    parser.add_argument("--xgboost-device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--ctgan-device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--hprd-network-file", type=Path, default=None)
    parser.add_argument("--kelly-knowledge-genes-file", type=Path, default=None)
    parser.add_argument(
        "--allow-cfg-supplement-proxy",
        action="store_true",
        help="Run Lee's outcome-derived CFG supplement proxy. This is excluded by default because it is not independent of the shared test.",
    )
    parser.add_argument(
        "--allow-incomplete-kelly-knowledge",
        action="store_true",
        help="Run a clearly labelled top-MAD-only ablation if Kelly's missing curated list is not supplied.",
    )
    parser.add_argument("--smoke", action="store_true", help="Run each underlying runner's reduced smoke profile on the first seed.")
    parser.add_argument("--dry-run", action="store_true", help="Print the subprocesses without executing them or writing results.")
    parser.add_argument(
        "--overwrite-paper-root",
        action="store_true",
        help="Delete only this paper's standardized output directory before starting.",
    )
    args = parser.parse_args(argv)
    if not 1 <= args.repeats <= len(TXT_SHARED_SEEDS):
        parser.error(f"--repeats must be between 1 and {len(TXT_SHARED_SEEDS)} for the canonical TxT seed set.")
    if args.smoke:
        args.repeats = 1
    return args


def _resolved(path: Path) -> Path:
    return path if path.is_absolute() else (ROOT / path).resolve()


def _ctgan_uses_cuda(mode: str, python_exe: str) -> bool:
    if mode == "cuda":
        return True
    if mode == "cpu":
        return False
    try:
        probe = subprocess.run(
            [
                str(python_exe),
                "-c",
                "import torch; print('1' if torch.cuda.is_available() else '0')",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        return probe.returncode == 0 and probe.stdout.strip().endswith("1")
    except (OSError, subprocess.SubprocessError):
        return False


def _common_command(
    args: argparse.Namespace,
    paper: str,
    task: str,
    result_dir: Path,
    split_dir: Path,
) -> list[str]:
    x_file, y_file = TASKS[task]
    command = [
        str(args.python_exe),
        "-u",
        str(PAPERS[paper]),
        "--x-file",
        str(x_file),
        "--y-file",
        str(y_file),
        "--result-root",
        str(result_dir),
        "--protocol",
        "batch_holdout",
        "--batch-scenarios",
        "shared_test",
        "--split-manifest-dir",
        str(split_dir),
        "--repeats",
        str(args.repeats),
        "--inner-val-ratio",
        "0.125",
        "--shared-test-size",
        "0.20",
        "--seed",
        "101",
        "--artifact-scope",
        "train_inner",
        "--threshold-mode",
        "fixed_0_5",
        "--skip-existing",
    ]
    if args.smoke:
        command.append("--smoke")
    return command


def build_command_specs(args: argparse.Namespace) -> list[CommandSpec]:
    paper = args.paper
    result_root = _resolved(args.result_root)
    split_dir = _resolved(args.split_manifest_dir) if args.split_manifest_dir else result_root / "_txt_shared_splits"
    paper_root = result_root / paper
    specs: list[CommandSpec] = []
    ctgan_cuda = paper == "hariharan-2026" and _ctgan_uses_cuda(args.ctgan_device, args.python_exe)

    for task in args.tasks:
        task_root = paper_root / task
        if paper == "hariharan-2026":
            common_hariharan = [
                "--feature-count-mode",
                "fixed",
                "--training-balance",
                "undersample",
                "--chi2-bins",
                "10",
                "--deep-epochs",
                "100",
                "--n-jobs",
                str(args.n_jobs),
            ]
            all_models = ["svm", "rf", "adaboost", "xgboost", "dnn", "cnn"]

            unaugmented_dir = task_root / "unaugmented_grid"
            unaugmented = _common_command(args, paper, task, unaugmented_dir, split_dir)
            unaugmented += [
                "--feature-selectors",
                "chi2",
                "anova",
                "rfe",
                "elasticnet",
                "--models",
                *all_models,
                "--augmentations",
                "none",
                *common_hariharan,
            ]
            specs.append(CommandSpec(paper, task, "unaugmented_grid", str(unaugmented_dir), tuple(unaugmented)))

            augmented_dir = task_root / "paper_augmented_grid"
            augmented = _common_command(args, paper, task, augmented_dir, split_dir)
            augmented += [
                "--feature-selectors",
                "chi2",
                "anova",
                "rfe",
                "elasticnet",
                "--models",
                *all_models,
                "--augmentations",
                "ctgan",
                *common_hariharan,
            ]
            if ctgan_cuda:
                augmented.append("--ctgan-cuda")
            specs.append(CommandSpec(paper, task, "paper_augmented_grid", str(augmented_dir), tuple(augmented)))

            extended_dir = task_root / "table11_dnn_k500"
            extended = _common_command(args, paper, task, extended_dir, split_dir)
            extended += [
                "--feature-selectors",
                "chi2",
                "anova",
                "rfe",
                "elasticnet",
                "lasso",
                "rf_importance",
                "--models",
                "dnn",
                "--augmentations",
                "none",
                "--feature-count-mode",
                "fixed",
                "--training-balance",
                "undersample",
                "--chi2-k",
                "500",
                "--anova-k",
                "500",
                "--rfe-k",
                "500",
                "--elasticnet-k",
                "500",
                "--lasso-k",
                "500",
                "--rf-importance-k",
                "500",
                "--chi2-bins",
                "10",
                "--deep-epochs",
                "100",
                "--n-jobs",
                str(args.n_jobs),
            ]
            specs.append(CommandSpec(paper, task, "table11_dnn_k500", str(extended_dir), tuple(extended)))
            continue

        command = _common_command(args, paper, task, task_root, split_dir)
        if paper == "lee-2020":
            command += [
                "--feature-sets",
                "deg",
                "vae",
                "tf_genes",
                "hub_genes",
                "cfg_genes",
                "--models",
                "lr",
                "l1_lr",
                "svm",
                "rf",
                "dnn",
                "--fdr-threshold",
                "0.01",
                "--deg-fallback-top-k",
                "0",
                "--scaler",
                "minmax",
                "--vae-epochs",
                "3000",
                "--paper-profile",
                "--class-weighting",
                "off",
                "--class-names",
                *TASK_CLASS_NAMES[task],
            ]
            if args.allow_cfg_supplement_proxy:
                command.append("--allow-cfg-supplement-proxy")
            if args.hprd_network_file:
                command += ["--hprd-network-file", str(_resolved(args.hprd_network_file))]
        elif paper == "kelly-2023":
            command += [
                "--feature-sets",
                "knowledge_genes",
                "vssrfe_lr",
                "lasso",
                "vae_latent",
                "--models",
                "lr",
                "svm",
                "xgboost",
                "rf",
                "mlp",
                "--implementation-profile",
                "paper",
                "--hyperparameter-mode",
                "fixed_paper",
                "--paper-lasso-alpha",
                "0.15135923730480524",
                "--paper-vssrfe-c",
                "0.012944980118048744",
                "--paper-vssrfe-n-genes",
                "159",
                "--scaler",
                "standard",
                "--vae-architecture",
                "basic",
                "--vae-learning-rate",
                "0.00001",
                "--vae-backend",
                "torch",
                "--vae-device",
                "cuda",
                "--vae-epochs",
                "1000",
                "--deep-patience",
                "10",
                "--vae-reconstruction-loss",
                "categorical_crossentropy",
                "--xgboost-device",
                args.xgboost_device,
                "--n-jobs",
                str(args.n_jobs),
            ]
            if args.kelly_knowledge_genes_file:
                command += ["--knowledge-genes-file", str(_resolved(args.kelly_knowledge_genes_file))]
            elif args.allow_incomplete_kelly_knowledge:
                command.append("--allow-incomplete-knowledge")
        elif paper == "one2mfusion-2023":
            command += [
                "--models",
                "cnn",
                "one2mfusion",
                "--implementation-profile",
                "paper",
                "--hyperparameter-mode",
                "fixed_paper",
                "--paper-lasso-alpha",
                "0.000001",
                "--gene-selection-mode",
                "nonzero",
                "--pixels",
                "90",
                "--fisher-groups",
                "15",
                "--image-gene-order",
                "fisher",
                "--epochs",
                "1003",
                "--batch-size-fusion",
                "30",
                "--batch-size-single",
                "30",
                "--learning-rate",
                "0.0001",
                "--patience",
                "10",
                "--start-from-epoch",
                "250",
                "--early-stopping-monitor",
                "val_loss",
                "--paper-like-preprocessing",
                "--n-jobs",
                str(args.n_jobs),
            ]
        elif paper == "diagnostics-2025":
            command += [
                "--models",
                "dl",
                "svm",
                "gbm",
                "rf",
                "--sampling",
                "no_smote",
                "borderline_smote",
                "--xgb-top-k",
                "300",
                "--sfbs-min-genes",
                "95",
                "--sfbs-max-genes",
                "95",
                "--sfbs-step",
                "1",
                "--sfbs-cv-folds",
                "5",
                "--sfbs-mode",
                "true",
                "--sfbs-target-genes",
                "95",
                "--dl-epochs",
                "4000",
                "--dl-batch-size",
                "5",
                "--n-jobs",
                str(args.n_jobs),
            ]
        else:
            raise AssertionError(f"Unhandled paper: {paper}")
        specs.append(CommandSpec(paper, task, "paper_matrix", str(task_root), tuple(command)))
    return specs


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _display_path(path: Path) -> str:
    path = path.resolve()
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _fingerprinted_files(args: argparse.Namespace, split_dir: Path) -> list[dict[str, object]]:
    files: set[Path] = {
        (SOURCE_DIR / "shared_test_splits.py").resolve(),
        (SOURCE_DIR / "strict_v2_utils.py").resolve(),
        SHARED_Y.resolve(),
        (ROOT / "pretraining_dataset" / "geo_downloads" / "GSE63060" / "eset_1_GPL6947" / "GSE63060_sample_metadata.tsv.gz").resolve(),
        (ROOT / "pretraining_dataset" / "geo_downloads" / "GSE63061" / "eset_1_GPL10558" / "GSE63061_sample_metadata.tsv.gz").resolve(),
    }
    for task in args.tasks:
        files.update(path.resolve() for path in TASKS[task])
    paper_source_dir = PAPERS[args.paper].parent
    files.update(
        path.resolve()
        for path in paper_source_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".py", ".r", ".txt"} and "__pycache__" not in path.parts
    )
    files.update((split_dir / f"seed_{seed}.csv").resolve() for seed in TXT_SHARED_SEEDS[: args.repeats])
    if args.hprd_network_file:
        files.add(_resolved(args.hprd_network_file))
    if args.kelly_knowledge_genes_file:
        files.add(_resolved(args.kelly_knowledge_genes_file))
    if args.paper == "lee-2020":
        supplement_dir = ROOT / "SOTA" / "Public Code" / "nature-2020"
        files.update(path.resolve() for path in supplement_dir.rglob("*") if path.is_file())

    missing = [path for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Cannot fingerprint missing benchmark inputs: {[str(path) for path in missing]}")
    return [
        {"path": _display_path(path), "bytes": path.stat().st_size, "sha256": _sha256_file(path)}
        for path in sorted(files, key=lambda item: str(item).lower())
    ]


def _dependency_versions(paper: str) -> dict[str, str]:
    versions: dict[str, str] = {"python": sys.version.split()[0]}
    distributions = [
        "numpy",
        "pandas",
        "scikit-learn",
        "tensorflow",
        "xgboost",
        "scikit-optimize",
        "mlxtend",
        "ctgan",
    ]
    # PyTorch is a Kelly-specific CUDA adaptation. Including it in every
    # paper's plan would invalidate resumable runs for papers that never use it.
    if paper == "kelly-2023":
        distributions.append("torch")
    for distribution in distributions:
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = "not-installed"
    return versions


def _plan_payload(args: argparse.Namespace, specs: Sequence[CommandSpec], split_dir: Path) -> dict[str, object]:
    command_rows = [asdict(spec) for spec in specs]
    serialized = json.dumps(command_rows, sort_keys=True, separators=(",", ":"))
    payload: dict[str, object] = {
        "schema_version": 2,
        "paper": args.paper,
        "tasks": list(args.tasks),
        "protocol": "TxT shared-test adaptation; exact shared multiclass 70/10/20 manifests; direct all-gene inputs excluded",
        "repeats": args.repeats,
        "seeds": list(TXT_SHARED_SEEDS[: args.repeats]),
        "split_manifest_dir": str(split_dir),
        "command_sha256": hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
        "commands": command_rows,
        "file_fingerprints": _fingerprinted_files(args, split_dir),
        "dependency_versions": _dependency_versions(args.paper),
    }
    plan_serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    payload["plan_sha256"] = hashlib.sha256(plan_serialized.encode("utf-8")).hexdigest()
    return payload


def _prepare_paper_root(paper_root: Path, result_root: Path, overwrite: bool) -> None:
    paper_root = paper_root.resolve()
    result_root = result_root.resolve()
    if result_root not in paper_root.parents:
        raise ValueError(f"Refusing to manage paper output outside result root: {paper_root}")
    if overwrite and paper_root.exists():
        shutil.rmtree(paper_root)
    if paper_root.exists() and any(paper_root.iterdir()) and not (paper_root / "orchestration_plan.json").is_file():
        raise RuntimeError(
            f"Refusing to adopt non-empty output without an orchestration plan: {paper_root}. "
            "Use a new --result-root or --overwrite-paper-root."
        )
    paper_root.mkdir(parents=True, exist_ok=True)


def _write_or_validate_plan(plan_path: Path, payload: dict[str, object]) -> None:
    if plan_path.exists():
        existing = json.loads(plan_path.read_text(encoding="utf-8"))
        if existing.get("plan_sha256") != payload.get("plan_sha256"):
            if _plans_semantically_compatible(existing, payload):
                plan_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
                print(f"Updated semantically compatible benchmark plan: {plan_path}", flush=True)
                return
            raise RuntimeError(
                f"Existing benchmark plan differs from this invocation: {plan_path}. "
                "Use a new --result-root or --overwrite-paper-root."
            )
        return
    plan_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _plans_semantically_compatible(existing: dict[str, object], current: dict[str, object]) -> bool:
    """Allow resume when only orchestration metadata changed, never experiment inputs.

    Older schema-v2 plans fingerprinted this wrapper itself. That made an edit to
    another paper's command builder invalidate otherwise identical completed runs.
    The generated commands already capture paper-specific wrapper behavior, so the
    wrapper fingerprint and descriptive protocol text are excluded from this
    compatibility comparison.
    """

    required = {
        "paper",
        "tasks",
        "repeats",
        "seeds",
        "split_manifest_dir",
        "commands",
        "file_fingerprints",
        "dependency_versions",
    }
    if not required.issubset(existing) or not required.issubset(current):
        return False

    def compatibility_view(plan: dict[str, object]) -> dict[str, object]:
        fingerprints = []
        for raw_row in plan.get("file_fingerprints", []):
            row = dict(raw_row)
            normalized_path = str(row.get("path", "")).replace("\\", "/").lower()
            if normalized_path.endswith("sota/source/run_shared_paper_benchmark.py"):
                continue
            fingerprints.append(row)
        fingerprints.sort(key=lambda row: str(row.get("path", "")).replace("\\", "/").lower())
        view = {
            "paper": plan.get("paper"),
            "tasks": plan.get("tasks"),
            "repeats": plan.get("repeats"),
            "seeds": plan.get("seeds"),
            "split_manifest_dir": plan.get("split_manifest_dir"),
            "commands": plan.get("commands"),
            "file_fingerprints": fingerprints,
            "dependency_versions": plan.get("dependency_versions"),
        }
        # Dataclass serialization keeps command tuples in memory, whereas JSON
        # reloads them as lists. Normalize both representations before compare.
        return json.loads(json.dumps(view, sort_keys=True, default=str))

    return compatibility_view(existing) == compatibility_view(current)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    args.result_root = _resolved(args.result_root)
    args.split_manifest_dir = _resolved(args.split_manifest_dir) if args.split_manifest_dir else args.result_root / "_txt_shared_splits"
    specs = build_command_specs(args)

    for spec in specs:
        print(f"[{spec.paper} | {spec.task} | {spec.phase}]", flush=True)
        print(subprocess.list2cmdline(list(spec.command)), flush=True)
    if args.dry_run:
        return

    for task in args.tasks:
        for path in TASKS[task]:
            if not path.is_file():
                raise FileNotFoundError(f"Missing task dataset file: {path}")
    paper_root = args.result_root / args.paper
    _prepare_paper_root(paper_root, args.result_root, args.overwrite_paper_root)
    write_txt_shared_splits(
        SHARED_Y,
        args.split_manifest_dir,
        seeds=TXT_SHARED_SEEDS[: args.repeats],
        overwrite=False,
    )

    plan = _plan_payload(args, specs, args.split_manifest_dir)
    _write_or_validate_plan(paper_root / "orchestration_plan.json", plan)

    for spec in specs:
        subprocess.run(list(spec.command), cwd=ROOT, check=True)
    print(f"Completed {args.paper}. Results: {paper_root}", flush=True)


if __name__ == "__main__":
    main()
