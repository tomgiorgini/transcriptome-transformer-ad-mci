#!/usr/bin/env python3
"""Run the baseline TxT key-attention workflow for an exact seed set.

The workflow is deliberately narrow: held-out test data, dense key-side
incoming attention, branch-free baseline checkpoints, and no query-side
analysis.  Existing exports are never overwritten.  They are skipped only
after the repository audit and an additional provenance/hash check both pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
EXTRACTOR = SCRIPT_DIR / "extract_txt_key_attention.py"
AUDITOR = SCRIPT_DIR / "audit_txt_baseline_attention_seeds.py"
SUMMARIZER = SCRIPT_DIR / "summarize_txt_key_attention_seeds.py"

DEFAULT_SEEDS = list(range(101, 111))
ATTENTION_FILES = (
    "key_attention_by_seed.csv",
    "key_attention_by_subject.npz",
    "key_attention_qc.csv",
    "key_attention_manifest.json",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Sequentially extract full-test baseline TxT key attention for an exact "
            "seed set, audit every export, and build the cross-seed consensus."
        )
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="mps")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--audit-output-dir", type=Path, default=None)
    parser.add_argument("--consensus-output-dir", type=Path, default=None)
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260812)
    parser.add_argument(
        "--stable-core-min-top20-frequency", type=float, default=0.7
    )
    return parser


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}.")
    return value


def normalize_run_root(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.name == "seeds10":
        resolved = resolved.parent
    seed_root = resolved / "seeds10"
    if not seed_root.is_dir():
        raise FileNotFoundError(f"Seed root does not exist: {seed_root}")
    return resolved


def discover_seed_dirs(run_root: Path, seeds: Sequence[int]) -> dict[int, Path]:
    seed_root = run_root / "seeds10"
    result: dict[int, Path] = {}
    for seed in seeds:
        matches = sorted(
            path.resolve()
            for path in seed_root.glob(f"*/seed_{seed}")
            if path.is_dir()
        )
        if len(matches) != 1:
            raise FileNotFoundError(
                f"Expected exactly one run directory for seed {seed}; found {matches}."
            )
        result[int(seed)] = matches[0]
    return result


def run_subprocess(command: Sequence[str], label: str) -> None:
    print(f"\n[{label}] {shlex.join(command)}", flush=True)
    result = subprocess.run(list(command), cwd=ROOT, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"{label} failed with exit code {result.returncode}.")


def audit_command(
    run_root: Path,
    seeds: Sequence[int],
    device: str,
    output_dir: Path,
) -> list[str]:
    return [
        sys.executable,
        str(AUDITOR),
        "--run-root",
        str(run_root),
        "--seeds",
        *[str(seed) for seed in seeds],
        "--expected-device",
        device,
        "--attention-split",
        "test",
        "--output-dir",
        str(output_dir),
    ]


def read_audit_results(output_dir: Path) -> dict[int, dict[str, Any]]:
    report = read_json(output_dir / "seed_audit_status.json")
    rows = report.get("results")
    if not isinstance(rows, list):
        raise ValueError("Audit JSON does not contain a results list.")
    result: dict[int, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or "seed" not in row:
            raise ValueError("Audit JSON contains an invalid seed result.")
        seed = int(row["seed"])
        if seed in result:
            raise ValueError(f"Audit JSON contains duplicate seed {seed}.")
        result[seed] = row
    return result


def artifact_path_map(run_dir: Path, args: dict[str, Any]) -> dict[str, Path]:
    def argument_path(name: str) -> Path:
        raw = args.get(name)
        if not isinstance(raw, str) or not raw:
            raise ValueError(f"args.json lacks a valid {name!r} path in {run_dir}.")
        path = Path(raw).expanduser()
        return path.resolve() if path.is_absolute() else (ROOT / path).resolve()

    return {
        "args": run_dir / "args.json",
        "x": argument_path("x_file"),
        "y": argument_path("y_file"),
        "split": argument_path("split_file"),
        "selected_genes": run_dir / "selected_genes.csv",
        "constructor_embedding": run_dir / "gene_embedding.csv",
        "extractor": EXTRACTOR,
    }


def validate_existing_export(run_dir: Path, seed: int) -> None:
    """Supplement the full content audit with strict current-file provenance."""
    attention_dir = run_dir / "key_attention" / "test"
    for name in ATTENTION_FILES:
        path = attention_dir / name
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(
                f"Refusing to skip seed {seed}: export artifact is missing/empty: {path}"
            )
    manifest_path = attention_dir / "key_attention_manifest.json"
    manifest = read_json(manifest_path)
    expected_scalars = {
        "seed": int(seed),
        "model_variant": "baseline",
        "split": "test",
        "is_full_split": True,
        "max_samples_requested": None,
        "vma_included": False,
        "query_metrics_included": False,
    }
    for key, expected in expected_scalars.items():
        if manifest.get(key) != expected:
            raise ValueError(
                f"Refusing to skip seed {seed}: manifest {key!r} is "
                f"{manifest.get(key)!r}, expected {expected!r}."
            )
    if int(manifest.get("samples", -1)) != int(
        manifest.get("full_split_samples", -2)
    ):
        raise ValueError(
            f"Refusing to skip seed {seed}: export is not the complete test split."
        )
    checkpoint = (run_dir / "best_model.pt").resolve()
    recorded_checkpoint = Path(str(manifest.get("checkpoint", ""))).expanduser()
    if not recorded_checkpoint.is_absolute():
        recorded_checkpoint = (ROOT / recorded_checkpoint).resolve()
    else:
        recorded_checkpoint = recorded_checkpoint.resolve()
    if recorded_checkpoint != checkpoint:
        raise ValueError(
            f"Refusing to skip seed {seed}: checkpoint path differs from best_model.pt."
        )
    if manifest.get("checkpoint_sha256") != sha256_file(checkpoint):
        raise ValueError(
            f"Refusing to skip seed {seed}: checkpoint SHA-256 is stale."
        )

    args = read_json(run_dir / "args.json")
    expected_paths = artifact_path_map(run_dir, args)
    recorded_hashes = manifest.get("artifact_sha256")
    if not isinstance(recorded_hashes, dict):
        raise ValueError(
            f"Refusing to skip seed {seed}: artifact_sha256 is absent or invalid."
        )
    missing = sorted(set(expected_paths).difference(recorded_hashes))
    if missing:
        raise ValueError(
            f"Refusing to skip seed {seed}: missing artifact hashes {missing}."
        )
    for name, path in expected_paths.items():
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(
                f"Refusing to skip seed {seed}: provenance artifact is missing/empty: {path}"
            )
        if recorded_hashes.get(name) != sha256_file(path):
            raise ValueError(
                f"Refusing to skip seed {seed}: provenance SHA-256 mismatch for {name}."
            )


def ensure_empty_extraction_target(run_dir: Path, seed: int) -> None:
    attention_dir = run_dir / "key_attention" / "test"
    if attention_dir.exists() and any(attention_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty attention target for seed {seed}: "
            f"{attention_dir}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    cli = build_parser().parse_args(argv)
    seeds = [int(seed) for seed in cli.seeds]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("--seeds must contain one or more unique integers.")
    if cli.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if cli.bootstrap_iterations <= 0:
        raise ValueError("--bootstrap-iterations must be positive.")
    if not 0.0 <= cli.stable_core_min_top20_frequency <= 1.0:
        raise ValueError("--stable-core-min-top20-frequency must be in [0, 1].")
    for dependency in (EXTRACTOR, AUDITOR, SUMMARIZER):
        if not dependency.is_file():
            raise FileNotFoundError(f"Required workflow script is missing: {dependency}")

    run_root = normalize_run_root(cli.run_root)
    seed_dirs = discover_seed_dirs(run_root, seeds)
    print("Baseline key-only test-attention batch", flush=True)
    print(f"Run root: {run_root}", flush=True)
    print(f"Seeds: {', '.join(map(str, seeds))}", flush=True)
    print(f"Requested device: {cli.device}", flush=True)

    with tempfile.TemporaryDirectory(prefix="txt-key-attention-preflight-") as temp:
        preflight_dir = Path(temp)
        run_subprocess(
            audit_command(run_root, seeds, cli.device, preflight_dir),
            "preflight audit",
        )
        preflight = read_audit_results(preflight_dir)
        if set(preflight) != set(seeds):
            raise ValueError("Preflight audit did not return exactly the requested seeds.")

        pending: list[int] = []
        for seed in seeds:
            status = str(preflight[seed].get("attention_status", ""))
            if status == "PASS":
                validate_existing_export(seed_dirs[seed], seed)
                print(
                    f"[seed {seed}] existing full-test export passed audit and "
                    "provenance validation; skipping.",
                    flush=True,
                )
            elif status == "not_present":
                ensure_empty_extraction_target(seed_dirs[seed], seed)
                pending.append(seed)
                print(f"[seed {seed}] no export present; queued.", flush=True)
            else:
                raise RuntimeError(
                    f"Seed {seed} has attention_status={status!r}; refusing overwrite."
                )

        for index, seed in enumerate(pending, start=1):
            run_dir = seed_dirs[seed]
            ensure_empty_extraction_target(run_dir, seed)
            print(
                f"\n[seed {seed}] extraction {index}/{len(pending)}", flush=True
            )
            run_subprocess(
                [
                    sys.executable,
                    str(EXTRACTOR),
                    "--run-dir",
                    str(run_dir),
                    "--checkpoint",
                    str(run_dir / "best_model.pt"),
                    "--split",
                    "test",
                    "--device",
                    cli.device,
                    "--batch-size",
                    str(cli.batch_size),
                ],
                f"seed {seed} key-attention extraction",
            )
            post_seed_dir = preflight_dir / f"post_seed_{seed}"
            run_subprocess(
                audit_command(run_root, [seed], cli.device, post_seed_dir),
                f"seed {seed} post-extraction audit",
            )
            post_result = read_audit_results(post_seed_dir).get(seed, {})
            if post_result.get("attention_status") != "PASS":
                raise RuntimeError(
                    f"Seed {seed} did not pass its post-extraction attention audit."
                )
            validate_existing_export(run_dir, seed)

    audit_output_dir = (
        cli.audit_output_dir.expanduser().resolve()
        if cli.audit_output_dir is not None
        else run_root / "audit_baseline_attention"
    )
    run_subprocess(
        audit_command(run_root, seeds, cli.device, audit_output_dir),
        "final all-seed audit",
    )
    final_audit = read_audit_results(audit_output_dir)
    if set(final_audit) != set(seeds) or any(
        row.get("status") != "PASS" or row.get("attention_status") != "PASS"
        for row in final_audit.values()
    ):
        raise RuntimeError("Final audit did not PASS every requested attention export.")

    consensus_output_dir = (
        cli.consensus_output_dir.expanduser().resolve()
        if cli.consensus_output_dir is not None
        else run_root / "key_attention_consensus"
    )
    run_subprocess(
        [
            sys.executable,
            str(SUMMARIZER),
            "--run-root",
            str(run_root),
            "--seeds",
            *[str(seed) for seed in seeds],
            "--output-dir",
            str(consensus_output_dir),
            "--bootstrap-iterations",
            str(cli.bootstrap_iterations),
            "--bootstrap-seed",
            str(cli.bootstrap_seed),
            "--stable-core-min-top20-frequency",
            str(cli.stable_core_min_top20_frequency),
        ],
        "cross-seed key-attention consensus",
    )
    print("\nBatch complete.", flush=True)
    print(f"Audit: {audit_output_dir}", flush=True)
    print(f"Consensus: {consensus_output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, FileExistsError, RuntimeError, TypeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1) from None
