from __future__ import annotations

import hashlib
import json
import math
import os
import shlex
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_WORKER = ROOT / "experiments" / "scripts" / "paper_comparison" / "train_txt_multitask.py"
DEFAULT_SHARED_DATASET = (
    ROOT / "task_dataset" / "processed" / "txt_pairwise_multitask" / "shared_ad_mci_ctl"
)
DEFAULT_PPI_EDGE_FILE = ROOT / "pretraining_dataset" / "ppi_networks" / "hippie_highconf_edges.csv"
DEFAULT_RESULT_ROOT = ROOT / "results" / "paper_comparison" / "txt_volumetric"

PROTOCOL_SEEDS = tuple(range(101, 111))
BETA_CANDIDATES = (0.0, 0.5, 1.0, 1.5)
TASK_NAMES = ("AD_vs_MCI", "AD_vs_CTL", "MCI_vs_CTL")
TASK_WEIGHTS = {"AD_vs_MCI": 0.70, "AD_vs_CTL": 0.15, "MCI_vs_CTL": 0.15}
LOSS_WEIGHTS = {"AD_vs_MCI": 0.50, "AD_vs_CTL": 0.25, "MCI_vs_CTL": 0.25}
CHECKPOINT_METRIC = "val_weighted_70_15_15_auc_minus_025_loss"


class ProtocolError(RuntimeError):
    """Raised when an experimental artifact is missing or internally inconsistent."""


def resolve_path(path: Path | str) -> Path:
    path = Path(path)
    return path if path.is_absolute() else (ROOT / path).resolve()


def float_tag(value: float) -> str:
    """Return a stable, path-safe representation without conflating distinct values."""
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"Expected a finite value, got {value!r}.")
    rendered = f"{value:.12g}"
    return rendered.replace("-", "m").replace(".", "p").replace("+", "")


def beta_run_name(beta: float) -> str:
    return f"beta_{float_tag(beta)}"


def seed_dir(result_root: Path, seed: int) -> Path:
    return result_root / f"seed_{int(seed)}"


def candidate_dir(result_root: Path, seed: int, beta: float) -> Path:
    return seed_dir(result_root, seed) / "candidates" / beta_run_name(beta)


def baseline_dir(result_root: Path, seed: int) -> Path:
    return seed_dir(result_root, seed) / "baseline"


def selected_test_dir(result_root: Path, seed: int) -> Path:
    return seed_dir(result_root, seed) / "selected_test"


def baseline_test_dir(result_root: Path, seed: int) -> Path:
    """Return the stable evaluation-only directory for the paired baseline."""

    return seed_dir(result_root, seed) / "baseline_test"


def selected_beta_path(result_root: Path, seed: int) -> Path:
    return seed_dir(result_root, seed) / "selected_beta.json"


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(json_ready(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


_SHA256_CACHE: dict[tuple[str, int, int], str] = {}


def cached_sha256_file(path: Path) -> str:
    """Hash immutable protocol inputs once per process, invalidating on stat changes."""

    resolved = path.resolve()
    stat = resolved.stat()
    key = (str(resolved), int(stat.st_size), int(stat.st_mtime_ns))
    digest = _SHA256_CACHE.get(key)
    if digest is None:
        digest = sha256_file(resolved)
        _SHA256_CACHE[key] = digest
    return digest


def command_text(command: Sequence[str]) -> str:
    return shlex.join([str(part) for part in command])


_FINGERPRINT_PATH_OPTIONS = {
    "--x-file": "<x-file>",
    "--y-file": "<y-file>",
    "--split-file": "<split-file>",
    "--ppi-edge-file": "<ppi-edge-file>",
}


def semantic_worker_argv(command: Sequence[str]) -> list[str]:
    """Canonicalize a worker command for path-neutral resume comparisons."""

    parts = [str(part) for part in command]
    try:
        first_option = next(index for index, part in enumerate(parts) if part.startswith("--"))
    except StopIteration:
        first_option = len(parts)
    parts = parts[first_option:]
    result: list[str] = []
    index = 0
    while index < len(parts):
        option = parts[index]
        if option == "--result-dir":
            index += 2
            continue
        if option in _FINGERPRINT_PATH_OPTIONS:
            if index + 1 >= len(parts) or parts[index + 1].startswith("--"):
                raise ProtocolError(f"Malformed worker command: {option} has no value.")
            result.extend([option, _FINGERPRINT_PATH_OPTIONS[option]])
            index += 2
            continue
        if option == "--evaluation-only-checkpoint":
            if index + 1 >= len(parts) or parts[index + 1].startswith("--"):
                raise ProtocolError(f"Malformed worker command: {option} has no value.")
            result.extend([option, "<checkpoint>"])
            index += 2
            continue
        if option == "--evaluation-only-checkpoints":
            result.extend([option, "<checkpoint-list>"])
            index += 1
            while index < len(parts) and not parts[index].startswith("--"):
                index += 1
            continue
        result.append(option)
        index += 1
    return result


def build_resume_fingerprint(
    command: Sequence[str],
    *,
    kind: str,
    input_files: dict[str, Path | str],
    checkpoint_files: Sequence[Path | str] = (),
) -> dict[str, Any]:
    """Build a semantic, content-addressed fingerprint for safe reuse."""

    inputs: dict[str, dict[str, str]] = {}
    semantic_input_hashes: dict[str, str] = {}
    for role, raw_path in sorted(input_files.items()):
        path = resolve_path(raw_path)
        if not path.exists():
            raise ProtocolError(f"Cannot fingerprint missing {role} input: {path}")
        digest = cached_sha256_file(path)
        inputs[str(role)] = {"path": str(path), "sha256": digest}
        semantic_input_hashes[str(role)] = digest

    checkpoints: list[dict[str, str]] = []
    checkpoint_hashes: list[str] = []
    for raw_path in checkpoint_files:
        path = resolve_path(raw_path)
        if not path.exists():
            raise ProtocolError(f"Cannot fingerprint missing evaluation checkpoint: {path}")
        digest = cached_sha256_file(path)
        checkpoints.append({"path": str(path), "sha256": digest})
        checkpoint_hashes.append(digest)

    semantic_payload = {
        "schema_version": 1,
        "kind": str(kind),
        "semantic_argv": semantic_worker_argv(command),
        "input_sha256": semantic_input_hashes,
        "checkpoint_sha256": checkpoint_hashes,
    }
    encoded = json.dumps(semantic_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        **semantic_payload,
        "inputs": inputs,
        "checkpoints": checkpoints,
        "semantic_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def validate_resume_fingerprint(run_dir: Path, expected: dict[str, Any]) -> None:
    """Refuse ``--skip-existing`` when provenance is absent or incompatible."""

    command_path = run_dir / "worker_command.json"
    if not command_path.exists():
        raise ProtocolError(
            f"Cannot safely reuse {run_dir}: missing worker_command.json with a resume fingerprint. "
            "Use a new result root or rerun without --skip-existing."
        )
    payload = read_json(command_path)
    observed = payload.get("resume_fingerprint") if isinstance(payload, dict) else None
    if not isinstance(observed, dict) or not observed.get("semantic_sha256"):
        raise ProtocolError(
            f"Cannot safely reuse {run_dir}: historical worker_command.json has no semantic "
            "resume fingerprint. Use a new result root or rerun without --skip-existing."
        )
    expected_digest = str(expected.get("semantic_sha256", ""))
    observed_digest = str(observed.get("semantic_sha256", ""))
    if observed_digest != expected_digest:
        raise ProtocolError(
            f"Refusing --skip-existing for {run_dir}: semantic resume fingerprint mismatch "
            f"(stored={observed_digest}, requested={expected_digest}). The command, inputs, "
            "split, PPI graph, or selected checkpoints changed; use a new result root."
        )


def save_worker_command(
    run_dir: Path,
    command: Sequence[str],
    *,
    kind: str,
    seed: int,
    beta: float | None,
    resume_fingerprint: dict[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "kind": kind,
        "seed": int(seed),
        "beta": None if beta is None else float(beta),
        "argv": [str(part) for part in command],
        "shell_display": command_text(command),
    }
    if resume_fingerprint is not None:
        payload["resume_fingerprint"] = resume_fingerprint
    write_json(
        run_dir / "worker_command.json",
        payload,
    )


def run_worker(command: Sequence[str], run_dir: Path, *, dry_run: bool = False) -> None:
    print(command_text(command), flush=True)
    if dry_run:
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "worker.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            [str(part) for part in command],
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        # Stream every worker line to the terminal while retaining the same
        # complete worker.log artifact for post-hoc inspection.
        assert process.stdout is not None
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            return_code = process.wait()
        except KeyboardInterrupt:
            process.terminate()
            process.wait()
            raise
    if return_code != 0:
        raise ProtocolError(
            f"Worker failed with exit code {return_code}. See {run_dir / 'worker.log'}."
        )


def _finite_float(value: Any, *, source: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ProtocolError(f"Non-numeric checkpoint score in {source}: {value!r}.") from exc
    if not math.isfinite(result):
        raise ProtocolError(f"Non-finite checkpoint score in {source}: {result!r}.")
    return result


def _score_from_ensemble_model_summary(path: Path) -> float | None:
    if not path.exists():
        return None
    payload = read_json(path)
    training = payload.get("training_summary", {})
    raw_value = training.get("ensemble_validation_score")
    if raw_value is None:
        return None
    metric = training.get("ensemble_validation_score_metric")
    if isinstance(raw_value, dict):
        metric = raw_value.get("checkpoint_metric", metric)
        raw_value = raw_value.get("value", raw_value.get("score"))
    if metric not in {None, CHECKPOINT_METRIC}:
        raise ProtocolError(
            f"{path} used ensemble checkpoint metric {metric!r}, expected "
            f"{CHECKPOINT_METRIC!r}."
        )
    if raw_value is None:
        raise ProtocolError(f"{path} contains ensemble_validation_score without a value.")
    return _finite_float(raw_value, source=f"{path}:ensemble_validation_score")


def _score_from_model_summary(path: Path) -> float | None:
    if not path.exists():
        return None
    payload = read_json(path)
    training = payload.get("training_summary", {})
    metric = training.get("checkpoint_metric")
    if metric not in {None, CHECKPOINT_METRIC}:
        raise ProtocolError(
            f"{path} used checkpoint metric {metric!r}, expected {CHECKPOINT_METRIC!r}."
        )
    value = training.get("best_checkpoint_value")
    return None if value is None else _finite_float(value, source=str(path))


def _score_from_training_log(path: Path) -> float | None:
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    if "checkpoint_value" not in frame.columns or frame.empty:
        return None
    if "checkpoint_metric" in frame.columns:
        observed = set(frame["checkpoint_metric"].dropna().astype(str))
        if observed and observed != {CHECKPOINT_METRIC}:
            raise ProtocolError(
                f"{path} contains checkpoint metrics {sorted(observed)!r}, expected only {CHECKPOINT_METRIC!r}."
            )
    values = pd.to_numeric(frame["checkpoint_value"], errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    return None if values.size == 0 else float(np.max(values))


def _score_from_metrics(path: Path) -> float | None:
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    required = {"split", "task", "roc_auc", "loss"}
    if not required.issubset(frame.columns):
        return None
    val = frame.loc[frame["split"].astype(str).str.lower() == "val"].copy()
    if val.empty:
        return None
    by_task = {str(row["task"]): row for _, row in val.iterrows()}
    missing = [task for task in TASK_NAMES if task not in by_task]
    if missing:
        raise ProtocolError(f"Cannot reconstruct validation score from {path}: missing tasks {missing}.")
    auc_term = sum(TASK_WEIGHTS[task] * _finite_float(by_task[task]["roc_auc"], source=str(path)) for task in TASK_NAMES)
    loss_term = sum(LOSS_WEIGHTS[task] * _finite_float(by_task[task]["loss"], source=str(path)) for task in TASK_NAMES)
    return float(auc_term - 0.25 * loss_term)


def validation_checkpoint_score(run_dir: Path) -> tuple[float, str]:
    """Read the exact validation-only checkpoint signal, with explicit fallbacks."""
    readers = (
        (_score_from_ensemble_model_summary, run_dir / "model_summary.json"),
        (_score_from_model_summary, run_dir / "model_summary.json"),
        (_score_from_training_log, run_dir / "training_log.csv"),
        (_score_from_metrics, run_dir / "metrics_summary.csv"),
    )
    for reader, path in readers:
        score = reader(path)
        if score is not None:
            return score, str(path)
    raise ProtocolError(
        f"No usable validation checkpoint score in {run_dir}. Expected model_summary.json, "
        "training_log.csv, or validation rows in metrics_summary.csv."
    )


@dataclass(frozen=True)
class BetaCandidate:
    seed: int
    beta: float
    score: float
    run_dir: str
    checkpoint: str
    score_source: str


@dataclass(frozen=True)
class BetaSelection:
    seed: int
    selected_beta: float
    selected_score: float
    selected_run_dir: str
    selected_checkpoint: str
    checkpoint_metric: str
    tie_break: str
    candidates: list[dict[str, Any]]
    selected_ensemble_checkpoints: list[str]
    selected_ensemble_size: int
    selected_ensemble_aggregation: str


def ensemble_checkpoint_paths(run_dir: Path) -> list[Path]:
    """Return the ordered probability-ensemble checkpoints, with historical fallback."""

    summary_path = run_dir / "model_summary.json"
    ranks: list[int] = []
    if summary_path.exists():
        payload = read_json(summary_path)
        training = payload.get("training_summary", {}) if isinstance(payload, dict) else {}
        metadata = training.get("checkpoint_ensemble", []) if isinstance(training, dict) else []
        if isinstance(metadata, list):
            for entry in metadata:
                if isinstance(entry, dict) and entry.get("rank") is not None:
                    ranks.append(int(entry["rank"]))
    if not ranks:
        for path in run_dir.glob("ensemble_checkpoint_rank*.pt"):
            suffix = path.stem.removeprefix("ensemble_checkpoint_rank")
            if suffix.isdigit():
                ranks.append(int(suffix))
    ranks = sorted(set(ranks))
    if ranks:
        paths = [run_dir / f"ensemble_checkpoint_rank{rank}.pt" for rank in ranks]
        missing = [str(path) for path in paths if not path.exists()]
        if missing:
            raise ProtocolError(
                f"Checkpoint ensemble metadata in {run_dir} references missing files: {missing}"
            )
        return paths
    checkpoint = run_dir / "best_model.pt"
    if not checkpoint.exists():
        raise ProtocolError(f"Missing checkpoint in {run_dir}: {checkpoint}")
    return [checkpoint]


def choose_beta(score_by_beta: dict[float, float]) -> tuple[float, float]:
    """Choose the largest finite validation score, breaking exact ties by lower beta."""
    if not score_by_beta:
        raise ValueError("At least one beta score is required.")
    candidates = []
    for beta, score in score_by_beta.items():
        beta = float(beta)
        score = float(score)
        if not math.isfinite(beta) or not math.isfinite(score):
            raise ValueError(f"Beta and score must be finite, got beta={beta!r}, score={score!r}.")
        candidates.append((beta, score))
    return sorted(candidates, key=lambda item: (-item[1], item[0]))[0]


def select_beta_for_seed(
    result_root: Path,
    seed: int,
    betas: Iterable[float],
    *,
    fixed_beta: float | None = None,
) -> BetaSelection:
    candidates: list[BetaCandidate] = []
    normalized_betas = sorted({float(beta) for beta in betas})
    if not normalized_betas:
        raise ValueError("At least one beta candidate is required.")
    for beta in normalized_betas:
        run_dir = candidate_dir(result_root, seed, beta)
        checkpoint = run_dir / "best_model.pt"
        if not checkpoint.exists():
            raise ProtocolError(f"Missing beta={beta:g} checkpoint for seed {seed}: {checkpoint}")
        score, score_source = validation_checkpoint_score(run_dir)
        candidates.append(
            BetaCandidate(
                seed=int(seed),
                beta=beta,
                score=score,
                run_dir=str(run_dir),
                checkpoint=str(checkpoint),
                score_source=score_source,
            )
        )
    if fixed_beta is None:
        # Descending validation score, then ascending beta. Python sorting is stable and
        # makes the exact-tie policy visible instead of hiding it behind a tolerance.
        selected_beta, _ = choose_beta({item.beta: item.score for item in candidates})
        tie_break = "lower_beta_on_exact_score_tie"
    else:
        selected_beta = float(fixed_beta)
        if selected_beta not in normalized_betas:
            raise ProtocolError(
                f"Fixed beta={selected_beta:g} is absent from candidates {normalized_betas}."
            )
        tie_break = "pre_registered_global_beta_no_per_seed_selection"
    selected = next(item for item in candidates if item.beta == selected_beta)
    selected_ensemble = ensemble_checkpoint_paths(Path(selected.run_dir))
    return BetaSelection(
        seed=int(seed),
        selected_beta=selected.beta,
        selected_score=selected.score,
        selected_run_dir=selected.run_dir,
        selected_checkpoint=selected.checkpoint,
        checkpoint_metric=CHECKPOINT_METRIC,
        tie_break=tie_break,
        candidates=[asdict(item) for item in candidates],
        selected_ensemble_checkpoints=[str(path) for path in selected_ensemble],
        selected_ensemble_size=len(selected_ensemble),
        selected_ensemble_aggregation="arithmetic_mean_softmax_probabilities",
    )


def write_beta_selection(result_root: Path, selection: BetaSelection) -> Path:
    path = selected_beta_path(result_root, selection.seed)
    write_json(path, asdict(selection))
    return path


def replace_cli_option(command: Sequence[str], option: str, value: str) -> list[str]:
    result = [str(part) for part in command]
    try:
        index = result.index(option)
    except ValueError:
        result.extend([option, value])
        return result
    if index + 1 >= len(result) or result[index + 1].startswith("--"):
        raise ProtocolError(f"Malformed saved worker command: {option} has no value.")
    result[index + 1] = value
    return result


def remove_cli_option(command: Sequence[str], option: str, *, takes_value: bool = True) -> list[str]:
    result: list[str] = []
    index = 0
    command = [str(part) for part in command]
    while index < len(command):
        if command[index] == option:
            index += 2 if takes_value else 1
        else:
            result.append(command[index])
            index += 1
    return result


def remove_cli_multi_option(command: Sequence[str], option: str) -> list[str]:
    """Remove an argparse ``nargs='+'`` option and all of its values."""

    result: list[str] = []
    index = 0
    command = [str(part) for part in command]
    while index < len(command):
        if command[index] != option:
            result.append(command[index])
            index += 1
            continue
        index += 1
        while index < len(command) and not command[index].startswith("--"):
            index += 1
    return result


def selection_checkpoint_paths(selection: BetaSelection | dict[str, Any]) -> list[Path]:
    """Read a new ensemble selection or fall back to the historical rank-1 field."""

    data = asdict(selection) if isinstance(selection, BetaSelection) else selection
    ensemble = data.get("selected_ensemble_checkpoints")
    if isinstance(ensemble, list) and ensemble:
        return [Path(str(path)) for path in ensemble]
    checkpoint = data.get("selected_checkpoint")
    if checkpoint in {None, ""}:
        raise ProtocolError("Selection JSON contains no selected checkpoint.")
    return [Path(str(checkpoint))]


def evaluation_command_from_selection(
    selection: BetaSelection | dict[str, Any],
    evaluation_dir: Path,
    *,
    python_exe: str | None = None,
    worker: Path | None = None,
    device: str | None = None,
) -> list[str]:
    data = asdict(selection) if isinstance(selection, BetaSelection) else selection
    candidate = Path(str(data["selected_run_dir"]))
    command_path = candidate / "worker_command.json"
    if not command_path.exists():
        raise ProtocolError(
            f"Cannot reproduce selected evaluation: missing saved training command {command_path}."
        )
    saved = read_json(command_path)
    command = [str(part) for part in saved.get("argv", [])]
    if len(command) < 2:
        raise ProtocolError(f"Invalid saved command in {command_path}.")
    command = remove_cli_option(command, "--evaluation-only-checkpoint")
    command = remove_cli_multi_option(command, "--evaluation-only-checkpoints")
    command = remove_cli_option(command, "--skip-final-test", takes_value=False)
    command = replace_cli_option(command, "--result-dir", str(evaluation_dir))
    command = replace_cli_option(command, "--evaluate-test", "on")
    command = replace_cli_option(command, "--evaluate-test-each-epoch", "off")
    if python_exe is not None:
        command[0] = str(python_exe)
    if worker is not None:
        command[2 if len(command) > 1 and command[1] == "-u" else 1] = str(worker)
    if device is not None:
        command = replace_cli_option(command, "--device", device)
    selected_ensemble = data.get("selected_ensemble_checkpoints")
    checkpoints = selection_checkpoint_paths(data)
    if isinstance(selected_ensemble, list) and selected_ensemble:
        command.extend(["--evaluation-only-checkpoints", *[str(path) for path in checkpoints]])
    else:
        command.extend(["--evaluation-only-checkpoint", str(checkpoints[0])])
    return command


def metrics_has_split(path: Path, split: str) -> bool:
    if not path.exists():
        return False
    try:
        frame = pd.read_csv(path, usecols=["split"])
    except (ValueError, pd.errors.EmptyDataError):
        return False
    return bool((frame["split"].astype(str).str.lower() == split.lower()).any())


BASE_REQUIRED_ARTIFACTS = (
    "args.json",
    "best_model.pt",
    "metrics_summary.csv",
    "model_summary.json",
    "selected_genes.csv",
    "training_log.csv",
)
VMA_REQUIRED_ARTIFACTS = BASE_REQUIRED_ARTIFACTS + (
    "induced_ppi_edges.csv",
    "ppi_graph_manifest.json",
    "volumetric_diagnostics.json",
)


def inspect_run_artifacts(run_dir: Path, *, variant: str, require_test: bool) -> dict[str, Any]:
    required = BASE_REQUIRED_ARTIFACTS if variant == "baseline" else VMA_REQUIRED_ARTIFACTS
    missing = [name for name in required if not (run_dir / name).exists()]
    test_present = metrics_has_split(run_dir / "metrics_summary.csv", "test")
    if require_test and not test_present:
        missing.append("metrics_summary.csv[test rows]")
    return {
        "run_dir": str(run_dir),
        "variant": variant,
        "required": list(required),
        "missing": missing,
        "test_metrics_present": test_present,
        "ok": not missing,
    }


IDENTIFIER_COLUMNS = {"split", "task", "samples", "seed", "beta", "variant"}
PREFERRED_METRICS = ("loss", "accuracy", "macro_f1", "weighted_f1", "balanced_accuracy", "roc_auc")


def _test_metrics_long(path: Path, seed: int, variant: str, beta: float | None) -> pd.DataFrame:
    if not path.exists():
        raise ProtocolError(f"Missing metrics file: {path}")
    frame = pd.read_csv(path)
    if "split" not in frame.columns or "task" not in frame.columns:
        raise ProtocolError(f"{path} must contain split and task columns.")
    frame = frame.loc[frame["split"].astype(str).str.lower() == "test"].copy()
    if frame.empty:
        raise ProtocolError(f"No test rows in {path}.")
    missing_tasks = sorted(set(TASK_NAMES) - set(frame["task"].astype(str)))
    if missing_tasks:
        raise ProtocolError(f"Missing test tasks in {path}: {missing_tasks}")
    numeric = []
    for column in frame.columns:
        if column in IDENTIFIER_COLUMNS:
            continue
        converted = pd.to_numeric(frame[column], errors="coerce")
        if converted.notna().any():
            numeric.append(column)
            frame[column] = converted
    ordered = [metric for metric in PREFERRED_METRICS if metric in numeric]
    ordered.extend(sorted(set(numeric) - set(ordered)))
    long = frame.melt(id_vars=["task"], value_vars=ordered, var_name="metric", value_name="value")
    long.insert(0, "seed", int(seed))
    long["variant"] = variant
    long["beta"] = math.nan if beta is None else float(beta)
    return long


def collect_paired_deltas(result_root: Path, seeds: Iterable[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_frames: list[pd.DataFrame] = []
    selections: list[dict[str, Any]] = []
    for seed in seeds:
        selection_file = selected_beta_path(result_root, seed)
        if not selection_file.exists():
            raise ProtocolError(f"Missing beta selection for seed {seed}: {selection_file}")
        selection = read_json(selection_file)
        beta = float(selection["selected_beta"])
        evaluation_dir = Path(selection.get("evaluation_dir", selected_test_dir(result_root, seed)))
        configured_baseline_evaluation = selection.get("baseline_evaluation_dir")
        if configured_baseline_evaluation:
            baseline_evaluation_dir = Path(str(configured_baseline_evaluation))
        else:
            baseline_evaluation_dir = baseline_test_dir(result_root, seed)
        baseline_metrics = baseline_evaluation_dir / "metrics_summary.csv"
        if not metrics_has_split(baseline_metrics, "test"):
            # Historical protocol roots stored baseline test rows directly in
            # the training directory and had no baseline_test evaluation run.
            legacy_metrics = baseline_dir(result_root, seed) / "metrics_summary.csv"
            if metrics_has_split(legacy_metrics, "test"):
                baseline_metrics = legacy_metrics
        metric_frames.append(
            _test_metrics_long(baseline_metrics, seed, "baseline", None)
        )
        metric_frames.append(
            _test_metrics_long(evaluation_dir / "metrics_summary.csv", seed, "ppi_volumetric", beta)
        )
        selections.append({"seed": int(seed), "selected_beta": beta, "selected_score": selection["selected_score"]})
    metrics = pd.concat(metric_frames, ignore_index=True)
    baseline = metrics.loc[metrics["variant"] == "baseline", ["seed", "task", "metric", "value"]].rename(
        columns={"value": "baseline_value"}
    )
    vma = metrics.loc[metrics["variant"] == "ppi_volumetric", ["seed", "task", "metric", "value", "beta"]].rename(
        columns={"value": "ppi_volumetric_value", "beta": "selected_beta"}
    )
    deltas = baseline.merge(vma, on=["seed", "task", "metric"], how="inner", validate="one_to_one")
    deltas["delta_ppi_volumetric_minus_baseline"] = (
        deltas["ppi_volumetric_value"] - deltas["baseline_value"]
    )
    return deltas.sort_values(["seed", "task", "metric"]).reset_index(drop=True), pd.DataFrame(selections)


def bootstrap_mean_ci(
    values: Sequence[float] | np.ndarray,
    *,
    replicates: int = 10_000,
    seed: int = 20260718,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return math.nan, math.nan
    if replicates <= 0:
        raise ValueError("bootstrap replicates must be positive.")
    if values.size == 1:
        value = float(values[0])
        return value, value
    rng = np.random.default_rng(seed)
    # Chunking bounds peak memory for large user-requested bootstrap counts.
    bootstrap_means = np.empty(replicates, dtype=np.float64)
    chunk = max(1, min(replicates, 100_000 // values.size))
    for start in range(0, replicates, chunk):
        stop = min(start + chunk, replicates)
        indices = rng.integers(0, values.size, size=(stop - start, values.size))
        bootstrap_means[start:stop] = values[indices].mean(axis=1)
    lower, upper = np.quantile(bootstrap_means, [0.025, 0.975])
    return float(lower), float(upper)


def summarize_deltas(
    deltas: pd.DataFrame,
    *,
    replicates: int = 10_000,
    bootstrap_seed: int = 20260718,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    delta_column = "delta_ppi_volumetric_minus_baseline"
    for group_index, ((task, metric), group) in enumerate(deltas.groupby(["task", "metric"], sort=True)):
        values = pd.to_numeric(group[delta_column], errors="coerce").to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        low, high = bootstrap_mean_ci(
            values,
            replicates=replicates,
            seed=bootstrap_seed + group_index,
        )
        rows.append(
            {
                "task": task,
                "metric": metric,
                "n_seeds": int(values.size),
                "delta_mean": float(np.mean(values)) if values.size else math.nan,
                "delta_std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0 if values.size else math.nan,
                "bootstrap_95_ci_low": low,
                "bootstrap_95_ci_high": high,
                "bootstrap_replicates": int(replicates),
                "bootstrap_seed": int(bootstrap_seed + group_index),
            }
        )
    return pd.DataFrame(rows)


def beta_frequencies(selections: pd.DataFrame, candidates: Iterable[float] = BETA_CANDIDATES) -> pd.DataFrame:
    counts = selections["selected_beta"].value_counts() if not selections.empty else pd.Series(dtype=int)
    total = int(len(selections))
    rows = []
    for beta in sorted({float(value) for value in candidates} | {float(value) for value in counts.index}):
        count = int(counts.get(beta, 0))
        rows.append(
            {
                "beta": beta,
                "selected_count": count,
                "total_seeds": total,
                "selected_frequency": float(count / total) if total else math.nan,
                "interpretation": "PPI pair-wise control" if beta == 0.0 else "volumetric penalty candidate",
            }
        )
    return pd.DataFrame(rows)


def write_summary_artifacts(
    result_root: Path,
    seeds: Iterable[int],
    *,
    betas: Iterable[float] = BETA_CANDIDATES,
    replicates: int = 10_000,
    bootstrap_seed: int = 20260718,
) -> dict[str, str]:
    seeds = tuple(int(seed) for seed in seeds)
    betas = tuple(float(beta) for beta in betas)
    deltas, selections = collect_paired_deltas(result_root, seeds)
    summary = summarize_deltas(deltas, replicates=replicates, bootstrap_seed=bootstrap_seed)
    frequencies = beta_frequencies(selections, betas)
    result_root.mkdir(parents=True, exist_ok=True)
    paths = {
        "paired_deltas": str(result_root / "paired_deltas.csv"),
        "paired_summary": str(result_root / "paired_summary.csv"),
        "beta_selections": str(result_root / "beta_selections.csv"),
        "beta_frequencies": str(result_root / "beta_selection_frequencies.csv"),
    }
    deltas.to_csv(paths["paired_deltas"], index=False)
    summary.to_csv(paths["paired_summary"], index=False)
    selections.to_csv(paths["beta_selections"], index=False)
    frequencies.to_csv(paths["beta_frequencies"], index=False)
    write_json(
        result_root / "summary_manifest.json",
        {
            "seeds": [int(seed) for seed in seeds],
            "beta_candidates": [float(beta) for beta in betas],
            "delta_definition": "ppi_volumetric - baseline",
            "bootstrap": {
                "method": "seed-level paired nonparametric bootstrap of the mean",
                "replicates": int(replicates),
                "base_seed": int(bootstrap_seed),
                "percentiles": [2.5, 97.5],
            },
            "artifacts": paths,
        },
    )
    return paths
