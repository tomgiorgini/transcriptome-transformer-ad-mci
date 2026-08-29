#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.scripts.paper_comparison import run_txt_multitask_cv_and_seeds as cv_runner
from experiments.scripts.paper_comparison import run_txt_multitask_final_grid as final_grid


@dataclass(frozen=True)
class FollowupJob:
    rank: int
    source_job_name: str
    source_arch: str
    source_split: str
    source_primary_roc_auc_mean: float
    source_primary_macro_f1_mean: float
    result_root: Path
    command: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the top TxT multitask CV grid configurations on repeated seed splits. "
            "This reads grid_ranked_summary.csv plus grid_jobs.csv and reruns selected jobs with --run-mode seeds."
        )
    )
    parser.add_argument("--grid-root", type=Path, default=ROOT / "results" / "paper_comparison" / "txt_multitask_final_grid")
    parser.add_argument("--ranked-summary", type=Path, default=None)
    parser.add_argument("--result-root", type=Path, default=None)
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--evaluation", default="cv5")
    parser.add_argument("--python-exe", default=None)
    parser.add_argument("--device", choices=["cuda", "cpu", "mps"], default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--early-stopping-patience", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else (ROOT / path).resolve()


def finite_float(value: object) -> float:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(numeric) if pd.notna(numeric) else float("nan")


def load_grid_config(grid_root: Path) -> dict[str, Any]:
    config_path = grid_root / "grid_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Grid config not found: {config_path}")
    return json.loads(config_path.read_text(encoding="utf-8"))


def load_ranked_summary(args: argparse.Namespace) -> pd.DataFrame:
    ranked_path = resolve(args.ranked_summary) if args.ranked_summary is not None else resolve(args.grid_root) / "grid_ranked_summary.csv"
    if not ranked_path.exists():
        final_grid.write_ranked_summary(resolve(args.grid_root))
    if not ranked_path.exists():
        raise FileNotFoundError(f"Ranked grid summary not found: {ranked_path}")
    ranked = pd.read_csv(ranked_path)
    if ranked.empty:
        raise ValueError(f"Ranked grid summary is empty: {ranked_path}")
    return ranked


def select_top_ranked_rows(ranked: pd.DataFrame, top_n: int, split: str, evaluation: str) -> pd.DataFrame:
    if top_n <= 0:
        raise ValueError("--top-n must be positive.")
    required = {"job_name", "evaluation", "arch", "split", "primary_roc_auc_mean", "primary_macro_f1_mean"}
    missing = sorted(required.difference(ranked.columns))
    if missing:
        raise ValueError(f"Ranked summary is missing columns: {', '.join(missing)}")
    selected = ranked[(ranked["split"].astype(str) == split) & (ranked["evaluation"].astype(str) == evaluation)].copy()
    if selected.empty:
        raise ValueError(f"No ranked rows found for split={split!r}, evaluation={evaluation!r}.")
    selected = selected.sort_values(
        ["primary_roc_auc_mean", "primary_macro_f1_mean", "auxiliary_roc_auc_mean", "job_name", "arch"],
        ascending=[False, False, False, True, True],
    )
    return selected.head(top_n).reset_index(drop=True)


def load_grid_jobs(grid_root: Path) -> pd.DataFrame:
    jobs_path = grid_root / "grid_jobs.csv"
    if not jobs_path.exists():
        raise FileNotFoundError(f"Grid job manifest not found: {jobs_path}")
    jobs = pd.read_csv(jobs_path)
    if jobs.empty:
        raise ValueError(f"Grid job manifest is empty: {jobs_path}")
    return jobs


def maybe_path(value: object) -> Path | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    return resolve(Path(text))


def grid_job_from_row(row: pd.Series, result_root: Path) -> final_grid.GridJob:
    def text_or_default(name: str, default: str) -> str:
        value = row.get(name, default)
        return default if value is None or pd.isna(value) or not str(value).strip() else str(value)

    raw_weights = row.get("task_loss_weights", "(1.0, 1.0, 1.0)")
    parsed_weights = ast.literal_eval(str(raw_weights)) if isinstance(raw_weights, str) else raw_weights

    return final_grid.GridJob(
        job_name=str(row["job_name"]),
        result_root=result_root,
        embedding_source=str(row["embedding_source"]),
        embed_file=maybe_path(row.get("embed_file")),
        embedding_gene_policy=str(row["embedding_gene_policy"]),
        gene_selection=str(row["gene_selection"]),
        max_genes=int(row["max_genes"]),
        augmentation=str(row["augmentation"]),
        d_model=int(row["d_model"]),
        d_ff=int(row["d_ff"]),
        dropout=float(row["dropout"]),
        ad_mci_gene_fraction=float(row["ad_mci_gene_fraction"]),
        encoder_sharing=text_or_default("encoder_sharing", "shared"),
        head_norm=text_or_default("head_norm", "batch"),
        embedding_rescale=text_or_default("embedding_rescale", "none"),
        task_loss_weights=tuple(float(value) for value in parsed_weights),
        checkpoint_metric=text_or_default("checkpoint_metric", "val_primary_auc_minus_025_loss"),
        pooling_mode=text_or_default("pooling_mode", "average"),
        attention_pooling_hidden_dim=int(row.get("attention_pooling_hidden_dim", 16)),
        primary_adapter_dim=int(row.get("primary_adapter_dim", 0)),
        expression_residual=text_or_default("expression_residual", "none"),
        checkpoint_ensemble_size=int(row.get("checkpoint_ensemble_size", 1)),
        ppi_integration=text_or_default("ppi_integration", "direct"),
        pca_neighbor_target_count=int(row.get("pca_neighbor_target_count", 0)),
        train_sampling=text_or_default("train_sampling", "random"),
    )


def followup_args_from_config(config: dict[str, Any], cli_args: argparse.Namespace, job: final_grid.GridJob, architecture_keys: list[str]) -> argparse.Namespace:
    def config_value(name: str, default: Any) -> Any:
        return config.get(name, default)

    return argparse.Namespace(
        python_exe=cli_args.python_exe or str(config_value("python_exe", sys.executable)),
        x_file=resolve(Path(config_value("x_file", final_grid.DEFAULT_SHARED_DIR / "X.csv"))),
        y_file=resolve(Path(config_value("y_file", final_grid.DEFAULT_SHARED_DIR / "y.csv"))),
        run_mode="seeds",
        architectures=architecture_keys,
        device=cli_args.device or str(config_value("device", "cuda")),
        batch_size=int(config_value("batch_size", 16)),
        epochs=int(cli_args.epochs if cli_args.epochs is not None else config_value("epochs", 100)),
        early_stopping_patience=int(
            cli_args.early_stopping_patience
            if cli_args.early_stopping_patience is not None
            else config_value("early_stopping_patience", 30)
        ),
        lr=float(config_value("lr", 1e-4)),
        lr_embedding=config_value("lr_embedding", None),
        freeze_embedding_epochs=int(config_value("freeze_embedding_epochs", 0)),
        weight_decay=float(config_value("weight_decay", 1e-4)),
        val_loss_stop_threshold=config_value("val_loss_stop_threshold", None),
        val_loss_stop_patience=int(config_value("val_loss_stop_patience", 0)),
        task_specific_pooling=str(config_value("task_specific_pooling", "off")),
        class_weighting=str(config_value("class_weighting", "off")),
        checkpoint_metric=str(config_value("checkpoint_metric", "val_primary_auc_minus_025_loss")),
        evaluate_test_each_epoch=str(config_value("evaluate_test_each_epoch", "off")),
        smote_k_neighbors=int(config_value("smote_k_neighbors", 5)),
        smote_m_neighbors=int(config_value("smote_m_neighbors", 10)),
        smote_kind=str(config_value("smote_kind", "borderline-1")),
        ctgan_epochs=int(config_value("ctgan_epochs", 100)),
        ctgan_batch_size=int(config_value("ctgan_batch_size", 128)),
        augmentation_target_multiplier=float(config_value("augmentation_target_multiplier", 2.0)),
        gan_epochs=int(config_value("gan_epochs", 100)),
        gan_batch_size=int(config_value("gan_batch_size", 64)),
        gan_latent_dim=int(config_value("gan_latent_dim", 64)),
        gan_learning_rate=float(config_value("gan_learning_rate", 0.001)),
        gan_target_multiplier=float(config_value("gan_target_multiplier", 1.5)),
        gan_sampling_strategy=str(config_value("gan_sampling_strategy", "balanced")),
        embedding_init_scale=float(config_value("embedding_init_scale", 0.02)),
        attention_pooling_dropout=float(config_value("attention_pooling_dropout", 0.1)),
        checkpoint_ensemble_min_gap=int(config_value("checkpoint_ensemble_min_gap", 3)),
        mask_aware_heads=str(config_value("mask_aware_heads", "off")),
        gradient_strategy=str(config_value("gradient_strategy", "weighted_sum")),
        gradient_diagnostics=str(config_value("gradient_diagnostics", "off")),
        gradient_diagnostic_interval=int(config_value("gradient_diagnostic_interval", 1)),
        pca_neighbor_components=int(config_value("pca_neighbor_components", 50)),
        pca_neighbor_k=int(config_value("pca_neighbor_k", 5)),
        pca_neighbor_gap_fraction=float(config_value("pca_neighbor_gap_fraction", 0.5)),
        ppi_gate_init=float(config_value("ppi_gate_init", 0.0)),
        seeds=[int(seed) for seed in (cli_args.seeds if cli_args.seeds is not None else config_value("seeds", [101, 102, 103, 104, 105, 106, 107, 108, 109, 110]))],
        seed_train_ratio=float(config_value("seed_train_ratio", 0.70)),
        seed_val_ratio=float(config_value("seed_val_ratio", 0.10)),
        seed_test_ratio=float(config_value("seed_test_ratio", 0.20)),
        skip_existing=bool(cli_args.skip_existing or config_value("skip_existing", False)),
        smoke=bool(cli_args.smoke),
        ppi_embedding_template=Path(config_value("ppi_embedding_template", final_grid.DEFAULT_PPI_TEMPLATE)),
        ppi_gene_policy=job.embedding_gene_policy,
        candidate_gene_file=(
            resolve(Path(config_value("candidate_gene_file", "")))
            if str(config_value("candidate_gene_file", "")).strip()
            else None
        ),
    )


def architecture_keys_for_summary_arch(config: dict[str, Any], job: final_grid.GridJob, source_arch: str) -> list[str]:
    configured = [str(arch) for arch in config.get("architectures", ["1l2h"])]
    matched: list[str] = []
    proxy = argparse.Namespace(
        d_model=job.d_model,
        d_ff=job.d_ff,
        dropout=job.dropout,
        batch_size=int(config.get("batch_size", 16)),
        task_specific_pooling="off",
        augmentation=job.augmentation,
        embed_file=job.embed_file,
        encoder_sharing=job.encoder_sharing,
        head_norm=job.head_norm,
        embedding_rescale=job.embedding_rescale,
        task_loss_weights=list(job.task_loss_weights),
        pooling_mode=job.pooling_mode,
        attention_pooling_hidden_dim=job.attention_pooling_hidden_dim,
        primary_adapter_dim=job.primary_adapter_dim,
        expression_residual=job.expression_residual,
        checkpoint_ensemble_size=job.checkpoint_ensemble_size,
    )
    for arch_key in configured:
        if arch_key not in cv_runner.ARCHITECTURES:
            continue
        generated_name = cv_runner.architecture_run_name(proxy, arch_key)
        legacy_name = generated_name.rsplit("_enc", 1)[0]
        if source_arch in {generated_name, legacy_name}:
            matched.append(arch_key)
    return matched or configured


def build_followup_jobs(args: argparse.Namespace) -> list[FollowupJob]:
    grid_root = resolve(args.grid_root)
    result_root = resolve(args.result_root) if args.result_root is not None else grid_root / "top_seed_validation"
    config = load_grid_config(grid_root)
    ranked = load_ranked_summary(args)
    selected_rows = select_top_ranked_rows(ranked, args.top_n, args.split, args.evaluation)
    grid_jobs = load_grid_jobs(grid_root)
    job_rows = {str(row["job_name"]): row for _, row in grid_jobs.dropna(subset=["job_name"]).iterrows()}

    jobs: list[FollowupJob] = []
    seen: set[tuple[str, str]] = set()
    for rank, row in enumerate(selected_rows.to_dict(orient="records"), start=1):
        job_name = str(row["job_name"])
        if job_name not in job_rows:
            raise ValueError(f"Ranked job {job_name!r} is missing from grid_jobs.csv.")
        source_arch = str(row["arch"])
        dedupe_key = (job_name, source_arch)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        job_result_root = result_root / job_name
        grid_job = grid_job_from_row(job_rows[job_name], job_result_root)
        arch_keys = architecture_keys_for_summary_arch(config, grid_job, source_arch)
        follow_args = followup_args_from_config(config, args, grid_job, arch_keys)
        command = final_grid.build_command(follow_args, grid_job)
        jobs.append(
            FollowupJob(
                rank=rank,
                source_job_name=job_name,
                source_arch=source_arch,
                source_split=str(row["split"]),
                source_primary_roc_auc_mean=finite_float(row.get("primary_roc_auc_mean")),
                source_primary_macro_f1_mean=finite_float(row.get("primary_macro_f1_mean")),
                result_root=job_result_root,
                command=command,
            )
        )
    return jobs


def write_followup_manifest(result_root: Path, jobs: list[FollowupJob]) -> Path:
    result_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for job in jobs:
        row = asdict(job)
        row["result_root"] = str(job.result_root)
        row["command"] = " ".join(job.command)
        rows.append(row)
    path = result_root / "seed_followup_jobs.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


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


def main() -> None:
    args = parse_args()
    result_root = resolve(args.result_root) if args.result_root is not None else resolve(args.grid_root) / "top_seed_validation"
    jobs = build_followup_jobs(args)
    manifest_path = write_followup_manifest(result_root, jobs)
    if args.dry_run:
        print(f"Dry run complete. Planned seed follow-up jobs: {len(jobs)}. Manifest: {manifest_path}")
        for job in jobs:
            print(f"[rank {job.rank}] {job.source_job_name}: {' '.join(job.command)}")
        return

    failures = []
    for index, job in enumerate(jobs, start=1):
        print("\n" + "=" * 80, flush=True)
        print(f"Seed follow-up job {index}/{len(jobs)} | rank={job.rank} | source={job.source_job_name}", flush=True)
        print("=" * 80, flush=True)
        return_code = run_command(job.command, job.result_root / "seed_followup.log")
        if return_code != 0:
            failures.append({"source_job_name": job.source_job_name, "return_code": return_code})
            pd.DataFrame(failures).to_csv(result_root / "seed_followup_failures.csv", index=False)
            raise RuntimeError(f"Seed follow-up failed: {job.source_job_name}. See {job.result_root / 'seed_followup.log'}")

    final_grid.write_ranked_summary(result_root)
    print(f"Seed follow-up complete. Outputs: {result_root}", flush=True)


if __name__ == "__main__":
    main()
