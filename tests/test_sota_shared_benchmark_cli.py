from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SOTA_SOURCE = ROOT / "SOTA" / "source"
if str(SOTA_SOURCE) not in sys.path:
    sys.path.insert(0, str(SOTA_SOURCE))

from run_shared_paper_benchmark import (
    _dependency_versions,
    _plans_semantically_compatible,
    _prepare_paper_root,
    _write_or_validate_plan,
    build_command_specs,
    parse_args,
)


def test_torch_provenance_is_kelly_specific() -> None:
    assert "torch" in _dependency_versions("kelly-2023")
    assert "torch" not in _dependency_versions("one2mfusion-2023")


def _args(paper: str, tmp_path: Path):
    return parse_args(
        [
            "--paper",
            paper,
            "--result-root",
            str(tmp_path / "results"),
            "--split-manifest-dir",
            str(tmp_path / "splits"),
            "--python-exe",
            sys.executable,
            "--ctgan-device",
            "cpu",
        ]
    )


def test_four_single_phase_papers_build_three_task_commands(tmp_path: Path) -> None:
    for paper in ("lee-2020", "kelly-2023", "one2mfusion-2023", "diagnostics-2025"):
        specs = build_command_specs(_args(paper, tmp_path))
        assert len(specs) == 3
        assert {spec.task for spec in specs} == {"ad_vs_mci", "ad_vs_ctl", "mci_vs_ctl"}
        for spec in specs:
            command = list(spec.command)
            assert command[command.index("--repeats") + 1] == "10"
            assert command[command.index("--split-manifest-dir") + 1] == str(tmp_path / "splits")
            assert command[command.index("--batch-scenarios") + 1] == "shared_test"
            assert "--skip-existing" in command


def test_paper_matrices_include_selected_branches_without_all_gene_inputs(tmp_path: Path) -> None:
    lee = " ".join(build_command_specs(_args("lee-2020", tmp_path))[0].command)
    for token in ("deg", "vae", "tf_genes", "hub_genes", "cfg_genes", "lr", "l1_lr", "svm", "rf", "dnn"):
        assert token in lee
    assert "--allow-cfg-supplement-proxy" not in lee

    kelly = " ".join(build_command_specs(_args("kelly-2023", tmp_path))[0].command)
    for token in ("knowledge_genes", "vssrfe_lr", "lasso", "vae_latent", "xgboost", "mlp"):
        assert token in kelly
    assert "all_genes" not in kelly
    assert "--include-deep" not in kelly
    assert "--hyperparameter-mode fixed_paper" in kelly
    assert "--vae-backend torch" in kelly
    assert "--vae-device cuda" in kelly
    assert "--disable-fixed-vssrfe-n-genes" not in kelly
    assert "--bayes-iter" not in kelly

    one2m = " ".join(build_command_specs(_args("one2mfusion-2023", tmp_path))[0].command)
    for token in ("cnn", "one2mfusion"):
        assert token in one2m
    assert "fnn" not in one2m

    diagnostics = " ".join(build_command_specs(_args("diagnostics-2025", tmp_path))[0].command)
    for token in ("dl", "svm", "gbm", "rf", "no_smote", "borderline_smote"):
        assert token in diagnostics


def test_hariharan_has_unaugmented_augmented_and_table11_phases_per_task(tmp_path: Path) -> None:
    specs = build_command_specs(_args("hariharan-2026", tmp_path))
    assert len(specs) == 9
    for task in ("ad_vs_mci", "ad_vs_ctl", "mci_vs_ctl"):
        task_specs = [spec for spec in specs if spec.task == task]
        assert {spec.phase for spec in task_specs} == {"unaugmented_grid", "paper_augmented_grid", "table11_dnn_k500"}
        primary = " ".join(next(spec.command for spec in task_specs if spec.phase == "unaugmented_grid"))
        for token in ("chi2", "anova", "rfe", "elasticnet", "svm", "rf", "adaboost", "xgboost", "dnn", "cnn", "ctgan"):
            if token != "ctgan":
                assert token in primary
        assert "all_genes" not in primary
        assert "ctgan" not in primary
        augmented = " ".join(next(spec.command for spec in task_specs if spec.phase == "paper_augmented_grid"))
        assert "ctgan" in augmented
        assert "all_genes" not in augmented
        table11 = list(next(spec.command for spec in task_specs if spec.phase == "table11_dnn_k500"))
        assert "lasso" in table11
        assert "rf_importance" in table11
        assert "ctgan" not in table11
        assert table11.count("500") >= 6


def test_resume_refuses_unplanned_nonempty_output_and_changed_plan(tmp_path: Path) -> None:
    result_root = tmp_path / "results"
    paper_root = result_root / "kelly-2023"
    paper_root.mkdir(parents=True)
    (paper_root / "stale.txt").write_text("old", encoding="utf-8")
    with pytest.raises(RuntimeError, match="non-empty output without an orchestration plan"):
        _prepare_paper_root(paper_root, result_root, overwrite=False)

    _prepare_paper_root(paper_root, result_root, overwrite=True)
    plan_path = paper_root / "orchestration_plan.json"
    first = {"plan_sha256": "abc"}
    _write_or_validate_plan(plan_path, first)
    _write_or_validate_plan(plan_path, first)
    with pytest.raises(RuntimeError, match="differs from this invocation"):
        _write_or_validate_plan(plan_path, {"plan_sha256": "def"})


def test_resume_accepts_only_wrapper_fingerprint_metadata_changes(tmp_path: Path) -> None:
    common = {
        "paper": "lee-2020",
        "tasks": ["ad_vs_mci"],
        "repeats": 1,
        "seeds": [101],
        "split_manifest_dir": "splits",
        "commands": [{"command": ("python", "runner.py", "--seed", "101")}],
        "dependency_versions": {"python": "3.12"},
    }
    old = {
        **common,
        "plan_sha256": "old",
        "protocol": "old description",
        "file_fingerprints": [
            {"path": "SOTA\\source\\run_shared_paper_benchmark.py", "bytes": 10, "sha256": "old-wrapper"},
            {"path": "SOTA\\source\\nature-2020\\run_experiments.py", "bytes": 20, "sha256": "paper-code"},
        ],
    }
    current = {
        **common,
        "commands": [{"command": ["python", "runner.py", "--seed", "101"]}],
        "plan_sha256": "new",
        "protocol": "new description",
        "file_fingerprints": [
            {"path": "SOTA\\source\\nature-2020\\run_experiments.py", "bytes": 20, "sha256": "paper-code"},
        ],
    }
    assert _plans_semantically_compatible(old, current)

    plan_path = tmp_path / "orchestration_plan.json"
    plan_path.write_text(json.dumps(old), encoding="utf-8")
    _write_or_validate_plan(plan_path, current)
    assert json.loads(plan_path.read_text(encoding="utf-8"))["plan_sha256"] == "new"

    changed_command = {**current, "plan_sha256": "changed", "commands": [{"command": ["different"]}]}
    assert not _plans_semantically_compatible(old, changed_command)
