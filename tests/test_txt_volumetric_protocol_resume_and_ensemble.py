from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pandas as pd
import pytest

from experiments.scripts.paper_comparison.txt_volumetric import common
from experiments.scripts.paper_comparison.txt_volumetric import run_paired_protocol as protocol


def _write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _training_summary(path: Path, *, best: float, ensemble: float, size: int = 1) -> None:
    common.write_json(
        path / "model_summary.json",
        {
            "training_summary": {
                "checkpoint_metric": common.CHECKPOINT_METRIC,
                "best_checkpoint_value": best,
                "ensemble_validation_score": ensemble,
                "ensemble_validation_score_metric": common.CHECKPOINT_METRIC,
                "checkpoint_ensemble": [
                    {"rank": rank, "epoch": rank, "checkpoint_value": best}
                    for rank in range(1, size + 1)
                ],
            }
        },
    )


def _saved_training_command(run_dir: Path) -> list[str]:
    command = [
        "python",
        "-u",
        "worker.py",
        "--x-file",
        "X.csv",
        "--y-file",
        "y.csv",
        "--split-file",
        "split.csv",
        "--result-dir",
        str(run_dir),
        "--evaluate-test",
        "off",
    ]
    common.save_worker_command(
        run_dir,
        command,
        kind="candidate",
        seed=101,
        beta=1.0,
    )
    return command


def test_validation_score_prefers_explicit_ensemble_score(tmp_path: Path) -> None:
    _training_summary(tmp_path, best=0.91, ensemble=0.73)

    score, source = common.validation_checkpoint_score(tmp_path)

    assert score == pytest.approx(0.73)
    assert source.endswith("model_summary.json")


def test_beta_selection_keeps_rank1_and_records_probability_ensemble(tmp_path: Path) -> None:
    for beta, score, size in ((0.0, 0.4, 1), (1.0, 0.8, 3)):
        run_dir = common.candidate_dir(tmp_path, 101, beta)
        _write(run_dir / "best_model.pt", f"best-{beta}")
        for rank in range(1, size + 1):
            _write(run_dir / f"ensemble_checkpoint_rank{rank}.pt", f"{beta}-{rank}")
        _training_summary(run_dir, best=score + 0.05, ensemble=score, size=size)

    selection = common.select_beta_for_seed(tmp_path, 101, (0.0, 1.0))

    assert selection.selected_beta == 1.0
    assert selection.selected_checkpoint.endswith("best_model.pt")
    assert selection.selected_ensemble_size == 3
    assert (
        selection.selected_ensemble_aggregation
        == "arithmetic_mean_softmax_probabilities"
    )
    assert [Path(path).name for path in selection.selected_ensemble_checkpoints] == [
        "ensemble_checkpoint_rank1.pt",
        "ensemble_checkpoint_rank2.pt",
        "ensemble_checkpoint_rank3.pt",
    ]


def test_fixed_beta_policy_never_uses_per_seed_argmax(tmp_path: Path) -> None:
    for beta, score in ((0.0, 0.9), (1.0, 0.4)):
        run_dir = common.candidate_dir(tmp_path, 201, beta)
        _write(run_dir / "best_model.pt", f"best-{beta}")
        _write(run_dir / "ensemble_checkpoint_rank1.pt", f"ensemble-{beta}")
        _training_summary(run_dir, best=score, ensemble=score)

    selection = common.select_beta_for_seed(
        tmp_path,
        201,
        (0.0, 1.0),
        fixed_beta=1.0,
    )

    assert selection.selected_beta == 1.0
    assert selection.tie_break == "pre_registered_global_beta_no_per_seed_selection"
    assert [candidate["beta"] for candidate in selection.candidates] == [0.0, 1.0]


def test_auto_beta_policy_freezes_primary_for_complete_v2_profile() -> None:
    args = Namespace(
        beta_policy="auto",
        primary_beta=1.0,
        volumetric_volume_mode="l2",
        volumetric_message_mode="expression_contrast",
        volumetric_output_norm="rms",
        volumetric_gate_mode="per_head",
        volumetric_backbone_gradient_mode="detached",
    )

    policy, beta = protocol.resolve_beta_policy(args, {"betas": [0.0, 1.0]})

    assert policy == "global_fixed"
    assert beta == 1.0


def test_auto_beta_policy_preserves_historical_multi_beta_argmax() -> None:
    args = Namespace(
        beta_policy="auto",
        primary_beta=1.0,
        volumetric_volume_mode="raw",
        volumetric_message_mode="legacy",
        volumetric_output_norm="none",
        volumetric_gate_mode="scalar",
        volumetric_backbone_gradient_mode="coupled",
    )

    policy, beta = protocol.resolve_beta_policy(args, {"betas": [0.0, 1.0]})

    assert policy == "per_seed_validation_argmax"
    assert beta is None


def test_paired_protocol_forwards_no_tupe_to_baseline_and_vma() -> None:
    args = protocol.build_parser().parse_args(["--tupe-mode", "off"])
    effective = protocol.effective_protocol(args)
    split = Path("split.csv")
    baseline = protocol.build_worker_command(
        args,
        effective,
        seed=101,
        split_file=split,
        result_dir=Path("baseline"),
        model_variant="baseline",
        beta=None,
        evaluate_test="off",
    )
    vma = protocol.build_worker_command(
        args,
        effective,
        seed=101,
        split_file=split,
        result_dir=Path("vma"),
        model_variant="ppi_volumetric",
        beta=1.0,
        evaluate_test="off",
    )

    for command in (baseline, vma):
        index = command.index("--tupe-mode")
        assert command[index + 1] == "off"


def test_evaluation_command_prefers_new_checkpoint_list_and_keeps_legacy_fallback(
    tmp_path: Path,
) -> None:
    candidate = tmp_path / "candidate"
    _saved_training_command(candidate)
    checkpoints = [candidate / "rank1.pt", candidate / "rank2.pt"]

    ensemble_command = common.evaluation_command_from_selection(
        {
            "selected_run_dir": str(candidate),
            "selected_checkpoint": str(candidate / "best_model.pt"),
            "selected_ensemble_checkpoints": [str(path) for path in checkpoints],
        },
        tmp_path / "ensemble_test",
    )
    legacy_command = common.evaluation_command_from_selection(
        {
            "selected_run_dir": str(candidate),
            "selected_checkpoint": str(candidate / "best_model.pt"),
        },
        tmp_path / "legacy_test",
    )

    list_index = ensemble_command.index("--evaluation-only-checkpoints")
    assert ensemble_command[list_index + 1 : list_index + 3] == [str(path) for path in checkpoints]
    assert "--evaluation-only-checkpoint" not in ensemble_command
    assert "--evaluation-only-checkpoints" not in legacy_command
    checkpoint_index = legacy_command.index("--evaluation-only-checkpoint")
    assert legacy_command[checkpoint_index + 1] == str(candidate / "best_model.pt")


def test_resume_fingerprint_is_path_neutral_but_content_sensitive(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    roles = {"x_file": "x", "y_file": "y", "split_file": "split", "ppi_edge_file": "ppi"}
    first_inputs = {role: _write(first / f"{name}.csv", name) for role, name in roles.items()}
    second_inputs = {role: _write(second / f"{name}.csv", name) for role, name in roles.items()}
    first_command = [
        "python",
        "worker.py",
        "--x-file",
        str(first_inputs["x_file"]),
        "--y-file",
        str(first_inputs["y_file"]),
        "--split-file",
        str(first_inputs["split_file"]),
        "--ppi-edge-file",
        str(first_inputs["ppi_edge_file"]),
        "--result-dir",
        str(first / "run"),
        "--seed",
        "101",
    ]
    second_command = [
        "different-python",
        "different-worker.py",
        "--x-file",
        str(second_inputs["x_file"]),
        "--y-file",
        str(second_inputs["y_file"]),
        "--split-file",
        str(second_inputs["split_file"]),
        "--ppi-edge-file",
        str(second_inputs["ppi_edge_file"]),
        "--result-dir",
        str(second / "run"),
        "--seed",
        "101",
    ]

    first_fingerprint = common.build_resume_fingerprint(
        first_command, kind="training", input_files=first_inputs
    )
    second_fingerprint = common.build_resume_fingerprint(
        second_command, kind="training", input_files=second_inputs
    )
    assert first_fingerprint["semantic_sha256"] == second_fingerprint["semantic_sha256"]

    _write(second_inputs["split_file"], "changed split")
    changed = common.build_resume_fingerprint(
        second_command, kind="training", input_files=second_inputs
    )
    assert changed["semantic_sha256"] != first_fingerprint["semantic_sha256"]


def test_skip_existing_mismatch_fails_before_overwriting_command(tmp_path: Path) -> None:
    inputs = {
        "x_file": _write(tmp_path / "X.csv", "x"),
        "y_file": _write(tmp_path / "y.csv", "y"),
        "split_file": _write(tmp_path / "split.csv", "split"),
        "ppi_edge_file": _write(tmp_path / "ppi.csv", "ppi"),
    }
    args = Namespace(
        x_file=inputs["x_file"],
        y_file=inputs["y_file"],
        ppi_edge_file=inputs["ppi_edge_file"],
        skip_existing=True,
        dry_run=True,
    )
    run_dir = tmp_path / "run"
    old_command = ["python", "worker.py", "--result-dir", str(run_dir), "--seed", "101"]
    old_fingerprint = common.build_resume_fingerprint(
        old_command, kind="training", input_files=inputs
    )
    common.save_worker_command(
        run_dir,
        old_command,
        kind="baseline_train",
        seed=101,
        beta=None,
        resume_fingerprint=old_fingerprint,
    )
    before = (run_dir / "worker_command.json").read_text(encoding="utf-8")
    changed_command = ["python", "worker.py", "--result-dir", str(run_dir), "--seed", "202"]

    with pytest.raises(common.ProtocolError, match="fingerprint mismatch"):
        protocol.run_or_reuse(
            args,
            run_dir=run_dir,
            command=changed_command,
            kind="baseline_train",
            seed=202,
            beta=None,
            complete=True,
            split_file=inputs["split_file"],
        )

    assert (run_dir / "worker_command.json").read_text(encoding="utf-8") == before


def test_skip_existing_does_not_overwrite_a_mismatched_split(tmp_path: Path) -> None:
    labels = pd.DataFrame(
        {
            "sample_id": [f"sample-{index}" for index in range(30)],
            "label": ["AD"] * 10 + ["MCI"] * 10 + ["CTL"] * 10,
        }
    )
    y_file = tmp_path / "y.csv"
    labels.to_csv(y_file, index=False)
    args = Namespace(result_root=tmp_path, y_file=y_file, skip_existing=False)
    effective = {"seeds": [101]}
    paths = protocol.ensure_splits(args, effective)
    split_path = paths[101]
    split_path.write_text("sample_id,split\nsample-0,test\n", encoding="utf-8")
    before = split_path.read_text(encoding="utf-8")
    args.skip_existing = True

    with pytest.raises(common.ProtocolError, match="stored split differs"):
        protocol.ensure_splits(args, effective)

    assert split_path.read_text(encoding="utf-8") == before


def _metrics(path: Path, *, value: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "split": "test",
                "task": task,
                "samples": 10,
                "loss": 0.5,
                "accuracy": value,
                "roc_auc": value,
            }
            for task in common.TASK_NAMES
        ]
    ).to_csv(path, index=False)


def test_summarizer_prefers_separate_baseline_test_and_falls_back_to_legacy(
    tmp_path: Path,
) -> None:
    seed = 101
    selected_dir = common.selected_test_dir(tmp_path, seed)
    separate_baseline_dir = common.baseline_test_dir(tmp_path, seed)
    _metrics(common.baseline_dir(tmp_path, seed) / "metrics_summary.csv", value=0.1)
    _metrics(separate_baseline_dir / "metrics_summary.csv", value=0.6)
    _metrics(selected_dir / "metrics_summary.csv", value=0.8)
    common.write_json(
        common.selected_beta_path(tmp_path, seed),
        {
            "selected_beta": 1.0,
            "selected_score": 0.7,
            "evaluation_dir": str(selected_dir),
            "baseline_evaluation_dir": str(separate_baseline_dir),
        },
    )

    separate, _ = common.collect_paired_deltas(tmp_path, [seed])
    assert set(separate["baseline_value"]) == {0.5, 0.6}

    (separate_baseline_dir / "metrics_summary.csv").unlink()
    legacy, _ = common.collect_paired_deltas(tmp_path, [seed])
    assert set(legacy["baseline_value"]) == {0.1, 0.5}


def test_select_evaluate_schedules_frozen_baseline_before_selected_vma(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seed = 101
    split_file = _write(tmp_path / "splits" / f"seed_{seed}.csv", "sample_id,split\n")
    baseline_run = common.baseline_dir(tmp_path, seed)
    candidate_run = common.candidate_dir(tmp_path, seed, 1.0)
    for run_dir, beta, score in ((baseline_run, None, 0.5), (candidate_run, 1.0, 0.7)):
        _write(run_dir / "best_model.pt", f"best-{beta}")
        _write(run_dir / "ensemble_checkpoint_rank1.pt", f"ensemble-{beta}")
        _training_summary(run_dir, best=score, ensemble=score, size=1)
        command = [
            "python",
            "-u",
            "worker.py",
            "--x-file",
            str(tmp_path / "X.csv"),
            "--y-file",
            str(tmp_path / "y.csv"),
            "--split-file",
            str(split_file),
            "--result-dir",
            str(run_dir),
            "--evaluate-test",
            "off",
            "--model-variant",
            "baseline" if beta is None else "ppi_volumetric",
        ]
        common.save_worker_command(
            run_dir,
            command,
            kind="training",
            seed=seed,
            beta=beta,
        )

    scheduled: list[dict] = []

    def fake_run_or_reuse(args, **kwargs):
        scheduled.append(kwargs)
        return False

    monkeypatch.setattr(protocol, "run_or_reuse", fake_run_or_reuse)
    args = Namespace(
        result_root=tmp_path,
        python_exe="python",
        worker=Path("worker.py"),
    )

    protocol.select_evaluate_stage(
        args,
        {"seeds": [seed], "betas": [1.0], "device": "cpu"},
    )

    assert [item["kind"] for item in scheduled] == [
        "baseline_checkpoint_test_evaluation",
        "selected_checkpoint_test_evaluation",
    ]
    assert scheduled[0]["run_dir"] == common.baseline_test_dir(tmp_path, seed)
    assert scheduled[1]["run_dir"] == common.selected_test_dir(tmp_path, seed)
    selection = common.read_json(common.selected_beta_path(tmp_path, seed))
    assert selection["baseline_evaluation_dir"] == str(common.baseline_test_dir(tmp_path, seed))
    assert selection["selected_ensemble_size"] == 1


def test_smoke_training_baseline_test_requirement_tracks_cli_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, bool]] = []

    def fake_inspect(run_dir: Path, *, variant: str, require_test: bool):
        calls.append((variant, require_test))
        return {"ok": True, "run_dir": str(run_dir), "variant": variant, "missing": []}

    monkeypatch.setattr(protocol, "inspect_run_artifacts", fake_inspect)
    args = Namespace(result_root=tmp_path, baseline_evaluate_test="off")
    protocol.smoke_artifact_report(args, {"seeds": [101], "betas": []})

    assert calls[0] == ("baseline", False)
    assert len(calls) == 1
