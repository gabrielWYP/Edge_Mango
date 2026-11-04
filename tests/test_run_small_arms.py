"""The plan of the four small arms: the protocol, the modalities and the order."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import run_small_arms as rs


def _args(tmp_path: Path, *extra: str):
    (tmp_path / "folds_lot").mkdir(exist_ok=True)
    stats = {"nir_clahe": {"roi": {"mean": 0.31, "std": 0.15}}}
    meta = {"folds": {str(k): {"nir_stats": stats} for k in range(5)}}
    (tmp_path / "folds_lot" / "folds_meta.json").write_text(json.dumps(meta))
    args = rs.parse_args(["--dataset", str(tmp_path), "--weights", "w.pt", *extra])
    args.nir_as_rgb_dir = tmp_path / "links"
    return args


def _train(args, arm: str) -> list[str]:
    return [str(x) for x in rs.plan_arm(args, arm, 0, 42)[0][1]]


def test_only_the_nir_student_reads_nir_in_its_rgb_slot(tmp_path) -> None:
    args = _args(tmp_path)
    assert f"rgb_dir={tmp_path / 'links'}" in _train(args, "est_nir")
    for arm in ("est_rgb", "kd_nir", "kd_rgb"):
        assert f"rgb_dir={tmp_path / 'rgb'}" in _train(args, arm)


@pytest.mark.parametrize("arm, teacher, reads", [("kd_nir", "est_nir", "nir"), ("kd_rgb", "est_rgb", "rgb")])
def test_a_kd_arm_distills_from_its_teacher_of_the_same_fold_and_seed(tmp_path, arm, teacher, reads) -> None:
    command = _train(_args(tmp_path), arm)
    assert "src.training.kd_train" in command and f"teacher_input={reads}" in command
    assert "teacher_kind=student" in command
    assert any(x.startswith("teacher_checkpoint=") and f"{teacher}/fold0/seed42" in x for x in command)


def test_every_arm_shares_the_protocol_and_is_evaluated_on_val_and_test(tmp_path) -> None:
    args = _args(tmp_path, "--epochs", "50", "--variant", "nano")
    for arm in rs.ARMS:
        steps = rs.plan_arm(args, arm, 0, 42)
        train = [str(x) for x in steps[0][1]]
        for flag in ("epochs=50", "seed=42", "patience=1000", "student_variant=nano", "student_pretrained_path=w.pt"):
            assert flag in train
        assert [s[0].rsplit("/", 1)[1] for s in steps] == [
            "train", "eval_val", "metrics_val", "eval_test", "metrics_test",
        ]


def test_the_evaluation_uses_the_arm_modality_and_variant(tmp_path) -> None:
    args = _args(tmp_path, "--variant", "nano")
    evaluation = [str(x) for x in rs.plan_arm(args, "est_nir", 0, 42)[1][1]]
    assert f"rgb_dir={tmp_path / 'links'}" in evaluation
    assert "student_variant=nano" in evaluation and "--model" in evaluation


def test_links_carry_the_rgb_name_and_are_not_redone(tmp_path) -> None:
    nir = tmp_path / "nir_clahe"
    nir.mkdir()
    (nir / "mango_nir_7.png").write_bytes(b"x")
    target = tmp_path / "links"
    assert rs.make_nir_as_rgb(tmp_path, target) == 1
    assert rs.make_nir_as_rgb(tmp_path, target) == 1
    assert (target / "mango_rgb_7.jpg").resolve() == (nir / "mango_nir_7.png").resolve()


def test_the_nir_student_pads_like_the_nir_its_kd_teacher_input_will_have(tmp_path) -> None:
    args = _args(tmp_path)
    expected = f"letterbox_value={int(0.31 * 255)}"
    steps = rs.plan_arm(args, "est_nir", 0, 42)
    assert expected in [str(x) for x in steps[0][1]]  # training
    assert expected in [str(x) for x in steps[1][1]]  # evaluation
    assert not any(x.startswith("letterbox_value") for x in _train(args, "est_rgb"))



def _main(tmp_path: Path, monkeypatch, *extra: str) -> tuple[Path, list[list[str]]]:
    """Run `main` for one arm with fake steps; returns its run directory and the commands."""
    _args(tmp_path)
    calls: list[list[str]] = []
    run = tmp_path / "ckpt" / "est_rgb" / "fold0" / "seed42"

    def execute(command, cwd):
        calls.append([str(x) for x in command])
        if len(calls) == 1:
            (run / "best_model.pt").write_bytes(b"trained")
        return 0

    monkeypatch.setattr(rs.rk, "_execute", execute)
    code = rs.main([
        "--dataset", str(tmp_path), "--weights", "w.pt", "--arms", "est_rgb", "--folds", "0",
        "--checkpoints-root", str(tmp_path / "ckpt"), "--reports-root", str(tmp_path / "reports"), *extra,
    ])
    assert code == 0
    return run, calls


def _interrupted_run(tmp_path: Path, *, started: bool = True) -> Path:
    run = tmp_path / "ckpt" / "est_rgb" / "fold0" / "seed42"
    run.mkdir(parents=True)
    (run / "best_model.pt").write_bytes(b"from an interrupted run")
    (run / rs.rk.RESUME_STATE).write_bytes(b"state")
    if started:
        rs.rk._write_identity(run / rs.rk.STARTED_MARKER, rs.IDENTITY)
    return run


def test_an_interrupted_training_continues_and_its_marker_records_the_plain_command(tmp_path, monkeypatch) -> None:
    run = _interrupted_run(tmp_path)
    _, calls = _main(tmp_path, monkeypatch)

    assert calls[0][-2:] == ["--resume", str(run / rs.rk.RESUME_STATE)]
    assert not list(run.parent.glob("seed42.interrupted-*"))
    assert "--resume" not in json.loads((run / rs.rk.COMPLETE_MARKER).read_text())["command"]
    assert not (run / rs.rk.RESUME_STATE).exists()


@pytest.mark.parametrize("started, extra", [(False, ()), (True, ("--no-resume",))])
def test_a_run_that_cannot_continue_is_kept_aside_and_trained_from_scratch(tmp_path, monkeypatch, started, extra) -> None:
    run = _interrupted_run(tmp_path, started=started)
    _, calls = _main(tmp_path, monkeypatch, *extra)

    assert "--resume" not in calls[0]
    assert (run.with_name("seed42.interrupted-1") / rs.rk.RESUME_STATE).read_bytes() == b"state"


def test_a_fresh_training_records_how_it_started_so_it_can_continue(tmp_path, monkeypatch) -> None:
    run, calls = _main(tmp_path, monkeypatch)

    assert "--resume" not in calls[0]
    assert json.loads((run / rs.rk.STARTED_MARKER).read_text())["identity"] == rs.IDENTITY


def test_a_finished_training_is_not_run_again(tmp_path, monkeypatch) -> None:
    run, first = _main(tmp_path, monkeypatch)
    _, second = _main(tmp_path, monkeypatch)
    assert "src.training.train" in first[0]
    assert not any("src.training.train" in command for command in second)
    assert (run / rs.rk.COMPLETE_MARKER).is_file()
