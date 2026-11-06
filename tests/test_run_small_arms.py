"""The plan of the small arms: the protocol, the modalities and the order."""

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
    for arm in ("est_rgb", "est_rgbnir", "kd_nir", "kd_rgb", "kd_rgbnir"):
        assert f"rgb_dir={tmp_path / 'rgb'}" in _train(args, arm)


@pytest.mark.parametrize(
    "arm, teacher, reads",
    [("kd_nir", "est_nir", "nir"), ("kd_rgb", "est_rgb", "rgb"), ("kd_rgbnir", "est_rgbnir", "rgbnir")],
)
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


def test_val_only_never_evaluates_on_test(tmp_path) -> None:
    steps = rs.plan_arm(_args(tmp_path, "--val-only"), "kd_nir", 0, 42)
    assert [s[0].rsplit("/", 1)[1] for s in steps] == ["train", "eval_val", "metrics_val"]
    commands = [[str(x) for x in s[1]] for s in steps]
    assert not any(x == "test" or "labels/test" in x for command in commands for x in command)


def test_the_extra_override_reaches_training_and_evaluation(tmp_path) -> None:
    steps = rs.plan_arm(_args(tmp_path, "--extra-override", "ignore_classes=[0]"), "est_rgb", 0, 42)
    train, evaluation = [str(x) for x in steps[0][1]], [str(x) for x in steps[1][1]]
    assert "ignore_classes=[0]" in train and "ignore_classes=[0]" in evaluation


@pytest.mark.parametrize("arm, folder", [("est_nir", "nir_as_rgb"), ("est_rgb", "rgb"), ("kd_nir", "rgb")])
def test_the_pixel_cache_follows_the_rgb_slot_and_only_in_training(tmp_path, arm, folder) -> None:
    steps = rs.plan_arm(_args(tmp_path, "--pixel-cache-root", "/c"), arm, 0, 42)
    assert f"pixel_cache_dir=/c/{folder}" in [str(x) for x in steps[0][1]]
    assert not any("pixel_cache_dir" in str(x) for s in steps[1:] for x in s[1])


def test_a_student_waits_for_its_teacher_and_the_limit_holds() -> None:
    import threading
    import time

    lock, running, peak, finished, order = threading.Lock(), [0], [0], [], []

    def run(arm, fold, seed):
        with lock:
            teacher = rs.ARMS[arm].teacher
            assert teacher is None or (teacher, fold, seed) in finished
            running[0] += 1
            peak[0] = max(peak[0], running[0])
            order.append(arm)
        time.sleep(0.01)
        with lock:
            running[0] -= 1
            finished.append((arm, fold, seed))
        return 0

    tasks = [(arm, fold, 42) for fold in range(3) for arm in rs.ARMS]
    assert rs.schedule(tasks, 3, run) == 0
    assert len(finished) == len(tasks) and peak[0] == 3


def test_a_failure_starts_nothing_new_and_is_returned() -> None:
    started = []

    def run(arm, fold, seed):
        started.append((arm, fold))
        return 7 if arm == "est_rgb" else 0

    tasks = [("est_rgb", 0, 42), ("kd_rgb", 0, 42), ("est_nir", 1, 42)]
    assert rs.schedule(tasks, 1, run) == 7
    assert started == [("est_rgb", 0)]


def test_an_interrupt_stops_the_running_processes_at_once(monkeypatch) -> None:
    import subprocess
    import sys
    import threading
    import time

    started = threading.Event()
    real_wait = rs.wait

    def run(arm, fold, seed):
        started.set()
        return rs._run_logged([sys.executable, "-c", "import time; time.sleep(60)"], None, dict(rs.os.environ))

    def interrupted_wait(*args, **kwargs):
        started.wait(5)
        while not rs._children:
            time.sleep(0.01)
        raise KeyboardInterrupt

    monkeypatch.setattr(rs, "wait", interrupted_wait)
    begin = time.time()
    with pytest.raises(KeyboardInterrupt):
        rs.schedule([("est_rgb", 0, 42), ("est_nir", 0, 42)], 1, run)
    assert time.time() - begin < 30 and not rs._children
    assert real_wait is not None and subprocess is not None


def _main(tmp_path: Path, monkeypatch, *extra: str) -> tuple[Path, list[list[str]]]:
    """Run `main` for one arm with fake steps; returns its run directory and the commands."""
    _args(tmp_path)
    calls: list[list[str]] = []
    run = tmp_path / "ckpt" / "est_rgb" / "fold0" / "seed42"

    def execute(command, log_path, env):
        calls.append([str(x) for x in command])
        if len(calls) == 1:
            (run / "best_model.pt").write_bytes(b"trained")
        return 0

    monkeypatch.setattr(rs, "_run_logged", execute)
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


def test_a_continued_pilot_keeps_its_overrides_and_never_reaches_test(tmp_path, monkeypatch) -> None:
    run = _interrupted_run(tmp_path)
    _, calls = _main(
        tmp_path, monkeypatch, "--val-only", "--pixel-cache-root", "/c", "--extra-override", "ignore_classes=[0]",
    )

    train, evaluation = calls[0], calls[1]
    assert train[-2:] == ["--resume", str(run / rs.rk.RESUME_STATE)]
    assert "pixel_cache_dir=/c/rgb" in train and "ignore_classes=[0]" in train
    assert "pixel_cache_dir=/c/rgb" not in evaluation and "ignore_classes=[0]" in evaluation
    assert not any(x == "test" or "labels/test" in x for command in calls for x in command)


def test_a_continued_run_appends_to_its_log(tmp_path, monkeypatch) -> None:
    import sys

    log = tmp_path / "run.log"
    command = [sys.executable, "-c", "print('attempt')"]
    assert rs._run_logged(command, log, dict(rs.os.environ)) == 0
    assert rs._run_logged(command, log, dict(rs.os.environ)) == 0
    assert log.read_text().count("attempt") == 2


def test_the_two_modality_student_reads_rgb_and_nir_and_is_a_planned_arm(tmp_path) -> None:
    args = _args(tmp_path, "--pixel-cache-root", "cache")
    steps = rs.plan_arm(args, "est_rgbnir", 0, 42)
    train, evaluation = [str(x) for x in steps[0][1]], [str(x) for x in steps[1][1]]
    assert "student_in_channels=4" in train and "student_in_channels=4" in evaluation
    assert f"rgb_dir={tmp_path / 'rgb'}" in train and f"nir_dir={tmp_path / 'nir_clahe'}" in train
    assert f"pixel_cache_dir={Path('cache') / 'rgb'}" in train
    assert not any(x.startswith("letterbox_value=") for x in train)
    assert "est_rgbnir" in args.arms and "kd_rgbnir" in args.arms and "kd_nir" not in args.arms
    assert not any("student_in_channels" in str(x) for x in _train(args, "est_rgb"))
