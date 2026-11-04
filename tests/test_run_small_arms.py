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
