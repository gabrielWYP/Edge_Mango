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
