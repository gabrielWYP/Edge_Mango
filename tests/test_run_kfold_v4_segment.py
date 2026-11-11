"""Segment twins of the v4 arms of `scripts/run_kfold.py`.

`est_etiquetas_seg` and the KD students `est_kd_rgb_segdt` / `est_kd_rgbnir_segdt` train and evaluate as
YOLOv8-seg students without `--extra-override task=segment`. The teachers are MasterModels, which cannot
segment, so a KD student can only distill from a detect teacher (`_segdt`, as in `run_small_arms.py`) and the
masters have no twin. Subprocesses are never started; the helpers are those of `test_run_kfold_v4.py`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts import evaluate_checkpoint
from scripts import run_kfold as rk
from src.training import kd_train, train
from src.training.config import TrainingConfig
from src.training.kd_config import KDConfig
from tests.test_run_kfold_v4 import (
    ARMS,
    EXPERIMENT,
    FakeExecutor,
    RTX3080,
    _argv_parses,
    _config,
    _flag,
    _is_eval,
    _is_train,
    _make_dataset,
    _options,
    _overrides,
    _run_of,
    _value,
)

SEG_ARMS = ("est_etiquetas_seg", "est_kd_rgb_segdt", "est_kd_rgbnir_segdt")
TEACHER_OF = {"est_kd_rgb_segdt": "maestro_rgb", "est_kd_rgbnir_segdt": "maestro_rgbnir"}
DETECT_TEACHER_OF = {"est_kd_rgb": "maestro_rgb", "est_kd_rgbnir": "maestro_rgbnir"}
BASE_OF = {"est_etiquetas_seg": "est_etiquetas", "est_kd_rgb_segdt": "est_kd_rgb", "est_kd_rgbnir_segdt": "est_kd_rgbnir"}


def _run(tmp_path: Path, arm: str, **options) -> FakeExecutor:
    arms = (TEACHER_OF[arm], arm) if arm in TEACHER_OF else (arm,)
    executor = FakeExecutor()
    rk.run_kfold(_options(tmp_path, arms=arms, folds=[0], **options), executor)
    return executor


def _train_cmd(executor: FakeExecutor, arm: str) -> list[str]:
    return next(c for c in executor.calls if _is_train(c) and _run_of(c)[0] == arm)


def _eval_cmds(executor: FakeExecutor, arm: str) -> list[str]:
    return [c for c in executor.calls if _is_eval(c) and f"/{arm}/" in _flag(c, "--output-dir")]


def test_the_table_has_the_segment_twins_and_none_for_the_masters() -> None:
    assert {a for a in rk.V4_ARMS if a.endswith(("_seg", "_segdt"))} == set(SEG_ARMS)
    for twin, base in BASE_OF.items():
        spec, plain = rk.V4_ARMS[twin], rk.V4_ARMS[base]
        assert spec.task == "segment" and plain.task == "detect"
        assert (spec.trainer, spec.teacher, spec.modality_ablation) == (plain.trainer, plain.teacher, plain.modality_ablation)


def test_a_twin_reads_the_config_of_its_base_arm(tmp_path: Path) -> None:
    executor = _run(tmp_path, "est_etiquetas_seg")
    assert _flag(_train_cmd(executor, "est_etiquetas_seg"), "--config") == str(_config("est_etiquetas"))
    assert rk.arm_name(_config("est_kd_rgb_segdt")) == "est_kd_rgb_segdt"
    assert rk.config_file(_config("est_kd_rgb_segdt")) == _config("est_kd_rgb")
    assert rk.config_file(_config("est_kd_rgb")) == _config("est_kd_rgb")


@pytest.mark.parametrize("arm", SEG_ARMS)
def test_a_segment_arm_trains_with_the_task_and_the_mask_options(tmp_path: Path, arm: str) -> None:
    train_cmd = _train_cmd(_run(tmp_path, arm), arm)
    assert _value(train_cmd, "task") == "segment"
    assert _value(train_cmd, "mask_ratio") == "4" and _value(train_cmd, "mask_weight") == "7.5"


def test_the_mask_options_come_from_the_runner_options(tmp_path: Path) -> None:
    train_cmd = _train_cmd(_run(tmp_path, "est_etiquetas_seg", mask_weight=5.0, mask_ratio=2), "est_etiquetas_seg")
    assert _value(train_cmd, "mask_ratio") == "2" and _value(train_cmd, "mask_weight") == "5.0"


@pytest.mark.parametrize("arm", SEG_ARMS)
def test_the_evaluation_reads_the_task_from_the_checkpoint_but_builds_the_masks(tmp_path: Path, arm: str) -> None:
    evals = _eval_cmds(_run(tmp_path, arm), arm)
    assert len(evals) == 2
    for cmd in evals:
        assert _value(cmd, "task") is None and _value(cmd, "mask_ratio") == "4"
        assert "--save-predictions" in cmd and _flag(cmd, "--model") == "student"


@pytest.mark.parametrize("arm", ("est_kd_rgb_segdt", "est_kd_rgbnir_segdt"))
def test_a_segment_student_distills_from_the_detect_master_teacher(tmp_path: Path, arm: str) -> None:
    executor = _run(tmp_path, arm)
    kd = _train_cmd(executor, arm)
    assert "src.training.kd_train" in kd and _value(kd, "task") == "segment"
    assert _value(kd, "teacher_checkpoint") == str(tmp_path / "ckpt" / TEACHER_OF[arm] / "fold0" / "seed42" / "best_model.pt")
    teacher = _train_cmd(executor, TEACHER_OF[arm])
    assert _value(teacher, "task") is None  # the master stays a detector
    assert all(_flag(c, "--config").endswith("v4_est_etiquetas.yaml") for c in _eval_cmds(executor, arm))


@pytest.mark.parametrize("arm", ARMS)
def test_detect_arms_carry_no_segment_option(tmp_path: Path, arm: str) -> None:
    arms = (DETECT_TEACHER_OF[arm], arm) if arm in DETECT_TEACHER_OF else (arm,)
    executor = FakeExecutor()
    rk.run_kfold(_options(tmp_path, arms=arms, folds=[0]), executor)
    for cmd in executor.calls:
        if "--override" in cmd:
            assert not any(t.startswith(("task=", "mask_")) for t in _overrides(cmd))


@pytest.mark.parametrize("arm", SEG_ARMS)
def test_the_commands_parse_on_their_real_entry_points(tmp_path: Path, monkeypatch, arm: str) -> None:
    executor = _run(tmp_path, arm)
    train_cmd = _train_cmd(executor, arm)
    module = kd_train if arm in TEACHER_OF else train
    args = _argv_parses(monkeypatch, module, train_cmd, 3)
    assert all(token.count("=") == 1 for token in args.override)
    for cmd in _eval_cmds(executor, arm):
        assert _argv_parses(monkeypatch, evaluate_checkpoint, cmd, 2).model == "student"


@pytest.mark.parametrize("arm", SEG_ARMS)
def test_every_segment_override_exists_on_the_config_that_receives_it(tmp_path: Path, arm: str) -> None:
    executor = _run(tmp_path, arm, epochs=1)
    own = [_train_cmd(executor, arm), *_eval_cmds(executor, arm)]
    for cmd in own:
        loader = KDConfig if "src.training.kd_train" in cmd else TrainingConfig
        config = loader.from_yaml(_flag(cmd, "--config"))
        for token in _overrides(cmd):
            assert hasattr(config, token.split("=", 1)[0]), token


def test_main_accepts_a_twin_config_that_has_no_file_of_its_own(tmp_path: Path, capsys) -> None:
    folds_dir = _make_dataset(tmp_path / "real")
    code = rk.main([
        "--folds-dir", str(folds_dir), "--machine", str(RTX3080), "--dry-run", "--folds", "0",
        "--arms", str(EXPERIMENT / "v4_est_etiquetas_seg.yaml"), "--mask-weight", "3", "--mask-ratio", "4",
        "--checkpoints-root", str(tmp_path / "ckpt"), "--reports-root", str(tmp_path / "reports"),
    ])
    assert code == 0
    out = capsys.readouterr().out
    assert "est_etiquetas_seg/fold0/seed42/train" in out and "task=segment" in out and "mask_weight=3.0" in out


def test_main_still_rejects_an_unknown_config(tmp_path: Path, capsys) -> None:
    code = rk.main(["--machine", str(RTX3080), "--dry-run", "--arms", str(EXPERIMENT / "v4_maestro_rgb_seg.yaml")])
    assert code == 1 and "not found" in capsys.readouterr().err
