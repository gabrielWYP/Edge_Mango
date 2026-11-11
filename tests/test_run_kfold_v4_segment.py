"""Segment twin of the v4 student arm of `scripts/run_kfold.py`.

`master_etiquetas_seg` (the undistilled student `est_etiquetas`, as a "master" like in `run_small_arms.py`) trains
and is evaluated as a YOLOv8-seg student without `--extra-override task=segment`. The KD students distill from a
MasterModel, which cannot segment, and the plan's masters are MasterModels too, so they have no twin here: the
segment plan with CrossKD is `run_small_arms.py --arms-preset seg`. No `_segdt` name exists. Subprocesses are never
started; the helpers are those of `test_run_kfold_v4.py`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts import evaluate_checkpoint
from scripts import run_kfold as rk
from src.training import train
from src.training.config import TrainingConfig
from tests.test_run_kfold_v4 import (
    ARMS,
    EXPERIMENT,
    RTX3080,
    FakeExecutor,
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

SEG = "master_etiquetas_seg"
DETECT_TEACHER_OF = {"est_kd_rgb": "maestro_rgb", "est_kd_rgbnir": "maestro_rgbnir"}


def _run(tmp_path: Path, **options) -> FakeExecutor:
    executor = FakeExecutor()
    rk.run_kfold(_options(tmp_path, arms=(SEG,), folds=[0], **options), executor)
    return executor


def _train_cmd(executor: FakeExecutor) -> list[str]:
    return next(c for c in executor.calls if _is_train(c) and _run_of(c)[0] == SEG)


def _eval_cmds(executor: FakeExecutor) -> list[list[str]]:
    return [c for c in executor.calls if _is_eval(c) and f"/{SEG}/" in _flag(c, "--output-dir")]


def test_the_table_has_one_twin_and_no_segdt_name() -> None:
    assert {a for a in rk.V4_ARMS if a.endswith("_seg") or a.endswith("_segdt")} == {SEG}
    spec, plain = rk.V4_ARMS[SEG], rk.V4_ARMS["est_etiquetas"]
    assert spec.task == "segment" and plain.task == "detect"
    assert (spec.trainer, spec.teacher, spec.modality_ablation) == (plain.trainer, plain.teacher, plain.modality_ablation)


def test_the_twin_reads_the_config_of_its_base_arm(tmp_path: Path) -> None:
    executor = _run(tmp_path)
    assert _flag(_train_cmd(executor), "--config") == str(_config("est_etiquetas"))
    assert rk.arm_name(_config("master_etiquetas_seg")) == SEG
    assert rk.config_file(_config("master_etiquetas_seg")) == _config("est_etiquetas")
    assert rk.config_file(_config("est_etiquetas")) == _config("est_etiquetas")


def test_the_twin_trains_with_the_task_and_the_mask_options(tmp_path: Path) -> None:
    train_cmd = _train_cmd(_run(tmp_path))
    assert _value(train_cmd, "task") == "segment"
    assert _value(train_cmd, "mask_ratio") == "4" and _value(train_cmd, "mask_weight") == "7.5"


def test_the_mask_options_come_from_the_runner_options(tmp_path: Path) -> None:
    train_cmd = _train_cmd(_run(tmp_path, mask_weight=5.0, mask_ratio=2))
    assert _value(train_cmd, "mask_ratio") == "2" and _value(train_cmd, "mask_weight") == "5.0"


def test_the_evaluation_reads_the_task_from_the_checkpoint_but_builds_the_masks(tmp_path: Path) -> None:
    evals = _eval_cmds(_run(tmp_path))
    assert len(evals) == 2
    for cmd in evals:
        assert _value(cmd, "task") is None and _value(cmd, "mask_ratio") == "4"
        assert "--save-predictions" in cmd and _flag(cmd, "--model") == "student"


@pytest.mark.parametrize("arm", ARMS)
def test_detect_arms_carry_no_segment_option(tmp_path: Path, arm: str) -> None:
    arms = (DETECT_TEACHER_OF[arm], arm) if arm in DETECT_TEACHER_OF else (arm,)
    executor = FakeExecutor()
    rk.run_kfold(_options(tmp_path, arms=arms, folds=[0]), executor)
    for cmd in executor.calls:
        if "--override" in cmd:
            assert not any(t.startswith(("task=", "mask_")) for t in _overrides(cmd))


def test_the_commands_parse_on_their_real_entry_points(tmp_path: Path, monkeypatch) -> None:
    executor = _run(tmp_path)
    args = _argv_parses(monkeypatch, train, _train_cmd(executor), 3)
    assert all(token.count("=") == 1 for token in args.override)
    for cmd in _eval_cmds(executor):
        assert _argv_parses(monkeypatch, evaluate_checkpoint, cmd, 2).model == "student"


def test_every_segment_override_exists_on_the_config_that_receives_it(tmp_path: Path) -> None:
    executor = _run(tmp_path, epochs=1)
    for cmd in [_train_cmd(executor), *_eval_cmds(executor)]:
        config = TrainingConfig.from_yaml(_flag(cmd, "--config"))
        for token in _overrides(cmd):
            assert hasattr(config, token.split("=", 1)[0]), token


def test_main_accepts_a_twin_config_that_has_no_file_of_its_own(tmp_path: Path, capsys) -> None:
    folds_dir = _make_dataset(tmp_path / "real")
    code = rk.main([
        "--folds-dir", str(folds_dir), "--machine", str(RTX3080), "--dry-run", "--folds", "0",
        "--arms", str(EXPERIMENT / "v4_master_etiquetas_seg.yaml"), "--mask-weight", "3", "--mask-ratio", "4",
        "--checkpoints-root", str(tmp_path / "ckpt"), "--reports-root", str(tmp_path / "reports"),
    ])
    assert code == 0
    out = capsys.readouterr().out
    assert "master_etiquetas_seg/fold0/seed42/train" in out and "task=segment" in out and "mask_weight=3.0" in out


def test_main_still_rejects_an_unknown_config(tmp_path: Path, capsys) -> None:
    code = rk.main(["--machine", str(RTX3080), "--dry-run", "--arms", str(EXPERIMENT / "v4_maestro_rgb_seg.yaml")])
    assert code == 1 and "not found" in capsys.readouterr().err
