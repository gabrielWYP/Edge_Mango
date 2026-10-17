"""Tests for scripts/run_kfold.py: command construction and resume logic.

Subprocesses are never started: `run_kfold` takes an `execute(cmd, cwd)`
callable, replaced here by a fake that writes the files a real run would.
"""

from __future__ import annotations

import json
import os
import shlex
import signal
from pathlib import Path

import pytest

from scripts import run_kfold as rk
from scripts.evaluate_checkpoint import _apply_overrides
from src.training.config import TrainingConfig

REPO = Path(__file__).resolve().parent.parent
TWOSTREAM = REPO / "configs" / "experiment" / "twostream.yaml"
ROI_CROP = REPO / "configs" / "experiment" / "twostream_roi_crop.yaml"
RTX3080 = REPO / "configs" / "machines" / "rtx3080.yaml"


# --------------------------------------------------------------------------
# Fixtures / fakes
# --------------------------------------------------------------------------


def _make_folds(root: Path, count: int = 5) -> Path:
    for k in range(count):
        fold = root / f"fold_{k}"
        (fold / "labels" / "train").mkdir(parents=True, exist_ok=True)
        (fold / "splits.json").write_text("{}")
    (root / "folds_meta.json").write_text("{}")
    return root


def _options(tmp_path: Path, **overrides) -> rk.RunnerOptions:
    kwargs = dict(
        folds_dir=_make_folds(tmp_path / "folds"),
        arms=[TWOSTREAM, ROI_CROP],
        folds=None,
        seeds=[42],
        machine=RTX3080,
        epochs=None,
        python="python",
        checkpoints_root=tmp_path / "ckpt",
        reports_root=tmp_path / "reports",
        dry_run=False,
        extra_overrides=[],
    )
    kwargs.update(overrides)
    return rk.RunnerOptions(**kwargs)


class FakeExecutor:
    """Records commands and writes the outputs a successful run would leave."""

    def __init__(self, fail_on: str | None = None, fail_code: int = 3, skip_checkpoint: bool = False):
        self.calls: list[list[str]] = []
        self.cwds: list[Path] = []
        self.fail_on = fail_on
        self.fail_code = fail_code
        self.skip_checkpoint = skip_checkpoint

    def __call__(self, cmd: list[str], cwd: Path) -> int:
        self.calls.append(cmd)
        self.cwds.append(cwd)
        kind = self.kind(cmd)
        if self.fail_on == kind:
            return self.fail_code
        if kind == "train":
            out = Path(next(t for t in cmd if t.startswith("output_dir=")).split("=", 1)[1])
            out.mkdir(parents=True, exist_ok=True)
            if not self.skip_checkpoint:
                (out / "best_model.pt").write_bytes(b"weights")
        else:
            out = Path(cmd[cmd.index("--output-dir") + 1])
            out.mkdir(parents=True, exist_ok=True)
            (out / "metrics.json").write_text("{}")
        return 0

    @staticmethod
    def kind(cmd: list[str]) -> str:
        if "src.training.train" in cmd:
            return "train"
        split = cmd[cmd.index("--split") + 1]
        return f"eval_{split}"

    def kinds(self) -> list[str]:
        return [self.kind(c) for c in self.calls]


# --------------------------------------------------------------------------
# Naming and discovery
# --------------------------------------------------------------------------


def test_default_arms_are_the_full_frame_and_roi_crop_configs() -> None:
    assert [rk.arm_name(a) for a in rk.DEFAULT_ARMS] == ["twostream", "twostream_roi_crop"]
    for arm in rk.DEFAULT_ARMS:
        assert (REPO / arm).is_file()


def test_discover_folds_returns_sorted_complete_fold_dirs_only(tmp_path: Path) -> None:
    for k in (10, 2, 0):
        (tmp_path / f"fold_{k}" / "labels").mkdir(parents=True)
        (tmp_path / f"fold_{k}" / "splits.json").write_text("{}")
    (tmp_path / "fold_7").mkdir()                              # no splits.json: not a fold
    (tmp_path / "fold_x").mkdir()
    (tmp_path / "folds_meta.json").write_text("{}")
    assert rk.discover_folds(tmp_path) == [0, 2, 10]


def test_discover_folds_fails_when_there_are_none(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="fold_"):
        rk.discover_folds(tmp_path)


def test_parse_int_list() -> None:
    assert rk.parse_int_list("0,2, 4") == [0, 2, 4]
    with pytest.raises(ValueError):
        rk.parse_int_list("a,b")


# --------------------------------------------------------------------------
# Command construction
# --------------------------------------------------------------------------


def test_train_command_shape(tmp_path: Path) -> None:
    fold = tmp_path / "folds" / "fold_3"
    run = tmp_path / "ckpt" / "twostream" / "fold3" / "seed42"
    cmd = rk.build_train_command("py", TWOSTREAM, RTX3080, fold, run, seed=42, epochs=None, extra_overrides=[])
    assert cmd == [
        "py", "-m", "src.training.train",
        "--config", str(TWOSTREAM), "--machine", str(RTX3080), "--model", "master",
        "--override",
        "seed=42", "patience=1000", "selection_metric=damage_ap50",
        f"labels_dir={fold / 'labels'}", f"split_manifest={fold / 'splits.json'}",
        f"output_dir={run}",
    ]


def test_train_command_appends_epochs_then_extra_overrides(tmp_path: Path) -> None:
    cmd = rk.build_train_command(
        "py", ROI_CROP, RTX3080, tmp_path / "f", tmp_path / "r", seed=7, epochs=1,
        extra_overrides=["rgb_dir=/x/rgb"],
    )
    overrides = cmd[cmd.index("--override") + 1:]
    assert "seed=7" in overrides
    assert overrides[-2:] == ["epochs=1", "rgb_dir=/x/rgb"]


def test_eval_command_reuses_the_training_data_overrides(tmp_path: Path) -> None:
    fold = tmp_path / "folds" / "fold_1"
    run = tmp_path / "ckpt" / "twostream_roi_crop" / "fold1" / "seed42"
    out = tmp_path / "reports" / "twostream_roi_crop" / "fold1_seed42" / "test"
    cmd = rk.build_eval_command(
        "py", ROI_CROP, run / "best_model.pt", "test", fold, run / "eval_test", out,
        machine_overrides=["batch_size=4"], extra_overrides=["rgb_dir=/x/rgb"],
    )
    assert cmd[:2] == ["py", "scripts/evaluate_checkpoint.py"]
    assert cmd[cmd.index("--config") + 1] == str(ROI_CROP)
    assert cmd[cmd.index("--checkpoint") + 1] == str(run / "best_model.pt")
    assert cmd[cmd.index("--split") + 1] == "test"
    assert cmd[cmd.index("--model") + 1] == "master"
    assert cmd[cmd.index("--output-dir") + 1] == str(out)
    overrides = cmd[cmd.index("--override") + 1:]
    assert overrides == [
        f"labels_dir={fold / 'labels'}", f"split_manifest={fold / 'splits.json'}",
        f"output_dir={run / 'eval_test'}", "batch_size=4", "rgb_dir=/x/rgb",
    ]
    # --override is the last flag, so its nargs="*" cannot swallow another one.
    assert "--output-dir" in cmd[: cmd.index("--override")]


def test_machine_overrides_come_from_the_profile_yaml(tmp_path: Path) -> None:
    assert rk.machine_overrides(RTX3080) == [
        "device=cuda", "batch_size=4", "num_workers=4", "pin_memory=true", "precision=fp32",
    ]
    profile = tmp_path / "m.yaml"
    profile.write_text("device: cpu\npin_memory: false\n")
    assert rk.machine_overrides(profile) == ["device=cpu", "pin_memory=false"]


def test_overrides_are_accepted_by_the_training_config_coercion() -> None:
    """Every key must exist on TrainingConfig (evaluate_checkpoint only warns on unknown
    keys) and values must come out typed, exactly as train.py / evaluate_checkpoint parse them."""
    fold = Path("/data/folds/fold_2")
    run = Path("/ckpt/twostream/fold2/seed42")
    train = rk.build_train_command("py", TWOSTREAM, RTX3080, fold, run, seed=42, epochs=1, extra_overrides=[])
    ev = rk.build_eval_command(
        "py", TWOSTREAM, run / "best_model.pt", "val", fold, run / "eval_val", Path("/rep/val"),
        machine_overrides=rk.machine_overrides(RTX3080), extra_overrides=[],
    )
    for cmd in (train, ev):
        overrides = cmd[cmd.index("--override") + 1:]
        config = TrainingConfig.from_yaml(TWOSTREAM)
        for token in overrides:
            assert hasattr(config, token.split("=", 1)[0]), token
        _apply_overrides(config, overrides)
        assert config.labels_dir == "/data/folds/fold_2/labels"
        assert config.split_manifest == "/data/folds/fold_2/splits.json"
        if cmd is ev:  # the train command gets the profile through --machine instead
            assert config.batch_size == 4 and config.num_workers == 4 and config.device == "cuda"
    config = TrainingConfig.from_yaml(TWOSTREAM)
    _apply_overrides(config, train[train.index("--override") + 1:])
    assert config.seed == 42 and isinstance(config.seed, int)
    assert config.patience == 1000
    assert config.selection_metric == "damage_ap50"
    assert config.epochs == 1
    assert config.output_dir == str(run)


# --------------------------------------------------------------------------
# Resume logic
# --------------------------------------------------------------------------


def test_run_is_complete_only_with_the_marker_and_a_checkpoint(tmp_path: Path) -> None:
    run = tmp_path / "run"
    assert not rk.is_train_complete(run)
    run.mkdir()
    (run / "best_model.pt").write_bytes(b"x")          # appears mid-training
    assert not rk.is_train_complete(run)
    rk.mark_train_complete(run, ["py", "-m", "src.training.train"])
    assert rk.is_train_complete(run)
    (run / "best_model.pt").unlink()
    assert not rk.is_train_complete(run)


def test_incomplete_run_is_moved_aside_not_deleted(tmp_path: Path) -> None:
    run = tmp_path / "ckpt" / "fold0" / "seed42"
    run.mkdir(parents=True)
    (run / "best_model.pt").write_bytes(b"partial")
    first = rk.move_aside_incomplete(run)
    assert first == run.with_name("seed42.interrupted-1")
    assert (first / "best_model.pt").read_bytes() == b"partial"
    assert not run.exists()
    run.mkdir()
    (run / "best_model.pt").write_bytes(b"partial2")
    assert rk.move_aside_incomplete(run) == run.with_name("seed42.interrupted-2")
    assert rk.move_aside_incomplete(run) is None        # nothing there any more


def test_incomplete_run_without_any_weights_is_simply_removed(tmp_path: Path) -> None:
    """Only TensorBoard logs of a failed attempt: nothing worth keeping, no clutter."""
    run = tmp_path / "ckpt" / "fold0" / "seed42"
    (run / "logs").mkdir(parents=True)
    (run / "logs" / "events.out.tfevents.1").write_bytes(b"x")
    assert rk.move_aside_incomplete(run) is None
    assert not run.exists()
    assert not list(run.parent.glob("seed42.interrupted-*"))


def test_fresh_run_executes_train_then_val_then_test_for_every_arm_fold_seed(tmp_path: Path) -> None:
    options = _options(tmp_path, folds=[0, 1], seeds=[42, 7])
    executor = FakeExecutor()
    summary = rk.run_kfold(options, executor)

    assert executor.kinds() == ["train", "eval_val", "eval_test"] * 8
    assert all(cwd == rk.PROJECT_ROOT for cwd in executor.cwds)
    # fold-major, then seed, then arm: both arms of a fold finish before the next fold.
    trained = [
        (Path(next(t for t in c if t.startswith("output_dir="))[len("output_dir="):]).parts[-3:])
        for c in executor.calls if "src.training.train" in c
    ]
    assert trained == [
        ("twostream", "fold0", "seed42"), ("twostream_roi_crop", "fold0", "seed42"),
        ("twostream", "fold0", "seed7"), ("twostream_roi_crop", "fold0", "seed7"),
        ("twostream", "fold1", "seed42"), ("twostream_roi_crop", "fold1", "seed42"),
        ("twostream", "fold1", "seed7"), ("twostream_roi_crop", "fold1", "seed7"),
    ]
    assert summary["twostream/fold0/seed42/train"] == "ran"
    run = tmp_path / "ckpt" / "twostream" / "fold0" / "seed42"
    assert rk.is_train_complete(run)
    for split in ("val", "test"):
        assert (tmp_path / "reports" / "twostream" / "fold0_seed42" / split / "metrics.json").is_file()


def test_rerunning_the_same_command_skips_everything(tmp_path: Path) -> None:
    options = _options(tmp_path, folds=[0])
    rk.run_kfold(options, FakeExecutor())
    second = FakeExecutor()
    summary = rk.run_kfold(options, second)
    assert second.calls == []
    assert set(summary.values()) == {"skipped"}


def test_run_without_marker_is_retrained_and_its_stale_evals_are_dropped(tmp_path: Path) -> None:
    options = _options(tmp_path, folds=[0], arms=[TWOSTREAM])
    run = tmp_path / "ckpt" / "twostream" / "fold0" / "seed42"
    run.mkdir(parents=True)
    (run / "best_model.pt").write_bytes(b"from an interrupted run")
    stale = tmp_path / "reports" / "twostream" / "fold0_seed42" / "val"
    stale.mkdir(parents=True)
    (stale / "metrics.json").write_text('{"stale": true}')

    executor = FakeExecutor()
    rk.run_kfold(options, executor)

    assert executor.kinds() == ["train", "eval_val", "eval_test"]
    assert (run.with_name("seed42.interrupted-1") / "best_model.pt").read_bytes() == b"from an interrupted run"
    assert (run / "best_model.pt").read_bytes() == b"weights"
    assert json.loads((stale / "metrics.json").read_text()) == {}      # rewritten, not the stale one


def test_only_the_missing_eval_is_rerun(tmp_path: Path) -> None:
    options = _options(tmp_path, folds=[0], arms=[ROI_CROP])
    rk.run_kfold(options, FakeExecutor())
    (tmp_path / "reports" / "twostream_roi_crop" / "fold0_seed42" / "test" / "metrics.json").unlink()
    executor = FakeExecutor()
    summary = rk.run_kfold(options, executor)
    assert executor.kinds() == ["eval_test"]
    assert summary["twostream_roi_crop/fold0/seed42/train"] == "skipped"
    assert summary["twostream_roi_crop/fold0/seed42/eval_val"] == "skipped"
    assert summary["twostream_roi_crop/fold0/seed42/eval_test"] == "ran"


def test_failed_training_stops_the_runner_with_a_clear_message(tmp_path: Path) -> None:
    options = _options(tmp_path, folds=[0, 1])
    executor = FakeExecutor(fail_on="train", fail_code=3)
    with pytest.raises(rk.RunnerError, match=r"twostream/fold0/seed42/train failed with exit code 3"):
        rk.run_kfold(options, executor)
    assert executor.kinds() == ["train"]                     # nothing after the failure
    assert not rk.is_train_complete(tmp_path / "ckpt" / "twostream" / "fold0" / "seed42")
    log = (tmp_path / "reports" / "runner.log").read_text()
    assert "FAILED" in log and "exit code 3" in log


def test_failed_eval_stops_the_runner_and_keeps_the_trained_run(tmp_path: Path) -> None:
    options = _options(tmp_path, folds=[0], arms=[TWOSTREAM])
    with pytest.raises(rk.RunnerError, match="eval_val"):
        rk.run_kfold(options, FakeExecutor(fail_on="eval_val", fail_code=1))
    assert rk.is_train_complete(tmp_path / "ckpt" / "twostream" / "fold0" / "seed42")
    # Resume: training is skipped, both evals run.
    executor = FakeExecutor()
    rk.run_kfold(options, executor)
    assert executor.kinds() == ["eval_val", "eval_test"]


def test_training_that_exits_zero_without_a_checkpoint_is_an_error(tmp_path: Path) -> None:
    options = _options(tmp_path, folds=[0], arms=[TWOSTREAM])
    with pytest.raises(rk.RunnerError, match="best_model.pt"):
        rk.run_kfold(options, FakeExecutor(skip_checkpoint=True))
    assert not rk.is_train_complete(tmp_path / "ckpt" / "twostream" / "fold0" / "seed42")


def test_runner_log_records_commands_and_outcomes(tmp_path: Path) -> None:
    options = _options(tmp_path, folds=[0], arms=[TWOSTREAM])
    executor = FakeExecutor()
    rk.run_kfold(options, executor)
    rk.run_kfold(options, FakeExecutor())                    # resume: skips
    lines = (tmp_path / "reports" / "runner.log").read_text().splitlines()
    train_cmd = shlex.join(executor.calls[0])
    assert any("START" in l and "twostream/fold0/seed42/train" in l and train_cmd in l for l in lines)
    assert any("OK" in l and "twostream/fold0/seed42/train" in l for l in lines)
    assert any("SKIP" in l and "twostream/fold0/seed42/train" in l for l in lines)
    assert sum("START" in l for l in lines) == 3


def test_unknown_fold_is_rejected_before_anything_runs(tmp_path: Path) -> None:
    options = _options(tmp_path, folds=[0, 9])
    executor = FakeExecutor()
    with pytest.raises(rk.RunnerError, match="fold 9"):
        rk.run_kfold(options, executor)
    assert executor.calls == []


def test_dry_run_executes_and_creates_nothing(tmp_path: Path, capsys) -> None:
    options = _options(tmp_path, folds=[0], dry_run=True)
    executor = FakeExecutor()
    summary = rk.run_kfold(options, executor)
    assert executor.calls == []
    assert not (tmp_path / "ckpt").exists() and not (tmp_path / "reports").exists()
    assert set(summary.values()) == {"dry-run"}
    out = capsys.readouterr().out
    assert out.count("src.training.train") == 2


def test_main_dry_run_prints_the_ten_train_commands(tmp_path: Path, capsys) -> None:
    folds = _make_folds(tmp_path / "folds")
    code = rk.main([
        "--folds-dir", str(folds), "--machine", str(RTX3080), "--dry-run",
        "--checkpoints-root", str(tmp_path / "ckpt"), "--reports-root", str(tmp_path / "reports"),
    ])
    assert code == 0
    lines = [l for l in capsys.readouterr().out.splitlines() if "src.training.train" in l]
    assert len(lines) == 10
    for line in lines:
        assert "seed=42" in line and "patience=1000" in line and "selection_metric=damage_ap50" in line
    assert sum("twostream_roi_crop" in l.split("--override")[0] for l in lines) == 5


def test_main_returns_nonzero_on_a_failed_step(tmp_path: Path, monkeypatch, capsys) -> None:
    folds = _make_folds(tmp_path / "folds")
    monkeypatch.setattr(rk, "_execute", FakeExecutor(fail_on="train", fail_code=5))
    code = rk.main([
        "--folds-dir", str(folds), "--machine", str(RTX3080), "--folds", "0",
        "--checkpoints-root", str(tmp_path / "ckpt"), "--reports-root", str(tmp_path / "reports"),
    ])
    assert code == 1
    assert "exit code 5" in capsys.readouterr().err


def test_main_exits_130_on_ctrl_c_and_tells_how_to_resume(tmp_path: Path, monkeypatch, capsys) -> None:
    folds = _make_folds(tmp_path / "folds")

    def interrupted(cmd, cwd):
        raise KeyboardInterrupt

    monkeypatch.setattr(rk, "_execute", interrupted)
    code = rk.main([
        "--folds-dir", str(folds), "--machine", str(RTX3080), "--folds", "0",
        "--checkpoints-root", str(tmp_path / "ckpt"), "--reports-root", str(tmp_path / "reports"),
    ])
    assert code == 130
    assert "re-run the same command" in capsys.readouterr().err.lower()


@pytest.mark.skipif(not hasattr(signal, "SIGTERM") or os.name == "nt", reason="POSIX signals")
def test_sigterm_is_handled_like_ctrl_c_and_the_handler_is_restored(tmp_path: Path, monkeypatch, capsys) -> None:
    """`kill <runner>` must stop the run cleanly (subprocess.run kills its child on
    KeyboardInterrupt) instead of orphaning a training process."""
    folds = _make_folds(tmp_path / "folds")
    before = signal.getsignal(signal.SIGTERM)

    def terminated(cmd, cwd):
        os.kill(os.getpid(), signal.SIGTERM)
        raise AssertionError("SIGTERM was not converted into an interrupt")

    monkeypatch.setattr(rk, "_execute", terminated)
    code = rk.main([
        "--folds-dir", str(folds), "--machine", str(RTX3080), "--folds", "0",
        "--checkpoints-root", str(tmp_path / "ckpt"), "--reports-root", str(tmp_path / "reports"),
    ])
    assert code == 130
    assert "re-run the same command" in capsys.readouterr().err.lower()
    assert signal.getsignal(signal.SIGTERM) == before
