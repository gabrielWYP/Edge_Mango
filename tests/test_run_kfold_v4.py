"""Tests for the E1 and E2 arms of scripts/run_kfold.py (plan-experimental, sections 5 and 6).

Subprocesses are never started: `run_kfold` takes an `execute(cmd, cwd)` callable,
replaced here by a fake that writes the files a real step would. The folds on disk
have the layout of `scripts/make_v4_folds.py`, with stats that differ per fold, NIR
variant and view, so a command can only hold the right numbers by reading the right
entry.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from scripts import evaluate_checkpoint, v4_metrics
from scripts import run_kfold as rk
from src.training import kd_train, train
from src.training.config import TrainingConfig
from src.training.kd_config import KDConfig

REPO = Path(__file__).resolve().parent.parent
EXPERIMENT = REPO / "configs" / "experiment"
RTX3080 = REPO / "configs" / "machines" / "rtx3080.yaml"
TWOSTREAM = EXPERIMENT / "twostream.yaml"

TEACHERS = ("maestro_rgbnir", "maestro_rgb", "maestro_nir")
STUDENTS = ("est_etiquetas", "est_kd_rgb", "est_kd_rgbnir")
ARMS = TEACHERS + STUDENTS
TEACHER_OF = {"est_kd_rgb": "maestro_rgb", "est_kd_rgbnir": "maestro_rgbnir"}
ABLATION = {"maestro_rgb": "nir_const", "maestro_nir": "rgb_const", "est_kd_rgb": "nir_const"}
STEPS_PER_RUN = ("train", "eval_val", "metrics_val", "eval_test", "metrics_test")


# --------------------------------------------------------------------------
# Fixtures / fakes
# --------------------------------------------------------------------------


def _config(arm: str, directory: Path = EXPERIMENT) -> Path:
    return directory / f"v4_{arm}.yaml"


def _stats(fold: int, variant: str, view: str) -> dict[str, float]:
    """Different numbers for every fold, variant and view."""
    base = {"nir": 0.1, "nir_clahe": 0.2}[variant] + {"frame": 0.01, "roi": 0.02}[view] + fold * 0.001
    return {"mean": round(base, 6), "std": round(base + 0.5, 6)}


def _make_dataset(root: Path, folds: int = 2) -> Path:
    """A dataset folder holding `folds_lot/` as `make_v4_folds.py` writes it; returns the folds dir."""
    dataset = root / "v4_dataset"
    dataset.mkdir(parents=True)
    index = dataset / "index.csv"
    index.write_text("stem,lot\nmango_rgb_1,1\n")
    folds_dir = dataset / "folds_lot"
    meta = {}
    for k in range(folds):
        fold = folds_dir / f"fold_{k}"
        for split in ("train", "val", "test"):
            (fold / "labels" / split).mkdir(parents=True)
            (fold / "labels" / split / f"{split}.txt").write_text("0 0.5 0.5 0.1 0.1\n")
        (fold / "splits.json").write_text("{}")
        meta[str(k)] = {
            "nir_stats": {
                variant: {view: _stats(k, variant, view) for view in ("frame", "roi")}
                for variant in ("nir", "nir_clahe")
            },
        }
    (folds_dir / "folds_meta.json").write_text(json.dumps({
        "folds": meta,
        "index_sha256": hashlib.sha256(index.read_bytes()).hexdigest(),
        "schema_version": 1,
    }))
    return folds_dir


def _options(tmp_path: Path, arms=ARMS, **overrides) -> rk.RunnerOptions:
    if not (tmp_path / "v4_dataset").exists():
        _make_dataset(tmp_path)
    kwargs = dict(
        folds_dir=tmp_path / "v4_dataset" / "folds_lot",
        arms=[arm if isinstance(arm, Path) else _config(arm) for arm in arms],
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


def _private_configs(tmp_path: Path) -> Path:
    """Copies of the six arm configs, so a test can edit them."""
    directory = tmp_path / "configs"
    directory.mkdir()
    for arm in ARMS:
        shutil.copy(_config(arm), directory)
    return directory


def _overrides(cmd: list[str]) -> list[str]:
    return cmd[cmd.index("--override") + 1:]


def _value(cmd: list[str], key: str) -> str | None:
    matches = [t.split("=", 1)[1] for t in _overrides(cmd) if t.startswith(f"{key}=")]
    assert len(matches) <= 1, f"{key} set twice: {matches}"
    return matches[0] if matches else None


def _flag(cmd: list[str], flag: str) -> str:
    return cmd[cmd.index(flag) + 1]


def _is_train(cmd: list[str]) -> bool:
    return "src.training.train" in cmd or "src.training.kd_train" in cmd


def _is_eval(cmd: list[str]) -> bool:
    return "scripts/evaluate_checkpoint.py" in cmd


def _is_metrics(cmd: list[str]) -> bool:
    return "scripts/v4_metrics.py" in cmd


def _run_of(cmd: list[str]) -> tuple[str, str, str]:
    """(arm, fold<k>, seed<S>) of a train command."""
    return tuple(Path(_value(cmd, "output_dir")).parts[-3:])


class FakeExecutor:
    """Records commands and writes the outputs a successful step would leave."""

    def __init__(self, omit: tuple[str, ...] = (), fail_on: str | None = None):
        self.calls: list[list[str]] = []
        self.omit = omit
        self.fail_on = fail_on
        self.teachers_seen: list[Path] = []

    def __call__(self, cmd: list[str], cwd: Path) -> int:
        self.calls.append(cmd)
        if self.fail_on and self.fail_on in " ".join(cmd):
            return 3
        if _is_train(cmd):
            out = Path(_value(cmd, "output_dir"))
            out.mkdir(parents=True, exist_ok=True)
            if "src.training.kd_train" in cmd:
                teacher = Path(_value(cmd, "teacher_checkpoint"))
                assert teacher.is_file(), f"KD started before its teacher exists: {teacher}"
                self.teachers_seen.append(teacher)
            (out / "best_model.pt").write_bytes(b"weights of " + str(out).encode())
        elif _is_eval(cmd):
            out = Path(_flag(cmd, "--output-dir"))
            out.mkdir(parents=True, exist_ok=True)
            (out / "metrics.json").write_text("{}")
            if "--save-predictions" in cmd and "predictions.jsonl" not in self.omit:
                (out / "predictions.jsonl").write_text("{}\n")
        elif "metrics_v4.json" not in self.omit:
            Path(_flag(cmd, "--output")).write_text("{}")
        return 0

    def trained(self) -> list[tuple[str, str, str]]:
        return [_run_of(c) for c in self.calls if _is_train(c)]


# --------------------------------------------------------------------------
# Arm definitions
# --------------------------------------------------------------------------


def test_v4_configs_take_the_names_of_the_plan() -> None:
    assert [rk.arm_name(_config(arm)) for arm in ARMS] == list(ARMS)


def test_other_configs_keep_their_file_name_as_arm_name() -> None:
    assert rk.arm_name(TWOSTREAM) == "twostream"
    assert rk.arm_name("configs/experiment/twostream_roi_crop.yaml") == "twostream_roi_crop"


def test_a_v4_config_the_runner_does_not_know_is_rejected(tmp_path: Path) -> None:
    unknown = tmp_path / "v4_maestro_grande.yaml"
    unknown.write_text(_config("maestro_rgb").read_text())
    executor = FakeExecutor()
    with pytest.raises(rk.RunnerError, match="v4_maestro_grande"):
        rk.run_kfold(_options(tmp_path, arms=[unknown]), executor)
    assert executor.calls == []


def test_the_three_teachers_share_one_architecture() -> None:
    loaded = {arm: yaml.safe_load(_config(arm).read_text()) for arm in TEACHERS}
    for arm, data in loaded.items():
        assert data.pop("output_dir").endswith(arm)
        config = TrainingConfig.from_yaml(_config(arm))
        assert config.model_type == "master"
        assert config.fusion_mode == "cross_attention"
        assert config.fusion_attention and config.use_nir
    assert loaded["maestro_rgb"] == loaded["maestro_rgbnir"] == loaded["maestro_nir"]


def test_the_student_arms_differ_only_in_what_distillation_needs() -> None:
    base = yaml.safe_load(_config("est_etiquetas").read_text())
    assert base.pop("output_dir").endswith("est_etiquetas")
    assert TrainingConfig.from_yaml(_config("est_etiquetas")).model_type == "student"
    kd_only = {
        "teacher_checkpoint", "kd_weight", "kd_temperature", "kd_loss_type", "distill_levels",
        "fusion_mode", "head_strides", "batch_size",
    }
    for arm in TEACHER_OF:
        data = yaml.safe_load(_config(arm).read_text())
        assert data.pop("output_dir").endswith(arm)
        assert {k: v for k, v in data.items() if k not in kd_only} == base, arm
        assert KDConfig.from_yaml(_config(arm)).model_type == "student"


def test_the_kd_configs_cannot_be_the_eval_config() -> None:
    """Why a KD arm evaluates with `v4_est_etiquetas.yaml`: `evaluate_checkpoint.py` loads
    a plain TrainingConfig, which refuses the KD keys."""
    with pytest.raises(TypeError):
        TrainingConfig.from_yaml(_config("est_kd_rgb"))


def test_every_arm_feeds_the_model_the_same_crop() -> None:
    crops = {
        arm: {k: v for k, v in yaml.safe_load(_config(arm).read_text()).items() if k.startswith("roi_crop")}
        for arm in ARMS
    }
    assert len({json.dumps(c, sort_keys=True) for c in crops.values()}) == 1


def test_the_kd_teacher_is_built_like_the_teachers() -> None:
    teacher = TrainingConfig.from_yaml(_config("maestro_rgb"))
    for arm in TEACHER_OF:
        kd = KDConfig.from_yaml(_config(arm))
        assert kd.fusion_mode == teacher.fusion_mode
        assert kd.head_strides == teacher.head_strides
        assert kd.backbone_variant == teacher.backbone_variant
        assert kd.num_classes == teacher.num_classes


# --------------------------------------------------------------------------
# Order, teachers and entry points
# --------------------------------------------------------------------------


def test_each_run_is_train_then_val_then_test_each_followed_by_its_metrics(tmp_path: Path) -> None:
    summary = rk.run_kfold(_options(tmp_path, folds=[0], arms=["maestro_rgb"]), FakeExecutor())
    assert list(summary) == [f"maestro_rgb/fold0/seed42/{step}" for step in STEPS_PER_RUN]
    assert set(summary.values()) == {"ran"}


def test_teachers_train_before_their_students_whatever_the_order_of_the_arms(tmp_path: Path) -> None:
    options = _options(tmp_path, arms=tuple(reversed(ARMS)), seeds=[42, 7])
    executor = FakeExecutor()
    rk.run_kfold(options, executor)
    trained = executor.trained()
    assert len(trained) == 6 * 2 * 2
    # Fold-major, then seed: one fold and seed finishes before the next one starts.
    assert [run[1:] for run in trained] == (
        [("fold0", "seed42")] * 6 + [("fold0", "seed7")] * 6 + [("fold1", "seed42")] * 6 + [("fold1", "seed7")] * 6
    )
    for student, teacher in TEACHER_OF.items():
        for fold in ("fold0", "fold1"):
            for seed in ("seed42", "seed7"):
                assert trained.index((teacher, fold, seed)) < trained.index((student, fold, seed))


def test_each_student_distills_from_the_teacher_of_its_fold_and_seed(tmp_path: Path) -> None:
    options = _options(tmp_path, seeds=[42, 7])
    executor = FakeExecutor()
    rk.run_kfold(options, executor)
    kd_calls = [c for c in executor.calls if "src.training.kd_train" in c]
    assert len(kd_calls) == 2 * 2 * 2
    for cmd in kd_calls:
        arm, fold, seed = _run_of(cmd)
        expected = tmp_path / "ckpt" / TEACHER_OF[arm] / fold / seed / "best_model.pt"
        assert _value(cmd, "teacher_checkpoint") == str(expected)
        assert _value(cmd, "seed") == seed.removeprefix("seed")


@pytest.mark.parametrize("arm", ARMS)
def test_each_arm_uses_its_training_entry_point_and_evaluates_with_the_matching_model(tmp_path: Path, arm: str) -> None:
    arms = (TEACHER_OF[arm], arm) if arm in TEACHER_OF else (arm,)
    executor = FakeExecutor()
    rk.run_kfold(_options(tmp_path, arms=arms, folds=[0]), executor)
    train_cmd = next(c for c in executor.calls if _is_train(c) and _run_of(c)[0] == arm)
    evals = [c for c in executor.calls if _is_eval(c) and f"/{arm}/" in _flag(c, "--output-dir")]
    assert len(evals) == 2

    if arm in TEACHERS:
        assert train_cmd[1:3] == ["-m", "src.training.train"]
        assert _flag(train_cmd, "--model") == "master"
        assert _flag(train_cmd, "--config") == str(_config(arm))
    elif arm == "est_etiquetas":
        assert train_cmd[1:3] == ["-m", "src.training.train"]
        assert _flag(train_cmd, "--model") == "student"
        assert _value(train_cmd, "teacher_checkpoint") is None
    else:
        assert train_cmd[1:3] == ["-m", "src.training.kd_train"]
        assert _flag(train_cmd, "--config") == str(_config(arm))
        # kd_train has no --model and no --machine: the student is its only model.
        assert "--model" not in train_cmd and "--machine" not in train_cmd

    for cmd in evals:
        assert _flag(cmd, "--checkpoint") == str(tmp_path / "ckpt" / arm / "fold0" / "seed42" / "best_model.pt")
        assert _flag(cmd, "--model") == ("master" if arm in TEACHERS else "student")
        # evaluate_checkpoint.py loads a TrainingConfig, which refuses the KD keys.
        eval_arm = arm if arm in TEACHERS else "est_etiquetas"
        assert _flag(cmd, "--config") == str(_config(eval_arm))


def test_the_kd_trainer_takes_the_machine_profile_except_the_batch_size(tmp_path: Path) -> None:
    """kd_train has no --machine, and KDTrainer no gradient accumulation: with the batch of
    the profile the KD arms would take twice the optimizer steps of `est_etiquetas`."""
    executor = FakeExecutor()
    rk.run_kfold(_options(tmp_path, folds=[0], arms=("maestro_rgb", "est_kd_rgb")), executor)
    kd = next(c for c in executor.calls if "src.training.kd_train" in c)
    assert _value(kd, "device") == "cuda"
    assert _value(kd, "num_workers") == "4"
    assert _value(kd, "pin_memory") == "true"
    assert _value(kd, "precision") == "fp32"
    assert _value(kd, "batch_size") is None
    config = KDConfig.from_yaml(_config("est_kd_rgb"))
    assert config.batch_size == config.effective_batch == 8
    # The evaluation of the same arm takes the whole profile, as every evaluation does.
    assert _value(next(c for c in executor.calls if _is_eval(c)), "batch_size") == "4"


def test_the_protocol_epochs_and_extra_overrides_reach_the_kd_command(tmp_path: Path) -> None:
    executor = FakeExecutor()
    options = _options(
        tmp_path, folds=[0], arms=("maestro_rgb", "est_kd_rgb"), epochs=1, extra_overrides=["lr=0.5"],
    )
    rk.run_kfold(options, executor)
    kd = next(c for c in executor.calls if "src.training.kd_train" in c)
    assert _value(kd, "patience") == "1000"
    assert _value(kd, "selection_metric") == "damage_ap50"
    assert _value(kd, "seed") == "42"
    assert _value(kd, "labels_dir") == str(options.folds_dir / "fold_0" / "labels")
    assert _value(kd, "split_manifest") == str(options.folds_dir / "fold_0" / "splits.json")
    assert _overrides(kd)[-2:] == ["epochs=1", "lr=0.5"]


@pytest.mark.parametrize("arm", ARMS)
def test_modality_ablation_goes_on_every_command_of_the_arm_that_has_one(tmp_path: Path, arm: str) -> None:
    arms = (TEACHER_OF[arm], arm) if arm in TEACHER_OF else (arm,)
    executor = FakeExecutor()
    rk.run_kfold(_options(tmp_path, arms=arms, folds=[0]), executor)
    own = [
        c for c in executor.calls
        if (_is_train(c) and _run_of(c)[0] == arm) or (_is_eval(c) and f"/{arm}/" in _flag(c, "--output-dir"))
    ]
    assert len(own) == 3
    for cmd in own:
        assert _value(cmd, "modality_ablation") == ABLATION.get(arm)


# --------------------------------------------------------------------------
# A missing or foreign teacher
# --------------------------------------------------------------------------


def test_a_missing_teacher_stops_before_anything_runs(tmp_path: Path) -> None:
    executor = FakeExecutor()
    with pytest.raises(rk.RunnerError) as error:
        rk.run_kfold(_options(tmp_path, arms=("est_etiquetas", "est_kd_rgb")), executor)
    message = str(error.value)
    assert "est_kd_rgb/fold0/seed42/train" in message
    assert str(tmp_path / "ckpt" / "maestro_rgb" / "fold0" / "seed42" / "best_model.pt") in message
    assert "maestro_rgb" in message
    assert executor.calls == []
    assert not (tmp_path / "ckpt").exists()


def test_a_teacher_missing_only_on_a_later_fold_and_seed_also_stops_at_the_start(tmp_path: Path) -> None:
    options = _options(tmp_path, seeds=[42, 7])
    rk.run_kfold(replace(options, arms=[_config("maestro_rgb")], seeds=[42]), FakeExecutor())
    executor = FakeExecutor()
    with pytest.raises(rk.RunnerError, match=r"est_kd_rgb/fold0/seed7/train"):
        rk.run_kfold(replace(options, arms=[_config("est_kd_rgb")]), executor)
    assert executor.calls == []


def test_an_interrupted_teacher_counts_as_missing(tmp_path: Path) -> None:
    """`best_model.pt` alone proves nothing: the trainer writes it after the first epoch."""
    options = _options(tmp_path, folds=[0], arms=("est_kd_rgbnir",))
    run = tmp_path / "ckpt" / "maestro_rgbnir" / "fold0" / "seed42"
    run.mkdir(parents=True)
    (run / "best_model.pt").write_bytes(b"after one epoch")
    executor = FakeExecutor()
    with pytest.raises(rk.RunnerError, match="run_complete.json"):
        rk.run_kfold(options, executor)
    assert executor.calls == []
    assert (run / "best_model.pt").read_bytes() == b"after one epoch"


def test_dry_run_reports_a_missing_teacher_it_would_not_train(tmp_path: Path) -> None:
    with pytest.raises(rk.RunnerError, match="maestro_rgb"):
        rk.run_kfold(_options(tmp_path, arms=("est_kd_rgb",), dry_run=True), FakeExecutor())


def test_a_student_run_alone_reuses_the_teachers_already_trained(tmp_path: Path) -> None:
    options = _options(tmp_path, seeds=[42])
    rk.run_kfold(replace(options, arms=[_config(arm) for arm in TEACHERS]), FakeExecutor())
    executor = FakeExecutor()
    summary = rk.run_kfold(replace(options, arms=[_config(arm) for arm in STUDENTS]), executor)
    assert {run[0] for run in executor.trained()} == set(STUDENTS)
    assert len(executor.teachers_seen) == 4
    assert not any(key.startswith("maestro") for key in summary)


def test_a_teacher_of_another_nir_variant_stops_the_student(tmp_path: Path) -> None:
    options = _options(tmp_path, folds=[0])
    rk.run_kfold(replace(options, arms=[_config("maestro_rgb")]), FakeExecutor())
    executor = FakeExecutor()
    with pytest.raises(rk.RunnerError, match=r"est_kd_rgb/fold0/seed42/train.*\(changed: command"):
        rk.run_kfold(replace(options, arms=[_config("est_kd_rgb")], nir_variant="nir_clahe"), executor)
    assert executor.calls == []


def test_a_retrained_teacher_stops_the_student_distilled_from_the_old_one(tmp_path: Path) -> None:
    options = _options(tmp_path, folds=[0], arms=("maestro_rgb", "est_kd_rgb"))
    rk.run_kfold(options, FakeExecutor())
    student = tmp_path / "ckpt" / "est_kd_rgb" / "fold0" / "seed42"
    marker = (student / "run_complete.json").read_text()
    (tmp_path / "ckpt" / "maestro_rgb" / "fold0" / "seed42" / "best_model.pt").write_bytes(b"other weights")

    executor = FakeExecutor()
    with pytest.raises(rk.RunnerError, match=r"est_kd_rgb/fold0/seed42/train.*\(changed: teacher\)"):
        rk.run_kfold(options, executor)
    assert [c for c in executor.calls if _is_train(c)] == []
    assert (student / "run_complete.json").read_text() == marker


# --------------------------------------------------------------------------
# Per-fold NIR normalization and data paths
# --------------------------------------------------------------------------


@pytest.mark.parametrize("variant", ["nir", "nir_clahe"])
def test_every_command_takes_the_nir_of_its_fold_and_variant(tmp_path: Path, variant: str) -> None:
    executor = FakeExecutor()
    options = _options(tmp_path, nir_variant=variant)
    rk.run_kfold(options, executor)
    dataset = tmp_path / "v4_dataset"
    fold_of = {str(options.folds_dir / f"fold_{k}" / "splits.json"): k for k in (0, 1)}
    steps = [c for c in executor.calls if _is_train(c) or _is_eval(c)]
    assert len(steps) == 6 * 2 * 3
    for cmd in steps:
        expected = _stats(fold_of[_value(cmd, "split_manifest")], variant, "roi")
        assert _value(cmd, "nir_mean") == str(expected["mean"])
        assert _value(cmd, "nir_std") == str(expected["std"])
        assert _value(cmd, "nir_dir") == str(dataset / variant)
        assert _value(cmd, "rgb_dir") == str(dataset / "rgb")
        assert _value(cmd, "nir_homography_path") == str(dataset / "homographies" / "identity.npy")


def test_nir_is_the_default_variant(tmp_path: Path) -> None:
    executor = FakeExecutor()
    rk.run_kfold(_options(tmp_path, folds=[0], arms=("maestro_rgbnir",)), executor)
    assert _value(executor.calls[0], "nir_mean") == str(_stats(0, "nir", "roi")["mean"])


def test_an_arm_without_the_crop_takes_the_full_frame_stats(tmp_path: Path) -> None:
    directory = _private_configs(tmp_path)
    config = _config("maestro_rgb", directory)
    data = yaml.safe_load(config.read_text())
    data["roi_crop"] = False
    config.write_text(yaml.safe_dump(data))
    executor = FakeExecutor()
    rk.run_kfold(_options(tmp_path, folds=[1], arms=[config]), executor)
    expected = _stats(1, "nir", "frame")
    for cmd in (c for c in executor.calls if _is_train(c) or _is_eval(c)):
        assert _value(cmd, "nir_mean") == str(expected["mean"])
        assert _value(cmd, "nir_std") == str(expected["std"])


def test_the_stats_follow_a_crop_changed_by_an_extra_override(tmp_path: Path) -> None:
    executor = FakeExecutor()
    options = _options(tmp_path, folds=[0], arms=("maestro_rgb",), extra_overrides=["roi_crop=false"])
    rk.run_kfold(options, executor)
    assert _value(executor.calls[0], "nir_mean") == str(_stats(0, "nir", "frame")["mean"])


def test_the_dataset_can_live_outside_the_parent_of_the_folds(tmp_path: Path) -> None:
    folds_dir = _make_dataset(tmp_path)
    moved = tmp_path / "elsewhere" / "folds_lot"
    moved.parent.mkdir()
    folds_dir.rename(moved)
    executor = FakeExecutor()
    options = _options(
        tmp_path, folds_dir=moved, dataset=tmp_path / "v4_dataset", folds=[0], arms=("maestro_rgb",),
    )
    rk.run_kfold(options, executor)
    assert _value(executor.calls[0], "rgb_dir") == str(tmp_path / "v4_dataset" / "rgb")


@pytest.mark.parametrize("meta", [None, "{}", '{"folds": {"0": {}}}', "not json"])
def test_folds_without_nir_stats_stop_the_v4_arms(tmp_path: Path, meta: str | None) -> None:
    options = _options(tmp_path, arms=("maestro_rgb",))
    path = options.folds_dir / "folds_meta.json"
    path.unlink() if meta is None else path.write_text(meta)
    executor = FakeExecutor()
    with pytest.raises(rk.RunnerError, match="make_v4_folds"):
        rk.run_kfold(options, executor)
    assert executor.calls == []


def test_folds_built_from_another_dataset_stop_the_v4_arms(tmp_path: Path) -> None:
    options = _options(tmp_path, arms=("maestro_rgb",))
    (tmp_path / "v4_dataset" / "index.csv").write_text("stem,lot\nmango_rgb_2,1\n")
    executor = FakeExecutor()
    with pytest.raises(rk.RunnerError, match="index.csv"):
        rk.run_kfold(options, executor)
    assert executor.calls == []


def test_a_dataset_without_index_stops_the_v4_arms(tmp_path: Path) -> None:
    options = _options(tmp_path, arms=("maestro_rgb",), dataset=tmp_path / "nowhere")
    with pytest.raises(rk.RunnerError, match="index.csv"):
        rk.run_kfold(options, FakeExecutor())


# --------------------------------------------------------------------------
# Predictions and v4 metrics
# --------------------------------------------------------------------------


def test_each_evaluation_saves_predictions_and_v4_metrics_read_them(tmp_path: Path) -> None:
    options = _options(tmp_path, folds=[1], arms=("maestro_rgbnir",))
    executor = FakeExecutor()
    rk.run_kfold(options, executor)
    report = tmp_path / "reports" / "maestro_rgbnir" / "fold1_seed42"
    for split, eval_cmd, metrics_cmd in (
        ("val", executor.calls[1], executor.calls[2]),
        ("test", executor.calls[3], executor.calls[4]),
    ):
        assert _is_eval(eval_cmd) and _flag(eval_cmd, "--split") == split
        # --override is nargs="*": every other flag has to come before it.
        assert eval_cmd.index("--save-predictions") < eval_cmd.index("--override")
        assert _flag(eval_cmd, "--output-dir") == str(report / split)
        assert metrics_cmd == [
            "python", "scripts/v4_metrics.py",
            "--predictions", str(report / split / "predictions.jsonl"),
            "--index", str(tmp_path / "v4_dataset" / "index.csv"),
            "--labels-dir", str(options.folds_dir / "fold_1" / "labels" / split),
            "--output", str(report / split / "metrics_v4.json"),
        ]


def test_an_evaluation_that_leaves_no_predictions_is_an_error(tmp_path: Path) -> None:
    options = _options(tmp_path, folds=[0], arms=("maestro_rgb",))
    with pytest.raises(rk.RunnerError, match="predictions.jsonl"):
        rk.run_kfold(options, FakeExecutor(omit=("predictions.jsonl",)))


def test_a_metrics_step_that_leaves_no_output_is_an_error(tmp_path: Path) -> None:
    options = _options(tmp_path, folds=[0], arms=("maestro_rgb",))
    with pytest.raises(rk.RunnerError, match="metrics_v4.json"):
        rk.run_kfold(options, FakeExecutor(omit=("metrics_v4.json",)))


def test_a_failed_metrics_step_stops_the_runner_and_resumes_there(tmp_path: Path) -> None:
    options = _options(tmp_path, folds=[0], arms=("maestro_rgb",))
    with pytest.raises(rk.RunnerError, match=r"maestro_rgb/fold0/seed42/metrics_val failed with exit code 3"):
        rk.run_kfold(options, FakeExecutor(fail_on="scripts/v4_metrics.py"))
    executor = FakeExecutor()
    summary = rk.run_kfold(options, executor)
    assert summary["maestro_rgb/fold0/seed42/eval_val"] == "skipped"
    assert summary["maestro_rgb/fold0/seed42/metrics_val"] == "ran"
    assert not any(_is_train(c) for c in executor.calls)


# --------------------------------------------------------------------------
# Resume
# --------------------------------------------------------------------------


def test_rerunning_the_same_command_skips_every_step(tmp_path: Path) -> None:
    options = _options(tmp_path, folds=[0])
    first = rk.run_kfold(options, FakeExecutor())
    assert len(first) == 6 * len(STEPS_PER_RUN)
    executor = FakeExecutor()
    summary = rk.run_kfold(options, executor)
    assert executor.calls == []
    assert list(summary) == list(first)
    assert set(summary.values()) == {"skipped"}


def test_only_the_missing_v4_metrics_are_rerun(tmp_path: Path) -> None:
    options = _options(tmp_path, folds=[0], arms=("maestro_rgb",))
    rk.run_kfold(options, FakeExecutor())
    (tmp_path / "reports" / "maestro_rgb" / "fold0_seed42" / "test" / "metrics_v4.json").unlink()
    executor = FakeExecutor()
    summary = rk.run_kfold(options, executor)
    assert [_is_metrics(c) for c in executor.calls] == [True]
    assert summary["maestro_rgb/fold0/seed42/metrics_test"] == "ran"
    assert summary["maestro_rgb/fold0/seed42/metrics_val"] == "skipped"


def test_missing_predictions_rerun_the_evaluation_and_its_metrics(tmp_path: Path) -> None:
    options = _options(tmp_path, folds=[0], arms=("maestro_rgb",))
    rk.run_kfold(options, FakeExecutor())
    (tmp_path / "reports" / "maestro_rgb" / "fold0_seed42" / "val" / "predictions.jsonl").unlink()
    executor = FakeExecutor()
    summary = rk.run_kfold(options, executor)
    assert [(_is_eval(c), _is_metrics(c)) for c in executor.calls] == [(True, False), (False, True)]
    assert summary["maestro_rgb/fold0/seed42/eval_val"] == "ran"
    assert summary["maestro_rgb/fold0/seed42/metrics_val"] == "ran"
    assert summary["maestro_rgb/fold0/seed42/eval_test"] == "skipped"


def test_a_student_resumes_without_retraining_the_finished_teacher(tmp_path: Path) -> None:
    options = _options(tmp_path, folds=[0], arms=("maestro_rgb", "est_kd_rgb"))
    with pytest.raises(rk.RunnerError, match="est_kd_rgb/fold0/seed42/train"):
        rk.run_kfold(options, FakeExecutor(fail_on="src.training.kd_train"))
    executor = FakeExecutor()
    summary = rk.run_kfold(options, executor)
    assert {key: status for key, status in summary.items() if key.startswith("maestro")} == {
        f"maestro_rgb/fold0/seed42/{step}": "skipped" for step in STEPS_PER_RUN
    }
    assert summary["est_kd_rgb/fold0/seed42/train"] == "ran"
    assert executor.trained() == [("est_kd_rgb", "fold0", "seed42")]


def test_a_kd_run_without_marker_is_retrained_like_any_other(tmp_path: Path) -> None:
    options = _options(tmp_path, folds=[0], arms=("maestro_rgb", "est_kd_rgb"))
    rk.run_kfold(options, FakeExecutor())
    student = tmp_path / "ckpt" / "est_kd_rgb" / "fold0" / "seed42"
    (student / "run_complete.json").unlink()
    executor = FakeExecutor()
    rk.run_kfold(options, executor)
    assert executor.trained() == [("est_kd_rgb", "fold0", "seed42")]
    assert (student.with_name("seed42.interrupted-1") / "best_model.pt").is_file()


def test_the_markers_of_a_kd_run_record_its_teacher_and_its_evaluation_config(tmp_path: Path) -> None:
    options = _options(tmp_path, folds=[0], arms=("maestro_rgb", "est_kd_rgb"))
    executor = FakeExecutor()
    rk.run_kfold(options, executor)
    student = tmp_path / "ckpt" / "est_kd_rgb" / "fold0" / "seed42"
    teacher = tmp_path / "ckpt" / "maestro_rgb" / "fold0" / "seed42" / "best_model.pt"
    identity = json.loads((student / "run_complete.json").read_text())["identity"]
    assert identity["teacher"] == hashlib.sha256(teacher.read_bytes()).hexdigest()
    report = tmp_path / "reports" / "est_kd_rgb" / "fold0_seed42" / "val"
    eval_identity = json.loads((report / "eval_complete.json").read_text())["identity"]
    assert eval_identity["run"] == identity["sha256"]
    assert eval_identity["eval_config"] == hashlib.sha256(_config("est_etiquetas").read_bytes()).hexdigest()
    metrics_identity = json.loads((report / "metrics_v4_complete.json").read_text())["identity"]
    assert metrics_identity["eval"] == eval_identity["sha256"]


def test_a_changed_evaluation_config_stops_the_kd_evaluations(tmp_path: Path) -> None:
    directory = _private_configs(tmp_path)
    arms = [_config("maestro_rgb", directory), _config("est_kd_rgb", directory)]
    options = _options(tmp_path, folds=[0], arms=arms)
    rk.run_kfold(options, FakeExecutor())
    config = _config("est_etiquetas", directory)
    config.write_text(config.read_text() + "# edit\n")
    executor = FakeExecutor()
    with pytest.raises(rk.RunnerError, match=r"est_kd_rgb/fold0/seed42/eval_val.*\(changed: eval_config\)"):
        rk.run_kfold(options, executor)
    assert executor.calls == []


def test_changed_folds_stop_a_finished_v4_run(tmp_path: Path) -> None:
    options = _options(tmp_path, folds=[0], arms=("maestro_rgb",))
    rk.run_kfold(options, FakeExecutor())
    path = options.folds_dir / "folds_meta.json"
    meta = json.loads(path.read_text())
    meta["labels_sha256"] = "other"
    path.write_text(json.dumps(meta))
    executor = FakeExecutor()
    with pytest.raises(rk.RunnerError, match=r"maestro_rgb/fold0/seed42/train.*\(changed: folds_meta\)"):
        rk.run_kfold(options, executor)
    assert executor.calls == []


def test_a_foreign_metrics_marker_stops_the_runner(tmp_path: Path) -> None:
    options = _options(tmp_path, folds=[0], arms=("maestro_rgb",))
    rk.run_kfold(options, FakeExecutor())
    marker = tmp_path / "reports" / "maestro_rgb" / "fold0_seed42" / "val" / "metrics_v4_complete.json"
    stale = json.loads(marker.read_text())
    stale["identity"].update(sha256="0" * 64, command="0" * 64)
    marker.write_text(json.dumps(stale))
    with pytest.raises(rk.RunnerError, match=r"metrics_val.*\(changed: command\)"):
        rk.run_kfold(options, FakeExecutor())
    marker.unlink()
    with pytest.raises(rk.RunnerError, match=r"metrics_val.*no identity"):
        rk.run_kfold(options, FakeExecutor())


# --------------------------------------------------------------------------
# Dry run and CLI
# --------------------------------------------------------------------------


def _plan(out: str) -> list[str]:
    """The step keys of a dry run, in the order they were printed."""
    return [line[1:].split("]", 1)[0] for line in out.splitlines() if line.startswith("[")]


def test_dry_run_prints_the_full_ordered_plan_and_creates_nothing(tmp_path: Path, capsys) -> None:
    executor = FakeExecutor()
    summary = rk.run_kfold(_options(tmp_path, seeds=[42, 7], dry_run=True), executor)
    assert executor.calls == []
    assert not (tmp_path / "ckpt").exists() and not (tmp_path / "reports").exists()
    expected = [
        f"{arm}/fold{fold}/seed{seed}/{step}"
        for fold in (0, 1) for seed in (42, 7) for arm in ARMS for step in STEPS_PER_RUN
    ]
    assert _plan(capsys.readouterr().out) == expected
    assert list(summary) == expected and set(summary.values()) == {"dry-run"}


def test_main_runs_the_arms_with_the_dataset_and_the_variant_of_the_command_line(tmp_path: Path, capsys) -> None:
    folds_dir = _make_dataset(tmp_path / "real")
    code = rk.main([
        "--folds-dir", str(folds_dir), "--dataset", str(folds_dir.parent), "--nir-variant", "nir_clahe",
        "--machine", str(RTX3080), "--dry-run", "--folds", "0",
        "--arms", str(_config("maestro_rgb")), str(_config("est_kd_rgb")),
        "--checkpoints-root", str(tmp_path / "ckpt"), "--reports-root", str(tmp_path / "reports"),
    ])
    assert code == 0
    out = capsys.readouterr().out
    assert f"nir_dir={folds_dir.parent / 'nir_clahe'}" in out
    assert f"nir_mean={_stats(0, 'nir_clahe', 'roi')['mean']}" in out
    assert "src.training.kd_train" in out and "scripts/v4_metrics.py" in out


def test_main_defaults_to_the_raw_nir_and_the_parent_of_the_folds(tmp_path: Path, capsys) -> None:
    folds_dir = _make_dataset(tmp_path / "real")
    code = rk.main([
        "--folds-dir", str(folds_dir), "--machine", str(RTX3080), "--dry-run", "--folds", "0",
        "--arms", str(_config("maestro_rgb")),
        "--checkpoints-root", str(tmp_path / "ckpt"), "--reports-root", str(tmp_path / "reports"),
    ])
    assert code == 0
    assert f"nir_dir={folds_dir.parent / 'nir'}" in capsys.readouterr().out


def test_main_refuses_an_unknown_nir_variant(tmp_path: Path, capsys) -> None:
    with pytest.raises(SystemExit) as error:
        rk.main(["--machine", str(RTX3080), "--nir-variant", "clahe"])
    assert error.value.code == 2
    assert "nir_clahe" in capsys.readouterr().err


def test_main_returns_nonzero_and_names_the_missing_teacher(tmp_path: Path, capsys) -> None:
    folds_dir = _make_dataset(tmp_path / "real")
    code = rk.main([
        "--folds-dir", str(folds_dir), "--machine", str(RTX3080), "--dry-run",
        "--arms", str(_config("est_kd_rgbnir")),
        "--checkpoints-root", str(tmp_path / "ckpt"), "--reports-root", str(tmp_path / "reports"),
    ])
    assert code == 1
    assert "maestro_rgbnir" in capsys.readouterr().err


# --------------------------------------------------------------------------
# The commands against the real entry points
# --------------------------------------------------------------------------


def _argv_parses(monkeypatch, module, cmd: list[str], start: int):
    monkeypatch.setattr(sys, "argv", ["prog", *cmd[start:]])
    return module.parse_args()


@pytest.mark.parametrize("arm", ARMS)
def test_the_commands_parse_on_their_real_entry_points(tmp_path: Path, monkeypatch, arm: str) -> None:
    arms = (TEACHER_OF[arm], arm) if arm in TEACHER_OF else (arm,)
    executor = FakeExecutor()
    rk.run_kfold(_options(tmp_path, arms=arms, folds=[0]), executor)
    train_cmd = next(c for c in executor.calls if _is_train(c) and _run_of(c)[0] == arm)
    module = kd_train if arm in TEACHER_OF else train
    args = _argv_parses(monkeypatch, module, train_cmd, 3)
    # kd_train splits every override on "=" and expects exactly two parts.
    assert all(token.count("=") == 1 for token in args.override)

    eval_cmd = next(c for c in executor.calls if _is_eval(c) and f"/{arm}/" in _flag(c, "--output-dir"))
    args = _argv_parses(monkeypatch, evaluate_checkpoint, eval_cmd, 2)
    assert args.save_predictions is True
    assert args.model == ("master" if arm in TEACHERS else "student")

    metrics_cmd = next(c for c in executor.calls if _is_metrics(c))
    args = v4_metrics.parse_args(metrics_cmd[2:])
    assert args.predictions.endswith("predictions.jsonl") and args.output.endswith("metrics_v4.json")


@pytest.mark.parametrize("arm", ARMS)
def test_every_override_exists_on_the_config_that_receives_it(tmp_path: Path, arm: str) -> None:
    """train.py and kd_train.py silently drop an unknown key, so a misspelt one would
    train the wrong arm without a word."""
    arms = (TEACHER_OF[arm], arm) if arm in TEACHER_OF else (arm,)
    executor = FakeExecutor()
    rk.run_kfold(_options(tmp_path, arms=arms, folds=[0], epochs=1), executor)
    own = [
        c for c in executor.calls
        if (_is_train(c) and _run_of(c)[0] == arm) or (_is_eval(c) and f"/{arm}/" in _flag(c, "--output-dir"))
    ]
    for cmd in own:
        loader = KDConfig if "src.training.kd_train" in cmd else TrainingConfig
        config = loader.from_yaml(_flag(cmd, "--config"))
        for token in _overrides(cmd):
            assert hasattr(config, token.split("=", 1)[0]), token


# --------------------------------------------------------------------------
# What the existing arms keep
# --------------------------------------------------------------------------


def test_existing_arms_ignore_the_v4_options_and_need_no_nir_stats(tmp_path: Path) -> None:
    folds_dir = tmp_path / "plain_folds"
    (folds_dir / "fold_0" / "labels" / "train").mkdir(parents=True)
    (folds_dir / "fold_0" / "splits.json").write_text("{}")
    base = dict(
        folds_dir=folds_dir, arms=[TWOSTREAM], folds=None, seeds=[42], machine=RTX3080, epochs=None,
        python="py", extra_overrides=[],
    )
    plain = FakeExecutor()
    summary = rk.run_kfold(
        rk.RunnerOptions(**base, checkpoints_root=tmp_path / "c", reports_root=tmp_path / "r"), plain,
    )
    assert list(summary) == [f"twostream/fold0/seed42/{step}" for step in ("train", "eval_val", "eval_test")]

    with_options = FakeExecutor()
    rk.run_kfold(
        rk.RunnerOptions(
            **base, checkpoints_root=tmp_path / "c2", reports_root=tmp_path / "r2",
            dataset=tmp_path / "nowhere", nir_variant="nir_clahe",
        ),
        with_options,
    )

    def normalized(calls: list[list[str]], checkpoints: str, reports: str) -> list[list[str]]:
        return [
            [t.replace(str(tmp_path / checkpoints), "C").replace(str(tmp_path / reports), "R") for t in cmd]
            for cmd in calls
        ]

    assert normalized(plain.calls, "c", "r") == normalized(with_options.calls, "c2", "r2")
    assert not any("--save-predictions" in c or _is_metrics(c) for c in plain.calls)
    for cmd in plain.calls:
        for key in ("nir_mean", "nir_std", "nir_dir", "rgb_dir", "modality_ablation", "teacher_checkpoint"):
            assert _value(cmd, key) is None


def test_eval_command_of_existing_arms_keeps_its_exact_shape(tmp_path: Path) -> None:
    fold, run, out = tmp_path / "f", tmp_path / "run", tmp_path / "out"
    assert rk.build_eval_command("py", TWOSTREAM, run / "best_model.pt", "val", fold, run / "ev", out) == [
        "py", "scripts/evaluate_checkpoint.py",
        "--config", str(TWOSTREAM), "--checkpoint", str(run / "best_model.pt"),
        "--split", "val", "--model", "master", "--output-dir", str(out),
        "--override",
        f"labels_dir={fold / 'labels'}", f"split_manifest={fold / 'splits.json'}", f"output_dir={run / 'ev'}",
    ]


def test_the_identity_of_existing_arms_keeps_its_parts(tmp_path: Path) -> None:
    """A new part would invalidate every marker on disk."""
    fold = tmp_path / "fold_0"
    (fold / "labels").mkdir(parents=True)
    (fold / "splits.json").write_text("{}")
    identity = rk.run_identity(["py"], TWOSTREAM, RTX3080, fold)
    assert sorted(identity) == ["command", "config", "fold", "machine", "sha256"]
    assert sorted(rk.eval_identity(identity["sha256"], "val", ["py"])) == ["command", "run", "sha256", "split"]
