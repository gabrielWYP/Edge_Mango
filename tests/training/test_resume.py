"""An interrupted run, continued from `resume_state.pt`, must end as if never interrupted."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.training.config import DAMAGE_CLASS_ID, TrainingConfig
from src.training.loop import Trainer
from src.training.resume import RESUME_STATE, atomic_torch_save, resume_config_mismatches


class _Crash(Exception):
    """Stands in for whatever killed the process."""


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)
        with torch.no_grad():
            self.linear.weight.fill_(0.3)
            self.linear.bias.fill_(-0.1)

    def freeze_backbone(self, freeze_stages: int) -> None:
        pass

    def unfreeze_backbone_stages(self, stages, unfreeze_stem: bool = False) -> None:
        pass

    def forward(self, rgb, nir=None):
        return {"preds": self.linear(rgb)}


class _RandomLoader:
    """Draws every batch from the global torch stream, like the real sampler and workers."""

    def __iter__(self):
        for _ in range(4):
            x = torch.randn(2, 1)
            yield {
                "rgb": x,
                "nir": torch.zeros_like(x),
                "bboxes": [torch.zeros(0, 4)] * 2,
                "labels": [2.0 * x + 1.0 + 0.1 * torch.randn(2, 1)],
            }


def _criterion(preds, targets):
    loss = ((preds - targets["labels"][0]) ** 2).mean()
    return loss, {"cls_loss": 0.0, "box_loss": 0.0}


def _trainer(output_dir: Path, *, crash_at: int | None = None, resume: bool = False, **config_kwargs) -> Trainer:
    """Validation is scripted from the validation count of this process, so that
    `crash_at` fires inside its Nth epoch, after training and before the epoch's state is saved."""
    config = TrainingConfig(
        **{
            "model_type": "student",
            "epochs": 6,
            "lr": 0.05,
            "batch_size": 2,
            "effective_batch": 2,
            "num_workers": 0,
            "precision": "fp32",
            "warmup_epochs": 2,
            "patience": 50,
            "save_interval": 1000,
            "device": "cpu",
            "nir_homography_path": None,
            "output_dir": str(output_dir),
            **config_kwargs,
        }
    )
    trainer = Trainer(_Model(), config, _RandomLoader(), _RandomLoader())
    trainer.criterion = _criterion
    trainer.writer = None
    trainer.seen_lrs = []
    if resume:
        trainer.resume_from = output_dir / RESUME_STATE

    def validate(epoch, phase):
        trainer.seen_lrs.append((phase, epoch))
        if crash_at is not None and len(trainer.seen_lrs) == crash_at:
            raise _Crash
        # Depends on the weights, so a wrong continuation changes which epoch is selected.
        score = float(-abs(trainer.eval_model.linear.weight.item() - 2.0))
        return {"map50": score, "map_50_95": 0.0, "per_class_ap_50": {DAMAGE_CLASS_ID: score}}

    trainer._validate = validate
    return trainer


def _run_interrupted(tmp_path: Path, crash_at: int, **config_kwargs) -> Trainer:
    torch.manual_seed(7)
    with pytest.raises(_Crash):
        _trainer(tmp_path, crash_at=crash_at, **config_kwargs).fit()
    # A new process seeds the stream again and rebuilds everything.
    torch.manual_seed(7)
    trainer = _trainer(tmp_path, resume=True, **config_kwargs)
    trainer.fit()
    return trainer


def _run_straight(tmp_path: Path, **config_kwargs) -> Trainer:
    torch.manual_seed(7)
    trainer = _trainer(tmp_path, **config_kwargs)
    trainer.fit()
    return trainer


def _assert_same_outcome(a: Trainer, b: Trainer) -> None:
    assert a.model.state_dict().keys() == b.model.state_dict().keys()
    for key, value in a.model.state_dict().items():
        assert torch.equal(value, b.model.state_dict()[key]), key
    if a.ema is not None:
        assert a.ema.updates == b.ema.updates
        for key, value in a.ema.state_dict().items():
            assert torch.equal(value, b.ema.state_dict()[key]), key
    assert a.best_score == b.best_score
    assert a.global_step == b.global_step
    assert a.loss_history.total_loss == b.loss_history.total_loss
    assert a.loss_history.epoch == b.loss_history.epoch
    assert a.loss_history.map50 == b.loss_history.map50


@pytest.mark.parametrize("crash_at", [2, 3, 5])  # during warmup, right after it, near the end
@pytest.mark.parametrize("ema", [False, True])
def test_a_continued_run_ends_exactly_like_an_uninterrupted_one(tmp_path, crash_at, ema) -> None:
    straight = _run_straight(tmp_path / "straight", ema=ema, ema_tau=5.0)
    continued = _run_interrupted(tmp_path / "continued", crash_at, ema=ema, ema_tau=5.0)
    _assert_same_outcome(continued, straight)


def test_the_continuation_retrains_only_the_lost_epoch_onwards(tmp_path) -> None:
    continued = _run_interrupted(tmp_path, crash_at=4)
    assert continued.seen_lrs == [(1, epoch) for epoch in (4, 5, 6)]


def test_the_best_checkpoint_of_a_continued_run_matches_and_says_it_was_continued(tmp_path) -> None:
    straight = _run_straight(tmp_path / "straight")
    continued = _run_interrupted(tmp_path / "continued", crash_at=4)

    saved = torch.load(tmp_path / "continued" / "best_model.pt", weights_only=False)
    reference = torch.load(tmp_path / "straight" / "best_model.pt", weights_only=False)
    assert saved["epoch"] == reference["epoch"]
    for key, value in reference["model_state_dict"].items():
        assert torch.equal(value, saved["model_state_dict"][key]), key

    assert straight.resume_history == () and reference["resume_history"] == []
    assert [(r["phase"], r["epoch"]) for r in continued.resume_history] == [(1, 3)]
    assert [(r["phase"], r["epoch"]) for r in saved["resume_history"]] == [(1, 3)]


def test_a_run_can_be_interrupted_more_than_once(tmp_path) -> None:
    straight = _run_straight(tmp_path / "straight")
    run = tmp_path / "continued"
    torch.manual_seed(7)
    with pytest.raises(_Crash):
        _trainer(run, crash_at=2).fit()
    torch.manual_seed(7)
    with pytest.raises(_Crash):
        _trainer(run, crash_at=2, resume=True).fit()
    torch.manual_seed(7)
    continued = _trainer(run, resume=True)
    continued.fit()

    _assert_same_outcome(continued, straight)
    assert [r["epoch"] for r in continued.resume_history] == [1, 2]


TWO_PHASE = {
    "model_type": "master",
    "schedule": "two_phase",
    "epochs_phase1": 3,
    "epochs_phase2": 3,
    "lr_phase1": 0.05,
    "lr_phase2": 0.01,
    "in_channels": 4,
}


@pytest.mark.parametrize("crash_at", [2, 4, 5])  # phase 1, first and second epoch of phase 2
def test_two_phase_continues_in_the_right_phase(tmp_path, crash_at, monkeypatch) -> None:
    monkeypatch.setattr(Trainer, "_module_grad_norms", lambda self: {})
    straight = _run_straight(tmp_path / "straight", **TWO_PHASE)
    continued = _run_interrupted(tmp_path / "continued", crash_at, **TWO_PHASE)
    _assert_same_outcome(continued, straight)
    assert continued.loss_history.phase == [1, 1, 1, 2, 2, 2]


def test_a_run_that_had_finished_trains_nothing_more(tmp_path) -> None:
    straight = _run_straight(tmp_path)
    again = _trainer(tmp_path, resume=True)
    again.fit()
    assert again.seen_lrs == []
    _assert_same_outcome(again, straight)


def test_a_run_that_had_stopped_early_stays_stopped(tmp_path) -> None:
    # patience=1: the second epoch without improvement ends it.
    first = _trainer(tmp_path, patience=1)
    scores = iter([0.5, 0.1])
    first._validate = lambda epoch, phase: {
        "map50": (s := next(scores)), "map_50_95": 0.0, "per_class_ap_50": {DAMAGE_CLASS_ID: s},
    }
    first.fit()

    again = _trainer(tmp_path, patience=1, resume=True)
    again.fit()
    assert again.seen_lrs == []


def test_the_learning_rate_continues_on_the_schedule(tmp_path) -> None:
    torch.manual_seed(7)
    with pytest.raises(_Crash):
        _trainer(tmp_path / "a", crash_at=4).fit()
    state = torch.load(tmp_path / "a" / RESUME_STATE, weights_only=False)

    torch.manual_seed(7)
    reference = _trainer(tmp_path / "b", crash_at=4)
    with pytest.raises(_Crash):
        reference.fit()
    reference_state = torch.load(tmp_path / "b" / RESUME_STATE, weights_only=False)

    assert state["epoch"] == 3
    assert state["optimizer"]["param_groups"][0]["lr"] == reference_state["optimizer"]["param_groups"][0]["lr"]
    assert state["optimizer"]["param_groups"][0]["lr"] != 0.05 / 2  # no longer the first warmup value


@pytest.mark.parametrize(
    "change, name", [({"lr": 0.01}, "lr"), ({"seed": 43}, "seed"), ({"ema": True}, "ema")],
)
def test_other_settings_are_refused_by_name(tmp_path, change, name) -> None:
    torch.manual_seed(7)
    with pytest.raises(_Crash):
        _trainer(tmp_path, crash_at=2).fit()
    with pytest.raises(ValueError, match=name):
        _trainer(tmp_path, resume=True, **change).fit()


def test_another_experiment_file_is_refused(tmp_path) -> None:
    first = _trainer(tmp_path, crash_at=2)
    first.experiment_sha256 = "aaa"
    with pytest.raises(_Crash):
        first.fit()
    again = _trainer(tmp_path, resume=True)
    again.experiment_sha256 = "bbb"
    with pytest.raises(ValueError, match="experiment_sha256"):
        again.fit()


def test_a_missing_state_is_an_error_not_a_fresh_start(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match=RESUME_STATE):
        _trainer(tmp_path, resume=True).fit()


def test_the_output_dir_may_differ() -> None:
    assert resume_config_mismatches({"output_dir": "a", "lr": 1}, {"output_dir": "b", "lr": 1}) == []
    assert resume_config_mismatches({"lr": 1}, {"lr": 2, "seed": 3}) == ["lr", "seed"]


def test_a_save_killed_midway_leaves_the_previous_file_intact(tmp_path, monkeypatch) -> None:
    path = tmp_path / "best_model.pt"
    atomic_torch_save({"epoch": 1}, path)

    def dies_midway(obj, target):
        Path(target).write_bytes(b"truncat")
        raise OSError("killed")

    monkeypatch.setattr(torch, "save", dies_midway)
    with pytest.raises(OSError):
        atomic_torch_save({"epoch": 2}, path)
    monkeypatch.undo()

    assert torch.load(path, weights_only=False) == {"epoch": 1}


def test_the_same_pilot_overrides_are_not_another_experiment(tmp_path) -> None:
    torch.manual_seed(7)
    settings = {"pixel_cache_dir": "/c/rgb", "ignore_classes": [0]}
    with pytest.raises(_Crash):
        _trainer(tmp_path, crash_at=2, **settings).fit()

    _trainer(tmp_path, resume=True, **settings).fit()

    with pytest.raises(ValueError, match="ignore_classes"):
        _trainer(tmp_path, resume=True, pixel_cache_dir="/c/rgb", ignore_classes=[]).fit()


def _record_lrs(trainer: Trainer) -> list[float]:
    """The learning rate each epoch really trains with."""
    seen: list[float] = []
    train_epoch = trainer._train_epoch

    def spy(optimizer, epoch, phase):
        seen.append(optimizer.param_groups[0]["lr"])
        return train_epoch(optimizer, epoch, phase)

    trainer._train_epoch = spy
    return seen


def test_the_warmup_rises_from_its_first_epoch_and_the_cosine_follows(tmp_path) -> None:
    trainer = _trainer(tmp_path)  # lr=0.05, warmup_epochs=2, epochs=6
    seen = _record_lrs(trainer)
    trainer.fit()

    assert seen[:3] == pytest.approx([0.025, 0.05, 0.05])
    assert seen[2] > seen[3] > seen[4] > seen[5] > 0


def test_a_continued_run_trains_with_the_same_learning_rates(tmp_path) -> None:
    straight = _trainer(tmp_path / "straight")
    expected = _record_lrs(straight)
    straight.fit()

    torch.manual_seed(7)
    with pytest.raises(_Crash):
        _trainer(tmp_path / "continued", crash_at=2).fit()
    continued = _trainer(tmp_path / "continued", resume=True)
    seen = _record_lrs(continued)
    continued.fit()

    assert seen == pytest.approx(expected[1:])
