"""Weight EMA (`src/training/ema.py`) and its integration in the trainers.

Why: with ~15-21 optimizer steps per epoch on the leave-one-lot-out folds,
the weights at the end of any single epoch are a noisy sample, so validation
AP flips from epoch to epoch and checkpoint selection on it is close to
random. An exponential moving average of the weights averages over the last
~1/(1 - decay) optimizer steps; when `ema=True` that average is what gets
validated, selected and written to `best_model.pt`.

Contract pinned here:
- parameters are blended, BatchNorm buffers are copied verbatim;
- the update runs once per OPTIMIZER step (never per micro-batch), and never
  touches the live training trajectory;
- validation, selection and checkpoints use the EMA weights when enabled,
  and the live weights — exactly as before — when not;
- eval-only trainers (no train loader) never build an EMA.
"""

from __future__ import annotations

import copy
import math
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.master.master_model import MasterModel
from src.models.student.student_model import StudentModel
from src.training.config import DAMAGE_CLASS_ID, TrainingConfig
from src.training.ema import ModelEMA
from src.training.kd_config import KDConfig
from src.training.kd_trainer import KDTrainer
from src.training.loop import Trainer


def _tiny_model() -> nn.Module:
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(3, 4), nn.BatchNorm1d(4), nn.Linear(4, 2))


def _perturb(model: nn.Module, seed: int = 1) -> None:
    """Move every parameter somewhere else, deterministically."""
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn(p.shape, generator=gen))


def _params(model: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in model.named_parameters()}


# --------------------------------------------------------------------------- #
# ModelEMA
# --------------------------------------------------------------------------- #


class TestConstruction:
    def test_shadow_is_an_independent_eval_copy(self):
        live = _tiny_model()
        ema = ModelEMA(live)

        assert ema.module is not live
        assert not ema.module.training
        assert all(not p.requires_grad for p in ema.module.parameters())
        for name, p in live.named_parameters():
            assert torch.equal(dict(ema.module.named_parameters())[name], p)

        _perturb(live)  # the live model moving must not move the shadow
        assert not torch.equal(ema.module[0].weight, live[0].weight)

    def test_live_model_is_left_untouched(self):
        live = _tiny_model()
        before = copy.deepcopy(live.state_dict())
        ModelEMA(live)
        assert all(torch.equal(before[k], v) for k, v in live.state_dict().items())
        assert live.training
        assert all(p.requires_grad for p in live.parameters())

    @pytest.mark.parametrize("decay", [0.0, 1.0, 1.2, -0.5])
    def test_decay_must_be_in_the_open_unit_interval(self, decay):
        with pytest.raises(ValueError, match="ema_decay"):
            ModelEMA(_tiny_model(), decay=decay)

    @pytest.mark.parametrize("tau", [0.0, -1.0])
    def test_tau_must_be_positive(self, tau):
        with pytest.raises(ValueError, match="ema_tau"):
            ModelEMA(_tiny_model(), tau=tau)


class TestDecaySchedule:
    def test_nothing_has_been_applied_before_the_first_update(self):
        assert ModelEMA(_tiny_model(), decay=0.99, tau=50.0).current_decay() == 0.0

    def test_warmup_formula(self):
        ema = ModelEMA(_tiny_model(), decay=0.99, tau=50.0)
        for n in (1, 10, 50, 500):
            ema.updates = n
            assert ema.current_decay() == pytest.approx(0.99 * (1 - math.exp(-n / 50.0)))

    def test_grows_monotonically_towards_the_asymptote(self):
        ema = ModelEMA(_tiny_model(), decay=0.99, tau=50.0)
        values = []
        for n in range(0, 2000, 25):
            ema.updates = n
            values.append(ema.current_decay())
        assert values == sorted(values)
        assert values[-1] == pytest.approx(0.99, abs=1e-6)
        assert values[-1] <= 0.99

    def test_early_updates_track_the_live_model_instead_of_the_init(self):
        """The reason for the warm-up: at update 1 the shadow is (almost)
        replaced by the live weights, not anchored to the random init."""
        live = _tiny_model()
        initial = live[0].weight.detach().clone()
        ema = ModelEMA(live, decay=0.999, tau=2.0)
        _perturb(live)
        ema.update(live)

        assert ema.current_decay() < 0.5
        shadow = ema.module[0].weight
        assert (shadow - live[0].weight).abs().sum() < (shadow - initial).abs().sum()


class TestUpdateMath:
    def test_one_update_blends_the_parameters(self):
        live = _tiny_model()
        ema = ModelEMA(live, decay=0.9, tau=10.0)
        initial = _params(ema.module)
        _perturb(live)

        ema.update(live)

        d = 0.9 * (1 - math.exp(-1 / 10.0))
        assert ema.updates == 1
        assert ema.current_decay() == pytest.approx(d)
        for name, p in ema.module.named_parameters():
            expected = initial[name] * d + dict(live.named_parameters())[name].detach() * (1 - d)
            assert torch.allclose(p, expected, atol=1e-6), name

    def test_a_sequence_of_updates_follows_the_recurrence(self):
        live = _tiny_model()
        ema = ModelEMA(live, decay=0.95, tau=4.0)
        expected = _params(live)
        for step in range(1, 6):
            _perturb(live, seed=step)
            ema.update(live)
            d = 0.95 * (1 - math.exp(-step / 4.0))
            for name, p in live.named_parameters():
                expected[name] = expected[name] * d + p.detach() * (1 - d)
        for name, p in ema.module.named_parameters():
            assert torch.allclose(p, expected[name], atol=1e-5), name

    def test_constant_live_weights_are_a_fixed_point(self):
        live = _tiny_model()
        ema = ModelEMA(live, decay=0.99, tau=5.0)
        reference = _params(live)
        for _ in range(20):
            ema.update(live)
        for name, p in ema.module.named_parameters():
            assert torch.allclose(p, reference[name], atol=1e-6), name

    def test_the_shadow_converges_to_new_constant_weights(self):
        live = _tiny_model()
        ema = ModelEMA(live, decay=0.9, tau=3.0)
        _perturb(live)
        for _ in range(400):
            ema.update(live)
        for name, p in live.named_parameters():
            assert torch.allclose(dict(ema.module.named_parameters())[name], p, atol=1e-4), name

    def test_frozen_live_parameters_are_still_tracked(self):
        """`freeze_backbone` flips `requires_grad` on the live model after the
        EMA was built; the update must not depend on it."""
        live = _tiny_model()
        ema = ModelEMA(live, decay=0.9, tau=5.0)
        for p in live[0].parameters():
            p.requires_grad_(False)
        before = ema.module[0].weight.clone()
        with torch.no_grad():
            live[0].weight.add_(1.0)

        ema.update(live)

        assert not torch.equal(ema.module[0].weight, before)

    def test_update_builds_no_autograd_graph(self):
        live = _tiny_model()
        ema = ModelEMA(live)
        ema.update(live)
        assert all(p.grad_fn is None and not p.requires_grad for p in ema.module.parameters())

    def test_resuming_continues_the_warmup(self):
        live = _tiny_model()
        ema = ModelEMA(live, decay=0.99, tau=50.0, updates=40)
        ema.update(live)
        assert ema.updates == 41
        assert ema.current_decay() == pytest.approx(0.99 * (1 - math.exp(-41 / 50.0)))


class TestBuffers:
    def test_batch_norm_buffers_are_copied_not_averaged(self):
        live = _tiny_model()
        ema = ModelEMA(live, decay=0.9, tau=10.0)
        live.train()
        for _ in range(3):
            live(torch.randn(8, 3) * 3 + 2)  # moves running_mean/var, counts batches

        ema.update(live)

        live_bn, ema_bn = live[1], ema.module[1]
        assert not torch.equal(live_bn.running_mean, torch.zeros(4))  # the stats did move
        assert torch.equal(ema_bn.running_mean, live_bn.running_mean)
        assert torch.equal(ema_bn.running_var, live_bn.running_var)
        assert ema_bn.num_batches_tracked.item() == live_bn.num_batches_tracked.item() == 3

    def test_parameters_are_still_blended_while_buffers_are_copied(self):
        live = _tiny_model()
        ema = ModelEMA(live, decay=0.9, tau=10.0)
        live.train()
        live(torch.randn(8, 3))
        _perturb(live)

        ema.update(live)

        assert not torch.equal(ema.module[0].weight, live[0].weight)
        assert torch.equal(ema.module[1].running_var, live[1].running_var)


class TestStateDict:
    def test_same_keys_and_loads_strictly_into_a_fresh_model(self):
        live = _tiny_model()
        ema = ModelEMA(live)
        assert list(ema.state_dict()) == list(live.state_dict())

        fresh = _tiny_model()
        _perturb(fresh, seed=9)
        fresh.load_state_dict(ema.state_dict(), strict=True)
        for k, v in ema.state_dict().items():
            assert torch.equal(fresh.state_dict()[k], v)

    def test_the_shadow_stays_in_eval_mode_when_the_live_model_trains(self):
        live = _tiny_model()
        ema = ModelEMA(live)
        live.train()
        ema.update(live)
        live.train()
        assert not ema.module.training
        assert all(not m.training for m in ema.module.modules())


# --------------------------------------------------------------------------- #
# Trainer integration - cheap linear model (student path, fake criterion)
# --------------------------------------------------------------------------- #

_XS = [0.5, -0.3, 1.2, 0.1, -0.7, 0.9, 0.2, -0.4]
_YS = [2.0 * x + 1.0 for x in _XS]


class _LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)
        with torch.no_grad():
            self.linear.weight.fill_(0.3)
            self.linear.bias.fill_(-0.1)

    def freeze_backbone(self, freeze_stages: int) -> None:
        pass

    def forward(self, rgb, nir=None):
        return {"preds": self.linear(rgb)}


def _mse_criterion(preds, targets):
    loss = ((preds - targets["labels"][0]) ** 2).mean()
    return loss, {"cls_loss": 0.0, "box_loss": 0.0}


class _Loader:
    """The fixed 8-sample set, in micro-batches of `group_size`."""

    def __init__(self, group_size: int):
        self.group_size = group_size

    def __iter__(self):
        for i in range(0, len(_XS), self.group_size):
            x = torch.tensor(_XS[i : i + self.group_size]).unsqueeze(1)
            y = torch.tensor(_YS[i : i + self.group_size]).unsqueeze(1)
            yield {
                "rgb": x,
                "nir": torch.zeros_like(x),
                "bboxes": [torch.zeros(0, 4)] * len(x),
                "labels": [y],
            }


def _linear_trainer(
    tmp_path, *, batch_size=2, effective_batch=8, train_loader="default", **config_kwargs
) -> Trainer:
    config = TrainingConfig(
        **{
            "model_type": "student",
            "batch_size": batch_size,
            "effective_batch": effective_batch,
            "num_workers": 0,
            "precision": "fp32",
            "grad_clip": 1e6,
            "warmup_epochs": 1,
            "patience": 50,
            "save_interval": 1000,
            "device": "cpu",
            "nir_homography_path": None,
            "output_dir": str(tmp_path),
            **config_kwargs,
        }
    )
    trainer = Trainer(
        model=_LinearModel(),
        config=config,
        train_loader=_Loader(batch_size) if train_loader == "default" else train_loader,
        val_loader=_Loader(batch_size),
    )
    trainer.criterion = _mse_criterion
    trainer.writer = None
    return trainer


class TestTrainerWiring:
    def test_default_builds_no_ema_and_evaluates_the_live_model(self, tmp_path):
        trainer = _linear_trainer(tmp_path)
        assert trainer.ema is None
        assert trainer.eval_model is trainer.model

    def test_enabled_builds_the_shadow_from_the_config(self, tmp_path):
        trainer = _linear_trainer(tmp_path, ema=True, ema_decay=0.97, ema_tau=12.0)
        assert isinstance(trainer.ema, ModelEMA)
        assert (trainer.ema.decay, trainer.ema.tau) == (0.97, 12.0)
        assert trainer.eval_model is trainer.ema.module
        assert trainer.eval_model is not trainer.model

    def test_eval_only_trainer_builds_no_ema_even_when_the_config_asks(self, tmp_path):
        """`scripts/evaluate_checkpoint.py` builds a Trainer without a train
        loader around weights that already ARE the EMA weights."""
        trainer = _linear_trainer(tmp_path, ema=True, train_loader=None)
        assert trainer.ema is None
        assert trainer.eval_model is trainer.model


class TestUpdateCadence:
    def test_one_update_per_optimizer_step_not_per_micro_batch(self, tmp_path):
        # batch_size=2, effective_batch=8 -> accumulate 4; 8 samples = 4
        # micro-batches = ONE optimizer step.
        trainer = _linear_trainer(tmp_path, batch_size=2, effective_batch=8, ema=True)
        optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.1)

        result = trainer._train_epoch(optimizer, epoch=1, phase=1)

        assert result["steps_taken"] == 1.0
        assert trainer.ema.updates == 1

    def test_partial_accumulation_window_is_flushed_and_updates_once(self, tmp_path):
        # batch_size=1, effective_batch=3 -> 8 micro-batches = 2 full steps
        # plus a flushed partial one.
        trainer = _linear_trainer(tmp_path, batch_size=1, effective_batch=3, ema=True)
        optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.1)

        result = trainer._train_epoch(optimizer, epoch=1, phase=1)

        assert result["steps_taken"] == 3.0
        assert trainer.ema.updates == 3

    def test_no_accumulation_updates_every_batch(self, tmp_path):
        trainer = _linear_trainer(tmp_path, batch_size=2, effective_batch=2, ema=True)
        optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.1)

        trainer._train_epoch(optimizer, epoch=1, phase=1)

        assert trainer.ema.updates == 4

    def test_the_shadow_lags_the_live_weights(self, tmp_path):
        trainer = _linear_trainer(
            tmp_path, batch_size=2, effective_batch=2, ema=True, ema_decay=0.9, ema_tau=20.0
        )
        optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.1)
        initial = trainer.model.linear.weight.item()

        trainer._train_epoch(optimizer, epoch=1, phase=1)

        live = trainer.model.linear.weight.item()
        shadow = trainer.ema.module.linear.weight.item()
        assert live != initial
        assert min(initial, live) < shadow < max(initial, live)

    def test_ema_never_changes_the_live_training_trajectory(self, tmp_path):
        """The EMA is observational: with it on or off the live weights after
        training are bit-identical."""
        finals = []
        for enabled in (False, True):
            trainer = _linear_trainer(
                tmp_path / str(enabled), batch_size=2, effective_batch=4, ema=enabled
            )
            optimizer = torch.optim.AdamW(trainer.model.parameters(), lr=0.05)
            for epoch in (1, 2, 3):
                trainer._train_epoch(optimizer, epoch=epoch, phase=1)
            finals.append(copy.deepcopy(trainer.model.state_dict()))
        assert all(torch.equal(finals[0][k], finals[1][k]) for k in finals[0])


def _val(map50: float) -> dict:
    return {
        "map50": map50,
        "map_50_95": 0.0,
        "per_class_ap_50": {0: map50, DAMAGE_CLASS_ID: map50},
    }


class TestCheckpoints:
    """Real `_train_epoch` and `_save_checkpoint`; validation is scripted so
    each epoch improves and best_model.pt is rewritten every epoch."""

    def _fit(self, tmp_path, **config_kwargs) -> Trainer:
        trainer = _linear_trainer(
            tmp_path, batch_size=2, effective_batch=2, **config_kwargs
        )
        scripted = iter([_val(0.1), _val(0.2), _val(0.3)])
        trainer._validate = lambda epoch, phase: next(scripted)
        trainer._train_phase(phase=1, epochs=3, lr=0.05)
        return trainer

    def test_best_model_holds_the_ema_weights_and_says_so(self, tmp_path):
        trainer = self._fit(tmp_path, ema=True, ema_decay=0.95, ema_tau=5.0)

        checkpoint = torch.load(tmp_path / "best_model.pt", weights_only=False)
        shadow, live = trainer.ema.state_dict(), trainer.model.state_dict()
        saved = checkpoint["model_state_dict"]

        assert checkpoint["ema"] is True
        assert checkpoint["ema_updates"] == trainer.ema.updates == 12
        assert all(torch.equal(saved[k], shadow[k]) for k in shadow)
        assert not all(torch.equal(saved[k], live[k]) for k in live)
        assert checkpoint["config"]["ema"] is True
        assert checkpoint["config"]["ema_decay"] == 0.95

    def test_periodic_checkpoints_hold_the_ema_weights_too(self, tmp_path):
        trainer = self._fit(tmp_path, ema=True, save_interval=3)
        checkpoint = torch.load(tmp_path / "checkpoint_epoch3.pt", weights_only=False)
        shadow = trainer.ema.state_dict()
        assert checkpoint["ema"] is True
        assert all(torch.equal(checkpoint["model_state_dict"][k], shadow[k]) for k in shadow)

    def test_default_best_model_holds_the_live_weights_and_is_marked_live(self, tmp_path):
        trainer = self._fit(tmp_path)

        checkpoint = torch.load(tmp_path / "best_model.pt", weights_only=False)
        live = trainer.model.state_dict()

        assert checkpoint["ema"] is False
        assert "ema_updates" not in checkpoint
        assert all(torch.equal(checkpoint["model_state_dict"][k], live[k]) for k in live)
        assert checkpoint["config"]["ema"] is False

    def test_the_checkpoint_loads_into_a_bare_model_like_any_other(self, tmp_path):
        self._fit(tmp_path, ema=True)
        checkpoint = torch.load(tmp_path / "best_model.pt", weights_only=False)
        _LinearModel().load_state_dict(checkpoint["model_state_dict"], strict=True)


# --------------------------------------------------------------------------- #
# Trainer integration - real (tiny) MasterModel: evaluation routing
# --------------------------------------------------------------------------- #

_IMAGE_SIZE = 64


def _master_batches(n: int = 1) -> list[dict]:
    gen = torch.Generator().manual_seed(0)
    return [
        {
            "rgb": torch.randn(2, 3, _IMAGE_SIZE, _IMAGE_SIZE, generator=gen),
            "nir": torch.randn(2, 1, _IMAGE_SIZE, _IMAGE_SIZE, generator=gen),
            "bboxes": [torch.tensor([[0.5, 0.5, 0.4, 0.4], [0.3, 0.3, 0.1, 0.1]])] * 2,
            "labels": [torch.tensor([0, 1])] * 2,
        }
        for _ in range(n)
    ]


class TestEvaluationUsesTheEmaWeights:
    def _trainer(self, tmp_path, **config_kwargs) -> Trainer:
        config = TrainingConfig(
            model_type="master",
            image_size=_IMAGE_SIZE,
            batch_size=2,
            effective_batch=2,
            num_workers=0,
            precision="fp32",
            device="cpu",
            nir_homography_path=None,
            output_dir=str(tmp_path),
            **config_kwargs,
        )
        model = MasterModel(num_classes=2, pretrained_backbone=False)
        trainer = Trainer(
            model=model,
            config=config,
            train_loader=_master_batches(),
            val_loader=_master_batches(),
        )
        trainer.writer = None
        return trainer

    @staticmethod
    def _count_forwards(module: nn.Module) -> list[int]:
        calls = [0]
        module.register_forward_hook(lambda *_: calls.__setitem__(0, calls[0] + 1))
        return calls

    def test_evaluate_runs_the_shadow_not_the_live_model(self, tmp_path):
        trainer = self._trainer(tmp_path, ema=True)
        live_calls = self._count_forwards(trainer.model)
        shadow_calls = self._count_forwards(trainer.ema.module)

        metrics = trainer.evaluate()

        assert shadow_calls[0] == 1
        assert live_calls[0] == 0
        assert "map50" in metrics

    def test_evaluate_runs_the_live_model_by_default(self, tmp_path):
        trainer = self._trainer(tmp_path)
        live_calls = self._count_forwards(trainer.model)

        trainer.evaluate()

        assert live_calls[0] == 1

    def test_validation_does_not_flip_the_live_model_out_of_train_mode_for_ema(self, tmp_path):
        trainer = self._trainer(tmp_path, ema=True)
        trainer.model.train()

        trainer.evaluate()

        assert trainer.model.training  # the live model was never put in eval
        assert not trainer.ema.module.training


# --------------------------------------------------------------------------- #
# KDTrainer - EMA is supported, the default is untouched
# --------------------------------------------------------------------------- #

_KD_IMAGE_SIZE = 64


def _kd_batch() -> dict:
    return {
        "rgb": torch.randn(1, 3, _KD_IMAGE_SIZE, _KD_IMAGE_SIZE),
        "nir": torch.randn(1, 1, _KD_IMAGE_SIZE, _KD_IMAGE_SIZE),
        "bboxes": [torch.tensor([[0.5, 0.5, 0.3, 0.3]])],
        "labels": [torch.tensor([0])],
    }


class _KDLoader:
    def __init__(self, n: int):
        self.n = n

    def __iter__(self):
        return (_kd_batch() for _ in range(self.n))


@pytest.fixture(scope="module")
def teacher_checkpoint(tmp_path_factory):
    teacher = MasterModel(num_classes=2, pretrained_backbone=False, backbone_variant="tiny")
    path = tmp_path_factory.mktemp("kd_ema") / "teacher.pt"
    torch.save({"model_state_dict": teacher.state_dict()}, path)
    return path


def _kd_trainer(teacher_checkpoint, tmp_path, **config_kwargs) -> KDTrainer:
    config = KDConfig(
        teacher_checkpoint=str(teacher_checkpoint),
        num_classes=2,
        backbone_variant="tiny",
        image_size=_KD_IMAGE_SIZE,
        batch_size=1,
        precision="fp32",
        num_workers=0,
        device="cpu",
        nir_homography_path=None,
        output_dir=str(tmp_path),
        **config_kwargs,
    )
    trainer = KDTrainer(
        model=StudentModel(num_classes=2),
        config=config,
        train_loader=_KDLoader(2),
        val_loader=_KDLoader(0),
    )
    trainer.writer = None
    return trainer


class TestKDTrainer:
    def test_default_has_no_ema(self, teacher_checkpoint, tmp_path):
        trainer = _kd_trainer(teacher_checkpoint, tmp_path)
        assert trainer.ema is None
        assert trainer.eval_model is trainer.model

    def test_ema_updates_once_per_optimizer_step(self, teacher_checkpoint, tmp_path):
        trainer = _kd_trainer(teacher_checkpoint, tmp_path, ema=True)
        optimizer = torch.optim.AdamW(trainer.model.parameters(), lr=1e-3)

        result = trainer._train_epoch(optimizer, epoch=1, phase=1)

        assert result["steps_taken"] == 2.0
        assert trainer.ema.updates == 2
        # The student carries BatchNorm: its buffers must follow the live model.
        live_bn = next(m for m in trainer.model.modules() if isinstance(m, nn.BatchNorm2d))
        ema_bn = next(m for m in trainer.ema.module.modules() if isinstance(m, nn.BatchNorm2d))
        assert torch.equal(live_bn.running_mean, ema_bn.running_mean)

    def test_checkpoint_keeps_the_adapters_outside_the_ema_weights(
        self, teacher_checkpoint, tmp_path
    ):
        trainer = _kd_trainer(teacher_checkpoint, tmp_path, ema=True)
        optimizer = torch.optim.AdamW(trainer.model.parameters(), lr=1e-3)
        trainer._train_epoch(optimizer, epoch=1, phase=1)

        trainer._save_checkpoint(1, 1, {"map50": 0.0}, "best_model.pt")

        checkpoint = torch.load(tmp_path / "best_model.pt", weights_only=False)
        assert checkpoint["ema"] is True
        assert "kd_adapters_state_dict" in checkpoint
        assert not any(k.startswith("kd_adapters") for k in checkpoint["model_state_dict"])
        StudentModel(num_classes=2).load_state_dict(checkpoint["model_state_dict"], strict=True)
