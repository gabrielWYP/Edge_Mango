"""Loading a checkpoint must rebuild the neck norm it was trained with.

GroupNorm and BatchNorm checkpoints have different state_dict schemas
(BatchNorm carries `running_mean`/`running_var`/`num_batches_tracked`), so a
model built with the wrong norm cannot load the weights. The checkpoint's own
recorded `config.neck_norm` is therefore the source of truth - exactly like
`head_strides` and `fusion_pos_encoding` - and a disagreement that cannot be
resolved from the recorded config fails with one actionable sentence instead
of a missing/unexpected-key dump.

Covered loaders: `scripts/evaluate_checkpoint.py::_load_model`,
`scripts/visualize_damage_predictions.py::load_model` and the KD teacher
(`KDTrainer._load_teacher`).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts import evaluate_checkpoint as ec
from scripts.visualize_damage_predictions import load_model as visualize_load_model
from src.models.master.master_model import MasterModel
from src.training.config import TrainingConfig
from src.training.kd_config import KDConfig
from src.training.kd_trainer import KDTrainer
from src.training.loop import Trainer

BN = nn.modules.batchnorm._BatchNorm
CPU = torch.device("cpu")


def _count(module: nn.Module, kind: type) -> int:
    return sum(isinstance(m, kind) for m in module.modules())


def _model(neck_norm: str) -> MasterModel:
    torch.manual_seed(0)
    return MasterModel(
        num_classes=2,
        pretrained_backbone=False,
        fusion_mode="cross_attention",
        neck_norm=neck_norm,
    )


def _eval_config(**overrides) -> TrainingConfig:
    """The eval config: cross-attention, with the DEFAULT (batch) norm unless
    told otherwise - i.e. what an old config file would say."""
    return TrainingConfig(
        model_type="master",
        fusion_mode="cross_attention",
        head_strides=[8, 16, 32],
        assigner_level_ranges=[64.0, 128.0],
        nir_homography_path=None,
        **overrides,
    )


def _payload(model: MasterModel, recorded: dict | None) -> dict:
    """What `Trainer._save_checkpoint` writes, minus the bulky extras."""
    payload = {
        "epoch": 3,
        "phase": 1,
        "arch_version": 1,
        "model_state_dict": model.state_dict(),
        "metrics": {},
        "best_map50": 0.0,
    }
    if recorded is not None:
        payload["config"] = {"head_strides": [8, 16, 32], **recorded}
    return payload


@pytest.fixture(scope="module")
def group_model() -> MasterModel:
    return _model("group")


@pytest.fixture(scope="module")
def batch_model() -> MasterModel:
    return _model("batch")


@pytest.fixture
def save(tmp_path):
    def _save(name: str, obj) -> str:
        path = tmp_path / name
        torch.save(obj, path)
        return str(path)

    return _save


class TestResolveNeckNorm:
    def test_the_recorded_norm_wins_over_the_eval_config(self, caplog):
        recorded = {"config": {"neck_norm": "group", "neck_gn_groups": 16}}
        with caplog.at_level(logging.INFO, logger=ec.logger.name):
            resolved = ec._resolve_neck_norm(recorded, _eval_config(neck_norm="batch"))
        assert resolved == ("group", 16)
        assert any("neck_norm" in r.getMessage() for r in caplog.records)

    def test_a_recorded_config_without_the_field_is_legacy_batch_norm(self):
        """Even if the eval config says `group`: that checkpoint was trained
        with BatchNorm, so its weights only fit a BatchNorm model."""
        resolved = ec._resolve_neck_norm(
            {"config": {"head_strides": [8, 16, 32]}}, _eval_config(neck_norm="group")
        )
        assert resolved == ("batch", 32)

    def test_no_recorded_config_falls_back_to_the_eval_config(self):
        """Nothing to read: trust the caller, and let the state_dict check
        catch a wrong guess."""
        assert ec._resolve_neck_norm(
            {"model_state_dict": {}}, _eval_config(neck_norm="group", neck_gn_groups=8)
        ) == ("group", 8)
        assert ec._resolve_neck_norm(None, _eval_config()) == ("batch", 32)


class TestEvaluateCheckpointLoadModel:
    def test_group_checkpoint_rebuilds_a_group_model_under_a_batch_config(
        self, save, group_model
    ):
        path = save("gn.pt", _payload(group_model, {"neck_norm": "group", "neck_gn_groups": 32}))

        model = ec._load_model("master", path, _eval_config(), CPU)

        assert model.neck_norm == "group"
        assert _count(model, BN) == 0
        assert not model.training
        for key, value in group_model.state_dict().items():
            assert torch.equal(model.state_dict()[key], value), key

    def test_batch_checkpoint_rebuilds_a_batch_model_under_a_group_config(
        self, save, batch_model
    ):
        """The inverse: a pre-existing BatchNorm checkpoint evaluated with the
        new `_stable` config must not be rebuilt as GroupNorm."""
        path = save("bn.pt", _payload(batch_model, {}))  # legacy: no neck_norm key

        model = ec._load_model("master", path, _eval_config(neck_norm="group"), CPU)

        assert model.neck_norm == "batch"
        assert _count(model, nn.BatchNorm2d) == 3

    def test_legacy_checkpoint_with_default_config_loads_unchanged(self, save, batch_model):
        path = save("bn.pt", _payload(batch_model, {}))
        model = ec._load_model("master", path, _eval_config(), CPU)
        assert _count(model, nn.BatchNorm2d) == 3

    def test_raw_group_state_dict_under_a_group_config_loads(self, save, group_model):
        path = save("raw_gn.pt", group_model.state_dict())
        model = ec._load_model("master", path, _eval_config(neck_norm="group"), CPU)
        assert _count(model, BN) == 0

    def test_raw_group_state_dict_under_a_batch_config_fails_loudly(self, save, group_model):
        path = save("raw_gn.pt", group_model.state_dict())
        with pytest.raises(ValueError, match="neck_norm") as excinfo:
            ec._load_model("master", path, _eval_config(), CPU)
        message = str(excinfo.value)
        assert path in message
        assert "group" in message and "batch" in message

    def test_raw_batch_state_dict_under_a_group_config_fails_loudly(self, save, batch_model):
        path = save("raw_bn.pt", batch_model.state_dict())
        with pytest.raises(ValueError, match="neck_norm"):
            ec._load_model("master", path, _eval_config(neck_norm="group"), CPU)

    def test_a_recorded_norm_that_contradicts_the_weights_fails_loudly(
        self, save, batch_model
    ):
        """BatchNorm weights under a config claiming `group`: the rebuilt
        GroupNorm model cannot take them, and the message says why."""
        path = save("liar.pt", _payload(batch_model, {"neck_norm": "group"}))
        with pytest.raises(ValueError, match="neck_norm"):
            ec._load_model("master", path, _eval_config(), CPU)

    def test_visualize_loader_rebuilds_the_checkpoints_norm_too(self, save, group_model):
        path = save("gn.pt", _payload(group_model, {"neck_norm": "group", "neck_gn_groups": 32}))

        model, strides = visualize_load_model("master", path, _eval_config(), CPU)

        assert _count(model, BN) == 0
        assert strides == [8, 16, 32]

    def test_visualize_loader_fails_loudly_on_a_mismatch(self, save, group_model):
        path = save("raw_gn.pt", group_model.state_dict())
        with pytest.raises(ValueError, match="neck_norm"):
            visualize_load_model("master", path, _eval_config(), CPU)


class TestKDTeacher:
    def _kd_config(self, path: str) -> KDConfig:
        return KDConfig(
            teacher_checkpoint=path,
            num_classes=2,
            backbone_variant="tiny",
            fusion_mode="cross_attention",
            head_strides=[8, 16, 32],
            device="cpu",
        )

    def test_teacher_is_rebuilt_with_the_norm_it_was_trained_with(self, save, group_model):
        path = save("t_gn.pt", _payload(group_model, {"neck_norm": "group", "neck_gn_groups": 32}))
        teacher = KDTrainer._load_teacher(self._kd_config(path))
        assert _count(teacher, BN) == 0

    def test_legacy_teacher_stays_batch_norm(self, save, batch_model):
        path = save("t_bn.pt", _payload(batch_model, {}))
        teacher = KDTrainer._load_teacher(self._kd_config(path))
        assert _count(teacher, nn.BatchNorm2d) == 3


# --------------------------------------------------------------------------- #
# A checkpoint written by the REAL Trainer with the stable recipe
# --------------------------------------------------------------------------- #

_IMAGE_SIZE = 128


def _batches() -> list[dict]:
    gen = torch.Generator().manual_seed(0)
    return [
        {
            "rgb": torch.randn(2, 3, _IMAGE_SIZE, _IMAGE_SIZE, generator=gen),
            "nir": torch.randn(2, 1, _IMAGE_SIZE, _IMAGE_SIZE, generator=gen),
            "bboxes": [torch.tensor([[0.5, 0.5, 0.4, 0.4], [0.3, 0.3, 0.2, 0.2]])] * 2,
            "labels": [torch.tensor([0, 1])] * 2,
        }
    ]


class TestRealTrainerCheckpointRoundTrip:
    def test_group_norm_ema_checkpoint_is_evaluable_from_a_plain_config(self, tmp_path):
        config = TrainingConfig(
            model_type="master",
            fusion_mode="cross_attention",
            head_strides=[8, 16, 32],
            assigner_level_ranges=[64.0, 128.0],
            neck_norm="group",
            ema=True,
            ema_decay=0.9,
            ema_tau=2.0,
            epochs=1,
            warmup_epochs=1,
            image_size=_IMAGE_SIZE,
            batch_size=2,
            effective_batch=2,
            num_workers=0,
            precision="fp32",
            device="cpu",
            nir_homography_path=None,
            output_dir=str(tmp_path),
        )
        trainer = Trainer(
            model=MasterModel(
                num_classes=2,
                pretrained_backbone=False,
                fusion_mode="cross_attention",
                neck_norm=config.neck_norm,
                neck_gn_groups=config.neck_gn_groups,
            ),
            config=config,
            train_loader=_batches(),
            val_loader=_batches(),
        )
        trainer.writer = None

        trainer.fit()

        checkpoint = torch.load(tmp_path / "best_model.pt", weights_only=False)
        assert checkpoint["ema"] is True
        assert checkpoint["config"]["neck_norm"] == "group"
        assert checkpoint["config"]["neck_gn_groups"] == 32
        assert checkpoint["arch_version"] == 1
        assert not [k for k in checkpoint["model_state_dict"] if k.endswith("running_mean")]

        # The eval config knows nothing about the norm: the checkpoint decides.
        model = ec._load_model("master", str(tmp_path / "best_model.pt"), _eval_config(), CPU)
        assert _count(model, BN) == 0
        shadow = trainer.ema.state_dict()
        assert all(torch.equal(model.state_dict()[k], shadow[k]) for k in shadow)
