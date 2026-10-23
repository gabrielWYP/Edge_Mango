"""`KDLoss` loss_type: plain MSE (default) vs PKD Pearson (arXiv 2207.02039).

PKD standardizes each feature map per channel before the MSE, so it must be
blind to a per-channel positive scale and shift of the student — the property
that separates it from raw-feature MSE in the mse-vs-pearson ablation. The
default must stay the pre-PKD MSE bit for bit, so existing configs and
checkpoints train exactly as before.
"""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.training.kd_config import KDConfig
from src.training.kd_loss import KD_LOSS_TYPES, KDLoss

_LEVEL_WEIGHTS = {"backbone": 1.0, "fpn": 0.1, "head_cls": 0.3, "head_reg": 0.5}


def _feats(seed: int) -> dict[str, list[torch.Tensor]]:
    """Two scales per level, (N, C, H, W) with N > 1 so batch pooling matters."""
    g = torch.Generator().manual_seed(seed)
    return {
        level: [
            torch.randn(2, 8, 6, 6, generator=g),
            torch.randn(2, 8, 3, 3, generator=g),
        ]
        for level in _LEVEL_WEIGHTS
    }


def _per_channel_affine(
    feats: dict[str, list[torch.Tensor]], seed: int
) -> dict[str, list[torch.Tensor]]:
    """`a_c * x + b_c` with a_c > 0, one (a, b) per channel shared over N, H, W."""
    g = torch.Generator().manual_seed(seed)
    out = {}
    for level, tensors in feats.items():
        out[level] = []
        for t in tensors:
            c = t.shape[1]
            scale = torch.rand(1, c, 1, 1, generator=g) * 4.0 + 0.5
            shift = torch.randn(1, c, 1, 1, generator=g) * 3.0
            out[level].append(t * scale + shift)
    return out


def _legacy_mse(teacher, student) -> torch.Tensor:
    """The pre-PKD `KDLoss.forward`, inlined as the reference."""
    total = torch.tensor(0.0)
    for level, weight in _LEVEL_WEIGHTS.items():
        level_mse = torch.tensor(0.0)
        for t, s in zip(teacher[level], student[level]):
            level_mse = level_mse + F.mse_loss(s, t.detach())
        level_mse = level_mse / len(teacher[level])
        total = total + weight * level_mse
    return total


class TestPearsonInvariance:
    def test_pearson_is_zero_for_per_channel_affine_student(self):
        teacher = _feats(0)
        student = _per_channel_affine(teacher, seed=1)
        total, per_level = KDLoss(_LEVEL_WEIGHTS, loss_type="pearson")(teacher, student)
        assert total.item() == pytest.approx(0.0, abs=1e-6)
        assert all(v == pytest.approx(0.0, abs=1e-6) for v in per_level.values())

    def test_mse_is_not_invariant_to_the_same_transform(self):
        teacher = _feats(0)
        student = _per_channel_affine(teacher, seed=1)
        total, _ = KDLoss(_LEVEL_WEIGHTS, loss_type="mse")(teacher, student)
        assert total.item() > 1.0

    def test_pearson_is_positive_for_unrelated_features(self):
        total, _ = KDLoss(_LEVEL_WEIGHTS, loss_type="pearson")(_feats(0), _feats(2))
        # Independent maps: r ≈ 0, so 0.5 * MSE(z_s, z_t) = 1 - r ≈ 1 per pair,
        # and the total ≈ sum of level weights.
        assert total.item() == pytest.approx(sum(_LEVEL_WEIGHTS.values()), rel=0.3)

    def test_pearson_rejects_shape_mismatch(self):
        teacher = _feats(0)
        student = _feats(1)
        student["fpn"][0] = torch.randn(2, 8, 5, 5)
        with pytest.raises(AssertionError, match="matching shapes"):
            KDLoss(_LEVEL_WEIGHTS, loss_type="pearson")(teacher, student)


class TestGradientFlow:
    @pytest.mark.parametrize("loss_type", KD_LOSS_TYPES)
    def test_gradients_reach_student_only(self, loss_type):
        teacher = {k: [t.requires_grad_() for t in v] for k, v in _feats(0).items()}
        student = {k: [s.requires_grad_() for s in v] for k, v in _feats(1).items()}
        total, _ = KDLoss(_LEVEL_WEIGHTS, loss_type=loss_type)(teacher, student)
        total.backward()
        for level in _LEVEL_WEIGHTS:
            for t, s in zip(teacher[level], student[level]):
                assert t.grad is None
                assert s.grad is not None and s.grad.abs().sum() > 0


class TestLossTypeSelection:
    def test_default_is_mse(self):
        assert KDLoss(_LEVEL_WEIGHTS).loss_type == "mse"

    def test_default_matches_legacy_mse_exactly(self):
        teacher, student = _feats(0), _feats(1)
        total, per_level = KDLoss(_LEVEL_WEIGHTS)(teacher, student)
        assert torch.equal(total, _legacy_mse(teacher, student))
        assert set(per_level) == set(_LEVEL_WEIGHTS)

    def test_unknown_loss_type_raises(self):
        with pytest.raises(ValueError, match="loss_type"):
            KDLoss(_LEVEL_WEIGHTS, loss_type="cosine")


class TestKDConfigLossType:
    def test_default_is_mse(self):
        assert KDConfig().kd_loss_type == "mse"

    def test_unknown_value_raises(self):
        with pytest.raises(ValueError, match="kd_loss_type"):
            KDConfig(kd_loss_type="pkd")

    @pytest.mark.parametrize("loss_type", KD_LOSS_TYPES)
    def test_yaml_round_trip(self, tmp_path, loss_type):
        KDConfig(kd_loss_type=loss_type).to_yaml(tmp_path / "kd.yaml")
        assert KDConfig.from_yaml(tmp_path / "kd.yaml").kd_loss_type == loss_type

    @pytest.mark.parametrize(
        "yaml_name", ["kd_training.yaml", "kd_training_twostream.yaml"]
    )
    def test_shipped_kd_configs_stay_mse(self, yaml_name):
        root = Path(__file__).parent.parent.parent
        assert KDConfig.from_yaml(root / "configs" / yaml_name).kd_loss_type == "mse"
