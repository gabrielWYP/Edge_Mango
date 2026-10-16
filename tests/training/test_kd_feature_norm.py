"""Opt-in PKD feature normalization for the KD feature loss.

`kd_feature_norm="pkd"` z-scores student and teacher features per channel
over (N, H, W) before the MSE and halves it (PKD, Cao et al. 2022). The
default "none" must keep the raw per-level MSE exactly.
"""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.training.kd_config import KDConfig
from src.training.kd_loss import KDLoss, pkd_loss

_LEVEL_WEIGHTS = {"backbone": 1.0, "fpn": 0.1, "head_cls": 0.3, "head_reg": 0.5}


def _feats(seed: int, channels: int = 8) -> dict[str, list[torch.Tensor]]:
    gen = torch.Generator().manual_seed(seed)
    return {
        level: [
            torch.randn(2, channels, 6, 6, generator=gen),
            torch.randn(2, channels, 3, 3, generator=gen),
        ]
        for level in _LEVEL_WEIGHTS
    }


def _channel_affine(x: torch.Tensor, seed: int) -> torch.Tensor:
    """Per-channel a_c * x + b_c with a_c > 0 and large offsets b_c."""
    gen = torch.Generator().manual_seed(seed)
    c = x.shape[1]
    a = torch.rand(c, generator=gen) * 10 + 0.1
    b = torch.randn(c, generator=gen) * 100
    return a.view(1, c, 1, 1) * x + b.view(1, c, 1, 1)


class TestPKDLoss:
    def test_invariant_to_student_channel_affine(self):
        s, t = torch.randn(2, 8, 6, 6), torch.randn(2, 8, 6, 6)
        assert pkd_loss(_channel_affine(s, seed=1), t) == pytest.approx(
            pkd_loss(s, t).item(), rel=1e-4
        )

    def test_invariant_to_teacher_channel_affine(self):
        s, t = torch.randn(2, 8, 6, 6), torch.randn(2, 8, 6, 6)
        assert pkd_loss(s, _channel_affine(t, seed=2)) == pytest.approx(
            pkd_loss(s, t).item(), rel=1e-4
        )

    def test_zero_when_student_equals_teacher(self):
        x = torch.randn(2, 8, 6, 6) * 50 + 300
        assert pkd_loss(x, x.clone()).item() == pytest.approx(0.0, abs=1e-6)

    def test_bounded_for_random_features(self):
        for seed in range(5):
            gen = torch.Generator().manual_seed(seed)
            s = torch.randn(4, 16, 8, 8, generator=gen) * 1000
            t = torch.randn(4, 16, 8, 8, generator=gen) + 340
            loss = pkd_loss(s, t).item()
            assert 0.0 <= loss <= 2.0 + 1e-4
        # Perfect anti-correlation is the worst case: the bound, up to the
        # (M - 1) / M factor of the unbiased std over M = N*H*W samples.
        x = torch.randn(4, 16, 8, 8)
        assert pkd_loss(x, -x).item() == pytest.approx(2.0 * 255 / 256, rel=1e-4)

    def test_teacher_receives_no_gradient(self):
        s = torch.randn(2, 8, 6, 6, requires_grad=True)
        t = torch.randn(2, 8, 6, 6, requires_grad=True)
        pkd_loss(s, t).backward()
        assert s.grad is not None
        assert t.grad is None


class TestKDLossFeatureNorm:
    def test_none_is_exactly_the_raw_mse(self):
        teacher, student = _feats(seed=0), _feats(seed=1)
        total, per_level = KDLoss(_LEVEL_WEIGHTS)(teacher, student)

        expected_total = torch.tensor(0.0)
        for level, weight in _LEVEL_WEIGHTS.items():
            level_mse = torch.tensor(0.0)
            for t, s in zip(teacher[level], student[level]):
                level_mse = level_mse + F.mse_loss(s, t)
            level_mse = level_mse / len(teacher[level])
            assert per_level[level] == level_mse.item()
            expected_total = expected_total + weight * level_mse
        assert torch.equal(total, expected_total)

    def test_default_is_none(self):
        assert KDLoss(_LEVEL_WEIGHTS).feature_norm == "none"

    def test_pkd_returns_every_level_and_uses_pkd_loss(self):
        teacher, student = _feats(seed=0), _feats(seed=1)
        total, per_level = KDLoss(_LEVEL_WEIGHTS, feature_norm="pkd")(teacher, student)

        assert set(per_level) == set(_LEVEL_WEIGHTS)
        expected = sum(
            weight
            * sum(pkd_loss(s, t).item() for t, s in zip(teacher[level], student[level]))
            / len(teacher[level])
            for level, weight in _LEVEL_WEIGHTS.items()
        )
        assert total.item() == pytest.approx(expected, rel=1e-5)

    def test_pkd_ignores_teacher_channel_offsets(self):
        teacher, student = _feats(seed=0), _feats(seed=1)
        shifted = {
            level: [_channel_affine(t, seed=i) for i, t in enumerate(ts)]
            for level, ts in teacher.items()
        }
        criterion = KDLoss(_LEVEL_WEIGHTS, feature_norm="pkd")
        assert criterion(shifted, student)[0].item() == pytest.approx(
            criterion(teacher, student)[0].item(), rel=1e-4
        )

    def test_rejects_unknown_feature_norm(self):
        with pytest.raises(ValueError, match="feature_norm"):
            KDLoss(_LEVEL_WEIGHTS, feature_norm="zscore")


class TestKDConfigFeatureNorm:
    def test_default_is_none(self):
        assert KDConfig().kd_feature_norm == "none"

    def test_accepts_pkd(self):
        assert KDConfig(kd_feature_norm="pkd").kd_feature_norm == "pkd"

    def test_rejects_unknown_value(self):
        with pytest.raises(ValueError, match="kd_feature_norm"):
            KDConfig(kd_feature_norm="PKD")
