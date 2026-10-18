"""`neck_norm` in the model: BatchNorm2d vs GroupNorm in the cross-attention neck.

The only BatchNorm layers in `MasterModel` are the `DualFPN.fusion_convs`
(`fusion_mode="cross_attention"`): the ConvNeXt backbone and the cross-modal
fusion use LayerNorm, the detection head has no normalization layer, and the
early-fusion `FPNNeck` has none either. So "BatchNorm in the neck and head"
means exactly those fusion convs, and `neck_norm="group"` must leave the
whole model without a single BatchNorm.

What these tests pin down:
- the default keeps `nn.BatchNorm2d` (and therefore every existing
  checkpoint schema);
- `group` removes every BatchNorm, keeps every output shape and the
  trainable-parameter count (BN and GN both carry weight + bias), and drops
  only the BN running-statistics buffers;
- a GroupNorm model's output for one image does not depend on what else is
  in the batch — in eval AND in train mode. BatchNorm fails that in train
  mode, which is the control proving the test can tell the two apart.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torchvision.ops import StochasticDepth

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.models.master.master_model import MasterModel
from src.models.master.neck import DualFPN

IMAGE_SIZE = 128
NUM_CLASSES = 2
BN = nn.modules.batchnorm._BatchNorm


def _build(neck_norm: str = "batch", **kwargs) -> MasterModel:
    return MasterModel(
        num_classes=NUM_CLASSES,
        pretrained_backbone=False,
        fusion_mode="cross_attention",
        neck_norm=neck_norm,
        **kwargs,
    )


def _count(module: nn.Module, kind: type) -> int:
    return sum(isinstance(m, kind) for m in module.modules())


def _backbone_features(batch: int = 1) -> list[torch.Tensor]:
    return [
        torch.randn(batch, ch, IMAGE_SIZE // (4 * 2**i), IMAGE_SIZE // (4 * 2**i))
        for i, ch in enumerate(MasterModel.STAGE_CHANNELS)
    ]


class TestDualFPNNorm:
    def test_default_is_batch_norm(self):
        neck = DualFPN()
        assert _count(neck, nn.BatchNorm2d) == 3
        assert _count(neck, nn.GroupNorm) == 0

    @pytest.mark.parametrize("emit_strides,levels", [((8, 16, 32), 3), ((4, 8, 16, 32), 4)])
    def test_group_replaces_one_norm_per_emitted_level(self, emit_strides, levels):
        neck = DualFPN(emit_strides=emit_strides, norm="group", gn_groups=32)
        assert _count(neck, BN) == 0
        groupnorms = [m for m in neck.modules() if isinstance(m, nn.GroupNorm)]
        assert len(groupnorms) == levels
        assert all(m.num_groups == 32 and m.num_channels == 256 for m in groupnorms)

    def test_output_shapes_are_unchanged(self):
        torch.manual_seed(0)
        fused, nir = _backbone_features(2), _backbone_features(2)
        shapes = {}
        for norm in ("batch", "group"):
            neck = DualFPN(norm=norm).eval()
            with torch.no_grad():
                shapes[norm] = [tuple(t.shape) for t in neck(fused, nir)]
        assert shapes["batch"] == shapes["group"]

    def test_group_has_the_same_parameters_and_loses_only_the_bn_buffers(self):
        batch_neck, group_neck = DualFPN(norm="batch"), DualFPN(norm="group")

        count = lambda m: sum(p.numel() for p in m.parameters())
        assert count(batch_neck) == count(group_neck)

        batch_keys, group_keys = set(batch_neck.state_dict()), set(group_neck.state_dict())
        assert group_keys < batch_keys
        assert {k.rsplit(".", 1)[-1] for k in batch_keys - group_keys} == {
            "running_mean",
            "running_var",
            "num_batches_tracked",
        }

    def test_unknown_norm_raises(self):
        with pytest.raises(ValueError, match="neck_norm"):
            DualFPN(norm="instance")


class TestMasterModelNorm:
    def test_default_keeps_batch_norm_in_the_fusion_convs_only(self):
        model = _build("batch")
        assert model.neck_norm == "batch"
        assert _count(model.neck, nn.BatchNorm2d) == 3
        # Nothing else in the model is a BatchNorm: backbone and fusion use
        # LayerNorm, the head has no normalization layer at all.
        assert _count(model, BN) == _count(model.neck, BN)
        assert _count(model.head, BN) == 0
        assert _count(model.head, nn.GroupNorm) == 0

    def test_group_leaves_no_batch_norm_anywhere(self):
        model = _build("group")
        assert model.neck_norm == "group"
        assert _count(model, BN) == 0
        assert _count(model.neck, nn.GroupNorm) == len(model.head_strides)
        assert _count(model.head, BN) == 0
        assert not [k for k in model.state_dict() if k.endswith("running_mean")]

    def test_group_count_follows_the_argument(self):
        model = _build("group", neck_gn_groups=16)
        groupnorms = [m for m in model.neck.modules() if isinstance(m, nn.GroupNorm)]
        assert {m.num_groups for m in groupnorms} == {16}

    def test_four_level_pyramid_gets_four_group_norms(self):
        model = _build("group", head_strides=[4, 8, 16, 32])
        assert _count(model.neck, nn.GroupNorm) == 4
        assert _count(model, BN) == 0

    def test_forward_shapes_and_parameter_count_match_batch_norm(self):
        torch.manual_seed(0)
        batch_model, group_model = _build("batch").eval(), _build("group").eval()
        rgb = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
        nir = torch.randn(1, 1, IMAGE_SIZE, IMAGE_SIZE)
        with torch.no_grad():
            batch_out, group_out = batch_model(rgb, nir), group_model(rgb, nir)

        assert set(batch_out) == set(group_out)
        for key in batch_out:
            assert [tuple(t.shape) for t in batch_out[key]] == [
                tuple(t.shape) for t in group_out[key]
            ], key
        assert (
            batch_model.count_parameters()["total"]
            == group_model.count_parameters()["total"]
        )

    def test_early_fusion_rejects_group_instead_of_ignoring_it(self):
        with pytest.raises(ValueError, match="early"):
            MasterModel(
                num_classes=NUM_CLASSES,
                pretrained_backbone=False,
                fusion_mode="early",
                neck_norm="group",
            )

    def test_early_fusion_default_is_still_buildable(self):
        model = MasterModel(num_classes=NUM_CLASSES, pretrained_backbone=False)
        assert model.neck_norm == "batch"
        assert _count(model, BN) == 0

    def test_unknown_norm_raises(self):
        with pytest.raises(ValueError, match="neck_norm"):
            _build("gn")

    def test_bad_group_count_raises(self):
        with pytest.raises(ValueError, match="neck_gn_groups"):
            _build("group", neck_gn_groups=0)


def _make_deterministic(model: MasterModel) -> MasterModel:
    """Remove every source of train-mode randomness except the norm layers.

    `fusion_dropout`/`fpn_dropout` are constructor arguments; the ConvNeXt
    blocks additionally carry stochastic depth (torchvision defaults to 0.1),
    which draws per-sample masks and would make any train-mode comparison
    fail for reasons unrelated to normalization.
    """
    for module in model.modules():
        if isinstance(module, StochasticDepth):
            module.p = 0.0
    return model


def _alone_vs_in_batch(model: MasterModel, batch: int = 3) -> tuple[list, list]:
    """Outputs for image 0 run alone, and for image 0 run inside a batch."""
    torch.manual_seed(0)
    rgb = torch.randn(batch, 3, IMAGE_SIZE, IMAGE_SIZE)
    nir = torch.randn(batch, 1, IMAGE_SIZE, IMAGE_SIZE)
    with torch.no_grad():
        alone = model(rgb[:1], nir[:1])["preds"]
        in_batch = [level[:1] for level in model(rgb, nir)["preds"]]
    return alone, in_batch


def _max_abs_diff(a: list[torch.Tensor], b: list[torch.Tensor]) -> float:
    return max((x - y).abs().max().item() for x, y in zip(a, b))


class TestBatchCompositionIndependence:
    @pytest.mark.parametrize("mode", ["eval", "train"])
    def test_group_norm_output_ignores_the_rest_of_the_batch(self, mode):
        model = _make_deterministic(_build("group", fusion_dropout=0.0, fpn_dropout=0.0))
        getattr(model, mode)()

        alone, in_batch = _alone_vs_in_batch(model)

        assert _max_abs_diff(alone, in_batch) < 1e-3

    def test_batch_norm_in_train_mode_does_depend_on_the_batch(self):
        """Control: the same comparison separates the two norms in train mode,
        so the group-norm assertions above are not vacuous."""
        model = _make_deterministic(_build("batch", fusion_dropout=0.0, fpn_dropout=0.0))
        model.train()

        alone, in_batch = _alone_vs_in_batch(model)

        assert _max_abs_diff(alone, in_batch) > 1e-2

    def test_batch_norm_in_eval_mode_is_batch_independent_too(self):
        """Eval-mode BN uses running statistics; only train mode couples the
        samples. (So eval-time equality alone could not tell the norms apart.)"""
        model = _make_deterministic(_build("batch", fusion_dropout=0.0, fpn_dropout=0.0)).eval()

        alone, in_batch = _alone_vs_in_batch(model)

        assert _max_abs_diff(alone, in_batch) < 1e-3
