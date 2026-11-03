"""`KDLoss` region balancing: a small lesion must not vanish in a global mean.

A 30 px lesion covers 0.22 % of a 640x640 map, so a plain MSE over the whole map
barely sees it. With `region_weights`, each region (damage, fruit, background)
is averaged over its own cells before weighting. The default stays the plain
per-map loss bit for bit, so existing configs train exactly as before.
"""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.master.master_model import MasterModel
from src.models.student.student_model import StudentModel
from src.training.kd_config import KDConfig
from src.training.kd_loss import KD_REGIONS, KDLoss, build_region_masks
from src.training.kd_trainer import KDTrainer

_WEIGHTS = {"damage": 0.5, "fruit": 0.4, "background": 0.1}
_LEVELS = {"fpn": 1.0}

# Fruit covers cells [2, 6) of an 8x8 map, damage only cell (3, 3).
_FRUIT_BOX = [0.5, 0.5, 0.5, 0.5]
_DAMAGE_BOX = [0.4375, 0.4375, 0.125, 0.125]


def _targets():
    return (
        [torch.tensor([_FRUIT_BOX, _DAMAGE_BOX])],
        [torch.tensor([0, 1])],
    )


def _pair(error_by_region: dict[str, float]):
    """Teacher of zeros and a student whose squared error is set per region."""
    bboxes, labels = _targets()
    masks = build_region_masks(bboxes, labels, 8, 8)
    student = torch.zeros(1, 2, 8, 8)
    for region, value in error_by_region.items():
        student[:, :, masks[region][0]] = value ** 0.5
    return {"fpn": [torch.zeros_like(student)]}, {"fpn": [student]}


def test_masks_partition_the_map_with_damage_on_top():
    bboxes, labels = _targets()
    masks = build_region_masks(bboxes, labels, 8, 8)

    assert set(masks) == set(KD_REGIONS)
    assert masks["damage"].sum().item() == 1
    assert masks["damage"][0, 3, 3].item()
    assert masks["fruit"].sum().item() == 15
    assert not masks["fruit"][0, 3, 3].item()
    assert masks["background"].sum().item() == 48
    total = masks["damage"].int() + masks["fruit"].int() + masks["background"].int()
    assert (total == 1).all()


def test_a_lesion_smaller_than_a_cell_still_marks_that_cell():
    bboxes = [torch.tensor([[0.5, 0.5, 0.001, 0.001]])]
    labels = [torch.tensor([1])]

    masks = build_region_masks(bboxes, labels, 4, 4)

    assert masks["damage"].sum().item() >= 1


def test_an_image_without_boxes_is_all_background():
    masks = build_region_masks([torch.zeros((0, 4))], [torch.zeros((0,), dtype=torch.long)], 4, 4)

    assert masks["background"].all()
    assert not masks["damage"].any()
    assert not masks["fruit"].any()


def test_default_stays_the_plain_mse_when_boxes_are_passed():
    teacher = {"fpn": [torch.randn(1, 2, 8, 8)]}
    student = {"fpn": [torch.randn(1, 2, 8, 8)]}
    bboxes, labels = _targets()

    loss, _ = KDLoss(_LEVELS)(teacher, student, bboxes, labels)

    assert loss.item() == F.mse_loss(student["fpn"][0], teacher["fpn"][0]).item()


def test_each_region_is_averaged_over_its_own_cells():
    teacher, student = _pair({"damage": 4.0, "fruit": 1.0, "background": 0.0})
    bboxes, labels = _targets()

    loss, _ = KDLoss(_LEVELS, region_weights=_WEIGHTS)(teacher, student, bboxes, labels)

    assert loss.item() == pytest.approx(0.5 * 4.0 + 0.4 * 1.0 + 0.1 * 0.0)


def test_one_damaged_cell_moves_the_balanced_loss_more_than_the_plain_one():
    teacher, student = _pair({"damage": 4.0, "fruit": 0.0, "background": 0.0})
    bboxes, labels = _targets()

    plain, _ = KDLoss(_LEVELS)(teacher, student)
    balanced, _ = KDLoss(_LEVELS, region_weights=_WEIGHTS)(teacher, student, bboxes, labels)

    assert plain.item() == pytest.approx(4.0 / 64)
    assert balanced.item() == pytest.approx(2.0)


def test_a_batch_without_damage_contributes_only_the_regions_it_has():
    student = torch.ones(1, 2, 4, 4)
    teacher = {"fpn": [torch.zeros_like(student)]}

    loss, _ = KDLoss(_LEVELS, region_weights=_WEIGHTS)(
        teacher, {"fpn": [student]}, [torch.zeros((0, 4))], [torch.zeros((0,), dtype=torch.long)]
    )

    assert loss.item() == pytest.approx(0.1)
    assert torch.isfinite(loss)


def test_balanced_pearson_keeps_the_pkd_scale_invariance():
    teacher = {"fpn": [torch.randn(2, 4, 8, 8)]}
    student = {"fpn": [teacher["fpn"][0] * 3.0 + 1.0]}
    bboxes = [torch.tensor([_FRUIT_BOX, _DAMAGE_BOX])] * 2
    labels = [torch.tensor([0, 1])] * 2

    loss, _ = KDLoss(_LEVELS, loss_type="pearson", region_weights=_WEIGHTS)(
        teacher, student, bboxes, labels
    )

    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_region_weights_require_boxes():
    teacher, student = _pair({"damage": 1.0})

    with pytest.raises(ValueError, match="bboxes"):
        KDLoss(_LEVELS, region_weights=_WEIGHTS)(teacher, student)


@pytest.mark.parametrize(
    "weights",
    [
        {"damage": 1.0},
        {"damage": 1.0, "fruit": 0.0, "typo": 0.0},
        {"damage": -1.0, "fruit": 1.0, "background": 1.0},
        {"damage": 0.0, "fruit": 0.0, "background": 0.0},
    ],
)
def test_invalid_region_weights_fail_loud(weights):
    with pytest.raises(ValueError, match="region"):
        KDLoss(_LEVELS, region_weights=weights)
    with pytest.raises(ValueError, match="region"):
        KDConfig(kd_region_weights=weights)


def test_config_default_keeps_the_plain_loss():
    assert KDConfig().kd_region_weights is None


def test_a_level_with_zero_weight_adds_nothing_and_sends_no_gradient():
    adapter = torch.nn.Conv2d(2, 2, 1)
    feat = torch.randn(1, 2, 4, 4)
    teacher = {"backbone": [torch.randn(1, 2, 4, 4)], "fpn": [torch.randn(1, 2, 4, 4)]}
    student = {"backbone": [adapter(feat)], "fpn": [torch.randn(1, 2, 4, 4, requires_grad=True)]}

    loss, per_level = KDLoss({"backbone": 0.0, "fpn": 1.0})(teacher, student)
    loss.backward()

    assert loss.item() == pytest.approx(per_level["fpn"])
    assert per_level["backbone"] > 0.0
    assert adapter.weight.grad is None or adapter.weight.grad.abs().sum().item() == 0.0


def test_config_round_trips_through_yaml(tmp_path):
    path = tmp_path / "kd.yaml"
    KDConfig(kd_region_weights=_WEIGHTS).to_yaml(path)

    assert KDConfig.from_yaml(path).kd_region_weights == _WEIGHTS


def test_trainer_feeds_the_batch_boxes_to_the_balanced_loss(tmp_path):
    teacher = MasterModel(num_classes=2, pretrained_backbone=False, backbone_variant="tiny")
    checkpoint = tmp_path / "teacher.pt"
    torch.save({"model_state_dict": teacher.state_dict()}, checkpoint)
    batch = {
        "rgb": torch.randn(1, 3, 64, 64),
        "nir": torch.randn(1, 1, 64, 64),
        "bboxes": [torch.tensor([_FRUIT_BOX, _DAMAGE_BOX])],
        "labels": [torch.tensor([0, 1])],
    }
    trainer = KDTrainer(
        model=StudentModel(num_classes=2),
        config=KDConfig(
            teacher_checkpoint=str(checkpoint),
            num_classes=2,
            backbone_variant="tiny",
            image_size=64,
            batch_size=1,
            precision="fp32",
            num_workers=0,
            output_dir=str(tmp_path),
            kd_region_weights=_WEIGHTS,
        ),
        train_loader=[batch],
        val_loader=[],
    )

    result = trainer._train_epoch(
        torch.optim.AdamW(trainer.model.parameters(), lr=1e-3), epoch=1, phase=1
    )

    assert trainer.kd_criterion.region_weights == _WEIGHTS
    assert result["kd_loss"] > 0.0


def _level_feats():
    teacher = {"fpn": [torch.zeros(1, 1, 2, 2) for _ in range(3)]}
    student = {"fpn": [torch.full((1, 1, 2, 2), float(v)) for v in (1.0, 2.0, 3.0)]}
    return teacher, student


def test_sublevels_keep_only_the_selected_maps():
    teacher, student = _level_feats()

    loss, per_level = KDLoss(_LEVELS, sublevels={"fpn": [0, 1]})(teacher, student)

    assert loss.item() == pytest.approx((1.0 + 4.0) / 2)
    assert per_level["fpn"] == pytest.approx(2.5)


def test_an_empty_sublevel_list_turns_the_group_off():
    teacher, student = _level_feats()

    loss, per_level = KDLoss(_LEVELS, sublevels={"fpn": []})(teacher, student)

    assert loss.item() == 0.0
    assert per_level["fpn"] == 0.0


def test_sublevels_default_keeps_every_map():
    teacher, student = _level_feats()

    loss, _ = KDLoss(_LEVELS)(teacher, student)

    assert loss.item() == pytest.approx((1.0 + 4.0 + 9.0) / 3)


@pytest.mark.parametrize("sublevels", [{"typo": [0]}, {"fpn": [-1]}, {"fpn": [0.5]}])
def test_invalid_sublevels_fail_loud(sublevels):
    with pytest.raises(ValueError, match="sublevels"):
        KDLoss(_LEVELS, sublevels=sublevels)


def test_config_rejects_a_sublevel_group_it_does_not_distill():
    with pytest.raises(ValueError, match="sublevels"):
        KDConfig(kd_sublevels={"typo": [0]})


def test_region_arm_config_loads_with_its_levels_and_regions():
    config = KDConfig.from_yaml(
        Path(__file__).parent.parent.parent / "configs/kd_training_twostream_regions.yaml"
    )

    assert config.kd_loss_type == "pearson"
    assert config.kd_sublevels["backbone"] == []
    assert sum(config.kd_region_weights.values()) == pytest.approx(1.0)
