"""The batched loss and assigner must match the per-image loop implementation they replaced.

`src/training/loss_reference.py` is the frozen original. Both are fed the same random batches
(0, 1 and several GT per image, detect and segment, with and without empty images).
"""

from __future__ import annotations

import pytest
import torch

from src.training.loss import TaskAlignedAssigner, YOLOv8Loss, _generate_anchors, pad_targets
from src.training.loss_reference import ReferenceTaskAlignedAssigner, ReferenceYOLOv8Loss

SIZE = 256
PROTO = SIZE // 4
STRIDES = [8, 16, 32]
NM = 6
NC = 2


def _predictions(batch: int, seed: int) -> list[torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    return [torch.randn(batch, NC + 4, SIZE // s, SIZE // s, generator=gen) for s in STRIDES]


def _targets(counts: list[int], seed: int, with_masks: bool = False) -> dict:
    gen = torch.Generator().manual_seed(1000 + seed)
    bboxes, labels, masks, has_mask = [], [], [], []
    for n in counts:
        centers = torch.rand(n, 2, generator=gen) * 0.6 + 0.2
        sizes = torch.rand(n, 2, generator=gen) * 0.35 + 0.05
        bboxes.append(torch.cat([centers, sizes], dim=1))
        labels.append(torch.randint(0, NC, (n,), generator=gen))
        masks.append((torch.rand(n, PROTO, PROTO, generator=gen) > 0.5).to(torch.uint8))
        has_mask.append(torch.rand(n, generator=gen) > 0.25)
    targets = {"bboxes": bboxes, "labels": labels}
    if with_masks:
        targets["masks"], targets["has_mask"] = masks, has_mask
    return targets


def _mask_outputs(batch: int, seed: int) -> dict:
    gen = torch.Generator().manual_seed(2000 + seed)
    return {
        "mask_coefs": [torch.randn(batch, NM, SIZE // s, SIZE // s, generator=gen) for s in STRIDES],
        "protos": torch.randn(batch, NM, PROTO, PROTO, generator=gen),
    }


def _leaves(preds: list[torch.Tensor]) -> list[torch.Tensor]:
    return [p.clone().requires_grad_(True) for p in preds]


def _padded(targets: dict) -> dict:
    boxes, labels, counts, masks, has_mask = pad_targets(targets)
    out = {"bboxes_padded": boxes, "labels_padded": labels, "counts": counts}
    if masks is not None:
        out["masks_padded"], out["has_mask_padded"] = masks, has_mask
    return out


COUNTS = [[0, 1, 3, 2], [1, 1, 1, 1], [0, 0, 0, 0], [4, 0, 2, 1], [2, 3, 1, 2]]


@pytest.mark.parametrize("seed", range(len(COUNTS)))
@pytest.mark.parametrize("kwargs", [
    {},
    {"level_ranges": [48, 96]},
    {"center_radius": 1.5},
    {"center_radius": 2.5, "level_ranges": [64, 128]},
])
@pytest.mark.parametrize("padded", [False, True])
def test_assigner_outputs_are_identical_to_the_loop(seed, kwargs, padded) -> None:
    counts = COUNTS[seed]
    B = len(counts)
    preds = _predictions(B, seed)
    feat = [(SIZE // s, SIZE // s) for s in STRIDES]
    anchors, anchor_strides, per_level = _generate_anchors(feat, STRIDES, torch.device("cpu"))
    cls = torch.cat([p[:, :NC].flatten(2).permute(0, 2, 1) for p in preds], dim=1).sigmoid()
    reg = torch.cat([p[:, NC:].flatten(2).permute(0, 2, 1) for p in preds], dim=1)
    boxes = YOLOv8Loss._decode_bboxes(reg, anchors)
    targets = _targets(counts, seed)
    gt_px = [b * SIZE for b in targets["bboxes"]]
    common = dict(anchor_strides=anchor_strides, num_per_level=per_level)

    ref = ReferenceTaskAlignedAssigner(**kwargs)
    old = ref(cls, boxes, targets["labels"], gt_px, anchors, STRIDES, **common)
    new_assigner = TaskAlignedAssigner(**kwargs)
    if padded:
        pb, pl, pc, _, _ = pad_targets({"bboxes": gt_px, "labels": targets["labels"]})
        new = new_assigner(cls, boxes, pl, pb, anchors, STRIDES, gt_counts=pc, **common)
    else:
        new = new_assigner(cls, boxes, targets["labels"], gt_px, anchors, STRIDES, **common)

    for a, b in zip(new, old):
        assert torch.equal(a, b)
    assert torch.equal(new_assigner.last_gt_idx, ref.last_gt_idx)


def test_assigner_stats_match_the_loop() -> None:
    counts = [2, 3, 1, 0]
    preds = _predictions(4, 11)
    feat = [(SIZE // s, SIZE // s) for s in STRIDES]
    anchors, anchor_strides, per_level = _generate_anchors(feat, STRIDES, torch.device("cpu"))
    cls = torch.cat([p[:, :NC].flatten(2).permute(0, 2, 1) for p in preds], dim=1).sigmoid()
    reg = torch.cat([p[:, NC:].flatten(2).permute(0, 2, 1) for p in preds], dim=1)
    boxes = YOLOv8Loss._decode_bboxes(reg, anchors)
    targets = _targets(counts, 11)
    gt_px = [b * SIZE for b in targets["bboxes"]]
    common = dict(anchor_strides=anchor_strides, num_per_level=per_level, collect_stats=True)
    ref = ReferenceTaskAlignedAssigner(**{k: v for k, v in common.items() if k == "collect_stats"})
    ref(cls, boxes, targets["labels"], gt_px, anchors, STRIDES,
        anchor_strides=anchor_strides, num_per_level=per_level)
    new = TaskAlignedAssigner(collect_stats=True)
    new(cls, boxes, targets["labels"], gt_px, anchors, STRIDES,
        anchor_strides=anchor_strides, num_per_level=per_level)
    assert new.last_stats == ref.last_stats and new.last_stats


def _run(loss_module, preds, targets, mask_outputs=None):
    leaves = _leaves(preds)
    extra = None
    mo_leaves = []
    if mask_outputs is not None:
        mo_leaves = [c.clone().requires_grad_(True) for c in mask_outputs["mask_coefs"]]
        proto = mask_outputs["protos"].clone().requires_grad_(True)
        mo_leaves.append(proto)
        extra = {"mask_coefs": mo_leaves[:-1], "protos": proto}
    loss, parts = (
        loss_module(leaves, targets) if extra is None else loss_module(leaves, targets, extra)
    )
    loss.backward()
    grads = [leaf.grad for leaf in leaves + mo_leaves]
    return loss.detach(), {k: float(v) for k, v in parts.items()}, grads


@pytest.mark.parametrize("seed", range(len(COUNTS)))
@pytest.mark.parametrize("with_masks", [False, True])
@pytest.mark.parametrize("padded", [False, True])
@pytest.mark.parametrize("kwargs", [{}, {"assigner_level_ranges": [48, 96], "assigner_center_radius": 1.0}])
def test_loss_values_and_gradients_match_the_loop(seed, with_masks, padded, kwargs) -> None:
    counts = COUNTS[seed]
    B = len(counts)
    preds = _predictions(B, seed)
    targets = _targets(counts, seed, with_masks=with_masks)
    mo = _mask_outputs(B, seed) if with_masks else None

    old_loss, old_parts, old_grads = _run(
        ReferenceYOLOv8Loss(num_classes=NC, strides=STRIDES, **kwargs), preds, targets, mo
    )
    new_targets = _padded(targets) if padded else targets
    new_loss, new_parts, new_grads = _run(
        YOLOv8Loss(num_classes=NC, strides=STRIDES, **kwargs), preds, new_targets, mo
    )

    torch.testing.assert_close(new_loss, old_loss, rtol=1e-5, atol=1e-6)
    assert new_parts.keys() == old_parts.keys()
    for key in old_parts:
        assert new_parts[key] == pytest.approx(old_parts[key], rel=1e-5, abs=1e-6), key
    for new_g, old_g in zip(new_grads, old_grads):
        torch.testing.assert_close(new_g, old_g, rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("seed", [0, 3])
def test_ignore_classes_matches_the_loop(seed) -> None:
    counts = COUNTS[seed]
    preds = _predictions(len(counts), seed)
    targets = _targets(counts, seed, with_masks=True)
    mo = _mask_outputs(len(counts), seed)
    kwargs = {"ignore_classes": [0], "class_weights": [1.0, 2.0]}
    old = _run(ReferenceYOLOv8Loss(num_classes=NC, strides=STRIDES, **kwargs), preds, targets, mo)
    for form in (targets, _padded(targets)):
        new = _run(YOLOv8Loss(num_classes=NC, strides=STRIDES, **kwargs), preds, form, mo)
        torch.testing.assert_close(new[0], old[0], rtol=1e-5, atol=1e-6)
        for key in old[1]:
            assert new[1][key] == pytest.approx(old[1][key], rel=1e-5, abs=1e-6)
        for new_g, old_g in zip(new[2], old[2]):
            torch.testing.assert_close(new_g, old_g, rtol=1e-4, atol=1e-6)


def test_ignore_classes_with_only_ignored_gt_uses_the_batch_floor() -> None:
    preds = _predictions(2, 5)
    targets = {
        "bboxes": [torch.tensor([[0.5, 0.5, 0.3, 0.3]]), torch.zeros(0, 4)],
        "labels": [torch.tensor([0]), torch.zeros(0, dtype=torch.long)],
    }
    kwargs = {"ignore_classes": [0]}
    old = _run(ReferenceYOLOv8Loss(num_classes=NC, strides=STRIDES, **kwargs), preds, targets)
    new = _run(YOLOv8Loss(num_classes=NC, strides=STRIDES, **kwargs), preds, targets)
    torch.testing.assert_close(new[0], old[0], rtol=1e-5, atol=1e-6)
    for new_g, old_g in zip(new[2], old[2]):
        torch.testing.assert_close(new_g, old_g, rtol=1e-4, atol=1e-6)


def test_loss_dict_is_floats_by_default_and_detached_tensors_on_request() -> None:
    preds = _predictions(2, 1)
    targets = _targets([1, 2], 1, with_masks=True)
    mo = _mask_outputs(2, 1)
    _, parts = YOLOv8Loss(num_classes=NC, strides=STRIDES)(_leaves(preds), targets, mo)
    assert all(isinstance(v, float) for v in parts.values())
    deferred = YOLOv8Loss(num_classes=NC, strides=STRIDES, log_as_tensors=True)
    _, tensor_parts = deferred(_leaves(preds), targets, mo)
    assert all(isinstance(v, torch.Tensor) and not v.requires_grad and v.ndim == 0
               for v in tensor_parts.values())
    for key, value in parts.items():
        assert float(tensor_parts[key]) == pytest.approx(value, rel=1e-6)


def test_pad_targets_layout() -> None:
    targets = _targets([0, 2, 1], 3, with_masks=True)
    boxes, labels, counts, masks, has_mask = pad_targets(targets)
    assert boxes.shape == (3, 2, 4) and labels.shape == (3, 2)
    assert counts.tolist() == [0, 2, 1]
    assert torch.equal(boxes[1], targets["bboxes"][1])
    assert torch.equal(boxes[2, 0], targets["bboxes"][2][0]) and not boxes[2, 1].any()
    assert masks.shape == (3, 2, PROTO, PROTO) and has_mask.shape == (3, 2)
    assert not has_mask[0].any()


def test_pad_targets_of_an_all_empty_batch() -> None:
    targets = {"bboxes": [torch.zeros(0, 4)] * 2, "labels": [torch.zeros(0, dtype=torch.long)] * 2}
    boxes, labels, counts, masks, has_mask = pad_targets(targets)
    assert boxes.shape == (2, 0, 4) and counts.tolist() == [0, 0] and masks is None
    loss, parts = YOLOv8Loss(num_classes=NC, strides=STRIDES)(_leaves(_predictions(2, 2)), targets)
    assert torch.isfinite(loss) and parts["box_loss"] == 0.0


@pytest.mark.parametrize("seed", range(3))
def test_ties_in_scores_and_distances_resolve_like_the_loop(seed) -> None:
    """Constant predictions make every alignment score and anchor distance tie."""
    counts = COUNTS[seed + 1]
    preds = [torch.zeros_like(p) for p in _predictions(len(counts), seed)]
    targets = _targets(counts, seed, with_masks=True)
    mo = _mask_outputs(len(counts), seed)
    old = _run(ReferenceYOLOv8Loss(num_classes=NC, strides=STRIDES), preds, targets, mo)
    new = _run(YOLOv8Loss(num_classes=NC, strides=STRIDES), preds, targets, mo)
    torch.testing.assert_close(new[0], old[0], rtol=1e-5, atol=1e-6)
    for new_g, old_g in zip(new[2], old[2]):
        torch.testing.assert_close(new_g, old_g, rtol=1e-4, atol=1e-6)
