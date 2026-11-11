"""YOLOv8-seg mask loss: positives from the TAL assignment, BCE inside the GT box, masks only."""

from __future__ import annotations

import pytest
import torch

from src.training.loss import TaskAlignedAssigner, YOLOv8Loss

SIZE = 64
PROTO = SIZE // 4  # 16
STRIDES = [8, 16, 32]
NM = 4


def _predictions(batch: int = 1, seed: int = 0) -> list[torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    return [
        torch.randn(batch, 6, SIZE // s, SIZE // s, generator=gen, requires_grad=True) for s in STRIDES
    ]


def _targets(masks: torch.Tensor | None = None, has_mask: bool = True) -> dict:
    # One damage instance covering the centre, 32x32 px = 8x8 prototype pixels.
    box = torch.tensor([[0.5, 0.5, 0.5, 0.5]])
    targets = {"bboxes": [box], "labels": [torch.tensor([1])]}
    if masks is not None:
        targets["masks"] = [masks]
        targets["has_mask"] = [torch.tensor([has_mask])]
    return targets


def _gt_mask() -> torch.Tensor:
    mask = torch.zeros(1, PROTO, PROTO, dtype=torch.uint8)
    mask[:, 4:12, 4:12] = 1  # exactly the box
    return mask


def _mask_outputs(coef_channel_value: float, proto_sign: float, requires_grad: bool = True) -> dict:
    """Prototype 0 is +/-10 inside the box and -/+10 outside; coefficient 0 is `coef_channel_value`."""
    protos = torch.zeros(1, NM, PROTO, PROTO)
    protos[0, 0] = -10.0 * proto_sign
    protos[0, 0, 4:12, 4:12] = 10.0 * proto_sign
    coefs = [torch.zeros(1, NM, SIZE // s, SIZE // s) for s in STRIDES]
    for c in coefs:
        c[:, 0] = coef_channel_value
    if requires_grad:
        protos.requires_grad_(True)
        for c in coefs:
            c.requires_grad_(True)
    return {"mask_coefs": coefs, "protos": protos}


def _loss(**kwargs) -> YOLOv8Loss:
    return YOLOv8Loss(strides=STRIDES, **kwargs)


def test_assigner_reports_the_gt_of_every_positive_anchor() -> None:
    assigner = TaskAlignedAssigner(topk=3)
    anchors = torch.tensor([[8.0, 8.0], [24.0, 8.0], [8.0, 24.0], [40.0, 40.0]])
    boxes = torch.tensor([[8.0, 8.0, 12.0, 12.0], [40.0, 40.0, 12.0, 12.0]])
    scores = torch.full((1, 4, 2), 0.5)
    pred = boxes[[0, 0, 0, 1]].unsqueeze(0)
    _, _, _, fg = assigner(
        scores, pred, [torch.tensor([0, 1])], [boxes], anchors, [8], anchor_strides=torch.full((4,), 8.0)
    )
    gt_idx = assigner.last_gt_idx
    assert gt_idx.shape == (1, 4)
    assert gt_idx[0, 0].item() == 0 and gt_idx[0, 3].item() == 1
    assert (gt_idx[~fg] == -1).all()


def test_detect_call_is_unchanged_and_has_no_mask_term() -> None:
    preds = _predictions()
    loss, parts = _loss()(preds, _targets())
    assert set(parts) == {"box_loss", "cls_loss"}
    # Mask targets alone (no mask outputs) change nothing: a detect student ignores them.
    loss_with_masks, _ = _loss()(preds, _targets(_gt_mask()))
    assert torch.equal(loss, loss_with_masks)


def test_mask_term_is_added_with_its_gain() -> None:
    preds = _predictions()
    targets = _targets(_gt_mask())
    base, _ = _loss(mask_weight=0.0)(preds, targets, _mask_outputs(0.0, 1.0))
    full, parts = _loss(mask_weight=2.0)(preds, targets, _mask_outputs(0.0, 1.0))
    assert "mask_loss" in parts and parts["mask_loss"] > 0
    assert full.item() - base.item() == pytest.approx(2.0 * parts["mask_loss"], rel=1e-4)


def test_mask_loss_is_small_for_a_matching_mask_and_large_for_an_inverted_one() -> None:
    preds, targets = _predictions(), _targets(_gt_mask())
    _, good = _loss()(preds, targets, _mask_outputs(1.0, +1.0))
    _, bad = _loss()(preds, targets, _mask_outputs(1.0, -1.0))
    assert good["mask_loss"] < 1e-3
    assert bad["mask_loss"] > 5.0


def test_only_the_gt_box_counts() -> None:
    # Right inside the box, wrong everywhere outside it: the crop makes the loss ~0.
    outputs = _mask_outputs(1.0, +1.0)
    with torch.no_grad():
        outputs["protos"][0, 0] = 10.0  # predicts "mask" over the whole map
    mask = torch.zeros(1, PROTO, PROTO, dtype=torch.uint8)
    mask[:, 4:12, 4:12] = 1
    _, parts = _loss()(_predictions(), _targets(mask), outputs)
    assert parts["mask_loss"] < 1e-3


def test_instances_without_a_mask_do_not_contribute() -> None:
    garbage = torch.ones(1, PROTO, PROTO, dtype=torch.uint8)
    loss, parts = _loss()(_predictions(), _targets(garbage, has_mask=False), _mask_outputs(1.0, -1.0))
    assert parts["mask_loss"] == 0.0
    detect_loss, _ = _loss()(_predictions(), _targets())
    assert loss.item() == pytest.approx(detect_loss.item(), rel=1e-6)


def test_gradient_reaches_protos_and_coefficients() -> None:
    outputs = _mask_outputs(0.3, 1.0)
    loss, _ = _loss()(_predictions(), _targets(_gt_mask()), outputs)
    loss.backward()
    assert outputs["protos"].grad is not None and outputs["protos"].grad.abs().sum() > 0
    assert any(c.grad is not None and c.grad.abs().sum() > 0 for c in outputs["mask_coefs"])


def test_gt_masks_at_another_resolution_are_resampled_to_the_prototypes() -> None:
    big = torch.zeros(1, SIZE, SIZE, dtype=torch.uint8)
    big[:, 16:48, 16:48] = 1  # the same box at input resolution
    _, parts = _loss()(_predictions(), _targets(big), _mask_outputs(1.0, +1.0))
    assert parts["mask_loss"] < 1e-2


def test_ignored_classes_filter_the_masks_with_the_boxes() -> None:
    targets = {
        "bboxes": [torch.tensor([[0.5, 0.5, 0.9, 0.9], [0.5, 0.5, 0.5, 0.5]])],
        "labels": [torch.tensor([0, 1])],
        "masks": [torch.stack([torch.ones(PROTO, PROTO, dtype=torch.uint8), _gt_mask()[0]])],
        "has_mask": [torch.tensor([True, True])],
    }
    _, parts = _loss(ignore_classes=[0])(_predictions(), targets, _mask_outputs(1.0, +1.0))
    assert parts["mask_loss"] < 1e-3  # scored against the damage mask, not the mango one


def test_batch_without_any_positive_gives_a_zero_mask_loss() -> None:
    targets = {"bboxes": [torch.zeros(0, 4)], "labels": [torch.zeros(0, dtype=torch.long)],
               "masks": [torch.zeros(0, PROTO, PROTO, dtype=torch.uint8)], "has_mask": [torch.zeros(0, dtype=torch.bool)]}
    loss, parts = _loss()(_predictions(), targets, _mask_outputs(0.0, 1.0))
    assert parts["mask_loss"] == 0.0 and torch.isfinite(loss)
