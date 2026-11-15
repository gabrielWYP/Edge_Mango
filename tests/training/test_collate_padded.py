"""`collate_fn` also hands the trainers one padded targets tensor instead of per-image lists."""

from __future__ import annotations

import pytest
import torch

from src.training.dataset import collate_fn
from src.training.loss import YOLOv8Loss

SIZE = 64
PROTO = SIZE // 4
STRIDES = [8, 16, 32]


def _item(n: int, seed: int, masks: bool = False) -> dict:
    gen = torch.Generator().manual_seed(seed)
    item = {
        "rgb": torch.zeros(3, SIZE, SIZE),
        "nir": torch.zeros(1, SIZE, SIZE),
        "bboxes": torch.cat([torch.rand(n, 2, generator=gen) * 0.5 + 0.25,
                             torch.rand(n, 2, generator=gen) * 0.3 + 0.1], dim=1),
        "labels": torch.randint(0, 2, (n,), generator=gen),
        "image_id": f"img{seed}",
    }
    if masks:
        item["masks"] = (torch.rand(n, PROTO, PROTO, generator=gen) > 0.5).to(torch.uint8)
        item["has_mask"] = torch.rand(n, generator=gen) > 0.3
    return item


def test_padded_fields_mirror_the_lists_and_the_lists_stay() -> None:
    items = [_item(0, 0), _item(2, 1), _item(1, 2)]
    batch = collate_fn(items)
    assert [len(b) for b in batch["bboxes"]] == [0, 2, 1]  # untouched list fields
    assert batch["bboxes_padded"].shape == (3, 2, 4)
    assert batch["labels_padded"].shape == (3, 2) and batch["labels_padded"].dtype == torch.long
    assert batch["counts"].tolist() == [0, 2, 1]
    for i, (boxes, labels) in enumerate(zip(batch["bboxes"], batch["labels"])):
        n = len(boxes)
        assert torch.equal(batch["bboxes_padded"][i, :n], boxes)
        assert torch.equal(batch["labels_padded"][i, :n], labels)
        assert not batch["bboxes_padded"][i, n:].any()


def test_padded_masks_follow_the_instances() -> None:
    items = [_item(1, 0, masks=True), _item(3, 1, masks=True)]
    batch = collate_fn(items)
    assert batch["masks_padded"].shape == (2, 3, PROTO, PROTO)
    assert batch["has_mask_padded"].shape == (2, 3)
    assert torch.equal(batch["masks_padded"][1], batch["masks"][1])
    assert not batch["has_mask_padded"][0, 1:].any()
    assert "masks_padded" not in collate_fn([_item(1, 0)])


def test_an_all_empty_batch_pads_to_zero_width() -> None:
    batch = collate_fn([_item(0, 0), _item(0, 1)])
    assert batch["bboxes_padded"].shape == (2, 0, 4) and batch["counts"].tolist() == [0, 0]


@pytest.mark.parametrize("masks", [False, True])
def test_loss_gives_identical_results_from_lists_and_from_the_collated_padding(masks) -> None:
    batch = collate_fn([_item(2, 0, masks), _item(0, 1, masks), _item(3, 2, masks)])
    gen = torch.Generator().manual_seed(7)
    preds = [torch.randn(3, 6, SIZE // s, SIZE // s, generator=gen) for s in STRIDES]
    mask_outputs = None
    if masks:
        mask_outputs = {
            "mask_coefs": [torch.randn(3, 4, SIZE // s, SIZE // s, generator=gen) for s in STRIDES],
            "protos": torch.randn(3, 4, PROTO, PROTO, generator=gen),
        }
    lists = {"bboxes": batch["bboxes"], "labels": batch["labels"]}
    padded = {k: batch[k] for k in ("bboxes_padded", "labels_padded", "counts")}
    if masks:
        lists.update(masks=batch["masks"], has_mask=batch["has_mask"])
        padded.update(masks_padded=batch["masks_padded"], has_mask_padded=batch["has_mask_padded"])
    criterion = YOLOv8Loss(num_classes=2, strides=STRIDES)
    results = []
    for targets in (lists, padded):
        leaves = [p.clone().requires_grad_(True) for p in preds]
        args = (leaves, targets) if mask_outputs is None else (leaves, targets, mask_outputs)
        loss, parts = criterion(*args)
        loss.backward()
        results.append((loss.detach(), parts, [leaf.grad for leaf in leaves]))
    assert torch.equal(results[0][0], results[1][0])
    assert results[0][1] == results[1][1]
    for a, b in zip(results[0][2], results[1][2]):
        assert torch.equal(a, b)
