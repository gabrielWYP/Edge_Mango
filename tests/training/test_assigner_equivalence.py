"""The vectorized conflict resolution must assign exactly as the per-anchor loop it replaced."""

from __future__ import annotations

import pytest
import torch

from src.training.loss import TaskAlignedAssigner


class _LoopAssigner(TaskAlignedAssigner):
    """The per-anchor loop, applied after the same top-k of the parent's first pass."""

    def __call__(self, pred_scores, pred_bboxes, gt_labels, gt_bboxes, anchors, strides, **kwargs):
        B, num_anchors, _ = pred_scores.shape
        classes = torch.full((B, num_anchors), -1, dtype=torch.int64)
        boxes = torch.zeros((B, num_anchors, 4))
        scores = torch.zeros((B, num_anchors))
        fg = torch.zeros((B, num_anchors), dtype=torch.bool)
        for b in range(B):
            for gt_i in range(len(gt_labels[b])):
                # One GT at a time through the real assigner gives its candidates.
                _, _, candidate_scores, candidate_mask = super().__call__(
                    pred_scores[b:b + 1], pred_bboxes[b:b + 1], [gt_labels[b][gt_i:gt_i + 1]],
                    [gt_bboxes[b][gt_i:gt_i + 1]], anchors, strides, **kwargs,
                )
                indices = torch.nonzero(candidate_mask[0]).squeeze(1)
                for idx, score in zip(indices.tolist(), candidate_scores[0, indices].tolist()):
                    if scores[b, idx] < score or classes[b, idx] == -1:
                        classes[b, idx] = gt_labels[b][gt_i]
                        boxes[b, idx] = gt_bboxes[b][gt_i]
                        scores[b, idx] = score
                        fg[b, idx] = True
        return classes, boxes, scores, fg


def _anchors():
    anchors, strides, per_level = [], [], []
    for stride in (8, 16, 32):
        size = 640 // stride
        ys, xs = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
        grid = (torch.stack([xs, ys], dim=-1).reshape(-1, 2).float() + 0.5) * stride
        anchors.append(grid)
        strides.append(torch.full((len(grid),), float(stride)))
        per_level.append(len(grid))
    return torch.cat(anchors), torch.cat(strides), per_level


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("level_ranges", [None, [64, 128]])
def test_assignment_is_exactly_that_of_the_per_anchor_loop(seed, level_ranges) -> None:
    torch.manual_seed(seed)
    anchors, anchor_strides, per_level = _anchors()
    B, A = 3, len(anchors)
    scores = torch.rand(B, A, 2)
    pred = torch.cat([anchors + torch.randn(A, 2) * 8, torch.rand(A, 2) * 300 + 20], dim=1).repeat(B, 1, 1)
    gts, labels = [], []
    for _ in range(B):
        n = int(torch.randint(0, 5, (1,)))
        centers, sizes = torch.rand(n, 2) * 400 + 120, torch.rand(n, 2) * 300 + 30
        gts.append(torch.cat([centers, sizes], dim=1))
        labels.append(torch.randint(0, 2, (n,)))
    kwargs = dict(anchor_strides=anchor_strides, num_per_level=per_level)
    new = TaskAlignedAssigner(level_ranges=level_ranges)(scores, pred, labels, gts, anchors, [8, 16, 32], **kwargs)
    old = _LoopAssigner(level_ranges=level_ranges)(scores, pred, labels, gts, anchors, [8, 16, 32], **kwargs)
    for a, b in zip(new, old):
        assert torch.equal(a, b)
