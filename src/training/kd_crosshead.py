"""CrossKD (Wang et al., CVPR 2024, arXiv 2306.11369) with an optional per-cell gate.

The student's mid head features (the first stem conv of each head branch, before
its activation) are aligned to the teacher's statistics, activated, and pushed
through the FROZEN teacher head tail (second stem conv and prediction conv). The resulting
cross-head predictions are pulled toward the teacher's own predictions, so the
gradient reaches the student's first stem conv and everything before it, while
the student's own second stem conv and prediction conv get none.

Semantics follow the official implementation (ATSS GIoU config):

- ``align_scale``: per channel over (N, H, W), ``(s - mean_s) / (std_s + 1e-6) * std_t + mean_t``.
- Classification: ``KDQualityFocalLoss`` with beta = 1, per cell and channel
  ``BCE(z_hat, p_t) * |sigmoid(z_hat) - p_t|`` with ``p_t = sigmoid(teacher logits)``.
- Regression: GIoU between the cross-head box and the teacher box, decoded to
  pixels along the same path as the detection loss, weighted per cell by the
  teacher's highest class probability.
- Normalization is batch-wide over every level. Deviation from the official code:
  both terms divide by the gated sum of the teacher's top probability, floored at 1.
  The official classification term divides by the assigner's positive count, which
  this trainer does not expose and which would not follow the gate.

The gate (``kd_gate``) multiplies both terms per cell. ``"none"`` keeps the whole
map, as CrossKD does. The others restrict it by how much the teacher's damage
decision depends on NIR (``nir_high`` / ``nir_low``) or by whether the teacher
beats the student on the cell (``teacher_better``, analogue of Chen et al. 2022).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torchvision.ops import generalized_box_iou_loss

from .config import DAMAGE_CLASS_ID
from .kd_loss import build_region_masks
from .loss import TaskAlignedAssigner, YOLOv8Loss, _generate_anchors

# Official ATSS GIoU CrossKD weights (loss_cls_kd, loss_reg_kd). Not config knobs.
CROSS_CLS_WEIGHT = 1.0
CROSS_REG_WEIGHT = 1.0

KD_MODES = ("feature", "crosshead")
KD_GATES = ("none", "nir_high", "nir_low", "teacher_better")
NIR_GATES = ("nir_high", "nir_low")

# Added to the std before dividing, as in the official ``align_scale``.
_ALIGN_EPS = 1e-6


def align_scale(stu: torch.Tensor, tea: torch.Tensor) -> torch.Tensor:
    """Give `stu` the per-channel mean and std of `tea`, both pooled over (N, H, W).

    ``tea`` is detached: the teacher is a target, never updated.
    """
    tea = tea.detach()
    n, c, h, w = stu.shape
    s = stu.transpose(0, 1).reshape(c, -1)
    t = tea.transpose(0, 1).reshape(c, -1)
    s_mean, s_std = s.mean(dim=1, keepdim=True), s.std(dim=1, keepdim=True)
    t_mean, t_std = t.mean(dim=1, keepdim=True), t.std(dim=1, keepdim=True)
    out = (s - s_mean) / (s_std + _ALIGN_EPS) * t_std + t_mean
    return out.reshape(c, n, h, w).transpose(0, 1)


def cross_head_predictions(
    teacher_level_head,
    stu_cls_mid: torch.Tensor,
    stu_reg_mid: torch.Tensor,
    tea_cls_mid: torch.Tensor,
    tea_reg_mid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Student mid features through one level of the frozen teacher head.

    Deliberately NOT under ``no_grad``: the teacher's parameters have
    ``requires_grad=False``, but the activations must stay differentiable so the
    gradient reaches the student.
    """
    cls_x = F.silu(align_scale(stu_cls_mid, tea_cls_mid))
    reg_x = F.silu(align_scale(stu_reg_mid, tea_reg_mid))
    cls_hat = teacher_level_head.cls_pred(teacher_level_head.cls_stem[1](cls_x))
    reg_hat = teacher_level_head.reg_pred(teacher_level_head.reg_stem[1](reg_x))
    return cls_hat, reg_hat


def quality_focal_term(cls_hat: torch.Tensor, p_t: torch.Tensor) -> torch.Tensor:
    """Official ``KDQualityFocalLoss`` (beta = 1), per element, unreduced."""
    bce = F.binary_cross_entropy_with_logits(cls_hat, p_t, reduction="none")
    return bce * (cls_hat.sigmoid() - p_t).abs()


def normalize_terms(
    cls_term: torch.Tensor, giou: torch.Tensor, pmax: torch.Tensor, gate: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batch-wide (L_cls, L_reg) from per-cell terms.

    Args:
        cls_term: Classification term already summed over classes, per cell.
        giou: GIoU loss, per cell.
        pmax: Teacher's highest class probability, per cell.
        gate: Gate weight, per cell.
    """
    denom = (gate * pmax).sum().clamp(min=1.0)
    l_cls = (gate * cls_term).sum() / denom
    l_reg = (gate * pmax * giou).sum() / denom
    return l_cls, l_reg


def _flatten_levels(maps: list[torch.Tensor]) -> torch.Tensor:
    """(N, C, H, W) per level -> (N, sum H*W, C)."""
    return torch.cat([m.flatten(2).permute(0, 2, 1) for m in maps], dim=1)


def _boxes_xyxy(reg: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
    """Raw regression outputs (N, A, 4) -> pixel xyxy, as the detection loss decodes them."""
    return TaskAlignedAssigner._cxcywh_to_xyxy(YOLOv8Loss._decode_bboxes(reg, anchors))


def crosshead_loss(
    student_out: dict,
    teacher_out: dict,
    teacher_head,
    gates: list[torch.Tensor],
    strides: list[int],
) -> tuple[torch.Tensor, dict[str, float]]:
    """CrossKD loss over every level of a batch.

    Args:
        student_out: StudentModel output (needs the ``*_mid`` entries).
        teacher_out: Frozen teacher's output on the same batch, computed without grad.
        teacher_head: The teacher's ``YOLOStudentHead`` (frozen, in eval).
        gates: Per level (N, H, W) float weights from `compute_gate`.
        strides: Stride of each level, as the detection loss uses.

    Returns:
        ``CROSS_CLS_WEIGHT * L_cls + CROSS_REG_WEIGHT * L_reg`` and a dict with
        ``cross_cls`` and ``cross_reg`` as floats.
    """
    cls_hats, reg_hats = [], []
    for level, level_head in enumerate(teacher_head.heads):
        cls_hat, reg_hat = cross_head_predictions(
            level_head,
            student_out["distill_head_cls_mid"][level].float(),
            student_out["distill_head_reg_mid"][level].float(),
            teacher_out["distill_head_cls_mid"][level].float(),
            teacher_out["distill_head_reg_mid"][level].float(),
        )
        cls_hats.append(cls_hat)
        reg_hats.append(reg_hat)

    device = cls_hats[0].device
    feat_sizes = [tuple(c.shape[2:]) for c in cls_hats]
    anchors, _, _ = _generate_anchors(feat_sizes, strides, device)

    cls_hat = _flatten_levels(cls_hats)  # (N, A, C)
    reg_hat = _flatten_levels(reg_hats)  # (N, A, 4)
    t_cls = _flatten_levels([t.detach().float() for t in teacher_out["cls_preds"]])
    t_reg = _flatten_levels([t.detach().float() for t in teacher_out["reg_preds"]])
    gate = torch.cat([g.flatten(1) for g in gates], dim=1).to(device).float()  # (N, A)

    p_t = t_cls.sigmoid()
    pmax = p_t.max(dim=-1).values
    cls_term = quality_focal_term(cls_hat, p_t).sum(dim=-1)

    stu_boxes = _boxes_xyxy(reg_hat, anchors).reshape(-1, 4)
    tea_boxes = _boxes_xyxy(t_reg, anchors).reshape(-1, 4)
    giou = generalized_box_iou_loss(stu_boxes, tea_boxes, reduction="none").reshape(pmax.shape)

    l_cls, l_reg = normalize_terms(cls_term, giou, pmax, gate)
    total = CROSS_CLS_WEIGHT * l_cls + CROSS_REG_WEIGHT * l_reg
    return total, {"cross_cls": float(l_cls.detach()), "cross_reg": float(l_reg.detach())}


# --- Gates ---------------------------------------------------------------------------------------


def nir_dependence(
    teacher_logits_full: list[torch.Tensor], teacher_logits_ablated: list[torch.Tensor]
) -> list[torch.Tensor]:
    """Per level (N, H, W): ``|sigmoid(z_full) - sigmoid(z_ablated)|`` on the damage channel.

    "Ablated" is the same teacher forward with a NIR of zeros (the normalized
    mean), the ablation of ``scripts/kd_feature_audit.py``.
    """
    return [
        (full[:, DAMAGE_CLASS_ID].sigmoid() - ablated[:, DAMAGE_CLASS_ID].sigmoid()).abs()
        for full, ablated in zip(teacher_logits_full, teacher_logits_ablated)
    ]


def _nir_split_gate(
    delta: list[torch.Tensor], bboxes: list[torch.Tensor], labels: list[torch.Tensor], high: bool
) -> list[torch.Tensor]:
    """High or low half of the fruit-or-damage cells of each image, split at the median of `delta`."""
    gates = []
    for level_delta in delta:
        n, h, w = level_delta.shape
        masks = build_region_masks(bboxes, labels, h, w, device=level_delta.device)
        universe = masks["damage"] | masks["fruit"]
        gate = torch.zeros_like(level_delta, dtype=torch.float32)
        for i in range(n):
            if not universe[i].any():
                continue
            median = level_delta[i][universe[i]].median()
            above = level_delta[i] > median
            gate[i] = ((above if high else ~above) & universe[i]).float()
        gates.append(gate)
    return gates


def _teacher_better_gate(
    teacher_logits: list[torch.Tensor],
    student_logits: list[torch.Tensor],
    bboxes: list[torch.Tensor],
    labels: list[torch.Tensor],
) -> list[torch.Tensor]:
    """1 where the teacher is right on the damage channel and strictly beats the student's BCE."""
    gates = []
    for t_logits, s_logits in zip(teacher_logits, student_logits):
        n, _, h, w = t_logits.shape
        y = build_region_masks(bboxes, labels, h, w, device=t_logits.device)["damage"]
        target = y.float()
        t_damage = t_logits[:, DAMAGE_CLASS_ID].detach().float()
        s_damage = s_logits[:, DAMAGE_CLASS_ID].detach().float()
        correct = (t_damage.sigmoid() > 0.5) == y
        bce_t = F.binary_cross_entropy_with_logits(t_damage, target, reduction="none")
        bce_s = F.binary_cross_entropy_with_logits(s_damage, target, reduction="none")
        gates.append((correct & (bce_t < bce_s)).float())
    return gates


def compute_gate(
    kind: str,
    *,
    teacher_cls_logits: list[torch.Tensor] | None = None,
    student_cls_logits: list[torch.Tensor] | None = None,
    nir_delta: list[torch.Tensor] | None = None,
    bboxes: list[torch.Tensor] | None = None,
    labels: list[torch.Tensor] | None = None,
) -> list[torch.Tensor]:
    """Per-level (N, H, W) float weights for ``kd_gate=kind``.

    ``nir_high`` and ``nir_low`` split the fruit-or-damage cells of each image
    and level at the median of `nir_delta` inside that set: ``nir_high`` keeps
    the cells strictly above it, ``nir_low`` the rest. They are disjoint and
    their union is the set. With all-equal deltas ``nir_low`` gets every cell
    and ``nir_high`` none; an image without fruit or damage gets zeros.
    """
    if kind not in KD_GATES:
        raise ValueError(f"Invalid kd_gate '{kind}'. Must be one of: {KD_GATES}")
    if kind == "none":
        return [torch.ones_like(t[:, 0], dtype=torch.float32) for t in teacher_cls_logits]
    if kind in NIR_GATES:
        return _nir_split_gate(nir_delta, bboxes, labels, high=kind == "nir_high")
    return _teacher_better_gate(teacher_cls_logits, student_cls_logits, bboxes, labels)
