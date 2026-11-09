"""CrossKD (Wang et al., CVPR 2024): the student's mid head features go through the frozen teacher head."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from src.models.student.head import DecoupledHead
from src.models.student.student_model import StudentModel
from src.training import kd_crosshead as ch
from src.training.kd_loss import build_region_masks

SIZE = 64

# Fruit covers cells [2, 6) of an 8x8 map, damage only cell (3, 3).
_FRUIT_BOX = [0.5, 0.5, 0.5, 0.5]
_DAMAGE_BOX = [0.4375, 0.4375, 0.125, 0.125]


def _targets():
    return [torch.tensor([_FRUIT_BOX, _DAMAGE_BOX])], [torch.tensor([0, 1])]


def _spread_head(model: StudentModel) -> None:
    """The YOLO init (std 0.01) makes every head output nearly constant: give the head real signal."""
    for module in model.head.modules():
        if isinstance(module, torch.nn.Conv2d):
            torch.nn.init.normal_(module.weight, std=0.3)


def _pair(seed: int = 0):
    torch.manual_seed(seed)
    student = StudentModel().train()
    teacher = StudentModel().eval()
    _spread_head(student)
    _spread_head(teacher)
    for p in teacher.parameters():
        p.requires_grad = False
    return student, teacher


def _strides(teacher_out) -> list[int]:
    return [SIZE // t.shape[-1] for t in teacher_out["cls_preds"]]


def _loss(student, teacher, gates=None, x=None):
    x = torch.randn(2, 3, SIZE, SIZE) if x is None else x
    with torch.no_grad():
        t_out = teacher(x)
    s_out = student(x)
    if gates is None:
        gates = ch.compute_gate("none", teacher_cls_logits=t_out["cls_preds"])
    return ch.crosshead_loss(s_out, t_out, teacher.head, gates, _strides(t_out)), s_out, t_out


# --- align_scale ---------------------------------------------------------------------------------


def test_align_scale_gives_the_student_the_teacher_channel_statistics() -> None:
    stu = torch.randn(4, 3, 5, 5) * 3 + 2
    tea = torch.randn(4, 3, 5, 5) * 0.5 - 1
    out = ch.align_scale(stu, tea)
    flat = lambda t: t.transpose(0, 1).reshape(3, -1)  # noqa: E731
    assert torch.allclose(flat(out).mean(1), flat(tea).mean(1), atol=1e-5)
    assert torch.allclose(flat(out).std(1), flat(tea).std(1), atol=1e-4)


def test_align_scale_matches_the_reference_formula_and_never_updates_the_teacher() -> None:
    stu = torch.randn(2, 3, 4, 4, requires_grad=True)
    tea = torch.randn(2, 3, 4, 4, requires_grad=True)
    out = ch.align_scale(stu, tea.detach())
    s, t = stu.transpose(0, 1).reshape(3, -1), tea.transpose(0, 1).reshape(3, -1)
    expected = (s - s.mean(1, keepdim=True)) / (s.std(1, keepdim=True) + 1e-6) * t.std(
        1, keepdim=True
    ) + t.mean(1, keepdim=True)
    assert torch.allclose(out.transpose(0, 1).reshape(3, -1), expected, atol=1e-5)
    out.sum().backward()
    assert stu.grad is not None and tea.grad is None


# --- terms ---------------------------------------------------------------------------------------


def test_classification_term_is_bce_times_the_absolute_probability_gap() -> None:
    z = torch.tensor([[2.0, -1.0]])
    p_t = torch.tensor([[0.3, 0.9]])
    expected = F.binary_cross_entropy_with_logits(z, p_t, reduction="none") * (z.sigmoid() - p_t).abs()
    assert torch.allclose(ch.quality_focal_term(z, p_t), expected)


def test_normalization_is_batch_wide_with_a_floor_of_one() -> None:
    cls_term = torch.tensor([1.0, 2.0, 3.0, 4.0])  # already summed over classes
    giou = torch.tensor([0.5, 0.5, 1.0, 1.0])
    pmax = torch.tensor([0.5, 0.5, 1.0, 1.0])
    gate = torch.tensor([1.0, 0.0, 1.0, 1.0])
    l_cls, l_reg = ch.normalize_terms(cls_term, giou, pmax, gate)
    denom = max((gate * pmax).sum().item(), 1.0)  # 2.5
    assert l_cls.item() == pytest.approx((1.0 + 3.0 + 4.0) / denom)
    assert l_reg.item() == pytest.approx((0.25 + 1.0 + 1.0) / denom)
    # Sum of gated pmax below one: the divisor is clamped to 1.
    l_cls, _ = ch.normalize_terms(cls_term, giou, pmax * 0.01, gate)
    assert l_cls.item() == pytest.approx(8.0)


def test_weights_are_the_official_atss_giou_ones() -> None:
    assert ch.CROSS_CLS_WEIGHT == 1.0 and ch.CROSS_REG_WEIGHT == 1.0


# --- cross-head loss -----------------------------------------------------------------------------


def test_an_aligned_student_has_nearly_no_loss() -> None:
    _, teacher = _pair()
    x = torch.randn(2, 3, SIZE, SIZE)
    with torch.no_grad():
        t_out = teacher(x)
    # A "student" whose mid features equal the teacher's is mapped onto them by align_scale.
    s_out = {k: t_out[k] for k in t_out}
    gates = ch.compute_gate("none", teacher_cls_logits=t_out["cls_preds"])
    total, parts = ch.crosshead_loss(s_out, t_out, teacher.head, gates, _strides(t_out))
    assert total.item() < 1e-3 and parts["cross_cls"] < 1e-3 and parts["cross_reg"] < 1e-3


def test_a_student_identical_to_the_teacher_reproduces_the_teacher_predictions() -> None:
    """The mid feature is pre-activation, so the cross path must activate it exactly once."""
    torch.manual_seed(0)
    head = DecoupledHead(c_in=8, c_stem=16).eval()
    for module in head.modules():
        if isinstance(module, torch.nn.Conv2d):
            torch.nn.init.normal_(module.weight, std=0.3)
    with torch.no_grad():
        out = head(torch.randn(2, 8, 12, 12))
        cls_mid, reg_mid = out["distill_cls_mid"], out["distill_reg_mid"]
        cls_hat, reg_hat = ch.cross_head_predictions(head, cls_mid, reg_mid, cls_mid, reg_mid)
        assert torch.allclose(cls_hat, out["cls_logits"], atol=1e-3)
        assert torch.allclose(reg_hat, out["reg_logits"], atol=1e-3)
        # An already activated feature would be activated twice on this path.
        twice, _ = ch.cross_head_predictions(
            head, F.silu(cls_mid), F.silu(reg_mid), F.silu(cls_mid), F.silu(reg_mid)
        )
        assert not torch.allclose(twice, out["cls_logits"], atol=1e-3)


def test_total_is_the_weighted_sum_and_parts_are_floats() -> None:
    student, teacher = _pair()
    (total, parts), _, _ = _loss(student, teacher)
    assert set(parts) == {"cross_cls", "cross_reg"} and all(isinstance(v, float) for v in parts.values())
    assert total.item() == pytest.approx(
        ch.CROSS_CLS_WEIGHT * parts["cross_cls"] + ch.CROSS_REG_WEIGHT * parts["cross_reg"], rel=1e-5
    )


def test_a_zero_gate_gives_zero_loss() -> None:
    student, teacher = _pair()
    x = torch.randn(2, 3, SIZE, SIZE)
    with torch.no_grad():
        t_out = teacher(x)
    zeros = [torch.zeros_like(c[:, 0]) for c in t_out["cls_preds"]]
    (total, _), _, _ = _loss(student, teacher, gates=zeros, x=x)
    assert total.item() == 0.0


def test_only_the_first_stem_conv_and_what_precedes_it_get_gradient() -> None:
    student, teacher = _pair()
    (total, _), _, _ = _loss(student, teacher)
    total.backward()
    # Deep levels of a random network carry almost no signal, so the finest level proves the path.
    finest = student.head.heads[0]
    for stem in (finest.cls_stem, finest.reg_stem):
        assert stem[0].conv.weight.grad is not None and stem[0].conv.weight.grad.abs().sum() > 0
    for head in student.head.heads:
        for stem in (head.cls_stem, head.reg_stem):
            assert stem[1].conv.weight.grad is None or stem[1].conv.weight.grad.abs().sum() == 0
        for pred in (head.cls_pred, head.reg_pred):
            assert pred.weight.grad is None or pred.weight.grad.abs().sum() == 0
    upstream = [p.grad for p in student.backbone.parameters() if p.grad is not None]
    assert upstream and sum(g.abs().sum() for g in upstream) > 0
    assert all(p.grad is None for p in teacher.parameters())


def test_the_teacher_stays_frozen_and_in_eval() -> None:
    student, teacher = _pair()
    _loss(student, teacher)[0][0].backward()
    assert not teacher.training
    assert all(not p.requires_grad for p in teacher.parameters())


# --- gates ---------------------------------------------------------------------------------------


def _delta_maps(values: torch.Tensor):
    """One level of an 8x8 grid for one image."""
    return [values.reshape(1, 8, 8)]


def test_nir_dependence_is_the_probability_gap_on_the_damage_channel() -> None:
    full = [torch.zeros(1, 2, 2, 2)]
    ablated = [torch.zeros(1, 2, 2, 2)]
    full[0][:, 1] = torch.tensor([[0.0, 2.0], [-2.0, 1.0]])
    full[0][:, 0] = 5.0  # other channel must not matter
    delta = ch.nir_dependence(full, ablated)
    expected = (full[0][:, 1].sigmoid() - 0.5).abs()
    assert len(delta) == 1 and torch.allclose(delta[0], expected)


def test_none_gate_is_one_everywhere() -> None:
    logits = [torch.randn(2, 2, 8, 8), torch.randn(2, 2, 4, 4)]
    gate = ch.compute_gate("none", teacher_cls_logits=logits)
    assert [g.shape for g in gate] == [(2, 8, 8), (2, 4, 4)] and all((g == 1).all() for g in gate)


def _universe():
    bboxes, labels = _targets()
    masks = build_region_masks(bboxes, labels, 8, 8)
    return (masks["damage"] | masks["fruit"])[0]


def test_nir_high_and_low_split_the_universe_at_the_median_and_are_disjoint() -> None:
    bboxes, labels = _targets()
    delta = _delta_maps(torch.rand(64, generator=torch.Generator().manual_seed(1)))
    high = ch.compute_gate("nir_high", nir_delta=delta, bboxes=bboxes, labels=labels)[0][0]
    low = ch.compute_gate("nir_low", nir_delta=delta, bboxes=bboxes, labels=labels)[0][0]
    universe = _universe()
    assert torch.equal((high + low) > 0, universe) and not ((high > 0) & (low > 0)).any()
    assert set(high.unique().tolist()) <= {0.0, 1.0}
    median = delta[0][0][universe].median()
    assert (delta[0][0][high > 0] > median).all() and (delta[0][0][low > 0] <= median).all()
    assert high.sum() > 0 and low.sum() > 0


def test_equal_deltas_put_the_whole_universe_in_nir_low() -> None:
    bboxes, labels = _targets()
    delta = _delta_maps(torch.full((64,), 0.3))
    high = ch.compute_gate("nir_high", nir_delta=delta, bboxes=bboxes, labels=labels)[0][0]
    low = ch.compute_gate("nir_low", nir_delta=delta, bboxes=bboxes, labels=labels)[0][0]
    assert high.sum() == 0 and torch.equal(low > 0, _universe())


def test_an_image_without_fruit_or_damage_gets_all_zero_gates() -> None:
    delta = _delta_maps(torch.rand(64))
    bboxes, labels = [torch.zeros(0, 4)], [torch.zeros(0, dtype=torch.long)]
    for kind in ("nir_high", "nir_low"):
        gate = ch.compute_gate(kind, nir_delta=delta, bboxes=bboxes, labels=labels)
        assert (gate[0] == 0).all()


def test_nir_gates_split_each_image_on_its_own_median() -> None:
    bboxes, labels = _targets()
    bboxes, labels = bboxes * 2, labels * 2
    base = torch.rand(64, generator=torch.Generator().manual_seed(2))
    delta = [torch.stack([base, base * 0.01]).reshape(2, 8, 8)]  # second image: same ranks, tiny scale
    high = ch.compute_gate("nir_high", nir_delta=delta, bboxes=bboxes, labels=labels)[0]
    assert torch.equal(high[0], high[1])


def test_teacher_better_requires_a_correct_teacher_with_lower_bce_than_the_student() -> None:
    bboxes, labels = _targets()
    masks = build_region_masks(bboxes, labels, 8, 8)
    y = masks["damage"][0]
    teacher = torch.full((1, 2, 8, 8), -4.0)
    student = torch.full((1, 2, 8, 8), -4.0)
    teacher[0, 1][y] = 3.0  # correct and confident on the lesion
    student[0, 1][y] = 1.0  # correct, but less so: teacher better
    teacher[0, 1][0, 0] = 3.0  # teacher wrong on a background cell
    student[0, 1][0, 0] = -3.0
    student[0, 1][7, 7] = -1.0  # both correct, student worse at a background cell
    gate = ch.compute_gate(
        "teacher_better", teacher_cls_logits=[teacher], student_cls_logits=[student],
        bboxes=bboxes, labels=labels,
    )[0][0]
    assert gate[y].item() == 1.0
    assert gate[0, 0].item() == 0.0  # wrong teacher
    assert gate[7, 7].item() == 1.0  # correct and strictly better
    assert gate[7, 0].item() == 0.0  # tie: not strictly lower


def test_teacher_better_does_not_use_student_gradients() -> None:
    bboxes, labels = _targets()
    student = torch.randn(1, 2, 8, 8, requires_grad=True)
    gate = ch.compute_gate(
        "teacher_better", teacher_cls_logits=[torch.randn(1, 2, 8, 8)], student_cls_logits=[student],
        bboxes=bboxes, labels=labels,
    )
    assert not gate[0].requires_grad


def test_an_unknown_gate_is_refused() -> None:
    with pytest.raises(ValueError, match="kd_gate"):
        ch.compute_gate("bogus", teacher_cls_logits=[torch.zeros(1, 2, 2, 2)])
