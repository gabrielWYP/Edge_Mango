"""Segment twins of the arm tables of the analysis scripts (`src/evaluation/arms.py`)."""

from __future__ import annotations

from scripts import export_run_results as export
from scripts import run_small_arms as small
from scripts import val_operating_points as vop
from scripts import zone_analysis as zones
from src.evaluation import arms

DETECT_ARMS = ("est_rgb", "est_rgbnir", "kd_rgb", "kd_rgbnir", "est_nir")


def test_every_arm_has_a_seg_twin_and_only_kd_arms_a_segdt_one() -> None:
    names = arms.with_segment_twins(["est_rgb", "kd_rgbnir", "crosskd_rgbnir_high"])
    assert names == ["est_rgb", "kd_rgbnir", "crosskd_rgbnir_high",
                     "est_rgb_seg", "kd_rgbnir_seg", "kd_rgbnir_segdt",
                     "crosskd_rgbnir_high_seg", "crosskd_rgbnir_high_segdt"]


def test_the_twin_names_are_the_arms_run_small_arms_defines() -> None:
    assert set(arms.with_segment_twins(DETECT_ARMS)) <= set(small.ARMS)


def test_detect_names_keep_their_order_at_the_front() -> None:
    assert arms.with_segment_twins(DETECT_ARMS)[: len(DETECT_ARMS)] == list(DETECT_ARMS)


def test_operating_point_styles_give_a_twin_its_base_colour_and_its_own_line_style() -> None:
    assert list(vop.ARMS)[:len(DETECT_ARMS)] == list(DETECT_ARMS)
    for base in ("est_rgb", "kd_rgbnir", "crosskd_rgbnir"):
        color, style, label = vop.ARMS[base]
        for suffix in ("_seg", "_segdt"):
            if f"{base}{suffix}" in vop.ARMS:
                twin_color, twin_style, twin_label = vop.ARMS[f"{base}{suffix}"]
                assert twin_color == color and twin_style != style and twin_label.startswith(f"{base}{suffix}")
    assert {"est_rgb_seg", "kd_rgbnir_seg", "kd_rgbnir_segdt", "crosskd_rgbnir_seg"} <= set(vop.ARMS)
    assert "est_rgb_segdt" not in vop.ARMS  # no detect teacher to distill from


def test_the_detect_operating_point_entries_are_untouched() -> None:
    assert vop.ARMS["est_rgb"] == ("#2a78d6", "-", "est_rgb (RGB)")


def test_zone_analysis_reads_the_segment_twins_after_the_detect_arms() -> None:
    assert zones.ARMS[:5] == DETECT_ARMS and "kd_rgb_segdt" in zones.ARMS


def test_the_export_reads_each_arm_once() -> None:
    assert len(export.EXPORT_ARMS) == len(set(export.EXPORT_ARMS))
    assert {"est_rgb_seg", "kd_rgb_segdt", "crosskd_rgb_seg"} <= set(export.EXPORT_ARMS)
