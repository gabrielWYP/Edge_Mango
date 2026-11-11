"""Segment twins of the arm tables of the analysis scripts (`src/evaluation/arms.py`)."""

from __future__ import annotations

from scripts import export_run_results as export
from scripts import run_small_arms as small
from scripts import val_operating_points as vop
from scripts import zone_analysis as zones
from src.evaluation import arms

DETECT_ARMS = ("est_rgb", "est_rgbnir", "kd_rgb", "kd_rgbnir", "est_nir")


def test_undistilled_arms_become_masters_and_distilled_ones_keep_their_name() -> None:
    names = arms.with_segment_twins(["est_rgb", "est_nir", "kd_rgbnir", "crosskd_rgbnir_high"])
    assert names == ["est_rgb", "est_nir", "kd_rgbnir", "crosskd_rgbnir_high",
                     "master_rgb_seg", "master_nir_seg", "kd_rgbnir_seg", "crosskd_rgbnir_high_seg"]
    assert not any(name.endswith("_segdt") for name in names)


def test_the_presets_list_the_masters_before_the_crosskd_arms_and_leave_out_the_mse_kd() -> None:
    core, full = arms.SEGMENT_PRESETS["seg"], arms.SEGMENT_PRESETS["seg-full"]
    assert core == ("master_rgb_seg", "master_rgbnir_seg", "master_nir_seg", "crosskd_rgb_seg", "crosskd_rgbnir_seg")
    assert full == core + ("crosskd_rgbnir_high_seg", "crosskd_rgbnir_low_seg", "crosskd_rgbnir_better_seg")
    assert not any(name.startswith("kd_") for name in full)


def test_the_twin_names_are_the_arms_run_small_arms_defines() -> None:
    assert set(arms.with_segment_twins(DETECT_ARMS)) <= set(small.ARMS)


def test_detect_names_keep_their_order_at_the_front() -> None:
    assert arms.with_segment_twins(DETECT_ARMS)[: len(DETECT_ARMS)] == list(DETECT_ARMS)


def test_operating_point_styles_give_a_twin_its_base_colour_and_its_own_line_style() -> None:
    assert list(vop.ARMS)[:len(DETECT_ARMS)] == list(DETECT_ARMS)
    for base, twin in (("est_rgb", "master_rgb_seg"), ("est_rgbnir", "master_rgbnir_seg"),
                       ("crosskd_rgbnir", "crosskd_rgbnir_seg"), ("kd_rgb", "kd_rgb_seg")):
        color, style, label = vop.ARMS[base]
        twin_color, twin_style, twin_label = vop.ARMS[twin]
        assert twin_color == color and twin_style != style and twin_label.startswith(twin)
    assert not any(name.endswith("_segdt") for name in vop.ARMS) and "est_rgb_seg" not in vop.ARMS


def test_the_detect_operating_point_entries_are_untouched() -> None:
    assert vop.ARMS["est_rgb"] == ("#2a78d6", "-", "est_rgb (RGB)")


def test_zone_analysis_reads_the_segment_twins_after_the_detect_arms() -> None:
    assert zones.ARMS[:5] == DETECT_ARMS and "master_rgb_seg" in zones.ARMS and "kd_rgb_seg" in zones.ARMS


def test_the_export_reads_each_arm_once() -> None:
    assert len(export.EXPORT_ARMS) == len(set(export.EXPORT_ARMS))
    assert {"master_rgb_seg", "crosskd_rgb_seg", "crosskd_rgbnir_better_seg"} <= set(export.EXPORT_ARMS)
    assert not any(name.endswith("_segdt") for name in export.EXPORT_ARMS)
