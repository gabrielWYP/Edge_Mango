"""The `thesis` plan: the segment preset plus the detect arm trained on polygon-derived boxes."""

from __future__ import annotations

import json
from pathlib import Path

from scripts import export_run_results as export
from scripts import run_small_arms as rs
from scripts import val_operating_points as vop
from scripts import zone_analysis as zones
from src.evaluation import arms

DET = "master_rgb_det"


def _args(tmp_path: Path, *extra: str):
    (tmp_path / "folds_lot").mkdir(exist_ok=True)
    stats = {"nir_clahe": {"roi": {"mean": 0.31, "std": 0.15}}}
    meta = {"folds": {str(k): {"nir_stats": stats} for k in range(5)}}
    (tmp_path / "folds_lot" / "folds_meta.json").write_text(json.dumps(meta))
    args = rs.parse_args(["--dataset", str(tmp_path), "--weights", "w.pt", *extra])
    args.nir_as_rgb_dir = tmp_path / "links"
    return args


def test_the_thesis_preset_is_the_seg_preset_plus_the_detect_arm_with_masters_first() -> None:
    thesis = arms.SEGMENT_PRESETS["thesis"]
    assert set(thesis) == set(arms.SEGMENT_PRESETS["seg"]) | {DET}
    assert len(thesis) == 6
    seen: set[str] = set()
    for arm in thesis:
        teacher = rs.ARMS[arm].teacher
        assert teacher is None or teacher in seen, arm
        seen.add(arm)


def test_the_detect_arm_is_an_rgb_detect_student_without_teacher() -> None:
    arm = rs.ARMS[DET]
    assert (arm.task, arm.teacher, arm.in_channels, arm.reads_nir) == ("detect", None, 3, False)
    assert rs.arm_input(DET) == "rgb"


def test_the_detect_arm_commands_equal_est_rgb_apart_from_the_arm_name(tmp_path) -> None:
    args = _args(tmp_path, "--arms-preset", "thesis")
    base = rs.plan_arm(args, "est_rgb", 3, 1337)
    det = rs.plan_arm(args, DET, 3, 1337)
    assert len(base) == len(det) == 5
    for (_, b_cmd, _), (_, d_cmd, _) in zip(base, det, strict=True):
        renamed = [str(x).replace("/est_rgb/", f"/{DET}/") for x in b_cmd]
        assert renamed == [str(x) for x in d_cmd]
    assert not any("task=segment" in str(x) for _, cmd, _ in det for x in cmd)


def test_the_thesis_dry_run_covers_six_arms_over_folds_and_seeds(tmp_path, capsys) -> None:
    _args(tmp_path)
    code = rs.main([
        "--dataset", str(tmp_path), "--arms-preset", "thesis", "--seeds", "42", "1337", "2024",
        "--folds", "0", "1", "2", "3", "4", "--dry-run", "--nir-as-rgb-dir", str(tmp_path / "links"),
    ])
    assert code == 0
    lines = [x for x in capsys.readouterr().out.splitlines() if x.startswith("[")]
    trains = [x for x in lines if "/train: " in x]
    assert len(trains) == 6 * 5 * 3 and len(lines) == 90 * 5


def test_the_mask_supervision_comparison_is_the_segment_master_against_the_detect_arm() -> None:
    assert export.DIFFERENCES["mask_supervision"] == ("master_rgb_seg", DET)


def test_mask_supervision_is_computed_on_box_metrics_only() -> None:
    ap = {}
    for metric in ("latent_ap50", "latent_ap30"):
        ap[(0, 42, "val", "best", metric, "master_rgb_seg")] = 0.5
        ap[(0, 42, "val", "best", metric, DET)] = 0.4
    ap[(0, 42, "val", "best", "latent_mask_ap50", "master_rgb_seg")] = 0.3  # the detect arm has no mask metric
    rows = [r for r in export.compute_differences(ap) if r["comparison"] == "mask_supervision"]
    assert sorted(r["metric"] for r in rows) == ["latent_ap30", "latent_ap50"]
    assert all(abs(r["delta"] - 0.1) < 1e-12 for r in rows)


def test_the_detect_arm_is_in_the_analysis_tables_with_the_est_rgb_colour() -> None:
    assert DET in export.EXPORT_ARMS and DET in zones.ARMS
    assert f"{DET}_seg" not in vop.ARMS and f"{DET}_seg" not in export.EXPORT_ARMS
    assert vop.ARMS[DET][0] == vop.ARMS["est_rgb"][0] and vop.ARMS[DET][1] != vop.ARMS["est_rgb"][1]
    assert vop.ARMS[DET][2].startswith(DET)
    assert list(vop.ARMS)[:5] == ["est_rgb", "est_rgbnir", "kd_rgb", "kd_rgbnir", "est_nir"]


def _image_sizes(args) -> set[str]:
    cmds = [cmd for arm in args.arms for _, cmd, _ in rs.plan_arm(args, arm, 0, 42)]
    return {str(x) for cmd in cmds for x in cmd if str(x).startswith("image_size=")}


def test_the_thesis_preset_trains_and_evaluates_every_arm_at_640(tmp_path) -> None:
    args = _args(tmp_path, "--arms-preset", "thesis")
    assert rs.THESIS_IMAGE_SIZE == 640
    assert _image_sizes(args) == {"image_size=640"}
    steps = [(key, cmd) for arm in args.arms for key, cmd, _ in rs.plan_arm(args, arm, 0, 42)]
    model_steps = [cmd for key, cmd in steps if "/metrics_" not in key]  # v4_metrics only reads predictions
    assert len(model_steps) == 6 * 3
    assert all("image_size=640" in [str(x) for x in cmd] for cmd in model_steps)


def test_an_explicit_image_size_wins_over_the_preset(tmp_path) -> None:
    assert _image_sizes(_args(tmp_path, "--arms-preset", "thesis", "--image-size", "800")) == {"image_size=800"}


def test_without_preset_or_flag_the_config_size_is_kept(tmp_path) -> None:
    assert _image_sizes(_args(tmp_path, "--arms", "est_rgb")) == set()
    assert _image_sizes(_args(tmp_path, "--arms", "master_rgb_seg", "--image-size", "800")) == {"image_size=800"}


def test_the_image_size_must_be_a_multiple_of_32_and_of_the_mask_ratio(tmp_path) -> None:
    import pytest

    with pytest.raises(SystemExit):
        _args(tmp_path, "--image-size", "810")
    with pytest.raises(SystemExit):
        _args(tmp_path, "--image-size", "800", "--mask-ratio", "3")


# Exploratory arm: an RGB student distilled with CrossKD from the NIR-only segment master (run after the thesis set).
EXPLORE = "crosskd_nir_seg"


def test_the_explore_preset_is_the_single_nir_crosskd_arm_and_not_part_of_the_thesis_preset() -> None:
    assert arms.SEGMENT_PRESETS["thesis_explore"] == (EXPLORE,)
    assert EXPLORE not in arms.SEGMENT_PRESETS["thesis"]
    assert len(arms.SEGMENT_PRESETS["thesis"]) == 6


def test_the_explore_arm_is_an_rgb_segment_student_with_the_nir_master_as_teacher() -> None:
    arm = rs.ARMS[EXPLORE]
    assert (arm.task, arm.in_channels, arm.reads_nir) == ("segment", 3, False)
    assert arm.teacher == "master_nir_seg" and rs.arm_input(arm.teacher) == "nir"
    assert arm.kd_overrides == rs.ARMS["crosskd_rgb_seg"].kd_overrides == ("kd_mode=crosshead",)


def test_the_explore_train_command_equals_crosskd_rgb_seg_apart_from_name_and_teacher(tmp_path) -> None:
    args = _args(tmp_path, "--arms-preset", "thesis_explore")
    base = rs.plan_arm(args, "crosskd_rgb_seg", 2, 42)
    explore = rs.plan_arm(args, EXPLORE, 2, 42)
    assert len(base) == len(explore) == 5
    for (_, b_cmd, _), (_, e_cmd, _) in zip(base, explore, strict=True):
        renamed = [str(x).replace("/crosskd_rgb_seg/", f"/{EXPLORE}/").replace("master_rgb_seg", "master_nir_seg")
                   .replace("teacher_input=rgb", "teacher_input=nir") for x in b_cmd]
        assert renamed == [str(x) for x in e_cmd]
    train = [str(x) for x in explore[0][1]]
    assert "teacher_input=nir" in train
    assert any(x.startswith("teacher_checkpoint=") and "/master_nir_seg/fold2/seed42/" in x for x in train)
    assert "student_in_channels=4" not in train


def test_the_explore_dry_run_plans_the_arm_only_and_defaults_to_640(tmp_path, capsys) -> None:
    _args(tmp_path)
    code = rs.main([
        "--dataset", str(tmp_path), "--arms-preset", "thesis_explore", "--folds", "0", "1", "2", "3", "4",
        "--seeds", "42", "--dry-run", "--nir-as-rgb-dir", str(tmp_path / "links"),
    ])
    assert code == 0
    lines = [x for x in capsys.readouterr().out.splitlines() if x.startswith("[")]
    assert len(lines) == 5 * 5 and all(f"{EXPLORE}/fold" in x.split(": ")[0] for x in lines)
    trains = [x for x in lines if "/train: " in x]
    assert len(trains) == 5 and all("image_size=640" in x and "teacher_input=nir" in x for x in trains)


def test_the_explore_preset_defaults_to_the_thesis_image_size(tmp_path) -> None:
    assert _image_sizes(_args(tmp_path, "--arms-preset", "thesis_explore")) == {"image_size=640"}
    assert _image_sizes(_args(tmp_path, "--arms-preset", "thesis_explore", "--image-size", "800")) == {"image_size=800"}


def test_the_explore_arm_is_in_the_analysis_tables_with_its_own_colour() -> None:
    assert EXPLORE in export.EXPORT_ARMS and EXPLORE in zones.ARMS and EXPLORE in vop.ARMS
    colour, style, label = vop.ARMS[EXPLORE]
    assert label.startswith(EXPLORE)
    assert colour not in {v[0] for k, v in vop.ARMS.items() if k != EXPLORE}
    assert len(export.EXPORT_ARMS) == len(set(export.EXPORT_ARMS)) and len(zones.ARMS) == len(set(zones.ARMS))
    assert "crosskd_nir" not in vop.ARMS  # no detect twin: the arm is segment only


def test_the_explore_comparisons_are_registered_and_use_every_metric() -> None:
    assert export.DIFFERENCES["crosskd_nir"] == (EXPLORE, "master_rgb_seg")
    assert export.DIFFERENCES["crosskd_nir_vs_rgbnir"] == (EXPLORE, "crosskd_rgbnir_seg")
    ap = {}
    for metric in ("latent_ap50", "latent_mask_ap50"):
        for arm, value in ((EXPLORE, 0.5), ("master_rgb_seg", 0.4), ("crosskd_rgbnir_seg", 0.45)):
            ap[(0, 42, "val", "best", metric, arm)] = value
    rows = [r for r in export.compute_differences(ap) if r["comparison"].startswith("crosskd_nir")]
    assert sorted((r["comparison"], r["metric"]) for r in rows) == [
        ("crosskd_nir", "latent_ap50"), ("crosskd_nir", "latent_mask_ap50"),
        ("crosskd_nir_vs_rgbnir", "latent_ap50"), ("crosskd_nir_vs_rgbnir", "latent_mask_ap50"),
    ]
