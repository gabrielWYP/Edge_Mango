"""The exploratory arm `master_twostem_rgbnir_seg`: master_rgbnir_seg with two fused backbones."""

from __future__ import annotations

import json
from pathlib import Path

from scripts import export_run_results as export
from scripts import run_small_arms as rs
from scripts import val_operating_points as vop
from scripts import zone_analysis as zones
from src.evaluation import arms

TWOSTEM = "master_twostem_rgbnir_seg"
BASE = "master_rgbnir_seg"


def _args(tmp_path: Path, *extra: str):
    (tmp_path / "folds_lot").mkdir(exist_ok=True)
    stats = {"nir_clahe": {"roi": {"mean": 0.31, "std": 0.15}}}
    meta = {"folds": {str(k): {"nir_stats": stats} for k in range(5)}}
    (tmp_path / "folds_lot" / "folds_meta.json").write_text(json.dumps(meta))
    args = rs.parse_args(["--dataset", str(tmp_path), "--weights", "w.pt", *extra])
    args.nir_as_rgb_dir = tmp_path / "links"
    return args


def test_the_arm_and_its_segment_name_are_registered() -> None:
    assert arms.segment_arm_name("est_twostem_rgbnir") == TWOSTEM
    assert rs.ARMS["est_twostem_rgbnir"].fusion == "mid"
    arm = rs.ARMS[TWOSTEM]
    assert (arm.task, arm.in_channels, arm.reads_nir, arm.teacher, arm.fusion) == ("segment", 4, False, None, "mid")
    assert rs.arm_input(TWOSTEM) == "rgbnir"
    assert all(rs.ARMS[name].fusion == "early" for name in rs.ARMS if "twostem" not in name)


def test_the_explore_twostem_preset_is_that_arm_alone_and_in_no_other_preset() -> None:
    assert arms.SEGMENT_PRESETS["thesis_explore_twostem"] == (TWOSTEM,)
    assert all(TWOSTEM not in names for key, names in arms.SEGMENT_PRESETS.items() if key != "thesis_explore_twostem")


def test_the_train_command_equals_master_rgbnir_seg_plus_the_fusion_override(tmp_path) -> None:
    args = _args(tmp_path, "--arms-preset", "thesis_explore_twostem")
    base = rs.plan_arm(args, BASE, 2, 42)
    two = rs.plan_arm(args, TWOSTEM, 2, 42)
    assert len(base) == len(two) == 5
    for i, ((b_key, b_cmd, _), (t_key, t_cmd, _)) in enumerate(zip(base, two, strict=True)):
        renamed = [str(x).replace(f"/{BASE}/", f"/{TWOSTEM}/") for x in b_cmd]
        assert [x for x in map(str, t_cmd) if x != "student_fusion=mid"] == renamed
        assert t_key == b_key.replace(BASE, TWOSTEM)
        # Training and the evaluation of the checkpoint carry it; the v4 metrics step reads predictions only.
        assert ("student_fusion=mid" in list(map(str, t_cmd))) == (b_key.endswith(("/train", "/eval_val", "/eval_test")))
    assert not any("student_fusion" in str(x) for _, cmd, _ in base for x in cmd)


def test_the_preset_defaults_to_640_and_is_planned_under_the_same_roots(tmp_path, capsys) -> None:
    _args(tmp_path)
    code = rs.main([
        "--dataset", str(tmp_path), "--arms-preset", "thesis_explore_twostem", "--folds", "0", "1", "2", "3", "4",
        "--seeds", "42", "--dry-run", "--nir-as-rgb-dir", str(tmp_path / "links"),
        "--checkpoints-root", "RUNS", "--reports-root", "RUNS/reports",
    ])
    assert code == 0
    lines = [x for x in capsys.readouterr().out.splitlines() if x.startswith("[")]
    assert len(lines) == 25 and all(f"{TWOSTEM}/fold" in x.split(": ")[0] for x in lines)
    trains = [x for x in lines if "/train: " in x]
    assert len(trains) == 5
    assert all("image_size=640" in x and "student_fusion=mid" in x and "student_in_channels=4" in x for x in trains)
    assert all(f"output_dir=RUNS/{TWOSTEM}/fold" in x for x in trains)


def test_the_evaluation_command_carries_the_same_fusion_so_it_rebuilds_the_model(tmp_path) -> None:
    steps = rs.plan_arm(_args(tmp_path), TWOSTEM, 0, 42)
    assert "student_fusion=mid" in [str(x) for x in steps[1][1]]


def test_the_analysis_scripts_find_the_arm_by_name_and_compare_it_to_both_references() -> None:
    assert TWOSTEM in export.EXPORT_ARMS and TWOSTEM in zones.ARMS and TWOSTEM in vop.ARMS
    assert vop.ARMS[TWOSTEM][2].startswith(TWOSTEM)
    assert vop.ARMS[TWOSTEM][0] not in {v[0] for k, v in vop.ARMS.items() if k != TWOSTEM}
    assert export.DIFFERENCES["twostem_vs_rgbnir"] == (TWOSTEM, BASE)
    assert export.DIFFERENCES["twostem_vs_nir"] == (TWOSTEM, "master_nir_seg")
    assert len(export.EXPORT_ARMS) == len(set(export.EXPORT_ARMS))


KD = "crosskd_twostem_seg"
KD_BASE = "crosskd_rgbnir_seg"


def test_the_crosskd_twostem_arm_is_crosskd_rgbnir_with_the_two_stem_teacher() -> None:
    assert arms.segment_arm_name("crosskd_twostem") == KD
    arm, base = rs.ARMS[KD], rs.ARMS[KD_BASE]
    assert arm.teacher == TWOSTEM and base.teacher == BASE
    assert arm.kd_overrides == base.kd_overrides == ("kd_mode=crosshead",)
    assert (arm.task, arm.in_channels, arm.fusion, arm.reads_nir) == (base.task, base.in_channels, base.fusion, base.reads_nir)
    assert rs.arm_input(KD) == rs.arm_input(KD_BASE)
    assert arms.SEGMENT_PRESETS["thesis_explore_twostem_kd"] == (KD,)
    assert all(KD not in names for key, names in arms.SEGMENT_PRESETS.items() if key != "thesis_explore_twostem_kd")


def test_the_kd_command_equals_crosskd_rgbnir_seg_with_the_two_stem_teacher_checkpoint(tmp_path) -> None:
    args = _args(tmp_path, "--arms-preset", "thesis_explore_twostem_kd", "--checkpoints-root", "RUNS")
    base = rs.plan_arm(args, KD_BASE, 2, 42)
    two = rs.plan_arm(args, KD, 2, 42)
    assert len(base) == len(two) == 5
    for (b_key, b_cmd, _), (t_key, t_cmd, _) in zip(base, two, strict=True):
        assert t_key == b_key.replace(KD_BASE, KD)
        renamed = [str(x).replace(f"/{KD_BASE}/", f"/{KD}/").replace(f"/{BASE}/", f"/{TWOSTEM}/") for x in b_cmd]
        assert list(map(str, t_cmd)) == renamed
    train = next(list(map(str, c)) for k, c, _ in two if k.endswith("/train"))
    teacher = next(x for x in train if x.startswith("teacher_checkpoint="))
    assert f"/{TWOSTEM}/fold2" in teacher and "seed42" in teacher
    assert "kd_mode=crosshead" in train and "student_fusion=mid" not in train


def test_the_kd_preset_dry_run_trains_no_teacher_and_defaults_to_640(tmp_path, capsys) -> None:
    _args(tmp_path)
    code = rs.main([
        "--dataset", str(tmp_path), "--arms-preset", "thesis_explore_twostem_kd", "--folds", "0", "1", "2", "3", "4",
        "--seeds", "42", "--dry-run", "--nir-as-rgb-dir", str(tmp_path / "links"),
        "--checkpoints-root", "RUNS", "--reports-root", "RUNS/reports",
    ])
    assert code == 0
    lines = [x for x in capsys.readouterr().out.splitlines() if x.startswith("[")]
    assert lines and all(f"{KD}/fold" in x.split(": ")[0] for x in lines)
    trains = [x for x in lines if "/train: " in x]
    assert len(trains) == 5
    assert all("image_size=640" in x and "kd_mode=crosshead" in x and f"RUNS/{TWOSTEM}/fold" in x for x in trains)
    assert not any(f"{TWOSTEM}/fold" in x.split(": ")[0] for x in lines)


def test_the_kd_arm_is_registered_in_the_analysis_scripts() -> None:
    assert KD in export.EXPORT_ARMS and KD in zones.ARMS and KD in vop.ARMS
    assert vop.ARMS[KD][2].startswith(KD)
    assert vop.ARMS[KD][0] not in {v[0] for k, v in vop.ARMS.items() if k != KD}
    assert export.DIFFERENCES["crosskd_twostem_vs_rgbnir"] == (KD, KD_BASE)
    assert export.DIFFERENCES["crosskd_twostem_vs_rgb"] == (KD, "crosskd_rgb_seg")
