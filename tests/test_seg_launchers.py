"""Segment arms in the launchers: arms, fold labels, evaluation, aggregation, export."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from scripts import aggregate_kfold as agg
from scripts import evaluate_checkpoint as ec
from scripts import export_run_results as ex
from scripts import make_grouped_folds as mgf
from scripts import make_v4_folds as mvf
from scripts import run_small_arms as rs
from src.evaluation.v4_metrics import read_damage_labels
from src.models.student.student_model import StudentModel
from src.training.config import TrainingConfig
from src.training.seg_labels import read_label_boxes

POLYGON_FILE = (
    "0 0.375 0.3 0.625 0.3 0.625 0.7 0.375 0.7\n"
    "1 0.45 0.45 0.55 0.45 0.5 0.6\n"
    "1 0.0 0.0 0.1 0.0 0.1 0.1\n"  # a vertex at 0: the box reader must not drop it
    "1 0.5 0.5 0.1 0.2\n"
)


# --- label readers --------------------------------------------------------------------------------


def test_read_label_boxes_handles_polygons_and_boxes(tmp_path: Path) -> None:
    path = tmp_path / "a.txt"
    path.write_text(POLYGON_FILE)
    boxes, classes = read_label_boxes(path)
    assert classes.tolist() == [0, 1, 1, 1]
    assert boxes[1].tolist() == pytest.approx([0.5, 0.525, 0.1, 0.15])
    assert boxes[3].tolist() == pytest.approx([0.5, 0.5, 0.1, 0.2])
    with pytest.raises(FileNotFoundError):
        read_label_boxes(tmp_path / "missing.txt")


def test_count_damage_boxes_counts_polygons(tmp_path: Path) -> None:
    path = tmp_path / "a.txt"
    path.write_text(POLYGON_FILE)
    assert mgf.count_damage_boxes(path) == 3


def test_read_damage_labels_returns_the_polygon_boxes(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text(POLYGON_FILE)
    boxes = read_damage_labels(tmp_path)["a"]
    assert boxes.shape == (3, 4)
    assert boxes[0].tolist() == pytest.approx([0.45, 0.45, 0.55, 0.6])


def test_fold_moments_read_a_polygon_label(tmp_path: Path) -> None:
    nir = tmp_path / "n.png"
    cv2.imwrite(str(nir), np.full((100, 200), 80, np.uint8))
    label = tmp_path / "n.txt"
    label.write_text(POLYGON_FILE)
    stem, moments = mvf.image_moments(("n", nir, label))
    assert stem == "n" and len(moments) == 6 and moments[3] < moments[0]  # the ROI is a part of the frame


def test_write_folds_keeps_polygon_lines_byte_for_byte(tmp_path: Path) -> None:
    source = tmp_path / "src.txt"
    source.write_text(POLYGON_FILE)
    manifests = [{"train": ["s"], "val": [], "test": []}]
    mgf.write_folds(tmp_path / "out", manifests, {"s": source}, {"folds": {}})
    assert (tmp_path / "out" / "fold_0" / "labels" / "train" / "s.txt").read_text() == POLYGON_FILE


# --- run_small_arms -------------------------------------------------------------------------------


def _args(tmp_path: Path, *extra: str):
    (tmp_path / "folds_lot").mkdir(exist_ok=True)
    stats = {"nir_clahe": {"roi": {"mean": 0.31, "std": 0.15}}}
    meta = {"folds": {str(k): {"nir_stats": stats} for k in range(5)}}
    (tmp_path / "folds_lot" / "folds_meta.json").write_text(json.dumps(meta))
    args = rs.parse_args(["--dataset", str(tmp_path), "--weights", "w.pt", *extra])
    args.nir_as_rgb_dir = tmp_path / "links"
    return args


def _train(args, arm: str) -> list[str]:
    return [str(x) for x in rs.plan_arm(args, arm, 0, 42)[0][1]]


SEG_KEYS = ("task=segment",)


def test_detect_arms_never_mention_the_task(tmp_path) -> None:
    args = _args(tmp_path)
    for arm in rs.PLANNED_ARMS + ["kd_nir"]:
        for command in (_train(args, arm), *[[str(x) for x in s[1]] for s in rs.plan_arm(args, arm, 0, 42)]):
            assert not any(x.startswith(("task=", "mask_", "load_masks")) for x in command), arm


def test_the_seg_names_are_an_opt_in_that_the_default_run_does_not_list(tmp_path) -> None:
    assert not any(a.endswith("_seg") for a in rs.PLANNED_ARMS)
    assert {"master_rgb_seg", "master_rgbnir_seg", "master_nir_seg", "kd_rgb_seg", "kd_rgbnir_seg",
            "crosskd_rgb_seg", "crosskd_rgbnir_seg"} <= set(rs.ARMS)
    assert not any(a.endswith("_segdt") or a in ("est_rgb_seg", "est_rgbnir_seg", "est_nir_seg") for a in rs.ARMS)
    assert rs.parse_args(["--dataset", str(tmp_path)]).arms == rs.PLANNED_ARMS


@pytest.mark.parametrize("arm", ["master_rgb_seg", "master_rgbnir_seg", "master_nir_seg", "kd_rgb_seg", "crosskd_rgbnir_seg"])
def test_a_seg_arm_trains_and_evaluates_as_segment(tmp_path, arm) -> None:
    args = _args(tmp_path, "--mask-weight", "5", "--mask-ratio", "4")
    steps = rs.plan_arm(args, arm, 0, 42)
    train = [str(x) for x in steps[0][1]]
    for flag in ("task=segment", "mask_ratio=4", "mask_weight=5.0"):
        assert flag in train, flag
    evaluation = [str(x) for x in steps[1][1]]
    # The evaluation reads the task from the checkpoint, so it is not overridden.
    assert not any(x.startswith("task=") for x in evaluation) and "mask_ratio=4" in evaluation


def test_a_seg_arm_keeps_the_modality_of_its_detect_twin(tmp_path) -> None:
    args = _args(tmp_path)
    assert f"rgb_dir={tmp_path / 'links'}" in _train(args, "master_nir_seg")
    assert "student_in_channels=4" in _train(args, "master_rgbnir_seg")
    assert "student_in_channels=4" not in _train(args, "master_rgb_seg")


@pytest.mark.parametrize("arm, master", [
    ("kd_rgb_seg", "master_rgb_seg"), ("kd_rgbnir_seg", "master_rgbnir_seg"),
    ("crosskd_rgb_seg", "master_rgb_seg"), ("crosskd_rgbnir_seg", "master_rgbnir_seg"),
    ("crosskd_rgbnir_high_seg", "master_rgbnir_seg"), ("crosskd_rgbnir_low_seg", "master_rgbnir_seg"),
    ("crosskd_rgbnir_better_seg", "master_rgbnir_seg"),
])
def test_kd_seg_arms_distill_from_the_master_of_their_fold_and_seed(tmp_path, arm, master) -> None:
    args = _args(tmp_path)
    command = _train(args, arm)
    assert "src.training.kd_train" in command and "task=segment" in command
    assert rs.ARMS[arm].teacher == master
    assert f"teacher_checkpoint={args.checkpoints_root / master / 'fold0' / 'seed42' / 'best_model.pt'}" in command


def test_the_crosskd_seg_arms_keep_the_overrides_of_their_detect_counterparts(tmp_path) -> None:
    args = _args(tmp_path)
    for detect in ("crosskd_rgb", "crosskd_rgbnir", "crosskd_rgbnir_high", "crosskd_rgbnir_low", "crosskd_rgbnir_better"):
        assert rs.ARMS[f"{detect}_seg"].kd_overrides == rs.ARMS[detect].kd_overrides
        assert "task=segment" not in _train(args, detect)
    assert "kd_gate=nir_high" in _train(args, "crosskd_rgbnir_high_seg")


def test_seg_arms_schedule_their_teacher_first(tmp_path) -> None:
    args = _args(tmp_path, "--arms", "crosskd_rgb_seg", "master_rgb_seg", "--folds", "0", "--dry-run")
    order = sorted(args.arms, key=lambda name: rs.ARMS[name].teacher is not None)
    assert order == ["master_rgb_seg", "crosskd_rgb_seg"]


def test_the_presets_expand_in_dependency_order(tmp_path) -> None:
    core = rs.parse_args(["--dataset", str(tmp_path), "--arms-preset", "seg"]).arms
    assert core == ["master_rgb_seg", "master_rgbnir_seg", "master_nir_seg", "crosskd_rgb_seg", "crosskd_rgbnir_seg"]
    full = rs.parse_args(["--dataset", str(tmp_path), "--arms-preset", "seg-full"]).arms
    assert full == core + ["crosskd_rgbnir_high_seg", "crosskd_rgbnir_low_seg", "crosskd_rgbnir_better_seg"]
    for preset in (core, full):
        seen = set()
        for arm in preset:
            teacher = rs.ARMS[arm].teacher
            assert teacher is None or teacher in seen, arm
            seen.add(arm)


def test_a_preset_and_explicit_arms_are_alternatives(tmp_path) -> None:
    with pytest.raises(SystemExit):
        rs.parse_args(["--dataset", str(tmp_path), "--arms-preset", "seg", "--arms", "est_rgb"])


def test_the_preset_dry_run_trains_every_line_as_segment_with_crosskd_teachers_from_the_masters(tmp_path, capsys) -> None:
    _args(tmp_path)  # writes the folds meta
    code = rs.main([
        "--dataset", str(tmp_path), "--weights", "w.pt", "--arms-preset", "seg-full", "--folds", "1", "--seeds", "7",
        "--dry-run", "--nir-as-rgb-dir", str(tmp_path / "links"),
    ])
    assert code == 0
    lines = [line for line in capsys.readouterr().out.splitlines() if line.startswith("[")]
    trains = [line for line in lines if "/train: " in line]
    assert len(trains) == 8
    for line in trains:
        assert "task=segment" in line
    for line in (x for x in trains if "] crosskd" in x):
        arm = line.split("] ")[1].split("/")[0]
        master = "master_rgb_seg" if arm == "crosskd_rgb_seg" else "master_rgbnir_seg"
        assert "kd_mode=crosshead" in line
        assert f"teacher_checkpoint={tmp_path.parent}" not in line  # roots are the defaults, relative
        assert f"/{master}/fold1/seed7/best_model.pt" in line, line
    assert all("_seg/" in line for line in lines)  # every step of the plan is a segment arm


def test_dry_run_prints_the_segment_command_without_running(tmp_path, capsys) -> None:
    _args(tmp_path)  # writes the folds meta
    code = rs.main([
        "--dataset", str(tmp_path), "--weights", "w.pt", "--arms", "master_rgb_seg",
        "--folds", "0", "--val-only", "--dry-run", "--nir-as-rgb-dir", str(tmp_path / "links"),
    ])
    out = capsys.readouterr().out
    assert code == 0 and "[run] master_rgb_seg/fold0/seed42/train" in out and "task=segment" in out
    assert "src.training.train" in out


# --- evaluate_checkpoint --------------------------------------------------------------------------


def _checkpoint(tmp_path: Path, task: str) -> Path:
    path = tmp_path / f"{task}.pt"
    model = StudentModel(num_classes=2, variant="nano", task=task)
    torch.save({"model_state_dict": model.state_dict(), "config": {"task": task, "roi_crop": False,
                                                                    "roi_crop_margin": 0.1}}, path)
    return path


def _config(**kw) -> TrainingConfig:
    return TrainingConfig(model_type="student", student_variant="nano", **kw)


def test_the_task_comes_from_the_checkpoint(tmp_path) -> None:
    config = _config()
    model = ec._load_model("student", str(_checkpoint(tmp_path, "segment")), config, torch.device("cpu"))
    assert model.task == "segment" and config.task == "segment"


def test_a_detect_checkpoint_stays_detect_even_for_a_segment_config(tmp_path) -> None:
    config = _config(task="segment")
    model = ec._load_model("student", str(_checkpoint(tmp_path, "detect")), config, torch.device("cpu"))
    assert model.task == "detect" and config.task == "detect"


def test_an_explicit_override_wins_over_the_checkpoint(tmp_path) -> None:
    config = _config(task="detect")
    with pytest.raises(RuntimeError):  # a segment checkpoint cannot load strictly as detect
        ec._load_model("student", str(_checkpoint(tmp_path, "segment")), config, torch.device("cpu"),
                       pin_task=True)


def test_a_checkpoint_without_task_is_detect(tmp_path) -> None:
    path = tmp_path / "old.pt"
    torch.save({"model_state_dict": StudentModel(num_classes=2, variant="nano").state_dict(),
                "config": {"roi_crop": False}}, path)
    config = _config()
    assert ec._load_model("student", str(path), config, torch.device("cpu")).task == "detect"


# --- aggregate_kfold and export_run_results -------------------------------------------------------


def _metrics(ap50: float, mask: float | None = None) -> dict:
    out = {
        "map50": 0.5, "map_50_95": 0.3,
        "per_class_ap_50": {"0": 1.0, "1": ap50}, "per_class_ap_50_95": {"0": 0.6, "1": 0.1},
        "per_class_precision": {"0": 1.0, "1": 0.2}, "per_class_recall": {"0": 1.0, "1": 0.5},
        "per_class_f1": {"0": 1.0, "1": 0.3},
        "per_class_counts": {"0": {"tp": 25, "fp": 0, "fn": 0}, "1": {"tp": 10, "fp": 20, "fn": 5}},
    }
    if mask is not None:
        out |= {"mask_map50": mask / 2, "mask_map_50_95": mask / 4,
                "mask_per_class_ap_50": {"0": 0.9, "1": mask}, "mask_per_class_ap_50_95": {"0": 0.5, "1": mask / 2}}
    return out


def _write(root: Path, arm: str, fold: int, split: str, metrics: dict) -> None:
    path = root / arm / f"fold{fold}_seed42" / split / "metrics.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics))


def _tree(root: Path, with_masks: bool) -> None:
    for arm, base in (("a", 0.2), ("b", 0.3)):
        for fold in (0, 1):
            for split in ("val", "test"):
                mask = base * 2 if with_masks else None
                _write(root, arm, fold, split, _metrics(base + fold / 10, mask))


def test_damage_summary_gains_mask_fields_only_when_present() -> None:
    assert "mask_ap50" not in agg.damage_summary(_metrics(0.3))
    summary = agg.damage_summary(_metrics(0.3, mask=0.4))
    assert summary["mask_ap50"] == pytest.approx(0.4) and summary["mask_ap50_95"] == pytest.approx(0.2)


def test_aggregate_adds_a_mask_block_for_segment_runs_and_not_for_detect(tmp_path) -> None:
    detect = tmp_path / "detect"
    _tree(detect, with_masks=False)
    kw = dict(arms=["a", "b"], folds=[0, 1], seeds=[42], baseline="a", treatment="b")
    plain = agg.aggregate(detect, **kw)
    assert "mask" not in plain and "Mask" not in agg.render_markdown(plain)

    seg = tmp_path / "seg"
    _tree(seg, with_masks=True)
    results = agg.aggregate(seg, **kw)
    block = results["mask"]
    assert block["metrics"] == ["mask_ap50", "mask_ap50_95"]
    assert block["per_fold"]["a"]["val"]["0"]["mask_ap50"] == pytest.approx(0.4)
    assert block["summary"]["b"]["test"]["mask_ap50"]["n"] == 2
    # Box statistics are exactly those of the detect-style aggregation of the same boxes.
    assert results["summary"] == agg.aggregate(seg, **kw)["summary"]
    assert "Mask AP" in agg.render_markdown(results)


def test_export_flattens_mask_scalars_and_lists_the_seg_arms() -> None:
    assert {"master_rgb_seg", "kd_rgbnir_seg", "crosskd_rgbnir_seg"} <= set(ex.EXPORT_ARMS)
    assert set(ex.ARMS) <= set(ex.EXPORT_ARMS)
    row: dict = {}
    ex.flatten("", {k: v for k, v in _metrics(0.3, mask=0.4).items() if k.startswith(ex.METRIC_PREFIXES)}, row)
    assert row["mask_map50"] == pytest.approx(0.2) and row["mask_per_class_ap_50.1"] == pytest.approx(0.4)
    detect_row: dict = {}
    ex.flatten("", {k: v for k, v in _metrics(0.3).items() if k.startswith(ex.METRIC_PREFIXES)}, detect_row)
    assert not any(k.startswith("mask") for k in detect_row)
