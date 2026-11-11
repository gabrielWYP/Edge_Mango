"""The NIR-dependence probe: labels, universe, cell sampling and the report. Never run on real checkpoints."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
import torch

from scripts import kd_nir_probe as probe
from src.training.kd_train import apply_overrides
from src.training.kd_config import KDConfig

_FRUIT_BOX = [0.5, 0.5, 0.5, 0.5]  # cells [2, 6) of an 8x8 map
_DAMAGE_BOX = [0.4375, 0.4375, 0.125, 0.125]  # cell (3, 3)


def _targets():
    return [torch.tensor([_FRUIT_BOX, _DAMAGE_BOX])], [torch.tensor([0, 1])]


def test_apply_overrides_types_values_after_the_field_and_skips_unknown_keys() -> None:
    config = KDConfig()
    apply_overrides(config, ["kd_weight=0.5", "epochs=3", "kd_mode=crosshead", "nonsense=1"])
    assert config.kd_weight == 0.5 and config.epochs == 3 and config.kd_mode == "crosshead"
    assert not hasattr(config, "nonsense")


def test_probe_labels_mark_cells_above_the_median_of_the_fruit_or_damage_universe() -> None:
    bboxes, labels = _targets()
    delta = [torch.rand(1, 8, 8, generator=torch.Generator().manual_seed(3))]
    (label, universe), = probe.probe_labels(delta, bboxes, labels)
    assert universe.sum() == 16 and (label & ~universe).sum() == 0
    median = delta[0][0][universe[0]].median()
    assert torch.equal(label[0], universe[0] & (delta[0][0] > median))
    assert 0 < label.sum() < universe.sum()


def test_an_image_without_universe_has_no_labelled_cell() -> None:
    delta = [torch.rand(1, 8, 8)]
    (label, universe), = probe.probe_labels(delta, [torch.zeros(0, 4)], [torch.zeros(0, dtype=torch.long)])
    assert not label.any() and not universe.any()


def test_collect_cells_keeps_only_the_universe_and_caps_each_image() -> None:
    feat = torch.arange(2 * 3 * 8 * 8, dtype=torch.float32).reshape(2, 3, 8, 8)
    universe = torch.zeros(2, 8, 8, dtype=torch.bool)
    universe[0, :4, :4] = True  # 16 cells
    universe[1, 0, :2] = True  # 2 cells
    label = universe.clone()
    label[0, 0, 0] = False
    x, y = probe.collect_cells(feat, label, universe, cap=5, generator=torch.Generator().manual_seed(0))
    assert x.shape == (5 + 2, 3) and y.shape == (7,) and x.dtype == np.float32
    assert y.sum() <= 7 and set(np.unique(y)) <= {0.0, 1.0}


def test_collect_cells_is_deterministic_for_a_seed() -> None:
    feat = torch.randn(1, 4, 8, 8)
    universe = torch.ones(1, 8, 8, dtype=torch.bool)
    label = torch.rand(1, 8, 8) > 0.5
    a = probe.collect_cells(feat, label, universe, 10, torch.Generator().manual_seed(1))
    b = probe.collect_cells(feat, label, universe, 10, torch.Generator().manual_seed(1))
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_score_returns_nan_when_a_split_has_a_single_class() -> None:
    x = np.random.RandomState(0).randn(20, 3).astype(np.float32)
    assert math.isnan(probe.score(x, np.zeros(20), x, np.zeros(20), seed=0)[0])


def test_score_separates_a_learnable_label_and_the_shuffled_control_does_not() -> None:
    rng = np.random.RandomState(0)
    y = (rng.rand(400) > 0.5).astype(np.float64)
    x = rng.randn(400, 4).astype(np.float32)
    x[:, 0] += 3 * y
    auroc, shuffled = probe.score(x[:200], y[:200], x[200:], y[200:], seed=0)
    assert auroc > 0.9 and 0.3 < shuffled < 0.7


def test_score_is_reproducible() -> None:
    rng = np.random.RandomState(1)
    y = (rng.rand(100) > 0.5).astype(np.float64)
    x = rng.randn(100, 3).astype(np.float32)
    assert probe.score(x, y, x, y, seed=5) == probe.score(x, y, x, y, seed=5)


def _result():
    return {
        "0": {"P3": {"auroc": 0.7, "shuffled_auroc": 0.5}, "P4": {"auroc": 0.6, "shuffled_auroc": 0.51},
              "P5": {"auroc": None, "shuffled_auroc": None}},
        "1": {"P3": {"auroc": 0.8, "shuffled_auroc": 0.49}, "P4": {"auroc": 0.7, "shuffled_auroc": 0.5},
              "P5": {"auroc": 0.55, "shuffled_auroc": 0.5}},
    }


def test_mean_over_folds_skips_undefined_levels() -> None:
    means = probe.mean_over_folds(_result())
    assert means["P3"]["auroc"] == pytest.approx(0.75)
    assert means["P5"]["auroc"] == pytest.approx(0.55)
    assert means["P4"]["shuffled_auroc"] == pytest.approx(0.505)


def test_the_table_has_a_row_per_fold_and_a_mean_row() -> None:
    text = probe.render(_result(), {"seed": 42, "variant": "wide"})
    rows = [line for line in text.splitlines() if line.startswith("| ")]
    assert any(r.startswith("| 0 |") for r in rows) and any(r.startswith("| 1 |") for r in rows)
    assert any(r.startswith("| mean |") for r in rows)
    assert "0.700 / 0.500" in text and "n/a" in text


def test_write_outputs_creates_probe_json_and_readme(tmp_path) -> None:
    probe.write_outputs(tmp_path / "out", _result(), {"seed": 42, "variant": "wide"})
    data = json.loads((tmp_path / "out" / "probe.json").read_text())
    assert set(data) == {"meta", "folds", "mean"} and data["mean"]["P3"]["auroc"] == pytest.approx(0.75)
    assert (tmp_path / "out" / "README.md").read_text().startswith("# ")


def _segment_checkpoint(path, in_channels: int) -> None:
    from src.models.student.student_model import StudentModel

    torch.manual_seed(0)
    model = StudentModel(num_classes=2, variant="nano", in_channels=in_channels, task="segment")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(),
                "config": {"task": "segment", "student_variant": "nano", "student_in_channels": in_channels}}, path)


def test_the_probe_reads_segment_teachers_with_the_task_from_the_checkpoint(tmp_path) -> None:
    for arm, channels in (("master_rgbnir_seg", 4), ("master_rgb_seg", 3)):
        _segment_checkpoint(tmp_path / "ckpt" / arm / "fold0" / "seed42" / "best_model.pt", channels)
    args = probe.parse_args(["--dataset", str(tmp_path), "--checkpoints-root", str(tmp_path / "ckpt"), "--segment"])
    teacher, control = probe.load_models(KDConfig(num_classes=2), args, 0, torch.device("cpu"))
    assert teacher.task == control.task == "segment" and teacher.in_channels == 4 and control.in_channels == 3
    result = probe.probe_fold(teacher, control, _fake_batches(), _fake_batches(2), seed=0, cap=64)
    assert list(result) == ["P3", "P4", "P5"]


def test_the_probe_arms_are_the_detect_ones_unless_segment_is_asked() -> None:
    args = probe.parse_args(["--dataset", "d", "--checkpoints-root", "c"])
    assert probe.arms_of(args) == ("est_rgbnir", "est_rgb")
    segment = probe.parse_args(["--dataset", "d", "--checkpoints-root", "c", "--segment"])
    assert probe.arms_of(segment) == ("master_rgbnir_seg", "master_rgb_seg")


def test_the_parser_defaults_follow_the_brief() -> None:
    args = probe.parse_args(["--dataset", "d", "--checkpoints-root", "c"])
    assert args.folds == [0, 1, 2, 3, 4] and args.seed == 42 and args.variant == "wide"
    assert str(args.output) == "reports/kd-nir-probe"


# --- end to end on tiny CPU tensors ---------------------------------------------------------------


def _fake_batches(n: int = 3, size: int = 64):
    g = torch.Generator().manual_seed(0)
    return [
        (
            torch.randn(2, 3, size, size, generator=g),
            torch.randn(2, 1, size, size, generator=g),
            [torch.tensor([[0.5, 0.5, 0.7, 0.7], [0.5, 0.5, 0.2, 0.2]])] * 2,
            [torch.tensor([0, 1])] * 2,
        )
        for _ in range(n)
    ]


def _models():
    from src.models.student.student_model import StudentModel

    torch.manual_seed(0)
    teacher = StudentModel(in_channels=4).eval()
    for module in teacher.head.modules():  # the YOLO init makes the cls head nearly constant
        if isinstance(module, torch.nn.Conv2d):
            torch.nn.init.normal_(module.weight, std=0.3)
    return teacher, StudentModel().eval()


def test_probe_fold_reports_every_level_with_bounded_aurocs() -> None:
    teacher, control = _models()
    result = probe.probe_fold(teacher, control, _fake_batches(), _fake_batches(2), seed=0, cap=64)
    assert list(result) == ["P3", "P4", "P5"]
    for entry in result.values():
        assert entry["n_train"] > 0 and entry["n_val"] > 0
        assert entry["auroc"] is None or 0.0 <= entry["auroc"] <= 1.0
        assert entry["shuffled_auroc"] is None or 0.0 <= entry["shuffled_auroc"] <= 1.0
    assert any(e["auroc"] is not None for e in result.values())


def test_fold_config_takes_the_data_settings_of_the_kd_arm_of_that_fold(tmp_path) -> None:
    (tmp_path / "folds_lot").mkdir()
    meta = {"folds": {"2": {"nir_stats": {"nir_clahe": {"roi": {"mean": 0.31, "std": 0.15}}}}}}
    (tmp_path / "folds_lot" / "folds_meta.json").write_text(json.dumps(meta))
    args = probe.parse_args(["--dataset", str(tmp_path), "--checkpoints-root", str(tmp_path / "ckpt")])
    config = probe.fold_config(args, 2)
    assert config.labels_dir == str(tmp_path / "folds_lot" / "fold_2" / "labels")
    assert config.split_manifest == str(tmp_path / "folds_lot" / "fold_2" / "splits.json")
    assert config.rgb_dir == str(tmp_path / "rgb") and config.nir_dir == str(tmp_path / "nir_clahe")
    assert config.nir_mean == 0.31 and config.nir_std == 0.15


def test_main_reads_train_and_val_only_and_writes_the_report(tmp_path, monkeypatch) -> None:
    (tmp_path / "folds_lot").mkdir()
    meta = {"folds": {str(k): {"nir_stats": {"nir_clahe": {"roi": {"mean": 0.31, "std": 0.15}}}} for k in range(2)}}
    (tmp_path / "folds_lot" / "folds_meta.json").write_text(json.dumps(meta))
    splits = []
    monkeypatch.setattr(probe, "dataset", lambda config, split: splits.append(split) or split)
    monkeypatch.setattr(probe, "batches", lambda ds, device, size: _fake_batches(2))
    monkeypatch.setattr(probe, "load_models", lambda config, args, fold, device: _models())
    code = probe.main([
        "--dataset", str(tmp_path), "--checkpoints-root", str(tmp_path / "ckpt"), "--folds", "0", "1",
        "--device", "cpu", "--output", str(tmp_path / "out"), "--cells-per-image", "32",
    ])
    assert code == 0 and set(splits) == {"train", "val"} and splits.count("train") == 2
    data = json.loads((tmp_path / "out" / "probe.json").read_text())
    assert list(data["folds"]) == ["0", "1"] and (tmp_path / "out" / "README.md").exists()
