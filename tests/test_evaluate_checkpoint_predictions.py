"""`scripts/evaluate_checkpoint.py --save-predictions`: per-image predictions next to metrics.json.

Drives the real `main()`, dataset, dataloader and `Trainer.evaluate` on images written to disk; only
the model is replaced by one whose head output is fixed, so every saved box is known in advance.

Geometry used below, all at `image_size` 64 with a student head (strides 8, 16, 32): the fixed damage
anchor is cell (3, 3) of the stride 8 level, so its center is (3 + 0.5) * 8 = 28 and its width and
height are 16. In letterbox pixels the box is therefore always xyxy (20, 20, 36, 36).
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
import torch.nn as nn
import yaml

from scripts import evaluate_checkpoint as ec
from scripts import v4_metrics as metrics_cli
from src.data_pipeline.v4_selection import LATENT_HOURS

IMAGE_SIZE = 64
LEVEL_SIDES = (8, 4, 2)
BELOW_ANY_THRESHOLD = 1e-4


def _logit(p: float) -> float:
    return math.log(p / (1 - p))


def _head_output(damage_score: float | None, mango_score: float | None = None) -> dict:
    """Head output with at most one damage and one mango candidate above `BELOW_ANY_THRESHOLD`."""
    preds, cls_preds = [], []
    for side in LEVEL_SIDES:
        cls = torch.full((1, 2, side, side), _logit(BELOW_ANY_THRESHOLD))
        reg = torch.zeros((1, 4, side, side))
        if side == LEVEL_SIDES[0]:
            if damage_score is not None:
                cls[0, 1, 3, 3] = _logit(damage_score)
            if mango_score is not None:
                cls[0, 0, 5, 5] = _logit(mango_score)
            reg[0, 2:, :, :] = math.log(16.0)
        cls_preds.append(cls)
        preds.append(torch.cat([cls, reg], dim=1))
    return {"preds": preds, "cls_preds": cls_preds}


class _FixedHead(nn.Module):
    def __init__(self, output: dict):
        super().__init__()
        self.output = output

    def forward(self, rgb):
        return self.output


def _write_photo(root: Path, stem: str, size: tuple[int, int], label: str) -> None:
    width, height = size
    cv2.imwrite(str(root / "rgb" / f"{stem}.jpg"), np.zeros((height, width, 3), np.uint8))
    nir_name = stem.replace("_rgb", "_nir")
    cv2.imwrite(str(root / "nir" / f"{nir_name}.png"), np.zeros((height, width), np.uint8))
    (root / "labels" / "test" / f"{stem}.txt").write_text(label)


def _dataset(root: Path, photos: dict[str, str], size: tuple[int, int] = (100, 50), **config) -> Path:
    for sub in ("rgb", "nir", "labels/test"):
        (root / sub).mkdir(parents=True)
    for stem, label in photos.items():
        _write_photo(root, stem, size, label)
    path = root / "config.yaml"
    path.write_text(yaml.safe_dump({
        "model_type": "student",
        "num_classes": 2,
        "image_size": IMAGE_SIZE,
        "precision": "fp32",
        "batch_size": 1,
        "effective_batch": 1,
        "num_workers": 0,
        "rgb_dir": str(root / "rgb"),
        "nir_dir": str(root / "nir"),
        "labels_dir": str(root / "labels"),
        "output_dir": str(root / "trainer"),
        "split_manifest": None,
        "nir_homography_path": None,
        **config,
    }))
    return path


def _main(monkeypatch, config: Path, out: Path, output: dict, *extra: str) -> int:
    monkeypatch.setattr(ec, "_load_model", lambda *args, **kwargs: _FixedHead(output))
    monkeypatch.setattr(sys, "argv", [
        "evaluate_checkpoint.py", "--config", str(config), "--checkpoint", "unused.pt", "--split", "test",
        "--device", "cpu", "--output-dir", str(out), *extra,
    ])
    return ec.main()


def _lines(out: Path) -> list[dict]:
    return [json.loads(line) for line in (out / "predictions.jsonl").read_text().splitlines()]


def test_save_predictions_writes_one_line_per_image_in_the_source_rgb_frame(tmp_path, monkeypatch) -> None:
    config = _dataset(tmp_path, {"mango_rgb_1": "", "mango_rgb_2": ""})
    out = tmp_path / "eval"
    assert _main(monkeypatch, config, out, _head_output(0.9, 0.8), "--save-predictions") == 0

    lines = _lines(out)
    assert [line["stem"] for line in lines] == ["mango_rgb_1", "mango_rgb_2"]
    damage = next(box for box in lines[0]["boxes"] if box["class"] == 1)
    assert damage["confidence"] == pytest.approx(0.9, abs=1e-5)
    # 100 x 50 image letterboxed to 64: scale 0.64, 32 px tall, pad_y 16, pad_x 0.
    # x: 20 / 0.64 = 31.25 and 36 / 0.64 = 56.25, over a width of 100.
    # y: (20 - 16) / 0.64 = 6.25 and (36 - 16) / 0.64 = 31.25, over a height of 50.
    assert damage["xyxy"] == pytest.approx([0.3125, 0.125, 0.5625, 0.625], abs=1e-5)
    mango = next(box for box in lines[0]["boxes"] if box["class"] == 0)
    assert mango["confidence"] == pytest.approx(0.8, abs=1e-5)
    assert (out / "metrics.json").is_file()


def test_saved_boxes_undo_the_roi_crop(tmp_path, monkeypatch) -> None:
    # 320 x 240 image, mango at x [120, 200], y [90, 150], margin 0.5 of its size: the window is
    # x [80, 240], y [60, 180], so 160 x 120, letterboxed with scale 0.4, pad_y 8, pad_x 0.
    label = "0 0.5 0.5 0.25 0.25\n"
    config = _dataset(tmp_path, {"mango_rgb_1": label}, size=(320, 240), roi_crop=True, roi_crop_margin=0.5)
    out = tmp_path / "eval"
    assert _main(monkeypatch, config, out, _head_output(0.9), "--save-predictions") == 0

    (line,) = _lines(out)
    (damage,) = [box for box in line["boxes"] if box["class"] == 1]
    # x: 80 + 20 / 0.4 = 130 and 80 + 36 / 0.4 = 170, over 320.
    # y: 60 + (20 - 8) / 0.4 = 90 and 60 + (36 - 8) / 0.4 = 130, over 240.
    assert damage["xyxy"] == pytest.approx([130 / 320, 90 / 240, 170 / 320, 130 / 240], abs=1e-5)


@pytest.mark.parametrize("roi_crop", [False, True])
def test_saved_predictions_and_label_files_share_a_frame(tmp_path, monkeypatch, roi_crop: bool) -> None:
    """`metrics.json` scores in the letterbox frame, `v4_metrics` against the label files: same AP."""
    if roi_crop:
        size, extra = (320, 240), {"roi_crop": True, "roi_crop_margin": 0.5}
        mango = "0 0.5 0.5 0.25 0.25\n"
        x1, y1, x2, y2 = 130 / 320, 90 / 240, 170 / 320, 130 / 240
    else:
        size, extra = (100, 50), {}
        mango = "0 0.5 0.5 0.9 0.9\n"
        x1, y1, x2, y2 = 0.3125, 0.125, 0.5625, 0.625
    damage = f"1 {(x1 + x2) / 2} {(y1 + y2) / 2} {x2 - x1} {y2 - y1}\n"
    # The fixed head predicts the same box on both photos: a false positive on the sano one, ranked
    # first because it comes first among equal scores. FP then TP gives AP 0.5 on both sides.
    config = _dataset(tmp_path, {"mango_rgb_1": mango, "mango_rgb_2": mango + damage}, size=size, **extra)
    out = tmp_path / "eval"
    assert _main(monkeypatch, config, out, _head_output(0.9), "--save-predictions") == 0

    index = tmp_path / "index.csv"
    with index.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["stem", "lot", "mango_id", "cara", "ronda_h", "zona", "provenance"])
        writer.writerow(["mango_rgb_1", 1, 1, 1, "sano", "", ""])
        writer.writerow(["mango_rgb_2", 1, 1, 1, LATENT_HOURS[0], "centro", "dibujada"])
    assert metrics_cli.main(["--predictions", str(out / "predictions.jsonl"), "--index", str(index),
                             "--labels-dir", str(tmp_path / "labels" / "test"),
                             "--output", str(out / "metrics_v4.json")]) == 0

    from_metrics_json = json.loads((out / "metrics.json").read_text())["per_class_ap_50"]["1"]
    from_predictions = json.loads((out / "metrics_v4.json").read_text())["latent_ap50"]["ap"]
    assert from_metrics_json == pytest.approx(0.5)
    assert from_predictions == pytest.approx(from_metrics_json)


def test_predictions_are_decoded_at_the_evaluation_threshold(tmp_path, monkeypatch) -> None:
    config = _dataset(tmp_path, {"mango_rgb_1": ""})
    out = tmp_path / "eval"
    # Below conf_threshold (0.25), above eval_conf_threshold (0.001).
    assert _main(monkeypatch, config, out, _head_output(0.05), "--save-predictions") == 0
    (line,) = _lines(out)
    (damage,) = [box for box in line["boxes"] if box["class"] == 1]
    assert damage["confidence"] == pytest.approx(0.05, abs=1e-5)


def test_an_image_with_no_candidate_still_gets_its_line(tmp_path, monkeypatch) -> None:
    config = _dataset(tmp_path, {"mango_rgb_1": ""})
    out = tmp_path / "eval"
    assert _main(monkeypatch, config, out, _head_output(None), "--save-predictions") == 0
    assert _lines(out) == [{"stem": "mango_rgb_1", "boxes": []}]


def test_without_the_flag_no_predictions_are_written_and_metrics_are_the_same(tmp_path, monkeypatch, capsys) -> None:
    config = _dataset(tmp_path, {"mango_rgb_1": "", "mango_rgb_2": ""})
    plain, flagged = tmp_path / "plain", tmp_path / "flagged"
    assert _main(monkeypatch, config, plain, _head_output(0.9)) == 0
    plain_stdout = capsys.readouterr().out
    assert _main(monkeypatch, config, flagged, _head_output(0.9), "--save-predictions") == 0

    assert sorted(p.name for p in plain.iterdir()) == ["metrics.json"]
    assert sorted(p.name for p in flagged.iterdir()) == ["metrics.json", "predictions.jsonl"]
    assert (plain / "metrics.json").read_bytes() == (flagged / "metrics.json").read_bytes()
    assert capsys.readouterr().out == plain_stdout


def test_save_predictions_is_refused_with_assigner_stats(tmp_path, monkeypatch, capsys) -> None:
    config = _dataset(tmp_path, {"mango_rgb_1": ""})
    with pytest.raises(SystemExit) as excinfo:
        _main(monkeypatch, config, tmp_path / "eval", _head_output(0.9), "--save-predictions", "--assigner-stats")
    assert excinfo.value.code == 2
    assert "--save-predictions" in capsys.readouterr().err
    assert not (tmp_path / "eval").exists()
