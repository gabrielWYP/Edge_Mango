"""`evaluate_checkpoint.py --save-predictions` for a segment model: masks in predictions.jsonl.

Same fixed-head setup and geometry as `test_evaluate_checkpoint_predictions.py` (image 64, damage
anchor cell (3, 3) of stride 8, letterbox box (20, 20, 36, 36)). The prototypes are 16 x 16, so the
box covers prototype cells 5..8 and the mask polygon, traced along pixel edges, is that box exactly.
"""

from __future__ import annotations

import csv
import json

import torch

from scripts import v4_metrics as metrics_cli
from src.data_pipeline.v4_selection import LATENT_HOURS
from tests.test_evaluate_checkpoint_predictions import _dataset, _head_output, _lines, _main

PROTO = 16


def _segment_output(damage_score: float = 0.9, mango_score: float | None = None, nm: int = 1) -> dict:
    output = _head_output(damage_score, mango_score)
    sides = [p.shape[-1] for p in output["preds"]]
    coefs = [torch.zeros(1, nm, side, side) for side in sides]
    coefs[0][0, 0, 3, 3] = 1.0  # the damage anchor
    coefs[0][0, 0, 5, 5] = 1.0  # the mango anchor
    return {**output, "mask_coefs": coefs, "protos": torch.full((1, nm, PROTO, PROTO), 10.0)}


def _corners(flat: list[float]) -> tuple[list[float], list[float]]:
    return sorted(set(flat[0::2])), sorted(set(flat[1::2]))


def test_a_damage_detection_carries_its_mask_polygon_in_the_source_frame(tmp_path, monkeypatch) -> None:
    config = _dataset(tmp_path, {"mango_rgb_1": ""}, task="segment")
    out = tmp_path / "eval"
    assert _main(monkeypatch, config, out, _segment_output(0.9, 0.8), "--save-predictions") == 0

    (line,) = _lines(out)
    (damage,) = [b for b in line["boxes"] if b["class"] == 1]
    (mango,) = [b for b in line["boxes"] if b["class"] == 0]
    # 100 x 50 image letterboxed to 64: x 20 / 0.64 = 31.25 and 36 / 0.64 = 56.25 over 100,
    # y (20 - 16) / 0.64 = 6.25 and (36 - 16) / 0.64 = 31.25 over 50.
    xs, ys = _corners(damage["mask"])
    assert xs == [0.3125, 0.5625] and ys == [0.125, 0.625]
    assert "mask" not in mango


def test_the_mask_polygon_undoes_the_roi_crop_like_the_box(tmp_path, monkeypatch) -> None:
    label = "0 0.5 0.5 0.25 0.25\n"
    config = _dataset(tmp_path, {"mango_rgb_1": label}, size=(320, 240), task="segment",
                      roi_crop=True, roi_crop_margin=0.5)
    out = tmp_path / "eval"
    assert _main(monkeypatch, config, out, _segment_output(0.9), "--save-predictions") == 0
    (line,) = _lines(out)
    (damage,) = [b for b in line["boxes"] if b["class"] == 1]
    xs, ys = _corners(damage["mask"])
    assert xs == [round(130 / 320, 6), round(170 / 320, 6)] and ys == [round(90 / 240, 6), round(130 / 240, 6)]


def test_an_empty_mask_is_stored_as_an_empty_polygon(tmp_path, monkeypatch) -> None:
    config = _dataset(tmp_path, {"mango_rgb_1": ""}, task="segment")
    out = tmp_path / "eval"
    output = _segment_output(0.9)
    output["protos"] = torch.full((1, 1, PROTO, PROTO), -10.0)
    assert _main(monkeypatch, config, out, output, "--save-predictions") == 0
    (line,) = _lines(out)
    (damage,) = [b for b in line["boxes"] if b["class"] == 1]
    assert damage["mask"] == []


def test_a_detect_run_writes_no_mask_field(tmp_path, monkeypatch) -> None:
    config = _dataset(tmp_path, {"mango_rgb_1": ""})
    out = tmp_path / "eval"
    assert _main(monkeypatch, config, out, _head_output(0.9, 0.8), "--save-predictions") == 0
    assert all("mask" not in box for line in _lines(out) for box in line["boxes"])


def test_saved_masks_feed_the_v4_mask_metrics_end_to_end(tmp_path, monkeypatch) -> None:
    # The 100 x 50 photo holds a damage polygon equal to the predicted mask (x 0.3125-0.5625, y 0.125-0.625).
    polygon = "1 0.3125 0.125 0.5625 0.125 0.5625 0.625 0.3125 0.625\n"
    config = _dataset(tmp_path, {"mango_rgb_1": "0 0.5 0.5 0.9 0.9\n" + polygon}, task="segment")
    out = tmp_path / "eval"
    assert _main(monkeypatch, config, out, _segment_output(0.9), "--save-predictions") == 0
    index = tmp_path / "index.csv"
    with index.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["stem", "lot", "mango_id", "cara", "ronda_h", "zona", "provenance"])
        writer.writerow(["mango_rgb_1", 1, 1, 1, LATENT_HOURS[0], "centro", "dibujada"])
    assert metrics_cli.main(["--predictions", str(out / "predictions.jsonl"), "--index", str(index),
                             "--labels-dir", str(tmp_path / "labels" / "test"),
                             "--output", str(out / "metrics_v4.json")]) == 0
    metrics = json.loads((out / "metrics_v4.json").read_text())
    assert metrics["latent_mask_ap50"]["ap"] == 1.0 and metrics["latent_ap50"]["ap"] == 1.0
