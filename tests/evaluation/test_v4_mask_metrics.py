"""Mask-IoU twins of the latent v4 metrics, through `scripts/v4_metrics.py`.

The predictions carry each damage detection's mask as a normalized polygon of the source frame
(`"mask": [x1, y1, x2, y2, ...]`), the label files YOLO-seg polygon lines. IoU is invariant to the
rasterization canvas, so every expectation is derived from areas on the unit square.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from scripts import v4_metrics as cli
from src.data_pipeline.v4_selection import LATENT_HOURS
from src.evaluation import v4_metrics as module
from src.training.seg_labels import mask_to_polygon

SANO = "sano"
LATENT = LATENT_HOURS[0]
DEVELOPED = max(LATENT_HOURS) + 2
FIELDS = ["stem", "lot", "mango_id", "cara", "ronda_h", "zona", "provenance"]

LOWER_LEFT = [0.1, 0.1, 0.1, 0.3, 0.3, 0.3]   # triangle below the diagonal of the square
UPPER_RIGHT = [0.1, 0.1, 0.3, 0.1, 0.3, 0.3]  # the other half: same box, no shared area
SQUARE = [0.1, 0.1, 0.3, 0.1, 0.3, 0.3, 0.1, 0.3]
TOP_40 = [0.1, 0.1, 0.3, 0.1, 0.3, 0.18, 0.1, 0.18]  # inside SQUARE, 40% of its area


def _xyxy(poly: list[float]) -> list[float]:
    xs, ys = poly[0::2], poly[1::2]
    return [min(xs), min(ys), max(xs), max(ys)]


def _line(cls: int, poly: list[float]) -> str:
    return f"{cls} " + " ".join(str(v) for v in poly)


def _photo(stem: str, hour, *, gt=(SQUARE,), gt_lines=None, preds=(), lot=1) -> dict:
    sano = hour == SANO
    return {
        "stem": stem, "lot": lot, "mango_id": stem, "cara": 1, "ronda_h": hour,
        "zona": "" if sano else "centro", "provenance": "" if sano else "dibujada",
        "gt": () if sano else gt, "gt_lines": gt_lines, "preds": preds,
    }


def _run(root: Path, photos: list[dict], *, with_masks: bool = True) -> dict:
    labels = root / "labels" / "test"
    labels.mkdir(parents=True)
    with (root / "index.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(photos)
    lines = []
    for photo in photos:
        rows = ["0 0.5 0.5 0.9 0.9"]
        rows += photo["gt_lines"] if photo["gt_lines"] is not None else [_line(1, p) for p in photo["gt"]]
        (labels / f"{photo['stem']}.txt").write_text("\n".join(rows) + "\n")
        boxes = []
        for confidence, poly in photo["preds"]:
            box = {"class": 1, "confidence": confidence, "xyxy": _xyxy(poly)}
            if with_masks:
                box["mask"] = poly
            boxes.append(box)
        lines.append(json.dumps({"stem": photo["stem"], "boxes": boxes}))
    (root / "predictions.jsonl").write_text("\n".join(lines) + "\n")
    out = root / "out" / "metrics_v4.json"
    code = cli.main(["--predictions", str(root / "predictions.jsonl"), "--index", str(root / "index.csv"),
                     "--labels-dir", str(labels), "--output", str(out)])
    assert code == 0
    return json.loads(out.read_text())


def test_same_box_but_disjoint_masks_scores_box_ap_one_and_mask_ap_zero(tmp_path: Path) -> None:
    photo = _photo("a", LATENT, gt=(LOWER_LEFT,), preds=[(0.9, UPPER_RIGHT)])
    metrics = _run(tmp_path, [photo])
    assert metrics["latent_ap50"]["ap"] == pytest.approx(1.0)
    assert metrics["latent_mask_ap50"]["ap"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["latent_mask_ap50"]["n_gt"] == 1


def test_identical_masks_score_one_and_a_sano_false_positive_ranked_first_halves_it(tmp_path: Path) -> None:
    ok = _run(tmp_path / "a", [_photo("a", LATENT, preds=[(0.9, SQUARE)])])
    assert ok["latent_mask_ap50"]["ap"] == pytest.approx(1.0)
    both = _run(tmp_path / "b", [_photo("s", SANO, preds=[(0.9, SQUARE)]), _photo("a", LATENT, preds=[(0.9, SQUARE)])])
    assert both["latent_mask_ap50"]["ap"] == pytest.approx(0.5)
    assert both["latent_mask_ap50"]["n_images"] == 2


def test_ap30_and_ap50_split_an_iou_of_040(tmp_path: Path) -> None:
    metrics = _run(tmp_path, [_photo("a", LATENT, preds=[(0.9, TOP_40)])])
    assert metrics["latent_mask_ap30"]["ap"] == pytest.approx(1.0)
    assert metrics["latent_mask_ap50"]["ap"] == pytest.approx(0.0)


def test_per_hour_and_developed_mask_ap_skip_sano_and_the_latent_window(tmp_path: Path) -> None:
    photos = [
        _photo("s", SANO, preds=[(0.9, SQUARE)]),
        _photo("l", LATENT, preds=[(0.9, SQUARE)]),
        _photo("d", DEVELOPED, preds=[(0.9, TOP_40)]),
    ]
    metrics = _run(tmp_path, photos)
    assert metrics["mask_ap50_by_hour"][str(LATENT)]["ap"] == pytest.approx(1.0)
    assert metrics["mask_ap50_by_hour"][str(DEVELOPED)]["ap"] == pytest.approx(0.0)
    assert metrics["developed_mask_ap50"]["ap"] == pytest.approx(0.0)
    assert metrics["developed_mask_ap50"]["n_images"] == 1


def test_a_box_only_ground_truth_line_is_left_out_of_the_mask_ground_truth(tmp_path: Path) -> None:
    box_line = "1 0.7 0.7 0.1 0.1"
    photo = _photo("a", LATENT, gt_lines=[_line(1, SQUARE), box_line], preds=[(0.9, SQUARE)])
    metrics = _run(tmp_path, [photo])
    assert metrics["latent_mask_ap50"]["n_gt"] == 1
    assert metrics["latent_ap50"]["n_gt"] == 2
    assert metrics["mask_gt_box_only"] == 1


def test_a_stratum_without_polygon_ground_truth_is_null(tmp_path: Path) -> None:
    metrics = _run(tmp_path, [_photo("s", SANO, preds=[(0.9, SQUARE)])])
    assert metrics["latent_mask_ap50"]["ap"] is None
    assert metrics["developed_mask_ap50"]["ap"] is None


def test_box_metrics_do_not_change_when_masks_are_present(tmp_path: Path) -> None:
    photos = [_photo("s", SANO, preds=[(0.8, SQUARE)]), _photo("a", LATENT, preds=[(0.9, SQUARE)])]
    with_masks = _run(tmp_path / "m", photos, with_masks=True)
    without = _run(tmp_path / "d", photos, with_masks=False)
    mask_keys = {"latent_mask_ap50", "latent_mask_ap30", "mask_ap50_by_hour", "developed_mask_ap50", "mask_gt_box_only"}
    assert mask_keys.isdisjoint(without)
    assert {k: v for k, v in with_masks.items() if k not in mask_keys} == without


def test_a_detection_without_a_mask_in_a_segment_file_is_rejected(tmp_path: Path, capsys) -> None:
    root = tmp_path
    labels = root / "labels" / "test"
    labels.mkdir(parents=True)
    (labels / "a.txt").write_text(_line(1, SQUARE) + "\n")
    (root / "index.csv").write_text(",".join(FIELDS) + f"\na,1,a,1,{LATENT},centro,dibujada\n")
    box = {"class": 1, "confidence": 0.9, "xyxy": _xyxy(SQUARE)}
    (root / "predictions.jsonl").write_text(json.dumps({"stem": "a", "boxes": [{**box, "mask": SQUARE}, box]}) + "\n")
    code = cli.main(["--predictions", str(root / "predictions.jsonl"), "--index", str(root / "index.csv"),
                     "--labels-dir", str(labels), "--output", str(root / "o.json")])
    assert code == 2 and "mask" in capsys.readouterr().err


def _area(polygon: np.ndarray) -> float:
    x, y = polygon[:, 0], polygon[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def test_mask_to_polygon_traces_pixel_edges_and_keeps_the_largest_blob() -> None:
    mask = np.zeros((40, 50), dtype=np.uint8)
    mask[5:25, 5:25] = 1
    mask[30:33, 30:33] = 1  # smaller blob, dropped
    polygon = mask_to_polygon(mask)
    assert polygon.min() >= 0 and polygon.max() <= 1
    # Pixel i covers [i, i + 1): the 20 x 20 block is 400 pixels of area, not 19 x 19.
    assert _area(polygon) * 40 * 50 == pytest.approx(400.0)
    assert polygon[:, 0].min() == pytest.approx(5 / 50) and polygon[:, 0].max() == pytest.approx(25 / 50)


def test_mask_to_polygon_of_an_empty_mask_is_none() -> None:
    assert mask_to_polygon(np.zeros((8, 8), dtype=np.uint8)) is None


def test_read_prediction_masks_is_none_for_a_detect_file(tmp_path: Path) -> None:
    path = tmp_path / "p.jsonl"
    path.write_text(json.dumps({"stem": "a", "boxes": [{"class": 1, "confidence": 0.5, "xyxy": [0, 0, 1, 1]}]}) + "\n")
    assert module.read_prediction_masks(path) is None
