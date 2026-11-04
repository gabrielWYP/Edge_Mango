"""Tests for the v4 leave-one-lot-out folds (scripts/make_v4_folds.py, plan 3.3)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from scripts import make_v4_folds as mvf
from scripts.make_grouped_folds import LeakageError, dump_json
from src.training.dataset import YOLODataset
from scripts.aggregate_kfold import load_fold_spec
from scripts.make_v4_folds import main as make_v4_folds_main

DAY = 86400
LOT_START = {1: 1792861156, 2: 1792861156 + DAY, 3: 1792861156 + 3 * DAY}  # lots 1-2 back to back
# NIR value inside the mango box per lot; 0 elsewhere. nir_clahe holds half of it.
LOT_VALUE = {1: 51, 2: 102, 3: 153}
SIZE = 100
MANGO = "0 0.4 0.4 0.4 0.4\n"  # pixels [20, 60) on both axes


def _write_dataset(root: Path, mango_lot: dict[int, int], photos_per_mango: int = 2) -> Path:
    for sub in ("rgb", "labels", *mvf.NIR_VARIANTS):
        (root / sub).mkdir(parents=True)
    rows = []
    for mango, lot in sorted(mango_lot.items()):
        for k in range(photos_per_mango):
            ts = LOT_START[lot] + 60 * (10 * mango + k)
            stem = f"mango_rgb_{ts}"
            damage = "1 0.3 0.3 0.05 0.05\n" if k else ""
            (root / "labels" / f"{stem}.txt").write_text(MANGO + damage)
            cv2.imwrite(str(root / "rgb" / f"{stem}.jpg"), np.zeros((SIZE, SIZE, 3), np.uint8))
            for variant, scale in (("nir", 1), ("nir_clahe", 2)):
                nir = np.zeros((SIZE, SIZE), np.uint8)
                nir[20:60, 20:60] = LOT_VALUE[lot] // scale
                cv2.imwrite(str(mvf.nir_path(root, variant, stem)), nir)
            rows.append({"stem": stem, "timestamp": ts, "lot": lot, "mango_id": mango})
    with (root / "index.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return root


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    return _write_dataset(tmp_path / "v4", {1: 1, 2: 1, 26: 2, 27: 2, 51: 3, 52: 3})


def test_each_lot_is_tested_once_and_validated_on_the_next(dataset: Path) -> None:
    meta = mvf.make_v4_folds(dataset, workers=1)

    assert [(f["test"], f["val"], f["train"]) for f in meta["folds"].values()] == [
        ("lot1", "lot2", ["lot3"]),
        ("lot2", "lot3", ["lot1"]),
        ("lot3", "lot1", ["lot2"]),
    ]
    assert meta["lots"]["lot1"] == {
        "damage_boxes": 2,
        "days": ["2026-10-24"],
        "images": 4,
        "mangos": 2,
        "negatives": 2,
    }
    fold = dataset / "folds_lot" / "fold_0"
    manifest = json.loads((fold / "splits.json").read_text())
    for split, lot in (("test", 1), ("val", 2), ("train", 3)):
        assert manifest[split] == sorted(
            r["stem"] for r in mvf.read_index(dataset / "index.csv") if r["lot"] == str(lot)
        )
        assert {p.stem for p in (fold / "labels" / split).glob("*.txt")} == set(manifest[split])
    loaded = json.loads((dataset / "folds_lot" / "folds_meta.json").read_text())
    assert loaded == json.loads(dump_json(meta))


def test_a_fold_loads_in_yolo_dataset(dataset: Path) -> None:
    mvf.make_v4_folds(dataset, workers=1)
    fold = dataset / "folds_lot" / "fold_1"
    data = YOLODataset(
        rgb_dir=dataset / "rgb",
        nir_dir=dataset / "nir",
        labels_dir=fold / "labels",
        split="test",
        manifest_path=fold / "splits.json",
        nir_homography_path=None,
    )
    assert len(data) == 4


def test_a_mango_in_two_lots_is_rejected(tmp_path: Path) -> None:
    dataset = _write_dataset(tmp_path / "v4", {1: 1, 2: 1, 26: 2, 27: 2, 51: 3, 52: 3})
    rows = mvf.read_index(dataset / "index.csv")
    rows[-1]["mango_id"] = "26"  # a lot-3 photo labelled as a lot-2 mango
    with pytest.raises(LeakageError, match="mango_id \\[26\\]"):
        mvf.check_mango_ids(rows, mvf.fold_manifests(*_lots_and_folds(rows)))


def test_a_repeated_stem_is_rejected(dataset: Path) -> None:
    rows = mvf.read_index(dataset / "index.csv")
    with pytest.raises(LeakageError, match="twice"):
        mvf.check_mango_ids(rows + rows[:1], mvf.fold_manifests(*_lots_and_folds(rows)))


def _lots_and_folds(rows):
    lots = mvf.lots_from_index(rows)
    return lots, mvf.assign_folds(lots)


def test_nir_stats_come_from_the_train_split_of_each_variant(dataset: Path) -> None:
    meta = mvf.make_v4_folds(dataset, workers=1)

    for fold, train_lot in (("0", 3), ("1", 1), ("2", 2)):
        for variant, scale in (("nir", 1), ("nir_clahe", 2)):
            value = LOT_VALUE[train_lot] // scale / 255
            stats = meta["folds"][fold]["nir_stats"][variant]
            # 40x40 mango pixels in a 100x100 frame and in the 48x48 ROI window [16, 64).
            for view, inside in (("frame", 0.16), ("roi", 1600 / 2304)):
                assert stats[view]["mean"] == pytest.approx(value * inside, abs=1e-6)
                std = value * np.sqrt(inside * (1 - inside))
                assert stats[view]["std"] == pytest.approx(std, abs=1e-6)


def test_the_fold_spec_is_written_for_the_aggregator(dataset: Path, tmp_path: Path) -> None:
    spec_path = tmp_path / "configs" / "v4_lot_5fold.json"
    assert make_v4_folds_main(["--dataset", str(dataset), "--workers", "1", "--fold-spec", str(spec_path)]) == 0

    meta = json.loads((dataset / "folds_lot" / "folds_meta.json").read_text())
    spec = json.loads(spec_path.read_text())
    assert spec["index_sha256"] == meta["index_sha256"]
    assert spec["labels_sha256"] == meta["labels_sha256"]
    entries = load_fold_spec(spec_path)
    assert [entry["fold"] for entry in entries] == sorted(int(k) for k in meta["folds"])
    for entry in entries:
        fold = meta["folds"][str(entry["fold"])]
        assert (entry["test"], entry["val"], entry["train"]) == (fold["test"], fold["val"], fold["train"])
        for split in ("test", "train", "val"):
            assert entry[f"num_{split}"] == fold["splits"][split]["images"]


def test_no_fold_spec_is_written_unless_asked(dataset: Path, tmp_path: Path, monkeypatch) -> None:
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    assert make_v4_folds_main(["--dataset", str(dataset), "--workers", "1"]) == 0
    assert list(cwd.iterdir()) == []
