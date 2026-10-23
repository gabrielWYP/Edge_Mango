"""Tests for scripts/export_aligned_corpus.py (aligned RGB+NIR corpus zip).

Everything here is synthetic: no real images, labels or manifests are read.
Stems are built from America/Lima wall-clock times so the lot and homography
rules are exercised exactly as they apply to the real `mango_rgb_<unix_ts>`
captures. Box coordinates are chosen as multiples of 5 px on a 1280x640 frame,
which are exact binary fractions once normalized, so the 50 % visibility
boundary is tested without float noise.
"""

from __future__ import annotations

import csv
import hashlib
import inspect
import io
import json
import os
import re
import shutil
import subprocess
import zipfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import cv2
import numpy as np
import pytest
import yaml

from scripts import export_aligned_corpus as eac
from src.training.config import TrainingConfig
from src.training.dataset import YOLODataset
from src.training.nir_registration import load_homography, register_nir_to_rgb
from src.training.roi_crop import crop_to_roi

LIMA = ZoneInfo("America/Lima")

# 1280x640 test frame: every coordinate below is a multiple of 5 px, so
# `px / 1280` and `px / 640` are exact binary fractions.
W, H = 1280, 640


def _stem(local: str, seq: int = 0) -> str:
    """`mango_rgb_<unix_ts>` for a Lima wall-clock time like '2026-10-18 10:00:00'."""
    moment = datetime.strptime(local, "%Y-%m-%d %H:%M:%S").replace(tzinfo=LIMA)
    return f"mango_rgb_{int(moment.timestamp()) + seq}"


def _norm(box_px: tuple[float, float, float, float], w: int = W, h: int = H) -> list[float]:
    """Pixel x1,y1,x2,y2 -> YOLO cx,cy,w,h normalized to a w x h frame."""
    x1, y1, x2, y2 = box_px
    return [(x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h]


# --------------------------------------------------------------------------
# Square window
# --------------------------------------------------------------------------


def test_square_window_centred_on_a_landscape_window() -> None:
    # 200x100 window centred on (600, 350): side 200, grown vertically.
    assert eac.square_window((500, 300, 700, 400), 1280, 720) == (500, 250, 700, 450)


def test_square_window_centred_on_a_portrait_window() -> None:
    # 100x300 window centred on (650, 250): side 300, grown horizontally.
    assert eac.square_window((600, 100, 700, 400), 1280, 720) == (500, 100, 800, 400)


def test_square_window_clamped_at_the_left_border() -> None:
    # Centering would start at x = -90; the square shifts right to stay inside.
    assert eac.square_window((0, 200, 20, 400), 1280, 720) == (0, 200, 200, 400)


def test_square_window_clamped_at_the_right_border() -> None:
    assert eac.square_window((1260, 200, 1280, 400), 1280, 720) == (1080, 200, 1280, 400)


def test_square_window_clamped_at_the_top_border() -> None:
    assert eac.square_window((500, 0, 700, 20), 1280, 720) == (500, 0, 700, 200)


def test_square_window_clamped_at_the_bottom_border() -> None:
    assert eac.square_window((500, 700, 700, 720), 1280, 720) == (500, 520, 700, 720)


def test_square_window_side_larger_than_the_image_height_is_clamped() -> None:
    # 1000x100 window: the side would be 1000 but the frame is only 720 tall.
    assert eac.square_window((100, 200, 1100, 300), 1280, 720) == (240, 0, 960, 720)
    # A full-width strip is clamped the same way and centred horizontally.
    assert eac.square_window((0, 0, 1280, 100), 1280, 720) == (280, 0, 1000, 720)


def test_square_window_odd_difference_rounds_down_deterministically() -> None:
    # 101x50 window: side 101, centre x2 = 2*0 + 101 = odd -> floor division.
    assert eac.square_window((300, 300, 401, 350), 1280, 720) == (300, 274, 401, 375)
    # Same rule horizontally: 50x101 window, x2 centre = 650 - 101 = 549 -> 274, not 275.
    assert eac.square_window((300, 300, 350, 401), 1280, 720) == (274, 300, 375, 401)


@pytest.mark.parametrize("seed", range(20))
def test_square_window_properties_hold_for_random_windows(seed: int) -> None:
    rng = np.random.default_rng(seed)
    x0 = int(rng.integers(0, 1200))
    y0 = int(rng.integers(0, 650))
    x1 = int(rng.integers(x0 + 1, 1281))
    y1 = int(rng.integers(y0 + 1, 721))
    X0, Y0, X1, Y1 = eac.square_window((x0, y0, x1, y1), 1280, 720)
    side = X1 - X0
    assert side == Y1 - Y0
    assert 0 <= X0 and X1 <= 1280 and 0 <= Y0 and Y1 <= 720
    assert side == min(max(x1 - x0, y1 - y0), 720)
    if side == max(x1 - x0, y1 - y0):  # not clamped: the original window is inside
        assert X0 <= x0 and y0 >= Y0 and x1 <= X1 and y1 <= Y1


def test_square_window_rejects_an_empty_window() -> None:
    with pytest.raises(ValueError):
        eac.square_window((10, 10, 10, 50), 1280, 720)


# --------------------------------------------------------------------------
# crop_to_roi window
# --------------------------------------------------------------------------


def _window_from_crop_to_roi(boxes: np.ndarray, classes: np.ndarray, w: int, h: int) -> tuple:
    """The window the real `crop_to_roi` cuts, read from position-coded planes."""
    xs = np.broadcast_to(np.arange(w, dtype=np.float32), (h, w))
    ys = np.broadcast_to(np.arange(h, dtype=np.float32)[:, None], (h, w))
    rgb = np.stack([ys, ys, ys], axis=-1)
    cropped_rgb, cropped_nir, _, _ = crop_to_roi(rgb, xs, boxes, classes, margin=0.10)
    ch, cw = cropped_nir.shape
    x0, y0 = int(cropped_nir[0, 0]), int(cropped_rgb[0, 0, 0])
    return (x0, y0, x0 + cw, y0 + ch)


def test_roi_window_matches_a_hand_computed_window() -> None:
    boxes = np.array([_norm((575, 290, 705, 430), 1280, 720)])
    classes = np.array([0])
    # roi 130x140 + 10 % per side: left 562 / right 718 / top 276 / bottom 444.
    assert eac.roi_window(boxes, classes, 1280, 720) == (562, 276, 718, 444)


@pytest.mark.parametrize("seed", range(25))
def test_roi_window_equals_the_window_crop_to_roi_cuts(seed: int) -> None:
    rng = np.random.default_rng(seed)
    w, h = 1280, 720
    mw, mh = rng.uniform(0.04, 0.45), rng.uniform(0.06, 0.8)
    mcx, mcy = rng.uniform(mw / 2, 1 - mw / 2), rng.uniform(mh / 2, 1 - mh / 2)
    boxes = [[mcx, mcy, mw, mh]]
    classes = [0]
    for _ in range(int(rng.integers(0, 4))):  # damage boxes, some off the mango
        boxes.append([rng.uniform(0.05, 0.95), rng.uniform(0.05, 0.95), 0.03, 0.05])
        classes.append(1)
    boxes_a, classes_a = np.array(boxes), np.array(classes)
    assert eac.roi_window(boxes_a, classes_a, w, h) == _window_from_crop_to_roi(boxes_a, classes_a, w, h)


def test_roi_window_requires_a_mango_box() -> None:
    with pytest.raises(ValueError, match="class 0"):
        eac.roi_window(np.array([_norm((100, 100, 200, 200))]), np.array([1]), W, H)


# --------------------------------------------------------------------------
# Box transform
# --------------------------------------------------------------------------


def test_transform_known_box_maps_to_known_coordinates() -> None:
    window = (100, 100, 420, 420)  # 320 px side -> 2x upscale to 640
    boxes = np.array([_norm((220, 240, 300, 280))])  # centre (260, 260), 80x40
    out = eac.transform_boxes(boxes, np.array([1]), W, H, window)
    np.testing.assert_allclose(out.boxes, [[0.5, 0.5, 0.25, 0.125]])
    np.testing.assert_allclose(out.boxes * 640, [[320, 320, 160, 80]])
    assert out.kept.tolist() == [True]
    assert out.visible.tolist() == [1.0]


def test_transform_box_half_outside_is_clipped_at_exactly_fifty_percent() -> None:
    window = (100, 100, 420, 420)
    # x 380..460 -> 380..420 visible: exactly 50 % of the area. Kept and clipped.
    boxes = np.array([_norm((380, 200, 460, 240))])
    out = eac.transform_boxes(boxes, np.array([1]), W, H, window)
    assert out.kept.tolist() == [True]
    assert out.visible[0] == pytest.approx(0.5)
    # clipped box: x 280..320 of the 320 window, y 100..140 -> cx 300/320, w 40/320.
    np.testing.assert_allclose(out.boxes, [[300 / 320, 120 / 320, 40 / 320, 40 / 320]])


def test_transform_box_mostly_outside_is_dropped() -> None:
    window = (100, 100, 420, 420)
    boxes = np.array([_norm((385, 200, 465, 240))])  # 35/80 = 43.75 % visible
    out = eac.transform_boxes(boxes, np.array([1]), W, H, window)
    assert out.kept.tolist() == [False]
    assert out.visible[0] == pytest.approx(0.4375)
    assert out.boxes.shape == (0, 4)
    assert out.classes.shape == (0,)


def test_transform_box_fully_outside_is_dropped_with_zero_visibility() -> None:
    window = (100, 100, 420, 420)
    boxes = np.array([_norm((600, 300, 640, 340))])
    out = eac.transform_boxes(boxes, np.array([1]), W, H, window)
    assert out.kept.tolist() == [False]
    assert out.visible.tolist() == [0.0]


def test_transform_min_visible_is_configurable() -> None:
    window = (100, 100, 420, 420)
    boxes = np.array([_norm((385, 200, 465, 240))])  # 43.75 % visible
    out = eac.transform_boxes(boxes, np.array([1]), W, H, window, min_visible=0.4)
    assert out.kept.tolist() == [True]


def test_transform_class_zero_always_survives_even_when_clamping_cuts_it() -> None:
    # Window narrower than the mango (a clamped square): only 20 % of it is inside.
    window = (500, 100, 600, 200)
    mango = _norm((580, 100, 680, 200))
    damage = _norm((520, 120, 540, 140))
    out = eac.transform_boxes(np.array([mango, damage]), np.array([0, 1]), W, H, window)
    assert out.kept.tolist() == [True, True]
    assert out.visible[0] == pytest.approx(0.2)
    # clipped mango: x 80..100 of a 100 px window.
    np.testing.assert_allclose(out.boxes[0], [0.9, 0.5, 0.2, 1.0])


def test_transform_keeps_input_order_and_classes() -> None:
    window = (100, 100, 420, 420)
    boxes = np.array(
        [
            _norm((220, 240, 300, 280)),  # damage, inside
            _norm((130, 130, 400, 400)),  # mango
            _norm((700, 300, 740, 340)),  # damage, outside -> dropped
            _norm((300, 300, 340, 340)),  # damage, inside
        ]
    )
    classes = np.array([1, 0, 1, 1])
    out = eac.transform_boxes(boxes, classes, W, H, window)
    assert out.kept.tolist() == [True, True, False, True]
    assert out.classes.tolist() == [1, 0, 1]
    assert len(out.boxes) == 3


def test_transform_handles_a_label_file_without_damage() -> None:
    out = eac.transform_boxes(
        np.array([_norm((130, 130, 400, 400))]), np.array([0]), W, H, (100, 100, 420, 420)
    )
    assert out.classes.tolist() == [0]


# --------------------------------------------------------------------------
# Resize
# --------------------------------------------------------------------------


def _noise(shape: tuple[int, ...], seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, size=shape, dtype=np.uint8)


def test_resize_upscales_with_lanczos() -> None:
    img = _noise((320, 320, 3))
    out = eac.resize_to_square(img, 640)
    assert out.shape == (640, 640, 3)
    np.testing.assert_array_equal(out, cv2.resize(img, (640, 640), interpolation=cv2.INTER_LANCZOS4))


def test_resize_downscales_with_area() -> None:
    img = _noise((1000, 1000))
    out = eac.resize_to_square(img, 640)
    assert out.shape == (640, 640)
    np.testing.assert_array_equal(out, cv2.resize(img, (640, 640), interpolation=cv2.INTER_AREA))


def test_resize_exact_size_is_untouched() -> None:
    img = _noise((640, 640))
    np.testing.assert_array_equal(eac.resize_to_square(img, 640), img)


def test_resize_rejects_a_non_square_crop() -> None:
    with pytest.raises(ValueError, match="square"):
        eac.resize_to_square(_noise((100, 120)), 640)


# --------------------------------------------------------------------------
# Homography selection by Lima local date
# --------------------------------------------------------------------------


def test_homography_for_a_lot6_capture_is_the_lot6_file() -> None:
    assert eac.homography_for_stem(_stem("2026-10-18 10:00:00")) == "matriz_homografia_lot6_20261018.npy"


def test_homography_for_earlier_captures_is_aruco() -> None:
    assert eac.homography_for_stem(_stem("2026-09-27 15:30:00")) == "matriz_homografia_aruco.npy"
    assert eac.homography_for_stem(_stem("2026-05-24 08:00:00")) == "matriz_homografia_aruco.npy"


def test_homography_selection_uses_lima_time_not_utc() -> None:
    # 23:30 on 10-17 Lima is 04:30 on 10-18 UTC: still the old rig.
    assert eac.homography_for_stem(_stem("2026-10-17 23:30:00")) == "matriz_homografia_aruco.npy"
    # 00:30 on 10-18 Lima is 05:30 UTC: already the new rig.
    assert eac.homography_for_stem(_stem("2026-10-18 00:30:00")) == "matriz_homografia_lot6_20261018.npy"


def test_homography_for_real_project_timestamps() -> None:
    assert eac.homography_for_stem("mango_rgb_1792348876") == "matriz_homografia_lot6_20261018.npy"
    assert eac.homography_for_stem("mango_rgb_1790524207") == "matriz_homografia_aruco.npy"


def test_homography_after_lot6_fails_loudly() -> None:
    with pytest.raises(ValueError, match="no homography"):
        eac.homography_for_stem(_stem("2026-10-19 09:00:00"))


def test_lot_homography_consistency_rejects_a_lot_split_across_matrices() -> None:
    stems = {"lot1": ["a", "b"], "lot2": ["c"]}
    good = {"a": "H1.npy", "b": "H1.npy", "c": "H2.npy"}
    eac.check_lot_homography_consistency(stems, good)
    bad = dict(good, b="H2.npy")
    with pytest.raises(ValueError, match="lot1"):
        eac.check_lot_homography_consistency(stems, bad)


# --------------------------------------------------------------------------
# NIR loader parity constants
# --------------------------------------------------------------------------


def test_nir_border_value_matches_what_the_training_loader_passes() -> None:
    loader_default = inspect.signature(YOLODataset.__init__).parameters["nir_mean"].default
    config_default = TrainingConfig.__dataclass_fields__["nir_mean"].default
    assert eac.NIR_MEAN == loader_default == config_default
    assert eac.nir_border_value() == int(loader_default * 255) == 14


def test_identity_homography_leaves_a_registered_nir_unchanged() -> None:
    # README promise: with `homographies/identity.npy` the repo loader does not move the NIR.
    nir = _noise((180, 320))
    out = register_nir_to_rgb(nir, np.eye(3), (320, 180), eac.nir_border_value())
    np.testing.assert_array_equal(out, nir)


# --------------------------------------------------------------------------
# Labels
# --------------------------------------------------------------------------


def test_parse_label_text_reads_classes_and_boxes() -> None:
    boxes, classes = eac.parse_label_text("0 0.5 0.5 0.25 0.5\n1 0.4 0.4 0.1 0.1\n\n")
    assert classes.tolist() == [0, 1]
    np.testing.assert_allclose(boxes, [[0.5, 0.5, 0.25, 0.5], [0.4, 0.4, 0.1, 0.1]])


def test_parse_label_text_rejects_malformed_lines() -> None:
    with pytest.raises(ValueError, match="line 2"):
        eac.parse_label_text("0 0.5 0.5 0.25 0.5\n1 0.4 0.4\n")
    with pytest.raises(ValueError, match="line 1"):
        eac.parse_label_text("zero 0.5 0.5 0.25 0.5\n")
    with pytest.raises(ValueError, match="line 1"):
        eac.parse_label_text("0 0.5 0.5 0.0 0.5\n")  # zero width


def test_format_label_text_uses_six_decimals_and_a_trailing_newline() -> None:
    text = eac.format_label_text(np.array([[0.5, 0.25, 0.125, 0.0625]]), np.array([1]))
    assert text == "1 0.500000 0.250000 0.125000 0.062500\n"
    assert eac.format_label_text(np.zeros((0, 4)), np.zeros((0,), dtype=int)) == ""


# --------------------------------------------------------------------------
# Deterministic zip writer
# --------------------------------------------------------------------------


def _tiny_tree(root: Path, mtime: int) -> None:
    (root / "images").mkdir(parents=True)
    files = {
        "images/a.jpg": b"\xff\xd8jpeg-bytes",
        "images/b.png": b"\x89PNGpng-bytes",
        "labels.txt": b"0 0.5 0.5 0.2 0.2\n" * 50,
        "z.csv": b"stem,split\nx,train\n",
    }
    for name, data in files.items():
        path = root / name
        path.write_bytes(data)
        os.utime(path, (mtime, mtime))


def _zip_tree(root: Path, out: Path) -> None:
    with eac.DeterministicZip(out) as archive:
        for path in sorted(p for p in root.rglob("*") if p.is_file()):
            archive.add_file(f"top/{path.relative_to(root).as_posix()}", path)


def test_zip_writer_is_byte_identical_across_runs_mtimes_and_locations(tmp_path: Path) -> None:
    a, b = tmp_path / "a", tmp_path / "b"
    _tiny_tree(a, mtime=1_000_000_000)
    _tiny_tree(b, mtime=1_700_000_000)
    _zip_tree(a, tmp_path / "one.zip")
    _zip_tree(b, tmp_path / "two.zip")
    assert (tmp_path / "one.zip").read_bytes() == (tmp_path / "two.zip").read_bytes()


def test_zip_writer_uses_fixed_timestamp_attributes_and_given_order(tmp_path: Path) -> None:
    root = tmp_path / "t"
    _tiny_tree(root, mtime=1_000_000_000)
    _zip_tree(root, tmp_path / "out.zip")
    with zipfile.ZipFile(tmp_path / "out.zip") as archive:
        infos = archive.infolist()
        assert [i.filename for i in infos] == [
            "top/images/a.jpg",
            "top/images/b.png",
            "top/labels.txt",
            "top/z.csv",
        ]
        assert {i.date_time for i in infos} == {(2026, 10, 21, 0, 0, 0)}
        assert len({i.external_attr for i in infos}) == 1
        assert {i.create_system for i in infos} == {3}
        by_name = {i.filename: i.compress_type for i in infos}
        # Already-compressed media is stored, text is deflated.
        assert by_name["top/images/a.jpg"] == zipfile.ZIP_STORED
        assert by_name["top/images/b.png"] == zipfile.ZIP_STORED
        assert by_name["top/labels.txt"] == zipfile.ZIP_DEFLATED
        assert archive.testzip() is None


def test_zip_writer_rejects_duplicate_entries(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="duplicate"):
        with eac.DeterministicZip(tmp_path / "d.zip") as archive:
            archive.add_bytes("top/x.txt", b"1")
            archive.add_bytes("top/x.txt", b"2")
    assert not (tmp_path / "d.zip").exists()  # an aborted write leaves no zip behind


def test_zip_writer_publishes_atomically(tmp_path: Path) -> None:
    out = tmp_path / "ok.zip"
    with eac.DeterministicZip(out) as archive:
        archive.add_bytes("top/x.txt", b"1")
        assert not out.exists()  # still writing to a temporary name
    assert out.is_file()
    assert [p.name for p in tmp_path.iterdir()] == ["ok.zip"]


# --------------------------------------------------------------------------
# Primary checkout
# --------------------------------------------------------------------------


@pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")
def test_primary_checkout_root_resolves_from_a_linked_worktree(tmp_path: Path) -> None:
    main = tmp_path / "main"
    main.mkdir()

    def git(*args: str, cwd: Path = main) -> None:
        subprocess.run(
            ["git", "-c", "user.name=t", "-c", "user.email=t@example.com", *args],
            cwd=cwd,
            check=True,
            capture_output=True,
        )

    git("init")
    git("commit", "--allow-empty", "-m", "init")
    linked = tmp_path / "linked"
    git("worktree", "add", str(linked), "-b", "side")
    assert eac.primary_checkout_root(linked) == main.resolve()
    assert eac.primary_checkout_root(main) == main.resolve()


@pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")
def test_default_export_zip_is_git_ignored() -> None:
    path = eac.default_output_path(Path("."), 252).as_posix()  # data/exports/mango_corpus_v4_252pairs.zip
    result = subprocess.run(["git", "check-ignore", "-q", path], cwd=eac.PROJECT_ROOT)
    assert result.returncode == 0, f"{path} would show up as untracked; add data/exports/ to .gitignore"


# --------------------------------------------------------------------------
# Pair resolution (preflight)
# --------------------------------------------------------------------------


def _write_manifest_tree(root: Path, manifest: dict[str, list[str]], *, unlabelled=(), no_nir=()) -> dict:
    """Create empty rgb/nir/label files for a manifest; returns the directory triple."""
    rgb, nir, labels = root / "rgb", root / "nir", root / "labels"
    rgb.mkdir(parents=True)
    nir.mkdir()
    for split, stems in manifest.items():
        (labels / split).mkdir(parents=True)
        for stem in stems:
            (rgb / f"{stem}.jpg").write_bytes(b"x")
            if stem not in no_nir:
                (nir / f"{stem.replace('_rgb', '_nir')}.jpg").write_bytes(b"x")
            if stem not in unlabelled:
                (labels / split / f"{stem}.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    return {"rgb_dir": rgb, "nir_dir": nir, "labels_dir": labels}


def _small_manifest() -> dict[str, list[str]]:
    stems = [_stem("2026-09-27 10:00:00", i) for i in range(6)]
    return {"train": stems[:4], "val": stems[4:5], "test": stems[5:]}


def test_resolve_pairs_lists_every_missing_file_before_anything_is_written(tmp_path: Path) -> None:
    manifest = _small_manifest()
    dirs = _write_manifest_tree(
        tmp_path, manifest, unlabelled=[manifest["train"][0], manifest["test"][0]], no_nir=[manifest["val"][0]]
    )
    with pytest.raises(eac.MissingInputsError) as caught:
        eac.resolve_pairs(manifest, **dirs)
    message = str(caught.value)
    assert f"{manifest['train'][0]}.txt" in message
    assert f"{manifest['test'][0]}.txt" in message
    assert manifest["val"][0].replace("_rgb", "_nir") + ".jpg" in message
    assert len(caught.value.missing) == 3


def test_resolve_pairs_only_labelled_skips_unlabelled_stems_but_not_missing_images(tmp_path: Path) -> None:
    manifest = _small_manifest()
    skipped_stem = manifest["train"][0]
    dirs = _write_manifest_tree(tmp_path, manifest, unlabelled=[skipped_stem])
    pairs, skipped = eac.resolve_pairs(manifest, **dirs, only_labelled=True)
    assert skipped == [skipped_stem]
    assert skipped_stem not in {p.stem for p in pairs}
    assert len(pairs) == 5

    shutil.rmtree(dirs["nir_dir"])
    dirs["nir_dir"].mkdir()
    with pytest.raises(eac.MissingInputsError):
        eac.resolve_pairs(manifest, **dirs, only_labelled=True)


def test_resolve_pairs_is_sorted_by_split_then_stem(tmp_path: Path) -> None:
    stems = [_stem("2026-09-27 10:00:00", i) for i in range(5)]
    manifest = {"train": [stems[2], stems[0], stems[3]], "val": [stems[1]], "test": [stems[4]]}
    dirs = _write_manifest_tree(tmp_path, manifest)
    pairs, skipped = eac.resolve_pairs(manifest, **dirs)
    assert skipped == []
    assert [(p.split, p.stem) for p in pairs] == [
        ("train", stems[0]),
        ("train", stems[2]),
        ("train", stems[3]),
        ("val", stems[1]),
        ("test", stems[4]),
    ]


def test_load_manifest_rejects_duplicates_and_unknown_splits(tmp_path: Path) -> None:
    path = tmp_path / "splits.json"
    path.write_text(json.dumps({"train": ["mango_rgb_1", "mango_rgb_1"], "val": [], "test": []}))
    with pytest.raises(ValueError, match="mango_rgb_1"):
        eac.load_manifest(path)
    path.write_text(json.dumps({"train": ["mango_rgb_1"], "val": [], "test": [], "extra": ["mango_rgb_2"]}))
    with pytest.raises(ValueError, match="extra"):
        eac.load_manifest(path)
    path.write_text(json.dumps({"train": ["mango_rgb_1"], "val": ["mango_rgb_1"], "test": []}))
    with pytest.raises(ValueError, match="mango_rgb_1"):
        eac.load_manifest(path)


def test_assign_lots_is_computed_on_the_whole_manifest() -> None:
    # Lot 2 has 5 stems, lot 1 has 6: both survive the min-size merge only when
    # every stem is counted, so a labelled-only filter must not change lot ids.
    day1 = [_stem("2026-09-27 10:00:00", i) for i in range(6)]
    day2 = [_stem("2026-10-18 10:00:00", i) for i in range(5)]
    lots = eac.assign_lots(day1 + day2)
    assert {lots[s].lot_id for s in day1} == {"lot1"}
    assert {lots[s].lot_id for s in day2} == {"lot2"}
    assert lots[day2[0]].days == ["2026-10-18"]


# --------------------------------------------------------------------------
# End to end on a synthetic corpus
# --------------------------------------------------------------------------

IMG_W, IMG_H = 320, 180
H_ARUCO = np.array([[1.0, 0.0, 7.0], [0.0, 1.0, 3.0], [0.0, 0.0, 1.0]])
H_LOT6 = np.array([[1.0, 0.0, 3.0], [0.0, 1.0, 9.0], [0.0, 0.0, 1.0]])
OLD_DAY, NEW_DAY = "2026-09-27 10:00:00", "2026-10-18 10:00:00"

MANGO_PX = (96, 45, 224, 135)  # 128x90 in the 320x180 frame
DAMAGE_INSIDE = (120, 70, 150, 100)
DAMAGE_OUTSIDE = (5, 5, 25, 25)  # in the frame, far from the mango window


def _label_text() -> str:
    rows = [(0, MANGO_PX), (1, DAMAGE_INSIDE), (1, DAMAGE_OUTSIDE)]
    return "".join(
        f"{cls} " + " ".join(f"{v:.6f}" for v in _norm(box, IMG_W, IMG_H)) + "\n" for cls, box in rows
    )


class Corpus:
    """A tiny on-disk data root + homography dir for the exporter."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.data_root = root / "primary"
        self.homography_dir = root / "notebooks"
        yolo = self.data_root / "data" / "annotations" / "yolo"
        self.rgb_dir = self.data_root / "data" / "cache" / "mango" / "rgb"
        self.nir_dir = self.data_root / "data" / "cache" / "mango" / "nir"
        self.labels_dir = yolo / "labels"
        self.manifest_path = yolo / "splits.json"
        self.homography_dir.mkdir(parents=True)
        np.save(self.homography_dir / "matriz_homografia_aruco.npy", H_ARUCO)
        np.save(self.homography_dir / "matriz_homografia_lot6_20261018.npy", H_LOT6)

        old = [_stem(OLD_DAY, i * 60) for i in range(5)]
        new = [_stem(NEW_DAY, i * 60) for i in range(5)]
        self.manifest = {
            "train": [old[0], old[1], old[2], new[0], new[1], new[2]],
            "val": [old[3], new[3]],
            "test": [old[4], new[4]],
        }
        self.old, self.new = old, new
        self.manifest_path.parent.mkdir(parents=True)
        self.manifest_path.write_text(json.dumps(self.manifest))
        self.rgb_dir.mkdir(parents=True)
        self.nir_dir.mkdir(parents=True)
        for split, stems in self.manifest.items():
            (self.labels_dir / split).mkdir(parents=True)
            for stem in stems:
                seed = int(stem.rsplit("_", 1)[1])
                rgb = _noise((IMG_H, IMG_W, 3), seed)
                nir = _noise((IMG_H, IMG_W, 3), seed + 1)
                assert cv2.imwrite(str(self.rgb_dir / f"{stem}.jpg"), rgb)
                assert cv2.imwrite(str(self.nir_dir / f"{stem.replace('_rgb', '_nir')}.jpg"), nir)
                (self.labels_dir / split / f"{stem}.txt").write_text(_label_text())

    def split_of(self, stem: str) -> str:
        return next(split for split, stems in self.manifest.items() if stem in stems)

    def run(self, output: Path, *extra: str) -> int:
        return eac.main(
            [
                "--data-root", str(self.data_root),
                "--homography-dir", str(self.homography_dir),
                "--output", str(output),
                *extra,
            ]
        )


@pytest.fixture()
def corpus(tmp_path: Path) -> Corpus:
    return Corpus(tmp_path)


@pytest.fixture()
def exported(corpus: Corpus, tmp_path: Path) -> tuple[Corpus, zipfile.ZipFile, Path]:
    out = tmp_path / "out" / "corpus.zip"
    assert corpus.run(out) == 0
    return corpus, zipfile.ZipFile(out), out


def test_export_zip_has_one_top_level_folder_and_the_documented_layout(exported) -> None:
    corpus, archive, _ = exported
    names = archive.namelist()
    assert {n.split("/")[0] for n in names} == {"mango_corpus_v4"}
    assert names == sorted(set(names), key=names.index)  # no duplicates
    root = "mango_corpus_v4/"
    for shared in (
        "README.md", "manifest.csv", "splits.json",
        "data_rgb.yaml", "data_nir.yaml", "data_rgb_roi640.yaml", "data_nir_roi640.yaml",
        "homographies/identity.npy", "homographies/matriz_homografia_aruco.npy",
        "homographies/matriz_homografia_lot6_20261018.npy", "roi640/crop_windows.csv",
    ):
        assert root + shared in names, shared
    for split, stems in corpus.manifest.items():
        for stem in stems:
            nir = stem.replace("_rgb", "_nir")
            for entry in (
                f"full_aligned/images/rgb/{split}/{stem}.jpg",
                f"full_aligned/images/nir/{split}/{nir}.png",
                f"full_aligned/labels/rgb/{split}/{stem}.txt",
                f"full_aligned/labels/nir/{split}/{nir}.txt",
                f"roi640/images/rgb/{split}/{stem}.jpg",
                f"roi640/images/nir/{split}/{nir}.png",
                f"roi640/labels/rgb/{split}/{stem}.txt",
                f"roi640/labels/nir/{split}/{nir}.txt",
            ):
                assert root + entry in names, entry
    assert len(names) == 8 * 10 + 11  # 8 files per pair, 11 shared files
    # The old shared `labels/<split>/` layout is gone.
    assert not [n for n in names if re.search(r"/labels/(train|val|test)/", n)]


def _ultralytics_label_path(image: str) -> str:
    """ultralytics.data.utils.img2label_paths for one posix path: last /images/ -> /labels/."""
    return "/labels/".join(image.rsplit("/images/", 1)).rsplit(".", 1)[0] + ".txt"


def test_every_image_has_its_label_under_the_ultralytics_substituted_path(exported) -> None:
    _, archive, _ = exported
    names = set(archive.namelist())
    images = [n for n in names if "/images/" in n]
    assert len(images) == 40
    for image in images:
        assert _ultralytics_label_path(image) in names, image
    labels = [n for n in names if "/labels/" in n]
    assert len(labels) == 40  # and no label without an image


def test_export_full_aligned_rgb_and_labels_are_byte_copies(exported) -> None:
    corpus, archive, _ = exported
    stem = corpus.new[1]
    split = corpus.split_of(stem)
    assert archive.read(f"mango_corpus_v4/full_aligned/images/rgb/{split}/{stem}.jpg") == (
        corpus.rgb_dir / f"{stem}.jpg"
    ).read_bytes()
    source_label = (corpus.labels_dir / split / f"{stem}.txt").read_bytes()
    nir_stem = stem.replace("_rgb", "_nir")
    assert archive.read(f"mango_corpus_v4/full_aligned/labels/rgb/{split}/{stem}.txt") == source_label
    # The NIR is registered to the RGB frame, so its labels are the very same bytes.
    assert archive.read(f"mango_corpus_v4/full_aligned/labels/nir/{split}/{nir_stem}.txt") == source_label
    assert archive.read("mango_corpus_v4/splits.json") == corpus.manifest_path.read_bytes()


def _decode(archive: zipfile.ZipFile, name: str, flags: int = cv2.IMREAD_UNCHANGED) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(archive.read(name), np.uint8), flags)


def _expected_registered_nir(corpus: Corpus, stem: str, matrix: np.ndarray) -> np.ndarray:
    nir = cv2.imread(str(corpus.nir_dir / f"{stem.replace('_rgb', '_nir')}.jpg"), cv2.IMREAD_GRAYSCALE)
    return register_nir_to_rgb(nir, matrix, (IMG_W, IMG_H), 14)


def test_export_nir_is_registered_with_the_homography_of_its_lot(exported) -> None:
    corpus, archive, _ = exported
    for stems, matrix, other in ((corpus.old, H_ARUCO, H_LOT6), (corpus.new, H_LOT6, H_ARUCO)):
        stem = stems[0]
        split = corpus.split_of(stem)
        name = f"mango_corpus_v4/full_aligned/images/nir/{split}/{stem.replace('_rgb', '_nir')}.png"
        got = _decode(archive, name)
        assert got.ndim == 2 and got.dtype == np.uint8  # single-channel 8-bit
        np.testing.assert_array_equal(got, _expected_registered_nir(corpus, stem, matrix))
        assert not np.array_equal(got, _expected_registered_nir(corpus, stem, other))


def test_export_nir_border_is_filled_with_the_loader_pad_value(exported) -> None:
    corpus, archive, _ = exported
    stem = corpus.old[0]
    name = f"mango_corpus_v4/full_aligned/images/nir/{corpus.split_of(stem)}/{stem.replace('_rgb', '_nir')}.png"
    got = _decode(archive, name)
    # registered(p) = NIR(H p) with H = translate(+7, +3): the last 7 columns and 3 rows have no NIR source.
    assert (got[:, -6:] == 14).all() and (got[-2:, :] == 14).all()
    assert not (got[:, :-8] == 14).all()  # while the interior is real data


def test_export_roi640_images_are_square_640_and_use_the_same_window(exported) -> None:
    corpus, archive, _ = exported
    rows = list(csv.DictReader(io.StringIO(archive.read("mango_corpus_v4/roi640/crop_windows.csv").decode())))
    assert len(rows) == 10
    assert list(rows[0]) == ["stem", "split", "lot", "x0", "y0", "x1", "y1", "side_px", "scale_to_640"]
    by_stem = {r["stem"]: r for r in rows}
    for stem in corpus.old + corpus.new:
        row = by_stem[stem]
        window = (int(row["x0"]), int(row["y0"]), int(row["x1"]), int(row["y1"]))
        side = int(row["side_px"])
        assert window[2] - window[0] == window[3] - window[1] == side
        assert float(row["scale_to_640"]) == pytest.approx(640 / side, abs=5e-7)
        # The window is the square of the crop_to_roi window of the label file.
        boxes, classes = eac.parse_label_text((corpus.labels_dir / row["split"] / f"{stem}.txt").read_text())
        assert window == eac.square_window(eac.roi_window(boxes, classes, IMG_W, IMG_H), IMG_W, IMG_H)

        split = row["split"]
        rgb = _decode(archive, f"mango_corpus_v4/roi640/images/rgb/{split}/{stem}.jpg")
        nir = _decode(archive, f"mango_corpus_v4/roi640/images/nir/{split}/{stem.replace('_rgb', '_nir')}.png")
        assert rgb.shape == (640, 640, 3) and nir.shape == (640, 640)

    # RGB crop = the source RGB cut with the row's window, resized, JPEG quality 95.
    rgb_stem = corpus.old[3]
    row = by_stem[rgb_stem]
    x0, y0, x1, y1 = (int(row[k]) for k in ("x0", "y0", "x1", "y1"))
    source = cv2.imread(str(corpus.rgb_dir / f"{rgb_stem}.jpg"))
    expected_jpeg = cv2.imencode(
        ".jpg", eac.resize_to_square(np.ascontiguousarray(source[y0:y1, x0:x1])), [cv2.IMWRITE_JPEG_QUALITY, 95]
    )[1].tobytes()
    assert archive.read(f"mango_corpus_v4/roi640/images/rgb/{row['split']}/{rgb_stem}.jpg") == expected_jpeg

    # NIR crop = the registered NIR cut with the row's window and resized.
    stem = corpus.new[2]
    row = by_stem[stem]
    x0, y0, x1, y1 = (int(row[k]) for k in ("x0", "y0", "x1", "y1"))
    registered = _expected_registered_nir(corpus, stem, H_LOT6)
    expected = eac.resize_to_square(np.ascontiguousarray(registered[y0:y1, x0:x1]), 640)
    got = _decode(archive, f"mango_corpus_v4/roi640/images/nir/{row['split']}/{stem.replace('_rgb', '_nir')}.png")
    np.testing.assert_array_equal(got, expected)


def test_export_roi640_labels_are_renormalized_and_drop_the_distant_damage(exported) -> None:
    corpus, archive, _ = exported
    stem = corpus.old[1]
    split = corpus.split_of(stem)
    roi_label = archive.read(f"mango_corpus_v4/roi640/labels/rgb/{split}/{stem}.txt")
    nir_label = archive.read(f"mango_corpus_v4/roi640/labels/nir/{split}/{stem.replace('_rgb', '_nir')}.txt")
    assert nir_label == roi_label  # same crop window for both modalities
    boxes, classes = eac.parse_label_text(roi_label.decode())
    assert classes.tolist() == [0, 1]  # the far damage box fell outside the window
    assert ((boxes >= 0) & (boxes <= 1)).all()
    row = next(
        r
        for r in csv.DictReader(io.StringIO(archive.read("mango_corpus_v4/roi640/crop_windows.csv").decode()))
        if r["stem"] == stem
    )
    x0, y0, side = int(row["x0"]), int(row["y0"]), int(row["side_px"])
    mango_cx = (MANGO_PX[0] + MANGO_PX[2]) / 2
    assert boxes[0][0] == pytest.approx((mango_cx - x0) / side, abs=1e-6)
    assert boxes[0][2] == pytest.approx((MANGO_PX[2] - MANGO_PX[0]) / side, abs=1e-6)


def test_export_manifest_csv_lists_lot_date_homography_and_box_counts(exported) -> None:
    corpus, archive, _ = exported
    rows = list(csv.DictReader(io.StringIO(archive.read("mango_corpus_v4/manifest.csv").decode())))
    assert list(rows[0]) == [
        "stem", "split", "lot", "capture_local_date", "homography_file", "n_mango", "n_damage",
    ]
    assert len(rows) == 10
    assert [r["split"] for r in rows] == ["train"] * 6 + ["val"] * 2 + ["test"] * 2
    by_stem = {r["stem"]: r for r in rows}
    old, new = by_stem[corpus.old[0]], by_stem[corpus.new[0]]
    assert (old["lot"], old["capture_local_date"]) == ("lot1", "2026-09-27")
    assert (new["lot"], new["capture_local_date"]) == ("lot2", "2026-10-18")
    assert old["homography_file"] == "homographies/matriz_homografia_aruco.npy"
    assert new["homography_file"] == "homographies/matriz_homografia_lot6_20261018.npy"
    assert (old["n_mango"], old["n_damage"]) == ("1", "2")


def test_export_shared_files_are_valid(exported) -> None:
    _, archive, _ = exported
    root = "mango_corpus_v4/"
    yamls = (
        ("data_rgb.yaml", "full_aligned", "rgb"),
        ("data_nir.yaml", "full_aligned", "nir"),
        ("data_rgb_roi640.yaml", "roi640", "rgb"),
        ("data_nir_roi640.yaml", "roi640", "nir"),
    )
    assert sorted(n[len(root):] for n in archive.namelist() if n.endswith(".yaml")) == sorted(y[0] for y in yamls)
    for name, folder, modality in yamls:
        config = yaml.safe_load(archive.read(root + name))
        assert config["names"] == {0: "mango", 1: "damage"}
        for split in ("train", "val", "test"):
            assert config[split] == f"{folder}/images/{modality}/{split}"
            # Ultralytics resolves labels by swapping /images/ for /labels/ in this path.
            assert _ultralytics_label_path(f"{config[split]}/x.png") == f"{folder}/labels/{modality}/{split}/x.txt"
        assert "path" not in config  # Ultralytics resolves a relative `path` against its own datasets dir

    identity = np.load(io.BytesIO(archive.read(root + "homographies/identity.npy")))
    np.testing.assert_array_equal(identity, np.eye(3))
    assert identity.dtype == np.float64
    shipped = np.load(io.BytesIO(archive.read(root + "homographies/matriz_homografia_lot6_20261018.npy")))
    np.testing.assert_array_equal(shipped, H_LOT6)


def test_export_readme_documents_the_pitfalls(exported) -> None:
    _, archive, _ = exported
    readme = archive.read("mango_corpus_v4/README.md").decode()
    for needle in (
        "ALREADY registered",
        "homographies/identity.npy",
        "nir_homography_path",
        "roi_crop",
        "twice",
        "INTER_LANCZOS4",
        "INTER_AREA",
        "matriz_homografia_lot6_20261018.npy",
        "8-bit",
        "lot1",
        "lot2",
        "min_visible",
        "4 px right and 9 px below",
        "labels/rgb/<split>",
        "labels/nir/<split>",
        "3 channels",
        "Which folder to use",
        "Known gap",
        "data_nir.yaml",
        "data_nir_roi640.yaml",
    ):
        assert needle in readme, needle
    assert "cp -r" not in readme  # no manual copy step any more
    assert "--only-labelled" not in readme  # a complete corpus does not carry the subset note


def test_export_is_byte_identical_on_a_rerun(corpus: Corpus, tmp_path: Path) -> None:
    first, second = tmp_path / "first.zip", tmp_path / "second.zip"
    assert corpus.run(first) == 0
    assert corpus.run(second) == 0
    assert hashlib.sha256(first.read_bytes()).hexdigest() == hashlib.sha256(second.read_bytes()).hexdigest()


def test_export_prints_a_summary_with_dropped_boxes(corpus: Corpus, tmp_path: Path, capsys) -> None:
    assert corpus.run(tmp_path / "s.zip") == 0
    out = capsys.readouterr().out
    assert "train" in out and "val" in out and "test" in out
    assert "lot1" in out and "lot2" in out
    assert "dropped" in out.lower()
    assert corpus.old[0] in out  # the far damage box of this stem is listed


def test_export_fails_before_writing_anything_when_a_label_is_missing(corpus: Corpus, tmp_path: Path, capsys) -> None:
    stem = corpus.new[0]
    (corpus.labels_dir / corpus.split_of(stem) / f"{stem}.txt").unlink()
    out = tmp_path / "never.zip"
    assert corpus.run(out) != 0
    assert not out.exists()
    assert list(tmp_path.glob("*.zip")) == []
    assert f"{stem}.txt" in capsys.readouterr().err


def test_export_only_labelled_limits_the_corpus_and_keeps_lot_ids(corpus: Corpus, tmp_path: Path, capsys) -> None:
    stem = corpus.new[0]
    (corpus.labels_dir / corpus.split_of(stem) / f"{stem}.txt").unlink()
    out = tmp_path / "subset.zip"
    assert corpus.run(out, "--only-labelled") == 0
    with zipfile.ZipFile(out) as archive:
        assert not [n for n in archive.namelist() if stem in n]
        rows = list(csv.DictReader(io.StringIO(archive.read("mango_corpus_v4/manifest.csv").decode())))
        assert len(rows) == 9
        # Lot ids come from the full manifest: the lot of the remaining 10-18 stems is still lot2.
        assert {r["lot"] for r in rows if r["capture_local_date"] == "2026-10-18"} == {"lot2"}
        readme = archive.read("mango_corpus_v4/README.md").decode()
        assert "--only-labelled" in readme and stem in readme  # a subset corpus says so, and what it left out
        # The shipped splits.json lists only what is in the zip.
        shipped = json.loads(archive.read("mango_corpus_v4/splits.json"))
        assert sum(len(v) for v in shipped.values()) == 9 and stem not in sum(shipped.values(), [])
    assert stem in capsys.readouterr().out  # the skipped stem is reported


def test_default_output_name_counts_the_exported_pairs() -> None:
    assert eac.default_output_path(Path("/data/root"), 252).as_posix() == (
        "/data/root/data/exports/mango_corpus_v4_252pairs.zip"
    )


def test_export_rejects_a_nir_whose_size_differs_from_the_rgb(corpus: Corpus, tmp_path: Path, capsys) -> None:
    stem = corpus.old[2]
    path = corpus.nir_dir / f"{stem.replace('_rgb', '_nir')}.jpg"
    cv2.imwrite(str(path), _noise((IMG_H // 2, IMG_W // 2, 3)))
    out = tmp_path / "bad.zip"
    assert corpus.run(out) != 0
    assert not out.exists()
    assert stem in capsys.readouterr().err
