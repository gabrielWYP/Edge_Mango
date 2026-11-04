"""`scripts/e0_detectability.py`: the boxes mode on synthetic datasets, and the centro mode it must not change."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from scripts import e0_detectability as e0
from src.data_pipeline.v4_selection import LATENT_HOURS
from src.evaluation import detectability as det

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "scripts" / "e0_detectability.py"

WIDTH, HEIGHT = 640, 480
AXES = np.array([150.0, 110.0])
CENTRE = np.array([WIDTH / 2, HEIGHT / 2])
BOX_SIZE = np.array([70.0, 60.0])
SKIN_BGR = np.array([40.0, 150.0, 175.0])
SKIN_NIR = 120.0
SHEET_BGR = np.array([215.0, 205.0, 190.0])
SHEET_NIR = 200.0
BLOTCH_SD = 8.0
ZONE_OFFSET = {"centro": (0, 0), "superior": (0, 0.55), "inferior": (0, -0.55), "izquierda": (0.55, 0), "derecha": (-0.55, 0)}
ZONES = tuple(ZONE_OFFSET)

CLEAN_HOUR, DRIFT_HOUR, DARK_HOUR = 0, 4, 12
DRIFT = 25.0  # every pixel of the photo, both cameras
DARK = 40.0  # inside the damage box only, every channel but blue
INDEX_FIELDS = ("stem", "timestamp", "lot", "mango_id", "cara", "ronda_h", "zona", "mango_x0", "mango_y0", "mango_x1",
                "mango_y1", "damage_x0", "damage_y0", "damage_x1", "damage_y1")


def _texture(rng):
    field = cv2.GaussianBlur(rng.normal(0, 1, (HEIGHT, WIDTH)).astype(np.float32), (0, 0), 9)
    return field / field.std() * BLOTCH_SD


def _render(texture, shift, box, drift, dark, rng):
    """(bgr, nir) of a fruit moved by `shift`, with `box` darkened by `dark` and the photo offset by `drift`."""
    yy, xx = np.mgrid[0:HEIGHT, 0:WIDTH]
    cx, cy = CENTRE + shift
    fruit = ((xx - cx) / AXES[0]) ** 2 + ((yy - cy) / AXES[1]) ** 2 <= 1
    moved = np.roll(texture, (int(shift[1]), int(shift[0])), axis=(0, 1))
    bgr = np.where(fruit[..., None], SKIN_BGR + moved[..., None], SHEET_BGR)
    nir = np.where(fruit, SKIN_NIR + moved, SHEET_NIR)
    if dark:
        x0, y0, x1, y1 = [int(v) for v in box]
        bgr[y0:y1, x0:x1, 1:] -= dark
        nir[y0:y1, x0:x1] -= dark
    bgr = bgr - drift + rng.normal(0, 1.5, bgr.shape)
    nir = nir - drift + rng.normal(0, 1.5, nir.shape)
    return np.clip(bgr, 1, 255).astype(np.uint8), np.clip(nir, 1, 255).astype(np.uint8)


def _write_photo(root, ts, bgr, nir):
    assert cv2.imwrite(str(root / "rgb" / f"mango_rgb_{ts}.jpg"), bgr, [cv2.IMWRITE_JPEG_QUALITY, 100])
    assert cv2.imwrite(str(root / "nir" / f"mango_nir_{ts}.png"), nir)


def _write_index(root, rows):
    with open(root / "index.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=INDEX_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def make_dataset(root, lots=(1, 2, 3), n_mangos=4, faces=(1, 2), seed=0):
    """A v4 dataset: clean at +0 h, a shared drift at +4 h and a darkened box at +12 h.

    Fruits move between photos and the indexed mango box has its own error, as the real mask has, so the
    measurement only lands on the box if the registration against `sano` is applied.
    """
    rng = np.random.default_rng(seed)
    (root / "rgb").mkdir(parents=True)
    (root / "nir").mkdir()
    rows, truth, ts, face_number = [], {}, 1000, 0
    for lot in lots:
        for mango in range(1, n_mangos + 1):
            mango_id = (lot - 1) * 25 + mango
            for cara in faces:
                zone = ZONES[face_number % len(ZONES)]
                face_number += 1
                texture = _texture(rng)
                for ronda in ("sano", CLEAN_HOUR, DRIFT_HOUR, DARK_HOUR):
                    shift = np.zeros(2) if ronda == "sano" else rng.integers(-5, 6, 2).astype(float)
                    middle = CENTRE + np.array(ZONE_OFFSET[zone]) * AXES + shift
                    box = np.r_[middle - BOX_SIZE / 2, middle + BOX_SIZE / 2]
                    bgr, nir = _render(texture, shift, box, DRIFT if ronda == DRIFT_HOUR else 0.0,
                                       DARK if ronda == DARK_HOUR else 0.0, rng)
                    ts += 1
                    _write_photo(root, ts, bgr, nir)
                    mango_box = np.r_[CENTRE + shift - AXES, CENTRE + shift + AXES] + np.tile(rng.integers(-6, 7, 2), 2)
                    row = {"stem": f"mango_rgb_{ts}", "timestamp": ts, "lot": lot, "mango_id": mango_id, "cara": cara,
                           "ronda_h": ronda, "zona": "" if ronda == "sano" else zone,
                           **dict(zip(INDEX_FIELDS[7:11], mango_box))}
                    if ronda != "sano":
                        row |= dict(zip(INDEX_FIELDS[11:15], box))
                    rows.append(row)
                    truth[(lot, mango_id, cara, ronda)] = {"shift": shift, "box": box, "mango_box": mango_box}
    _write_index(root, rows)
    return rows, truth


def run_script(*args):
    return subprocess.run([sys.executable, str(SCRIPT), *map(str, args)], capture_output=True, text=True, cwd=PROJECT_ROOT)


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    """The clean dataset, plus one photo without a box, one without a file and one face without `sano`."""
    root = tmp_path_factory.mktemp("dataset")
    rows, truth = make_dataset(root)
    other = next(r for r in rows if r["ronda_h"] == DARK_HOUR)
    extra = [
        {**other, "stem": "mango_rgb_9001", "timestamp": 9001, "ronda_h": 8,
         "damage_x0": "", "damage_y0": "", "damage_x1": "", "damage_y1": ""},
        {**other, "stem": "mango_rgb_9002", "timestamp": 9002, "ronda_h": 10},
        {**other, "stem": "mango_rgb_9003", "timestamp": 9003, "mango_id": 99, "cara": 1, "lot": 3},
    ]
    bgr, nir = _render(_texture(np.random.default_rng(1)), np.zeros(2), np.zeros(4), 0.0, 0.0, np.random.default_rng(2))
    _write_photo(root, 9001, bgr, nir)
    _write_photo(root, 9003, bgr, nir)
    _write_index(root, rows + extra)
    out = root / "out"
    result = run_script("--mode", "boxes", "--dataset", root, "--out", out, "--n-boot", 50, "--workers", 2)
    assert result.returncode == 0, result.stderr
    return {
        "root": root,
        "out": out,
        "truth": truth,
        "stdout": result.stdout,
        "faces": read_csv(out / "faces.csv"),
        "summary": json.loads((out / "summary.json").read_text(encoding="utf-8")),
    }


def _ok(faces, hour):
    return [r for r in faces if r["status"] == "ok" and r["hours"] == str(hour)]


def test_darkened_box_gives_a_known_delta_and_a_high_auroc(synthetic):
    rows = _ok(synthetic["faces"], DARK_HOUR)
    assert rows
    for row in rows:
        for ch in ("R", "G", "NIR"):  # the plan's formula: a darker box gives a negative delta
            assert float(row[f"{ch}_delta"]) == pytest.approx(-DARK / float(row[f"{ch}_sd"]), rel=0.2)
    hour = synthetic["summary"]["per_hour"][str(DARK_HOUR)]
    for modality in ("RGB", "NIR", "RGB+NIR"):
        assert hour["auroc"][modality]["auroc"] > 0.95
    assert hour["delta"]["NIR"]["mean"] < 0


def test_delta_is_scaled_by_the_sd_of_the_fruit_in_sano(synthetic):
    # The blotches are the only variation of the skin; the mask rim adds a little.
    for row in _ok(synthetic["faces"], DARK_HOUR):
        assert float(row["NIR_sd"]) == pytest.approx(BLOTCH_SD, rel=0.35)


def test_shared_drift_cancels_in_the_difference_of_differences(synthetic):
    rows = _ok(synthetic["faces"], DRIFT_HOUR)
    assert rows
    for row in rows:
        for ch in ("R", "G", "NIR"):
            assert float(row[f"{ch}_impact_t"]) - float(row[f"{ch}_impact_sano"]) == pytest.approx(-DRIFT, abs=5)
            assert float(row[f"{ch}_control_t"]) - float(row[f"{ch}_control_sano"]) == pytest.approx(-DRIFT, abs=5)
            assert abs(float(row[f"{ch}_delta"])) < 0.2 * DRIFT / float(row[f"{ch}_sd"])


def test_control_vs_control_is_about_chance_at_every_hour(synthetic):
    summary = synthetic["summary"]
    assert summary["hours"] == sorted(summary["hours"])
    assert {CLEAN_HOUR, DRIFT_HOUR, DARK_HOUR} <= set(summary["hours"])
    for hour in (CLEAN_HOUR, DRIFT_HOUR, DARK_HOUR):
        for modality in ("RGB", "NIR", "RGB+NIR"):
            assert summary["per_hour"][str(hour)]["negative_control"][modality]["auroc"] == pytest.approx(0.5, abs=0.2)


def test_auroc_without_damage_stays_near_chance(synthetic):
    # No damage at +0 h, and at +4 h only the shared drift: neither separates impact from control.
    for hour in (CLEAN_HOUR, DRIFT_HOUR):
        for modality in ("RGB", "NIR", "RGB+NIR"):
            assert synthetic["summary"]["per_hour"][str(hour)]["auroc"][modality]["auroc"] == pytest.approx(0.5, abs=0.2)


def test_box_reaches_the_sano_photo_through_the_registration(synthetic):
    truth = synthetic["truth"]
    rows = _ok(synthetic["faces"], DARK_HOUR)
    assert rows
    for row in rows:
        key = (int(row["lot"]), int(row["mango"]), int(row["face"]))
        sano, moment = truth[(*key, "sano")], truth[(*key, DARK_HOUR)]
        # Where the box of this photo lies in the frame of the sano photo, then in the sano crop.
        in_sano_frame = moment["box"] - np.tile(moment["shift"], 2)
        to_crop = e0.crop_matrix(*e0.box_centre(sano["mango_box"]), e0.crop_side(sano["mango_box"]))
        measured = [float(row[f"box_{k}"]) for k in ("x0", "y0", "x1", "y1")]
        assert measured == pytest.approx(det.map_box(in_sano_frame, to_crop), abs=2.5)


def test_photo_without_a_box_or_a_file_and_face_without_sano_are_excluded_and_counted(synthetic):
    assert synthetic["summary"]["exclusions"] == {"no_damage_box": 1, "no_fruit_or_file": 1, "no_sano": 1}
    statuses = {(r["mango"], r["moment"], r["status"]) for r in synthetic["faces"] if r["status"] != "ok"}
    assert ("99", str(DARK_HOUR), "no_sano") in statuses


def test_dataset_without_any_box_stops_with_a_clear_error(tmp_path):
    root = tmp_path / "dataset"
    rows, _ = make_dataset(root, lots=(1,), n_mangos=1, faces=(1,))
    for row in rows:
        for field in INDEX_FIELDS[11:15]:
            row[field] = ""
    _write_index(root, rows)
    result = run_script("--mode", "boxes", "--dataset", root, "--out", tmp_path / "out")
    assert result.returncode == 1
    assert "no photo has a damage box" in result.stderr
    assert not (tmp_path / "out" / "summary.json").exists()


def test_dataset_without_the_index_or_a_column_stops_with_a_clear_error(tmp_path):
    result = run_script("--mode", "boxes", "--dataset", tmp_path, "--out", tmp_path / "out")
    assert result.returncode == 1
    assert "index.csv" in result.stderr
    (tmp_path / "index.csv").write_text("stem,lot\nmango_rgb_1,1\n", encoding="utf-8")
    result = run_script("--mode", "boxes", "--dataset", tmp_path, "--out", tmp_path / "out")
    assert result.returncode == 1
    assert "damage_x0" in result.stderr


def test_report_files_hold_the_curves_the_table_and_the_figure(synthetic):
    out = synthetic["out"]
    assert set(synthetic["summary"]["lots"]) == {1, 2, 3}
    table = read_csv(out / "table.csv")
    keys = {(r["metric"], r["name"], r["hours"]) for r in table}
    for hour in (CLEAN_HOUR, DRIFT_HOUR, DARK_HOUR):
        for ch in ("R", "G", "B", "NIR", "NIR/R"):
            assert ("delta", ch, str(hour)) in keys
        for modality in ("RGB", "NIR", "RGB+NIR"):
            assert ("auroc", modality, str(hour)) in keys
            assert ("negative_control", modality, str(hour)) in keys
    row = next(r for r in table if (r["metric"], r["name"], r["hours"]) == ("delta", "NIR", str(DARK_HOUR)))
    assert float(row["lot_min"]) <= float(row["median_per_mango"]) <= float(row["lot_max"])
    assert int(row["n_lots"]) == 3
    figure = cv2.imread(str(out / "figure.png"))
    assert figure is not None and figure.shape[1] > figure.shape[0]
    report = (out / "README.md").read_text(encoding="utf-8")
    assert "figure.png" in report
    assert "boxes version" in report
    patches = read_csv(out / "patches.csv")
    assert {"impact", "control", "control2"} == {p["region"] for p in patches}
    assert f"{len(synthetic['faces'])} rows" in synthetic["stdout"]
    assert f"{len(patches)} patches" in synthetic["stdout"]


def test_report_states_the_latent_quantities_of_the_reading_table_with_the_latent_constant(synthetic):
    summary = synthetic["summary"]
    assert summary["latent_hours"] == list(LATENT_HOURS)
    expected = {str(h) for h in summary["hours"] if h in LATENT_HOURS}
    assert set(summary["reading"]) == expected
    for reading in summary["reading"].values():
        assert {"NIR-RGB", "RGB-negative_control", "negative_control_covers_chance"} <= set(reading)
    report = (synthetic["out"] / "README.md").read_text(encoding="utf-8")
    for token in ("NIR - RGB", "RGB - negative control", "Hours whose interval excludes 0.5"):
        assert token in report


@pytest.fixture(scope="module")
def small_dataset(tmp_path_factory):
    root = tmp_path_factory.mktemp("small")
    rows, truth = make_dataset(root, lots=(1,), n_mangos=2, faces=(1,), seed=4)
    return root, rows, truth


def _task(root, **changes):
    task = e0.box_tasks(root)[0]
    return {**task, **changes}


def _statuses(rows):
    return {r["moment"]: r["status"] for r in rows}


def test_face_with_every_photo_readable_measures_all_its_moments(small_dataset):
    rows, patches = e0.measure_face_boxes(_task(small_dataset[0]))
    assert set(_statuses(rows).values()) == {"ok"}
    assert {p["region"] for p in patches} == {"impact", "control", "control2"}


def test_unreadable_sano_photo_excludes_the_whole_face(small_dataset):
    task = _task(small_dataset[0])
    rows, patches = e0.measure_face_boxes({**task, "sano": {**task["sano"], "stem": "missing", "timestamp": 1}})
    assert _statuses(rows) == {"sano": "no_fruit_or_file", **{str(h): "no_sano" for h in task["moments"]}}
    assert patches == []


@pytest.mark.parametrize(
    "box, status",
    [
        ([0.0, 0.0, 30.0, 30.0], "box_outside_fruit"),  # a corner of the sheet
        ([175.0, 135.0, 465.0, 345.0], "no_control_zone"),  # most of the fruit
    ],
)
def test_box_with_no_fruit_under_it_or_no_room_for_a_control_is_excluded(small_dataset, box, status):
    task = _task(small_dataset[0])
    moments = {h: {**photo, "damage": box} for h, photo in task["moments"].items()}
    rows, patches = e0.measure_face_boxes({**task, "moments": moments})
    assert {r["status"] for r in rows if r["moment"] != "sano"} == {status}
    assert patches == []


def test_face_without_a_zone_is_excluded(small_dataset):
    rows, patches = e0.measure_face_boxes(_task(small_dataset[0], zone=""))
    assert {r["status"] for r in rows if r["moment"] != "sano"} == {"no_zone"}
    assert patches == []


def test_moment_that_does_not_align_is_excluded(small_dataset, monkeypatch):
    monkeypatch.setattr(e0, "align", lambda ref, mov: None)
    rows, patches = e0.measure_face_boxes(_task(small_dataset[0]))
    assert {r["status"] for r in rows if r["moment"] != "sano"} == {"not_aligned"}
    assert patches == []


def test_max_mangos_per_lot_keeps_the_first_mangos_of_each_lot(tmp_path):
    root = tmp_path / "dataset"
    make_dataset(root, lots=(1, 2), n_mangos=3, faces=(1,))
    tasks = e0.box_tasks(root, 2)
    assert sorted({(t["lot"], t["mango"]) for t in tasks}) == [(1, 1), (1, 2), (2, 26), (2, 27)]
    assert len(e0.box_tasks(root)) == 6


def test_a_conflicting_or_unknown_zone_stops_with_a_clear_error(tmp_path):
    root = tmp_path / "dataset"
    rows, _ = make_dataset(root, lots=(1,), n_mangos=1, faces=(1,))
    rows[1]["zona"] = "arriba"
    _write_index(root, rows)
    with pytest.raises(ValueError, match="arriba"):
        e0.box_tasks(root)


def test_boxes_mode_needs_the_dataset_and_refuses_the_options_of_centro(tmp_path):
    result = run_script("--mode", "boxes")
    assert result.returncode == 2 and "--dataset" in result.stderr
    result = run_script("--mode", "boxes", "--dataset", tmp_path, "--lote", "a", "b")
    assert result.returncode == 2 and "--lote" in result.stderr
    result = run_script("--mode", "boxes", "--dataset", tmp_path, "--n-perm", 5)
    assert result.returncode == 2 and "--n-perm" in result.stderr


def test_centro_mode_refuses_the_options_of_boxes(tmp_path):
    result = run_script("--lote", "a", "b", "--dataset", tmp_path)
    assert result.returncode == 2 and "--dataset" in result.stderr
    result = run_script("--lote", "a", "b", "--max-mangos-per-lot", 2)
    assert result.returncode == 2 and "--max-mangos-per-lot" in result.stderr


def test_boxes_mode_does_not_overwrite_the_centro_report_by_default(tmp_path):
    args = e0.parse_args(["--mode", "boxes", "--dataset", str(tmp_path)])
    assert args.out == PROJECT_ROOT / "reports" / "e0-detectability" / "boxes"


# --- centro mode: what it did before the boxes mode existed ---------------------------------------


def make_jornada(root):
    """A captura_v3 jornada: 3 mangos x 2 faces, +0 and +2 h, one late photo."""
    rng = np.random.default_rng(3)
    (root / "news" / "rgb").mkdir(parents=True)
    (root / "news" / "nir").mkdir(parents=True)
    zones = {1: {1: "centro", 2: "superior"}, 2: {1: "inferior", 2: "centro"}, 3: {1: "derecha", 2: "izquierda"}}
    jornada = {"id": "lote1-test", "tolerancia_min": 10, "mangos": [1, 2, 3], "caras": [1, 2], "intervalos_h": [0, 2],
               "sanas": {}, "rondas": {}, "zonas": {str(m): {str(c): z for c, z in f.items()} for m, f in zones.items()}}
    ts = 500
    for mango, faces in zones.items():
        for cara in faces:
            texture = _texture(rng)
            for hour in ("sano", 0, 2):
                ts += 1
                bgr, nir = _render(texture, rng.integers(-5, 6, 2).astype(float), np.zeros(4), 0.0, 0.0, rng)
                assert cv2.imwrite(str(root / "news" / "rgb" / f"mango_rgb_{ts}.jpg"), bgr)
                assert cv2.imwrite(str(root / "news" / "nir" / f"mango_nir_{ts}.jpg"), nir)
                if hour == "sano":
                    jornada["sanas"].setdefault(str(mango), {})[str(cara)] = {"timestamp": ts}
                else:
                    late = mango == 3 and cara == 2 and hour == 2
                    jornada["rondas"].setdefault(str(mango), {}).setdefault(str(hour), {})[str(cara)] = {
                        "timestamp": ts, "desviacion_min": 30 if late else 1}
    (root / "jornada.json").write_text(json.dumps(jornada), encoding="utf-8")
    np.save(root / "H.npy", np.eye(3))
    return root


def test_centro_is_still_the_default_and_keeps_its_outputs(tmp_path):
    root = make_jornada(tmp_path / "jornada")
    out = tmp_path / "out"
    result = run_script("--lote", root, root / "H.npy", "--out", out, "--n-boot", 20, "--n-perm", 5)
    assert result.returncode == 0, result.stderr
    assert result.stdout == f"6 faces, 18 rows -> {out}\n"
    faces = read_csv(out / "faces.csv")
    assert list(faces[0]) == ["lot", "mango", "face", "zone", "moment", "hours", "status", "R", "G", "B", "lum", "NIR", "NIR/R"]
    assert len(faces) == 18
    assert sum(r["status"] == "out_of_tolerance" for r in faces) == 1
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["n_boot"] == 20 and summary["n_perm"] == 5 and summary["seed"] == 0
    assert summary["oof_grouping"] == "mango"
    assert set(summary) == {"lots", "hours", "n_boot", "n_perm", "seed", "oof_grouping", "exclusions", "per_hour",
                            "placebo_sano", "onset_hour"}
    assert (out / "README.md").read_text(encoding="utf-8").startswith("# E0: impact detectability per hour (centro-only version)")


def test_explicit_centro_mode_equals_the_default(tmp_path):
    root = make_jornada(tmp_path / "jornada")
    common = ("--lote", root, root / "H.npy", "--n-boot", 20, "--n-perm", 5)
    assert run_script(*common, "--out", tmp_path / "a").returncode == 0
    assert run_script("--mode", "centro", *common, "--out", tmp_path / "b").returncode == 0
    for name in ("faces.csv", "summary.json", "README.md"):
        assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes()


def test_centro_still_requires_lote():
    result = run_script()
    assert result.returncode == 2
    assert "error: the following arguments are required: --lote" in result.stderr


def test_centro_keeps_its_option_defaults():
    args = e0.parse_args(["--lote", "a", "b"])
    assert args.mode == "centro"
    assert args.out == PROJECT_ROOT / "reports" / "e0-detectability"
    assert (args.n_boot, args.n_perm, args.seed, args.workers) == (2000, 500, 0, None)
    assert args.lote == [["a", "b"]]
