"""CLI-level tests for scripts/correct_white_balance.py on synthetic data (never the real dataset)."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from capture.calidad import cargar_reducido
from scripts import correct_white_balance as cwb
from src.data_pipeline import white_balance as wb

JORNADA = "lote-test"
SHEET = (90, 200, 190)
CAST = {  # per-photo auto-WB style tint (linear gains)
    (3, 2): (4.5, 0.9, 1.1),
    (4, 1): (3.8, 1.0, 1.0),
    (4, 2): (4.2, 0.95, 1.05),
}
BASE_TS = 1_000_000


def mango_color(mango):
    return (95 + 10 * mango, 95, 30)


def good_scene(mango, jitter=0):
    h, w = 120, 160
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[3:-3, 4:-4] = np.clip(np.array(SHEET) + jitter, 0, 255)
    yy, xx = np.mgrid[:h, :w]
    ellipse = ((yy - h / 2) / (0.22 * h)) ** 2 + ((xx - w / 2) / (0.20 * w)) ** 2 <= 1
    img[ellipse] = mango_color(mango)
    return img


def write_photo(root, ts, mango, cara, ronda, img, jornada=JORNADA):
    (root / "news" / "rgb").mkdir(parents=True, exist_ok=True)
    (root / "news" / "meta").mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(root / "news" / "rgb" / f"mango_rgb_{ts}.jpg", quality=95)
    meta = {"timestamp": ts, "mango_id": mango, "cara": cara, "ronda_h": ronda,
            "fase": "sano" if ronda is None else "post_impacto", "jornada": {"id": jornada}}
    (root / "news" / "meta" / f"mango_meta_{ts}.json").write_text(json.dumps(meta))


@pytest.fixture
def data_root(tmp_path):
    """Mangos 1-4 x faces 1-2 at +2 h and +4 h; at +4 h the photos 3-2, 4-1, 4-2 have the auto-WB tint."""
    root = tmp_path / "data"
    ts = BASE_TS
    for ronda in (2, 4):
        for mango in range(1, 5):
            for cara in (1, 2):
                img = good_scene(mango, jitter=(mango + cara) % 3)
                if ronda == 4 and (mango, cara) in CAST:
                    img, _ = wb.apply_gains(img, CAST[(mango, cara)])
                write_photo(root, ts, mango, cara, ronda, img)
                ts += 10
    write_photo(root, ts, 1, 1, 2, good_scene(1), jornada="otra-jornada")   # must be ignored
    return root


def tree_hashes(path):
    return {str(p.relative_to(path)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(path.rglob("*")) if p.is_file()}


def run(data_root, tmp_path, *extra, faces="3:2,4", expected="3"):
    argv = ["--data-root", str(data_root), "--jornada", JORNADA, "--ronda", "4", "--faces", faces,
            "--expected-count", expected, "--figure-dir", str(tmp_path / "fig"),
            "--summary-json", str(tmp_path / "summary.json"), *extra]
    return cwb.main(argv)


def ts_of(data_root, mango, cara, ronda):
    for p in (data_root / "news" / "meta").glob("*.json"):
        m = json.loads(p.read_text())
        if (m["mango_id"], m["cara"], m["ronda_h"], m["jornada"]["id"]) == (mango, cara, ronda, JORNADA):
            return m["timestamp"]
    raise KeyError((mango, cara, ronda))


def border_means(data_root, ts, frac):
    return wb.border_means_linear(cargar_reducido(data_root / "news" / "rgb" / f"mango_rgb_{ts}.jpg"), frac)


# --- face spec -----------------------------------------------------------------------------

def test_parse_face_spec_lot1_affected_set_has_57_photos():
    slots = cwb.parse_face_spec("11:4,12-25")
    assert len(slots) == 57
    assert (11, 4) in slots and (11, 3) not in slots
    assert all((m, c) in slots for m in range(12, 26) for c in range(1, 5))
    assert (26, 1) not in slots and (10, 4) not in slots


@pytest.mark.parametrize("spec,expected", [
    ("5", {(5, 1), (5, 2), (5, 3), (5, 4)}),
    ("5:1-2", {(5, 1), (5, 2)}),
    ("1-2:3", {(1, 3), (2, 3)}),
    ("7:4,7:4", {(7, 4)}),
])
def test_parse_face_spec_grammar(spec, expected):
    assert cwb.parse_face_spec(spec) == expected


@pytest.mark.parametrize("spec", ["", "abc", "12-", "25-12", "11:5", "11:0", "0", ",", "3:"])
def test_parse_face_spec_rejects_malformed_specs(spec):
    with pytest.raises(ValueError):
        cwb.parse_face_spec(spec)


def test_largest_gap_threshold_splits_a_bimodal_distribution():
    threshold, lo, hi = cwb.largest_gap_threshold([0.24, 0.25, 0.23, 0.26, 0.85, 0.9, 1.1])
    assert (lo, hi) == (0.26, 0.85)
    assert 0.26 < threshold < 0.85


def test_largest_gap_threshold_refuses_unimodal_data():
    with pytest.raises(cwb.WhiteBalanceError):
        cwb.largest_gap_threshold([0.24, 0.25, 0.26, 0.27, 0.28])


# --- end to end --------------------------------------------------------------------------------

def test_corrects_only_the_selected_photos_and_writes_a_manifest(data_root, tmp_path):
    before = tree_hashes(data_root)
    assert run(data_root, tmp_path) == 0

    out_dir = data_root / "news" / "rgb_wb"
    selected = {ts_of(data_root, 3, 2, 4), ts_of(data_root, 4, 1, 4), ts_of(data_root, 4, 2, 4)}
    assert {p.name for p in out_dir.glob("*.jpg")} == {f"mango_rgb_{t}.jpg" for t in selected}

    after = tree_hashes(data_root)
    assert {k: v for k, v in after.items() if not k.startswith("news/rgb_wb")} == before   # originals untouched

    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["jornada"] == JORNADA and manifest["ronda_h"] == 4
    assert {f["timestamp"] for f in manifest["files"]} == selected
    good_round4 = {ts_of(data_root, m, c, 4) for m in range(1, 5) for c in (1, 2)} - selected
    assert set(manifest["reference"]["timestamps"]) == good_round4
    assert manifest["threshold"]["rg_linear"] > 0.2
    assert manifest["method"]

    ref_rg, ref_bg = manifest["reference"]["chroma"]["rg"], manifest["reference"]["chroma"]["bg"]
    for entry in manifest["files"]:
        assert len(entry["gains"]) == 3 and entry["gains"][0] < 0.5          # R is pulled down
        assert entry["before"]["rg"] > 0.5
        assert entry["after"]["rg"] == pytest.approx(ref_rg, rel=0.05)
        assert entry["after"]["bg"] == pytest.approx(ref_bg, rel=0.05)
        assert 0.0 <= entry["clipped_fraction"] <= 1.0
        src = data_root / "news" / "rgb" / f"mango_rgb_{entry['timestamp']}.jpg"
        out = out_dir / f"mango_rgb_{entry['timestamp']}.jpg"
        assert entry["source_sha256"] == hashlib.sha256(src.read_bytes()).hexdigest()
        assert entry["output_sha256"] == hashlib.sha256(out.read_bytes()).hexdigest()
        with Image.open(out) as im:
            assert im.size == (160, 120)

    validation = manifest["validation"]
    assert validation["formula"].startswith("CIE76")
    assert validation["median_delta_e_after"] < 0.25 * validation["median_delta_e_before"]
    for entry in manifest["files"]:
        assert entry["validation"]["delta_e_after"] < entry["validation"]["delta_e_before"]
        assert len(entry["validation"]["delta_lab_after"]) == 3
    assert len(validation["median_delta_lab_after"]) == 3
    assert validation["control_unaffected_photos"]["n"] == 5


def test_writes_summary_and_before_after_figure(data_root, tmp_path):
    assert run(data_root, tmp_path) == 0
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["n_corrected"] == 3
    figures = list((tmp_path / "fig").glob("*.jpg"))
    assert figures and all(f.stat().st_size < 300_000 for f in figures)


def test_dry_run_writes_nothing(data_root, tmp_path):
    before = tree_hashes(data_root)
    assert run(data_root, tmp_path, "--dry-run") == 0
    assert tree_hashes(data_root) == before
    assert not (data_root / "news" / "rgb_wb").exists()
    assert not (tmp_path / "summary.json").exists() and not (tmp_path / "fig").exists()


def test_missing_counterpart_is_reported_as_null_not_a_crash(data_root, tmp_path):
    victim = ts_of(data_root, 4, 1, 2)
    (data_root / "news" / "rgb" / f"mango_rgb_{victim}.jpg").unlink()
    (data_root / "news" / "meta" / f"mango_meta_{victim}.json").unlink()
    assert run(data_root, tmp_path) == 0
    manifest = json.loads((data_root / "news" / "rgb_wb" / "manifest.json").read_text())
    by_ts = {f["timestamp"]: f for f in manifest["files"]}
    assert by_ts[ts_of(data_root, 4, 1, 4)]["validation"] is None
    assert by_ts[ts_of(data_root, 4, 2, 4)]["validation"] is not None


# --- aborts: nothing may be written ----------------------------------------------------------------

def assert_aborted(code, data_root, before, capsys, *needles):
    assert code == 2
    err = capsys.readouterr().err
    for needle in needles:
        assert str(needle) in err
    assert tree_hashes(data_root) == before
    assert not (data_root / "news" / "rgb_wb").exists()


def test_aborts_when_selected_count_differs_from_expected(data_root, tmp_path, capsys):
    before = tree_hashes(data_root)
    code = run(data_root, tmp_path, expected="57")
    assert_aborted(code, data_root, before, capsys, "expected 57", "3")


def test_aborts_when_an_outlier_is_not_selected(data_root, tmp_path, capsys):
    before = tree_hashes(data_root)
    missed = ts_of(data_root, 4, 2, 4)
    code = run(data_root, tmp_path, faces="3:2,4:1", expected="2")
    assert_aborted(code, data_root, before, capsys, "outliers not selected", missed)


def test_aborts_when_a_selected_photo_is_not_an_outlier(data_root, tmp_path, capsys):
    before = tree_hashes(data_root)
    wrongly = ts_of(data_root, 1, 1, 4)
    code = run(data_root, tmp_path, faces="1:1,3:2,4", expected="4")
    assert_aborted(code, data_root, before, capsys, "selected but not outliers", wrongly)


def test_aborts_when_another_round_has_an_off_colour_photo(data_root, tmp_path, capsys):
    bad, _ = wb.apply_gains(good_scene(2), (4.0, 1.0, 1.0))
    ts = ts_of(data_root, 2, 1, 2)
    write_photo(data_root, ts, 2, 1, 2, bad)
    before = tree_hashes(data_root)
    code = run(data_root, tmp_path)
    assert_aborted(code, data_root, before, capsys, "outliers not selected", ts)


PROTECTED = ("rgb", "nir", "meta")


@pytest.fixture
def full_data_root(data_root):
    (data_root / "news" / "nir").mkdir()
    return data_root


@pytest.mark.parametrize("out_subdir", [
    "news/rgb", "news/../news/rgb", "./news/rgb/", "../elsewhere", "news/rgb/sub",
    "news/nir", "news/nir/sub", "news/meta", "news/meta/sub", "news/zzz/../meta",
])
def test_refuses_output_dir_that_is_a_news_original_or_escapes_the_data_root(
        full_data_root, tmp_path, capsys, out_subdir):
    before = tree_hashes(full_data_root)
    code = run(full_data_root, tmp_path, "--out-subdir", out_subdir)
    assert code == 2
    assert "out-subdir" in capsys.readouterr().err
    assert tree_hashes(full_data_root) == before
    assert not (tmp_path / "elsewhere").exists()


@pytest.mark.parametrize("name", PROTECTED)
def test_refuses_output_dir_symlinked_onto_a_news_original(full_data_root, tmp_path, capsys, name):
    (full_data_root / "news" / "alias").symlink_to(full_data_root / "news" / name, target_is_directory=True)
    before = tree_hashes(full_data_root)
    code = run(full_data_root, tmp_path, "--out-subdir", "news/alias")
    assert code == 2
    assert "out-subdir" in capsys.readouterr().err
    assert tree_hashes(full_data_root) == before


@pytest.mark.parametrize("flag,suffix", [("--figure-dir", ""), ("--summary-json", "summary.json")])
@pytest.mark.parametrize("name", PROTECTED)
@pytest.mark.parametrize("style", ["direct", "dotdot", "symlink"])
def test_refuses_figure_dir_and_summary_json_inside_a_news_original(
        full_data_root, tmp_path, capsys, flag, suffix, name, style):
    target = full_data_root / "news" / name
    if style == "direct":
        value = target / suffix
    elif style == "dotdot":
        value = full_data_root / "news" / "zzz" / ".." / name / suffix
    else:
        (tmp_path / "alias").symlink_to(target, target_is_directory=True)
        value = tmp_path / "alias" / suffix
    before = tree_hashes(full_data_root)
    code = run(full_data_root, tmp_path, flag, str(value))
    assert code == 2
    assert flag in capsys.readouterr().err
    assert tree_hashes(full_data_root) == before


def test_summary_json_cannot_overwrite_a_metadata_file(data_root, tmp_path, capsys):
    meta = next((data_root / "news" / "meta").glob("*.json"))
    original = meta.read_bytes()
    assert run(data_root, tmp_path, "--summary-json", str(meta)) == 2
    assert "--summary-json" in capsys.readouterr().err
    assert meta.read_bytes() == original


def test_summary_json_symlinked_onto_a_metadata_file_is_refused(data_root, tmp_path, capsys):
    meta = next((data_root / "news" / "meta").glob("*.json"))
    original = meta.read_bytes()
    link = tmp_path / "link.json"
    link.symlink_to(meta)
    assert run(data_root, tmp_path, "--summary-json", str(link)) == 2
    assert "--summary-json" in capsys.readouterr().err
    assert meta.read_bytes() == original


def test_aborts_on_unreadable_selection_spec(data_root, tmp_path, capsys):
    before = tree_hashes(data_root)
    code = run(data_root, tmp_path, faces="nonsense")
    assert_aborted(code, data_root, before, capsys, "faces")


# --- degenerate frames ---------------------------------------------------------------------------

def test_a_black_frame_aborts_with_a_clear_message_instead_of_a_division_error(data_root, tmp_path, capsys):
    ts = BASE_TS + 999
    write_photo(data_root, ts, 9, 1, 2, np.zeros((120, 160, 3), dtype=np.uint8))
    before = tree_hashes(data_root)
    code = run(data_root, tmp_path)
    assert_aborted(code, data_root, before, capsys, f"mango_rgb_{ts}.jpg", "zero")


def test_white_balance_error_is_shared_between_module_and_cli():
    assert cwb.WhiteBalanceError is wb.WhiteBalanceError
    assert issubclass(wb.WhiteBalanceError, ValueError)


# --- estimator and parameter coverage -----------------------------------------------------------------

def test_border_frac_changes_what_is_measured(data_root, tmp_path):
    means = {}
    for frac in (0.04, 0.20):
        out = f"news/rgb_wb_{frac}"
        assert run(data_root, tmp_path, "--border-frac", str(frac), "--out-subdir", out) == 0
        manifest = json.loads((data_root / out / "manifest.json").read_text())
        assert manifest["border_frac"] == frac
        expected = np.median([border_means(data_root, t, frac) for t in manifest["reference"]["timestamps"]], axis=0)
        np.testing.assert_allclose(manifest["reference"]["means_linear"], expected, atol=1e-6)
        means[frac] = np.array(manifest["reference"]["means_linear"])
    assert not np.allclose(means[0.04], means[0.20], rtol=0.5)


def test_reference_is_the_per_channel_median_not_the_mean(data_root, tmp_path):
    victim = ts_of(data_root, 1, 1, 4)
    dim, _ = wb.apply_gains(good_scene(1, jitter=2), (0.3, 0.3, 0.3))    # same chroma, much darker
    write_photo(data_root, victim, 1, 1, 4, dim)
    assert run(data_root, tmp_path) == 0
    manifest = json.loads((data_root / "news" / "rgb_wb" / "manifest.json").read_text())
    assert victim in manifest["reference"]["timestamps"]
    stack = np.array([border_means(data_root, t, 0.10) for t in manifest["reference"]["timestamps"]])
    median, mean = np.median(stack, axis=0), stack.mean(axis=0)
    assert not np.allclose(median, mean, rtol=0.10)       # the dim photo separates the two estimators
    np.testing.assert_allclose(manifest["reference"]["means_linear"], median, atol=1e-6)


def test_describe_keeps_the_precision_asked_for():
    stats = cwb.describe([0.0, 4e-6, 1.2e-5], digits=6)
    assert stats == {"min": 0.0, "median": 4e-6, "max": 1.2e-5}
