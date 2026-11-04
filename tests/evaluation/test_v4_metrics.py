"""Stratified v4 metrics (plan-experimental section 7), through `scripts/v4_metrics.py`.

Every expected value is computed by hand in the test that asserts it. Hours come from
`LATENT_HOURS` so the tests keep their meaning if the latent window constant changes.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scripts import v4_metrics as cli
from src.data_pipeline.v4_selection import LATENT_HOURS
from src.evaluation import v4_metrics as metrics_module

SANO = "sano"
GT = (0.1, 0.1, 0.3, 0.3)
MISS = (0.6, 0.6, 0.8, 0.8)        # no overlap with GT
LATENT = LATENT_HOURS[0]
LAST_LATENT = max(LATENT_HOURS)
DEVELOPED = LAST_LATENT + 2        # first post-impact hour after the window
DEVELOPED_2 = LAST_LATENT + 4
FIELDS = ["stem", "lot", "mango_id", "cara", "ronda_h", "zona", "provenance"]


def _photo(stem: str, hour, *, gt=None, damage=(), mango=(), lot=1, mango_id=None, cara=1,
           zona="centro", provenance="dibujada") -> dict:
    """One evaluated photo. `damage` and `mango` are `(confidence, xyxy)` predictions."""
    is_sano = hour == SANO
    return {
        "stem": stem,
        "lot": lot,
        "mango_id": stem if mango_id is None else mango_id,
        "cara": cara,
        "ronda_h": hour,
        "zona": "" if is_sano else zona,
        "provenance": "" if is_sano else provenance,
        "gt": () if is_sano else ((GT,) if gt is None else gt),
        "damage": damage,
        "mango": mango,
    }


def _box(cls: int, confidence: float, xyxy) -> dict:
    return {"class": cls, "confidence": confidence, "xyxy": list(xyxy)}


def _write(root: Path, photos: list[dict], other_lots: list[dict] = ()) -> list[str]:
    """Write the three inputs and return the CLI arguments. `other_lots` only reach the index."""
    labels = root / "labels" / "test"
    labels.mkdir(parents=True)
    with (root / "index.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows([*photos, *other_lots])
    lines = []
    for photo in photos:
        rows = ["0 0.5 0.5 0.9 0.9"]  # the mango box every label file has
        for x1, y1, x2, y2 in photo["gt"]:
            rows.append(f"1 {(x1 + x2) / 2} {(y1 + y2) / 2} {x2 - x1} {y2 - y1}")
        (labels / f"{photo['stem']}.txt").write_text("\n".join(rows) + "\n")
        boxes = [_box(0, c, b) for c, b in photo["mango"]] + [_box(1, c, b) for c, b in photo["damage"]]
        lines.append(json.dumps({"stem": photo["stem"], "boxes": boxes}))
    (root / "predictions.jsonl").write_text("\n".join(lines) + "\n")
    return ["--predictions", str(root / "predictions.jsonl"), "--index", str(root / "index.csv"),
            "--labels-dir", str(labels), "--output", str(root / "out" / "metrics_v4.json")]


def _run(root: Path, photos: list[dict], *extra: str, other_lots: list[dict] = ()) -> dict:
    """Run the CLI, assert it succeeded, and return the file it wrote."""
    argv = _write(root, photos, other_lots)
    assert cli.main([*argv, *extra]) == 0
    return json.loads((root / "out" / "metrics_v4.json").read_text())


# --------------------------------------------------------------------------
# Primary and secondary 1
# --------------------------------------------------------------------------


def test_primary_is_ap50_on_latent_plus_sano_and_skips_developed(tmp_path: Path, capsys) -> None:
    result = _run(tmp_path, [
        _photo("a_latent", LATENT, damage=[(0.8, GT)]),                # true positive, ranked 2nd
        _photo("b_sano", SANO, damage=[(0.9, MISS)]),                  # false positive, ranked 1st
        _photo("c_developed", DEVELOPED, damage=[(0.95, MISS)]),       # outside the set
    ])
    # Ranked FP then TP: precision 1/2 at recall 1, so AP = 0.5. With c it would be 1/3 at recall 1/2.
    assert result["latent_ap50"] == {"ap": pytest.approx(0.5), "n_gt": 1, "n_images": 2}
    assert result["damage_class_id"] == 1
    assert result["num_images"] == 3
    assert result["latent_hours"] == list(LATENT_HOURS)
    out = capsys.readouterr().out
    assert "latent AP50: 0.5000" in out and "metrics_v4.json" in out


def test_mango_predictions_and_boxes_do_not_enter_the_damage_metrics(tmp_path: Path) -> None:
    result = _run(tmp_path, [_photo("a", LATENT, damage=[(0.5, GT)], mango=[(0.99, MISS)])])
    # A mango prediction scored above the damage one would be a damage FP if it leaked in.
    assert result["latent_ap50"]["ap"] == pytest.approx(1.0)
    assert result["latent_ap50"]["n_gt"] == 1  # the mango line of the label file is not counted


def test_latent_ap30_is_the_same_photo_set_at_iou_030(tmp_path: Path) -> None:
    gt = (0.0, 0.0, 0.4, 0.4)
    overlap = (0.0, 0.0, 0.4, 0.16)  # inside the GT: IoU = 0.064 / 0.16 = 0.4
    result = _run(tmp_path, [
        _photo("a_latent", LATENT, gt=(gt,), damage=[(0.8, overlap)]),
        _photo("b_sano", SANO, damage=[(0.9, MISS)]),
        _photo("c_developed", DEVELOPED, gt=(gt,), damage=[(0.95, MISS)]),
    ])
    assert result["latent_ap50"] == {"ap": pytest.approx(0.0), "n_gt": 1, "n_images": 2}  # IoU 0.4 < 0.5
    # At IoU 0.3 the same ranking is FP (sano) then TP: precision 1/2 at recall 1.
    assert result["latent_ap30"] == {"ap": pytest.approx(0.5), "n_gt": 1, "n_images": 2}


# --------------------------------------------------------------------------
# Latent window and hours
# --------------------------------------------------------------------------


def test_ap50_by_hour_covers_every_hour_of_the_index_without_sano(tmp_path: Path) -> None:
    result = _run(tmp_path, [
        _photo("a", LATENT, damage=[(0.9, GT)]),                                   # AP 1
        _photo("b", DEVELOPED, damage=[(0.9, MISS), (0.8, GT)]),                   # FP then TP: AP 1/2
        _photo("c_sano", SANO, damage=[(0.99, MISS)]),                             # in no hour
    ], other_lots=[_photo("z", DEVELOPED_2)])
    hours = result["ap50_by_hour"]
    assert hours[str(LATENT)] == {"ap": pytest.approx(1.0), "n_gt": 1, "n_images": 1}
    assert hours[str(DEVELOPED)] == {"ap": pytest.approx(0.5), "n_gt": 1, "n_images": 1}
    assert hours[str(DEVELOPED_2)] == {"ap": None, "n_gt": 0, "n_images": 0}  # no photo in this lot
    assert SANO not in hours
    assert set(hours) == {str(h) for h in (*LATENT_HOURS, DEVELOPED, DEVELOPED_2)}


def test_developed_is_every_post_impact_hour_outside_the_latent_window(tmp_path: Path) -> None:
    result = _run(tmp_path, [
        _photo("a", DEVELOPED, damage=[(0.9, GT)]),                # TP
        _photo("b", DEVELOPED_2),                                  # GT never found: recall stops at 1/2
        _photo("c", LATENT, damage=[(0.99, MISS)]),                # a FP outside the set
    ])
    assert result["developed_hours"] == [DEVELOPED, DEVELOPED_2]
    assert result["ap50_developed"] == {"ap": pytest.approx(0.5), "n_gt": 2, "n_images": 2}


def test_the_latent_window_comes_from_the_constant(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(metrics_module, "LATENT_HOURS", (LATENT, DEVELOPED))
    result = _run(tmp_path, [
        _photo("a", LATENT, damage=[(0.9, GT)]),
        _photo("b", DEVELOPED),                                    # now latent: its GT counts, unmatched
        _photo("c", DEVELOPED_2),                                  # still developed
    ])
    assert result["latent_hours"] == [LATENT, DEVELOPED]
    assert result["developed_hours"] == [DEVELOPED_2]
    assert result["latent_ap50"] == {"ap": pytest.approx(0.5), "n_gt": 2, "n_images": 2}


# --------------------------------------------------------------------------
# Zone and provenance
# --------------------------------------------------------------------------


def test_ap50_by_zone_and_sano_stays_out(tmp_path: Path) -> None:
    result = _run(tmp_path, [
        _photo("a", LATENT, zona="centro", damage=[(0.9, GT)]),                    # AP 1
        _photo("b", LATENT, zona="derecha", damage=[(0.9, MISS), (0.8, GT)]),      # AP 1/2
        _photo("c_sano", SANO, damage=[(0.99, MISS)]),                             # would lower both
    ], other_lots=[_photo("z", LATENT, zona="inferior")])
    assert result["ap50_by_zone"] == {
        "centro": {"ap": pytest.approx(1.0), "n_gt": 1, "n_images": 1},
        "derecha": {"ap": pytest.approx(0.5), "n_gt": 1, "n_images": 1},
        "inferior": {"ap": None, "n_gt": 0, "n_images": 0},
    }


def test_ap50_by_provenance(tmp_path: Path) -> None:
    result = _run(tmp_path, [
        _photo("a", LATENT, provenance="proyectada", damage=[(0.9, GT)]),
        _photo("b", DEVELOPED, provenance="dibujada", damage=[(0.9, MISS), (0.8, GT)]),
        _photo("c_sano", SANO, damage=[(0.99, MISS)]),
    ])
    assert result["ap50_by_provenance"] == {
        "dibujada": {"ap": pytest.approx(0.5), "n_gt": 1, "n_images": 1},
        "proyectada": {"ap": pytest.approx(1.0), "n_gt": 1, "n_images": 1},
    }


# --------------------------------------------------------------------------
# False positives on sano
# --------------------------------------------------------------------------


def test_fp_per_image_on_sano_counts_damage_boxes_at_the_operating_point(tmp_path: Path) -> None:
    result = _run(tmp_path, [
        _photo("a", SANO, damage=[(0.9, MISS), (0.3, MISS), (0.1, MISS)], mango=[(0.99, MISS)]),
        _photo("b", SANO, damage=[(0.25, MISS)]),                  # at the threshold: counted
        _photo("c", LATENT, damage=[(0.9, MISS)]),                 # not a sano photo
    ])
    # 3 damage boxes at >= 0.25 on 2 sano photos.
    assert result["sano_fp_per_image"] == {
        "conf_threshold": 0.25, "fp_per_image": pytest.approx(1.5), "n_fp": 3, "n_images": 2,
    }


def test_conf_threshold_option_moves_the_operating_point(tmp_path: Path) -> None:
    result = _run(tmp_path, [_photo("a", SANO, damage=[(0.9, MISS), (0.3, MISS)])], "--conf-threshold", "0.5")
    assert result["sano_fp_per_image"] == {
        "conf_threshold": 0.5, "fp_per_image": pytest.approx(1.0), "n_fp": 1, "n_images": 1,
    }


# --------------------------------------------------------------------------
# Per-face AUROC
# --------------------------------------------------------------------------


def test_face_auroc_ranks_latent_against_sano_on_the_maximum_damage_confidence(tmp_path: Path) -> None:
    result = _run(tmp_path, [
        _photo("p1", LATENT, damage=[(0.9, MISS), (0.2, MISS)]),                   # score 0.9
        _photo("p2", LAST_LATENT, damage=[(0.4, MISS)], mango=[(0.99, MISS)]),     # score 0.4, not 0.99
        _photo("n1", SANO, damage=[(0.5, MISS)]),
        _photo("n2", SANO, damage=[(0.1, MISS)]),
        _photo("d", DEVELOPED, damage=[(0.01, MISS)]),                             # neither group
    ])
    # Pairs (p, n): 0.9>0.5, 0.9>0.1, 0.4<0.5, 0.4>0.1 -> 3 of 4.
    assert result["face_auroc"]["auroc"] == pytest.approx(0.75)
    assert result["face_auroc"]["n_positive"] == 2
    assert result["face_auroc"]["n_negative"] == 2


def test_face_auroc_keeps_sano_out_of_the_positives(tmp_path: Path) -> None:
    result = _run(tmp_path, [
        _photo("p", LATENT, damage=[(0.6, MISS)]),
        _photo("n", SANO, damage=[(0.9, MISS)]),
    ])
    assert result["face_auroc"]["auroc"] == pytest.approx(0.0)
    assert result["face_auroc"]["n_positive"] == 1
    assert result["face_auroc"]["n_negative"] == 1


def test_a_face_with_no_predictions_scores_zero(tmp_path: Path) -> None:
    result = _run(tmp_path, [
        _photo("p1", LATENT, damage=[(0.5, MISS)]),
        _photo("p2", LATENT, mango=[(0.9, MISS)]),                 # no damage prediction: score 0
        _photo("n1", SANO),                                        # no prediction at all: score 0
        _photo("n2", SANO, damage=[(0.1, MISS)]),
    ])
    # Pairs: 0.5>0, 0.5>0.1, 0=0 counts 1/2, 0<0.1 -> 2.5 of 4.
    assert result["face_auroc"]["auroc"] == pytest.approx(0.625)
    assert result["face_auroc"]["n_positive_no_detection"] == 1
    assert result["face_auroc"]["n_negative_no_detection"] == 1


def test_face_auroc_reports_how_many_faces_hold_both_groups(tmp_path: Path) -> None:
    result = _run(tmp_path, [
        _photo("a0", LATENT, mango_id=1, cara=1),
        _photo("a1", LAST_LATENT, mango_id=1, cara=1),
        _photo("a2", SANO, mango_id=1, cara=1),                    # paired face
        _photo("b0", LATENT, mango_id=1, cara=2),                  # no sano photo
        _photo("c0", SANO, mango_id=2, cara=1),                    # no latent photo
    ])
    assert result["face_auroc"]["n_paired_faces"] == 1
    assert result["face_auroc"]["n_positive"] == 3
    assert result["face_auroc"]["n_negative"] == 2


# --------------------------------------------------------------------------
# Empty strata
# --------------------------------------------------------------------------


def test_strata_with_no_ground_truth_report_null_never_zero(tmp_path: Path) -> None:
    result = _run(tmp_path, [_photo("a", SANO, damage=[(0.9, MISS)])])
    assert result["latent_ap50"] == {"ap": None, "n_gt": 0, "n_images": 1}
    assert result["latent_ap30"] == {"ap": None, "n_gt": 0, "n_images": 1}
    assert result["ap50_developed"] == {"ap": None, "n_gt": 0, "n_images": 0}
    assert result["face_auroc"]["auroc"] is None  # no positives
    raw = (tmp_path / "out" / "metrics_v4.json").read_text()
    assert '"ap": null' in raw and '"auroc": null' in raw


def test_no_sano_photo_leaves_auroc_and_fp_null(tmp_path: Path) -> None:
    result = _run(tmp_path, [_photo("a", LATENT, damage=[(0.9, GT)])])
    assert result["face_auroc"]["auroc"] is None
    assert result["sano_fp_per_image"] == {
        "conf_threshold": 0.25, "fp_per_image": None, "n_fp": 0, "n_images": 0,
    }


# --------------------------------------------------------------------------
# The inputs must describe the same photos
# --------------------------------------------------------------------------


def _assert_rejected(argv: list[str], root: Path, capsys, needle: str) -> None:
    assert cli.main(argv) == 2
    err = capsys.readouterr().err
    assert "ERROR" in err and needle in err
    assert not (root / "out" / "metrics_v4.json").exists()


def test_a_label_file_without_a_prediction_line_is_rejected(tmp_path: Path, capsys) -> None:
    argv = _write(tmp_path, [_photo("a", LATENT)])
    (tmp_path / "labels" / "test" / "orphan.txt").write_text("1 0.5 0.5 0.1 0.1\n")
    _assert_rejected(argv, tmp_path, capsys, "orphan")


def test_a_prediction_line_without_a_label_file_is_rejected(tmp_path: Path, capsys) -> None:
    argv = _write(tmp_path, [_photo("a", LATENT), _photo("b", LATENT)])
    (tmp_path / "labels" / "test" / "b.txt").unlink()
    _assert_rejected(argv, tmp_path, capsys, "b")


def test_a_photo_missing_from_the_index_is_rejected(tmp_path: Path, capsys) -> None:
    argv = _write(tmp_path, [_photo("a", LATENT)])
    (tmp_path / "index.csv").write_text(",".join(FIELDS) + "\n")
    _assert_rejected(argv, tmp_path, capsys, "a")


def test_a_repeated_stem_is_rejected(tmp_path: Path, capsys) -> None:
    argv = _write(tmp_path, [_photo("a", LATENT)])
    with (tmp_path / "predictions.jsonl").open("a") as handle:
        handle.write(json.dumps({"stem": "a", "boxes": []}) + "\n")
    _assert_rejected(argv, tmp_path, capsys, "a")


@pytest.mark.parametrize("line", ["{not json", json.dumps({"boxes": []}), json.dumps({"stem": "a", "boxes": [{}]})])
def test_a_malformed_prediction_line_is_rejected(tmp_path: Path, capsys, line: str) -> None:
    argv = _write(tmp_path, [_photo("a", LATENT)])
    (tmp_path / "predictions.jsonl").write_text(line + "\n")
    _assert_rejected(argv, tmp_path, capsys, "predictions.jsonl")


def test_an_index_without_a_required_column_is_rejected(tmp_path: Path, capsys) -> None:
    argv = _write(tmp_path, [_photo("a", LATENT)])
    (tmp_path / "index.csv").write_text("stem,lot\na,1\n")
    _assert_rejected(argv, tmp_path, capsys, "ronda_h")


@pytest.mark.parametrize("threshold", ["-0.1", "1.5", "nan"])
def test_an_out_of_range_conf_threshold_is_rejected(tmp_path: Path, capsys, threshold: str) -> None:
    argv = _write(tmp_path, [_photo("a", LATENT)])
    _assert_rejected([*argv, "--conf-threshold", threshold], tmp_path, capsys, "--conf-threshold")


def test_the_output_file_is_deterministic(tmp_path: Path) -> None:
    argv = _write(tmp_path, [_photo("a", LATENT, damage=[(0.9, GT)]), _photo("b", SANO)])
    assert cli.main(argv) == 0
    first = (tmp_path / "out" / "metrics_v4.json").read_bytes()
    assert cli.main(argv) == 0
    assert (tmp_path / "out" / "metrics_v4.json").read_bytes() == first
