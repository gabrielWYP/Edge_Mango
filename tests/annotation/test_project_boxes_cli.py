"""Tests for the project_boxes command line entry point."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scripts.project_boxes import main
from src.annotation.box_projection import box_to_percent
from tests.annotation.export_synth import task, write_export
from tests.annotation.synthetic import JORNADA_1
from tests.annotation.world_synth import HAND, HEIGHT, WIDTH, write_rgb_world


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    return write_rgb_world(tmp_path_factory.mktemp("cli_world"))


@pytest.fixture(scope="module")
def outputs(world, tmp_path_factory):
    data_root, export, doc_root = world
    root = tmp_path_factory.mktemp("cli_out")
    code = main(
        [
            "--data-root",
            str(data_root),
            "--export",
            str(export),
            "--document-root",
            str(doc_root),
            "--out-dir",
            str(root / "ls"),
            "--report-dir",
            str(root / "report"),
            "--mask-sheet",
            str(root / "masks.jpg"),
            "--nir-display",
            "original",
            "--workers",
            "1",
        ]
    )
    return code, root


def test_run_succeeds_and_writes_every_output(outputs) -> None:
    code, root = outputs

    assert code == 0
    assert sorted(p.name for p in (root / "ls").iterdir()) == [
        "predictions_8h.json",
        "projection_10h_for_validation.json",
    ]
    assert sorted(p.name for p in (root / "report").iterdir()) == [
        "contact_sheet.jpg",
        "cycle_check.csv",
        "projection_report.csv",
        "summary.json",
    ]


def test_mask_sheet_is_written_on_request(outputs) -> None:
    _, root = outputs

    assert (root / "masks.jpg").stat().st_size > 0


def test_report_has_a_row_per_face_and_moment(outputs) -> None:
    _, root = outputs

    rows = list(csv.DictReader((root / "report" / "projection_report.csv").open(encoding="utf-8")))

    assert sorted((r["cara"], r["ronda_h"]) for r in rows) == [
        ("1", "10"),
        ("1", "8"),
        ("2", "10"),
        ("2", "8"),
    ]
    assert {r["tier"] for r in rows} <= {"A", "B", "C"}


def test_summary_lists_tiers_and_consistency(outputs) -> None:
    _, root = outputs

    summary = json.loads((root / "report" / "summary.json").read_text(encoding="utf-8"))

    assert summary["hand_boxes"] == {
        "nir": 0,
        "rgb": 2,
        "faces_without_box": [],
        "tasks_without_annotation": [],
    }
    assert summary["tiers"][f"{JORNADA_1}|10"]["A"] == 2
    assert "cycle" in summary["consistency"]
    assert len(summary["export_sha256"]) == 64


def hand_export(world, tmp_path: Path) -> Path:
    """The +12 h boxes plus a hand +10 h box for face 1 (RGB) and none for face 2."""
    _, export, _ = world
    tasks = json.loads(export.read_text(encoding="utf-8"))
    tasks.append(
        task(
            cara=1,
            hour=10,
            ts=2,
            box=("damage_rgb", *box_to_percent(HAND, WIDTH, HEIGHT).values()),
            size=(WIDTH, HEIGHT),
        )
    )
    return write_export(tmp_path, tasks, "hand_12h_10h.json")


def test_validation_mode_scores_the_10h_projections(world, outputs, tmp_path: Path, capsys) -> None:
    _, root = outputs

    code = main(
        [
            "--validate",
            str(hand_export(world, tmp_path)),
            "--out-dir",
            str(root / "ls"),
            "--report-dir",
            str(root / "report"),
        ]
    )

    assert code == 0
    printed = capsys.readouterr().out
    assert "Validation at +10 h" in printed
    assert "copy of the +12 h box" in printed
    saved = json.loads((root / "report" / "validation_10h.json").read_text(encoding="utf-8"))
    assert saved["moment"] == "10" and saved["tiers"]["A"]["n"] == 1
    assert saved["copy_baseline"]["all"]["n"] == 1


def test_a_hand_export_with_other_moments_does_not_change_the_projection(
    world, tmp_path: Path
) -> None:
    data_root, _, doc_root = world
    code = main(
        [
            "--data-root",
            str(data_root),
            "--export",
            str(hand_export(world, tmp_path)),
            "--document-root",
            str(doc_root),
            "--out-dir",
            str(tmp_path / "ls"),
            "--report-dir",
            str(tmp_path / "report"),
            "--nir-display",
            "original",
            "--workers",
            "1",
            "--limit-faces",
            "1",
        ]
    )

    summary = json.loads((tmp_path / "report" / "summary.json").read_text(encoding="utf-8"))
    assert code == 0
    assert summary["hand_boxes"]["rgb"] == 2  # the +10 h box is not a source box


def test_validation_needs_a_projection_file(tmp_path: Path, capsys) -> None:
    code = main(["--validate", str(tmp_path / "hand.json")])

    assert code == 1
    assert "--projections" in capsys.readouterr().err


def test_missing_arguments_are_reported(capsys) -> None:
    assert main([]) == 1
    assert "--data-root" in capsys.readouterr().err


def test_a_face_without_its_12h_photo_is_an_error(world, tmp_path: Path, capsys) -> None:
    data_root, _, doc_root = world
    export = write_export(
        tmp_path,
        [task(mango=99, ts=77, box=("damage_rgb", 10.0, 10.0, 5.0, 5.0), size=(WIDTH, HEIGHT))],
    )

    code = main(
        [
            "--data-root",
            str(data_root),
            "--export",
            str(export),
            "--document-root",
            str(doc_root),
            "--out-dir",
            str(tmp_path / "ls"),
            "--report-dir",
            str(tmp_path / "report"),
            "--nir-display",
            "original",
            "--workers",
            "1",
        ]
    )

    assert code == 1
    assert "12 h" in capsys.readouterr().err
