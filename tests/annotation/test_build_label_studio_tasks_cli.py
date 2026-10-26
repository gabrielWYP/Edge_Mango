"""Tests for the build_label_studio_tasks command line entry point."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.build_label_studio_tasks import main
from tests.annotation.synthetic import JORNADA_1, JORNADA_2, make_meta, write_dataset


@pytest.fixture
def dataset(tmp_path: Path) -> tuple[Path, Path]:
    doc_root = tmp_path / "data"
    data_root = doc_root / "cache" / "captura_v3"
    metas = [
        make_meta(1, mango_id=1, cara=1, ronda_h=12),
        make_meta(2, mango_id=1, cara=2, ronda_h=12),
        make_meta(3, mango_id=1, cara=1, ronda_h=10),
        make_meta(4, mango_id=1, cara=1, ronda_h=None),
        make_meta(5, jornada=JORNADA_2, mango_id=26, cara=1, ronda_h=12),
    ]
    write_dataset(data_root, metas)
    return doc_root, data_root


def run(dataset, out: Path, *extra: str) -> int:
    doc_root, data_root = dataset
    return main(
        [
            "--data-root", str(data_root),
            "--document-root", str(doc_root),
            "--out", str(out),
            *extra,
        ]
    )


def test_writes_tasks_and_prints_a_summary(dataset, tmp_path: Path, capsys) -> None:
    out = tmp_path / "out" / "tasks.json"

    assert run(dataset, out, "--moments", "12") == 0

    tasks = json.loads(out.read_text(encoding="utf-8"))
    assert [t["data"]["timestamp"] for t in tasks] == [1, 2, 5]
    printed = capsys.readouterr().out
    assert "Tasks: 3" in printed
    assert str(out) in printed


def test_moments_default_to_12h(dataset, tmp_path: Path) -> None:
    out = tmp_path / "tasks.json"

    assert run(dataset, out) == 0

    tasks = json.loads(out.read_text(encoding="utf-8"))
    assert {t["data"]["ronda_h"] for t in tasks} == {12}


def test_several_moments_including_sano(dataset, tmp_path: Path) -> None:
    out = tmp_path / "tasks.json"

    assert run(dataset, out, "--moments", "12,10,sano") == 0

    tasks = json.loads(out.read_text(encoding="utf-8"))
    assert [t["data"]["ronda_h"] for t in tasks[:3]] == [None, 10, 12]


def test_jornadas_option_filters_lots(dataset, tmp_path: Path) -> None:
    out = tmp_path / "tasks.json"

    assert run(dataset, out, "--jornadas", JORNADA_1) == 0

    tasks = json.loads(out.read_text(encoding="utf-8"))
    assert {t["data"]["jornada"] for t in tasks} == {JORNADA_1}


def test_errors_exit_nonzero_without_writing(tmp_path: Path, capsys) -> None:
    data_root = tmp_path / "elsewhere" / "captura_v3"
    write_dataset(data_root, [make_meta(1)])
    doc_root = tmp_path / "data"
    doc_root.mkdir()
    out = tmp_path / "tasks.json"

    code = main(
        [
            "--data-root", str(data_root),
            "--document-root", str(doc_root),
            "--out", str(out),
        ]
    )

    assert code == 1
    assert "outside the document root" in capsys.readouterr().err
    assert not out.exists()


def test_invalid_moments_exit_nonzero(dataset, tmp_path: Path, capsys) -> None:
    out = tmp_path / "tasks.json"

    assert run(dataset, out, "--moments", "twelve") == 1
    assert "twelve" in capsys.readouterr().err
    assert not out.exists()


def test_document_root_defaults_to_the_folder_above_cache(dataset, tmp_path: Path) -> None:
    _, data_root = dataset
    out = tmp_path / "tasks.json"

    assert main(["--data-root", str(data_root), "--out", str(out)]) == 0

    tasks = json.loads(out.read_text(encoding="utf-8"))
    assert tasks[0]["data"]["rgb"].startswith(
        "/data/local-files/?d=cache/captura_v3/news/rgb/"
    )


def test_ambiguous_document_root_asks_for_it_explicitly(tmp_path: Path, capsys) -> None:
    data_root = tmp_path / "elsewhere" / "captura_v3"
    write_dataset(data_root, [make_meta(1)])
    out = tmp_path / "tasks.json"

    assert main(["--data-root", str(data_root), "--out", str(out)]) == 1
    assert "--document-root" in capsys.readouterr().err
    assert not out.exists()
