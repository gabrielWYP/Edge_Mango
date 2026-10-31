"""Tests for the build_reference_images command line entry point."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from scripts.build_reference_images import main
from src.annotation.reference_images import NO_REFERENCE_NAME, REF_DIR
from tests.annotation.test_reference_images import (
    IMAGE_SIZE,
    NIR_SRC,
    face_task,
    nir_face,
    write_source,
)


@pytest.fixture
def document_root(tmp_path: Path) -> Path:
    root = tmp_path / "data"
    write_source(root, NIR_SRC.format(ts=100), 100)
    write_source(root, NIR_SRC.format(ts=200), 200)
    return root


@pytest.fixture
def labels(tmp_path: Path) -> Path:
    tasks = nir_face() + [
        face_task(200, 12, ref="mango_nir_200.jpg", cara=2),
        face_task(201, 8, ref="mango_nir_200.jpg", cara=2, annotated=False),
    ]
    path = tmp_path / "export.json"
    path.write_text(json.dumps(tasks))
    return path


def run(labels: Path, document_root: Path, *extra: str) -> int:
    return main(["--labels", str(labels), "--document-root", str(document_root), *extra])


def listing(root: Path) -> list[str]:
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())


def test_writes_the_references_and_reports_the_counts(
    labels: Path, document_root: Path, capsys
) -> None:
    before = set(listing(document_root))

    assert run(labels, document_root) == 0

    printed = capsys.readouterr().out
    assert "Written: 3" in printed
    assert "Skipped: 0" in printed
    ref_dir = REF_DIR.as_posix()
    assert set(listing(document_root)) - before == {
        f"{ref_dir}/mango_nir_100.jpg",
        f"{ref_dir}/mango_nir_200.jpg",
        f"{ref_dir}/{NO_REFERENCE_NAME}",
    }
    with Image.open(document_root / REF_DIR / "mango_nir_100.jpg") as image:
        assert image.size == IMAGE_SIZE


def test_rerun_skips_existing_files_unless_overwrite(
    labels: Path, document_root: Path, capsys
) -> None:
    run(labels, document_root)
    capsys.readouterr()

    assert run(labels, document_root) == 0
    printed = capsys.readouterr().out
    assert "Written: 0" in printed
    assert "Skipped: 3" in printed

    assert run(labels, document_root, "--overwrite") == 0
    printed = capsys.readouterr().out
    assert "Written: 3" in printed
    assert "Skipped: 0" in printed


def test_missing_source_image_exits_nonzero_and_writes_nothing(
    labels: Path, document_root: Path, capsys
) -> None:
    (document_root / NIR_SRC.format(ts=200)).unlink()
    before = listing(document_root)

    assert run(labels, document_root) == 1

    assert "mango_nir_200.jpg" in capsys.readouterr().err
    assert listing(document_root) == before


def test_inconsistent_export_exits_nonzero_naming_the_face(
    tmp_path: Path, document_root: Path, capsys
) -> None:
    # The +12 h task of the face is missing from the export.
    broken = tmp_path / "broken.json"
    broken.write_text(json.dumps([nir_face()[2]]))

    assert run(broken, document_root) == 1

    error = capsys.readouterr().err
    assert error.startswith("error:")
    assert "mango 1" in error
    assert not (document_root / REF_DIR).exists()


def test_missing_labels_file_exits_nonzero(tmp_path: Path, document_root: Path, capsys) -> None:
    assert run(tmp_path / "absent.json", document_root) == 1
    assert "absent.json" in capsys.readouterr().err


def test_document_root_is_required(labels: Path) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["--labels", str(labels)])

    assert exit_info.value.code == 2
