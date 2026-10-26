"""Tests for the build_nir_display command line entry point."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from scripts.build_nir_display import main
from tests.annotation.synthetic import JORNADA_2, make_meta, write_dataset


@pytest.fixture
def data_root(tmp_path: Path) -> Path:
    metas = [
        make_meta(1, mango_id=1, cara=1, ronda_h=12),
        make_meta(2, mango_id=1, cara=2, ronda_h=12),
        make_meta(3, mango_id=1, cara=1, ronda_h=10),
        make_meta(4, jornada=JORNADA_2, mango_id=26, cara=1, ronda_h=12),
        # Same slot as 4, taken late: the duplicate rule must drop it.
        make_meta(
            5, jornada=JORNADA_2, mango_id=26, cara=1, ronda_h=12, en_tolerancia=False
        ),
    ]
    return write_dataset(
        tmp_path / "data" / "cache" / "captura_v3", metas, nir_shape=(61, 97)
    )


def written(data_root: Path) -> list[str]:
    folder = data_root / "news" / "nir_vis"
    return sorted(p.name for p in folder.glob("*.jpg")) if folder.exists() else []


def test_writes_images_for_the_selected_moment_only(data_root: Path, capsys) -> None:
    assert main(["--data-root", str(data_root), "--moments", "12"]) == 0

    assert written(data_root) == [
        "mango_nir_1.jpg",
        "mango_nir_2.jpg",
        "mango_nir_4.jpg",
    ]
    printed = capsys.readouterr().out
    assert "Selected: 3" in printed
    assert "Written: 3" in printed
    assert "Skipped: 0" in printed
    with Image.open(data_root / "news" / "nir_vis" / "mango_nir_1.jpg") as image:
        assert (image.mode, image.size) == ("L", (97, 61))


def test_moments_default_to_12h(data_root: Path) -> None:
    assert main(["--data-root", str(data_root)]) == 0

    assert "mango_nir_3.jpg" not in written(data_root)


def test_jornadas_and_several_moments(data_root: Path) -> None:
    assert main(["--data-root", str(data_root), "--moments", "10,12", "--jornadas", "1"]) == 0

    assert written(data_root) == ["mango_nir_1.jpg", "mango_nir_2.jpg", "mango_nir_3.jpg"]


def test_rerun_skips_existing_files_unless_overwrite(data_root: Path, capsys) -> None:
    main(["--data-root", str(data_root)])
    capsys.readouterr()

    assert main(["--data-root", str(data_root)]) == 0
    assert "Skipped: 3" in capsys.readouterr().out

    assert main(["--data-root", str(data_root), "--overwrite"]) == 0
    printed = capsys.readouterr().out
    assert "Written: 3" in printed
    assert "Skipped: 0" in printed


def test_invalid_moments_exit_nonzero(data_root: Path, capsys) -> None:
    assert main(["--data-root", str(data_root), "--moments", "twelve"]) == 1
    assert "twelve" in capsys.readouterr().err
    assert written(data_root) == []


def test_unreadable_source_exits_nonzero(data_root: Path, capsys) -> None:
    (data_root / "news" / "nir" / "mango_nir_2.jpg").write_bytes(b"broken")

    assert main(["--data-root", str(data_root)]) == 1
    assert "mango_nir_2.jpg" in capsys.readouterr().err
