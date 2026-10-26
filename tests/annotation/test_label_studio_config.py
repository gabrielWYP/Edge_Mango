"""Tests for the captura_v3 Label Studio labeling config."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from src.annotation.captura_v3_tasks import build_tasks
from tests.annotation.synthetic import make_meta, write_dataset

CONFIG_PATH = (
    Path(__file__).resolve().parents[2]
    / "configs"
    / "label_studio"
    / "captura_v3_damage.xml"
)


@pytest.fixture(scope="module")
def config_text() -> str:
    return CONFIG_PATH.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def root(config_text: str) -> ET.Element:
    return ET.fromstring(config_text)


def by_name(root: ET.Element, name: str) -> ET.Element:
    matches = [el for el in root.iter() if el.get("name") == name]
    assert len(matches) == 1, f"expected exactly one tag named {name!r}"
    return matches[0]


def test_config_passes_label_studio_validation(
    config_text: str, tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("LABEL_STUDIO_BASE_DATA_DIR", str(tmp_path / "ls_data"))
    label_config = pytest.importorskip("label_studio.core.label_config")

    label_config.validate_label_config(config_text)


def test_label_studio_validator_rejects_a_broken_config(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("LABEL_STUDIO_BASE_DATA_DIR", str(tmp_path / "ls_data"))
    label_config = pytest.importorskip("label_studio.core.label_config")
    broken = (
        '<View><RectangleLabels name="a" toName="missing">'
        '<Label value="damage"/></RectangleLabels></View>'
    )

    with pytest.raises(Exception, match="toName"):
        label_config.validate_label_config(broken)


def test_images_have_zoom_and_brightness_contrast_controls(root: ET.Element) -> None:
    for name, field in (("rgb", "$rgb"), ("nir", "$nir")):
        image = by_name(root, name)
        assert image.tag == "Image"
        assert image.get("value") == field
        assert image.get("zoom") == "true"
        assert image.get("zoomControl") == "true"
        assert image.get("brightnessControl") == "true"
        assert image.get("contrastControl") == "true"


def test_images_sit_side_by_side_in_a_flex_view(root: ET.Element) -> None:
    flex = [
        view
        for view in root.iter("View")
        if "display: flex" in (view.get("style") or "")
    ]
    assert len(flex) == 1
    names = {el.get("name") for el in flex[0].iter() if el.tag == "Image"}
    assert names == {"rgb", "nir"}


def test_header_and_info_are_shown_above_the_images(root: ET.Element) -> None:
    headers = [el.get("value") for el in root.findall("Header")]
    assert headers[:2] == ["$header", "$info"]


@pytest.mark.parametrize(
    ("name", "to_name"), [("damage_rgb", "rgb"), ("damage_nir", "nir")]
)
def test_each_image_has_a_single_damage_box_label(
    root: ET.Element, name: str, to_name: str
) -> None:
    control = by_name(root, name)
    assert control.tag == "RectangleLabels"
    assert control.get("toName") == to_name
    assert [label.get("value") for label in control.findall("Label")] == ["damage"]


@pytest.mark.parametrize(
    ("name", "to_name"), [("visible_rgb", "rgb"), ("visible_nir", "nir")]
)
def test_visibility_questions_are_required_single_choice(
    root: ET.Element, name: str, to_name: str
) -> None:
    control = by_name(root, name)
    assert control.tag == "Choices"
    assert control.get("toName") == to_name
    assert control.get("required") == "true"
    assert control.get("choice") == "single-radio"
    assert control.get("perRegion") in (None, "false")
    assert [c.get("value") for c in control.findall("Choice")] == [
        "yes",
        "no",
        "unsure",
    ]


def test_notes_are_an_optional_text_area(root: ET.Element) -> None:
    notes = by_name(root, "notes")
    assert notes.tag == "TextArea"
    assert notes.get("required") in (None, "false")


def test_every_task_field_used_by_the_config_is_produced_by_the_builder(
    config_text: str, tmp_path: Path
) -> None:
    doc_root = tmp_path / "data"
    data_root = doc_root / "cache" / "captura_v3"
    write_dataset(data_root, [make_meta(1)])
    task = build_tasks(data_root, doc_root, (12,)).tasks[0]

    used = set(re.findall(r'value="\$(\w+)"', config_text))

    assert used == {"rgb", "nir", "header", "info"}
    assert used <= set(task["data"])
