"""Tests for the read-only +12 h reference images shown next to the +10 h and earlier tasks."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from src.annotation.reference_images import (
    BOX_COLOR,
    NO_REFERENCE_NAME,
    PLACEHOLDER_SIZE,
    REF_DIR,
    ReferenceJob,
    box_thickness,
    build_reference_images,
    collect_reference_jobs,
    render_placeholder,
    render_reference,
)
from tests.annotation.export_synth import H, W, task
from tests.annotation.synthetic import JORNADA_1, JORNADA_2

URL = "/data/local-files/?d="
NIR_SRC = "cache/captura_v3/news/nir_vis/mango_nir_{ts}.jpg"
RGB_SRC = "cache/captura_v3/news/rgb/mango_rgb_{ts}.jpg"
IMAGE_SIZE = (328, 247)  # width, height: about a tenth of the 3280x2464 originals


def ref_url(name: str) -> str:
    return f"{URL}{REF_DIR.as_posix()}/{name}"


def face_task(ts: int, hour: int | None, *, ref: str | None, rgb: str | None = None, **kwargs) -> dict:
    """A task with the display fields of the real export (``rgb``, ``nir``, ``ref``)."""
    item = task(ts=ts, hour=hour, **kwargs)
    item["data"]["nir"] = URL + NIR_SRC.format(ts=ts)
    item["data"]["rgb"] = URL + (rgb or RGB_SRC.format(ts=ts))
    if ref is not None:
        item["data"]["ref"] = ref_url(ref)
    return item


def nir_face(**kwargs) -> list[dict]:
    """One face: its +12 h task (box on NIR) and two earlier rounds that point at it."""
    return [
        face_task(100, 12, ref="mango_nir_100.jpg", **kwargs),
        face_task(101, 10, ref=NO_REFERENCE_NAME, annotated=False),
        face_task(102, 8, ref="mango_nir_100.jpg", annotated=False),
    ]


def by_ref(jobs: list[ReferenceJob]) -> dict[str, ReferenceJob]:
    return {job.ref.name: job for job in jobs}


def test_nir_reference_points_at_the_nir_image_of_the_12h_task() -> None:
    jobs = by_ref(collect_reference_jobs(nir_face()))

    job = jobs["mango_nir_100.jpg"]
    assert job.ref == REF_DIR / "mango_nir_100.jpg"
    assert job.source == Path(NIR_SRC.format(ts=100))
    assert job.box_size == (W, H)
    x, y, w, h = job.box
    assert (x, y, w, h) == pytest.approx((0.10 * W, 0.20 * H, 0.08 * W, 0.06 * H))


def test_rgb_reference_reuses_the_displayed_white_balanced_rgb() -> None:
    wb = "cache/captura_v3/news/rgb_wb/mango_rgb_100.jpg"
    tasks = [
        face_task(100, 12, ref="mango_rgb_100.jpg", rgb=wb, box=("damage_rgb", 5.0, 5.0, 10.0, 10.0)),
        face_task(102, 8, ref="mango_rgb_100.jpg", annotated=False),
    ]

    (job,) = collect_reference_jobs(tasks)

    assert job.source == Path(wb)
    assert job.box == pytest.approx((0.05 * W, 0.05 * H, 0.10 * W, 0.10 * H))


def test_placeholder_job_has_no_source_and_no_box() -> None:
    jobs = by_ref(collect_reference_jobs(nir_face()))

    placeholder = jobs[NO_REFERENCE_NAME]
    assert placeholder.ref == REF_DIR / NO_REFERENCE_NAME
    assert placeholder.source is None
    assert placeholder.box is None


def test_one_job_per_reference_file_in_a_stable_order() -> None:
    tasks = nir_face() + [
        face_task(200, 12, ref="mango_nir_200.jpg", cara=2),
        face_task(201, 10, ref=NO_REFERENCE_NAME, cara=2, annotated=False),
    ]

    names = [job.ref.name for job in collect_reference_jobs(tasks)]

    assert names == ["mango_nir_100.jpg", "mango_nir_200.jpg", NO_REFERENCE_NAME]


def test_tasks_without_ref_are_ignored() -> None:
    tasks = [face_task(100, 12, ref=None), face_task(101, 10, ref=None, annotated=False)]

    assert collect_reference_jobs(tasks) == []


def test_faces_are_told_apart_by_jornada_mango_and_cara() -> None:
    tasks = [
        face_task(100, 12, ref="mango_nir_100.jpg", jornada=JORNADA_1, mango=1, cara=1),
        face_task(300, 12, ref="mango_nir_300.jpg", jornada=JORNADA_2, mango=1, cara=1),
        face_task(301, 8, ref="mango_nir_300.jpg", jornada=JORNADA_2, mango=1, cara=1, annotated=False),
    ]

    names = [job.ref.name for job in collect_reference_jobs(tasks)]

    assert names == ["mango_nir_100.jpg", "mango_nir_300.jpg"]


def test_latest_non_cancelled_annotation_gives_the_box() -> None:
    item = face_task(100, 12, ref="mango_nir_100.jpg", box=("damage_nir", 10.0, 10.0, 10.0, 10.0))
    old = item["annotations"][0]
    old["updated_at"] = "2026-10-26T10:00:00Z"
    newer = task(ts=100, box=("damage_nir", 50.0, 50.0, 10.0, 10.0))["annotations"][0]
    newer["updated_at"] = "2026-10-26T11:00:00Z"
    cancelled = task(ts=100, box=("damage_nir", 70.0, 70.0, 5.0, 5.0), cancelled=True)["annotations"][0]
    cancelled["updated_at"] = "2026-10-26T12:00:00Z"
    item["annotations"] = [old, newer, cancelled]

    (job,) = collect_reference_jobs([item])

    assert job.box[0] == pytest.approx(0.50 * W)


@pytest.mark.parametrize(
    ("tasks", "message"),
    [
        pytest.param(
            [face_task(102, 8, ref="mango_nir_100.jpg", annotated=False)],
            "no \\+12 h task",
            id="no-12h-task",
        ),
        pytest.param(
            [face_task(100, 12, ref="mango_nir_100.jpg", annotated=False)],
            "no annotation",
            id="12h-unannotated",
        ),
        pytest.param(
            [face_task(100, 12, ref="mango_nir_100.jpg", box=None)],
            "no box",
            id="12h-without-box",
        ),
        pytest.param(
            [face_task(100, 12, ref="mango_rgb_100.jpg")],
            "modality",
            id="prefix-vs-box-modality",
        ),
        pytest.param(
            [face_task(100, 12, ref="mango_nir_999.jpg")],
            "timestamp",
            id="timestamp-mismatch",
        ),
        pytest.param(
            [face_task(100, 12, ref="mango_nir_100.jpg", cancelled=True)],
            "no annotation",
            id="12h-only-cancelled",
        ),
    ],
)
def test_inconsistent_export_names_the_face(tasks: list[dict], message: str) -> None:
    with pytest.raises(ValueError, match=message) as error:
        collect_reference_jobs(tasks)

    assert JORNADA_1 in str(error.value)
    assert "mango 1" in str(error.value)
    assert "face 1" in str(error.value)


def test_two_boxes_on_the_12h_task_are_rejected() -> None:
    item = face_task(100, 12, ref="mango_nir_100.jpg")
    extra = task(ts=100, box=("damage_rgb", 1.0, 1.0, 2.0, 2.0))["annotations"][0]["result"][-1]
    item["annotations"][0]["result"].append(extra)

    with pytest.raises(ValueError, match="2 boxes"):
        collect_reference_jobs([item])


@pytest.mark.parametrize(
    "ref",
    [
        f"{URL}cache/captura_v3/news/ref_12h/other_100.jpg",
        f"{URL}cache/captura_v3/news/ref_12h/../nir/mango_nir_100.jpg",
        f"{URL}cache/captura_v3/news/nir/mango_nir_100.jpg",
        "/somewhere/else/mango_nir_100.jpg",
    ],
)
def test_ref_outside_the_reference_folder_is_rejected(ref: str) -> None:
    item = face_task(100, 12, ref=None)
    item["data"]["ref"] = ref

    with pytest.raises(ValueError, match="ref"):
        collect_reference_jobs([item])


def test_source_that_is_not_a_local_file_url_is_rejected() -> None:
    item = face_task(100, 12, ref="mango_nir_100.jpg")
    item["data"]["nir"] = "https://example.com/mango_nir_100.jpg"

    with pytest.raises(ValueError, match="local-files"):
        collect_reference_jobs([item])


@pytest.mark.parametrize("path", ["../secret.jpg", "/etc/mango_nir_100.jpg", "cache/../../x.jpg"])
def test_source_outside_the_document_root_is_rejected(path: str) -> None:
    item = face_task(100, 12, ref="mango_nir_100.jpg")
    item["data"]["nir"] = f"/data/local-files/?d={path}"

    with pytest.raises(ValueError, match="local-files"):
        collect_reference_jobs([item])


# ---------------------------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------------------------


def gray_image(width: int = 400, height: int = 300) -> np.ndarray:
    return np.full((height, width, 3), 100, dtype=np.uint8)


def test_box_thickness_scales_with_the_image_width() -> None:
    assert box_thickness(3280) == 13
    assert box_thickness(328) == 2
    assert box_thickness(10) == 2


def test_render_reference_outlines_the_box_and_keeps_the_inside() -> None:
    image = gray_image()
    # 100x100 pixel box at (100, 50) in a 400x300 original: same scale as the image.
    out = render_reference(image, (100.0, 50.0, 100.0, 100.0), (400, 300))

    assert out.shape == image.shape
    assert tuple(out[50, 150]) == BOX_COLOR  # top edge
    assert tuple(out[150, 150]) == BOX_COLOR  # bottom edge
    assert tuple(out[100, 100]) == BOX_COLOR  # left edge
    assert tuple(out[100, 200]) == BOX_COLOR  # right edge
    assert tuple(out[100, 150]) == (100, 100, 100)  # inside untouched
    assert tuple(out[10, 10]) == (100, 100, 100)  # outside untouched
    assert tuple(image[50, 150]) == (100, 100, 100)  # input not modified


def test_render_reference_scales_the_box_to_the_image_size() -> None:
    image = gray_image(200, 150)
    # Box of a 400x300 original drawn on a half-size image.
    out = render_reference(image, (100.0, 50.0, 100.0, 100.0), (400, 300))

    assert tuple(out[25, 75]) == BOX_COLOR  # top edge at (50, 25)..(100, 75)
    assert tuple(out[50, 75]) == (100, 100, 100)


def test_render_reference_keeps_a_box_that_touches_the_border() -> None:
    image = gray_image()

    out = render_reference(image, (0.0, 0.0, 400.0, 300.0), (400, 300))

    assert out.shape == image.shape
    assert tuple(out[0, 200]) == BOX_COLOR
    assert tuple(out[299, 200]) == BOX_COLOR


def test_render_reference_handles_a_single_channel_image() -> None:
    gray = np.full((300, 400), 100, dtype=np.uint8)

    out = render_reference(gray, (100.0, 50.0, 100.0, 100.0), (400, 300))

    assert out.shape == (300, 400, 3)
    assert tuple(out[50, 150]) == BOX_COLOR


def test_render_placeholder_is_a_neutral_gray_with_text() -> None:
    out = render_placeholder((640, 480))

    assert out.shape == (480, 640, 3)
    assert np.all(out[..., 0] == out[..., 1]) and np.all(out[..., 1] == out[..., 2])
    background = int(np.bincount(out[..., 0].ravel()).argmax())
    assert 100 <= background <= 160
    text = out[..., 0] != background
    assert text.sum() > 200
    ys, xs = np.nonzero(text)
    # The text sits around the middle of the image.
    assert abs((xs.min() + xs.max()) / 2 - 320) < 40
    assert abs((ys.min() + ys.max()) / 2 - 240) < 40


def test_placeholder_defaults_to_the_capture_size() -> None:
    assert PLACEHOLDER_SIZE == (W, H)


# ---------------------------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------------------------


def write_source(root: Path, relpath: str, ts: int, size: tuple[int, int] = IMAGE_SIZE) -> None:
    rng = np.random.default_rng(ts)
    image = rng.integers(0, 256, size=(size[1], size[0], 3), dtype=np.uint8)
    path = root / relpath
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), image)


@pytest.fixture
def document_root(tmp_path: Path) -> Path:
    root = tmp_path / "data"
    write_source(root, NIR_SRC.format(ts=100), 100)
    write_source(root, NIR_SRC.format(ts=200), 200)
    return root


@pytest.fixture
def jobs() -> list[ReferenceJob]:
    tasks = nir_face() + [face_task(200, 12, ref="mango_nir_200.jpg", cara=2)]
    return collect_reference_jobs(tasks)


def listing(root: Path) -> list[str]:
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())


def test_build_writes_every_reference_in_the_reference_folder(
    document_root: Path, jobs: list[ReferenceJob]
) -> None:
    before = listing(document_root)

    report = build_reference_images(document_root, jobs)

    assert len(report.written) == 3
    assert report.skipped == []
    assert report.out_dir == document_root / REF_DIR
    new = sorted(set(listing(document_root)) - set(before))
    assert new == [
        f"{REF_DIR.as_posix()}/mango_nir_100.jpg",
        f"{REF_DIR.as_posix()}/mango_nir_200.jpg",
        f"{REF_DIR.as_posix()}/no_reference.jpg",
    ]
    with Image.open(document_root / REF_DIR / "mango_nir_100.jpg") as image:
        assert (image.format, image.size) == ("JPEG", IMAGE_SIZE)
    with Image.open(document_root / REF_DIR / NO_REFERENCE_NAME) as image:
        assert (image.format, image.size) == ("JPEG", PLACEHOLDER_SIZE)


def test_written_reference_shows_the_box(document_root: Path, jobs: list[ReferenceJob]) -> None:
    build_reference_images(document_root, jobs)

    out = cv2.imread(str(document_root / REF_DIR / "mango_nir_100.jpg"))
    source = cv2.imread(str(document_root / NIR_SRC.format(ts=100)))
    width, height = IMAGE_SIZE
    top = round(0.20 * height)
    center_x = round((0.10 + 0.04) * width)
    red = out[top, center_x].astype(int)
    assert red[2] > 180 and red[0] < 90 and red[1] < 90
    # Far from the box the photo is the source photo (JPEG round trip only).
    diff = np.abs(out[-20:, -20:].astype(int) - source[-20:, -20:].astype(int))
    assert diff.mean() < 12


def test_rerun_skips_existing_files_unless_overwrite(
    document_root: Path, jobs: list[ReferenceJob]
) -> None:
    build_reference_images(document_root, jobs)
    target = document_root / REF_DIR / "mango_nir_100.jpg"
    target.write_bytes(b"stale")

    again = build_reference_images(document_root, jobs)

    assert again.written == []
    assert len(again.skipped) == 3
    assert target.read_bytes() == b"stale"

    forced = build_reference_images(document_root, jobs, overwrite=True)

    assert len(forced.written) == 3
    assert target.read_bytes() != b"stale"


def test_missing_source_fails_before_writing_anything(
    document_root: Path, jobs: list[ReferenceJob]
) -> None:
    (document_root / NIR_SRC.format(ts=200)).unlink()
    before = listing(document_root)

    with pytest.raises(FileNotFoundError, match="mango_nir_200.jpg"):
        build_reference_images(document_root, jobs)

    assert listing(document_root) == before


def test_unreadable_source_is_an_error(document_root: Path, jobs: list[ReferenceJob]) -> None:
    (document_root / NIR_SRC.format(ts=100)).write_bytes(b"broken")

    with pytest.raises(ValueError, match="mango_nir_100.jpg"):
        build_reference_images(document_root, jobs)


def test_refuses_outputs_outside_the_reference_folder(document_root: Path) -> None:
    sneaky = ReferenceJob(
        ref=Path("cache/captura_v3/news/nir/mango_nir_100.jpg"),
        source=Path(NIR_SRC.format(ts=100)),
        box=(1.0, 1.0, 5.0, 5.0),
        box_size=(W, H),
    )
    before = (document_root / sneaky.ref).read_bytes() if (document_root / sneaky.ref).exists() else None

    with pytest.raises(ValueError, match="ref_12h"):
        build_reference_images(document_root, [sneaky], overwrite=True)

    assert not (document_root / REF_DIR).exists()
    after = (document_root / sneaky.ref).read_bytes() if (document_root / sneaky.ref).exists() else None
    assert after == before


def test_refuses_a_reference_folder_that_links_elsewhere(
    document_root: Path, jobs: list[ReferenceJob], tmp_path: Path
) -> None:
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (document_root / REF_DIR).parent.mkdir(parents=True, exist_ok=True)
    (document_root / REF_DIR).symlink_to(elsewhere, target_is_directory=True)

    with pytest.raises(ValueError, match="outside"):
        build_reference_images(document_root, jobs)

    assert list(elsewhere.iterdir()) == []
