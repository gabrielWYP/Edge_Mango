"""Tests for the NIR line artifact detector and interpolation (plan 3.1 rule 4)."""

from __future__ import annotations

import numpy as np
import pytest

from src.data_pipeline.nir_lines import HALO, NirLine, detect_lines, interpolate_lines


def _sheet(value: float = 60.0, shape: tuple[int, int] = (120, 800)) -> np.ndarray:
    rng = np.random.default_rng(0)
    return np.clip(value + rng.normal(0, 3, size=shape), 0, 255).astype(np.float32)


def _draw_line(gray: np.ndarray, row: int, x0: int, x1: int) -> None:
    """A dropped row stretch plus its JPEG halo on the rows next to it."""
    gray[row, x0:x1] = 8
    gray[[row - 1, row + 1], x0:x1] = 35


class TestDetectLines:
    def test_finds_a_dropped_row_stretch(self) -> None:
        gray = _sheet()
        _draw_line(gray, 60, 200, 480)
        (line,) = detect_lines(gray)
        assert line.row == 60
        assert (line.x0, line.x1) == pytest.approx((200, 480), abs=5)

    def test_a_line_on_a_dark_background_still_counts(self) -> None:
        gray = _sheet(value=25.0)
        gray[60, 300:400] = 6
        assert [line.row for line in detect_lines(gray)] == [60]

    def test_rows_of_one_line_merge(self) -> None:
        gray = _sheet()
        gray[60:62, 200:480] = 5
        assert len(detect_lines(gray)) == 1

    @pytest.mark.parametrize(
        "draw",
        [
            pytest.param(lambda g: g.__setitem__(np.s_[60:, :], 20), id="horizontal edge"),
            pytest.param(lambda g: g.__setitem__(np.s_[50:70, 100:600], 5), id="dark region"),
            pytest.param(lambda g: g.__setitem__(np.s_[60, 300:330], 5), id="short segment"),
        ],
    )
    def test_ignores_what_is_not_a_line(self, draw) -> None:
        gray = _sheet()
        draw(gray)
        assert detect_lines(gray) == []


class TestInterpolateLines:
    def test_rebuilds_the_line_rows_and_nothing_else(self) -> None:
        clean = np.tile(np.linspace(40, 140, 120)[:, None, None], (1, 800, 3)).astype(np.uint8)
        broken = clean.copy()
        broken[59:62, 200:480] = 8
        line = NirLine(60, 200, 480)
        fixed = interpolate_lines(broken, [line])
        rebuilt = np.s_[60 - HALO : 60 + HALO + 1, 190:490]
        assert np.abs(fixed[rebuilt].astype(int) - clean[rebuilt]).max() <= 1
        untouched = np.ones(clean.shape[:2], dtype=bool)
        untouched[rebuilt] = False
        assert np.array_equal(fixed[untouched], broken[untouched])

    def test_a_line_near_the_border_uses_the_border_row(self) -> None:
        image = np.full((40, 300), 90, dtype=np.uint8)
        image[3, 50:250] = 5
        fixed = interpolate_lines(image, [NirLine(3, 50, 250)])
        assert fixed[3, 50:250].min() == 90

    def test_no_lines_returns_an_equal_image(self) -> None:
        image = _sheet().astype(np.uint8)
        assert np.array_equal(interpolate_lines(image, []), image)
