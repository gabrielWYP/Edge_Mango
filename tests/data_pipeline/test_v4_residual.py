"""Tests for the per-photo residual shift of the fruit, on synthetic images with a known shift."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.annotation.mango_registration import rgb_mango_mask
from src.data_pipeline.v4_residual import (
    MIN_CORR,
    SEARCH_PX,
    SOURCE_FRUIT,
    SOURCE_MEDIAN,
    SOURCE_UNMEASURED,
    FruitMatch,
    measure_fruit_match,
    resolve_shifts,
)
from src.data_pipeline.v4_selection import RGB_SIZE
from tests.data_pipeline.v4_synth import fruit_photos

TOLERANCE_PX = 2.0


def _measure(rgb: np.ndarray, nir: np.ndarray) -> FruitMatch:
    small = cv2.resize(rgb, (RGB_SIZE[0] // 4, RGB_SIZE[1] // 4), interpolation=cv2.INTER_AREA)
    return measure_fruit_match(cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY), nir, rgb_mango_mask(small))


def _central_rectangle_shift(rgb: np.ndarray, nir: np.ndarray) -> tuple[float, float]:
    """The finding's measure, where the sheet also votes: here to prove the fixture bites."""

    def gradient(image: np.ndarray) -> np.ndarray:
        small = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        smooth = cv2.GaussianBlur(small, (0, 0), 1.5)
        return cv2.magnitude(
            cv2.Sobel(smooth, cv2.CV_32F, 1, 0), cv2.Sobel(smooth, cv2.CV_32F, 0, 1)
        )

    rgb_grad = gradient(cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY))
    nir_grad = gradient(nir)
    height, width = rgb_grad.shape
    top, bottom = int(0.2 * height), int(0.85 * height)
    left, right = int(0.3 * width), int(0.8 * width)
    margin = SEARCH_PX // 2
    scores = cv2.matchTemplate(
        rgb_grad[top - margin : bottom + margin, left - margin : right + margin],
        nir_grad[top:bottom, left:right],
        cv2.TM_CCOEFF_NORMED,
    )
    _, _, _, (x, y) = cv2.minMaxLoc(scores)
    return (x - margin) * 2.0, (y - margin) * 2.0


class TestMeasure:
    @pytest.mark.parametrize("shift", [(84, 20), (-56, 4), (0, 0)])
    def test_recovers_a_known_shift(self, shift: tuple[int, int]) -> None:
        match = _measure(*fruit_photos(shift))
        assert (match.dx, match.dy) == pytest.approx(shift, abs=TOLERANCE_PX)
        assert match.reliable and not match.at_limit
        assert match.corr > MIN_CORR

    def test_reports_the_correlation_at_zero_shift_too(self) -> None:
        match = _measure(*fruit_photos((84, 20)))
        assert match.corr_zero < match.corr - 0.2

    def test_sheet_and_shadow_outside_the_fruit_do_not_vote(self) -> None:
        rgb, nir = fruit_photos((84, 20), sheet_clutter=True)
        assert _central_rectangle_shift(rgb, nir) == pytest.approx((0, 0), abs=TOLERANCE_PX)
        match = _measure(rgb, nir)
        assert (match.dx, match.dy) == pytest.approx((84, 20), abs=TOLERANCE_PX)

    def test_a_best_shift_at_the_search_limit_is_unreliable(self) -> None:
        match = _measure(*fruit_photos((SEARCH_PX, 0)))
        assert match.at_limit
        assert not match.reliable

    def test_an_unrelated_nir_is_unreliable(self) -> None:
        rgb, _ = fruit_photos()
        noise = np.random.default_rng(3).normal(120, 25, (RGB_SIZE[1], RGB_SIZE[0]))
        match = _measure(rgb, np.clip(noise, 0, 255).astype(np.uint8))
        assert match.corr < MIN_CORR
        assert not match.reliable

    def test_marks_on_the_skin_measure_the_parallax_against_the_outline(self) -> None:
        match = _measure(*fruit_photos((84, 20), surface=(50, 0)))
        assert (match.dx, match.dy) == pytest.approx((84, 20), abs=TOLERANCE_PX)
        assert match.surface == pytest.approx((50, 0), abs=TOLERANCE_PX)

    def test_a_fruit_with_a_blank_skin_has_no_surface_match(self) -> None:
        width, height = RGB_SIZE
        rgb = np.full((height, width, 3), (230, 220, 150), dtype=np.uint8)
        nir = np.full((height, width), 190, dtype=np.uint8)
        cv2.ellipse(rgb, (1600, 1200), (500, 400), 0, 0, 360, (30, 200, 220), -1)
        cv2.ellipse(nir, (1516, 1180), (500, 400), 0, 0, 360, 100, -1)
        match = _measure(rgb, nir)
        assert (match.dx, match.dy) == pytest.approx((84, 20), abs=TOLERANCE_PX)
        assert match.surface is None


def _match(dx: float, dy: float, corr: float = 0.6, at_limit: bool = False, surface=(0.0, 0.0)):
    return FruitMatch(dx, dy, corr, 0.1, at_limit, surface)


class TestResolve:
    def test_a_reliable_match_keeps_its_own_shift(self) -> None:
        shifts, _ = resolve_shifts([(1, "12")] * 3, [_match(10, 2), _match(20, 4), _match(90, 8)])
        assert [(s.dx, s.dy, s.source) for s in shifts] == [
            (10, 2, SOURCE_FRUIT),
            (20, 4, SOURCE_FRUIT),
            (90, 8, SOURCE_FRUIT),
        ]
        assert shifts[0].corr == 0.6

    @pytest.mark.parametrize(
        "bad",
        [
            _match(300, 300, corr=MIN_CORR - 0.01),
            _match(160, 0, at_limit=True),
        ],
    )
    def test_an_unreliable_match_falls_back_to_the_median_of_its_group(
        self, bad: FruitMatch
    ) -> None:
        keys = [(1, "12"), (1, "12"), (1, "12"), (1, "12"), (1, "10"), (2, "12")]
        matches = [_match(10, 2), bad, _match(20, 4), _match(90, 8), _match(500, 1), _match(600, 1)]
        shifts, _ = resolve_shifts(keys, matches)
        assert (shifts[1].dx, shifts[1].dy, shifts[1].source) == (20, 4, SOURCE_MEDIAN)
        assert shifts[1].corr == round(bad.corr, 3)

    def test_the_fallback_does_not_depend_on_the_photo_order(self) -> None:
        matches = [_match(10, 2), _match(20, 4), _match(0, 0, corr=0.0), _match(90, 8)]
        keys = [(1, "12")] * 4
        forward, _ = resolve_shifts(keys, matches)
        backward, _ = resolve_shifts(keys, matches[::-1])
        assert forward[2] == backward[1]

    def test_a_group_without_an_accepted_photo_stays_unshifted_and_says_so(self) -> None:
        shifts, summary = resolve_shifts(
            [(1, "12"), (1, "10")], [_match(50, 50, corr=0.0), _match(10, 2)]
        )
        assert (shifts[0].dx, shifts[0].dy, shifts[0].source) == (0, 0, SOURCE_UNMEASURED)
        assert summary[(1, "12")]["unmeasured"] == 1
        assert summary[(1, "12")]["accepted"] == 0

    def test_the_group_parallax_is_added_to_each_photo_and_to_the_fallback(self) -> None:
        matches = [
            _match(10, 2, surface=(40, 0)),
            _match(20, 4, surface=(50, 2)),
            _match(90, 8, surface=(300, 0)),  # a false peak does not move the median
            _match(0, 0, corr=0.0),
        ]
        shifts, summary = resolve_shifts([(1, "12")] * 4, matches)
        assert (shifts[0].dx, shifts[0].dy) == (60, 2)
        assert (shifts[1].dx, shifts[1].dy) == (70, 4)
        assert (shifts[3].dx, shifts[3].dy, shifts[3].source) == (70, 4, SOURCE_MEDIAN)
        assert (summary[(1, "12")]["parallax_dx"], summary[(1, "12")]["parallax_dy"]) == (50, 0)

    def test_an_unreliable_photo_does_not_vote_for_the_parallax(self) -> None:
        matches = [
            _match(10, 2, surface=(40, 0)),
            _match(0, 0, corr=0.0, surface=(140, 16)),
            _match(10, 2, corr=0.0, surface=(140, 16)),
        ]
        _, summary = resolve_shifts([(1, "12")] * 3, matches)
        assert (summary[(1, "12")]["parallax_dx"], summary[(1, "12")]["parallax_dy"]) == (40, 0)

    def test_a_group_without_surface_matches_has_no_parallax(self) -> None:
        shifts, summary = resolve_shifts([(1, "12")], [_match(10, 2, surface=None)])
        assert (shifts[0].dx, shifts[0].dy) == (10, 2)
        assert summary[(1, "12")]["parallax_dx"] == 0

    def test_the_summary_counts_and_medians_per_group(self) -> None:
        keys = [(1, "12"), (1, "12"), (1, "12"), (1, "10")]
        matches = [_match(10, 2), _match(20, 4), _match(0, 0, corr=0.0), _match(30, 6)]
        _, summary = resolve_shifts(keys, matches)
        assert summary[(1, "12")] == {
            "median_dx": 15.0,
            "median_dy": 3.0,
            "parallax_dx": 0.0,
            "parallax_dy": 0.0,
            "accepted": 2,
            "fallback": 1,
            "unmeasured": 0,
            "median_corr": 0.6,
            "median_corr_zero": 0.1,
        }
        assert summary[(1, "10")]["accepted"] == 1

    def test_a_photo_without_a_mango_has_no_shift_and_does_not_count(self) -> None:
        shifts, summary = resolve_shifts([(1, "12"), (1, "12")], [None, _match(10, 2)])
        assert shifts[0] is None
        assert summary[(1, "12")]["accepted"] == 1
        assert summary[(1, "12")]["fallback"] == 0
