"""Tests for mango masks and the two independent registration methods."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.annotation.mango_registration import (
    Similarity,
    nir_mango_mask,
    prepare,
    register_dense,
    register_sift,
    rgb_mango_mask,
)
from tests.annotation.fruit_synth import (
    CENTER,
    MOTION,
    SIZE,
    fruit_layer,
    fruit_mask,
    motion_about_center,
    scene,
    sheet,
)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a > 0, b > 0
    return float((a & b).sum() / max((a | b).sum(), 1))


def corner_error(estimate: Similarity, truth: Similarity) -> float:
    corners = np.array([[200.0, 150], [450, 150], [450, 330], [200, 330]])
    return float(np.abs(estimate.apply(corners) - truth.apply(corners)).max())


class TestSimilarity:
    def test_apply_rotates_scales_and_translates(self) -> None:
        sim = Similarity(scale=2.0, angle_deg=90.0, tx=10.0, ty=5.0)

        np.testing.assert_allclose(sim.apply(np.array([[1.0, 0.0]])), [[10.0, 7.0]], atol=1e-9)

    def test_inverse_round_trips(self) -> None:
        sim = Similarity(1.07, -12.0, 33.0, -8.0)
        points = np.array([[10.0, 20.0], [300.0, 40.0]])

        np.testing.assert_allclose(sim.inverse().apply(sim.apply(points)), points, atol=1e-9)

    def test_then_applies_self_first(self) -> None:
        first, second = Similarity(1.1, 5.0, 3.0, 4.0), Similarity(0.9, -20.0, -7.0, 2.0)
        points = np.array([[12.0, 34.0]])

        np.testing.assert_allclose(
            first.then(second).apply(points), second.apply(first.apply(points)), atol=1e-9
        )

    def test_from_matrix_recovers_parameters(self) -> None:
        sim = Similarity(1.2, 15.0, 4.0, 9.0)

        back = Similarity.from_matrix(sim.matrix())

        assert back.scale == pytest.approx(1.2)
        assert back.angle_deg == pytest.approx(15.0)
        assert (back.tx, back.ty) == pytest.approx((4.0, 9.0))

    def test_rescaled_matches_images_resized_by_the_factor(self) -> None:
        sim = Similarity(1.05, 7.0, 40.0, -20.0)
        point = np.array([[100.0, 60.0]])

        half = sim.rescaled(0.5)

        np.testing.assert_allclose(half.apply(point * 0.5), sim.apply(point) * 0.5, atol=1e-9)

    def test_cv_matrix_moves_a_pixel_where_apply_says(self) -> None:
        sim = Similarity(1.0, 0.0, 20.0, 10.0)
        image = np.zeros((60, 80), np.uint8)
        image[11, 15] = 255  # pixel centre at u = (15.5, 11.5)

        warped = cv2.warpAffine(image, sim.cv_matrix(), (80, 60), flags=cv2.INTER_NEAREST)

        ys, xs = np.nonzero(warped)
        assert (xs[0] + 0.5, ys[0] + 0.5) == pytest.approx(
            tuple(sim.apply(np.array([[15.5, 11.5]]))[0])
        )


class TestMasks:
    def test_rgb_mask_selects_the_yellow_fruit_not_the_cyan_sheet(self) -> None:
        bgr = np.zeros((300, 400, 3), np.uint8)
        bgr[20:280, 10:390] = (190, 200, 90)  # cyan-green sheet (B, G, R)
        cv2.ellipse(bgr, (130, 150), (60, 40), 0, 0, 360, (90, 110, 50), -1)  # shadow on the sheet
        cv2.ellipse(bgr, (200, 150), (70, 55), 0, 0, 360, (40, 170, 190), -1)  # yellow fruit
        truth = np.zeros((300, 400), np.uint8)
        cv2.ellipse(truth, (200, 150), (70, 55), 0, 0, 360, 255, -1)

        mask = rgb_mango_mask(bgr)

        assert iou(mask, truth) > 0.85
        assert mask[5, 5] == 0 and mask[290, 200] == 0

    def test_nir_mask_covers_fruit_and_shadow_but_not_the_sheet_or_its_edges(self) -> None:
        h, w = 240, 320
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        empty = 150 + 60 * np.exp(-(((xx - 200) / 150) ** 2 + ((yy - 120) / 100) ** 2))
        empty[:8] = 3  # dark band, as at the tray edge
        image = empty * 0.9  # illumination changed between the two shots
        shadow = np.zeros((h, w), np.float32)
        cv2.ellipse(shadow, (110, 125), (60, 38), 0, 0, 360, 1.0, -1)
        image *= 1 - 0.7 * cv2.GaussianBlur(shadow, (0, 0), 3)
        fruit = np.zeros((h, w), np.uint8)
        cv2.ellipse(fruit, (150, 120), (40, 36), 0, 0, 360, 255, -1)
        image[fruit > 0] = 120

        mask = nir_mango_mask(image, empty)

        assert (mask[fruit > 0] > 0).mean() > 0.95
        assert mask[200:, 250:].sum() == 0
        assert mask[:20].sum() == 0


@pytest.fixture(scope="module")
def reference():
    texture, mask = fruit_layer(seed=1)
    background = sheet(seed=2)
    return texture, mask, background


def prepared_pair(reference, motion: Similarity | None = MOTION, other_texture=None):
    texture, mask, background = reference
    ref_image, ref_mask = scene(background, texture, mask)
    mov_texture = texture if other_texture is None else other_texture
    mov_image, mov_mask = scene(background, mov_texture, mask, motion)
    grow = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    return (
        prepare(ref_image, cv2.dilate(ref_mask, grow), 1.0),
        prepare(mov_image, cv2.dilate(mov_mask, grow), 1.0),
    )


class TestSift:
    def test_recovers_a_known_similarity(self, reference) -> None:
        ref, mov = prepared_pair(reference)

        result = register_sift(ref, mov)

        assert result.valid and result.inliers >= 15
        assert corner_error(result.sim, MOTION) < 3.0
        assert result.rmse < 3.0

    def test_static_background_does_not_pull_the_transform_to_identity(self, reference) -> None:
        texture, mask, background = reference
        ref_image, _ = scene(background, texture, mask)
        mov_image, _ = scene(background, texture, mask, MOTION)
        everything = np.full(SIZE, 255, np.uint8)

        unmasked = register_sift(
            prepare(ref_image, everything, 1.0), prepare(mov_image, everything, 1.0)
        )
        ref, mov = prepared_pair(reference)
        masked = register_sift(ref, mov)

        assert corner_error(unmasked.sim, Similarity()) < 3.0  # the sheet wins without a mask
        assert corner_error(masked.sim, MOTION) < 3.0  # the fruit wins with it

    def test_unrelated_fruits_are_not_valid(self, reference) -> None:
        other, _ = fruit_layer(seed=99)
        ref, mov = prepared_pair(reference, other_texture=other)

        result = register_sift(ref, mov)

        assert not result.valid
        assert result.inliers < 15

    def test_works_at_a_reduced_working_scale(self, reference) -> None:
        texture, mask, background = reference
        ref_image, ref_mask = scene(background, texture, mask)
        mov_image, mov_mask = scene(background, texture, mask, MOTION)

        def reduced(image, m):
            small = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
            small_mask = cv2.resize(m, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
            return prepare(small, small_mask, 0.5)

        result = register_sift(reduced(ref_image, ref_mask), reduced(mov_image, mov_mask))

        assert result.valid
        assert corner_error(result.sim, MOTION) < 4.0

    def test_an_image_without_keypoints_is_invalid(self) -> None:
        flat = np.full(SIZE, 120, np.uint8)
        prepared = prepare(flat, fruit_mask(), 1.0)

        result = register_sift(prepared, prepared)

        assert not result.valid and result.sim is None and result.inliers == 0


class TestDense:
    def test_recovers_a_known_similarity(self, reference) -> None:
        ref, mov = prepared_pair(reference)

        result = register_dense(ref, mov)

        assert result.valid and result.score > 0.6
        assert corner_error(result.sim, MOTION) < 8.0

    def test_recovers_a_pure_shift_and_a_pure_rotation(self, reference) -> None:
        for motion in (
            motion_about_center(1.0, 0.0, (-60.0, 35.0)),
            motion_about_center(1.0, -14.0, (0.0, 0.0)),
        ):
            ref, mov = prepared_pair(reference, motion)

            result = register_dense(ref, mov)

            assert result.valid
            assert corner_error(result.sim, motion) < 8.0

    def test_static_background_does_not_pull_the_transform_to_identity(self, reference) -> None:
        ref, mov = prepared_pair(reference, motion_about_center(1.0, 0.0, (70.0, 0.0)))

        result = register_dense(ref, mov)

        assert result.sim.tx == pytest.approx(70.0, abs=6.0)

    def test_unrelated_fruits_are_not_valid(self, reference) -> None:
        other, _ = fruit_layer(seed=99)
        ref, mov = prepared_pair(reference, other_texture=other)

        result = register_dense(ref, mov)

        assert not result.valid
        assert result.score < 0.45

    def test_min_score_decides_validity(self, reference) -> None:
        ref, mov = prepared_pair(reference)

        assert register_dense(ref, mov, min_score=0.999).valid is False
        assert register_dense(ref, mov, min_score=0.1).valid is True

    def test_works_at_a_reduced_working_scale(self, reference) -> None:
        texture, mask, background = reference
        ref_image, ref_mask = scene(background, texture, mask)
        mov_image, mov_mask = scene(background, texture, mask, MOTION)

        def reduced(image, m):
            small = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
            small_mask = cv2.resize(m, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
            return prepare(small, small_mask, 0.5)

        result = register_dense(reduced(ref_image, ref_mask), reduced(mov_image, mov_mask))

        assert result.valid
        assert corner_error(result.sim, MOTION) < 10.0


def test_center_constant_is_inside_the_frame() -> None:
    assert 0 < CENTER[0] < SIZE[1] and 0 < CENTER[1] < SIZE[0]
