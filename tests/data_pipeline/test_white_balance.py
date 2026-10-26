"""Pure white-balance helpers: sRGB<->linear, border chroma, von Kries gains, mango mask, CIELAB."""

import numpy as np
import pytest

from capture.calidad import region_borde
from src.data_pipeline import white_balance as wb

SHEET = (90, 200, 190)  # sRGB cyan sheet, linear R/G ~0.18
MANGO = (210, 170, 40)


def scene(shape=(120, 160), sheet=SHEET, mango=MANGO):
    """Uniform sheet with a centered mango ellipse (the border frame sees only the sheet)."""
    h, w = shape
    img = np.empty((h, w, 3), dtype=np.uint8)
    img[:] = sheet
    yy, xx = np.mgrid[:h, :w]
    ellipse = ((yy - h / 2) / (0.22 * h)) ** 2 + ((xx - w / 2) / (0.20 * w)) ** 2 <= 1
    img[ellipse] = mango
    return img


# --- sRGB <-> linear ---------------------------------------------------------------

def test_srgb_to_linear_known_values():
    lin = wb.srgb_to_linear(np.array([0, 10, 128, 255], dtype=np.uint8))
    assert lin[0] == 0.0 and lin[3] == pytest.approx(1.0)
    assert lin[1] == pytest.approx(10 / 255 / 12.92)       # linear toe segment
    assert lin[2] == pytest.approx(0.21586, abs=1e-4)      # power segment


def test_linear_roundtrip_is_lossless_for_every_uint8_level():
    levels = np.arange(256, dtype=np.uint8)
    back = wb.linear_to_srgb(wb.srgb_to_linear(levels))
    assert back.dtype == np.uint8
    np.testing.assert_array_equal(back, levels)


def test_linear_to_srgb_clips_out_of_range():
    out = wb.linear_to_srgb(np.array([-0.2, 0.0, 1.0, 3.5]))
    assert out.dtype == np.uint8
    np.testing.assert_array_equal(out, [0, 0, 255, 255])


# --- border means / chroma -----------------------------------------------------------

def test_border_means_use_the_capture_qc_region_in_linear_space():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, size=(100, 200, 3), dtype=np.uint8)
    mask = region_borde(img.shape, 0.10)
    expected = wb.srgb_to_linear(img)[mask].mean(axis=0)
    np.testing.assert_allclose(wb.border_means_linear(img, 0.10), expected)


def test_border_means_ignore_the_center():
    a = scene(mango=(255, 0, 0))
    b = scene(mango=(0, 0, 255))
    np.testing.assert_allclose(wb.border_means_linear(a), wb.border_means_linear(b))


def test_border_means_average_in_linear_not_encoded_space():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:, :50] = 255                      # half the frame white, half black
    mean_lin = wb.border_means_linear(img, 0.5)
    assert mean_lin == pytest.approx([0.5, 0.5, 0.5], abs=0.02)   # encoded mean would be ~0.21 linear


def test_chroma_is_r_over_g_and_b_over_g():
    rg, bg = wb.chroma(np.array([0.2, 0.5, 0.4]))
    assert rg == pytest.approx(0.4) and bg == pytest.approx(0.8)


# --- gains -----------------------------------------------------------------------------

def test_von_kries_gains_are_reference_over_observed():
    gains = wb.von_kries_gains(np.array([0.1, 0.5, 0.4]), np.array([0.4, 0.5, 0.2]))
    np.testing.assert_allclose(gains, [0.25, 1.0, 2.0])


def test_von_kries_gains_reject_non_positive_observed():
    with pytest.raises(ValueError):
        wb.von_kries_gains(np.array([0.1, 0.5, 0.4]), np.array([0.0, 0.5, 0.2]))


def test_a_zero_channel_raises_a_clear_white_balance_error():
    with pytest.raises(wb.WhiteBalanceError, match="zero"):
        wb.chroma(np.array([0.1, 0.0, 0.2]))
    with pytest.raises(wb.WhiteBalanceError, match="zero"):
        wb.von_kries_gains(np.array([0.1, 0.5, 0.4]), np.array([0.0, 0.5, 0.2]))


def test_recovers_a_known_channel_cast_and_matches_reference_chroma():
    reference = scene(mango=(120, 100, 30))          # dim enough that the cast does not clip it
    cast = np.array([4.5, 0.9, 1.1])                 # auto-WB style tint, applied in linear space
    observed, _ = wb.apply_gains(reference, cast)

    gains = wb.von_kries_gains(wb.border_means_linear(reference), wb.border_means_linear(observed))
    np.testing.assert_allclose(gains, 1 / cast, rtol=0.02)

    corrected, clipped = wb.apply_gains(observed, gains)
    assert clipped == 0.0
    ref_rg, ref_bg = wb.chroma(wb.border_means_linear(reference))
    out_rg, out_bg = wb.chroma(wb.border_means_linear(corrected))
    assert out_rg == pytest.approx(ref_rg, rel=0.02)
    assert out_bg == pytest.approx(ref_bg, rel=0.02)
    # the mango (center) is restored too, not only the border
    center = (slice(50, 70), slice(70, 90))
    diff = np.abs(corrected[center].astype(int) - reference[center].astype(int))
    assert diff.max() <= 4


def test_apply_gains_identity_returns_same_pixels_and_a_copy():
    img = scene()
    out, clipped = wb.apply_gains(img, np.ones(3))
    np.testing.assert_array_equal(out, img)
    assert out is not img and clipped == 0.0
    assert out.dtype == np.uint8 and out.shape == img.shape


def test_apply_gains_reports_the_fraction_of_clipped_pixels():
    img = np.full((10, 10, 3), 20, dtype=np.uint8)
    img[:5] = 200                                     # half the pixels are bright
    out, clipped = wb.apply_gains(img, np.array([2.0, 1.0, 1.0]))   # linear 0.578 * 2 > 1 on R
    assert clipped == pytest.approx(0.5)
    assert out[:5, :, 0].min() == 255 and out[:5, :, 1].max() == 200
    assert out[5:, :, 0].max() < 255


def test_apply_gains_does_not_modify_its_input():
    img = scene()
    before = img.copy()
    wb.apply_gains(img, np.array([0.5, 1.2, 1.0]))
    np.testing.assert_array_equal(img, before)


@pytest.mark.parametrize("gains", [[1, 1], [1, 1, 1, 1], [-1, 1, 1], [0, 1, 1], [np.nan, 1, 1]])
def test_apply_gains_rejects_invalid_gains(gains):
    with pytest.raises(ValueError):
        wb.apply_gains(scene(), np.array(gains, dtype=float))


def test_apply_gains_rejects_non_uint8_images():
    with pytest.raises(ValueError):
        wb.apply_gains(scene().astype(np.float32), np.ones(3))


# --- CIELAB / mango mask -----------------------------------------------------------------

def test_srgb_to_lab_known_values():
    lab = wb.srgb_to_lab(np.array([[255, 255, 255], [0, 0, 0], [255, 0, 0]], dtype=np.uint8))
    np.testing.assert_allclose(lab[0], [100, 0, 0], atol=0.1)
    np.testing.assert_allclose(lab[1], [0, 0, 0], atol=0.1)
    np.testing.assert_allclose(lab[2], [53.24, 80.09, 67.20], atol=0.2)   # D65, sRGB red


def test_delta_e76_is_euclidean_distance_in_lab():
    assert wb.delta_e76([50, 10, 10], [50, 13, 14]) == pytest.approx(5.0)


def test_mango_mask_selects_the_fruit_and_not_the_shadow_or_dark_frame():
    img = np.full((120, 160, 3), 8, dtype=np.uint8)               # dark tray frame
    img[6:114, 8:152] = SHEET
    img[60:100, 20:70] = (30, 70, 70)                              # cast shadow: dark cyan
    yy, xx = np.mgrid[:120, :160]
    fruit = ((yy - 55) / 25) ** 2 + ((xx - 85) / 32) ** 2 <= 1
    img[fruit] = MANGO
    img[10:14, 140:144] = MANGO                                    # small stray yellow speck
    mask = wb.mango_mask(img)

    assert mask.dtype == bool and mask.shape == fruit.shape
    inter = (mask & fruit).sum()
    assert inter / fruit.sum() > 0.80                              # most of the fruit (mask is eroded)
    assert inter / mask.sum() > 0.98                               # nothing outside the fruit
    assert not mask[60:100, 20:50].any()                           # shadow excluded
    assert not mask[10:14, 140:144].any()                          # only the largest component


def test_mango_mask_is_empty_when_there_is_no_fruit():
    assert not wb.mango_mask(scene(mango=SHEET)).any()
