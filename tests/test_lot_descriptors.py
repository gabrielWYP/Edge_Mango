"""The per-lot capture measures: fruit mask and damage contrast."""

from __future__ import annotations

import numpy as np

from scripts import lot_descriptors as ld


def test_the_fruit_is_the_ellipse_inscribed_in_the_mango_box() -> None:
    mask = ld.fruit_mask((100, 100), (40.0, 40.0, 360.0, 360.0))  # 10..90 at the reduced scale
    assert mask[50, 50] and not mask[12, 12] and not mask[5, 50]
    assert abs(mask.sum() - np.pi * 40 * 40) / (np.pi * 40 * 40) < 0.05


def test_the_damage_contrast_is_the_box_against_the_rest_of_the_fruit() -> None:
    rng = np.random.default_rng(0)
    channel = rng.normal(100.0, 10.0, (100, 100)).astype(np.float32)
    fruit = ld.fruit_mask((100, 100), (40.0, 40.0, 360.0, 360.0))
    damage = ld.box_mask((100, 100), (160.0, 160.0, 240.0, 240.0))
    assert ld.damage_contrast(channel, fruit, damage) < 0.3
    channel[damage] -= 20.0  # a darker bruise, two standard deviations below the fruit
    assert abs(ld.damage_contrast(channel, fruit, damage) - 2.0) < 0.3


def test_a_box_too_small_gives_no_contrast() -> None:
    fruit = ld.fruit_mask((100, 100), (40.0, 40.0, 360.0, 360.0))
    damage = ld.box_mask((100, 100), (200.0, 200.0, 208.0, 208.0))
    assert ld.damage_contrast(np.ones((100, 100), np.float32), fruit, damage) is None
