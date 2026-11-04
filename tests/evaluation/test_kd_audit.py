"""`src/evaluation/kd_audit.py`: region sums and the damage-vs-fruit probe."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.evaluation.kd_audit import collect_probe_cells, probe_auroc, region_sums


def _masks():
    damage = torch.zeros(1, 4, 4, dtype=torch.bool)
    damage[0, 0, 0] = True
    fruit = torch.zeros(1, 4, 4, dtype=torch.bool)
    fruit[0, 1, :] = True
    return {"damage": damage, "fruit": fruit, "background": ~(damage | fruit)}


def test_region_sums_add_up_per_region():
    per_cell = torch.ones(1, 4, 4)
    per_cell[0, 0, 0] = 10.0

    sums = region_sums(per_cell, _masks())

    assert sums["damage"] == (10.0, 1)
    assert sums["fruit"] == (4.0, 4)
    assert sums["background"] == (11.0, 11)


def test_collect_keeps_all_damage_and_caps_the_fruit_cells():
    feat = torch.randn(1, 3, 4, 4)

    x, y = collect_probe_cells(feat, _masks(), 2, torch.Generator().manual_seed(0))

    assert x.shape == (3, 3)
    assert y.tolist() == [1.0, 0.0, 0.0]


def test_probe_separates_a_real_signal_and_not_noise():
    rng = np.random.default_rng(0)
    y = np.array([0, 1] * 300, dtype=float)
    noise = rng.normal(size=(600, 4))
    signal = noise + y[:, None] * 3.0

    assert probe_auroc(signal[:300], y[:300], signal[300:], y[300:]) > 0.95
    assert probe_auroc(noise[:300], y[:300], noise[300:], y[300:]) == pytest.approx(0.5, abs=0.1)
