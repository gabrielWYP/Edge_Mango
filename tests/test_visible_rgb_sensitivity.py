"""Sensitivity without the latent photos visible in RGB: which photos stay in each set."""

from __future__ import annotations

import torch

from scripts import visible_rgb_sensitivity as vrs
from src.evaluation.v4_metrics import EvaluatedPhoto


def _photo(stem, hour) -> EvaluatedPhoto:
    empty = torch.zeros((0, 4), dtype=torch.float64)
    return EvaluatedPhoto(
        stem=stem, hour=hour, zona="", provenance="", face=("1", "1", "a"),
        gt_boxes=empty, pred_boxes=empty, pred_scores=torch.zeros(0, dtype=torch.float64),
    )


def test_only_latent_photos_visible_in_rgb_are_removed() -> None:
    photos = [_photo("a", 0), _photo("b", 2), _photo("c", 6), _photo("d", 8), _photo("e", None)]
    index = {"a": {"visible_rgb": "yes"}, "b": {"visible_rgb": "unsure"}, "c": {"visible_rgb": "no"},
             "d": {"visible_rgb": "yes"}, "e": {"visible_rgb": ""}}
    latent, kept = vrs.latent_sets(photos, index)
    assert [p.stem for p in latent] == ["a", "b", "c", "e"]
    assert [p.stem for p in kept] == ["b", "c", "e"]
