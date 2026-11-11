"""Segment twins of the arm names the analysis scripts read.

`scripts/run_small_arms.py` defines, for every arm, a `<arm>_seg` twin (segment student) and, for the KD
arms, a `<arm>_segdt` twin (segment student, detect teacher). The scripts that look for runs by arm name
use these helpers so a segment run is found, drawn and exported like its detect arm, and the detect
arms stay first and unchanged.
"""

from __future__ import annotations

from collections.abc import Iterable

SEGMENT_SUFFIXES = ("_seg", "_segdt")
KD_PREFIXES = ("kd_", "crosskd_")  # the arms with a teacher, so the only ones with a `_segdt` twin


def segment_twin_names(arm: str) -> list[str]:
    """`[<arm>_seg]`, plus `<arm>_segdt` when the arm distills from a teacher."""
    twins = [f"{arm}_seg"]
    if arm.startswith(KD_PREFIXES):
        twins.append(f"{arm}_segdt")
    return twins


def with_segment_twins(arms: Iterable[str]) -> list[str]:
    """The arms as given, then the segment twins of each in the same order."""
    arms = list(arms)
    return [*arms, *(twin for arm in arms for twin in segment_twin_names(arm))]
