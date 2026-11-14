"""Names of the segment arms the analysis scripts read.

`scripts/run_small_arms.py` defines a segment twin of every arm, trained from scratch on polygon labels. An
undistilled arm is a "master" there (`est_rgb` -> `master_rgb_seg`) and a distilled one keeps its name
(`crosskd_rgb` -> `crosskd_rgb_seg`, distilling from `master_rgb_seg` of the same fold and seed). A segment student with a
detect teacher is not an arm: the code path exists in the trainer, the name does not. The scripts that look for
runs by arm name use these helpers so a segment run is found, drawn and exported like its detect arm, and the
detect arms stay first and unchanged.
"""

from __future__ import annotations

from collections.abc import Iterable

SEGMENT_SUFFIX = "_seg"
# Presets of `run_small_arms.py --arms-preset`, masters before the CrossKD arms that distill from them. The MSE
# feature-KD arms (`kd_rgb_seg`, `kd_rgbnir_seg`) stay runnable by name and are in no preset.
SEGMENT_PRESETS = {
    "seg": ("master_rgb_seg", "master_rgbnir_seg", "master_nir_seg", "crosskd_rgb_seg", "crosskd_rgbnir_seg"),
    "seg-full": (
        "master_rgb_seg", "master_rgbnir_seg", "master_nir_seg", "crosskd_rgb_seg", "crosskd_rgbnir_seg",
        "crosskd_rgbnir_high_seg", "crosskd_rgbnir_low_seg", "crosskd_rgbnir_better_seg",
    ),
    # The final thesis plan: `seg` plus the detect arm trained on the polygon-derived boxes (`DETECT_MASTER_ARM`).
    "thesis": (
        "master_rgb_seg", "master_rgbnir_seg", "master_nir_seg", "master_rgb_det", "crosskd_rgb_seg", "crosskd_rgbnir_seg",
    ),
}
# Detect RGB student with exactly the training config of `est_rgb`, trained on the polygon v4 dataset: its labels are the
# tight boxes of the polygons. Its output folder is the only difference from `est_rgb`. Mask supervision is the
# comparison `master_rgb_seg - master_rgb_det`. It is not a segment twin of any arm.
DETECT_MASTER_ARM = "master_rgb_det"


def segment_arm_name(arm: str) -> str:
    """`est_rgb` -> `master_rgb_seg`; any other arm gets the suffix (`kd_rgbnir` -> `kd_rgbnir_seg`)."""
    if arm.startswith("est_"):
        return f"master_{arm.removeprefix('est_')}{SEGMENT_SUFFIX}"
    return f"{arm}{SEGMENT_SUFFIX}"


def segment_twin_names(arm: str) -> list[str]:
    """The segment twin of a detect arm, as a one-item list."""
    return [segment_arm_name(arm)]


def with_segment_twins(arms: Iterable[str]) -> list[str]:
    """The arms as given, then the segment twin of each in the same order."""
    arms = list(arms)
    return [*arms, *(segment_arm_name(arm) for arm in arms)]
