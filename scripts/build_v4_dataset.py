#!/usr/bin/env python3
"""Build the v4 training dataset from captura_v3 (lots 1-5) with the NIR registered per lot.

See `src/data_pipeline/v4_dataset.py` for the layout and the inclusion rules. Train on it with

    --override rgb_dir=<out>/rgb nir_dir=<out>/nir \\
               nir_homography_path=<out>/homographies/identity.npy

plus the fold labels and manifest (`labels_dir`, `split_manifest`) built from `<out>/labels`.
Use `nir_dir=<out>/nir_clahe` for the CLAHE NIR, with its own `nir_mean`/`nir_std`.

With `--polygons <export.json>` the labels are YOLO-seg polygons instead of boxes (see
`src/data_pipeline/v4_polygons.py`); train on them with `task=segment`. Without it nothing changes.

By default each photo gets a translation on top of its homography, measured on the fruit, and
`report.json` holds the residual per lot and moment; `--residual-shift none` leaves it out.

Typical use (a linked worktree reads the untracked data of the primary checkout):

    ./.venv/bin/python scripts/build_v4_dataset.py --workers 8
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.export_aligned_corpus import primary_checkout_root  # noqa: E402
from src.data_pipeline.v4_dataset import RESIDUAL_MODES, build_dataset  # noqa: E402

DEFAULT_DATA_ROOT = "data/cache/captura_v3"
DEFAULT_OUTPUT = "data/cache/v4_captura_v3"
DEFAULT_LABELS = PROJECT_ROOT / "data/labels/captura_v3_lots1-5_12h_0h.json"
DEFAULT_RULES = PROJECT_ROOT / "configs/data/v4_homographies.yaml"
DEFAULT_IDENTITY = PROJECT_ROOT / "configs/data/v4_identity_exclusions.yaml"
DEFAULT_TOLERANCE = PROJECT_ROOT / "configs/data/v4_tolerance_exceptions.yaml"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help=f"captura_v3 cache (default: <primary checkout>/{DEFAULT_DATA_ROOT})",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=DEFAULT_LABELS,
        help="Label Studio JSON export with the damage boxes",
    )
    parser.add_argument(
        "--polygons",
        type=Path,
        default=None,
        help="Label Studio polygon export (both views in the RGB frame): write YOLO-seg labels, "
        "damage from the export and the mango from the fruit-mask contour. Replaces --labels. "
        "Default: box labels",
    )
    parser.add_argument(
        "--homographies",
        type=Path,
        default=DEFAULT_RULES,
        help="per-lot RGB -> NIR homography rules (YAML)",
    )
    parser.add_argument(
        "--identity-exclusions",
        type=Path,
        default=DEFAULT_IDENTITY,
        help="photos whose SIFT identity was not confirmed (YAML)",
    )
    parser.add_argument(
        "--tolerance-exceptions",
        type=Path,
        default=DEFAULT_TOLERANCE,
        help="rounds whose late photos are kept against inclusion rule 1 (YAML)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=f"new or empty folder (default: <primary checkout>/{DEFAULT_OUTPUT})",
    )
    parser.add_argument("--workers", type=int, default=1, help="parallel processes")
    parser.add_argument(
        "--residual-shift",
        choices=RESIDUAL_MODES,
        default="fruit",
        help="fruit: add a per-photo translation measured on the fruit to the NIR and to the "
        "boxes drawn on it; none: the homography alone (default: fruit)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    primary = primary_checkout_root(PROJECT_ROOT)
    try:
        report = build_dataset(
            data_root=args.data_root or primary / DEFAULT_DATA_ROOT,
            labels_path=args.labels,
            rules_path=args.homographies,
            identity_path=args.identity_exclusions,
            tolerance_path=args.tolerance_exceptions,
            out_dir=args.output or primary / DEFAULT_OUTPUT,
            repo_root=PROJECT_ROOT,
            workers=args.workers,
            residual_shift=args.residual_shift,
            polygons_path=args.polygons,
        )
    except (FileNotFoundError, FileExistsError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"included {report['num_included']}, excluded {report['num_excluded']}")
    for lot, counts in report["included"].items():
        print(f"  {lot} included: {counts}")
    for lot, counts in report["excluded"].items():
        print(f"  {lot} excluded: {counts}")
    for lot, moments in report["residual"].items():
        for moment, values in moments.items():
            name = moment if moment == "sano" else f"+{moment} h"
            print(
                f"  {lot} {name} residual: {values['median_dx']:+.0f}, "
                f"{values['median_dy']:+.0f} px, "
                f"accepted {values['accepted']}, fallback {values['fallback']}"
                + (f", NOT MEASURED {values['unmeasured']}" if values["unmeasured"] else "")
            )
    if report["damage_mostly_outside_mango"]:
        print(
            f"  damage boxes mostly outside the mango box: {report['damage_mostly_outside_mango']}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
