#!/usr/bin/env python3
"""Build contrast-enhanced grayscale NIR images for captura_v3 annotation.

Usage:
    .venv/bin/python scripts/build_nir_display.py \
        --data-root data/cache/captura_v3 \
        --moments 12

Writes ``<data-root>/news/nir_vis/mango_nir_<timestamp>.jpg`` (single-channel JPEG with the
exact size of the original NIR) plus ``manifest.json`` (method, parameters, source sha256).
Captures are selected like in ``build_label_studio_tasks.py`` (same ``--moments``,
``--jornadas`` and duplicate rule). Existing images are skipped unless ``--overwrite``;
``news/rgb``, ``news/nir`` and ``news/meta`` are never written.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.annotation.captura_v3_tasks import (  # noqa: E402
    parse_jornadas,
    parse_moments,
    select_captures,
)
from src.annotation.nir_display import build_nir_display  # noqa: E402


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build contrast-enhanced grayscale NIR images (news/nir_vis) for captura_v3."
    )
    parser.add_argument("--data-root", type=Path, required=True,
                        help="captura_v3 folder containing news/{nir,meta}.")
    parser.add_argument("--moments", default="12",
                        help="Hours after the impact and/or 'sano', comma separated (default: 12).")
    parser.add_argument("--jornadas", default=None,
                        help="Comma-separated jornada ids or lot numbers (default: all).")
    parser.add_argument("--overwrite", action="store_true",
                        help="Regenerate images that already exist.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    start = time.perf_counter()
    try:
        moments = parse_moments(args.moments)
        selection = select_captures(args.data_root, moments, parse_jornadas(args.jornadas))
        report = build_nir_display(
            args.data_root, [c.timestamp for c in selection.kept], overwrite=args.overwrite
        )
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"Selected: {len(selection.kept)}")
    print(f"Written: {len(report.written)}")
    print(f"Skipped: {len(report.skipped)}")
    print(f"Output: {report.out_dir}")
    print(f"Elapsed: {time.perf_counter() - start:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
