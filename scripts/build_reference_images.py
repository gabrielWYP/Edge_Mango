#!/usr/bin/env python3
"""Rebuild the read-only +12 h reference images of the captura_v3 damage project.

Usage:
    .venv/bin/python scripts/build_reference_images.py \
        --labels data/labels/captura_v3_lots1-3_12h_0h.json \
        --document-root "$PWD/data"

Reads the Label Studio JSON export and writes, for every ``data.ref`` in it,
``<document-root>/cache/captura_v3/news/ref_12h/mango_{nir|rgb}_<timestamp>.jpg``: the image the same
face's +12 h task displays with its hand-drawn damage box outlined (same size, JPEG), plus the gray
``no_reference.jpg`` placeholder. Existing images are skipped unless ``--overwrite``. It fails if a
source image is missing or the export is inconsistent, and never writes outside ``ref_12h``.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.annotation.reference_images import (  # noqa: E402
    build_reference_images,
    load_reference_jobs,
)


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild the +12 h reference images (news/ref_12h) from a Label Studio export."
    )
    parser.add_argument("--labels", type=Path, required=True,
                        help="Label Studio JSON export (a list of tasks with data and annotations).")
    parser.add_argument("--document-root", type=Path, required=True,
                        help="Folder Label Studio serves files from (the '?d=' paths are relative to it).")
    parser.add_argument("--overwrite", action="store_true",
                        help="Regenerate images that already exist.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    start = time.perf_counter()
    try:
        jobs = load_reference_jobs(args.labels)
        report = build_reference_images(args.document_root, jobs, overwrite=args.overwrite)
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"References: {len(jobs)}")
    print(f"Written: {len(report.written)}")
    print(f"Skipped: {len(report.skipped)}")
    print(f"Output: {report.out_dir}")
    print(f"Elapsed: {time.perf_counter() - start:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
