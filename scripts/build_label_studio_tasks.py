#!/usr/bin/env python3
"""Build the Label Studio task file for captura_v3 impact-damage annotation.

Usage:
    .venv/bin/python scripts/build_label_studio_tasks.py \
        --data-root data/cache/captura_v3 \
        --document-root data \
        --moments 12 \
        --out data/label_studio_captura_v3/tasks_12h.json

``--moments`` takes hours after the impact (``12``, ``12,10``) and/or ``sano``.
``--document-root`` must be the folder Label Studio serves local files from
(``LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT``) and must contain ``--data-root``. It
defaults to the folder above the single ``cache`` directory of ``--data-root``;
pass it explicitly when the path has no ``cache`` folder or more than one.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.annotation.captura_v3_tasks import (  # noqa: E402
    build_tasks,
    format_summary,
    infer_document_root,
    parse_moments,
    write_tasks,
)


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Label Studio tasks (RGB + NIR side by side) for captura_v3."
    )
    parser.add_argument("--data-root", type=Path, required=True,
                        help="captura_v3 folder containing news/{rgb,nir,meta,rgb_wb}.")
    parser.add_argument("--document-root", type=Path, default=None,
                        help="Label Studio LOCAL_FILES_DOCUMENT_ROOT (default: the folder "
                             "above the 'cache' directory of --data-root).")
    parser.add_argument("--moments", default="12",
                        help="Hours after the impact and/or 'sano', comma separated (default: 12).")
    parser.add_argument("--jornadas", default=None,
                        help="Comma-separated jornada ids or lot numbers (default: all).")
    parser.add_argument("--out", type=Path, required=True, help="Output tasks JSON path.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        moments = parse_moments(args.moments)
        jornadas = (
            tuple(token.strip() for token in args.jornadas.split(",") if token.strip())
            if args.jornadas
            else None
        )
        document_root = args.document_root or infer_document_root(args.data_root)
        result = build_tasks(args.data_root, document_root, moments, jornadas)
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    write_tasks(result.tasks, args.out)
    print(format_summary(result))
    print(f"Wrote {len(result.tasks)} tasks to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
