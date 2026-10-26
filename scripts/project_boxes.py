#!/usr/bin/env python3
"""Project hand-drawn +12 h damage boxes onto the earlier photos of the same mango face.

Usage:
    .venv/bin/python scripts/project_boxes.py \
        --data-root data/cache/captura_v3 \
        --export data/labels_lote_1_2.json \
        --out-dir data/label_studio_captura_v3 \
        --report-dir docs/findings/2026-10-26-box-projection \
        --homography data/cache/captura_v3/homografias/20261024/lote1_oficial/matriz_homografia_lote1_20261024_centros.npy

For each face with a +12 h box and each earlier moment, the +12 h photo is registered to the
earlier one in the modality the box was drawn on (SIFT and a dense masked-ZNCC search), the box
is carried across and gets a tier: A (two methods agree), B (one method), C (no projection).
With ``--homography`` the RGB registration is also carried into the NIR frame as two more
estimates for NIR boxes.

Writes ``predictions_<h>h.json`` (Label Studio tasks with a ``predictions`` entry on tiers A and
B) for every projected moment except +10 h, whose projections go only to
``projection_10h_for_validation.json`` (the blind validation set), plus ``projection_report.csv``,
``cycle_check.csv``, ``summary.json`` and ``contact_sheet.jpg`` in ``--report-dir``.

    --validate <hand_export.json>   compare the validation set with a later hand export
    --calibrate N                   registration scores of N unrelated pairs per modality
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.annotation.box_projection import (  # noqa: E402
    HandBoxes,
    format_validation,
    load_hand_boxes,
    moment_text,
    plan_faces,
    validate_projections,
    validation_to_dict,
    write_report,
)
from src.annotation.captura_v3_tasks import (
    NIR_DISPLAYS,
    describe_slot,
    infer_document_root,
)  # noqa: E402
from src.annotation.projection_run import (  # noqa: E402
    DENSE_MIN_SCORE,
    PhotoLoader,
    ProjectionRun,
    RunConfig,
    calibrate_nulls,
    consistency_summary,
    format_tier_table,
    render_contact_sheet,
    render_mask_sheet,
    run_projection,
    tier_table,
    validity_table,
    write_predictions,
)

VALIDATION_FILE = "projection_10h_for_validation.json"


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Project +12 h damage boxes onto earlier photos.")
    parser.add_argument("--data-root", type=Path, help="captura_v3 folder (news/, calib/).")
    parser.add_argument(
        "--export", type=Path, help="Label Studio JSON export with the +12 h boxes."
    )
    parser.add_argument("--out-dir", type=Path, help="Where the Label Studio prediction files go.")
    parser.add_argument(
        "--report-dir", type=Path, help="Where the CSV report, summary and sheet go."
    )
    parser.add_argument(
        "--document-root",
        type=Path,
        default=None,
        help="Label Studio LOCAL_FILES_DOCUMENT_ROOT (default: above the 'cache' folder).",
    )
    parser.add_argument(
        "--homography",
        type=Path,
        default=None,
        help="RGB->NIR homography (.npy); enables the RGB bridge for NIR boxes.",
    )
    parser.add_argument(
        "--homography-jornada",
        default=None,
        help="Jornada the homography was measured on (default: the first one).",
    )
    parser.add_argument(
        "--nir-display",
        choices=NIR_DISPLAYS,
        default="enhanced",
        help="NIR image the prediction tasks show (default: enhanced, as the +12 h round).",
    )
    parser.add_argument("--workers", type=int, default=min(10, os.cpu_count() or 1))
    parser.add_argument(
        "--limit-faces", type=int, default=None, help="Only the first N faces (testing)."
    )
    parser.add_argument("--no-cycles", action="store_true", help="Skip the +10 h cycle check.")
    parser.add_argument("--contact-sheet-faces", type=int, default=6)
    parser.add_argument(
        "--mask-sheet",
        type=Path,
        default=None,
        help="Also write a JPEG with the fruit masks of a sample of photos here.",
    )
    parser.add_argument(
        "--validate",
        type=Path,
        default=None,
        metavar="HAND_EXPORT",
        help="Compare the +10 h projections with a later hand export and exit.",
    )
    parser.add_argument(
        "--source-export",
        type=Path,
        default=None,
        help="Export with the +12 h source boxes for --validate (default: the hand export).",
    )
    parser.add_argument(
        "--projections",
        type=Path,
        default=None,
        help="Projection file for --validate (default: <out-dir>/" + VALIDATION_FILE + ").",
    )
    parser.add_argument(
        "--calibrate",
        type=int,
        default=None,
        metavar="N",
        help="Score N unrelated pairs per modality and exit.",
    )
    return parser.parse_args(argv)


def run_validation(args: argparse.Namespace) -> int:
    projections = args.projections or (args.out_dir / VALIDATION_FILE if args.out_dir else None)
    if projections is None:
        print("error: --validate needs --projections or --out-dir", file=sys.stderr)
        return 1
    report_csv = args.report_dir / "projection_report.csv" if args.report_dir else None
    if report_csv is not None and not report_csv.is_file():
        report_csv = None
    try:
        report = validate_projections(args.validate, projections, report_csv, args.source_export)
    except (ValueError, FileNotFoundError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(format_validation(report))
    if args.report_dir is not None:
        out = args.report_dir / f"validation_{moment_text(report.moment)}h.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(validation_to_dict(report), indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {out}")
    return 0


def run_calibration(args: argparse.Namespace) -> int:
    hand = load_hand_boxes(args.export, hours={12})
    planned = plan_faces(args.data_root, hand.boxes)
    loader = PhotoLoader(args.data_root)
    for modality in ("rgb", "nir"):
        nulls = calibrate_nulls(
            loader, planned.plans, modality=modality, pairs=args.calibrate, workers=args.workers
        )
        inliers, scores = np.array(nulls["sift_inliers"]), np.array(nulls["dense_score"])
        print(
            f"{modality}: unrelated pairs {len(scores)}  SIFT inliers max {inliers.max():.0f} "
            f"p99 {np.percentile(inliers, 99):.0f}  dense ZNCC max {scores.max():.3f} "
            f"p99 {np.percentile(scores, 99):.3f} median {np.median(scores):.3f}"
        )
    return 0


def summary_payload(
    run: ProjectionRun, args: argparse.Namespace, seconds: float, export_sha256: str
) -> dict:
    table = tier_table(run.records)
    return {
        "export_sha256": export_sha256,
        "hand_boxes": {
            "nir": sum(b.modality == "nir" for b in run.hand.boxes),
            "rgb": sum(b.modality == "rgb" for b in run.hand.boxes),
            "faces_without_box": [list(f) for f in run.hand.without_box],
            "tasks_without_annotation": [list(f) for f in run.hand.unannotated],
        },
        "faces_projected": len({(r.face, r.hand.modality) for r in run.records}),
        "dropped_duplicates": [
            describe_slot(d.slot) + f" (dropped {d.timestamp})" for d in run.planned.dropped
        ],
        "missing_slots": [describe_slot(s) for s in run.planned.missing],
        "dense_min_score": dict(DENSE_MIN_SCORE),
        "lot_shifts": {lot: list(shift) for lot, shift in run.shifts.items()},
        "tiers": {
            f"{lot}|{moment_text(moment)}": dict(counts) for (lot, moment), counts in table.items()
        },
        "validity": {
            f"{mod}|{moment_text(moment)}": row
            for (mod, moment), row in validity_table(run.records).items()
        },
        "consistency": consistency_summary(run.records, run.cycles),
        "seconds": round(seconds, 1),
        "workers": args.workers,
    }


def print_overview(run: ProjectionRun) -> None:
    hand: HandBoxes = run.hand
    print(
        f"Hand boxes: {sum(b.modality == 'nir' for b in hand.boxes)} NIR, "
        f"{sum(b.modality == 'rgb' for b in hand.boxes)} RGB"
    )
    for face in hand.without_box:
        print(f"  no box drawn, skipped: {face.jornada} mango {face.mango_id} face {face.cara}")
    print(
        f"Dropped duplicates: {len(run.planned.dropped)}; missing slots: {len(run.planned.missing)}"
    )
    for slot in run.planned.missing:
        print(f"  missing {describe_slot(slot)}")
    if run.shifts:
        print(
            "Lot shifts of the RGB->NIR map (px): "
            + ", ".join(f"{k} {v[0]:+.0f},{v[1]:+.0f}" for k, v in run.shifts.items())
        )
    print(format_tier_table(tier_table(run.records)))
    stats = consistency_summary(run.records, run.cycles)
    for method, row in stats["cycle"].items():
        print(
            f"cycle 12->t->10 vs 12->t [{method}]: n={row['n']} median {row['median_px']:.1f} px, "
            f"p90 {row['p90_px']:.1f} px, <10% side {row['share_under_10pct_side'] * 100:.0f}%"
        )
    for name, row in stats["sift_vs_dense"].items():
        print(
            f"SIFT vs dense [{name}]: n={row['n']} agree {row['share_agree'] * 100:.0f}% "
            f"(<10% side {row['share_under_10pct_side'] * 100:.0f}%)"
        )


def write_mask_sheet(run: ProjectionRun, loader: PhotoLoader, path: Path, count: int = 16) -> None:
    """A spread of photos (every moment, both lots, both modalities) with their fruit masks."""
    pool = [(r.face, r.hand.modality, r.moment) for r in run.records]
    step = max(1, len(pool) // count)
    items = []
    for face, modality, moment in pool[::step][:count]:
        capture = loader.capture(face, moment)
        if capture is not None:
            items.append((capture, modality))
    render_mask_sheet(loader, items, path)


def write_cycle_report(run: ProjectionRun, path: Path) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "jornada",
                "mango_id",
                "cara",
                "ronda_h",
                "method",
                "displacement_px",
                "displacement_frac_of_side",
            ]
        )
        for row in run.cycles:
            face = row["face"]
            writer.writerow(
                [
                    face.jornada,
                    face.mango_id,
                    face.cara,
                    moment_text(row["moment"]),
                    row["method"],
                    f"{row['displacement_px']:.2f}",
                    f"{row['displacement_frac']:.4f}",
                ]
            )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.validate is not None:
        return run_validation(args)
    missing = [n for n in ("data_root", "export") if getattr(args, n) is None]
    if missing:
        print("error: --data-root and --export are required", file=sys.stderr)
        return 1
    try:
        if args.calibrate is not None:
            return run_calibration(args)
        if args.out_dir is None or args.report_dir is None:
            print("error: --out-dir and --report-dir are required", file=sys.stderr)
            return 1
        start = time.perf_counter()
        export_sha256 = hashlib.sha256(
            args.export.read_bytes()
        ).hexdigest()  # the export may change meanwhile
        config = RunConfig(
            data_root=args.data_root,
            export=args.export,
            workers=args.workers,
            with_cycles=not args.no_cycles,
            limit_faces=args.limit_faces,
            homography=args.homography,
            homography_jornada=args.homography_jornada,
        )
        run = run_projection(config)
        document_root = args.document_root or infer_document_root(args.data_root)
        written = write_predictions(
            run.records, args.data_root, document_root, args.out_dir, nir_display=args.nir_display
        )
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    seconds = time.perf_counter() - start
    write_report(run.records, args.report_dir / "projection_report.csv")
    write_cycle_report(run, args.report_dir / "cycle_check.csv")
    render_contact_sheet(
        run.records,
        PhotoLoader(args.data_root),
        args.report_dir / "contact_sheet.jpg",
        faces=args.contact_sheet_faces,
    )
    if args.mask_sheet is not None:
        write_mask_sheet(run, PhotoLoader(args.data_root), args.mask_sheet)
    (args.report_dir / "summary.json").write_text(
        json.dumps(summary_payload(run, args, seconds, export_sha256), indent=2) + "\n",
        encoding="utf-8",
    )
    print_overview(run)
    for path in written:
        print(f"Wrote {path}")
    print(f"Elapsed: {seconds:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
