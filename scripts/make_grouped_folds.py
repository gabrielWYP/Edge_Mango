#!/usr/bin/env python3
"""Build leave-one-lot-out CV folds with zero mango leakage.

The canonical `splits.json` assigns images to train/val/test at random, one
image at a time. Each physical mango was photographed on several sides and
reappears minutes, hours and a day later, so a per-image split puts the same
mango in train and test (a heavily bruised mango sits at 05-31 09:46:25 in test
and 09:50:21 in train). Clustering by timestamp gap does not recover clean
per-mango groups; the dependable unit is the CAPTURE LOT: every lot used new
mangos, and a lot is a run of consecutive capture days.

Lots are found from the capture timestamp in the stem (`mango_rgb_<unix_ts>`):

  1. the local calendar day (`--tz`, default America/Lima) of every stem;
  2. consecutive days whose gap is <= `--lot-gap-days` (default 1) form a lot;
  3. a lot smaller than `--min-lot-size` (default 5) merges into the
     chronologically nearest lot (smallest time gap between the two; the
     earlier lot on a tie).

Lot ids are chronological (`lot1..lotN`). K = N folds, leave-one-lot-out:

    fold k:  test = lot k,  val = lot (k + 1) mod K,  train = every other lot

Outputs (deterministic: the same inputs give byte-identical files):

    <output-dir>/fold_{k}/splits.json          same schema as the canonical manifest
    <output-dir>/fold_{k}/labels/{train,val,test}/<stem>.txt   COPIES of the labels
    <output-dir>/folds_meta.json               per-lot / per-fold counts + leakage report
    <assignment-file>                          small git-tracked copy of the assignment

`YOLODataset` reads `<labels_dir>/<split>/<stem>.txt` and verifies it against
the manifest, so a fold trains with

    --override labels_dir=<output-dir>/fold_{k}/labels \\
               split_manifest=<output-dir>/fold_{k}/splits.json

and no dataset code changes. Labels are copied, never symlinked, so the folds
work on machines (and filesystems) without symlink support.

Usage:
    python scripts/make_grouped_folds.py
    python scripts/make_grouped_folds.py --check   # compare with the tracked file, write nothing
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import logging
import re
import shutil
import sys
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

PROJECT_ROOT = Path(__file__).resolve().parent.parent

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
SPLITS = ("train", "val", "test")
DAMAGE_CLASS_ID = 1
DEFAULT_TZ = "America/Lima"
SECONDS_PER_DAY = 86400

DEFAULT_MANIFEST = "data/annotations/yolo/splits.json"
DEFAULT_LABELS_DIR = "data/annotations/yolo/labels"
DEFAULT_OUTPUT_DIR = "data/annotations/yolo/folds_lot"
DEFAULT_ASSIGNMENT_FILE = "configs/folds/lot_5fold.json"

_TIMESTAMP_RE = re.compile(r"_(\d{9,})$")
_FOLD_DIR_RE = re.compile(r"^fold_(\d+)$")


class LeakageError(ValueError):
    """The generated folds would let a mango (or a re-photograph) cross splits."""


@dataclass
class Lot:
    """A run of consecutive capture days; every lot used new mangos."""

    lot_id: str
    days: list[str]
    stems: list[str] = field(default_factory=list)

    @property
    def size(self) -> int:
        return len(self.stems)

    @property
    def start(self) -> int:
        return min(stem_timestamp(s) for s in self.stems)

    @property
    def end(self) -> int:
        return max(stem_timestamp(s) for s in self.stems)


@dataclass
class Fold:
    """One leave-one-lot-out fold, expressed as lot ids."""

    index: int
    test: str
    val: str
    train: list[str]


# --------------------------------------------------------------------------
# Lots
# --------------------------------------------------------------------------


def stem_timestamp(stem: str) -> int:
    """Parse the trailing unix timestamp of a `mango_rgb_<unix_ts>` stem."""
    match = _TIMESTAMP_RE.search(stem)
    if match is None:
        raise ValueError(f"Cannot parse a capture timestamp from stem '{stem}'.")
    return int(match.group(1))


def _zone(tz_name: str) -> ZoneInfo:
    try:
        return ZoneInfo(tz_name)
    except ZoneInfoNotFoundError:
        raise ValueError(
            f"Timezone '{tz_name}' not found. Check the name; on machines without a system "
            "timezone database (e.g. Windows) run `pip install tzdata`."
        ) from None


def local_date(timestamp: int, tz_name: str = DEFAULT_TZ) -> date:
    """Local calendar day of a unix timestamp in `tz_name`."""
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).astimezone(_zone(tz_name)).date()


def merge_small_lots(lots: list[Lot], min_lot_size: int) -> list[Lot]:
    """Merge every lot smaller than `min_lot_size` into its nearest neighbour.

    "Nearest" is the previous or next lot with the smallest time gap between
    the two lots' closest captures; the earlier lot wins a tie. The smallest
    offending lot (earliest on ties) goes first, repeating until none is left
    or a single lot remains. Chronological order is preserved.
    """
    lots = list(lots)
    while len(lots) > 1:
        small = [i for i, lot in enumerate(lots) if lot.size < min_lot_size]
        if not small:
            break
        i = min(small, key=lambda idx: (lots[idx].size, lots[idx].start))
        candidates = []
        if i > 0:
            candidates.append((lots[i].start - lots[i - 1].end, i - 1))
        if i < len(lots) - 1:
            candidates.append((lots[i + 1].start - lots[i].end, i + 1))
        _, j = min(candidates)
        lo, hi = sorted((i, j))
        logger.info(
            "Merging lot of %d image(s) (%s) into its nearest lot (%s, %d image(s))",
            lots[i].size, lots[i].days, lots[j].days, lots[j].size,
        )
        lots[lo:hi + 1] = [
            Lot("", sorted({*lots[lo].days, *lots[hi].days}), sorted(lots[lo].stems + lots[hi].stems))
        ]
    return lots


def group_into_lots(
    stems: list[str],
    tz_name: str = DEFAULT_TZ,
    lot_gap_days: int = 1,
    min_lot_size: int = 5,
) -> list[Lot]:
    """Group stems into chronological capture lots (`lot1..lotN`)."""
    if lot_gap_days < 0:
        raise ValueError(f"lot_gap_days must be >= 0, got {lot_gap_days}.")
    if min_lot_size < 1:
        raise ValueError(f"min_lot_size must be >= 1, got {min_lot_size}.")

    by_day: dict[date, list[str]] = {}
    for stem in sorted(set(stems)):
        by_day.setdefault(local_date(stem_timestamp(stem), tz_name), []).append(stem)

    runs: list[list[date]] = []
    for day in sorted(by_day):
        if runs and (day - runs[-1][-1]).days <= lot_gap_days:
            runs[-1].append(day)
        else:
            runs.append([day])

    lots = [
        Lot("", [d.isoformat() for d in run], sorted(s for d in run for s in by_day[d]))
        for run in runs
    ]
    lots = merge_small_lots(lots, min_lot_size)
    for number, lot in enumerate(lots, start=1):
        lot.lot_id = f"lot{number}"
    return lots


# --------------------------------------------------------------------------
# Folds
# --------------------------------------------------------------------------


def assign_folds(lots: list[Lot]) -> list[Fold]:
    """Leave-one-lot-out: fold k tests lot k, validates on lot (k+1) mod K."""
    k = len(lots)
    if k < 3:
        raise ValueError(
            f"Leave-one-lot-out needs at least 3 lots (train, val and test each hold whole "
            f"lots), found {k}: {[(lot.lot_id, lot.days) for lot in lots]}. "
            "Lower --min-lot-size or --lot-gap-days."
        )
    ids = [lot.lot_id for lot in lots]
    return [
        Fold(
            index=i,
            test=ids[i],
            val=ids[(i + 1) % k],
            train=[lot_id for lot_id in ids if lot_id not in (ids[i], ids[(i + 1) % k])],
        )
        for i in range(k)
    ]


def fold_manifests(lots: list[Lot], folds: list[Fold]) -> list[dict[str, list[str]]]:
    """One splits.json-schema manifest per fold (sorted stems)."""
    by_id = {lot.lot_id: lot for lot in lots}
    return [
        {
            "test": sorted(by_id[fold.test].stems),
            "train": sorted(s for lot_id in fold.train for s in by_id[lot_id].stems),
            "val": sorted(by_id[fold.val].stems),
        }
        for fold in folds
    ]


def _closest_pair(eval_stems: list[str], train_stems: list[str]) -> tuple[int, str, str]:
    """(gap_seconds, eval_stem, train_stem) of the closest eval/train capture pair."""
    train_sorted = sorted((stem_timestamp(s), s) for s in train_stems)
    train_times = [t for t, _ in train_sorted]
    best: tuple[int, str, str] | None = None
    for stem in sorted(eval_stems):
        ts = stem_timestamp(stem)
        pos = bisect.bisect_left(train_times, ts)
        for idx in (pos - 1, pos):
            if 0 <= idx < len(train_sorted):
                candidate = (abs(ts - train_sorted[idx][0]), stem, train_sorted[idx][1])
                if best is None or candidate < best:
                    best = candidate
    if best is None:
        raise LeakageError("Cannot measure the train/eval gap: a split is empty.")
    return best


def leakage_report(lots: list[Lot], folds: list[Fold]) -> dict:
    """Verify and describe the absence of leakage; raise `LeakageError` otherwise.

    Checks, on the manifests the folds actually produce: every stem belongs to
    one lot only; train/val/test are pairwise disjoint in every fold; every
    stem is a test stem in exactly one fold; and the closest val/test capture
    to any train capture is more than one day away (true by construction,
    because lots are separated by at least two calendar days).
    """
    owner: dict[str, str] = {}
    for lot in lots:
        for stem in lot.stems:
            if stem in owner:
                raise LeakageError(
                    f"Stem {stem} is in more than one lot ({owner[stem]} and {lot.lot_id})."
                )
            owner[stem] = lot.lot_id

    manifests = fold_manifests(lots, folds)
    tested: dict[str, int] = {}
    fold_reports: dict[str, dict] = {}
    for fold, manifest in zip(folds, manifests):
        sets = {split: set(manifest[split]) for split in SPLITS}
        for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
            shared = sets[a] & sets[b]
            if shared:
                raise LeakageError(
                    f"fold {fold.index}: {a} and {b} are not disjoint ({len(shared)} shared, "
                    f"e.g. {sorted(shared)[:3]})."
                )
        for stem in manifest["test"]:
            tested[stem] = tested.get(stem, 0) + 1

        val_gap = _closest_pair(manifest["val"], manifest["train"])
        test_gap = _closest_pair(manifest["test"], manifest["train"])
        gap, eval_stem, train_stem = min(val_gap, test_gap)
        if gap <= SECONDS_PER_DAY:
            raise LeakageError(
                f"fold {fold.index}: a val/test capture is only {gap} s from a train capture "
                f"({eval_stem} vs {train_stem}); the gap must exceed one day."
            )
        fold_reports[str(fold.index)] = {
            "closest_pair": [eval_stem, train_stem],
            "min_gap_days": round(gap / SECONDS_PER_DAY, 6),
            "min_gap_seconds": gap,
            "test_min_gap_seconds": test_gap[0],
            "val_min_gap_seconds": val_gap[0],
        }

    if sorted(tested) != sorted(owner) or any(count != 1 for count in tested.values()):
        raise LeakageError(
            "Not every stem is a test stem in exactly one fold "
            f"({len(tested)} tested of {len(owner)} stems)."
        )
    overall = min(r["min_gap_seconds"] for r in fold_reports.values())
    return {
        "all_disjoint": True,
        "every_stem_tested_exactly_once": True,
        "folds": fold_reports,
        "min_gap_days": round(overall / SECONDS_PER_DAY, 6),
        "min_gap_seconds": overall,
    }


# --------------------------------------------------------------------------
# Inputs: manifest + labels
# --------------------------------------------------------------------------


def dump_json(obj) -> str:
    return json.dumps(obj, indent=2, sort_keys=True) + "\n"


def canonical_split_of(manifest: dict) -> dict[str, str]:
    """stem -> canonical split, rejecting a stem listed in two splits."""
    split_of: dict[str, str] = {}
    for split in SPLITS:
        for stem in manifest.get(split, []):
            if stem in split_of:
                raise ValueError(
                    f"Stem {stem} is listed in more than one split of the manifest "
                    f"({split_of[stem]} and {split})."
                )
            split_of[stem] = split
    return split_of


def resolve_label_sources(split_of: dict[str, str], labels_dir: Path) -> dict[str, Path]:
    """stem -> `<labels_dir>/<canonical split>/<stem>.txt`; every file must exist."""
    sources = {stem: labels_dir / split / f"{stem}.txt" for stem, split in sorted(split_of.items())}
    missing = [path for path in sources.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} label file(s) listed by the manifest are missing under {labels_dir}: "
            + ", ".join(str(p.relative_to(labels_dir)) for p in missing[:5])
            + (" ..." if len(missing) > 5 else "")
        )
    return sources


def count_damage_boxes(label_path: Path) -> int:
    """Number of class-1 boxes in a label file (same parsing as `YOLODataset`)."""
    count = 0
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        if float(parts[3]) <= 0 or float(parts[4]) <= 0:
            continue
        if int(parts[0]) == DAMAGE_CLASS_ID:
            count += 1
    return count


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --------------------------------------------------------------------------
# Plan, assignment, meta
# --------------------------------------------------------------------------


@dataclass
class Plan:
    """Everything derived from the manifest alone (no labels needed)."""

    manifest_sha256: str
    split_of: dict[str, str]
    lots: list[Lot]
    folds: list[Fold]
    manifests: list[dict[str, list[str]]]
    params: dict
    assignment: dict
    leakage: dict


def build_plan(
    manifest_path: Path,
    tz_name: str = DEFAULT_TZ,
    lot_gap_days: int = 1,
    min_lot_size: int = 5,
) -> Plan:
    manifest_bytes = manifest_path.read_bytes()
    split_of = canonical_split_of(json.loads(manifest_bytes))
    if not split_of:
        raise ValueError(f"Manifest {manifest_path} lists no stems.")
    lots = group_into_lots(list(split_of), tz_name, lot_gap_days, min_lot_size)
    folds = assign_folds(lots)
    leakage = leakage_report(lots, folds)
    manifests = fold_manifests(lots, folds)
    params = {"lot_gap_days": lot_gap_days, "min_lot_size": min_lot_size, "tz": tz_name}
    assignment = {
        "folds": [
            {
                "fold": fold.index,
                "num_test": len(m["test"]),
                "num_train": len(m["train"]),
                "num_val": len(m["val"]),
                "test": fold.test,
                "train": fold.train,
                "val": fold.val,
            }
            for fold, m in zip(folds, manifests)
        ],
        "lots": {
            lot.lot_id: {"days": lot.days, "num_images": lot.size, "stems": lot.stems} for lot in lots
        },
        "num_images": len(split_of),
        "params": params,
        "schema_version": SCHEMA_VERSION,
        "source_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
    }
    return Plan(
        manifest_sha256=assignment["source_manifest_sha256"],
        split_of=split_of, lots=lots, folds=folds, manifests=manifests,
        params=params, assignment=assignment, leakage=leakage,
    )


def _split_counts(stems: list[str], damage: dict[str, int]) -> dict[str, int]:
    with_damage = sum(1 for s in stems if damage[s] > 0)
    return {
        "damage_boxes": sum(damage[s] for s in stems),
        "images": len(stems),
        "negatives": len(stems) - with_damage,
        "with_damage": with_damage,
    }


def build_meta(plan: Plan, label_sources: dict[str, Path]) -> dict:
    """Per-lot and per-fold counts (images, damage boxes, negatives) + leakage report."""
    damage = {stem: count_damage_boxes(path) for stem, path in label_sources.items()}
    tz_name = plan.params["tz"]
    tz = _zone(tz_name)

    def local_iso(ts: int) -> str:
        return datetime.fromtimestamp(ts, tz=timezone.utc).astimezone(tz).isoformat()

    def lot_counts(lot: Lot) -> dict[str, int]:
        counts = _split_counts(lot.stems, damage)
        counts["num_images"] = counts.pop("images")
        return counts

    labels_digest = hashlib.sha256()
    for stem, path in sorted(label_sources.items()):
        labels_digest.update(f"{stem}\t{_sha256_file(path)}\n".encode())

    return {
        "damage_class_id": DAMAGE_CLASS_ID,
        "folds": {
            str(fold.index): {
                "splits": {split: _split_counts(manifest[split], damage) for split in SPLITS},
                "test": fold.test,
                "train": fold.train,
                "val": fold.val,
            }
            for fold, manifest in zip(plan.folds, plan.manifests)
        },
        "labels_sha256": labels_digest.hexdigest(),
        "leakage": plan.leakage,
        "lots": {
            lot.lot_id: {
                "days": lot.days,
                "first_capture": local_iso(lot.start),
                "last_capture": local_iso(lot.end),
                **lot_counts(lot),
            }
            for lot in plan.lots
        },
        "num_images": len(plan.split_of),
        "params": plan.params,
        "schema_version": SCHEMA_VERSION,
        "source_manifest_sha256": plan.manifest_sha256,
    }


# --------------------------------------------------------------------------
# Materialization
# --------------------------------------------------------------------------


def _overlaps(a: Path, b: Path) -> bool:
    a, b = a.resolve(), b.resolve()
    return a == b or a in b.parents or b in a.parents


def _fold_dirs(output_dir: Path) -> list[Path]:
    if not output_dir.is_dir():
        return []
    return sorted(
        (p for p in output_dir.iterdir() if p.is_dir() and _FOLD_DIR_RE.match(p.name)),
        key=lambda p: int(_FOLD_DIR_RE.match(p.name).group(1)),
    )


def write_folds(
    output_dir: Path,
    manifests: list[dict[str, list[str]]],
    label_sources: dict[str, Path],
    meta: dict,
) -> None:
    """(Re)write `fold_k/{splits.json,labels/}` and `folds_meta.json`.

    Existing `fold_<n>` directories under `output_dir` are replaced: a stale
    label file would make `YOLODataset` reject the split, and a leftover
    fold from a previous grouping would be picked up by the runner.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale in _fold_dirs(output_dir):
        shutil.rmtree(stale)
    for index, manifest in enumerate(manifests):
        fold_dir = output_dir / f"fold_{index}"
        (fold_dir / "splits.json").parent.mkdir(parents=True, exist_ok=True)
        (fold_dir / "splits.json").write_text(dump_json(manifest))
        for split in SPLITS:
            split_dir = fold_dir / "labels" / split
            split_dir.mkdir(parents=True, exist_ok=True)
            for stem in manifest[split]:
                shutil.copyfile(label_sources[stem], split_dir / f"{stem}.txt")
    (output_dir / "folds_meta.json").write_text(dump_json(meta))


def make_grouped_folds(
    manifest_path: Path,
    labels_dir: Path,
    output_dir: Path,
    assignment_file: Path | None = None,
    tz_name: str = DEFAULT_TZ,
    lot_gap_days: int = 1,
    min_lot_size: int = 5,
) -> dict:
    """Generate and write the folds (and the tracked assignment). Returns the meta dict."""
    manifest_path, labels_dir, output_dir = Path(manifest_path), Path(labels_dir), Path(output_dir)
    if _overlaps(output_dir, labels_dir):
        raise ValueError(
            f"--output-dir {output_dir} overlaps the canonical labels dir {labels_dir}; "
            "refusing to write (and clean) folds there."
        )
    plan = build_plan(manifest_path, tz_name, lot_gap_days, min_lot_size)
    label_sources = resolve_label_sources(plan.split_of, labels_dir)
    meta = build_meta(plan, label_sources)

    for lot in plan.lots:
        counts = meta["lots"][lot.lot_id]
        logger.info(
            "%s  %s  %3d images, %3d damage boxes, %3d negatives",
            lot.lot_id, "+".join(lot.days), counts["num_images"], counts["damage_boxes"], counts["negatives"],
        )
    write_folds(output_dir, plan.manifests, label_sources, meta)
    for fold in plan.folds:
        splits = meta["folds"][str(fold.index)]["splits"]
        logger.info(
            "fold_%d: train=%d val=%d test=%d (test=%s val=%s)", fold.index,
            splits["train"]["images"], splits["val"]["images"], splits["test"]["images"],
            fold.test, fold.val,
        )
    if assignment_file is not None:
        assignment_file = Path(assignment_file)
        assignment_file.parent.mkdir(parents=True, exist_ok=True)
        assignment_file.write_text(dump_json(plan.assignment))
        logger.info("Wrote tracked assignment %s", assignment_file)
    logger.info("Wrote %d folds + folds_meta.json to %s", len(plan.folds), output_dir)
    return meta


# --------------------------------------------------------------------------
# --check
# --------------------------------------------------------------------------


def _diff(expected, actual, path: str = "") -> list[str]:
    """Human-readable differences between two JSON values (bounded)."""
    if isinstance(expected, dict) and isinstance(actual, dict):
        out = []
        for key in sorted(set(expected) | set(actual)):
            where = f"{path}.{key}" if path else str(key)
            if key not in actual:
                out.append(f"{where}: only in regenerated")
            elif key not in expected:
                out.append(f"{where}: only in tracked file")
            else:
                out += _diff(expected[key], actual[key], where)
        return out
    if expected == actual:
        return []
    if isinstance(expected, list) and isinstance(actual, list) and all(
        isinstance(x, str) for x in expected + actual
    ):
        only_new, only_old = sorted(set(expected) - set(actual)), sorted(set(actual) - set(expected))
        return [
            f"{path}: regenerated has {len(only_new)} item(s) the tracked file lacks "
            f"and lacks {len(only_old)} it has"
            + (f" (e.g. {only_new[:2]} / {only_old[:2]})" if only_new or only_old else " (order differs)")
        ]
    if isinstance(expected, list) and isinstance(actual, list) and len(expected) == len(actual):
        out = []
        for i, (e, a) in enumerate(zip(expected, actual)):
            out += _diff(e, a, f"{path}[{i}]")
        return out
    return [f"{path}: tracked {actual!r} != regenerated {expected!r}"]


def _verify_materialized(
    output_dir: Path, plan: Plan, labels_dir: Path,
) -> tuple[list[str], list[str]]:
    """Compare existing `fold_k/` dirs with the regenerated folds. -> (problems, notes)."""
    existing = _fold_dirs(output_dir)
    if not existing:
        return [], [f"no materialized folds under {output_dir}; only the tracked assignment was checked"]
    problems: list[str] = []
    expected_names = {f"fold_{i}" for i in range(len(plan.manifests))}
    for stale in existing:
        if stale.name not in expected_names:
            problems.append(f"{stale.name}: not part of the regenerated folds")
    try:
        sources = resolve_label_sources(plan.split_of, labels_dir)
    except FileNotFoundError as error:
        return problems + [str(error)], []
    for index, manifest in enumerate(plan.manifests):
        fold_dir = output_dir / f"fold_{index}"
        name = fold_dir.name
        if not fold_dir.is_dir():
            problems.append(f"{name}: missing")
            continue
        manifest_file = fold_dir / "splits.json"
        if not manifest_file.is_file() or manifest_file.read_text() != dump_json(manifest):
            problems.append(f"{name}/splits.json: missing or different from the regenerated manifest")
        for split in SPLITS:
            split_dir = fold_dir / "labels" / split
            on_disk = {p.stem for p in split_dir.glob("*.txt")} if split_dir.is_dir() else set()
            wanted = set(manifest[split])
            if on_disk != wanted:
                problems.append(
                    f"{name}/labels/{split}: {len(wanted - on_disk)} label file(s) missing, "
                    f"{len(on_disk - wanted)} unexpected"
                )
                continue
            changed = [s for s in sorted(wanted) if (split_dir / f"{s}.txt").read_bytes() != sources[s].read_bytes()]
            if changed:
                problems.append(f"{name}/labels/{split}: {len(changed)} label file(s) differ from the source (e.g. {changed[:2]})")
    return problems, []


def check_folds(
    manifest_path: Path,
    labels_dir: Path,
    output_dir: Path,
    assignment_file: Path,
    tz_name: str = DEFAULT_TZ,
    lot_gap_days: int = 1,
    min_lot_size: int = 5,
) -> tuple[list[str], list[str]]:
    """Regenerate in memory and compare with the tracked file. -> (problems, notes).

    Also verifies the materialized `fold_k/` dirs under `output_dir` when they
    exist. Writes nothing.
    """
    plan = build_plan(Path(manifest_path), tz_name, lot_gap_days, min_lot_size)
    assignment_file = Path(assignment_file)
    if not assignment_file.is_file():
        return [f"tracked assignment file not found: {assignment_file}"], []
    tracked_text = assignment_file.read_text()
    problems: list[str] = []
    if tracked_text != dump_json(plan.assignment):
        try:
            problems += _diff(plan.assignment, json.loads(tracked_text))
        except json.JSONDecodeError as error:
            problems.append(f"{assignment_file} is not valid JSON: {error}")
        if not problems:
            problems.append(f"{assignment_file} differs from the regenerated file only in formatting")
    mat_problems, notes = _verify_materialized(Path(output_dir), plan, Path(labels_dir))
    return problems + mat_problems, notes


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build leave-one-lot-out CV folds (zero mango leakage).",
    )
    parser.add_argument("--manifest", default=None,
                        help=f"Canonical split manifest (default: <repo>/{DEFAULT_MANIFEST}).")
    parser.add_argument("--labels-dir", default=None,
                        help=f"Canonical labels root holding <split>/<stem>.txt "
                             f"(default: <repo>/{DEFAULT_LABELS_DIR}).")
    parser.add_argument("--output-dir", default=None,
                        help=f"Where fold_k/ and folds_meta.json go (default: <repo>/{DEFAULT_OUTPUT_DIR}).")
    parser.add_argument("--assignment-file", default=None,
                        help=f"Git-tracked copy of the fold assignment (default: <repo>/{DEFAULT_ASSIGNMENT_FILE}).")
    parser.add_argument("--tz", default=DEFAULT_TZ,
                        help=f"Timezone defining the capture calendar day (default: {DEFAULT_TZ}).")
    parser.add_argument("--lot-gap-days", type=int, default=1,
                        help="Consecutive capture days at most this far apart form one lot (default: 1).")
    parser.add_argument("--min-lot-size", type=int, default=5,
                        help="Lots with fewer images merge into the nearest lot (default: 5).")
    parser.add_argument("--check", action="store_true",
                        help="Regenerate in memory and fail if it differs from the tracked "
                             "assignment file (and from any existing fold_k/ dirs). Writes nothing.")
    return parser.parse_args(argv)


def _path(value: str | None, default: str) -> Path:
    """Explicit paths are cwd-relative; defaults are anchored at the repo root."""
    return Path(value) if value is not None else PROJECT_ROOT / default


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args(argv)
    manifest = _path(args.manifest, DEFAULT_MANIFEST)
    labels_dir = _path(args.labels_dir, DEFAULT_LABELS_DIR)
    output_dir = _path(args.output_dir, DEFAULT_OUTPUT_DIR)
    assignment_file = _path(args.assignment_file, DEFAULT_ASSIGNMENT_FILE)
    params = dict(tz_name=args.tz, lot_gap_days=args.lot_gap_days, min_lot_size=args.min_lot_size)

    try:
        if args.check:
            problems, notes = check_folds(manifest, labels_dir, output_dir, assignment_file, **params)
            for note in notes:
                print(f"NOTE: {note}")
            if problems:
                print(f"DRIFT: the regenerated folds differ from {assignment_file}:")
                for problem in problems:
                    print(f"  - {problem}")
                return 1
            print(f"OK: the regenerated folds match {assignment_file}")
            return 0

        make_grouped_folds(manifest, labels_dir, output_dir, assignment_file, **params)
    except (OSError, ValueError) as error:  # includes FileNotFoundError and LeakageError
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
