"""Tests for scripts/make_grouped_folds.py (leave-one-lot-out folds).

Synthetic stems are built from local America/Lima wall-clock times, so the
lot rules are exercised exactly as they apply to the real `mango_rgb_<unix_ts>`
captures.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from scripts import make_grouped_folds as mgf
from src.training.dataset import YOLODataset

LIMA = ZoneInfo("America/Lima")

# Local capture day -> number of images. Mirrors the real dataset's shape:
# the lone 05-24 image is 7 days from the nearest lot, 06-05/06-06 are
# consecutive days of one lot, and so on. Expected lots: 24/51/42/69/50.
DAY_SIZES = {
    "2026-05-24": 1, "2026-05-31": 23,
    "2026-06-05": 40, "2026-06-06": 11,
    "2026-06-20": 20, "2026-06-21": 22,
    "2026-06-29": 41, "2026-06-30": 28,
    "2026-09-27": 50,
}
EXPECTED_LOT_SIZES = [24, 51, 42, 69, 50]


def _stem(day: str, hour: int = 10, minute: int = 0, second: int = 0) -> str:
    local = datetime.fromisoformat(f"{day}T{hour:02d}:{minute:02d}:{second:02d}").replace(tzinfo=LIMA)
    return f"mango_rgb_{int(local.timestamp())}"


def _stems_for(day_sizes: dict[str, int]) -> list[str]:
    stems = []
    for day, count in day_sizes.items():
        for i in range(count):
            stems.append(_stem(day, 8 + i // 60, i % 60))
    return sorted(stems)


def _write_dataset(root: Path, day_sizes: dict[str, int] = DAY_SIZES) -> tuple[Path, Path, list[str]]:
    """Write a canonical manifest + label tree; return (manifest, labels_dir, stems)."""
    stems = _stems_for(day_sizes)
    manifest = {"train": stems[::3], "val": stems[1::3], "test": stems[2::3]}
    manifest_path = root / "splits.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    labels_dir = root / "labels"
    for split, split_stems in manifest.items():
        (labels_dir / split).mkdir(parents=True, exist_ok=True)
        for stem in split_stems:
            idx = stems.index(stem)
            # Every image has a mango (class 0); every third also has damage
            # boxes (class 1), 1 + idx % 2 of them. Distinct bytes per stem.
            lines = [f"0 0.5 0.5 0.4 0.{idx % 9 + 1}"]
            if idx % 3 == 0:
                lines += [f"1 0.4 0.4 0.1 0.{idx % 9 + 1}"] * (1 + idx % 2)
            (labels_dir / split / f"{stem}.txt").write_text("\n".join(lines) + "\n")
    return manifest_path, labels_dir, stems


def _generate(tmp_path: Path, name: str = "out", **overrides) -> dict:
    """Generate folds into `tmp_path/<name>` (writing the dataset on first use)."""
    if not (tmp_path / "splits.json").exists():
        _write_dataset(tmp_path)
    kwargs = dict(
        manifest_path=tmp_path / "splits.json",
        labels_dir=tmp_path / "labels",
        output_dir=tmp_path / name,
        assignment_file=tmp_path / f"{name}_assignment.json",
    )
    kwargs.update(overrides)
    return mgf.make_grouped_folds(**kwargs)


# --------------------------------------------------------------------------
# Lot grouping
# --------------------------------------------------------------------------


def test_local_day_uses_timezone_not_utc() -> None:
    # 22:30 in Lima on 06-05 is already 06-06 in UTC; it must stay in 06-05's lot.
    late = _stem("2026-06-05", 22, 30)
    early = _stem("2026-06-05", 8, 0)
    assert mgf.local_date(mgf.stem_timestamp(late), "America/Lima").isoformat() == "2026-06-05"
    lots = mgf.group_into_lots([late, early], "America/Lima", lot_gap_days=1, min_lot_size=1)
    assert [lot.days for lot in lots] == [["2026-06-05"]]


def test_unknown_timezone_fails_with_an_install_hint() -> None:
    with pytest.raises(ValueError, match="tzdata"):
        mgf.group_into_lots([_stem("2026-06-05")], "Not/AZone")


def test_stem_timestamp_rejects_unparseable_stem() -> None:
    with pytest.raises(ValueError, match="timestamp"):
        mgf.stem_timestamp("mango_rgb_notanumber")


def test_consecutive_days_join_one_lot_and_gaps_split_lots() -> None:
    stems = [_stem("2026-06-05", 10, i) for i in range(6)] + [_stem("2026-06-06", 10, i) for i in range(6)]
    lots = mgf.group_into_lots(stems, "America/Lima", lot_gap_days=1, min_lot_size=5)
    assert [(lot.days, len(lot.stems)) for lot in lots] == [(["2026-06-05", "2026-06-06"], 12)]

    # A one-day hole (06-05 -> 06-07) is a gap of 2 days: separate lots at
    # lot_gap_days=1, one lot at lot_gap_days=2.
    stems = [_stem("2026-06-05", 10, i) for i in range(6)] + [_stem("2026-06-07", 10, i) for i in range(6)]
    split = mgf.group_into_lots(stems, "America/Lima", lot_gap_days=1, min_lot_size=5)
    assert [lot.days for lot in split] == [["2026-06-05"], ["2026-06-07"]]
    joined = mgf.group_into_lots(stems, "America/Lima", lot_gap_days=2, min_lot_size=5)
    assert [lot.days for lot in joined] == [["2026-06-05", "2026-06-07"]]


def test_small_lot_merges_into_chronologically_nearest_lot() -> None:
    # The lone 05-24 image is 7 days before 05-31 and has no earlier neighbour.
    stems = (
        [_stem("2026-05-24")]
        + [_stem("2026-05-31", 10, i) for i in range(6)]
        + [_stem("2026-06-20", 10, i) for i in range(6)]
    )
    lots = mgf.group_into_lots(stems, "America/Lima", lot_gap_days=1, min_lot_size=5)
    assert [(lot.lot_id, lot.days, len(lot.stems)) for lot in lots] == [
        ("lot1", ["2026-05-24", "2026-05-31"], 7),
        ("lot2", ["2026-06-20"], 6),
    ]


def test_small_lot_in_the_middle_goes_to_the_closer_neighbour() -> None:
    near_next = (
        [_stem("2026-06-01", 10, i) for i in range(6)]
        + [_stem("2026-06-09")]                                   # 8 days after, 2 days before
        + [_stem("2026-06-11", 10, i) for i in range(6)]
    )
    lots = mgf.group_into_lots(near_next, "America/Lima", lot_gap_days=1, min_lot_size=5)
    assert [lot.days for lot in lots] == [["2026-06-01"], ["2026-06-09", "2026-06-11"]]

    near_prev = (
        [_stem("2026-06-01", 10, i) for i in range(6)]
        + [_stem("2026-06-03")]                                   # 2 days after, 8 days before
        + [_stem("2026-06-11", 10, i) for i in range(6)]
    )
    lots = mgf.group_into_lots(near_prev, "America/Lima", lot_gap_days=1, min_lot_size=5)
    assert [lot.days for lot in lots] == [["2026-06-01", "2026-06-03"], ["2026-06-11"]]


def test_lot_ids_are_chronological_whatever_the_input_order() -> None:
    stems = _stems_for(DAY_SIZES)
    forward = mgf.group_into_lots(stems, "America/Lima")
    backward = mgf.group_into_lots(list(reversed(stems)), "America/Lima")
    assert [lot.lot_id for lot in forward] == ["lot1", "lot2", "lot3", "lot4", "lot5"]
    assert [(lot.lot_id, lot.days, lot.stems) for lot in forward] == [
        (lot.lot_id, lot.days, lot.stems) for lot in backward
    ]
    starts = [lot.start for lot in forward]
    assert starts == sorted(starts)


def test_real_capture_pattern_yields_the_five_expected_lots() -> None:
    lots = mgf.group_into_lots(_stems_for(DAY_SIZES), "America/Lima")
    assert [len(lot.stems) for lot in lots] == EXPECTED_LOT_SIZES
    assert lots[0].days == ["2026-05-24", "2026-05-31"]
    assert lots[1].days == ["2026-06-05", "2026-06-06"]
    assert lots[2].days == ["2026-06-20", "2026-06-21"]
    assert lots[3].days == ["2026-06-29", "2026-06-30"]
    assert lots[4].days == ["2026-09-27"]


# --------------------------------------------------------------------------
# Fold assignment
# --------------------------------------------------------------------------


def test_fewer_than_three_lots_fails_clearly() -> None:
    lots = mgf.group_into_lots(_stems_for({"2026-06-05": 6, "2026-06-20": 6}), "America/Lima")
    assert len(lots) == 2
    with pytest.raises(ValueError, match="at least 3 lots"):
        mgf.assign_folds(lots)


def test_each_lot_is_test_exactly_once_and_val_is_the_next_lot() -> None:
    lots = mgf.group_into_lots(_stems_for(DAY_SIZES), "America/Lima")
    folds = mgf.assign_folds(lots)
    ids = [lot.lot_id for lot in lots]
    k = len(lots)
    assert len(folds) == k == 5
    assert sorted(fold.test for fold in folds) == sorted(ids)
    for i, fold in enumerate(folds):
        assert fold.index == i
        assert fold.test == ids[i]
        assert fold.val == ids[(i + 1) % k]
        assert fold.train == [lot_id for lot_id in ids if lot_id not in (fold.test, fold.val)]
        assert len({fold.test, fold.val, *fold.train}) == k


def test_fold_manifests_are_disjoint_complete_and_sorted() -> None:
    stems = _stems_for(DAY_SIZES)
    lots = mgf.group_into_lots(stems, "America/Lima")
    manifests = mgf.fold_manifests(lots, mgf.assign_folds(lots))
    tested: dict[str, int] = {}
    for manifest in manifests:
        assert set(manifest) == {"train", "val", "test"}
        for split in manifest:
            assert manifest[split] == sorted(manifest[split])
        assert not set(manifest["train"]) & set(manifest["val"])
        assert not set(manifest["train"]) & set(manifest["test"])
        assert not set(manifest["val"]) & set(manifest["test"])
        assert set(manifest["train"]) | set(manifest["val"]) | set(manifest["test"]) == set(stems)
        for stem in manifest["test"]:
            tested[stem] = tested.get(stem, 0) + 1
    assert tested == {stem: 1 for stem in stems}


# --------------------------------------------------------------------------
# Leakage report
# --------------------------------------------------------------------------


def test_leakage_report_gap_is_more_than_a_day_for_every_fold() -> None:
    lots = mgf.group_into_lots(_stems_for(DAY_SIZES), "America/Lima")
    folds = mgf.assign_folds(lots)
    report = mgf.leakage_report(lots, folds)
    assert report["all_disjoint"] is True
    assert report["every_stem_tested_exactly_once"] is True
    assert set(report["folds"]) == {"0", "1", "2", "3", "4"}
    for fold_report in report["folds"].values():
        assert fold_report["min_gap_seconds"] > 86400
        assert fold_report["min_gap_days"] > 1
        eval_stem, train_stem = fold_report["closest_pair"]
        assert abs(mgf.stem_timestamp(eval_stem) - mgf.stem_timestamp(train_stem)) == fold_report["min_gap_seconds"]
    assert report["min_gap_seconds"] == min(f["min_gap_seconds"] for f in report["folds"].values())


def test_leakage_report_rejects_a_stem_shared_between_splits() -> None:
    lots = mgf.group_into_lots(_stems_for(DAY_SIZES), "America/Lima")
    folds = mgf.assign_folds(lots)
    lots[2].stems.append(lots[0].stems[0])  # one stem now sits in two lots
    with pytest.raises(mgf.LeakageError, match="disjoint|more than one"):
        mgf.leakage_report(lots, folds)


def test_leakage_report_rejects_lots_closer_than_a_day() -> None:
    # Two lots only 6 hours apart: a train/test pair would be a re-photograph.
    stems = (
        [_stem("2026-06-05", 6, i) for i in range(6)]
        + [_stem("2026-06-05", 18, i) for i in range(6)]
        + [_stem("2026-06-20", 10, i) for i in range(6)]
    )
    # Force the same-day stems into separate lots, as a mis-built grouping would.
    lots = [
        mgf.Lot("lot1", ["2026-06-05"], sorted(stems[:6])),
        mgf.Lot("lot2", ["2026-06-05"], sorted(stems[6:12])),
        mgf.Lot("lot3", ["2026-06-20"], sorted(stems[12:])),
    ]
    with pytest.raises(mgf.LeakageError, match="gap"):
        mgf.leakage_report(lots, mgf.assign_folds(lots))


# --------------------------------------------------------------------------
# Materialization
# --------------------------------------------------------------------------


def test_materialized_folds_have_manifest_schema_and_copied_labels(tmp_path: Path) -> None:
    manifest_path, labels_dir, stems = _write_dataset(tmp_path)
    mgf.make_grouped_folds(
        manifest_path=manifest_path, labels_dir=labels_dir,
        output_dir=tmp_path / "out", assignment_file=tmp_path / "assignment.json",
    )
    canonical = {
        stem: split
        for split, split_stems in json.loads(manifest_path.read_text()).items()
        for stem in split_stems
    }
    tested = []
    for k in range(5):
        fold_dir = tmp_path / "out" / f"fold_{k}"
        manifest = json.loads((fold_dir / "splits.json").read_text())
        assert set(manifest) == {"train", "val", "test"}
        for split, split_stems in manifest.items():
            on_disk = sorted(p.stem for p in (fold_dir / "labels" / split).glob("*.txt"))
            assert on_disk == split_stems
            for stem in split_stems:
                copy = fold_dir / "labels" / split / f"{stem}.txt"
                source = labels_dir / canonical[stem] / f"{stem}.txt"
                assert copy.read_bytes() == source.read_bytes()
                assert not copy.is_symlink()
        tested += manifest["test"]
    assert sorted(tested) == stems


def test_yolodataset_accepts_materialized_folds_without_code_changes(tmp_path: Path) -> None:
    """The dataset verifies each split's label dir against the fold manifest."""
    _generate(tmp_path)
    for k in range(5):
        fold_dir = tmp_path / "out" / f"fold_{k}"
        manifest = json.loads((fold_dir / "splits.json").read_text())
        for split in ("train", "val", "test"):
            YOLODataset(
                rgb_dir=tmp_path / "no_rgb", nir_dir=tmp_path / "no_nir",
                labels_dir=fold_dir / "labels", split=split,
                manifest_path=fold_dir / "splits.json", nir_homography_path=None,
            )  # raises ValueError if the label dir diverges from the manifest
            assert len(manifest[split]) > 0


def test_regenerating_into_the_same_directory_removes_stale_files(tmp_path: Path) -> None:
    manifest_path, labels_dir, _ = _write_dataset(tmp_path)
    out = tmp_path / "out"
    mgf.make_grouped_folds(manifest_path=manifest_path, labels_dir=labels_dir, output_dir=out,
                           assignment_file=tmp_path / "a.json")
    assert (out / "fold_4").is_dir()
    # A coarser grouping yields fewer lots -> fewer folds; fold_4 must not linger.
    mgf.make_grouped_folds(manifest_path=manifest_path, labels_dir=labels_dir, output_dir=out,
                           assignment_file=tmp_path / "a.json", lot_gap_days=10)
    meta = json.loads((out / "folds_meta.json").read_text())
    k = len(meta["lots"])
    assert k < 5
    assert sorted(p.name for p in out.glob("fold_*")) == [f"fold_{i}" for i in range(k)]
    for i in range(k):
        manifest = json.loads((out / f"fold_{i}" / "splits.json").read_text())
        for split, split_stems in manifest.items():
            assert sorted(p.stem for p in (out / f"fold_{i}" / "labels" / split).glob("*.txt")) == split_stems


def test_missing_label_file_fails_clearly(tmp_path: Path) -> None:
    manifest_path, labels_dir, stems = _write_dataset(tmp_path)
    victim = next(labels_dir.glob("*/*.txt"))
    victim.unlink()
    with pytest.raises(FileNotFoundError, match=victim.stem):
        mgf.make_grouped_folds(manifest_path=manifest_path, labels_dir=labels_dir,
                               output_dir=tmp_path / "out", assignment_file=tmp_path / "a.json")


def test_output_dir_overlapping_the_labels_dir_is_refused(tmp_path: Path) -> None:
    manifest_path, labels_dir, _ = _write_dataset(tmp_path)
    for bad in (labels_dir, labels_dir / "train", tmp_path):
        with pytest.raises(ValueError, match="overlap"):
            mgf.make_grouped_folds(manifest_path=manifest_path, labels_dir=labels_dir,
                                   output_dir=bad, assignment_file=tmp_path / "a.json")
    assert (labels_dir / "train").is_dir() and any((labels_dir / "train").glob("*.txt"))


def test_a_stem_in_two_canonical_splits_is_rejected(tmp_path: Path) -> None:
    manifest_path, labels_dir, stems = _write_dataset(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    manifest["val"].append(manifest["train"][0])
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="more than one split"):
        mgf.make_grouped_folds(manifest_path=manifest_path, labels_dir=labels_dir,
                               output_dir=tmp_path / "out", assignment_file=tmp_path / "a.json")


# --------------------------------------------------------------------------
# Meta + tracked assignment
# --------------------------------------------------------------------------


def test_meta_counts_images_damage_boxes_and_negatives(tmp_path: Path) -> None:
    meta = _generate(tmp_path)
    stems = _stems_for(DAY_SIZES)
    damage = {stem: (1 + i % 2 if i % 3 == 0 else 0) for i, stem in enumerate(stems)}
    assert [meta["lots"][f"lot{i}"]["num_images"] for i in range(1, 6)] == EXPECTED_LOT_SIZES
    lot_stems = {lot.lot_id: lot.stems for lot in mgf.group_into_lots(stems, "America/Lima")}
    for lot_id, lot in meta["lots"].items():
        assert lot["damage_boxes"] == sum(damage[s] for s in lot_stems[lot_id])
        assert lot["negatives"] == sum(1 for s in lot_stems[lot_id] if damage[s] == 0)
    fold0 = meta["folds"]["0"]
    assert fold0["test"] == "lot1" and fold0["val"] == "lot2" and fold0["train"] == ["lot3", "lot4", "lot5"]
    assert fold0["splits"]["test"]["images"] == 24
    assert fold0["splits"]["val"]["images"] == 51
    assert fold0["splits"]["train"]["images"] == 42 + 69 + 50
    for split_counts in fold0["splits"].values():
        assert split_counts["images"] == split_counts["with_damage"] + split_counts["negatives"]
    assert sum(f["splits"]["test"]["damage_boxes"] for f in meta["folds"].values()) == sum(damage.values())
    assert meta["leakage"]["every_stem_tested_exactly_once"] is True
    assert (tmp_path / "out" / "folds_meta.json").is_file()


def test_tracked_assignment_lists_lots_folds_and_the_manifest_hash(tmp_path: Path) -> None:
    import hashlib

    _generate(tmp_path)
    assignment = json.loads((tmp_path / "out_assignment.json").read_text())
    assert assignment["source_manifest_sha256"] == hashlib.sha256((tmp_path / "splits.json").read_bytes()).hexdigest()
    assert assignment["params"] == {"lot_gap_days": 1, "min_lot_size": 5, "tz": "America/Lima"}
    assert assignment["num_images"] == sum(DAY_SIZES.values())
    assert [assignment["lots"][f"lot{i}"]["num_images"] for i in range(1, 6)] == EXPECTED_LOT_SIZES
    assert assignment["lots"]["lot1"]["days"] == ["2026-05-24", "2026-05-31"]
    assert len(assignment["lots"]["lot1"]["stems"]) == 24
    assert [f["test"] for f in assignment["folds"]] == ["lot1", "lot2", "lot3", "lot4", "lot5"]
    assert [f["val"] for f in assignment["folds"]] == ["lot2", "lot3", "lot4", "lot5", "lot1"]
    # No machine-specific content: it must be identical across machines.
    text = (tmp_path / "out_assignment.json").read_text()
    assert str(tmp_path) not in text


# --------------------------------------------------------------------------
# Determinism and --check
# --------------------------------------------------------------------------


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


def test_two_runs_are_byte_identical(tmp_path: Path) -> None:
    _write_dataset(tmp_path)
    _generate(tmp_path, "a")
    _generate(tmp_path, "b")
    assert _tree_bytes(tmp_path / "a") == _tree_bytes(tmp_path / "b")
    assert (tmp_path / "a_assignment.json").read_bytes() == (tmp_path / "b_assignment.json").read_bytes()


def _cli(tmp_path: Path, *extra: str) -> int:
    return mgf.main([
        "--manifest", str(tmp_path / "splits.json"),
        "--labels-dir", str(tmp_path / "labels"),
        "--output-dir", str(tmp_path / "cli_out"),
        "--assignment-file", str(tmp_path / "tracked.json"),
        *extra,
    ])


def test_check_passes_on_identical_regeneration_and_writes_nothing(tmp_path: Path, capsys) -> None:
    _write_dataset(tmp_path)
    assert _cli(tmp_path) == 0
    before = _tree_bytes(tmp_path)
    assert _cli(tmp_path, "--check") == 0
    assert _tree_bytes(tmp_path) == before
    assert "OK" in capsys.readouterr().out


def test_check_detects_a_drifted_fold_assignment(tmp_path: Path, capsys) -> None:
    _write_dataset(tmp_path)
    assert _cli(tmp_path) == 0
    tracked = tmp_path / "tracked.json"
    drifted = json.loads(tracked.read_text())
    stem = drifted["lots"]["lot1"]["stems"].pop()
    drifted["lots"]["lot2"]["stems"].append(stem)
    tracked.write_text(json.dumps(drifted, indent=2, sort_keys=True) + "\n")
    assert _cli(tmp_path, "--check") == 1
    assert "DRIFT" in capsys.readouterr().out


def test_check_detects_a_changed_source_manifest(tmp_path: Path, capsys) -> None:
    _write_dataset(tmp_path)
    assert _cli(tmp_path) == 0
    manifest_path = tmp_path / "splits.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["train"], manifest["val"] = manifest["val"], manifest["train"]  # same stems, other file
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    assert _cli(tmp_path, "--check") == 1
    assert "source_manifest_sha256" in capsys.readouterr().out


def test_check_fails_when_the_tracked_file_is_missing(tmp_path: Path, capsys) -> None:
    _write_dataset(tmp_path)
    assert _cli(tmp_path, "--check") == 1
    assert "not found" in capsys.readouterr().out


def test_check_also_verifies_materialized_folds_when_present(tmp_path: Path, capsys) -> None:
    _write_dataset(tmp_path)
    assert _cli(tmp_path) == 0
    victim = next((tmp_path / "cli_out" / "fold_0" / "labels" / "test").glob("*.txt"))
    victim.unlink()
    assert _cli(tmp_path, "--check") == 1
    assert "fold_0" in capsys.readouterr().out


def test_cli_reports_a_clear_error_for_too_few_lots(tmp_path: Path, capsys) -> None:
    _write_dataset(tmp_path, {"2026-06-05": 10, "2026-06-20": 10})
    assert _cli(tmp_path) == 2
    assert "at least 3 lots" in capsys.readouterr().err
    assert not (tmp_path / "cli_out").exists()
    assert not (tmp_path / "tracked.json").exists()
