"""Regression tests for split-manifest integrity (Phase 0 / W0).

Reproduces the verified leakage defect: `prepare_yolo_splits.py`'s
`write_empty_label_files` never pruned stray label files from earlier
shuffles, and `YOLODataset._load_pairs` read directory contents instead of
the manifest — so the effective split was the union of every historical
assignment. See openspec/changes/damage-map-audit/design.md Prerequisite P1.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.prepare_yolo_splits import (
    apply_reconciliation,
    plan_reconciliation,
    reconcile_splits,
    reconcile_splits_resolving_conflicts,
    resolve_conflict,
    write_conflict_resolution_log,
    write_empty_label_files,
)
from src.training.dataset import YOLODataset


def _write_label(path: Path, content: str = "0 0.5 0.5 0.1 0.1\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def _leaked_fixture(tmp_path: Path) -> tuple[Path, dict[str, list[str]]]:
    """Build a label-directory tree with the exact leakage shape from the audit.

    Manifest: train=[t1, t2], val=[v1], test=[]
    Disk: labels/train has an extra stray copy of the val-only stem, and
    labels/val has an extra stray copy of a train-only stem — reproducing
    the observed train/val leakage from an earlier shuffle.
    """
    labels_dir = tmp_path / "labels"
    manifest = {
        "train": ["t1", "t2"],
        "val": ["v1"],
        "test": [],
    }

    # Correctly-placed files.
    for stem in manifest["train"]:
        _write_label(labels_dir / "train" / f"{stem}.txt")
    for stem in manifest["val"]:
        _write_label(labels_dir / "val" / f"{stem}.txt")

    # Stray leakage: t1 also lingers in val/ (byte-identical to train/t1.txt),
    # and v1 also lingers in train/ (byte-identical to val/v1.txt).
    _write_label(labels_dir / "val" / "t1.txt", content=(labels_dir / "train" / "t1.txt").read_text())
    _write_label(labels_dir / "train" / "v1.txt", content=(labels_dir / "val" / "v1.txt").read_text())

    return labels_dir, manifest


def test_leakage_is_reproduced_on_fixture(tmp_path: Path) -> None:
    """train/val/test directory contents disagree with the manifest before reconciliation."""
    labels_dir, manifest = _leaked_fixture(tmp_path)

    train_stems = {p.stem for p in (labels_dir / "train").glob("*.txt")}
    val_stems = {p.stem for p in (labels_dir / "val").glob("*.txt")}

    assert train_stems & val_stems == {"t1", "v1"}
    assert train_stems != set(manifest["train"])
    assert val_stems != set(manifest["val"])


def test_dry_run_reconcile_mutates_nothing(tmp_path: Path) -> None:
    labels_dir, manifest = _leaked_fixture(tmp_path)

    before_train = sorted(p.name for p in (labels_dir / "train").glob("*.txt"))
    before_val = sorted(p.name for p in (labels_dir / "val").glob("*.txt"))

    report = reconcile_splits(labels_dir, manifest, dry_run=True)

    after_train = sorted(p.name for p in (labels_dir / "train").glob("*.txt"))
    after_val = sorted(p.name for p in (labels_dir / "val").glob("*.txt"))

    assert before_train == after_train
    assert before_val == after_val
    assert report.applied is False
    assert not report.has_conflicts
    assert report.strays["train"] == ["v1"]
    assert report.strays["val"] == ["t1"]


def test_reconcile_removes_only_manifest_orphaned_stems(tmp_path: Path) -> None:
    labels_dir, manifest = _leaked_fixture(tmp_path)

    report = reconcile_splits(labels_dir, manifest, dry_run=False)

    assert report.applied is True
    assert not (labels_dir / "train" / "v1.txt").exists()
    assert not (labels_dir / "val" / "t1.txt").exists()

    # Correctly-manifested files must survive untouched.
    remaining_train = {p.stem for p in (labels_dir / "train").glob("*.txt")}
    remaining_val = {p.stem for p in (labels_dir / "val").glob("*.txt")}
    assert remaining_train == set(manifest["train"])
    assert remaining_val == set(manifest["val"])


def test_reconcile_reports_disk_stem_absent_from_manifest_without_deleting(tmp_path: Path) -> None:
    labels_dir, manifest = _leaked_fixture(tmp_path)
    _write_label(labels_dir / "train" / "unmanifested_stem.txt")

    report = reconcile_splits(labels_dir, manifest, dry_run=False)

    assert report.unmanifested == ["unmanifested_stem"]
    # Reported, never deleted, even in a real (non-dry-run) reconciliation.
    assert (labels_dir / "train" / "unmanifested_stem.txt").exists()


def test_reconcile_refuses_to_delete_on_byte_mismatch_conflict(tmp_path: Path) -> None:
    labels_dir, manifest = _leaked_fixture(tmp_path)
    # Corrupt the stray copy so it no longer matches the manifest-correct file.
    (labels_dir / "val" / "t1.txt").write_text("1 0.9 0.9 0.05 0.05\n")

    report = reconcile_splits(labels_dir, manifest, dry_run=False)

    assert report.applied is False
    assert report.conflicts["val"] == ["t1"]
    # Nothing was deleted anywhere while a conflict exists (fail closed / all-or-nothing).
    assert (labels_dir / "val" / "t1.txt").exists()
    assert (labels_dir / "train" / "v1.txt").exists()


def test_apply_reconciliation_raises_on_conflicting_report(tmp_path: Path) -> None:
    labels_dir, manifest = _leaked_fixture(tmp_path)
    (labels_dir / "val" / "t1.txt").write_text("1 0.9 0.9 0.05 0.05\n")

    report = plan_reconciliation(labels_dir, manifest)
    assert report.has_conflicts

    with pytest.raises(ValueError, match="conflicts"):
        apply_reconciliation(labels_dir, report)


def test_write_empty_label_files_prunes_strays_not_in_target_split(tmp_path: Path) -> None:
    labels_dir = tmp_path / "labels"
    # Simulate a stray left behind by an earlier shuffle.
    _write_label(labels_dir / "train" / "old_shuffle_stem.txt")

    split_map = {"train": ["new_stem"], "val": [], "test": []}
    write_empty_label_files(split_map, labels_dir)

    train_stems = {p.stem for p in (labels_dir / "train").glob("*.txt")}
    assert train_stems == {"new_stem"}


class TestYOLODatasetManifestGuard:
    """W0: YOLODataset._load_pairs must fail closed on a manifest/directory mismatch."""

    def _make_pair(self, root: Path, split: str, stem: str) -> None:
        rgb_dir = root / "rgb"
        nir_dir = root / "nir"
        labels_dir = root / "labels" / split
        rgb_dir.mkdir(parents=True, exist_ok=True)
        nir_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        (rgb_dir / f"{stem}.jpg").touch()
        (nir_dir / f"{stem.replace('_rgb', '_nir')}.jpg").touch()
        _write_label(labels_dir / f"{stem}.txt")

    def test_no_manifest_path_skips_guard(self, tmp_path: Path) -> None:
        """Backward compatibility: omitting manifest_path performs no check."""
        self._make_pair(tmp_path, "train", "mango_rgb_001")

        dataset = YOLODataset(
            rgb_dir=tmp_path / "rgb",
            nir_dir=tmp_path / "nir",
            labels_dir=tmp_path / "labels",
            split="train",
            nir_homography_path=None,
        )
        assert len(dataset) == 1

    def test_matching_manifest_loads_cleanly(self, tmp_path: Path) -> None:
        self._make_pair(tmp_path, "train", "mango_rgb_001")
        manifest_path = tmp_path / "splits.json"
        manifest_path.write_text(json.dumps({"train": ["mango_rgb_001"], "val": [], "test": []}))

        dataset = YOLODataset(
            rgb_dir=tmp_path / "rgb",
            nir_dir=tmp_path / "nir",
            labels_dir=tmp_path / "labels",
            split="train",
            manifest_path=manifest_path,
            nir_homography_path=None,
        )
        assert len(dataset) == 1

    def test_diverging_directory_raises(self, tmp_path: Path) -> None:
        self._make_pair(tmp_path, "train", "mango_rgb_001")
        # Manifest disagrees with what's on disk for "train".
        manifest_path = tmp_path / "splits.json"
        manifest_path.write_text(json.dumps({"train": [], "val": ["mango_rgb_001"], "test": []}))

        with pytest.raises(ValueError, match="diverges from"):
            YOLODataset(
                rgb_dir=tmp_path / "rgb",
                nir_dir=tmp_path / "nir",
                labels_dir=tmp_path / "labels",
                split="train",
                manifest_path=manifest_path,
                nir_homography_path=None,
            )

    def test_gt_count_report_matches_direct_recount(self, tmp_path: Path) -> None:
        self._make_pair(tmp_path, "train", "mango_rgb_001")
        dataset = YOLODataset(
            rgb_dir=tmp_path / "rgb",
            nir_dir=tmp_path / "nir",
            labels_dir=tmp_path / "labels",
            split="train",
            nir_homography_path=None,
        )
        report = dataset.gt_count_report()
        assert report["loaded"] == report["on_disk"]
        assert report["image_count"] == 1


class TestConflictResolutionQ11:
    """Round 5 Q11: opt-in resolution of annotation conflicts.

    Rule: keep the copy with more class-1 (damage) boxes, regardless of
    which split holds it. Tiebreak: (1) larger total class-1 area,
    (2) the manifest-correct copy.
    """

    def test_max_damage_count_wins_regardless_of_split(self, tmp_path: Path) -> None:
        """The wrong-split copy wins when it has strictly more damage boxes."""
        labels_dir = tmp_path / "labels"
        wrong_path = labels_dir / "train" / "stem_a.txt"
        correct_path = labels_dir / "val" / "stem_a.txt"
        _write_label(wrong_path, "0 0.5 0.5 0.1 0.1\n")  # 0 damage boxes (mango only)
        _write_label(
            correct_path,
            "0 0.5 0.5 0.1 0.1\n1 0.2 0.2 0.05 0.05\n1 0.6 0.6 0.05 0.05\n",  # 2 damage boxes
        )

        decision = resolve_conflict(
            "stem_a", "train", wrong_path, "val", correct_path, manifest_split="val",
        )

        assert decision.rule == "max_damage_count"
        assert decision.kept_split == "val"
        assert decision.kept_damage_count == 2
        assert decision.discarded_split == "train"
        assert decision.discarded_damage_count == 0

    def test_wrong_split_copy_can_win_over_manifest_correct_copy(self, tmp_path: Path) -> None:
        """Damage count decides the winner even when it means keeping the
        copy that is NOT where the manifest says the stem belongs."""
        labels_dir = tmp_path / "labels"
        train_path = labels_dir / "train" / "stem_b.txt"
        val_path = labels_dir / "val" / "stem_b.txt"
        _write_label(train_path, "0 0.5 0.5 0.1 0.1\n1 0.1 0.1 0.02 0.02\n" * 6)  # 6 damage boxes
        _write_label(val_path, "0 0.5 0.5 0.1 0.1\n")  # 0 damage boxes, but this is manifest-correct

        decision = resolve_conflict(
            "stem_b", "train", train_path, "val", val_path, manifest_split="val",
        )

        assert decision.rule == "max_damage_count"
        assert decision.kept_split == "train"  # wins on damage count despite being "wrong"
        assert decision.manifest_split == "val"  # destination is still governed by the manifest

    def test_tiebreak_area_prefers_larger_total_damage_area(self, tmp_path: Path) -> None:
        labels_dir = tmp_path / "labels"
        train_path = labels_dir / "train" / "stem_c.txt"
        val_path = labels_dir / "val" / "stem_c.txt"
        # Both have exactly 1 damage box (equal count) but different area.
        _write_label(train_path, "1 0.5 0.5 0.05 0.05\n")  # area = 0.0025
        _write_label(val_path, "1 0.5 0.5 0.20 0.20\n")  # area = 0.04 (larger)

        decision = resolve_conflict(
            "stem_c", "train", train_path, "val", val_path, manifest_split="train",
        )

        assert decision.rule == "tiebreak_area"
        assert decision.kept_split == "val"
        assert decision.kept_damage_area > decision.discarded_damage_area

    def test_tiebreak_manifest_copy_when_count_and_area_both_tie(self, tmp_path: Path) -> None:
        labels_dir = tmp_path / "labels"
        train_path = labels_dir / "train" / "stem_d.txt"
        val_path = labels_dir / "val" / "stem_d.txt"
        # Identical count and identical total area, different coordinates
        # (mirrors the real mango_rgb_1780238853 case: same box shape, moved).
        _write_label(train_path, "1 0.3 0.435 0.05 0.05\n")
        _write_label(val_path, "1 0.3 0.358 0.05 0.05\n")

        decision = resolve_conflict(
            "stem_d", "train", train_path, "val", val_path, manifest_split="val",
        )

        assert decision.rule == "tiebreak_manifest_copy"
        assert decision.kept_split == "val"  # the manifest-correct copy

    def test_reconcile_resolving_conflicts_is_opt_in_default_still_fails_closed(self, tmp_path: Path) -> None:
        """Without --resolve-conflicts (i.e. calling reconcile_splits directly),
        a conflict must still abort with nothing deleted — Q11 does not
        weaken the default fail-closed behavior."""
        labels_dir, manifest = _leaked_fixture(tmp_path)
        (labels_dir / "val" / "t1.txt").write_text("1 0.9 0.9 0.05 0.05\n")  # force a real conflict

        report = reconcile_splits(labels_dir, manifest, dry_run=False)

        assert report.applied is False
        assert report.has_conflicts

    def test_dry_run_resolving_conflicts_mutates_nothing(self, tmp_path: Path) -> None:
        labels_dir, manifest = _leaked_fixture(tmp_path)
        (labels_dir / "val" / "t1.txt").write_text("1 0.9 0.9 0.05 0.05\n1 0.1 0.1 0.05 0.05\n")

        before = sorted(p.read_bytes() for p in labels_dir.rglob("*.txt"))
        report, decisions = reconcile_splits_resolving_conflicts(labels_dir, manifest, dry_run=True)
        after = sorted(p.read_bytes() for p in labels_dir.rglob("*.txt"))

        assert before == after
        assert len(decisions) == 1
        assert decisions[0].stem == "t1"

    def test_resolving_conflicts_applies_winner_and_removes_loser(self, tmp_path: Path) -> None:
        labels_dir, manifest = _leaked_fixture(tmp_path)
        # train/t1.txt (manifest-correct) has 0 damage boxes; val/t1.txt
        # (the leaked stray copy) has 2 — the stray should win and be
        # promoted into the manifest-correct train/ location.
        winning_content = "0 0.5 0.5 0.1 0.1\n1 0.2 0.2 0.05 0.05\n1 0.6 0.6 0.05 0.05\n"
        (labels_dir / "val" / "t1.txt").write_text(winning_content)

        report, decisions = reconcile_splits_resolving_conflicts(labels_dir, manifest, dry_run=False)

        assert report.applied is True
        assert not report.has_conflicts
        # Winner promoted to the manifest-correct (train) location.
        assert (labels_dir / "train" / "t1.txt").read_text() == winning_content
        # Loser (val copy) removed.
        assert not (labels_dir / "val" / "t1.txt").exists()
        # The other pre-existing (non-conflicting) leaked stray still gets cleaned up.
        assert not (labels_dir / "train" / "v1.txt").exists()

    def test_unmanifested_stem_still_reported_never_deleted_with_resolve_conflicts(self, tmp_path: Path) -> None:
        labels_dir, manifest = _leaked_fixture(tmp_path)
        _write_label(labels_dir / "train" / "orphan_stem.txt")
        (labels_dir / "val" / "t1.txt").write_text("1 0.9 0.9 0.05 0.05\n")

        report, decisions = reconcile_splits_resolving_conflicts(labels_dir, manifest, dry_run=False)

        assert report.unmanifested == ["orphan_stem"]
        assert (labels_dir / "train" / "orphan_stem.txt").exists()

    def test_decision_log_is_written_and_auditable(self, tmp_path: Path) -> None:
        labels_dir, manifest = _leaked_fixture(tmp_path)
        (labels_dir / "val" / "t1.txt").write_text("1 0.9 0.9 0.05 0.05\n")

        _, decisions = reconcile_splits_resolving_conflicts(labels_dir, manifest, dry_run=True)
        log_path = tmp_path / "w0-reconcile.md"
        write_conflict_resolution_log(decisions, unmanifested=[], output_path=log_path)

        content = log_path.read_text()
        assert "Conflict Resolution Log (Q11)" in content
        assert "t1" in content
        assert decisions[0].rule in content
