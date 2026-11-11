"""`scripts/top1_localization.py` on a synthetic runs root: discovery, the test-reading rule and the outputs."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scripts import top1_localization as cli
from src.data_pipeline.v4_selection import LATENT_HOURS

LATENT = sorted(LATENT_HOURS)[:2]
DEVELOPED = max(LATENT_HOURS) + 2
GT = [0.25, 0.25, 0.75, 0.75]
MISS = [0.8, 0.8, 0.95, 0.95]
SQUARE = [0.25, 0.25, 0.75, 0.25, 0.75, 0.75, 0.25, 0.75]  # flat normalized polygon of GT
MISS_SQUARE = [0.8, 0.8, 0.95, 0.8, 0.95, 0.95, 0.8, 0.95]
FOLDS = (0, 1)


def _photos(fold: int) -> list[tuple[str, int | None]]:
    return [(f"f{fold}_l{h}", h) for h in LATENT] + [(f"f{fold}_d", DEVELOPED), (f"f{fold}_s1", None), (f"f{fold}_s2", None)]


def _dataset(root: Path) -> Path:
    dataset = root / "dataset"
    rows = [("stem", "lot", "mango_id", "cara", "ronda_h", "zona", "provenance")]
    meta = {}
    for fold in FOLDS:
        for split in ("val", "test"):
            labels = dataset / "folds_lot" / f"fold_{fold}" / "labels" / split
            labels.mkdir(parents=True)
            for stem, hour in _photos(fold):
                stem = f"{stem}_{split}"
                rows.append((stem, str(fold), stem, "a", "sano" if hour is None else str(hour), "z" if hour is not None else "", "p" if hour is not None else ""))
                (labels / f"{stem}.txt").write_text("" if hour is None else "1 0.5 0.5 0.5 0.5\n")
        meta[str(fold)] = {"val": f"lot{fold}", "test": f"lot{fold + 2}", "train": ["lot4", "lot5"],
                           "splits": {"train": {"images": 100}, "val": {"images": 20}, "test": {"images": 20}}}
    with (dataset / "index.csv").open("w", newline="") as fh:
        csv.writer(fh).writerows(rows)
    (dataset / "folds_lot" / "folds_meta.json").write_text(json.dumps({"folds": meta}))
    return dataset


def _evaluation(runs: Path, arm: str, fold: int, split: str, scorer, seed: int = 42, checkpoint: str = "reports") -> None:
    """`scorer(hour, stem)` returns the predicted boxes of a photo: [(confidence, xyxy, flat mask or None)]."""
    run = runs / checkpoint / arm / f"fold{fold}_seed{seed}" / split
    run.mkdir(parents=True)
    (run / "metrics_v4.json").write_text("{}")
    lines = []
    for stem, hour in _photos(fold):
        stem = f"{stem}_{split}"
        boxes = []
        for confidence, xyxy, mask in scorer(hour, stem):
            box = {"class": 1, "confidence": confidence, "xyxy": xyxy}
            if mask is not None:
                box["mask"] = mask
            boxes.append(box)
        lines.append(json.dumps({"stem": stem, "boxes": boxes}))
    (run / "predictions.jsonl").write_text("\n".join(lines) + "\n")


def perfect(hour, stem):
    return [] if hour is None else [(0.9, GT, None)]


def right_ranking_wrong_boxes(hour, stem):
    return [(0.1, MISS, None)] if hour is None else [(0.9, MISS, None)]


def half_right(hour, stem):  # right box on the first latent photo only
    if hour is None:
        return []
    return [(0.9, GT if hour == LATENT[0] else MISS, None)]


def segment_perfect(hour, stem):
    return [] if hour is None else [(0.9, GT, SQUARE)]


def segment_wrong_mask(hour, stem):  # the box is right, the mask is not
    return [] if hour is None else [(0.9, GT, MISS_SQUARE)]


@pytest.fixture
def tree(tmp_path: Path):
    dataset = _dataset(tmp_path)
    runs = tmp_path / "runs"
    for fold in FOLDS:
        for split in ("val", "test"):
            _evaluation(runs, "est_rgbnir", fold, split, perfect)
            _evaluation(runs, "est_rgb", fold, split, right_ranking_wrong_boxes if fold == 0 else half_right)
    return runs, dataset, tmp_path / "out"


def rows(path: Path) -> list[dict]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def run(tree, *extra) -> int:
    runs, dataset, out = tree
    return cli.main(["--runs-root", str(runs), "--dataset", str(dataset), "--output", str(out), *extra])


def test_the_default_split_is_val_and_test_is_never_read(tree) -> None:
    assert run(tree) == 0
    evaluations = rows(tree[2] / "top1_runs.csv")
    assert {r["split"] for r in evaluations} == {"val"}
    assert len(evaluations) == 4  # two arms x two folds, seed 42, best checkpoint


def test_each_evaluation_row_holds_the_metrics_of_its_model(tree) -> None:
    assert run(tree) == 0
    by_key = {(r["arm"], r["fold"]): r for r in rows(tree[2] / "top1_runs.csv")}
    perfect_row = by_key[("est_rgbnir", "0")]
    assert (perfect_row["checkpoint"], perfect_row["seed"], perfect_row["split"]) == ("best", "42", "val")
    assert float(perfect_row["corloc_primary_iou50"]) == 1.0 and float(perfect_row["lroc_auc_iou50"]) == 1.0
    assert float(perfect_row["corloc_developed_iou50"]) == 1.0 and float(perfect_row["corloc_all_iou30"]) == 1.0
    wrong = by_key[("est_rgb", "0")]
    assert float(wrong["corloc_primary_iou50"]) == 0.0 and float(wrong["lroc_auc_iou50"]) == 0.0
    assert float(wrong["auroc_top1"]) == 1.0  # the ranking is right even though the boxes are not
    half = by_key[("est_rgb", "1")]
    assert float(half["corloc_primary_iou50"]) == 0.5
    assert float(half["fpf_at_0.25"]) == 0.0 and float(half["tpf_at_0.25"]) == 1.0
    assert float(half["tpf_loc_at_0.25_iou50"]) == 0.5
    assert "mask_corloc_primary_iou50" not in perfect_row  # detect run


def test_the_lroc_curves_come_in_a_long_csv(tree) -> None:
    assert run(tree) == 0
    curves = rows(tree[2] / "top1_lroc_curves.csv")
    mine = [r for r in curves if (r["arm"], r["fold"], r["iou"]) == ("est_rgbnir", "0", "0.5")]
    assert [(float(r["fpf"]), float(r["tpf_loc"])) for r in mine] == [(0.0, 0.0), (0.0, 1.0), (1.0, 1.0)]
    assert [r["threshold"] for r in mine] == ["inf", "0.9", "-inf"]
    assert {"arm", "fold", "seed", "split", "checkpoint", "iou", "threshold", "fpf", "tpf_loc"} <= set(mine[0])
    assert {r["iou"] for r in curves} == {"0.5", "0.3"}


def test_the_paired_differences_follow_the_export_comparisons(tree) -> None:
    assert run(tree) == 0
    differences = rows(tree[2] / "top1_differences.csv")
    h1 = {(r["fold"], r["metric"]): r for r in differences if r["comparison"] == "H1"}
    assert (h1[("0", "corloc_primary_iou50")]["treatment"], h1[("0", "corloc_primary_iou50")]["baseline"]) == ("est_rgbnir", "est_rgb")
    assert float(h1[("0", "corloc_primary_iou50")]["delta"]) == 1.0
    assert float(h1[("1", "corloc_primary_iou50")]["delta"]) == 0.5
    assert float(h1[("0", "lroc_auc_iou50")]["delta"]) == 1.0
    assert {r["comparison"] for r in differences} == {"H1"}  # the other pairs have no arm in the tree


def test_the_lot_level_inference_pools_the_seeds_of_a_fold(tree) -> None:
    assert run(tree) == 0
    inference = rows(tree[2] / "top1_inference.csv")
    row = next(r for r in inference if r["comparison"] == "H1" and r["metric"] == "corloc_primary_iou50")
    assert row["n_folds"] == "2" and float(row["mean_delta"]) == pytest.approx(0.75)
    assert float(row["corrected_t.mean"]) == pytest.approx(0.75)
    assert "bayes.p_right" in row


def test_test_needs_an_explicit_split_and_the_pre_registered_sets(tree, capsys) -> None:
    assert run(tree, "--split", "test") == 2
    assert "--test-folds" in capsys.readouterr().err
    assert run(tree, "--split", "test", "--test-folds", "0") == 2


def test_test_reads_only_the_registered_folds_seeds_and_arms(tree) -> None:
    assert run(tree, "--split", "test", "--test-folds", "0", "--test-seeds", "42", "--test-arms", "est_rgbnir") == 0
    evaluations = rows(tree[2] / "top1_runs.csv")
    assert [(r["arm"], r["fold"], r["split"]) for r in evaluations] == [("est_rgbnir", "0", "test")]
    assert rows(tree[2] / "top1_differences.csv") == []  # one arm only: no pair


def test_the_epoch80_checkpoint_is_discovered(tree) -> None:
    runs, dataset, _ = tree
    _evaluation(runs, "est_rgbnir", 0, "val", perfect, checkpoint="reports_epoch80")
    assert run(tree) == 0
    assert {(r["checkpoint"]) for r in rows(tree[2] / "top1_runs.csv")} == {"best", "epoch80"}


def test_segment_runs_add_the_mask_corloc(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    runs = tmp_path / "runs"
    for fold in FOLDS:
        _evaluation(runs, "master_rgb_seg", fold, "val", segment_wrong_mask)
        _evaluation(runs, "master_rgbnir_seg", fold, "val", segment_perfect)
        for arm in ("master_rgb_seg", "master_rgbnir_seg"):
            for split in ("val",):
                labels = dataset / "folds_lot" / f"fold_{fold}" / "labels" / split
                for stem, hour in _photos(fold):
                    if hour is not None:  # polygon ground truth: class x1 y1 ... xn yn
                        pts = " ".join(map(str, SQUARE))
                        (labels / f"{stem}_{split}.txt").write_text(f"1 {pts}\n")
    out = tmp_path / "out"
    assert cli.main(["--runs-root", str(runs), "--dataset", str(dataset), "--output", str(out)]) == 0
    by_arm = {r["arm"]: r for r in rows(out / "top1_runs.csv") if r["fold"] == "0"}
    assert float(by_arm["master_rgbnir_seg"]["mask_corloc_primary_iou50"]) == 1.0
    assert float(by_arm["master_rgb_seg"]["mask_corloc_primary_iou50"]) == 0.0
    assert float(by_arm["master_rgb_seg"]["corloc_primary_iou50"]) == 1.0  # the box is right, the mask is not
    metrics = {r["metric"] for r in rows(out / "top1_differences.csv") if r["comparison"] == "H1_seg"}
    assert metrics == {"corloc_primary_iou50", "lroc_auc_iou50", "mask_corloc_primary_iou50"}


def test_the_lroc_plot_is_optional(tree) -> None:
    pytest.importorskip("matplotlib")
    assert run(tree) == 0
    assert not (tree[2] / "top1_lroc_curves_val.png").exists()
    assert run(tree, "--plot") == 0
    assert (tree[2] / "top1_lroc_curves_val.png").stat().st_size > 0


def test_the_mean_curve_averages_the_folds_on_a_common_fpf_grid() -> None:
    import numpy as np

    from src.evaluation.top1_localization import LrocCurve

    def curve(fpf, tpf):
        return LrocCurve(np.array([np.inf] + [0.0] * (len(fpf) - 1)), np.array(fpf, float), np.array(tpf, float))

    grid = np.array([0.0, 0.5, 1.0])
    mean = cli.mean_curve([curve([0, 0, 1], [0, 1, 1]), curve([0, 1], [0, 0])], grid)
    assert list(mean) == [0.5, 0.5, 0.5]  # (1 + 0) / 2 at FPF 0 after the vertical rise, and so on
