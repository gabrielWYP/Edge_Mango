"""Tests for scripts/aggregate_kfold.py: per-fold tables, mean +/- sigma, paired difference."""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import pytest

from scripts import aggregate_kfold as agg

BASE = "twostream"
TREAT = "twostream_roi_crop"


def _metrics(ap50: float, *, ap50_95: float = 0.1, precision: float = 0.2, recall: float = 0.5,
             tp: int = 10, fn: int = 5, fp: int = 20,
             ap30: float | None = None, ap_center: float | None = None,
             center_recall: float | None = None) -> dict:
    """The JSON `scripts/evaluate_checkpoint.py` writes (class ids are string keys).

    Without `ap30` / `ap_center` / `center_recall` the file is in the OLD format, i.e. it
    predates the localization-tolerant metrics and carries none of their keys.
    """
    metrics = _old_metrics(ap50, ap50_95=ap50_95, precision=precision, recall=recall,
                           tp=tp, fn=fn, fp=fp)
    if ap30 is not None:
        metrics["per_class_ap_30"] = {"0": 1.0, "1": ap30}
    if ap_center is not None:
        metrics["per_class_ap_center"] = {"0": 1.0, "1": ap_center}
    if center_recall is not None:
        metrics["per_class_center_recall"] = {"0": 1.0, "1": center_recall}
    return metrics


def _old_metrics(ap50: float, *, ap50_95: float, precision: float, recall: float,
                 tp: int, fn: int, fp: int) -> dict:
    return {
        "map50": (1.0 + ap50) / 2,
        "map_50_95": 0.3,
        "per_class_ap_50": {"0": 1.0, "1": ap50},
        "per_class_ap_50_95": {"0": 0.6, "1": ap50_95},
        "per_class_precision": {"0": 1.0, "1": precision},
        "per_class_recall": {"0": 1.0, "1": recall},
        "per_class_f1": {"0": 1.0, "1": 0.3},
        "per_class_counts": {"0": {"tp": 25, "fp": 0, "fn": 0}, "1": {"tp": tp, "fp": fp, "fn": fn}},
        "roi_crop": False,
        "roi_crop_margin": 0.1,
    }


def _write(root: Path, arm: str, fold: int, seed: int, split: str, metrics: dict | str) -> Path:
    path = root / arm / f"fold{fold}_seed{seed}" / split / "metrics.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(metrics if isinstance(metrics, str) else json.dumps(metrics))
    return path


def _aggregate(root: Path, folds=(0, 1, 2), seeds=(42,)) -> dict:
    return agg.aggregate(root, arms=[BASE, TREAT], folds=list(folds), seeds=list(seeds),
                         baseline=BASE, treatment=TREAT)


# --------------------------------------------------------------------------
# Building blocks
# --------------------------------------------------------------------------


def test_damage_summary_reads_class_one_and_counts_gt_boxes() -> None:
    summary = agg.damage_summary(_metrics(0.398344, ap50_95=0.139928, precision=0.165414,
                                          recall=0.594595, tp=22, fn=15, fp=111))
    assert summary["ap50"] == pytest.approx(0.398344)
    assert summary["ap50_95"] == pytest.approx(0.139928)
    assert summary["precision"] == pytest.approx(0.165414)
    assert summary["recall"] == pytest.approx(0.594595)
    assert summary["num_gt"] == 37          # tp + fn at the operating point == GT boxes


def test_damage_summary_accepts_integer_class_keys() -> None:
    raw = _metrics(0.5)
    ints = {k: ({int(c): v for c, v in val.items()} if isinstance(val, dict) else val) for k, val in raw.items()}
    assert agg.damage_summary(ints) == agg.damage_summary(raw)


def test_damage_summary_rejects_metrics_without_the_damage_class() -> None:
    raw = _metrics(0.5)
    raw["per_class_ap_50"] = {"0": 1.0}
    with pytest.raises(KeyError):
        agg.damage_summary(raw)


def test_mean_std_uses_the_sample_standard_deviation() -> None:
    result = agg.mean_std([0.2, 0.4, 0.6])
    assert result == {"mean": pytest.approx(0.4), "std": pytest.approx(statistics.stdev([0.2, 0.4, 0.6])), "n": 3}
    assert result["std"] == pytest.approx(0.2)
    assert agg.mean_std([0.3]) == {"mean": pytest.approx(0.3), "std": None, "n": 1}
    assert agg.mean_std([]) == {"mean": None, "std": None, "n": 0}


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------


def _full_tree(root: Path) -> None:
    base_ap = {0: 0.20, 1: 0.30, 2: 0.40}
    treat_ap = {0: 0.50, 1: 0.45, 2: 0.70}
    for fold in (0, 1, 2):
        for split, shift in (("val", 0.1), ("test", 0.0)):
            _write(root, BASE, fold, 42, split, _metrics(base_ap[fold] + shift, tp=10 + fold, fn=5))
            _write(root, TREAT, fold, 42, split, _metrics(treat_ap[fold] + shift, tp=10 + fold, fn=5))


def test_per_fold_values_mean_std_and_paired_difference(tmp_path: Path) -> None:
    _full_tree(tmp_path)
    results = _aggregate(tmp_path)

    test_base = results["per_fold"][BASE]["test"]
    assert {f: v["ap50"] for f, v in test_base.items()} == {"0": pytest.approx(0.20), "1": pytest.approx(0.30),
                                                           "2": pytest.approx(0.40)}
    assert test_base["2"]["num_gt"] == 12 + 5

    summary = results["summary"][BASE]["test"]["ap50"]
    assert summary["mean"] == pytest.approx(0.30)
    assert summary["std"] == pytest.approx(statistics.stdev([0.2, 0.3, 0.4]))
    assert summary["n"] == 3
    assert results["summary"][TREAT]["val"]["ap50"]["mean"] == pytest.approx(statistics.fmean([0.6, 0.55, 0.8]))

    deltas = {f: v["ap50"] for f, v in results["paired"]["per_fold"]["test"].items()}
    assert deltas == {"0": pytest.approx(0.30), "1": pytest.approx(0.15), "2": pytest.approx(0.30)}
    paired = results["paired"]["summary"]["test"]["ap50"]
    assert paired["mean"] == pytest.approx(statistics.fmean([0.30, 0.15, 0.30]))
    assert paired["std"] == pytest.approx(statistics.stdev([0.30, 0.15, 0.30]))
    assert paired["n"] == 3
    assert results["paired"]["baseline"] == BASE and results["paired"]["treatment"] == TREAT
    assert results["missing"] == []


def test_missing_runs_are_reported_and_excluded_from_the_statistics(tmp_path: Path) -> None:
    _full_tree(tmp_path)
    (tmp_path / TREAT / "fold1_seed42" / "test" / "metrics.json").unlink()      # one eval missing
    for split in ("val", "test"):                                               # a whole run missing
        (tmp_path / BASE / "fold2_seed42" / split / "metrics.json").unlink()

    results = _aggregate(tmp_path)

    missing = {(m["arm"], m["fold"], m["seed"], m["split"]) for m in results["missing"]}
    assert missing == {
        (TREAT, 1, 42, "test"),
        (BASE, 2, 42, "val"),
        (BASE, 2, 42, "test"),
    }
    assert all(m["status"] == "missing" for m in results["missing"])
    assert set(results["per_fold"][BASE]["test"]) == {"0", "1"}
    assert results["summary"][BASE]["test"]["ap50"]["n"] == 2
    # Pairs exist only where both arms have the fold: test -> {0}, val -> {0, 1}.
    assert set(results["paired"]["per_fold"]["test"]) == {"0"}
    assert set(results["paired"]["per_fold"]["val"]) == {"0", "1"}
    assert results["paired"]["summary"]["test"]["ap50"] == {"mean": pytest.approx(0.30), "std": None, "n": 1}


def test_unreadable_metrics_are_reported_not_dropped(tmp_path: Path) -> None:
    _full_tree(tmp_path)
    _write(tmp_path, BASE, 0, 42, "test", "{not json")
    _write(tmp_path, TREAT, 0, 42, "val", json.dumps({"map50": 0.5}))           # no damage class
    results = _aggregate(tmp_path)
    by_key = {(m["arm"], m["fold"], m["split"]): m for m in results["missing"]}
    assert by_key[(BASE, 0, "test")]["status"] == "invalid"
    assert by_key[(TREAT, 0, "val")]["status"] == "invalid"
    assert "reason" in by_key[(BASE, 0, "test")]
    assert "0" not in results["per_fold"][BASE]["test"]


def test_multiple_seeds_average_within_a_fold_and_pair_on_common_seeds(tmp_path: Path) -> None:
    for seed, base_ap, treat_ap in ((42, 0.2, 0.4), (7, 0.4, 0.5)):
        _write(tmp_path, BASE, 0, seed, "test", _metrics(base_ap))
        _write(tmp_path, TREAT, 0, seed, "test", _metrics(treat_ap))
    _write(tmp_path, TREAT, 1, 42, "test", _metrics(0.9))                       # treat has seed 42 only...
    _write(tmp_path, BASE, 1, 7, "test", _metrics(0.1))                         # ...base has seed 7 only
    results = _aggregate(tmp_path, folds=(0, 1), seeds=(42, 7))

    fold0 = results["per_fold"][BASE]["test"]["0"]
    assert fold0["ap50"] == pytest.approx(0.3) and fold0["seeds"] == [7, 42]
    # fold 0 pairs seed 42 (+0.2) and seed 7 (+0.1) -> +0.15; fold 1 has no common seed.
    assert results["paired"]["per_fold"]["test"]["0"]["ap50"] == pytest.approx(0.15)
    assert "1" not in results["paired"]["per_fold"]["test"]


def test_aggregate_with_no_results_has_empty_statistics(tmp_path: Path) -> None:
    results = _aggregate(tmp_path)
    assert len(results["missing"]) == 2 * 3 * 2
    assert results["summary"][BASE]["test"]["ap50"] == {"mean": None, "std": None, "n": 0}
    assert results["paired"]["summary"]["test"]["ap50"]["n"] == 0


# --------------------------------------------------------------------------
# Rendering and CLI
# --------------------------------------------------------------------------


def test_markdown_has_tables_mean_sigma_paired_diff_and_missing_marks(tmp_path: Path) -> None:
    _full_tree(tmp_path)
    (tmp_path / TREAT / "fold1_seed42" / "test" / "metrics.json").unlink()
    text = agg.render_markdown(_aggregate(tmp_path))

    assert f"### {BASE}" in text and f"### {TREAT}" in text
    assert "mean ± σ" in text
    assert "0.2000" in text and "0.3000 ± 0.1000" in text         # per-fold value; mean ± sigma (test, base)
    assert f"Paired difference {TREAT} − {BASE}" in text
    assert "+0.3000" in text                                      # fold 0 delta
    assert "missing" in text.lower()
    assert "fold 1" in text and "test" in text                     # the missing-run list names the run


def test_main_writes_json_and_markdown(tmp_path: Path, capsys) -> None:
    _full_tree(tmp_path)
    assert agg.main(["--reports-root", str(tmp_path), "--folds", "0,1,2"]) == 0
    data = json.loads((tmp_path / "kfold_results.json").read_text())
    assert data["summary"][BASE]["test"]["ap50"]["mean"] == pytest.approx(0.30)
    assert (tmp_path / "kfold_results.md").read_text().startswith("# ")
    # Deterministic: a second run gives the same bytes.
    first = (tmp_path / "kfold_results.json").read_bytes(), (tmp_path / "kfold_results.md").read_bytes()
    assert agg.main(["--reports-root", str(tmp_path), "--folds", "0,1,2"]) == 0
    assert ((tmp_path / "kfold_results.json").read_bytes(), (tmp_path / "kfold_results.md").read_bytes()) == first


def test_main_with_default_folds_reports_the_unrun_folds_as_missing(tmp_path: Path) -> None:
    _write(tmp_path, TREAT, 0, 42, "val", _metrics(0.5))
    assert agg.main(["--reports-root", str(tmp_path)]) == 0
    data = json.loads((tmp_path / "kfold_results.json").read_text())
    assert data["folds"] == [0, 1, 2, 3, 4]
    assert len(data["missing"]) == 2 * 5 * 2 - 1


def test_main_fails_when_no_result_exists_and_writes_nothing(tmp_path: Path, capsys) -> None:
    assert agg.main(["--reports-root", str(tmp_path)]) == 1
    assert "No evaluation results" in capsys.readouterr().err
    assert not (tmp_path / "kfold_results.json").exists()


# --------------------------------------------------------------------------
# Localization-tolerant metrics: AP@0.3 and AP_center next to AP50
# --------------------------------------------------------------------------


def test_damage_summary_reads_ap30_and_ap_center() -> None:
    summary = agg.damage_summary(_metrics(0.1, ap30=0.45, ap_center=0.6))
    assert summary["ap30"] == pytest.approx(0.45)
    assert summary["ap_center"] == pytest.approx(0.6)
    assert summary["ap50"] == pytest.approx(0.1)


def test_damage_summary_of_an_old_metrics_file_reports_the_new_metrics_as_missing() -> None:
    summary = agg.damage_summary(_metrics(0.1))                 # no per_class_ap_30 / _center keys
    assert summary["ap30"] is None and summary["ap_center"] is None
    assert summary["ap50"] == pytest.approx(0.1)                # everything else still reads


def test_new_metrics_are_listed_next_to_ap50() -> None:
    assert list(agg.METRICS[:3]) == ["ap50", "ap30", "ap_center"]
    assert agg.METRIC_LABELS["ap30"] == "AP@0.3" and agg.METRIC_LABELS["ap_center"] == "AP_center"


def _localization_tree(root: Path) -> None:
    base = {0: (0.20, 0.50, 0.60), 1: (0.30, 0.55, 0.65), 2: (0.40, 0.60, 0.80)}
    treat = {0: (0.50, 0.70, 0.75), 1: (0.45, 0.60, 0.70), 2: (0.70, 0.90, 0.95)}
    for fold in (0, 1, 2):
        for split in ("val", "test"):
            for arm, table in ((BASE, base), (TREAT, treat)):
                ap50, ap30, apc = table[fold]
                _write(root, arm, fold, 42, split, _metrics(ap50, ap30=ap30, ap_center=apc))


def test_per_fold_mean_std_and_paired_difference_of_the_new_metrics(tmp_path: Path) -> None:
    _localization_tree(tmp_path)
    results = _aggregate(tmp_path)

    base_test = results["per_fold"][BASE]["test"]
    assert {f: v["ap30"] for f, v in base_test.items()} == {
        "0": pytest.approx(0.50), "1": pytest.approx(0.55), "2": pytest.approx(0.60)}
    assert {f: v["ap_center"] for f, v in base_test.items()} == {
        "0": pytest.approx(0.60), "1": pytest.approx(0.65), "2": pytest.approx(0.80)}

    stat = results["summary"][BASE]["test"]["ap30"]
    assert stat["mean"] == pytest.approx(0.55)
    assert stat["std"] == pytest.approx(statistics.stdev([0.50, 0.55, 0.60]))
    assert stat["n"] == 3

    deltas = [0.70 - 0.50, 0.60 - 0.55, 0.90 - 0.60]
    paired = results["paired"]["summary"]["test"]["ap30"]
    assert paired["mean"] == pytest.approx(statistics.fmean(deltas))
    assert paired["std"] == pytest.approx(statistics.stdev(deltas))
    center_deltas = [0.75 - 0.60, 0.70 - 0.65, 0.95 - 0.80]
    assert results["paired"]["summary"]["test"]["ap_center"]["mean"] == pytest.approx(
        statistics.fmean(center_deltas))
    assert results["missing_metrics"] == []


def test_old_format_files_do_not_crash_and_leave_the_old_columns_untouched(tmp_path: Path) -> None:
    _full_tree(tmp_path)                                        # old format: no new keys at all
    results = _aggregate(tmp_path)

    assert results["missing"] == []                             # the runs themselves are fine
    assert results["summary"][BASE]["test"]["ap50"]["mean"] == pytest.approx(0.30)
    for arm in (BASE, TREAT):
        assert results["per_fold"][arm]["test"]["0"]["ap30"] is None
        assert results["summary"][arm]["test"]["ap30"] == {"mean": None, "std": None, "n": 0}
        assert results["summary"][arm]["test"]["ap_center"] == {"mean": None, "std": None, "n": 0}
    assert results["paired"]["summary"]["test"]["ap30"]["n"] == 0
    assert results["paired"]["per_fold"]["test"]["0"]["ap30"] is None
    assert results["paired"]["summary"]["test"]["ap50"]["mean"] == pytest.approx(
        statistics.fmean([0.30, 0.15, 0.30]))
    # Each old-format evaluation is named, so the user knows what to re-run.
    assert len(results["missing_metrics"]) == 2 * 3 * 2
    assert {tuple(m["metrics"]) for m in results["missing_metrics"]} == {("ap30", "ap_center")}


def test_a_mix_of_old_and_new_files_uses_only_the_runs_that_have_the_metric(tmp_path: Path) -> None:
    _localization_tree(tmp_path)
    _write(tmp_path, BASE, 1, 42, "test", _metrics(0.30))       # baseline fold 1 re-written in the old format
    results = _aggregate(tmp_path)

    assert results["per_fold"][BASE]["test"]["1"]["ap30"] is None
    assert results["summary"][BASE]["test"]["ap30"]["n"] == 2   # folds 0 and 2 only
    assert results["summary"][BASE]["test"]["ap50"]["n"] == 3
    # Fold 1 cannot be paired on ap30, but is still paired on ap50.
    assert results["paired"]["per_fold"]["test"]["1"]["ap30"] is None
    assert results["paired"]["per_fold"]["test"]["1"]["ap50"] == pytest.approx(0.15)
    assert results["paired"]["summary"]["test"]["ap30"]["n"] == 2
    assert [m for m in results["missing_metrics"]] == [
        {"arm": BASE, "fold": 1, "seed": 42, "split": "test", "metrics": ["ap30", "ap_center"]}]


def test_markdown_has_the_new_columns_and_marks_missing_ones(tmp_path: Path) -> None:
    _localization_tree(tmp_path)
    _write(tmp_path, BASE, 1, 42, "test", _metrics(0.30))
    text = agg.render_markdown(_aggregate(tmp_path))

    assert "AP@0.3" in text and "AP_center" in text
    assert "ΔAP@0.3" in text and "ΔAP_center" in text
    header = next(line for line in text.splitlines() if line.startswith("| fold |") and "AP50-95" in line)
    assert header.index("AP50 ") < header.index("AP@0.3") < header.index("AP_center") < header.index("AP50-95")
    assert "| 1 | 42 | 0.3000 | — | — |" in text                # the old-format fold: "—" cells
    assert "AP@0.3, AP_center" in text and "re-run" in text.lower()


def test_markdown_of_a_complete_tree_has_no_missing_metrics_section(tmp_path: Path) -> None:
    _localization_tree(tmp_path)
    assert "Missing metrics" not in agg.render_markdown(_aggregate(tmp_path))


# --------------------------------------------------------------------------
# The paired section needs two different arms
# --------------------------------------------------------------------------


def test_a_single_arm_skips_the_paired_difference_instead_of_comparing_it_with_itself(
        tmp_path: Path) -> None:
    _localization_tree(tmp_path)
    results = agg.aggregate(tmp_path, arms=[TREAT], folds=[0, 1, 2], seeds=[42],
                            baseline=TREAT, treatment=TREAT)

    paired = results["paired"]
    assert paired["skipped"] and TREAT in paired["skipped"]
    assert paired["per_fold"] == {} and paired["summary"] == {}
    assert results["summary"][TREAT]["test"]["ap50"]["n"] == 3       # the per-arm tables are intact

    text = agg.render_markdown(results)
    assert "skipped" in text.lower()
    assert "0.0000 ± 0.0000" not in text                             # the old meaningless 0 ± 0
    assert "Paired difference" in text                                # the note says WHY it is skipped
    assert "+0.0000" not in text


def test_identical_baseline_and_treatment_skip_the_paired_difference(tmp_path: Path) -> None:
    _localization_tree(tmp_path)
    results = agg.aggregate(tmp_path, arms=[BASE, TREAT], folds=[0, 1, 2], seeds=[42],
                            baseline=BASE, treatment=BASE)
    assert results["paired"]["skipped"]
    assert results["paired"]["per_fold"] == {}
    assert set(results["summary"]) == {BASE, TREAT}


def test_two_different_arms_are_still_paired(tmp_path: Path) -> None:
    _localization_tree(tmp_path)
    results = _aggregate(tmp_path)
    assert results["paired"]["skipped"] is None
    assert "Skipped" not in agg.render_markdown(results)


def test_main_with_one_arm_succeeds_and_writes_the_note(tmp_path: Path) -> None:
    _localization_tree(tmp_path)
    assert agg.main(["--reports-root", str(tmp_path), "--arms", TREAT, "--folds", "0,1,2"]) == 0
    data = json.loads((tmp_path / "kfold_results.json").read_text())
    assert data["paired"]["skipped"]
    text = (tmp_path / "kfold_results.md").read_text()
    assert "0.0000 ± 0.0000" not in text and "skipped" in text.lower()
