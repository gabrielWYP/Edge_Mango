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
             tp: int = 10, fn: int = 5, fp: int = 20) -> dict:
    """The JSON `scripts/evaluate_checkpoint.py` writes (class ids are string keys)."""
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
