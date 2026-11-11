"""Tests for scripts/aggregate_kfold.py: per-fold tables, mean +/- sigma, paired difference, fold-level inference."""

from __future__ import annotations

import dataclasses
import json
import statistics
from pathlib import Path

import pytest

from scripts import aggregate_kfold as agg
from src.evaluation import fold_inference as fi

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


# --------------------------------------------------------------------------
# Fold-level inference: fixtures
# --------------------------------------------------------------------------

FIVE = (0, 1, 2, 3, 4)
SEEDS = (42, 1337)
# (num_test, num_train) per fold of the tmp fold spec; pooled ratio = 150 / 500 = 0.3 (the repo spec gives 1/3).
SIZES = {0: (10, 100), 1: (20, 100), 2: (30, 100), 3: (40, 100), 4: (50, 100)}
DELTAS = {0: 0.10, 1: 0.04, 2: 0.12, 3: -0.02, 4: 0.08}      # treatment - baseline test AP50 per fold
SEED_NOISE = {42: 0.01, 1337: -0.01}                          # cancels in the seed average


def _spec_rows(sizes: dict[int, tuple[int, int]] = SIZES) -> list[dict]:
    return [
        {"fold": k, "num_test": te, "num_train": tr, "num_val": 7,
         "test": f"lot{k}", "val": f"lot{(k + 1) % 5}", "train": []}
        for k, (te, tr) in sizes.items()
    ]


def _write_spec(tmp_path: Path, sizes: dict[int, tuple[int, int]] = SIZES, name: str = "folds.json") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps({"schema_version": 1, "folds": _spec_rows(sizes)}))
    return path


def _fam(ap50: float) -> dict:
    """Metrics that move with AP50, so every metric has a non-constant paired difference."""
    return _metrics(ap50, ap50_95=ap50 / 2, precision=0.2 + ap50 / 10, recall=0.5 + ap50 / 10)


def _synthetic_tree(root: Path, deltas: dict[int, float] = DELTAS, folds=FIVE, seeds=SEEDS) -> None:
    """5 folds x 2 seeds x 2 arms x 2 splits with known paired differences."""
    for fold in folds:
        for seed in seeds:
            for split, shift in (("val", 0.1), ("test", 0.0)):
                base = 0.30 + 0.02 * fold + shift
                _write(root, BASE, fold, seed, split, _fam(base))
                _write(root, TREAT, fold, seed, split, _fam(base + deltas[fold] + SEED_NOISE[seed]))


def _inferred(root: Path, spec: list[dict] | None = None, folds=FIVE, seeds=SEEDS, **kwargs) -> dict:
    return agg.aggregate(root, arms=[BASE, TREAT], folds=list(folds), seeds=list(seeds),
                         baseline=BASE, treatment=TREAT,
                         fold_spec=_spec_rows() if spec is None else spec, **kwargs)


def _fold_deltas(results: dict, split: str = "test", metric: str = "ap50", folds=FIVE) -> list[float]:
    rows = results["paired"]["per_fold"][split]
    return [rows[str(f)][metric] for f in folds if str(f) in rows]


# --------------------------------------------------------------------------
# Per-seed paired values and backward compatibility
# --------------------------------------------------------------------------


def test_aggregate_without_a_fold_spec_adds_no_inference_and_keeps_the_old_keys(tmp_path: Path) -> None:
    _full_tree(tmp_path)
    results = _aggregate(tmp_path)
    assert "inference" not in results["paired"]
    assert set(results["paired"]) >= {"baseline", "treatment", "per_fold", "summary", "per_fold_seed"}
    assert set(results) >= {"arms", "damage_class_id", "folds", "metrics", "missing", "paired",
                            "per_fold", "reports_root", "seeds", "splits", "summary"}


def test_paired_per_fold_seed_keeps_every_seed_delta(tmp_path: Path) -> None:
    _synthetic_tree(tmp_path)
    results = _inferred(tmp_path)
    per_seed = results["paired"]["per_fold_seed"]["test"]["ap50"]
    assert set(per_seed) == {str(f) for f in FIVE}
    for fold in FIVE:
        assert set(per_seed[str(fold)]) == {"42", "1337"}
        assert per_seed[str(fold)]["42"] == pytest.approx(DELTAS[fold] + 0.01)
        assert per_seed[str(fold)]["1337"] == pytest.approx(DELTAS[fold] - 0.01)
        # the fold value is the mean of its seed values
        assert results["paired"]["per_fold"]["test"][str(fold)]["ap50"] == pytest.approx(
            statistics.fmean(per_seed[str(fold)].values()))
    assert set(results["paired"]["per_fold_seed"]["val"]) == {"ap50", "ap50_95", "precision", "recall"}


def test_per_fold_seed_only_holds_seeds_both_arms_have(tmp_path: Path) -> None:
    _synthetic_tree(tmp_path)
    (tmp_path / TREAT / "fold2_seed1337" / "test" / "metrics.json").unlink()
    per_seed = _inferred(tmp_path)["paired"]["per_fold_seed"]["test"]["ap50"]
    assert set(per_seed["2"]) == {"42"}
    assert set(per_seed["0"]) == {"42", "1337"}


# --------------------------------------------------------------------------
# Fold-level inference block
# --------------------------------------------------------------------------


def test_inference_block_equals_the_pure_functions_on_the_fold_deltas(tmp_path: Path) -> None:
    _synthetic_tree(tmp_path)
    results = _inferred(tmp_path)
    deltas = _fold_deltas(results)
    assert deltas == pytest.approx([DELTAS[f] for f in FIVE])
    ratio = 150 / 500

    entry = results["paired"]["inference"]["test"]["ap50"]
    assert entry["n_folds"] == 5
    assert entry["n_positive"] == 4                              # fold 3 is negative
    assert entry["test_train_ratio"] == pytest.approx(ratio)     # from the tmp spec, not the repo's 1/3
    assert entry["corrected_t"] == dataclasses.asdict(fi.corrected_resampled_t(deltas, entry["test_train_ratio"]))
    assert entry["bayes"] == dataclasses.asdict(
        fi.correlated_bayesian_t(deltas, entry["test_train_ratio"], rope=0.0))
    assert entry["sign_flip"] == dataclasses.asdict(fi.sign_flip_test(deltas))
    assert entry["corrected_t"]["df"] == 4
    assert entry["sign_flip"]["exact"] is True and entry["sign_flip"]["n_nonzero"] == 5
    assert entry["sign_flip"]["min_attainable_p_two_sided"] == pytest.approx(0.0625)
    assert entry["bayes"]["p_rope"] == 0.0


def test_inference_covers_every_split_and_metric(tmp_path: Path) -> None:
    _synthetic_tree(tmp_path)
    inference = _inferred(tmp_path)["paired"]["inference"]
    assert set(inference) == {"val", "test"}
    for split in inference:
        assert set(inference[split]) == {"ap50", "ap50_95", "precision", "recall"}
        assert all("error" not in entry for entry in inference[split].values())


def test_inference_results_are_json_serialisable(tmp_path: Path) -> None:
    _synthetic_tree(tmp_path)
    text = json.dumps(_inferred(tmp_path), indent=2, sort_keys=True)
    assert json.loads(text)["paired"]["inference"]["test"]["ap50"]["n_folds"] == 5


def test_rope_and_confidence_are_forwarded(tmp_path: Path) -> None:
    _synthetic_tree(tmp_path)
    plain = _inferred(tmp_path)["paired"]["inference"]["test"]["ap50"]
    tuned = _inferred(tmp_path, rope=0.03, confidence=0.9)["paired"]["inference"]["test"]["ap50"]
    assert plain["bayes"]["rope"] == 0.0 and plain["corrected_t"]["confidence"] == 0.95
    assert tuned["bayes"]["rope"] == 0.03 and tuned["bayes"]["p_rope"] > 0.0
    assert tuned["corrected_t"]["confidence"] == 0.9 and tuned["bayes"]["credibility"] == 0.9
    assert tuned["corrected_t"]["t"] == plain["corrected_t"]["t"]


def test_ratio_is_pooled_over_the_folds_that_have_a_pair(tmp_path: Path) -> None:
    _synthetic_tree(tmp_path)
    for seed in SEEDS:                                           # fold 3 has no baseline test run
        (tmp_path / BASE / f"fold3_seed{seed}" / "test" / "metrics.json").unlink()
    entry = _inferred(tmp_path)["paired"]["inference"]["test"]["ap50"]
    assert entry["n_folds"] == 4
    assert entry["test_train_ratio"] == pytest.approx((10 + 20 + 30 + 50) / 400)
    assert entry["n_positive"] == 4


def test_zero_variance_deltas_record_an_error_instead_of_crashing(tmp_path: Path) -> None:
    for fold in FIVE:                                            # dyadic values: the deltas are exactly equal
        for seed in SEEDS:
            for split in ("val", "test"):
                base = 0.25 + 0.0625 * fold
                _write(tmp_path, BASE, fold, seed, split, _metrics(base))
                _write(tmp_path, TREAT, fold, seed, split, _metrics(base + 0.125))
    inference = _inferred(tmp_path)["paired"]["inference"]
    entry = inference["test"]["ap50"]
    assert set(entry) == {"error"} and "zero variance" in entry["error"]
    assert "error" in inference["test"]["precision"]             # constant 0 difference as well
    assert inference["val"]["ap50"] == entry


def test_fewer_than_two_paired_folds_record_an_error(tmp_path: Path) -> None:
    _synthetic_tree(tmp_path, folds=(0,))
    one = _inferred(tmp_path)["paired"]["inference"]["test"]["ap50"]
    assert set(one) == {"error"} and "at least 2" in one["error"]
    none = _inferred(tmp_path / "empty")["paired"]["inference"]["test"]["ap50"]
    assert set(none) == {"error"}


def test_unknown_fold_id_is_a_clear_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"fold.*7.*fold spec"):
        _inferred(tmp_path, folds=(0, 1, 7))


# --------------------------------------------------------------------------
# Fold spec loading
# --------------------------------------------------------------------------


def test_load_fold_spec_reads_the_repo_spec_and_a_tmp_spec(tmp_path: Path) -> None:
    rows = agg.load_fold_spec(agg.PROJECT_ROOT / agg.DEFAULT_FOLD_SPEC)
    assert [r["fold"] for r in rows] == [0, 1, 2, 3, 4]
    assert fi.test_train_ratio(rows) == pytest.approx(236 / 708)
    assert agg.load_fold_spec(_write_spec(tmp_path))[2]["num_test"] == 30


@pytest.mark.parametrize(
    "content",
    [
        "{not json",
        json.dumps({"no_folds": []}),
        json.dumps({"folds": []}),
        json.dumps({"folds": [{"fold": 0, "num_test": 5}]}),                                    # no num_train
        json.dumps({"folds": [{"fold": 0, "num_test": 5, "num_train": 9},
                              {"fold": 0, "num_test": 6, "num_train": 9}]}),                    # duplicate id
    ],
)
def test_load_fold_spec_rejects_malformed_files(tmp_path: Path, content: str) -> None:
    path = tmp_path / "bad.json"
    path.write_text(content)
    with pytest.raises(ValueError):
        agg.load_fold_spec(path)


def test_load_fold_spec_rejects_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="nope.json"):
        agg.load_fold_spec(tmp_path / "nope.json")


# --------------------------------------------------------------------------
# Markdown and CLI
# --------------------------------------------------------------------------


def test_markdown_has_a_paired_inference_section(tmp_path: Path) -> None:
    _synthetic_tree(tmp_path)
    text = agg.render_markdown(_inferred(tmp_path, rope=0.02))
    assert "## Paired inference" in text
    section = text.split("## Paired inference")[1].split("## Missing runs")[0]
    assert "95% CI" in section and "P(left)" in section and "P(rope)" in section and "P(right)" in section
    assert "4/5" in section                                       # folds with a positive delta
    assert "0.0625" in section                                    # smallest attainable sign-flip p
    assert "0.3000" in section                                    # test/train ratio
    assert "0.02" in section                                      # the ROPE
    assert "| test |" in section and "| val |" in section
    assert text.index("## Paired inference") < text.index("## Missing runs")


def test_markdown_shows_the_error_instead_of_numbers(tmp_path: Path) -> None:
    for fold in FIVE:
        for seed in SEEDS:
            for split in ("val", "test"):
                base = 0.25 + 0.0625 * fold
                _write(tmp_path, BASE, fold, seed, split, _metrics(base))
                _write(tmp_path, TREAT, fold, seed, split, _metrics(base + 0.125))
    section = agg.render_markdown(_inferred(tmp_path)).split("## Paired inference")[1]
    assert "zero variance" in section


def test_markdown_without_a_fold_spec_has_no_inference_section(tmp_path: Path) -> None:
    _full_tree(tmp_path)
    assert "Paired inference" not in agg.render_markdown(_aggregate(tmp_path))


def test_main_writes_inference_with_an_explicit_fold_spec(tmp_path: Path) -> None:
    reports = tmp_path / "reports"
    _synthetic_tree(reports)
    spec = _write_spec(tmp_path)
    argv = ["--reports-root", str(reports), "--seeds", "42", "1337", "--fold-spec", str(spec),
            "--rope", "0.02", "--confidence", "0.9"]
    assert agg.main(argv) == 0
    data = json.loads((reports / "kfold_results.json").read_text())
    entry = data["paired"]["inference"]["test"]["ap50"]
    assert entry["bayes"]["rope"] == 0.02 and entry["corrected_t"]["confidence"] == 0.9
    assert entry["test_train_ratio"] == pytest.approx(0.3)
    assert data["fold_spec"] == str(spec)
    assert "## Paired inference" in (reports / "kfold_results.md").read_text()
    # still deterministic
    first = (reports / "kfold_results.json").read_bytes(), (reports / "kfold_results.md").read_bytes()
    assert agg.main(argv) == 0
    assert ((reports / "kfold_results.json").read_bytes(), (reports / "kfold_results.md").read_bytes()) == first


def test_main_defaults_to_the_repo_fold_spec(tmp_path: Path) -> None:
    _synthetic_tree(tmp_path)
    assert agg.main(["--reports-root", str(tmp_path), "--seeds", "42", "1337"]) == 0
    entry = json.loads((tmp_path / "kfold_results.json").read_text())["paired"]["inference"]["test"]["ap50"]
    assert entry["test_train_ratio"] == pytest.approx(236 / 708)


def test_main_does_not_crash_on_zero_variance_deltas(tmp_path: Path) -> None:
    for fold in FIVE:
        for split in ("val", "test"):
            base = 0.25 + 0.0625 * fold
            _write(tmp_path, BASE, fold, 42, split, _metrics(base))
            _write(tmp_path, TREAT, fold, 42, split, _metrics(base + 0.125))
    assert agg.main(["--reports-root", str(tmp_path)]) == 0
    data = json.loads((tmp_path / "kfold_results.json").read_text())
    assert "error" in data["paired"]["inference"]["test"]["ap50"]


def test_main_rejects_a_fold_missing_from_the_spec_and_writes_nothing(tmp_path: Path, capsys) -> None:
    _synthetic_tree(tmp_path)
    spec = _write_spec(tmp_path, sizes={k: SIZES[k] for k in (0, 1, 2)})
    assert agg.main(["--reports-root", str(tmp_path), "--fold-spec", str(spec)]) == 2
    err = capsys.readouterr().err
    assert "ERROR" in err and "3" in err and "4" in err and "fold spec" in err
    assert not (tmp_path / "kfold_results.json").exists()


def test_main_rejects_a_missing_fold_spec(tmp_path: Path, capsys) -> None:
    _synthetic_tree(tmp_path)
    assert agg.main(["--reports-root", str(tmp_path), "--fold-spec", str(tmp_path / "nope.json")]) == 2
    assert "nope.json" in capsys.readouterr().err
    assert not (tmp_path / "kfold_results.json").exists()


@pytest.mark.parametrize("flags", [["--rope", "-0.01"], ["--confidence", "1.5"], ["--confidence", "0"]])
def test_main_rejects_invalid_rope_and_confidence(tmp_path: Path, capsys, flags: list[str]) -> None:
    _synthetic_tree(tmp_path)
    assert agg.main(["--reports-root", str(tmp_path), *flags]) == 2
    assert "ERROR" in capsys.readouterr().err


# --------------------------------------------------------------------------
# metrics_v4.json: latent AP50, latent AP30 and per-face AUROC
# --------------------------------------------------------------------------


def _v4(latent_ap50: float | None, latent_ap30: float | None = None, face_auroc: float | None = None) -> dict:
    """The fields of the JSON `scripts/v4_metrics.py` writes that the aggregation reads."""
    return {
        "latent_ap50": {"ap": latent_ap50, "n_gt": 9, "n_images": 30},
        "latent_ap30": {"ap": latent_ap30, "n_gt": 9, "n_images": 30},
        "face_auroc": {"auroc": face_auroc, "n_positive": 20, "n_negative": 10},
    }


def _write_v4(root: Path, arm: str, fold: int, seed: int, split: str, metrics: dict | str) -> Path:
    path = root / arm / f"fold{fold}_seed{seed}" / split / "metrics_v4.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(metrics if isinstance(metrics, str) else json.dumps(metrics))
    return path


def _v4_tree(root: Path, deltas: dict[int, float] = DELTAS, folds=FIVE, seeds=SEEDS, split: str = "test") -> None:
    """`_synthetic_tree` plus a metrics_v4.json per run, with known paired differences per metric."""
    _synthetic_tree(root, folds=folds, seeds=seeds)
    for fold in folds:
        for seed in seeds:
            base = 0.20 + 0.01 * fold
            noisy = deltas[fold] + SEED_NOISE[seed]
            _write_v4(root, BASE, fold, seed, split, _v4(base, base + 0.1, 0.6))
            _write_v4(root, TREAT, fold, seed, split, _v4(base + noisy, base + 0.1 + 2 * noisy, 0.6 + noisy / 2))


def test_v4_metrics_are_aggregated_per_fold_arm_and_seed(tmp_path: Path) -> None:
    _synthetic_tree(tmp_path)
    for fold in FIVE:
        _write_v4(tmp_path, BASE, fold, 42, "test", _v4(0.2, 0.4, 0.6))
        _write_v4(tmp_path, BASE, fold, 1337, "test", _v4(0.4, 0.6, 0.8))
    results = _inferred(tmp_path)

    row = results["per_fold"][BASE]["test"]["3"]
    assert row["latent_ap50"] == pytest.approx(0.3)     # (0.2 + 0.4) / 2
    assert row["latent_ap30"] == pytest.approx(0.5)     # (0.4 + 0.6) / 2
    assert row["face_auroc"] == pytest.approx(0.7)      # (0.6 + 0.8) / 2
    assert row["v4_seeds"]["latent_ap50"] == [42, 1337]
    assert results["summary"][BASE]["test"]["latent_ap50"]["mean"] == pytest.approx(0.3)
    assert results["summary"][BASE]["test"]["latent_ap50"]["n"] == 5
    assert results["metrics"] == [*agg.METRICS, *agg.V4_METRICS]
    # The metrics.json numbers of the same runs are unchanged by the merge.
    assert row["ap50"] == pytest.approx(0.36)
    assert row["seeds"] == [42, 1337]


def test_v4_metrics_enter_the_same_paired_inference(tmp_path: Path) -> None:
    _v4_tree(tmp_path)
    results = _inferred(tmp_path, rope=0.02)
    paired = results["paired"]

    deltas = [DELTAS[f] for f in FIVE]
    assert [paired["per_fold"]["test"][str(f)]["latent_ap50"] for f in FIVE] == pytest.approx(deltas)
    assert [paired["per_fold"]["test"][str(f)]["latent_ap30"] for f in FIVE] == pytest.approx([2 * d for d in deltas])
    assert [paired["per_fold"]["test"][str(f)]["face_auroc"] for f in FIVE] == pytest.approx([d / 2 for d in deltas])
    assert paired["summary"]["test"]["latent_ap50"]["mean"] == pytest.approx(statistics.fmean(deltas))
    per_seed = paired["per_fold_seed"]["test"]["latent_ap50"]["2"]
    assert per_seed["42"] == pytest.approx(DELTAS[2] + 0.01)
    assert per_seed["1337"] == pytest.approx(DELTAS[2] - 0.01)

    for metric, scale in (("latent_ap50", 1.0), ("latent_ap30", 2.0), ("face_auroc", 0.5)):
        expected = agg.paired_inference([d * scale for d in deltas], _spec_rows(), 0.02, 0.95)
        entry = paired["inference"]["test"][metric]
        assert entry["n_folds"] == 5
        assert entry["corrected_t"]["mean"] == pytest.approx(expected["corrected_t"]["mean"])
        assert entry["corrected_t"]["p_two_sided"] == pytest.approx(expected["corrected_t"]["p_two_sided"])
        assert entry["bayes"]["p_right"] == pytest.approx(expected["bayes"]["p_right"])
        assert entry["sign_flip"]["min_attainable_p_two_sided"] == pytest.approx(0.0625)


def test_a_null_v4_metric_stays_out_of_every_statistic_and_is_listed(tmp_path: Path) -> None:
    _v4_tree(tmp_path)
    for arm in (BASE, TREAT):
        for seed in SEEDS:
            _write_v4(tmp_path, arm, 4, seed, "test", _v4(None, None, None))
    results = _inferred(tmp_path)

    assert "latent_ap50" not in results["per_fold"][BASE]["test"]["4"]
    assert "latent_ap50" not in results["paired"]["per_fold"]["test"]["4"]
    assert set(results["paired"]["per_fold_seed"]["test"]["latent_ap50"]) == {"0", "1", "2", "3"}
    assert results["summary"][BASE]["test"]["latent_ap50"]["n"] == 4
    assert results["paired"]["inference"]["test"]["latent_ap50"]["n_folds"] == 4
    nulls = [m for m in results["v4"]["missing"] if m["status"] == "null"]
    assert {(m["arm"], m["fold"], m["seed"]) for m in nulls} == {(a, 4, s) for a in (BASE, TREAT) for s in SEEDS}
    assert all(m["split"] == "test" for m in nulls)
    # A fold whose latent stratum is empty still counts for what its metrics.json holds.
    assert results["paired"]["inference"]["test"]["ap50"]["n_folds"] == 5
    assert results["missing"] == []


def test_a_fold_with_a_v4_file_on_only_one_arm_has_no_pair(tmp_path: Path) -> None:
    _v4_tree(tmp_path)
    for seed in SEEDS:
        (tmp_path / TREAT / f"fold1_seed{seed}" / "test" / "metrics_v4.json").unlink()
    results = _inferred(tmp_path)

    assert "latent_ap50" not in results["paired"]["per_fold"]["test"]["1"]
    assert "latent_ap50" in results["per_fold"][BASE]["test"]["1"]
    absent = [m for m in results["v4"]["missing"] if m["status"] == "missing"]
    assert {(m["arm"], m["fold"], m["seed"]) for m in absent} == {(TREAT, 1, s) for s in SEEDS}


def test_an_unreadable_v4_file_is_listed_as_invalid_and_keeps_the_rest(tmp_path: Path) -> None:
    _v4_tree(tmp_path)
    _write_v4(tmp_path, BASE, 0, 42, "test", "{not json")
    _write_v4(tmp_path, BASE, 1, 42, "test", json.dumps({"latent_ap50": {}}))
    results = _inferred(tmp_path)

    invalid = [m for m in results["v4"]["missing"] if m["status"] == "invalid"]
    assert {(m["arm"], m["fold"], m["seed"]) for m in invalid} == {(BASE, 0, 42), (BASE, 1, 42)}
    assert all("metrics_v4.json" in m["reason"] for m in invalid)
    assert results["per_fold"][BASE]["test"]["0"]["v4_seeds"]["latent_ap50"] == [1337]


def test_a_split_with_no_v4_file_gets_no_v4_block(tmp_path: Path) -> None:
    _v4_tree(tmp_path, split="test")
    results = _inferred(tmp_path)

    assert results["v4"]["splits"] == ["test"]
    assert "latent_ap50" not in results["summary"][BASE]["val"]
    assert "latent_ap50" not in results["paired"]["inference"]["val"]
    assert all(m["split"] == "test" for m in results["v4"]["missing"])


def test_without_any_v4_file_the_results_stay_as_they_were(tmp_path: Path) -> None:
    _synthetic_tree(tmp_path)
    results = _inferred(tmp_path, rope=0.02)

    assert "v4" not in results
    assert results["metrics"] == list(agg.METRICS)
    for split in agg.SPLITS:
        assert set(results["paired"]["summary"][split]) == set(agg.METRICS)
        assert set(results["paired"]["inference"][split]) == set(agg.METRICS)
        assert set(results["paired"]["per_fold_seed"][split]) == set(agg.METRICS)
        assert set(results["paired"]["per_fold"][split]["0"]) == {*agg.METRICS, "seeds"}
        assert set(results["per_fold"][BASE][split]["0"]) == {*agg.METRICS, "num_gt", "seeds"}
    assert "Latent" not in agg.render_markdown(results)


def test_markdown_reports_the_v4_metrics_and_their_inference(tmp_path: Path) -> None:
    _v4_tree(tmp_path)
    _write_v4(tmp_path, TREAT, 4, 42, "test", _v4(None, None, None))
    markdown = agg.render_markdown(_inferred(tmp_path, rope=0.02))

    section = markdown[markdown.index("## Latent-window metrics"):markdown.index("## Missing runs")]
    assert f"#### {BASE}" in section and f"#### {TREAT}" in section
    assert "| fold | seeds | Latent AP50 | Latent AP30 | Face AUROC |" in section
    assert f"Paired difference {TREAT} − {BASE}" in section
    assert "| split | metric | n | mean Δ |" in section
    assert "| test | Latent AP50 | 5 |" in section
    assert "| test | Face AUROC | 5 |" in section
    assert "null" in section  # the fold 4 run with no value
    assert "significant" not in section.lower()


def test_main_writes_the_v4_block_and_stays_deterministic(tmp_path: Path) -> None:
    reports = tmp_path / "reports"
    _v4_tree(reports)
    spec = _write_spec(tmp_path)
    argv = ["--reports-root", str(reports), "--seeds", "42", "1337", "--fold-spec", str(spec), "--rope", "0.02"]
    assert agg.main(argv) == 0
    data = json.loads((reports / "kfold_results.json").read_text())
    assert data["paired"]["inference"]["test"]["latent_ap50"]["n_folds"] == 5
    assert data["v4"]["metrics"] == list(agg.V4_METRICS)
    assert "## Latent-window metrics" in (reports / "kfold_results.md").read_text()
    first = (reports / "kfold_results.json").read_bytes(), (reports / "kfold_results.md").read_bytes()
    assert agg.main(argv) == 0
    assert ((reports / "kfold_results.json").read_bytes(), (reports / "kfold_results.md").read_bytes()) == first


def _v4_fold_meta(sizes: dict[int, tuple[int, int]] = SIZES) -> dict:
    """The shape `scripts/make_v4_folds.py` writes to folds_meta.json: folds keyed by id, sizes under `splits`."""
    return {
        "folds": {
            str(k): {
                "splits": {
                    "test": {"images": te, "damage_boxes": 1, "mangos": 1, "negatives": 1},
                    "train": {"images": tr, "damage_boxes": 1, "mangos": 1, "negatives": 1},
                    "val": {"images": 7, "damage_boxes": 1, "mangos": 1, "negatives": 1},
                },
                "test": f"lot{k + 1}",
                "train": [],
                "val": f"lot{(k + 1) % 5 + 1}",
            }
            for k, (te, tr) in sizes.items()
        },
        "num_images": 0,
    }


def test_load_fold_spec_reads_the_v4_folds_meta(tmp_path: Path) -> None:
    path = tmp_path / "folds_meta.json"
    path.write_text(json.dumps(_v4_fold_meta()))
    rows = agg.load_fold_spec(path)
    assert [r["fold"] for r in rows] == [0, 1, 2, 3, 4]
    assert rows[2]["num_test"] == SIZES[2][0]
    assert rows[2]["num_train"] == SIZES[2][1]
    assert rows[2]["num_val"] == 7
    assert rows[2]["test"] == "lot3"
    assert fi.test_train_ratio(rows) == pytest.approx(fi.test_train_ratio(_spec_rows()))


def test_main_takes_the_v4_folds_meta_through_fold_spec(tmp_path: Path) -> None:
    reports = tmp_path / "reports"
    _synthetic_tree(reports)
    spec = tmp_path / "folds_meta.json"
    spec.write_text(json.dumps(_v4_fold_meta()))
    assert agg.main(["--reports-root", str(reports), "--seeds", "42", "1337", "--fold-spec", str(spec)]) == 0
    entry = json.loads((reports / "kfold_results.json").read_text())["paired"]["inference"]["test"]["ap50"]
    assert entry["test_train_ratio"] == pytest.approx(0.3)


@pytest.mark.parametrize(
    "folds",
    [
        {},
        {"0": 1},
        {"0": {"splits": {"test": {"images": 5}}}},                             # no train size
        {"x": {"splits": {"test": {"images": 5}, "train": {"images": 9}}}},     # not a fold id
    ],
)
def test_load_fold_spec_rejects_a_keyed_folds_block_it_cannot_read(tmp_path: Path, folds: dict) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"folds": folds}))
    with pytest.raises(ValueError, match="bad.json"):
        agg.load_fold_spec(path)


def test_load_fold_spec_returns_a_list_spec_as_written(tmp_path: Path) -> None:
    path = _write_spec(tmp_path)
    assert agg.load_fold_spec(path) == _spec_rows()


# --------------------------------------------------------------------------
# Mask twins of the latent metrics (segment runs)
# --------------------------------------------------------------------------


def _v4_with_masks(base: dict, mask_ap50: float | None, mask_ap30: float | None) -> dict:
    return {**base, "latent_mask_ap50": {"ap": mask_ap50, "n_gt": 9, "n_images": 30},
            "latent_mask_ap30": {"ap": mask_ap30, "n_gt": 9, "n_images": 30}}


def _v4_mask_tree(root: Path) -> None:
    _synthetic_tree(root)
    for fold in FIVE:
        for seed in SEEDS:
            noisy = DELTAS[fold] + SEED_NOISE[seed]
            _write_v4(root, BASE, fold, seed, "test", _v4_with_masks(_v4(0.3, 0.4, 0.6), 0.1, 0.2))
            _write_v4(root, TREAT, fold, seed, "test", _v4_with_masks(_v4(0.3, 0.4, 0.6), 0.1 + noisy, 0.2 + 3 * noisy))


def test_mask_latent_metrics_follow_the_box_ones_through_pairs_and_inference(tmp_path: Path) -> None:
    _v4_mask_tree(tmp_path)
    results = _inferred(tmp_path, rope=0.02)
    deltas = [DELTAS[f] for f in FIVE]

    assert results["v4"]["metrics"] == [*agg.V4_METRICS, *agg.V4_MASK_METRICS]
    assert results["per_fold"][BASE]["test"]["2"]["latent_mask_ap50"] == pytest.approx(0.1)
    paired = results["paired"]
    assert [paired["per_fold"]["test"][str(f)]["latent_mask_ap50"] for f in FIVE] == pytest.approx(deltas)
    assert [paired["per_fold"]["test"][str(f)]["latent_mask_ap30"] for f in FIVE] == pytest.approx([3 * d for d in deltas])
    expected = agg.paired_inference(deltas, _spec_rows(), 0.02, 0.95)
    got = paired["inference"]["test"]["latent_mask_ap50"]["corrected_t"]
    assert got["p_two_sided"] == pytest.approx(expected["corrected_t"]["p_two_sided"])
    # The box metrics beside them are the ones they always were.
    assert [paired["per_fold"]["test"][str(f)]["latent_ap50"] for f in FIVE] == pytest.approx([0.0] * 5)
    assert "Latent mask AP50" in agg.render_markdown(results)


def test_without_mask_keys_the_v4_block_is_what_it_was(tmp_path: Path) -> None:
    _v4_tree(tmp_path)
    results = _inferred(tmp_path)
    assert results["v4"]["metrics"] == list(agg.V4_METRICS)
    assert "latent_mask_ap50" not in results["paired"]["per_fold"]["test"]["0"]
    assert "mask AP50" not in agg.render_markdown(results).replace("Mask AP50", "")
