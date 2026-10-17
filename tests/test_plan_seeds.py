"""Tests for scripts/plan_seeds.py: seed budget from priors or from a kfold_results.json."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from scripts import aggregate_kfold as agg
from scripts import plan_seeds as ps
from src.evaluation import seed_planning as sp

# Helpers of the aggregator tests (fold-spec writer, synthetic metrics tree); tmp spec ratio = 150 / 500 = 0.3.
from tests.test_aggregate_kfold import FIVE, _synthetic_tree, _write_spec

RATIO = 0.3

# fold x seed paired differences of a hypothetical test-AP50 comparison (5 folds, 3 seeds)
MATRIX = {
    0: {42: 0.115, 1337: 0.085, 2024: 0.100},
    1: {42: 0.030, 1337: 0.050, 2024: 0.040},
    2: {42: 0.140, 1337: 0.110, 2024: 0.120},
    3: {42: -0.010, 1337: -0.040, 2024: -0.020},
    4: {42: 0.090, 1337: 0.060, 2024: 0.080},
}


def _write_results(path: Path, matrix: dict, split: str = "test", metric: str = "ap50") -> Path:
    per_fold_seed = {split: {metric: {str(f): {str(s): v for s, v in seeds.items()} for f, seeds in matrix.items()}}}
    path.write_text(json.dumps({"paired": {"per_fold_seed": per_fold_seed}}))
    return path


def _matrix(matrix: dict = MATRIX) -> np.ndarray:
    return np.array([[matrix[f][s] for s in sorted(matrix[f])] for f in sorted(matrix)])


# --------------------------------------------------------------------------
# Priors mode
# --------------------------------------------------------------------------

PRIORS = ["--sigma-fold", "0.05", "--sigma-seed", "0.144", "--folds", "5", "--ratio", "0.333", "--max-seeds", "8"]


def test_priors_mode_prints_the_table_and_the_plateau(capsys) -> None:
    assert ps.main(PRIORS) == 0
    out = capsys.readouterr().out
    rows = sp.project_seed_budget(0.05 ** 2, 0.144 ** 2, 5, 0.333, seeds=range(1, 9))
    assert "| seeds | SE | MDE |" in out
    for row in rows:
        assert f"| {row.n_seeds} | {row.se:.4f} | {row.mde:.4f} |" in out
    assert "| 9 |" not in out
    plateau = sp.plateau_se(0.05 ** 2, 5, 0.333)
    assert "Plateau SE" in out and f"{plateau:.4f}" in out
    assert "0.0500" in out and "0.1440" in out                    # the priors, echoed as sigmas
    assert "alpha" in out and "power" in out


def test_priors_mode_alpha_and_power_move_the_mde(tmp_path: Path) -> None:
    default, strict = tmp_path / "a.json", tmp_path / "b.json"
    assert ps.main([*PRIORS, "--json-out", str(default)]) == 0
    assert ps.main([*PRIORS, "--alpha", "0.01", "--power", "0.9", "--json-out", str(strict)]) == 0
    a, b = json.loads(default.read_text()), json.loads(strict.read_text())
    assert b["alpha"] == 0.01 and b["power"] == 0.9
    assert b["rows"][0]["mde"] > a["rows"][0]["mde"] and b["rows"][0]["se"] == a["rows"][0]["se"]


def test_priors_mode_takes_the_ratio_from_the_fold_spec_when_omitted(tmp_path: Path) -> None:
    spec, out = _write_spec(tmp_path), tmp_path / "plan.json"
    argv = ["--sigma-fold", "0.05", "--sigma-seed", "0.144", "--fold-spec", str(spec), "--json-out", str(out)]
    assert ps.main(argv) == 0
    plan = json.loads(out.read_text())
    assert plan["test_train_ratio"] == pytest.approx(RATIO)
    assert plan["n_folds"] == 5 and len(plan["rows"]) == 10        # --folds and --max-seeds defaults


def test_json_out_is_sorted_indented_and_newline_terminated(tmp_path: Path) -> None:
    out = tmp_path / "nested" / "plan.json"
    assert ps.main([*PRIORS, "--json-out", str(out)]) == 0
    text = out.read_text()
    assert text.endswith("}\n")
    assert text == json.dumps(json.loads(text), indent=2, sort_keys=True) + "\n"
    plan = json.loads(text)
    assert plan["mode"] == "priors"
    assert plan["plateau_se"] == pytest.approx(sp.plateau_se(0.05 ** 2, 5, 0.333))
    assert [r["n_seeds"] for r in plan["rows"]] == list(range(1, 9))


@pytest.mark.parametrize(
    "argv, fragment",
    [
        ([], "--results"),                                                    # neither mode
        (["--sigma-fold", "0.05"], "--sigma-seed"),
        (["--sigma-seed", "0.1"], "--sigma-fold"),
        (["--sigma-fold", "-0.05", "--sigma-seed", "0.1", "--ratio", "0.3"], "sigma"),
        (["--sigma-fold", "0.05", "--sigma-seed", "0.1", "--ratio", "0.3", "--folds", "1"], "folds"),
        (["--sigma-fold", "0.05", "--sigma-seed", "0.1", "--ratio", "-1"], "ratio"),
        (["--sigma-fold", "0.05", "--sigma-seed", "0.1", "--ratio", "0.3", "--max-seeds", "0"], "max-seeds"),
        (["--sigma-fold", "0.05", "--sigma-seed", "0.1", "--ratio", "0.3", "--alpha", "1.5"], "alpha"),
        (["--sigma-fold", "0.05", "--sigma-seed", "0.1", "--ratio", "0.3", "--power", "0"], "power"),
    ],
)
def test_invalid_priors_arguments_exit_non_zero_with_a_message(capsys, argv: list[str], fragment: str) -> None:
    assert ps.main(argv) == 2
    err = capsys.readouterr().err
    assert "ERROR" in err and fragment in err


def test_priors_mode_without_a_ratio_and_without_a_fold_spec_fails_clearly(tmp_path: Path, capsys) -> None:
    argv = ["--sigma-fold", "0.05", "--sigma-seed", "0.1", "--fold-spec", str(tmp_path / "nope.json")]
    assert ps.main(argv) == 2
    assert "nope.json" in capsys.readouterr().err


# --------------------------------------------------------------------------
# Results mode
# --------------------------------------------------------------------------


def test_results_mode_estimates_the_components_from_the_per_seed_values(tmp_path: Path, capsys) -> None:
    results, spec, out = _write_results(tmp_path / "kfold_results.json", MATRIX), _write_spec(tmp_path), tmp_path / "p.json"
    argv = ["--results", str(results), "--split", "test", "--metric", "ap50", "--fold-spec", str(spec),
            "--max-seeds", "6", "--json-out", str(out)]
    assert ps.main(argv) == 0

    expected = sp.estimate_variance_components(_matrix())
    plan = json.loads(out.read_text())
    assert plan["mode"] == "results"
    assert plan["sigma2_seed"] == pytest.approx(expected.sigma2_seed)
    assert plan["sigma2_fold"] == pytest.approx(expected.sigma2_fold)
    assert plan["truncated"] is False
    assert (plan["n_folds"], plan["n_seeds_observed"]) == (5, 3)
    assert plan["test_train_ratio"] == pytest.approx(RATIO)
    assert plan["source"] == {"metric": "ap50", "results": str(results), "split": "test"}
    rows = sp.project_seed_budget(expected.sigma2_fold, expected.sigma2_seed, 5, RATIO, seeds=range(1, 7))
    assert [(r["n_seeds"], r["se"], r["mde"]) for r in plan["rows"]] == pytest.approx(
        [(r.n_seeds, r.se, r.mde) for r in rows])
    assert plan["plateau_se"] == pytest.approx(sp.plateau_se(expected.sigma2_fold, 5, RATIO))

    out_text = capsys.readouterr().out
    assert "| seeds | SE | MDE |" in out_text and "Plateau SE" in out_text
    assert f"{math.sqrt(expected.sigma2_seed):.4f}" in out_text      # estimated sigma_seed
    assert f"{math.sqrt(expected.sigma2_fold):.4f}" in out_text      # estimated sigma_fold


def test_results_mode_reads_the_requested_split_and_metric(tmp_path: Path) -> None:
    results = _write_results(tmp_path / "r.json", MATRIX, split="val", metric="recall")
    spec, out = _write_spec(tmp_path), tmp_path / "p.json"
    assert ps.main(["--results", str(results), "--split", "val", "--metric", "recall",
                    "--fold-spec", str(spec), "--json-out", str(out)]) == 0
    assert json.loads(out.read_text())["source"]["split"] == "val"
    assert ps.main(["--results", str(results), "--fold-spec", str(spec)]) == 2     # defaults: test / ap50


def test_results_mode_with_a_single_seed_requires_sigma_seed(tmp_path: Path, capsys) -> None:
    single = {f: {42: seeds[42]} for f, seeds in MATRIX.items()}
    results, spec = _write_results(tmp_path / "r.json", single), _write_spec(tmp_path)
    assert ps.main(["--results", str(results), "--fold-spec", str(spec)]) == 2
    err = capsys.readouterr().err
    assert "ERROR" in err and "--sigma-seed" in err

    out = tmp_path / "p.json"
    assert ps.main(["--results", str(results), "--fold-spec", str(spec), "--sigma-seed", "0.02",
                    "--json-out", str(out)]) == 0
    plan = json.loads(out.read_text())
    expected = sp.estimate_variance_components(_matrix(single), sigma2_seed=0.02 ** 2)
    assert plan["sigma2_seed"] == pytest.approx(0.02 ** 2)
    assert plan["sigma2_fold"] == pytest.approx(expected.sigma2_fold)
    assert plan["n_seeds_observed"] == 1


def test_results_mode_uses_only_the_seeds_every_fold_has(tmp_path: Path, capsys) -> None:
    ragged = {f: dict(seeds) for f, seeds in MATRIX.items()}
    del ragged[2][2024]                                            # fold 2 lacks seed 2024
    results, spec, out = _write_results(tmp_path / "r.json", ragged), _write_spec(tmp_path), tmp_path / "p.json"
    assert ps.main(["--results", str(results), "--fold-spec", str(spec), "--json-out", str(out)]) == 0
    plan = json.loads(out.read_text())
    assert plan["n_seeds_observed"] == 2
    balanced = {f: {s: v for s, v in seeds.items() if s != 2024} for f, seeds in MATRIX.items()}
    assert plan["sigma2_seed"] == pytest.approx(sp.estimate_variance_components(_matrix(balanced)).sigma2_seed)
    assert "2024" in capsys.readouterr().err                       # the dropped seed is named


def test_results_mode_ratio_covers_only_the_folds_present(tmp_path: Path) -> None:
    partial = {f: seeds for f, seeds in MATRIX.items() if f != 3}
    results, spec, out = _write_results(tmp_path / "r.json", partial), _write_spec(tmp_path), tmp_path / "p.json"
    assert ps.main(["--results", str(results), "--fold-spec", str(spec), "--json-out", str(out)]) == 0
    plan = json.loads(out.read_text())
    assert plan["n_folds"] == 4
    assert plan["test_train_ratio"] == pytest.approx((10 + 20 + 30 + 50) / 400)


def test_results_mode_flags_a_truncated_fold_component(tmp_path: Path, capsys) -> None:
    flat = {f: {42: 0.0, 1337: 0.2} for f in FIVE}                  # identical fold means, noisy seeds
    results, spec, out = _write_results(tmp_path / "r.json", flat), _write_spec(tmp_path), tmp_path / "p.json"
    assert ps.main(["--results", str(results), "--fold-spec", str(spec), "--json-out", str(out)]) == 0
    plan = json.loads(out.read_text())
    assert plan["truncated"] is True and plan["sigma2_fold"] == 0.0 and plan["plateau_se"] == 0.0
    assert "truncated" in capsys.readouterr().out.lower()


@pytest.mark.parametrize(
    "payload, fragment",
    [
        ({"paired": {"per_fold": {}}}, "aggregate_kfold.py"),                          # results from an old aggregator
        ({"paired": {"per_fold_seed": {"test": {}}}}, "ap50"),                         # metric absent
        ({"paired": {"per_fold_seed": {"val": {"ap50": {"0": {"42": 0.1}}}}}}, "test"),  # split absent
        ({"paired": {"per_fold_seed": {"test": {"ap50": {"0": {"42": 0.1}}}}}}, "at least 2 folds"),
    ],
)
def test_results_mode_reports_unusable_results_files(tmp_path: Path, capsys, payload: dict, fragment: str) -> None:
    results = tmp_path / "r.json"
    results.write_text(json.dumps(payload))
    assert ps.main(["--results", str(results), "--fold-spec", str(_write_spec(tmp_path)),
                    "--sigma-seed", "0.02"]) == 2
    err = capsys.readouterr().err
    assert "ERROR" in err and fragment in err


def test_results_mode_rejects_unreadable_files_and_unknown_folds(tmp_path: Path, capsys) -> None:
    spec = _write_spec(tmp_path)
    assert ps.main(["--results", str(tmp_path / "missing.json"), "--fold-spec", str(spec)]) == 2
    assert "missing.json" in capsys.readouterr().err

    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert ps.main(["--results", str(bad), "--fold-spec", str(spec)]) == 2
    assert "bad.json" in capsys.readouterr().err

    results = _write_results(tmp_path / "r.json", {**MATRIX, 9: MATRIX[0]})
    assert ps.main(["--results", str(results), "--fold-spec", str(spec)]) == 2
    err = capsys.readouterr().err
    assert "9" in err and "fold spec" in err


def test_results_mode_rejects_prior_only_flags(tmp_path: Path, capsys) -> None:
    results, spec = _write_results(tmp_path / "r.json", MATRIX), _write_spec(tmp_path)
    for extra in (["--sigma-fold", "0.05"], ["--folds", "5"], ["--ratio", "0.3"]):
        assert ps.main(["--results", str(results), "--fold-spec", str(spec), *extra]) == 2
        assert "--results" in capsys.readouterr().err


def test_end_to_end_from_the_aggregator_output(tmp_path: Path) -> None:
    reports = tmp_path / "reports"
    _synthetic_tree(reports)
    spec = _write_spec(tmp_path)
    assert agg.main(["--reports-root", str(reports), "--seeds", "42", "1337", "--fold-spec", str(spec)]) == 0
    out = tmp_path / "plan.json"
    assert ps.main(["--results", str(reports / "kfold_results.json"), "--fold-spec", str(spec),
                    "--json-out", str(out)]) == 0
    plan = json.loads(out.read_text())
    assert plan["n_folds"] == 5 and plan["n_seeds_observed"] == 2
    # the synthetic tree has seed offsets of +/-0.01 around each fold's delta -> sigma2_seed = 2 * 0.01**2
    assert plan["sigma2_seed"] == pytest.approx(2e-4, rel=1e-6)
    assert plan["test_train_ratio"] == pytest.approx(RATIO)
