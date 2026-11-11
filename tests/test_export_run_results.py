"""The differences of `export_run_results.py` and the arms it may read."""

from __future__ import annotations

from types import SimpleNamespace

from scripts import export_run_results as ex

OLD_ARMS = ("est_rgb", "est_rgbnir", "kd_rgb", "kd_rgbnir", "est_nir")
CROSSKD_ARMS = ("crosskd_rgb", "crosskd_rgbnir", "crosskd_rgbnir_high", "crosskd_rgbnir_low", "crosskd_rgbnir_better")


def _ap(arms: dict[str, float], fold: int = 0, seed: int = 42, split: str = "val") -> dict:
    return {(fold, seed, split, "best", "latent_ap50", arm): value for arm, value in arms.items()}


def test_the_new_comparisons_name_the_arms_the_brief_fixes() -> None:
    assert ex.DIFFERENCES["crosskd_H2"] == ("crosskd_rgbnir", "crosskd_rgb")
    assert ex.DIFFERENCES["crosskd_vs_mse"] == ("crosskd_rgbnir", "kd_rgbnir")
    assert ex.DIFFERENCES["crosskd_vs_est"] == ("crosskd_rgbnir", "est_rgb")
    assert ex.DIFFERENCES["gate_high_vs_low"] == ("crosskd_rgbnir_high", "crosskd_rgbnir_low")
    assert ex.DIFFERENCES["gate_better_vs_none"] == ("crosskd_rgbnir_better", "crosskd_rgbnir")


def test_existing_runs_yield_exactly_the_rows_they_always_did() -> None:
    ap = _ap({"est_rgb": 0.10, "est_rgbnir": 0.30, "kd_rgb": 0.20, "kd_rgbnir": 0.25})
    rows = ex.compute_differences(ap)
    assert [(r["comparison"], r["treatment"], r["baseline"]) for r in rows] == [
        ("H1", "est_rgbnir", "est_rgb"),
        ("H2", "kd_rgbnir", "kd_rgb"),
        ("secondary", "kd_rgbnir", "est_rgb"),
        ("exploratory", "kd_rgb", "est_rgb"),
    ]
    h1 = rows[0]
    assert h1 == {
        "fold": 0, "seed": 42, "split": "val", "checkpoint": "best", "metric": "latent_ap50",
        "comparison": "H1", "treatment": "est_rgbnir", "baseline": "est_rgb",
        "treatment_ap": 0.30, "baseline_ap": 0.10, "delta": 0.30 - 0.10,
    }


def test_a_difference_whose_arms_were_not_exported_is_skipped() -> None:
    ap = _ap({"est_rgb": 0.1, "crosskd_rgbnir": 0.4})
    assert [r["comparison"] for r in ex.compute_differences(ap)] == ["crosskd_vs_est"]


def test_crosskd_comparisons_appear_when_their_arms_exist() -> None:
    ap = _ap({"crosskd_rgb": 0.1, "crosskd_rgbnir": 0.3, "crosskd_rgbnir_high": 0.5, "crosskd_rgbnir_low": 0.2,
              "crosskd_rgbnir_better": 0.4})
    names = {r["comparison"]: r["delta"] for r in ex.compute_differences(ap)}
    assert names["crosskd_H2"] == 0.3 - 0.1
    assert names["gate_high_vs_low"] == 0.5 - 0.2
    assert names["gate_better_vs_none"] == 0.4 - 0.3


def test_segment_twins_get_the_four_plan_comparisons_for_box_and_mask_metrics() -> None:
    ap = {}
    for metric, values in (("latent_ap50", (0.10, 0.30)), ("latent_mask_ap50", (0.05, 0.20)),
                           ("latent_mask_ap30", (0.15, 0.40))):
        ap[(0, 42, "val", "best", metric, "est_rgb_seg")] = values[0]
        ap[(0, 42, "val", "best", metric, "est_rgbnir_seg")] = values[1]
    rows = ex.compute_differences(ap)
    assert {(r["comparison"], r["metric"]) for r in rows} == {
        ("H1_seg", "latent_ap50"), ("H1_seg", "latent_mask_ap50"), ("H1_seg", "latent_mask_ap30")}
    mask = next(r for r in rows if r["metric"] == "latent_mask_ap50")
    assert (mask["treatment"], mask["baseline"]) == ("est_rgbnir_seg", "est_rgb_seg")
    assert mask["delta"] == 0.20 - 0.05


def test_segdt_twins_have_their_own_comparisons_and_detect_arms_get_no_mask_rows() -> None:
    ap = {(0, 42, "val", "best", "latent_mask_ap50", arm): v
          for arm, v in {"kd_rgb_segdt": 0.1, "kd_rgbnir_segdt": 0.3}.items()}
    assert [r["comparison"] for r in ex.compute_differences(ap)] == ["H2_segdt"]
    assert {r["metric"] for r in ex.compute_differences(_ap({"est_rgb": 0.1, "est_rgbnir": 0.3}))} == {"latent_ap50"}


def test_ap_entries_hold_the_mask_metrics_only_when_the_run_has_them() -> None:
    box_only = {"latent_ap50": {"ap": 0.1}, "latent_ap30": {"ap": 0.2}}
    assert ex.ap_entries(box_only) == {"latent_ap50": 0.1, "latent_ap30": 0.2}
    segment = {**box_only, "latent_mask_ap50": {"ap": 0.3}, "latent_mask_ap30": {"ap": None}}
    assert ex.ap_entries(segment) == {"latent_ap50": 0.1, "latent_ap30": 0.2,
                                      "latent_mask_ap50": 0.3, "latent_mask_ap30": None}


def test_the_crosskd_arms_can_be_exported(tmp_path) -> None:
    for arm in (*OLD_ARMS, *CROSSKD_ARMS):
        run = tmp_path / "reports" / arm / "fold0_seed42" / "val"
        run.mkdir(parents=True)
        (run / "metrics_v4.json").write_text("{}")
    args = SimpleNamespace(runs_root=tmp_path, test_folds=[], test_seeds=[], test_arms=None)
    found = {arm for arm, *_ in ex.evaluations(args, {})}
    assert found == {*OLD_ARMS, *CROSSKD_ARMS}
