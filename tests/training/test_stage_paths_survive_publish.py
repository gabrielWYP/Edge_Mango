"""A published run's summaries must point at files that exist.

`scripts/run_final_training_pipeline.py` stages the whole run under
`<output_root>/.partial/<run_id>-<uuid>/` and, on success,
`publish_atomic_run` renames that directory to `<output_root>/<run_id>/`.
`finalize_stage` had already written `stage_summary.json` — and the pipeline
its `run_summary.json` — with `output_dir` and `checkpoint` recorded as
paths under the staging directory. After the rename every one of them was
dead. Confirmed on `20260901T200124175077Z` and `20260831T224505981163Z`:
`Path(result.checkpoint).exists()` was False for all three stages of both.

The fix records `output_dir` and `checkpoint` relative to the run root
(`maestro/best_model_<run_id>_mAP0p5073/best_model.pt`). The root is the
one thing that moves, and everything beneath it moves with it, so the record
cannot be invalidated by publishing — by construction, not by a rewrite step
that a future mover could forget. `run_summary.json` keeps `run_dir` as the
anchor readers join against.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import scripts.run_final_training_pipeline as pipeline
from src.training.run_artifacts import (
    collect_stage_results,
    finalize_stage,
    load_stage_result,
)

RUN_ID = "20260901T000000Z"
SHA = "c" * 64


def _plant_checkpoint(path: Path, best_map50: float = 0.5) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "best_map50": best_map50,
            "model_state_dict": {},
            "experiment_sha256": SHA,
            "config": {"effective_batch": 8, "precision": "fp32", "device": "cpu"},
        },
        path,
    )


def _finalize_under(root: Path, stage_key: str = "maestro") -> object:
    tmp_dir = root / ".stage_tmp" / stage_key
    _plant_checkpoint(tmp_dir / "best_model.pt")
    return finalize_stage(
        tmp_dir=tmp_dir,
        final_parent=root / stage_key,
        run_id=RUN_ID,
        command=["python"],
        stage_key=stage_key,
        stage_label="Maestro",
    )


class TestStageResultIsRunRelative:
    def test_paths_are_relative_to_the_run_root(self, tmp_path):
        result = _finalize_under(tmp_path)

        assert not Path(result.output_dir).is_absolute()
        assert Path(result.output_dir).parts[0] == "maestro"
        assert result.checkpoint == f"{result.output_dir}/best_model.pt"
        assert (tmp_path / result.checkpoint).exists()

    def test_dry_run_plans_the_same_shape(self, tmp_path):
        result = finalize_stage(
            tmp_dir=tmp_path / ".stage_tmp" / "estudiante",
            final_parent=tmp_path / "estudiante",
            run_id=RUN_ID,
            command=["python"],
            stage_key="estudiante",
            stage_label="Estudiante",
            dry_run=True,
        )

        assert Path(result.output_dir).parts[0] == "estudiante"
        assert not Path(result.checkpoint).is_absolute()


class TestPublishingCannotBreakTheRecord:
    def test_stage_summary_still_resolves_after_the_root_is_renamed(self, tmp_path):
        staging = tmp_path / ".partial" / f"{RUN_ID}-deadbeef"
        published = tmp_path / RUN_ID
        result = _finalize_under(staging)

        os.rename(staging, published)

        reloaded = load_stage_result(
            published / result.output_dir / "stage_summary.json"
        )
        assert (published / reloaded.checkpoint).exists()
        assert reloaded.experiment_sha256 == SHA

    def test_collect_stage_results_resolves_against_the_published_root(self, tmp_path):
        staging = tmp_path / ".partial" / f"{RUN_ID}-deadbeef"
        published = tmp_path / RUN_ID
        _finalize_under(staging, "maestro")
        _finalize_under(staging, "estudiante")

        os.rename(staging, published)

        results = collect_stage_results(published)
        assert [r.stage_key for r in results] == ["maestro", "estudiante"]
        for result in results:
            assert (published / result.checkpoint).exists()


class TestThePipelinePublishesLivePaths:
    """The end-to-end shape of the defect: run main(), then follow the JSON."""

    def test_run_summary_checkpoints_exist_after_publish(self, tmp_path, monkeypatch):
        def _fake_training(command: list[str], dry_run: bool) -> None:
            overrides = command[command.index("--override") + 1 :]
            output_dir = next(o for o in overrides if o.startswith("output_dir=")).split("=", 1)[1]
            _plant_checkpoint(Path(output_dir) / "best_model.pt")

        monkeypatch.setattr(pipeline, "run_command", _fake_training)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "run_final_training_pipeline",
                "--output-root",
                str(tmp_path),
                "--timestamp",
                RUN_ID,
            ],
        )

        assert pipeline.main() == 0

        run_dir = tmp_path / RUN_ID
        summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
        assert summary["run_id"] == RUN_ID
        assert [r["stage_key"] for r in summary["results"]] == [
            "maestro",
            "estudiante",
            "destilado",
        ]
        for result in summary["results"]:
            checkpoint = Path(summary["run_dir"]) / result["checkpoint"]
            assert checkpoint.exists(), f"{result['stage_key']}: {checkpoint} is dead"
            assert result["experiment_sha256"] == SHA
        assert not (tmp_path / ".partial" / f"{RUN_ID}-").exists()

    def test_kd_stage_receives_a_resolvable_teacher_checkpoint(self, tmp_path, monkeypatch):
        """The teacher path is consumed *during* the run, from the staging dir."""
        seen: dict[str, str] = {}

        def _fake_training(command: list[str], dry_run: bool) -> None:
            overrides = command[command.index("--override") + 1 :]
            output_dir = next(o for o in overrides if o.startswith("output_dir=")).split("=", 1)[1]
            _plant_checkpoint(Path(output_dir) / "best_model.pt")
            teacher = [o for o in overrides if o.startswith("teacher_checkpoint=")]
            if teacher:
                seen["teacher"] = teacher[0].split("=", 1)[1]
                assert Path(seen["teacher"]).exists(), seen["teacher"]

        monkeypatch.setattr(pipeline, "run_command", _fake_training)
        monkeypatch.setattr(
            sys,
            "argv",
            ["run_final_training_pipeline", "--output-root", str(tmp_path), "--timestamp", RUN_ID],
        )

        assert pipeline.main() == 0
        assert "teacher" in seen
