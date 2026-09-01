"""Every published stage must record what made it comparable.

`StageResult` carries the four D-H hardware-portability fields, and its own
docstring states the rule: two runs are comparable only if
`experiment_sha256` and `effective_batch` both match. `train.py` passes them
when it finalises its own `--versioned-run` stage, but
`scripts/run_final_training_pipeline.py` — the entry point that actually
produces the three-stage final runs — calls `finalize_stage` without them,
so every `run_summary.json` it wrote recorded:

    "device": null, "effective_batch": null,
    "experiment_sha256": null, "precision": null

Confirmed in `checkpoints/final_runs/20260831T224505981163Z/run_summary.json`.
The comparability rule had no data to run on, and this investigation had to
reconstruct the batch size by parsing the stored `command` string.

`_save_checkpoint` already records all four (`experiment_sha256` at the top
level, the rest inside `config`), and `finalize_stage` already opens that
checkpoint to read `best_map50`. Reading provenance from the same file makes
it ground truth from the run that actually happened rather than a second
derivation that can drift from it — and fixes both callers at once.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.training.run_artifacts import finalize_stage

SHA = "a" * 64


def _write_checkpoint(path: Path, *, provenance: bool = True) -> None:
    payload: dict = {"best_map50": 0.4806, "model_state_dict": {}}
    if provenance:
        payload["experiment_sha256"] = SHA
        payload["config"] = {
            "effective_batch": 8,
            "precision": "fp32",
            "device": "cuda",
            "batch_size": 4,
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def _finalize(tmp_path: Path, **kwargs):
    tmp_dir = tmp_path / "stage_tmp"
    _write_checkpoint(tmp_dir / "best_model.pt", provenance=kwargs.pop("provenance", True))
    return finalize_stage(
        tmp_dir=tmp_dir,
        final_parent=tmp_path / "maestro",
        run_id="20260901T000000Z",
        command=["python", "-m", "src.training.train"],
        stage_key="maestro",
        stage_label="Maestro",
        **kwargs,
    )


def test_provenance_is_read_from_the_checkpoint_when_not_passed(tmp_path):
    result = _finalize(tmp_path)

    assert result.experiment_sha256 == SHA
    assert result.effective_batch == 8
    assert result.precision == "fp32"
    assert result.device == "cuda"


def test_stage_summary_on_disk_carries_the_provenance(tmp_path):
    """The pipeline's run_summary.json is built from these files."""
    import json

    result = _finalize(tmp_path)
    summary = json.loads((Path(result.output_dir) / "stage_summary.json").read_text())

    assert summary["experiment_sha256"] == SHA
    assert summary["effective_batch"] == 8
    assert summary["precision"] == "fp32"
    assert summary["device"] == "cuda"


def test_explicit_arguments_win_over_the_checkpoint(tmp_path):
    """`train.py` passes these itself; the caller stays authoritative."""
    result = _finalize(
        tmp_path,
        experiment_sha256="b" * 64,
        effective_batch=16,
        precision="fp16",
        device="cpu",
    )

    assert result.experiment_sha256 == "b" * 64
    assert result.effective_batch == 16
    assert result.precision == "fp16"
    assert result.device == "cpu"


def test_a_checkpoint_without_provenance_leaves_the_fields_none(tmp_path):
    """Checkpoints written before these keys existed must still finalise."""
    result = _finalize(tmp_path, provenance=False)

    assert result.best_map50 == 0.4806
    assert result.experiment_sha256 is None
    assert result.effective_batch is None
    assert result.precision is None
    assert result.device is None
