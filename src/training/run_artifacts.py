"""Utilities for timestamped, atomic training run artifacts."""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)


STAGE_LABELS: dict[str, str] = {
    "maestro": "Maestro",
    "estudiante": "Estudiante",
    "destilado": "Destilado",
}


@dataclass(frozen=True)
class StageResult:
    """Metadata for one completed training stage.

    The four hardware-portability fields (W9, fusion-redesign D-H) are
    optional and default to `None` so existing `stage_summary.json` files
    without them still load via `StageResult(**data)`. `experiment_sha256`
    and `effective_batch` together are the comparability rule: two runs are
    comparable only if both match.

    `output_dir` and `checkpoint` are recorded **relative to the run root**
    (`maestro/best_model_<run_id>_mAP0p5073/best_model.pt`), never as paths
    that include the root itself. The final pipeline stages a whole run
    under `.partial/<run_id>-<uuid>/` and publishes it by renaming that
    directory, so any recorded path that named the staging root was dead the
    moment the run succeeded — every checkpoint path in
    `20260901T200124175077Z`'s `run_summary.json` resolved to nothing.
    Readers join these against the run directory they found the summary in
    (`run_summary.json` carries it as `run_dir`).
    """

    stage_key: str
    stage_label: str
    output_dir: str
    checkpoint: str
    best_map50: float
    command: list[str]
    experiment_sha256: str | None = None
    effective_batch: int | None = None
    precision: str | None = None
    device: str | None = None


@dataclass(frozen=True)
class VersionedStagePlan:
    """Resolved paths for one atomic versioned stage run."""

    run_id: str
    run_dir: Path
    stage_key: str
    stage_label: str
    tmp_dir: Path
    final_parent: Path


@dataclass(frozen=True)
class AtomicRunPlan:
    """Private staging location and lock for an all-or-nothing pipeline run."""

    run_id: str
    run_dir: Path
    staging_dir: Path
    lock_dir: Path


def utc_timestamp() -> str:
    """Return a filesystem-safe UTC timestamp."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def final_stage_name(run_id: str, best_map50: float) -> str:
    """Build the final directory name for one stage."""
    metric = f"{best_map50:.4f}".replace(".", "p")
    return f"best_model_{run_id}_mAP{metric}"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomically write a deterministic JSON document."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as tmp:
        tmp.write(json.dumps(payload, indent=2, sort_keys=True))
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, path)


def prepare_atomic_run(output_root: str | Path, run_id: str) -> AtomicRunPlan:
    """Reserve a run ID and create a private staging directory.

    The lock is created with ``mkdir``, which is atomic on a shared local
    filesystem. This prevents two processes using the same explicit timestamp
    from writing into the same run. Nothing under ``final_runs/<run_id>`` is
    visible until :func:`publish_atomic_run` succeeds.
    """
    root = Path(output_root)
    run_dir = root / run_id
    lock_dir = root / ".locks" / run_id
    root.mkdir(parents=True, exist_ok=True)
    lock_dir.parent.mkdir(parents=True, exist_ok=True)
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {run_dir}")
    try:
        lock_dir.mkdir()
    except FileExistsError as error:
        raise FileExistsError(f"Run ID is already active: {run_id}") from error

    staging_dir = root / ".partial" / f"{run_id}-{uuid.uuid4().hex}"
    try:
        staging_dir.mkdir(parents=True)
    except Exception:
        lock_dir.rmdir()
        raise
    return AtomicRunPlan(run_id, run_dir, staging_dir, lock_dir)


def publish_atomic_run(plan: AtomicRunPlan) -> None:
    """Atomically publish a fully completed staged run directory."""
    if plan.run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {plan.run_dir}")
    if not plan.staging_dir.exists():
        raise FileNotFoundError(f"Missing staged run directory: {plan.staging_dir}")
    os.rename(plan.staging_dir, plan.run_dir)


def release_atomic_run(plan: AtomicRunPlan) -> None:
    """Release a run reservation without deleting failed-run evidence."""
    try:
        plan.lock_dir.rmdir()
    except FileNotFoundError:
        pass


def _read_checkpoint(checkpoint_path: Path) -> dict:
    """Load a stage checkpoint once.

    Stage checkpoints are hundreds of megabytes, so callers that need both
    the metric and the provenance read them through this and pull both out
    of the same payload rather than loading the file twice.
    """
    import torch

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing stage checkpoint: {checkpoint_path}")

    return torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)


def _best_map50_of(checkpoint: dict, checkpoint_path: Path) -> float:
    metric = checkpoint.get("best_map50")
    if metric is None:
        metric = checkpoint.get("metrics", {}).get("map50")
    if metric is None:
        raise ValueError(f"Checkpoint does not contain best mAP@0.5: {checkpoint_path}")
    return float(metric)


def load_best_map50(checkpoint_path: Path) -> float:
    """Read the best mAP@0.5 from a saved checkpoint."""
    return _best_map50_of(_read_checkpoint(checkpoint_path), checkpoint_path)


def checkpoint_provenance(checkpoint: dict) -> dict:
    """Extract the four D-H comparability fields from a checkpoint payload.

    `Trainer._save_checkpoint` records `experiment_sha256` at the top level
    and the rest inside `config`, so the run that actually happened is the
    source of truth here. Deriving them a second time from the config files
    would let the record drift from the run — which is precisely the failure
    this data exists to make detectable.

    Returns `None` for any field a pre-D-H checkpoint does not carry.
    """
    config = checkpoint.get("config") or {}
    device = config.get("device")
    return {
        "experiment_sha256": checkpoint.get("experiment_sha256"),
        "effective_batch": config.get("effective_batch"),
        "precision": config.get("precision"),
        # `device` may be a torch.device; `stage_summary.json` must stay
        # JSON-serialisable.
        "device": None if device is None else str(device),
    }


def build_versioned_stage_plan(
    *,
    output_root: str | Path,
    stage_key: str,
    run_id: str | None = None,
) -> VersionedStagePlan:
    """Resolve temporary and final paths for one versioned stage.

    Args:
        output_root: Root directory for timestamped runs.
        stage_key: One of ``maestro``, ``estudiante`` or ``destilado``.
        run_id: Optional existing timestamp to reuse across independent runs.

    Returns:
        VersionedStagePlan with temporary and final parent directories.
    """
    if stage_key not in STAGE_LABELS:
        valid = ", ".join(sorted(STAGE_LABELS))
        raise ValueError(f"Invalid stage_key '{stage_key}'. Expected one of: {valid}")

    resolved_run_id = run_id or utc_timestamp()
    run_dir = Path(output_root) / resolved_run_id
    return VersionedStagePlan(
        run_id=resolved_run_id,
        run_dir=run_dir,
        stage_key=stage_key,
        stage_label=STAGE_LABELS[stage_key],
        tmp_dir=run_dir / ".partial" / stage_key,
        final_parent=run_dir / stage_key,
    )


def finalize_stage(
    *,
    tmp_dir: Path,
    final_parent: Path,
    run_id: str,
    command: list[str],
    stage_key: str,
    stage_label: str,
    dry_run: bool = False,
    experiment_sha256: str | None = None,
    effective_batch: int | None = None,
    precision: str | None = None,
    device: str | None = None,
) -> StageResult:
    """Move a successful temporary stage directory into its final location."""
    # `final_parent` is `<run root>/<stage_key>` for both callers, so the
    # run root is its parent. Paths are recorded relative to it — see the
    # `StageResult` docstring for why absolute-to-staging paths were dead.
    run_root = final_parent.parent

    if dry_run:
        planned_dir = final_parent / final_stage_name(run_id, 0.0)
        planned_rel = planned_dir.relative_to(run_root)
        return StageResult(
            stage_key=stage_key,
            stage_label=stage_label,
            output_dir=planned_rel.as_posix(),
            checkpoint=(planned_rel / "best_model.pt").as_posix(),
            best_map50=0.0,
            command=command,
            experiment_sha256=experiment_sha256,
            effective_batch=effective_batch,
            precision=precision,
            device=device,
        )

    checkpoint_path = tmp_dir / "best_model.pt"
    checkpoint = _read_checkpoint(checkpoint_path)
    best_map50 = _best_map50_of(checkpoint, checkpoint_path)

    # Provenance falls back to what the run recorded, so a caller that
    # launches training as a subprocess — `run_final_training_pipeline.py`
    # does — still publishes the comparability fields instead of nulls. An
    # explicit argument stays authoritative for callers that hold the
    # resolved config themselves (`train.py --versioned-run`).
    recorded = checkpoint_provenance(checkpoint)
    if experiment_sha256 is None:
        experiment_sha256 = recorded["experiment_sha256"]
    if effective_batch is None:
        effective_batch = recorded["effective_batch"]
    if precision is None:
        precision = recorded["precision"]
    if device is None:
        device = recorded["device"]

    final_dir = final_parent / final_stage_name(run_id, best_map50)

    if final_parent.exists() and any(final_parent.iterdir()):
        raise FileExistsError(f"Stage directory already contains artifacts: {final_parent}")
    if final_dir.exists():
        raise FileExistsError(f"Final stage directory already exists: {final_dir}")

    final_parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(tmp_dir), str(final_dir))

    final_rel = final_dir.relative_to(run_root)
    result = StageResult(
        stage_key=stage_key,
        stage_label=stage_label,
        output_dir=final_rel.as_posix(),
        checkpoint=(final_rel / "best_model.pt").as_posix(),
        best_map50=best_map50,
        command=command,
        experiment_sha256=experiment_sha256,
        effective_batch=effective_batch,
        precision=precision,
        device=device,
    )
    write_json(final_dir / "stage_summary.json", asdict(result))
    return result


def load_stage_result(path: Path) -> StageResult:
    """Load one stage summary JSON."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return StageResult(**data)


def collect_stage_results(run_dir: Path) -> list[StageResult]:
    """Collect stage summaries already written under a run directory."""
    results: list[StageResult] = []
    for stage_key in STAGE_LABELS:
        for summary_path in sorted((run_dir / stage_key).glob("*/stage_summary.json")):
            results.append(load_stage_result(summary_path))
    return results


def generate_run_summary_image(run_dir: Path, results: list[StageResult]) -> None:
    """Generate a compact PNG summary for immediate visual inspection."""
    if not results:
        return

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib unavailable; skipping final summary PNG")
        return

    labels = [result.stage_label for result in results]
    scores = [result.best_map50 for result in results]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(labels, scores, color=["#4C78A8", "#F58518", "#54A24B"][: len(results)])
    ax.set_ylim(0.0, max(1.0, max(scores) * 1.15))
    ax.set_ylabel("Best mAP@0.5")
    ax.set_title("Training run summary")
    ax.grid(axis="y", alpha=0.25)

    for bar, score in zip(bars, scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{score:.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    fig.tight_layout()
    fig.savefig(run_dir / "final_metrics_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_run_summary(run_dir: Path, run_id: str, results: list[StageResult]) -> None:
    """Write root-level JSON and PNG summaries for a timestamped run."""
    summary = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "results": [asdict(result) for result in results],
    }
    write_json(run_dir / "run_summary.json", summary)
    generate_run_summary_image(run_dir, results)
