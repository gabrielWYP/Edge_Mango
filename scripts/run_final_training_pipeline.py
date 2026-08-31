#!/usr/bin/env python3
"""Run the final three-stage model training pipeline.

This orchestrator trains:

1. MasterModel (RGB+NIR teacher)
2. StudentModel baseline (RGB-only)
3. Distilled StudentModel (RGB-only student with frozen master teacher)

Each stage writes to a temporary directory first and is moved into its final
timestamped directory only after the stage finishes successfully. The final
layout is:

    checkpoints/final_runs/<run_id>/
      maestro/best_model_<run_id>_mAP<score>/
      estudiante/best_model_<run_id>_mAP<score>/
      destilado/best_model_<run_id>_mAP<score>/

The stage folders contain weights, metrics CSV, PNG training curves, TensorBoard
logs, and a ``stage_summary.json`` file with the command and best validation mAP.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training.run_artifacts import (
    StageResult,
    finalize_stage,
    generate_run_summary_image,
    prepare_atomic_run,
    publish_atomic_run,
    release_atomic_run,
    utc_timestamp,
    write_json,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the final training pipeline."""
    parser = argparse.ArgumentParser(
        description="Train master, student baseline, and distilled student."
    )
    parser.add_argument(
        "--master-config",
        default="configs/training_mango.yaml",
        help="YAML config for MasterModel training.",
    )
    parser.add_argument(
        "--student-config",
        default="configs/training_student.yaml",
        help="YAML config for baseline StudentModel training.",
    )
    parser.add_argument(
        "--kd-config",
        default="configs/kd_training.yaml",
        help="YAML config for knowledge distillation training.",
    )
    parser.add_argument(
        "--machine",
        default=None,
        help=(
            "Path to a machine-profile YAML (configs/machines/*.yaml), forwarded "
            "to all three stages. A profile sets only device/batch_size/"
            "num_workers/pin_memory/precision, so it changes HOW the run "
            "executes on this hardware, never the experiment being measured: "
            "effective_batch is held by the stage config and grad_accum_steps "
            "is derived from it (fusion-redesign D-H)."
        ),
    )
    parser.add_argument(
        "--output-root",
        default="checkpoints/final_runs",
        help="Root directory for timestamped final runs.",
    )
    parser.add_argument(
        "--timestamp",
        default=None,
        help="Optional run ID. Defaults to a UTC timestamp with microseconds.",
    )
    parser.add_argument(
        "--skip-master",
        action="store_true",
        help="Skip master training and use --teacher-checkpoint for KD.",
    )
    parser.add_argument(
        "--skip-student",
        action="store_true",
        help="Skip baseline student training.",
    )
    parser.add_argument(
        "--skip-kd",
        action="store_true",
        help="Skip distilled student training.",
    )
    parser.add_argument(
        "--teacher-checkpoint",
        default=None,
        help="Teacher checkpoint to use when --skip-master is set.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable for subprocess training commands.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands and planned folders without running training.",
    )
    return parser.parse_args()


def run_command(command: list[str], dry_run: bool) -> None:
    """Run a subprocess command with logging.

    Args:
        command: Command and arguments to execute.
        dry_run: If true, only log the command.
    """
    logger.info("Running: %s", " ".join(command))
    if dry_run:
        return
    subprocess.run(command, check=True)


def run_training_stage(
    *,
    python: str,
    module: str,
    config_path: str,
    output_dir: Path,
    dry_run: bool,
    overrides: list[str] | None = None,
    model: str | None = None,
    machine: str | None = None,
) -> list[str]:
    """Execute a training module with output-dir override.

    `--machine` is placed before `--override` because both entry points apply
    the profile first and the overrides second; keeping the recorded command
    in that order means the stored command reproduces the run exactly.
    """
    command = [python, "-m", module, "--config", config_path]
    if model is not None:
        command.extend(["--model", model])
    if machine is not None:
        command.extend(["--machine", machine])
    effective_overrides = [f"output_dir={output_dir}", *(overrides or [])]
    command.extend(["--override", *effective_overrides])
    run_command(command, dry_run)
    return command


def main() -> int:
    """Run the final pipeline end to end."""
    args = parse_args()
    run_id = args.timestamp or utc_timestamp()
    run_dir = Path(args.output_root) / run_id
    run_plan = None
    working_dir = run_dir
    if not args.dry_run:
        run_plan = prepare_atomic_run(args.output_root, run_id)
        working_dir = run_plan.staging_dir

    logger.info("Final run id: %s", run_id)
    logger.info("Final run directory: %s", run_dir)
    if run_plan is not None:
        logger.info("Private staging directory: %s", working_dir)

    results: list[StageResult] = []
    teacher_checkpoint = args.teacher_checkpoint

    try:
        if not args.skip_master:
            tmp_dir = working_dir / ".stage_tmp" / "maestro"
            command = run_training_stage(
                python=args.python,
                module="src.training.train",
                config_path=args.master_config,
                output_dir=tmp_dir,
                dry_run=args.dry_run,
                model="master",
                machine=args.machine,
            )
            master = finalize_stage(
                tmp_dir=tmp_dir,
                final_parent=working_dir / "maestro",
                run_id=run_id,
                command=command,
                stage_key="maestro",
                stage_label="Maestro",
                dry_run=args.dry_run,
            )
            results.append(master)
            teacher_checkpoint = master.checkpoint
        elif not teacher_checkpoint:
            raise ValueError("--teacher-checkpoint is required when --skip-master is set")

        if not args.skip_student:
            tmp_dir = working_dir / ".stage_tmp" / "estudiante"
            command = run_training_stage(
                python=args.python,
                module="src.training.train",
                config_path=args.student_config,
                output_dir=tmp_dir,
                dry_run=args.dry_run,
                model="student",
                machine=args.machine,
            )
            results.append(
                finalize_stage(
                    tmp_dir=tmp_dir,
                    final_parent=working_dir / "estudiante",
                    run_id=run_id,
                    command=command,
                    stage_key="estudiante",
                    stage_label="Estudiante",
                    dry_run=args.dry_run,
                )
            )

        if not args.skip_kd:
            tmp_dir = working_dir / ".stage_tmp" / "destilado"
            command = run_training_stage(
                python=args.python,
                module="src.training.kd_train",
                config_path=args.kd_config,
                output_dir=tmp_dir,
                dry_run=args.dry_run,
                overrides=[f"teacher_checkpoint={teacher_checkpoint}"],
                machine=args.machine,
            )
            results.append(
                finalize_stage(
                    tmp_dir=tmp_dir,
                    final_parent=working_dir / "destilado",
                    run_id=run_id,
                    command=command,
                    stage_key="destilado",
                    stage_label="Destilado",
                    dry_run=args.dry_run,
                )
            )

        summary = {
            "run_id": run_id,
            "run_dir": str(run_dir),
            "results": [result.__dict__ for result in results],
        }
        if not args.dry_run:
            write_json(working_dir / "run_summary.json", summary)
            generate_run_summary_image(working_dir, results)
            publish_atomic_run(run_plan)
        else:
            logger.info("Dry-run summary: %s", summary)
    finally:
        if run_plan is not None:
            release_atomic_run(run_plan)

    logger.info("Final training pipeline complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
