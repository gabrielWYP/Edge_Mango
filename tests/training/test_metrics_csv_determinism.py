"""`metrics_history.csv` must lay its columns out the same way every run.

`LossHistory.update` seeded `self.extra_losses` by iterating
`set(self.extra_losses) | set(extra_losses)`. Set iteration order for
strings depends on their hashes, and Python randomises string hashing per
process (PYTHONHASHSEED), so the insertion order of `extra_losses` — and
therefore the `fieldnames` built from `.keys()` in `_write_metrics_csv` —
differed between runs of identical code.

Observed on two real runs of the same experiment: `steps_taken` is column
16 in `20260831T223558592803Z` and column 6 in `20260831T224505981163Z`.
Header-keyed readers cope, but any positional comparison across runs
silently yields garbage — during this investigation it made a frozen stem
look like a trained one.

The order also has to be *meaningful*, not merely stable: the counters and
per-module gradient norms are read side by side, so they must keep the
order the trainer reports them in rather than being alphabetised apart.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.training.metrics import LossHistory, _write_metrics_csv

REPO_ROOT = Path(__file__).parent.parent.parent

# The order the trainer reports these in: D-G counters, then D-F per-module
# gradient norms walking stem -> stages -> neck -> head levels.
TRAINER_ORDER = [
    "oom_skipped",
    "nan_skipped",
    "steps_taken",
    "grad_norm_backbone_stem",
    "grad_norm_backbone_stage0",
    "grad_norm_backbone_stage1",
    "grad_norm_backbone_stage2",
    "grad_norm_backbone_stage3",
    "grad_norm_neck",
    "grad_norm_head_level0",
    "grad_norm_head_level1",
    "grad_norm_head_level2",
]

_HEADER_PROBE = textwrap.dedent(
    """
    import pathlib, sys, tempfile
    sys.path.insert(0, {root!r})
    from src.training.metrics import LossHistory, _write_metrics_csv

    history = LossHistory()
    history.update(
        cls=1.0, box=1.0, total=1.0, map50=0.1, map_50_95=0.1,
        extra_losses={{key: 1.0 for key in {order!r}}},
    )
    out = pathlib.Path(tempfile.mkdtemp())
    _write_metrics_csv(history, out)
    print((out / "metrics_history.csv").read_text().splitlines()[0])
    """
)


def _header_under_hash_seed(seed: str) -> str:
    env = {**os.environ, "PYTHONHASHSEED": seed}
    script = _HEADER_PROBE.format(root=str(REPO_ROOT), order=TRAINER_ORDER)
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=True,
        cwd=REPO_ROOT,
        env=env,
    )
    return completed.stdout.strip()


def test_column_order_is_identical_across_hash_seeds():
    """The defect reproduced: four seeds used to give four column orders."""
    headers = {_header_under_hash_seed(seed) for seed in ("0", "1", "2", "3")}

    assert len(headers) == 1, (
        "metrics_history.csv column order depends on PYTHONHASHSEED:\n"
        + "\n".join(sorted(headers))
    )


def test_extra_columns_keep_the_order_the_trainer_reports_them_in(tmp_path):
    history = LossHistory()
    history.update(
        cls=1.0,
        box=1.0,
        total=1.0,
        map50=0.1,
        map_50_95=0.1,
        extra_losses={key: 1.0 for key in TRAINER_ORDER},
    )

    _write_metrics_csv(history, tmp_path)
    header = (tmp_path / "metrics_history.csv").read_text().splitlines()[0].split(",")

    assert header[5 : 5 + len(TRAINER_ORDER)] == TRAINER_ORDER


def test_a_key_appearing_late_is_appended_not_reordered(tmp_path):
    """A metric that only exists in phase 2 must not shuffle phase 1's columns."""
    history = LossHistory()
    history.update(
        cls=1.0, box=1.0, total=1.0, map50=0.1, map_50_95=0.1,
        extra_losses={"steps_taken": 7.0, "oom_skipped": 0.0},
    )
    history.update(
        cls=1.0, box=1.0, total=1.0, map50=0.2, map_50_95=0.1,
        extra_losses={"steps_taken": 7.0, "oom_skipped": 0.0, "grad_norm_backbone_stem": 3.0},
    )

    _write_metrics_csv(history, tmp_path)
    header = (tmp_path / "metrics_history.csv").read_text().splitlines()[0].split(",")

    assert header[5:8] == ["steps_taken", "oom_skipped", "grad_norm_backbone_stem"]
