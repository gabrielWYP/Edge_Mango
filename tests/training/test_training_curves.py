"""Loss plots show losses only; telemetry stays in the CSV.

Verified defect: `LossHistory.extra_losses` also carries the D-G counters and the
D-F gradient norms, and both loss plots drew every entry. `steps_taken` (about
300) set the y axis, so the real losses lay flat near zero in `loss_curves.png`.
"""

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib.figure import Figure

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.training.metrics import LossHistory, generate_training_curves

LOSSES = {"total_loss", "cls_loss", "box_loss", "kd_loss"}
TELEMETRY = {"oom_skipped", "nan_skipped", "steps_taken", "grad_norm_neck"}


def _history() -> LossHistory:
    history = LossHistory()
    for epoch in (1, 2):
        history.update(
            cls=1.0,
            box=0.5,
            total=4.25,
            epoch=epoch,
            per_class_ap_50={0: 1.0, 1: 0.2},
            extra_losses={
                "kd_loss": 0.3,
                "oom_skipped": 0.0,
                "nan_skipped": 0.0,
                "steps_taken": 298.0,
                "grad_norm_neck": 1.5,
            },
        )
    return history


def _loss_axes_labels(monkeypatch, output_dir: Path) -> dict[str, set[str]]:
    """Run the public report and return the loss-axes line labels per saved figure."""
    seen: dict[str, set[str]] = {}
    real_savefig = Figure.savefig

    def recording_savefig(fig, fname, *args, **kwargs):
        for ax in fig.axes:
            if ax.get_title() == "Training losses":
                seen[Path(fname).name] = {line.get_label() for line in ax.get_lines()}
        return real_savefig(fig, fname, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", recording_savefig)
    generate_training_curves(_history(), output_dir)
    return seen


def test_loss_plots_draw_only_losses(tmp_path, monkeypatch):
    seen = _loss_axes_labels(monkeypatch, tmp_path)

    assert seen == {"loss_curves.png": LOSSES, "training_curves.png": LOSSES}
    assert (tmp_path / "loss_curves.png").stat().st_size > 0
    assert (tmp_path / "training_curves.png").stat().st_size > 0


def test_csv_keeps_telemetry_columns(tmp_path):
    generate_training_curves(_history(), tmp_path)

    with open(tmp_path / "metrics_history.csv", newline="") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 2
    assert LOSSES | TELEMETRY <= set(rows[0])
    assert float(rows[1]["steps_taken"]) == 298.0
    assert float(rows[1]["grad_norm_neck"]) == 1.5
