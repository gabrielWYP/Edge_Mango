"""A machine profile's `batch_size` must not change what gets measured.

D-H (fusion-redesign design.md) makes `batch_size` a *machine* knob: a
profile may set it freely, and `grad_accum_steps` is derived so that
`effective_batch` — the experiment constant — stays fixed. Two runs are
declared comparable when `experiment_sha256` and `effective_batch` match.

That guarantee was incomplete. Both entry points built the training loader
with `drop_last=True`, so the physical batch size decided how much of the
training set the model ever saw, and how many optimizer steps an epoch
contained:

    50 train images, effective_batch=8
      batch 2 -> 25 micro-batches, 50 images seen, ceil(25/4) = 7 steps
      batch 4 -> 12 micro-batches, 48 images seen, ceil(12/2) = 6 steps

Verified against two real runs of the same experiment at the same
`effective_batch`: `20260831T224505981163Z` (batch 2) recorded
`steps_taken=7` per epoch, `20260831T223558592803Z` (batch 4) recorded 6.
Because early stopping counts *epochs*, that 7-vs-6 compounds over a run
into a materially different training budget while both checkpoints claim
the same `effective_batch`.

`build_dataloader` already defaults to `drop_last=False`; the entry points
overrode it. With the default restored, the accumulation loop's end-of-epoch
flush turns the leftover micro-batch into a final (smaller) optimizer step,
so every legal batch size sees the whole dataset in the same number of steps.
"""

from __future__ import annotations

import ast
import math
import sys
from pathlib import Path

import pytest
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.training.dataset import build_dataloader
from src.training.machine import derive_grad_accum_steps

TRAIN_IMAGES = 50
EFFECTIVE_BATCH = 8
# `derive_grad_accum_steps` rejects a batch size that does not divide
# `effective_batch`, so these are exactly the legal batch sizes.
LEGAL_BATCH_SIZES = [1, 2, 4, 8]

ENTRY_POINTS = [
    Path("src/training/train.py"),
    Path("src/training/kd_train.py"),
]


class _CountingDataset(Dataset):
    """Minimal dataset — this suite counts samples, it does not train."""

    def __init__(self, size: int) -> None:
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> int:
        return index


def _loader(batch_size: int, *, drop_last: bool):
    return build_dataloader(
        _CountingDataset(TRAIN_IMAGES),
        batch_size=batch_size,
        num_workers=0,
        collate_fn=list,
        pin_memory=False,
        drop_last=drop_last,
    )


def _steps_per_epoch(num_micro_batches: int, accum: int) -> int:
    """Optimizer steps an epoch performs, including the end-of-epoch flush."""
    return math.ceil(num_micro_batches / accum)


class TestImagesSeenIsIndependentOfBatchSize:
    """The machine knob must not change the training data."""

    @pytest.mark.parametrize("batch_size", LEGAL_BATCH_SIZES)
    def test_every_legal_batch_size_sees_the_whole_dataset(self, batch_size):
        loader = _loader(batch_size, drop_last=False)

        seen = sum(len(batch) for batch in loader)

        assert seen == TRAIN_IMAGES

    def test_drop_last_is_what_broke_the_invariant(self):
        """Regression witness: the exact asymmetry observed in production."""
        seen_at_2 = sum(len(b) for b in _loader(2, drop_last=True))
        seen_at_4 = sum(len(b) for b in _loader(4, drop_last=True))

        assert seen_at_2 == 50
        assert seen_at_4 == 48


class TestOptimizerStepBudgetIsIndependentOfBatchSize:
    """Equal `effective_batch` must mean an equal number of updates."""

    def test_steps_per_epoch_are_identical_across_legal_batch_sizes(self):
        steps = {}
        for batch_size in LEGAL_BATCH_SIZES:
            accum = derive_grad_accum_steps(EFFECTIVE_BATCH, batch_size)
            num_micro = len(_loader(batch_size, drop_last=False))
            steps[batch_size] = _steps_per_epoch(num_micro, accum)

        assert len(set(steps.values())) == 1, steps

    def test_drop_last_made_the_budget_depend_on_the_machine(self):
        """The 7-vs-6 recorded by the two real runs."""
        steps = {}
        for batch_size in (2, 4):
            accum = derive_grad_accum_steps(EFFECTIVE_BATCH, batch_size)
            num_micro = len(_loader(batch_size, drop_last=True))
            steps[batch_size] = _steps_per_epoch(num_micro, accum)

        assert steps == {2: 7, 4: 6}


class TestEntryPointsDoNotReEnableDropLast:
    """One entry point re-enabling it silently restores the whole defect."""

    @pytest.mark.parametrize("module_path", ENTRY_POINTS, ids=lambda p: p.name)
    def test_no_build_dataloader_call_passes_drop_last_true(self, module_path):
        tree = ast.parse(module_path.read_text(encoding="utf-8"))

        offenders = [
            node.lineno
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and getattr(node.func, "id", None) == "build_dataloader"
            for keyword in node.keywords
            if keyword.arg == "drop_last"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value is True
        ]

        assert offenders == [], (
            f"{module_path} passes drop_last=True at line(s) {offenders}; "
            "that makes batch_size change the images seen and the optimizer "
            "step count, breaking D-H comparability."
        )
