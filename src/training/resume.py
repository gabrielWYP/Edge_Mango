"""
What a training run needs to continue exactly where it stopped.

`best_model.pt` and `checkpoint_epoch<N>.pt` hold the weights validation ran on
(the EMA when `config.ema` is on) and nothing else: restarting from them would
lose the live weights, the AdamW moments, the position on the LR schedule and
the random stream, and train a different model than an uninterrupted run. The
trainer therefore also rewrites one `resume_state.pt` after every epoch.

Saves are atomic: the file is written next to its destination and then renamed,
so a run killed mid-save leaves the previous file intact, never a truncated one.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch

RESUME_STATE = "resume_state.pt"

# Config keys that may differ between an interrupted run and its continuation.
# Where the files go does not change the model being trained.
RESUME_IGNORED_CONFIG_KEYS = frozenset({"output_dir"})


def atomic_torch_save(obj, path: Path) -> None:
    """`torch.save` that never leaves a partial file at `path`."""
    path = Path(path)
    tmp = path.with_name(f"{path.name}.tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def capture_rng_state() -> dict:
    """Every random stream of the main process.

    The DataLoader draws the sampler order and each worker's seed from the
    global torch stream, so restoring it at an epoch boundary also restores
    the augmentations of the epochs to come.
    """
    return {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }


def restore_rng_state(state: dict) -> None:
    """Undo `capture_rng_state`. The CUDA streams need the same number of devices."""
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])


def resume_config_mismatches(saved: dict, current: dict) -> list[str]:
    """Names of the config keys that differ, sorted. Empty means same experiment."""
    keys = (set(saved) | set(current)) - RESUME_IGNORED_CONFIG_KEYS
    missing = object()
    return sorted(k for k in keys if saved.get(k, missing) != current.get(k, missing))
