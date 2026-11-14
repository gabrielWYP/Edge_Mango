"""Thread caps for the training processes.

The training host has few cores and runs several trainings at once, each with torch, OpenMP
and OpenCV pools: every process and DataLoader worker wants all the cores, and they only
fight over them. Workers do single-threaded work per item, and the main process drives the GPU.
"""

from __future__ import annotations

from typing import Callable

import cv2
import torch

MAIN_PROCESS_THREADS = 2


def limit_worker_threads(worker_id: int) -> None:
    """DataLoader `worker_init_fn`: one torch and one OpenCV thread in the worker."""
    torch.set_num_threads(1)
    cv2.setNumThreads(1)


def cap_main_threads(max_threads: int = MAIN_PROCESS_THREADS) -> None:
    """Lower torch's intra-op threads of this process to `max_threads` (never raises them)."""
    if torch.get_num_threads() > max_threads:
        torch.set_num_threads(max_threads)


def compose_worker_init(*fns: Callable[[int], None] | None) -> Callable[[int], None]:
    """One `worker_init_fn` that runs each given function (None skipped) in order."""
    active = [fn for fn in fns if fn is not None]

    def init(worker_id: int) -> None:
        for fn in active:
            fn(worker_id)

    return init
