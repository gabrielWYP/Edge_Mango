#!/usr/bin/env python3
"""Turn an official Ultralytics checkpoint (e.g. yolov8n-seg.pt) into a plain state_dict.

`src/models/student/pretrained.py` reads a plain `{"model.<i>...": tensor}` file, so training never
needs the ultralytics package. The official `.pt` pickles whole `ultralytics` modules; this script
unpickles them as empty `nn.Module` stand-ins, without importing or running ultralytics code, and
keeps only the tensors (cast to float32).

    python scripts/convert_ultralytics_weights.py yolov8n-seg.pt data/weights/yolov8n-seg_state.pt

The yolov8n-seg file initialises every student arm: detect students take its backbone, neck and
box/class head and ignore the Proto and mask-coefficient layers; segment students take those too.
"""

from __future__ import annotations

import argparse
import pickle
import sys
import types
from pathlib import Path

import torch
from torch import nn

_FOREIGN_PREFIX = "ultralytics"


class _Placeholder(nn.Module):
    """Stand-in for any pickled ultralytics module: keeps its submodules, parameters and buffers."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)


class _Unpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        if module == _FOREIGN_PREFIX or module.startswith(_FOREIGN_PREFIX + "."):
            return _Placeholder
        return super().find_class(module, name)


_PICKLE = types.SimpleNamespace(Unpickler=_Unpickler, load=pickle.load, __name__="pickle")


def convert(src: str | Path, dst: str | Path) -> list[str]:
    """Write the float32 state_dict of `src`'s model to `dst`; return its keys."""
    checkpoint = torch.load(src, map_location="cpu", weights_only=False, pickle_module=_PICKLE)
    model = checkpoint["ema"] if checkpoint.get("ema") is not None else checkpoint["model"]
    state = {
        name: (tensor.float() if tensor.is_floating_point() else tensor).clone()
        for name, tensor in model.state_dict().items()
    }
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, dst)
    return list(state)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("src", type=Path, help="official Ultralytics .pt checkpoint")
    parser.add_argument("dst", type=Path, help="output plain state_dict (.pt)")
    args = parser.parse_args(argv)
    keys = convert(args.src, args.dst)
    print(f"{len(keys)} tensors -> {args.dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
