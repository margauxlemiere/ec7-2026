"""Small helpers the engine gives you. **Do not modify this file.**

You get one thing here: `show`, which writes an intermediate image to `out/`
when the engine runs with `--show`, and does nothing otherwise. A vision
pipeline is debugged by looking at it, and this is how you look.

    from harness import show

    show("mask", mask)              # any 2D or 3D numpy array
"""
from __future__ import annotations

import pathlib

import cv2
import numpy as np

_state = {"enabled": False, "dir": pathlib.Path("out"), "stem": "frame"}


def enable(directory: str = "out") -> None:
    _state["enabled"] = True
    _state["dir"] = pathlib.Path(directory)
    _state["dir"].mkdir(parents=True, exist_ok=True)


def frame(stem: str) -> None:
    _state["stem"] = stem


def show(tag: str, image) -> None:
    """Write `image` to out/<frame>-<tag>.png, when --show is on."""
    if not _state["enabled"]:
        return
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        lo, hi = float(np.min(arr)), float(np.max(arr))
        arr = np.zeros_like(arr, dtype=np.uint8) if hi - lo < 1e-9 else \
            ((arr - lo) / (hi - lo) * 255).astype(np.uint8)
    if arr.ndim == 3 and arr.shape[2] == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(_state["dir"] / f"{_state['stem']}-{tag}.png"), arr)
