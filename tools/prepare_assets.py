#!/usr/bin/env python3
"""Fill `assets/` with the flat texture of every target of this lab.

    python tools/prepare_assets.py

Nine cylinder markers, from the nine pages of `markers.pdf`, and six road signs.
`assets/` is committed with this project, so you only need this script if you
change the target set, or if you want to check what it was built from.

It reads the course resources when they are next to it (the course repository),
and needs `pdftoppm`, from poppler-utils.
"""
from __future__ import annotations

import json
import pathlib
import subprocess
import tempfile

import cv2
import numpy as np
from PIL import Image

HERE = pathlib.Path(__file__).resolve().parent.parent
ASSETS = HERE / "assets"
CANDIDATES = [HERE / "resources", HERE.parent.parent / "resources"]
SIGNS = ["up", "down", "left", "right", "stop", "free"]


def resources() -> pathlib.Path:
    for path in CANDIDATES:
        if (path / "markers.pdf").exists():
            return path
    raise SystemExit("markers.pdf not found — looked in " +
                     ", ".join(str(p) for p in CANDIDATES))


def crop_to_ink(im: np.ndarray) -> np.ndarray:
    """The printed band, edge to edge: the white page margin is not the marker."""
    rgb = im[..., :3].astype(np.float64) / 255.0
    ink = (rgb.max(axis=2) < 0.93) | (rgb.std(axis=2) > 0.08)
    rows, cols = np.where(ink.any(axis=1))[0], np.where(ink.any(axis=0))[0]
    return im[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]


def main() -> int:
    src = resources()
    ASSETS.mkdir(exist_ok=True)

    dpi, ink_px = 150, []
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["pdftoppm", "-r", str(dpi), "-png", str(src / "markers.pdf"),
                        f"{tmp}/p"], check=True)
        pages = sorted(pathlib.Path(tmp).glob("p-*.png"))
        for n, page in enumerate(pages, start=1):
            im = cv2.imread(str(page), cv2.IMREAD_COLOR)
            out = crop_to_ink(im)
            ink_px.append(out.shape[0])
            alpha = np.full(out.shape[:2], 255, dtype=np.uint8)
            cv2.imwrite(str(ASSETS / f"sheet-{n}.png"), np.dstack([out, alpha]))
    # `free.png` is greyscale plus alpha where the others are colour plus alpha.
    # Everything downstream reads four channels, so they are normalised here.
    for name in SIGNS:
        Image.open(src / "road-signs" / f"{name}.png").convert("RGBA").save(
            ASSETS / f"sign-{name}.png")

    # What a detector measures is the printed band, not the sheet of paper: the
    # ink stops short of the edge, and the difference is 4 % of the distance.
    height_mm = round(max(ink_px) / dpi * 25.4, 1)
    (ASSETS / "geometry.json").write_text(json.dumps({
        "cylinder_ink_height_mm": height_mm,
        "cylinder_paper_height_mm": 210.0,
        "sign_diameter_mm": 150.0,
        "source": "markers.pdf printed on A4 landscape, road-signs.pdf on A4",
        "note": "the cylinder value is the printed band, measured on the PDF at "
                f"{dpi} dpi. Measure the one you printed before you trust it.",
    }, indent=2) + "\n", encoding="utf-8")

    print(f"{len(list(ASSETS.glob('*.png')))} assets in {ASSETS}, "
          f"printed band {height_mm} mm tall")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
