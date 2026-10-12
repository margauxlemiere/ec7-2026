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


# The model signs of the platform, measured on 18 September 2026: an outer disc
# of 70 mm, and on the stop sign a red octagon of 57 mm across its flats, on
# white. The printed stop sign of road-signs.pdf fills its disc with the
# octagon, so the model one is rebuilt here.
SIGN_MM, STOP_MM = 70.0, 57.0


def colour_width(path: pathlib.Path, colour: str) -> float:
    """The width of the red or blue part of a sign, as a fraction of the sign."""
    im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    hsv = cv2.cvtColor(im[..., :3], cv2.COLOR_BGR2HSV)
    h, s = hsv[..., 0], hsv[..., 1]
    if colour == "red":
        mask = ((h < 10) | (h > 170)) & (s > 100)
    else:
        mask = (h > 95) & (h < 130) & (s > 100)
    cols = np.where(mask.any(axis=0))[0]
    return (cols[-1] - cols[0] + 1) / im.shape[1]


def model_stop() -> None:
    """The red octagon at 57 mm, centred on a white disc of 70 mm."""
    path = ASSETS / "sign-stop.png"
    printed = Image.open(path).convert("RGBA")
    side = printed.width
    scale = (STOP_MM / SIGN_MM) / colour_width(path, "red")
    inner = printed.resize((round(side * scale),) * 2, Image.LANCZOS)
    disc = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    arr = np.zeros((side, side), np.uint8)
    cv2.circle(arr, (side // 2, side // 2), side // 2 - 1, 255, -1, lineType=cv2.LINE_AA)
    mask = Image.fromarray(arr)
    disc.paste(Image.new("RGBA", (side, side), (246, 246, 244, 255)), (0, 0), mask)
    offset = (side - inner.width) // 2
    disc.alpha_composite(inner, (offset, offset))
    disc.save(path)


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
    model_stop()
    blue_mm = round(SIGN_MM * colour_width(ASSETS / "sign-up.png", "blue"), 1)

    # What a detector measures is the printed band, not the sheet of paper: the
    # ink stops short of the edge, and the difference is 4 % of the distance.
    height_mm = round(max(ink_px) / dpi * 25.4, 1)
    (ASSETS / "geometry.json").write_text(json.dumps({
        "cylinder_ink_height_mm": height_mm,
        "cylinder_paper_height_mm": 210.0,
        "sign_diameter_mm": SIGN_MM,
        "stop_octagon_mm": STOP_MM,
        "blue_disc_mm": blue_mm,
        "source": "markers.pdf printed on A4 landscape; the model signs of the "
                  "platform, measured on 18 September 2026",
        "note": "the cylinder value is the printed band, measured on the PDF at "
                f"{dpi} dpi. The signs: the outer disc, the red octagon across "
                "its flats, and the blue disc of the direction signs.",
    }, indent=2) + "\n", encoding="utf-8")

    print(f"{len(list(ASSETS.glob('*.png')))} assets in {ASSETS}, "
          f"printed band {height_mm} mm tall")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
