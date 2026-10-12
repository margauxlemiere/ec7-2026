#!/usr/bin/env python3
"""Add the robot's photographs to data/labels.csv, as the set `robot`.

    python tools/add_robot_photos.py

Run it after tools/render_scenes.py, which writes labels.csv from scratch.

A photograph says what it shows in its file name, in data/robot/:

    sign-stop_060cm.png          one target, 60 cm from the camera, on its axis
    sign-left_090cm_r15.png      the same at 90 cm, 15 degrees to the right
    sheet-3_120cm_l10.png        a marker at 1.2 m, 10 degrees to the left
    none_01.png                  no target at all

The distance is measured with a tape, from the lens to the front of the target.
Other files in data/robot/ are left alone: the camera survey of 20 September.
"""
from __future__ import annotations

import csv
import pathlib
import re

HERE = pathlib.Path(__file__).resolve().parent.parent
DATA = HERE / "data"
FIELDS = ["file", "condition", "target", "family", "distance_m", "bearing_deg",
          "yaw_deg", "occlusion", "blur", "shadow"]
NAME = re.compile(r"^(?P<target>sheet-[1-9]|sign-(?:up|down|left|right|stop|free))"
                  r"_(?P<cm>\d+)cm(?:_(?P<side>[lr])(?P<deg>\d+))?(?:_.*)?$")
NONE = re.compile(r"^none(?:_.*)?$")


def row_of(path: pathlib.Path) -> dict | None:
    stem = path.stem
    base = {"file": f"robot/{path.name}", "condition": "robot", "target": "",
            "family": "", "distance_m": "", "bearing_deg": "", "yaw_deg": "",
            "occlusion": 0, "blur": 0, "shadow": 0}
    if NONE.match(stem):
        return base
    m = NAME.match(stem)
    if not m:
        return None
    bearing = 0.0
    if m["side"]:
        bearing = float(m["deg"]) * (-1 if m["side"] == "l" else 1)
    target = m["target"]
    return base | {"target": target,
                   "family": "cylinder" if target.startswith("sheet") else "sign",
                   "distance_m": int(m["cm"]) / 100, "bearing_deg": bearing}


def main() -> int:
    labels = DATA / "labels.csv"
    rows = [r for r in csv.DictReader(labels.open(encoding="utf-8"))
            if r["condition"] != "robot"]
    photos = sorted(p for p in (DATA / "robot").iterdir()
                    if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    added = [r for r in map(row_of, photos) if r]
    with labels.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows + added)
    print(f"{len(added)} robot photographs added to {labels.relative_to(HERE)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
