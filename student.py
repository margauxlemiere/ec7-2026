"""Lab 3 — what you write. Three functions, and the constants above them.

The robot has one camera and two things to recognise in the lane:

    the nine paper cylinders of markers.pdf      -> detect_marker
    the six road signs of road-signs.pdf         -> detect_sign

Each detector answers for its own family and `unknown` for everything else,
including the other family. `unknown` is a legitimate answer with a behaviour
attached, not a failure to answer.

Run `python engine.py --input files --set even --show` and open `out/` after
every change. A vision pipeline is debugged by looking at it.
"""
from __future__ import annotations

import numpy as np

from harness import show

# --- the constants that turn pixels into metres ----------------------------
# The focal length of the provided set is in data/camera.json. The real size of
# a target is not given to you: it is printed on the sheet, and the robot only
# knows a distance because someone went and measured it.
FOCAL_PX = 530.0
CYLINDER_HEIGHT_M = 0.250        # PLACEHOLDER — measure the sheet you printed
SIGN_DIAMETER_M = 0.120          # PLACEHOLDER — the sign sheet states its size

SHEETS = [f"sheet-{i}" for i in range(1, 10)]
SIGNS = ["sign-up", "sign-down", "sign-left", "sign-right", "sign-stop", "sign-free"]


def detect_marker(image: np.ndarray) -> tuple[str, float | None, float | None, float]:
    """Find a paper cylinder in `image`.

    image:  H x W x 3, RGB, uint8.
    return: (target, distance_m, bearing_deg, confidence)
            target      one of SHEETS, or "unknown"
            distance_m  from the apparent height, or None
            bearing_deg to the left is negative, or None
            confidence  0.0 to 1.0, produced by your pipeline, not invented

    Four steps, none of them learned:
      1. threshold in HSV, on hue rather than on RGB;
      2. morphology, to close the holes;
      3. contours, keeping the connected regions;
      4. a shape filter: aspect ratio, area, and the band pattern that names
         the sheet.

    Two sheets carry the same two colours in the other order, and four differ
    only by a stripe count. "red and green are present" names neither 3 nor 4.
    """
    show("input", image)
    return "unknown", None, None, 0.0


def detect_sign(image: np.ndarray) -> tuple[str, float | None, float | None, float]:
    """Find a road sign in `image`. Same signature, different thresholds.

    A sign is a shape with a colour: a red octagon, a blue disc, a white disc
    crossed by a black bar. Shape and colour name the family; only the
    pictogram separates `sign-up` from `sign-down`, which are the same disc and
    the same arrow, turned.

    Distance comes from the apparent diameter, the same way.
    """
    return "unknown", None, None, 0.0


def policy(detection: dict, state: dict) -> str:
    """What the robot does with one detection. Returns one action.

    detection: {"target", "distance_m", "bearing_deg", "confidence"}
    state:     {"speed", "person_ahead", "person_distance_m", "history"}
               `history` is every (detection, state, action) already played, so
               a stop sign that has left the field of view is still in reach.
    actions:   forward, slow, stop, turn_left, turn_right, report

    The four situations of scenario.md §4.2: a stop at 1.2 m, the same stop at
    0.3 m, the moment it leaves the field of view, and `unknown`. Three
    outcomes are forbidden, and `python engine.py --task policy` names them
    when you produce one.
    """
    return "stop"
