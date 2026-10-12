"""Step 3 — The signs, by colour and by shape.

You write one function here: `detect_sign`. It finds a road sign in an image,
and says which one, how far, and in which direction. Run it with:

    python engine.py --input files --set even --show

The engine scores you twice. `family_accuracy`: stop, direction or free (end
of restrictions). `accuracy`: the exact name, which also separates `sign-up`
from `sign-down`.

Read this file from top to bottom. Do what each TODO asks.
"""

# Type hints, as in step 2.
from __future__ import annotations

# This line imports everything from step2.py: detect_marker, your two renamed
# functions, FOCAL_PX, show… You do not copy them again.
from step2 import *

# Already imported by step2.py. We import them again so that your editor knows
# what `math`, `cv2` and `np` are in this file.
import math
import cv2
import numpy as np


# -----------------------------------------------------------------------------
# The sizes that turn pixels into metres
# -----------------------------------------------------------------------------

# The signs are the model signs of the platform: a disc on a post. The diameter
# of the disc, in metres, is in assets/geometry.json: "sign_diameter_mm".
SIGN_DIAMETER_M = None  # TODO: a number, in metres

# What your mask sees is smaller than the sign. Open assets/sign-stop.png and
# assets/sign-up.png: the red octagon sits on a white disc, and the blue disc
# stops before a thin white rim. assets/geometry.json gives both widths:
# "stop_octagon_mm" and "blue_disc_mm". Each ratio is that width divided by
# "sign_diameter_mm". Without them, every distance to a sign is too long.
RED_OF_SHEET = None   # TODO: a number below 1. You can write 57 / 70.
BLUE_OF_SHEET = None  # TODO: a number a little below 1


# -----------------------------------------------------------------------------
# The shape test
# -----------------------------------------------------------------------------
# A sign is a shape with a colour: a red octagon, a blue disc, a white disc
# crossed by a black bar. Slide 16 ("Shape names the family, not the sign"):
# the fraction of its box that a shape fills separates the three things a red
# mask finds. A disc fills 0.79 of its box, an octagon 0.83, and a red crate
# fills all of it: 1.00.

# TODO: check the 0.79 of the disc. A disc of radius r has an area of
# pi * r ** 2, and its box is a square of side 2 * r. Which fraction of the
# box does it fill? Then choose the two bounds of the test: both signs must
# pass, and the crate must not. At 30 px wide, a mask is ragged: keep a
# margin. Try 0.68 and 0.93.
FILL_MIN = None  # TODO
FILL_MAX = None  # TODO

# The "circularity" of a shape is 1.0 for a disc, and lower for any other
# shape: 0.95 for an octagon, 0.79 for a square, near 0 for a long thin shape.
# Below this value, the region is not a sign.
CIRCULARITY_MIN = None  # TODO: a number between 0 and 1. Try 0.72.


def shape_of(contour: np.ndarray, w: int, height: int) -> tuple[float, float]:
    """How much of its box the region fills, and its circularity."""
    # The area inside the contour, and the length of the contour, in pixels.
    # `True` says that the contour is closed.
    # https://docs.opencv.org/4.x/dd/d49/tutorial_py_contour_features.html
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    # The fraction of the box. max(1, …) avoids a division by zero.
    fill = area / max(1, w * height)
    # TODO: the circularity, from `area` and `perimeter`. It is
    #   k * area / perimeter ** 2,
    # with the number k that gives exactly 1 for a disc. A disc of radius r
    # has an area of pi * r ** 2 and a perimeter of 2 * pi * r: find k.
    # Write max(1.0, perimeter ** 2) instead of perimeter ** 2, to avoid a
    # division by zero. `math.pi` is pi.
    circularity = None
    return fill, circularity


# -----------------------------------------------------------------------------
# The arrow: the name inside the family
# -----------------------------------------------------------------------------

# TODO: this function is given. Read it line by line. Then give it a name, in
# two or three words joined by "_". Rename it here, and in `detect_sign`.
def useful_function_3_to_rename(contour: np.ndarray, mask: np.ndarray,
                                x: int, y: int, w: int, height: int) -> np.ndarray:
    # An image of zeros, as large as the mask. np.zeros(shape, type).
    # https://numpy.org/doc/stable/reference/generated/numpy.zeros.html
    filled = np.zeros(mask.shape, np.uint8)
    # Paint the inside of the contour with 1: the whole disc, arrow included.
    # [contour] is a list of one contour; -1 means "every contour of the
    # list"; 1 is the value painted; cv2.FILLED fills the inside.
    # https://docs.opencv.org/4.x/d4/d73/tutorial_py_contours_begin.html
    cv2.drawContours(filled, [contour], -1, 1, cv2.FILLED)
    # 1 - mask is 1 where the mask is 0: where the pixel is not blue.
    # `&` keeps the pixels that are inside the disc AND not blue.
    hole = filled & (1 - mask)
    # Cut the box of the sign out of it, as True and False.
    return hole[y:y + height, x:x + w].astype(bool)


def arrow_direction(white: np.ndarray) -> str | None:
    """"up", "down", "left" or "right", or None if there is no arrow.

    The head of the arrow is its wide part; the shaft is the thin part.
    """
    show("sign-arrow", white.astype(np.uint8) * 255)
    # How many white pixels in each row (axis=1: along the row), and in each
    # column (axis=0: down the column).
    rows, cols = white.sum(axis=1), white.sum(axis=0)
    if white.sum() < 20:
        return None
    # An arrow is long in its own direction. np.count_nonzero counts the
    # True values: here, the rows (or columns) with some white in them.
    # https://numpy.org/doc/stable/reference/generated/numpy.count_nonzero.html
    vertical = np.count_nonzero(rows > 0) >= np.count_nonzero(cols > 0)
    # `a if condition else b` is a value: a when the condition is True, b
    # otherwise. The profile is the width of the white along the arrow.
    profile = rows if vertical else cols
    # np.where(condition)[0] is the list of the positions where the condition
    # is True: the rows (or columns) that hold some arrow.
    # https://numpy.org/doc/stable/reference/generated/numpy.where.html
    live = np.where(profile > 0)[0]
    if len(live) < 5:
        return None
    # The head: the positions where the arrow is at least 80 % of its widest.
    head = live[profile[live] >= 0.80 * profile[live].max()]
    # The middle of the arrow, between its first and last position.
    middle = (live[0] + live[-1]) / 2

    # TODO: the direction, in words. Rows are counted from the top of the
    # image, and columns from the left.
    #   - vertical arrow: "up" if the head (its mean position, head.mean())
    #     is above the middle, "down" otherwise;
    #   - horizontal arrow: "left" if the head is left of the middle, "right"
    #     otherwise.
    # The form `a if condition else b`, above, writes each case in one line.
    return None


# -----------------------------------------------------------------------------
# The white disc crossed by a black bar
# -----------------------------------------------------------------------------

def detect_free(hsv: np.ndarray, image: np.ndarray) -> tuple[str, float | None, float | None, float]:
    """The end of restrictions: pale, bright, with a dark bar across it."""
    s, v = hsv[..., 1], hsv[..., 2]
    # TODO: `pale` is True where the pixel is white: little colour (s below
    # 70) and bright (v above 170). Write it as in step 2, with `&`, then
    # turn it into 0/1 with .astype(np.uint8).
    pale = None
    pale = cv2.morphologyEx(pale, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    contours, _ = cv2.findContours(pale, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # The three largest regions, largest first. sorted(list, key=f,
    # reverse=True) sorts from the largest f to the smallest; `[:3]` keeps the
    # first three. https://docs.python.org/3/library/functions.html#sorted
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:3]:
        x, y, w, height = cv2.boundingRect(contour)
        # Too small, or not round enough in its box: the next region.
        if height < 20 or w < 16 or not 0.70 <= w / height <= 1.45:
            continue
        # The brightness inside the box, and where it is dark: the bar.
        patch = v[y:y + height, x:x + w]
        dark = patch < 110
        # TODO: skip this region (`continue`) unless the dark part is between
        # 5 % and 32 % of the box: a bar, and nothing else. `dark.mean()` is
        # that fraction.

        # The bar goes across, at an angle: its pixels are spread from left
        # to right. np.nonzero gives the rows and the columns of the True
        # pixels; np.std(xs) is how spread the columns are.
        # https://numpy.org/doc/stable/reference/generated/numpy.nonzero.html
        # https://numpy.org/doc/stable/reference/generated/numpy.std.html
        ys, xs = np.nonzero(dark)
        if len(xs) < 12 or np.std(xs) < 0.10 * w:    # a bar, not a corner
            continue
        # TODO: the distance, as in step 2. The white disc is the whole sign:
        # use SIGN_DIAMETER_M, with no ratio.
        distance = None
        # TODO: the bearing: the same two lines as in step 2.
        bearing = None
        return "sign-free", distance, bearing, 0.58
    return "unknown", None, None, 0.0


# -----------------------------------------------------------------------------
# Your function
# -----------------------------------------------------------------------------

def detect_sign(image: np.ndarray) -> tuple[str, float | None, float | None, float]:
    # Same input and same output as detect_marker. Target: one of "sign-up",
    # "sign-down", "sign-left", "sign-right", "sign-stop", "sign-free", or
    # "unknown". A marker is "unknown" here: it is not a sign.

    # 1. TODO: the image in HSV: the same line as in step 2.
    hsv = None
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    # 2. Two colour masks. Red: hue near 0 or near 180, strong, not too dark.
    red = ((h < 10) | (h > 170)) & (s > 100) & (v > 55)
    #    TODO: blue. Blue is at 240 degrees on the colour wheel: 120 in
    #    OpenCV. In words: hue above 95 and below 130, s above 100, v above
    #    40. On an array, `95 < h < 130` does not work: write two comparisons
    #    joined by `&`.
    blue = None

    # 3. Each mask in turn. The loop gives `name` and `mask` at each turn:
    #    first ("red", red), then ("blue", blue).
    for name, mask in (("red", red), ("blue", blue)):
        # A small square structuring element, 5 x 5: it closes small holes in
        # every direction, not the tall gap of a marker.
        mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE,
                                np.ones((5, 5), np.uint8))
        # An f-string: {name} is replaced by the value of name.
        show(f"sign-{name}", mask * 255)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # The three largest regions, largest first (as in detect_free).
        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:3]:
            x, y, w, height = cv2.boundingRect(contour)
            # Too small, or too far from a square box: the next region.
            if height < 18 or w < 14 or not 0.62 <= w / height <= 1.55:
                continue
            fill, circularity = shape_of(contour, w, height)
            # TODO: the shape test. Skip this region (`continue`) if `fill` is
            # not between FILL_MIN and FILL_MAX, or if `circularity` is below
            # CIRCULARITY_MIN.

            # 4. Distance and bearing. The size the mask sees: the diameter
            #    of the sign times the ratio of its colour.
            seen = SIGN_DIAMETER_M * (RED_OF_SHEET if name == "red" else BLUE_OF_SHEET)
            # TODO: the distance, as in step 2, with `seen` as the real size.
            distance = None
            # TODO: the bearing: the same two lines as in step 2.
            bearing = None

            # 5. The name. A red octagon is a stop sign.
            if name == "red":
                return "sign-stop", distance, bearing, float(np.clip(circularity, 0, 1))
            # A blue disc: the arrow in it gives the name.
            arrow = arrow_direction(useful_function_3_to_rename(contour, mask, x, y, w, height))
            if arrow is None:
                continue
            return f"sign-{arrow}", distance, bearing, 0.70

    # 6. No red and no blue sign: maybe the white disc.
    return detect_free(hsv, image)
