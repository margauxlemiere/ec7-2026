"""Step 2 — The markers.

You write one function here: `detect_marker`. It finds a paper cylinder in an
image, and says which of the nine sheets it is, how far, and in which
direction. Run it with:

    python engine.py --input files --set even --show

Then open `out/`: your intermediate images are there. Open them again after
every change. A vision pipeline is debugged by looking.

Read this file from top to bottom. Do what each TODO asks.
"""

# The type hints below (`image: np.ndarray`, `-> str`…) say what each function
# takes and returns. Python does not check them, but your editor uses them to
# help you. This line lets us write them in the modern way.
from __future__ import annotations

# math has the functions of trigonometry: math.atan, math.degrees.
# https://docs.python.org/3/library/math.html
import math

# OpenCV is the library for images. Its Python name is cv2. We use it to change
# the colour space, to close holes in a mask, and to find regions. Its Python
# tutorials are short and have pictures:
# https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html
import cv2

# numpy is the library for arrays of numbers. An image is an array of
# H x W x 3 numbers: H rows, W columns, 3 colours (red, green, blue), each
# between 0 and 255. Everybody imports it under the short name `np`.
import numpy as np

# `show(name, array)` writes an image to out/ when the engine runs with
# --show, and does nothing otherwise. It is in harness.py.
from harness import show


# -----------------------------------------------------------------------------
# The constants that turn pixels into metres
# -----------------------------------------------------------------------------

# The focal length of the camera, in pixels. The images of data/ were rendered
# with a known camera. data/camera.json describes it: open it in your editor.
FOCAL_PX = None  # TODO: the value of "focal_px" in data/camera.json

# The real height of the marker: from the top of its top black band to the
# bottom of its bottom black band, in metres. Slide 9 of the lecture ("Distance
# from apparent size") needs it: d = f * H_real / h_pixels.
# assets/geometry.json gives it in millimetres: "cylinder_ink_height_mm".
# Open the file in your editor. You can write a division: 201.5 / 1000.
CYLINDER_HEIGHT_M = None  # TODO: a number, in metres (not in mm)


# -----------------------------------------------------------------------------
# The nine sheets
# -----------------------------------------------------------------------------

# A vertical line down the middle of a marker crosses its bands, from top to
# bottom. We write each band as one letter: "k" black, "w" white, "r" red,
# "g" green. Sheet 1 is black, red, black: "krk".
#
# TODO: complete the patterns of sheets 3, 4, 6, 7 and 8. The table of
# scenario.md §4.1 describes them, and assets/sheet-3.png… show them. Sheet 5
# has 2 black stripes on white: "kwk". Sheets 6, 7 and 8 have 3, 4 and 5.
PATTERNS: dict[str, str | None] = {
    "sheet-1": "krk",
    "sheet-2": "kgk",
    "sheet-3": None,
    "sheet-4": None,
    "sheet-5": "kwk",
    "sheet-6": None,
    "sheet-7": None,
    "sheet-8": None,
    "sheet-9": "k",
}


# -----------------------------------------------------------------------------
# Two functions are given. Read them line by line. Then give each one a name
# that says what it does.
# -----------------------------------------------------------------------------

# TODO: this function turns one pixel into one letter. Give it a name in two
# or three words joined by "_". Rename it here, and in `detect_marker` below.
def useful_function_1_to_rename(hsv_pixel: np.ndarray) -> str:
    # The pixel is in HSV, not in RGB: hue (the colour on the colour wheel),
    # saturation (how strong the colour is), value (how bright it is). Slide 7
    # ("HSV separates hue from brightness") says why: when the light changes,
    # the hue stays, the value moves.
    # In OpenCV, the three numbers go from 0 to 255, except the hue: an angle
    # from 0 to 360 degrees, stored divided by 2, so from 0 to 179.
    # `h, s, v = hsv_pixel` puts the three numbers in three names at once.
    h, s, v = hsv_pixel
    # Too dark: black ink, whatever its colour.
    if v < 90:
        return "k"
    # A strong colour, bright enough to trust its hue.
    if s > 90 and v > 60:
        # Red is at 0 degrees on the colour wheel, which is also 360: its hue
        # is near 0 or near 180 in OpenCV.
        if h < 12 or h > 168:
            return "r"
        # Green is at 120 degrees: 60 in OpenCV.
        if 40 < h < 90:
            return "g"
    # Anything else: the white paper.
    return "w"


# TODO: this function turns a list of letters, one per row of pixels, into a
# short word, one letter per band. Give it a name in two or three words
# joined by "_". Rename it here, and in `detect_marker` below.
def useful_function_2_to_rename(labels: list[str], min_run: int) -> str:
    # Example: labels = ["k", "k", "k", "r", "r", "r", "r", "k", "k", "k"],
    # min_run = 3 gives "krk".
    # A "run" is a group of the same letter, one after the other. A run
    # shorter than `min_run` rows is noise: we drop it.
    # `min_run` is the setting that decides. Too large, a real thin band
    # disappears. Too small, the edge of a shadow adds a band: sheet 6 in a
    # shadow reads as sheet 7.
    out = []                 # the letters kept, one per band
    current = None           # the letter of the current run
    count = 0                # how many rows the current run has so far
    # We add None at the end, so that the last run also closes.
    for label in labels + [None]:
        # Same letter as before: the run goes on.
        if label == current:
            count += 1
            continue
        # A new letter: the current run ends. Keep its letter if the run is
        # long enough, and if it is not the same letter as the band before
        # (a dropped run between two runs of "k" must not give "kk").
        if current is not None and count >= min_run and (not out or out[-1] != current):
            out.append(current)
        # A new run starts, with one row.
        current, count = label, 1
    # "".join(out) makes one word of the letters. .strip("w") removes the
    # white paper above and below the marker.
    return "".join(out).strip("w")


# -----------------------------------------------------------------------------
# Your function
# -----------------------------------------------------------------------------

# Sheet 5 is black, white, black: two black bands, and nothing between them.
# To see one marker, not two, we close the white gap with a tall, thin
# "structuring element" (a small rectangle of ones). How tall, in pixels?
# At 1 m, the marker is about 106 px tall in the image (slide 9), and the
# white gap of sheet 5 is 60 % of it: about 64 px. The rectangle must be
# taller than the gap. Closer than 1 m, the gap grows: keep that in mind in
# step 4.
CLOSE_HEIGHT_PX = None  # TODO: a whole number of pixels. Try 81.


def detect_marker(image: np.ndarray) -> tuple[str, float | None, float | None, float]:
    # `image` is H x W x 3, in RGB, one byte (0 to 255) per colour.
    # Return a tuple of four values: (target, distance_m, bearing_deg,
    # confidence), for example ("sheet-3", 1.12, -4.5, 0.83).
    #   target      one of the keys of PATTERNS, or "unknown"
    #   distance_m  from the apparent height, in metres
    #   bearing_deg the angle to the marker; to the left is negative
    #   confidence  between 0 and 1
    # "unknown" is an answer, not a failure: the robot does something with it.

    # 1. The colour threshold (slide 13: "The classical pipeline, in four
    #    steps").
    #    TODO: convert the image from RGB to HSV. cv2.cvtColor takes the image
    #    and a code that says which conversion. The codes all start with
    #    cv2.COLOR_: find the right one in the documentation, and replace
    #    COLOR_None with it. Careful: our image is in RGB, not in BGR.
    #    https://docs.opencv.org/4.x/df/d9d/tutorial_py_colorspaces.html
    hsv = cv2.cvtColor(image, cv2.COLOR_None)
    #    `hsv[..., 1]` is the second number of every pixel: an H x W array of
    #    saturations. The `...` means "all the rows, all the columns".
    s, v = hsv[..., 1], hsv[..., 2]

    #    TODO: `ink` is an H x W array of True and False: True where the pixel
    #    is ink. In words: the pixel is dark (v below 95), or it is a strong,
    #    bright colour (s above 95 and v above 55).
    #    On arrays, `and` and `or` do not work: numpy uses `&` for "and", `|`
    #    for "or", on every pixel at once. Put each comparison in parentheses,
    #    as in (v < 95): `&` and `|` are computed before `<`.
    ink = None

    # 2. Morphology: close the holes, and join the top and bottom bands of
    #    sheet 5. A "closing" fills every gap smaller than the structuring
    #    element. `.astype(np.uint8)` turns True/False into 1/0: OpenCV wants
    #    numbers.
    #    https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html
    #    TODO: np.ones((rows, columns), np.uint8) is the structuring element.
    #    Replace None with the number of rows: the height you chose above.
    #    Three columns: we close vertically only.
    mask = cv2.morphologyEx(ink.astype(np.uint8), cv2.MORPH_CLOSE,
                            np.ones((None, 3), np.uint8))
    #    Look at it: out/<image>-marker-mask.png. 1 * 255 is white.
    show("marker-mask", mask * 255)

    # 3. Contours: the outline of each white region of the mask.
    #    RETR_EXTERNAL keeps the outer outlines only; CHAIN_APPROX_SIMPLE
    #    stores them with few points. `_` is a value we do not use.
    #    https://docs.opencv.org/4.x/d4/d73/tutorial_py_contours_begin.html
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #    cv2.boundingRect gives the smallest upright box around a contour:
    #    (x, y, w, h), its left column, its top row, its width and its height.
    #    https://docs.opencv.org/4.x/dd/d49/tutorial_py_contour_features.html
    #    The form `[f(c) for c in contours]` is a "list comprehension": a list
    #    with one value per contour. It is the same as a for loop that calls
    #    .append().
    boxes = [cv2.boundingRect(c) for c in contours]

    # 4. The shape filter. A marker is a rolled A4 sheet, standing: 297 mm
    #    around, so 297 / pi = 95 mm wide, and about 200 mm tall. Seen from
    #    the front, height / width is about 2.1. At an angle, or half hidden,
    #    it changes: we accept from 0.9 to 2.6.
    #    TODO: keep only the boxes b that are a possible marker. `b[2]` is the
    #    width, `b[3]` the height. In words: at least 24 px tall, at least 8 px
    #    wide, and height / width between 0.9 and 2.6. Python accepts
    #    `0.9 <= a <= 2.6`. Write the condition after `if`.
    boxes = [b for b in boxes if True]
    if not boxes:
        return "unknown", None, None, 0.0

    #    The tallest box. max(list, key=f) returns the element whose f is the
    #    largest. `lambda b: b[3]` is a small function without a name: it
    #    takes b and returns its height.
    #    https://docs.python.org/3/library/functions.html#max
    x, y, w, height = max(boxes, key=lambda b: b[3])

    # 5. Read the bands, down a thin column at the centre of the box: from
    #    the column x + w // 2 - half to the column x + w // 2 + half.
    #    `a:b` in an index keeps the rows (or columns) from a to b - 1.
    half = max(1, w // 8)
    column = hsv[y:y + height, x + w // 2 - half: x + w // 2 + half + 1]
    #    `column` has `height` rows. Each row, column[row], is a small array:
    #    one line per pixel, one column per channel (H, S, V).
    #    TODO: the median pixel of each row: np.median of column[row] takes the
    #    median of each channel over the pixels of the row, when you give it
    #    the right `axis`. axis=0 goes down the lines, axis=1 across the
    #    columns. Which one gives three numbers, H, S and V? Replace None.
    #    https://numpy.org/doc/stable/reference/generated/numpy.median.html
    labels = [useful_function_1_to_rename(np.median(column[row], axis=None))
              for row in range(height)]
    #    A run must last a twentieth of the height, and at least 3 rows.
    word = useful_function_2_to_rename(labels, max(3, height // 20))

    #    TODO: the sheet whose pattern is `word`. In words: start with
    #    sheet = None; for each name and pattern of PATTERNS.items(), if the
    #    pattern is equal to word, the sheet is that name.
    sheet = None
    if sheet is None:
        return "unknown", None, None, 0.0

    # 6. Distance and bearing.
    #    TODO: the distance, in metres, from the apparent height `height`, in
    #    pixels. Slide 9: d = f * H_real / h_pixels. Use FOCAL_PX and
    #    CYLINDER_HEIGHT_M.
    distance = None

    #    How far the centre of the box is from the centre of the image, in
    #    pixels. image.shape is (H, W, 3): image.shape[1] is the width.
    offset = x + w / 2 - image.shape[1] / 2
    #    TODO: the bearing, in degrees. In the camera, tan(angle) = offset /
    #    focal. math.atan gives the angle in radians, math.degrees turns it
    #    into degrees.
    bearing = None

    # 7. The confidence: the fraction of ink in the box. `.mean()` of an array
    #    of True/False is the fraction of True. np.clip keeps a number between
    #    0 and 1. https://numpy.org/doc/stable/reference/generated/numpy.clip.html
    ink_fraction = float(ink[y:y + height, x:x + w].mean())
    return sheet, distance, bearing, float(np.clip(0.40 + 0.9 * ink_fraction, 0, 1))
