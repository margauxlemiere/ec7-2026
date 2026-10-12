#!/usr/bin/env python3
"""Build the image set of lab 3: markers and road signs, in five light conditions.

    python tools/render_scenes.py                 # writes data/, 250 images
    python tools/render_scenes.py --per 10        # a small set, to try it out

These images are **renderings, not photographs**. They are here so that everyone
measures on the same input, and so that the two failures this lab is about are
certain to be present: a hard-edged shadow that adds a band to a striped marker,
and a washed-out pixel whose hue means nothing. A photograph shoot cannot
promise either on the day.

Real light enters this lab twice: through `--input device`, and through the
photographs your assignment asks you to take.

One camera throughout: 640x480, focal length 530 px, 0.18 m above the floor,
which is roughly a Pi camera on a TurtleBot3. The ground truth is therefore
exact: a target of real size S at distance Z is f * S / Z pixels tall, and a
target at bearing b is at column 320 + f * tan(b).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib

import cv2
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent.parent
ASSETS = HERE / "assets"
DATA = HERE / "data"

WIDTH, HEIGHT = 640, 480
FOCAL_PX = 530.0                   # roughly a Pi camera v2 over 640 px of width
CAMERA_HEIGHT_M = 0.18             # the camera sits low, on the robot
SIGN_CENTRE_M = 0.19               # the model sign: 225 mm tall on its post, a 70 mm disc

# The two real sizes that turn an apparent size into a distance. They are
# measured on the PDFs by `tools/prepare_assets.py`, not guessed here.
_GEOMETRY = json.loads((ASSETS / "geometry.json").read_text()) \
    if (ASSETS / "geometry.json").exists() else {}
CYLINDER_HEIGHT_M = _GEOMETRY.get("cylinder_ink_height_mm", 201.5) / 1000.0
SIGN_DIAMETER_M = _GEOMETRY.get("sign_diameter_mm", 150.0) / 1000.0

SHEETS = [f"sheet-{i}" for i in range(1, 10)]
SIGNS = ["sign-up", "sign-down", "sign-left", "sign-right", "sign-stop", "sign-free"]
TARGETS = SHEETS + SIGNS
STRIPED = ["sheet-5", "sheet-6", "sheet-7", "sheet-8"]
CONDITIONS = ["even", "low-sun", "shadow", "mixed", "absent"]

# warmth of the light, and its strength
LIGHT = {"even": (0.05, 1.00), "low-sun": (0.85, 1.10), "shadow": (0.40, 1.02),
         "mixed": (0.30, 1.04), "absent": (0.20, 1.00)}


# --------------------------------------------------------------------- assets
def load_assets() -> dict[str, np.ndarray]:
    """The flat texture of every target, RGBA in 0-1."""
    out = {}
    for name in TARGETS:
        path = ASSETS / f"{name}.png"
        if not path.exists():
            raise SystemExit(f"missing {path} — run python tools/prepare_assets.py first")
        im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        out[name] = cv2.cvtColor(im, cv2.COLOR_BGRA2RGBA).astype(np.float64) / 255.0
    return out


def tint_of(warm: float) -> np.ndarray:
    return np.array([1.0 + 0.13 * warm, 1.0 - 0.02 * warm, 1.0 - 0.18 * warm])


# ------------------------------------------------------------------ the scene
def lane(rng: np.random.Generator, warm: float, bright: float) -> np.ndarray:
    """A workshop lane: a wall, a floor, grain. Not photorealistic, and clean."""
    horizon = int(HEIGHT * 0.50)                       # the camera looks straight ahead
    y = np.linspace(0, 1, HEIGHT)[:, None]
    wall = 0.68 - 0.14 * np.clip((horizon / HEIGHT - y) / (horizon / HEIGHT), 0, 1)
    floor = 0.50 - 0.10 * np.clip((y - horizon / HEIGHT) / (1 - horizon / HEIGHT), 0, 1)
    grey = np.where(np.arange(HEIGHT)[:, None] < horizon, wall, floor)
    img = np.repeat(np.repeat(grey, WIDTH, axis=1)[:, :, None], 3, axis=2)

    # The wall meets the floor in a seam, and a lane edge is painted on the
    # floor. Both run the full width of the image, so neither is allowed to be
    # as dark as ink: a horizontal bar of ink across the frame would join every
    # marker to every other thing, and that is a quarrel with the room rather
    # than with the markers.
    img[horizon:horizon + 2, :] *= 0.88
    x0 = int(rng.integers(20, WIDTH - 60))             # a conveyor upright, to reject
    img[:horizon, x0:x0 + 9] *= 0.82
    y0 = int(rng.integers(int(HEIGHT * 0.72), HEIGHT - 6))
    img[y0:y0 + 3, :] = np.clip(img[y0:y0 + 3, :] * 0.35 + 0.62, 0, 1)

    img += rng.normal(0, 0.008, img.shape)
    return np.clip(img * bright * tint_of(warm)[None, None, :], 0, 1)


def rolled(flat: np.ndarray, height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    """A sheet rolled into a cylinder: (rgb, alpha).

    The bands run across the full width, so rolling the sheet is a resampling
    in x by the angle, plus the shading of a curved surface. That shading is
    why a marker read from the side is harder than one read face on.
    """
    h_src, w_src = flat.shape[:2]
    xs = np.linspace(-1.0, 1.0, width)
    ang = np.arcsin(np.clip(xs, -1, 1))
    u = (ang / (math.pi / 2) * 0.5 + 0.5) * (w_src - 1)
    ys = np.linspace(0, h_src - 1, height).astype(int)
    body = flat[np.ix_(ys, np.clip(u.astype(int), 0, w_src - 1))][..., :3]
    lam = np.sqrt(np.clip(1.0 - xs ** 2, 0, 1))
    shade = 0.30 + 0.70 * lam ** 0.5
    spec = 0.10 * np.exp(-((xs + 0.30) / 0.20) ** 2)   # printed paper is shiny
    body = np.clip(body * shade[None, :, None] + spec[None, :, None], 0, 1)
    return body, np.ones((height, width))


def sign_face(flat: np.ndarray, side: int) -> tuple[np.ndarray, np.ndarray]:
    """A model road sign, a disc on a post, facing the lane: (rgb, alpha)."""
    im = cv2.resize(flat, (side, side), interpolation=cv2.INTER_AREA)
    return im[..., :3], im[..., 3]


def yaw(rgb: np.ndarray, alpha: np.ndarray, degrees: float):
    """Turn the target away from the camera, around its vertical axis.

    A plane seen at an angle loses width by cos(angle), and its receding edge
    loses height. The result is cropped to what is left, so the caller can place
    it by its real bounding box.
    """
    if abs(degrees) < 1.0:
        return rgb, alpha
    h, w = alpha.shape
    k = math.cos(math.radians(abs(degrees)))
    edge = h * 0.38 * abs(math.sin(math.radians(degrees))) / 2
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    if degrees > 0:                                    # the right edge recedes
        dst = np.float32([[0, 0], [w * k, edge], [w * k, h - edge], [0, h]])
    else:                                              # the left edge recedes
        dst = np.float32([[w * (1 - k), edge], [w, 0], [w, h], [w * (1 - k), h - edge]])
    m = cv2.getPerspectiveTransform(src, dst)
    rgb = cv2.warpPerspective(rgb, m, (w, h), flags=cv2.INTER_LINEAR)
    alpha = cv2.warpPerspective(alpha, m, (w, h), flags=cv2.INTER_LINEAR)
    cols = np.where(alpha.max(axis=0) > 0.05)[0]
    if len(cols):
        rgb, alpha = rgb[:, cols[0]:cols[-1] + 1], alpha[:, cols[0]:cols[-1] + 1]
    return rgb, alpha


def paste(scene: np.ndarray, rgb: np.ndarray, alpha: np.ndarray, x: int, y: int) -> None:
    h, w = alpha.shape
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(WIDTH, x + w), min(HEIGHT, y + h)
    if x1 <= x0 or y1 <= y0:
        return
    sub = (slice(y0 - y, y1 - y), slice(x0 - x, x1 - x))
    a = np.clip(alpha[sub], 0, 1)[..., None]
    scene[y0:y1, x0:x1] = rgb[sub] * a + scene[y0:y1, x0:x1] * (1 - a)


def hard_shadow(scene: np.ndarray, rng, box: tuple, thickness_px: float,
                strength: float = 0.55) -> None:
    """A shadow with a hard edge, laid across the target by a shelf or a pillar.

    The hard edge is the point: a threshold cannot tell it from ink, so a
    striped marker gains a band it does not have, and sheet 6 reads as sheet 7.
    Its thickness is that of one band of the pattern.

    The band falls full strength on the target and much weaker on what
    surrounds it. A shadow dark enough to add a band to printed white is dark
    enough to turn a whole wall into ink, and a set where every image is one
    huge black region measures the renderer rather than the marker.
    """
    bx, by, bw, bh = box
    angle = math.radians(rng.uniform(6, 24) * float(rng.choice([-1, 1])))
    yy = np.arange(HEIGHT)[:, None] / HEIGHT
    xx = np.arange(WIDTH)[None, :] / WIDTH
    d = yy * math.cos(angle) - xx * math.sin(angle)
    cy = (by + bh * rng.uniform(0.28, 0.72)) / HEIGHT
    cx = (bx + bw / 2) / WIDTH
    centre = cy * math.cos(angle) - cx * math.sin(angle)
    thickness = max(thickness_px, 4) / HEIGHT
    band = np.abs(d - centre) < thickness / 2

    on_target = np.zeros((HEIGHT, WIDTH), dtype=bool)
    on_target[:, max(0, bx):bx + bw] = True
    scene[band & on_target] *= (1.0 - strength)
    scene[band & ~on_target] *= (1.0 - strength * 0.40)


def low_sun(scene: np.ndarray, rng) -> None:
    """Afternoon light from one side: bright there, dim across the lane.

    No hard edge here. What this condition costs you is the hue of the ink,
    which a warm low sun moves, and a threshold tuned in the middle of the
    frame, which no longer holds at its edges.
    """
    ramp = np.linspace(*((1.22, 0.78) if rng.random() < 0.5 else (0.78, 1.22)), WIDTH)
    scene *= ramp[None, :, None]
    np.clip(scene, 0, 1, out=scene)


def sun_patch(scene: np.ndarray, rng, box: tuple) -> None:
    """Half the target in a sunlit patch, with a hard edge down the middle.

    Hue is meaningless where saturation is near zero, and that is what a patch
    of direct sun does to printed ink. One global threshold cannot hold on both
    sides of this edge.
    """
    bx, by, bw, bh = box
    cut = bx + int(bw * rng.uniform(0.35, 0.65))
    lean = rng.uniform(-0.18, 0.18)
    top = max(0, by - int(bh * 0.5))
    bottom = min(HEIGHT, by + int(bh * 1.4))
    right = min(WIDTH, cut + int(max(bw * 1.6, 40)))
    for row in range(top, bottom):
        x0 = int(np.clip(cut + lean * (row - top), 0, WIDTH))
        if x0 >= right:
            continue
        region = scene[row, x0:right]
        # Direct sun raises the value and takes the colour out of the ink. The
        # bands are still there in luminance; the hue is gone, which is the
        # point of the condition.
        grey = region.mean(axis=-1, keepdims=True)
        scene[row, x0:right] = np.clip((region * 0.20 + grey * 0.80) * 1.28 + 0.05, 0, 1)


def distractors(scene: np.ndarray, rng) -> None:
    """What lives in a lane and is not a target: the false positive set."""
    for _ in range(int(rng.integers(2, 5))):
        kind = int(rng.integers(0, 4))
        x = int(rng.integers(10, WIDTH - 90))
        y = int(rng.integers(int(HEIGHT * 0.25), HEIGHT - 110))
        w, h = int(rng.integers(28, 85)), int(rng.integers(28, 105))
        if kind == 0:                                  # a red crate, hue of sheet 1
            colour = np.array([0.70, 0.10, 0.09])
        elif kind == 1:                                # a green bin, hue of sheet 2
            colour = np.array([0.10, 0.52, 0.24])
        elif kind == 2:                                # a white disc, shape of a sign
            colour, h = np.array([0.90, 0.90, 0.88]), w
        else:                                          # a black cable duct
            colour, w = np.array([0.10, 0.10, 0.11]), int(rng.integers(10, 24))
        patch = np.ones((h, w, 3)) * colour
        patch *= (0.85 + 0.30 * np.linspace(0, 1, w))[None, :, None]
        if kind == 2:
            yy, xx = np.mgrid[0:h, 0:w]
            mask = ((yy - h / 2) / (h / 2)) ** 2 + ((xx - w / 2) / (w / 2)) ** 2 <= 1
            paste(scene, patch, mask.astype(float), x, y)
        else:
            paste(scene, patch, np.ones((h, w)), x, y)


# ------------------------------------------------------------------ one image
def place(target: str, distance: float, bearing: float) -> tuple[float, int, int]:
    """Where the target lands: (centre column, top row, apparent size in px)."""
    cx, cy = WIDTH / 2, HEIGHT / 2
    u = cx + FOCAL_PX * math.tan(math.radians(bearing))
    if target.startswith("sheet"):
        size = FOCAL_PX * CYLINDER_HEIGHT_M / distance
        base = cy + FOCAL_PX * CAMERA_HEIGHT_M / distance          # it stands on the floor
        top = base - size
    else:
        size = FOCAL_PX * SIGN_DIAMETER_M / distance
        centre = cy + FOCAL_PX * (CAMERA_HEIGHT_M - SIGN_CENTRE_M) / distance
        top = centre - size / 2
    return u, int(round(top)), int(round(size))


def render(assets: dict, target: str | None, condition: str,
           rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    warm, bright = LIGHT[condition]
    scene = lane(rng, warm, bright)
    meta = {"distance_m": "", "bearing_deg": "", "yaw_deg": "",
            "occlusion": 0, "blur": 0, "shadow": 0}
    box, stripe_px = None, None

    if target:
        # Two of the sets exist to make a readable marker read *wrongly*: the
        # shadow that adds a band, the sun that takes the colour out. Both need
        # the marker to be readable in the first place, so they stay near.
        # A model sign is 70 mm wide: past 1.3 m its octagon is under 25 px.
        if target.startswith("sign"):
            distance = float(rng.uniform(0.30, 1.30))
        else:
            near = 1.5 if condition in ("shadow", "mixed") else 2.5
            distance = float(rng.uniform(0.45, near))
        bearing = float(rng.uniform(-22, 22))
        yaw_deg = float(rng.uniform(-40, 40))
        centre_u, top, size = place(target, distance, bearing)

        flat = assets[target]
        if target.startswith("sheet"):
            rgb, alpha = rolled(flat, size, max(6, int(size * 0.72)))
            stripe_px = size / 7.0                     # about one band of the pattern
        else:
            rgb, alpha = sign_face(flat, max(8, size))
        rgb, alpha = yaw(rgb, alpha, yaw_deg)
        rgb = np.clip(rgb * bright * tint_of(warm)[None, None, :], 0, 1)

        # The bearing is ground truth, so the target is centred on the column the
        # bearing says, after the turn has taken some of its width away.
        oh, ow = alpha.shape
        left = int(round(centre_u - ow / 2))
        if rng.random() < 0.18:                        # partly hidden by something
            cut = max(1, int(ow * rng.uniform(0.20, 0.40)))
            alpha = alpha.copy()
            if rng.integers(0, 2):
                alpha[:, :cut] = 0
            else:
                alpha[:, ow - cut:] = 0
            meta["occlusion"] = 1

        if target.startswith("sign"):                  # the post it is mounted on
            post = max(2, size // 14)
            foot = int(HEIGHT / 2 + FOCAL_PX * CAMERA_HEIGHT_M / distance)
            px = left + ow // 2 - post // 2
            paste(scene, np.full((max(1, foot - top - oh), post, 3), 0.34),
                  np.ones((max(1, foot - top - oh), post)), px, top + oh)

        paste(scene, rgb, alpha, left, top)
        box = (max(0, left), max(0, top), ow, oh)
        foot = min(HEIGHT - 1, top + oh)
        scene[foot:foot + 2, max(0, left):left + ow] *= 0.78      # contact with the floor
        meta.update(distance_m=round(distance, 3), bearing_deg=round(bearing, 1),
                    yaw_deg=round(yaw_deg, 1))

    if condition == "shadow" and box:
        hard_shadow(scene, rng, box,
                    thickness_px=(stripe_px or box[3] / 7) * rng.uniform(0.9, 1.6))
        meta["shadow"] = 1
    elif condition == "low-sun":
        low_sun(scene, rng)
    elif condition == "mixed" and box:
        sun_patch(scene, rng, box)
    if condition == "absent":
        distractors(scene, rng)

    if rng.random() < 0.22:                            # the robot moves while it looks
        k = int(rng.integers(5, 13))
        kernel = np.zeros((k, k))
        kernel[k // 2, :] = 1.0 / k
        scene = cv2.filter2D(scene, -1, kernel)
        meta["blur"] = 1

    scene = np.clip(scene + rng.normal(0, 0.012, scene.shape), 0, 1)
    return scene, meta


# ----------------------------------------------------------------------- main
def pick_target(condition: str, i: int, rng) -> str | None:
    if condition == "absent":
        return None
    if condition == "shadow" and i % 2 == 0:
        return STRIPED[(i // 2) % 4]
    return TARGETS[int(rng.integers(0, len(TARGETS)))]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--per", type=int, default=50, help="images per condition")
    ap.add_argument("--seed", type=int, default=20261012)
    args = ap.parse_args()

    assets = load_assets()
    rng = np.random.default_rng(args.seed)
    rows = []
    for condition in CONDITIONS:
        (DATA / condition).mkdir(parents=True, exist_ok=True)
        for i in range(args.per):
            target = pick_target(condition, i, rng)
            scene, meta = render(assets, target, condition, rng)
            name = f"{condition}/{condition}-{i:03d}.jpg"
            cv2.imwrite(str(DATA / name),
                        cv2.cvtColor((scene * 255).astype(np.uint8), cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, 88])
            family = "none" if target is None else \
                ("cylinder" if target.startswith("sheet") else "sign")
            rows.append({"file": name, "condition": condition,
                         "target": target or "none", "family": family, **meta})

    with (DATA / "labels.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # The intrinsics are here; the real sizes of the targets are in
    # assets/geometry.json.
    (DATA / "camera.json").write_text(json.dumps({
        "width": WIDTH, "height": HEIGHT, "focal_px": FOCAL_PX,
        "camera_height_m": CAMERA_HEIGHT_M,
        "note": "this set is rendered, so the focal length is exact. The real "
                "size of each target is in assets/geometry.json.",
    }, indent=2) + "\n", encoding="utf-8")

    print(f"{len(rows)} images in {DATA}, labels.csv written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
