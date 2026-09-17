#!/usr/bin/env python3
"""Lab 3 — the engine. It feeds, it times, it scores. **Do not modify it.**

    python engine.py --input files --set even --show
    python engine.py --input files --set all
    python engine.py --input files --set absent
    python engine.py --task policy
    python engine.py --input device

Two detectors, scored apart, because they fail apart:

    detect_marker(image)   the nine paper cylinders of markers.pdf
    detect_sign(image)     the six road signs of road-signs.pdf

Each one answers `unknown` on everything that is not its own family, and that
includes the other family. A red octagon is an excellent false positive for a
detector that looks for a red band, and this is where you find out.

Nothing is averaged across conditions. One number over five lighting conditions
hides the one that breaks you, and that one is what decides whether the robot
can be let out of the lab.
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import statistics
import sys
import time
import traceback

import cv2
import numpy as np

import harness
import student

HERE = pathlib.Path(__file__).resolve().parent
DATA = HERE / "data"
CONDITIONS = ["even", "low-sun", "shadow", "mixed", "absent"]
FPS = 10.0                       # the robot looks ten times a second
ACTIONS = {"forward", "slow", "stop", "turn_left", "turn_right", "report"}
DETECTORS = {"marker": ("cylinder", "detect_marker"), "sign": ("sign", "detect_sign")}
# A sign has a family before it has a name: a red octagon, a blue disc with an
# arrow in it, a white disc crossed out. The lab asks for the family, and the
# assignment asks which arrow.
SIGN_FAMILY = {"sign-stop": "stop", "sign-free": "free", "sign-up": "direction",
               "sign-down": "direction", "sign-left": "direction",
               "sign-right": "direction"}


# ------------------------------------------------------------------- calling
def call(name: str, image: np.ndarray):
    """Run one detector on one image. A crash is a result, not a stop."""
    function = getattr(student, name, None)
    if function is None:
        return ("unknown", None, None, 0.0), 0.0, f"student.{name} is missing"
    start = time.perf_counter()
    try:
        answer = function(image)
    except Exception:                                  # noqa: BLE001 — we report it
        return ("unknown", None, None, 0.0), (time.perf_counter() - start) * 1e3, \
            traceback.format_exc(limit=2).strip().splitlines()[-1]
    latency = (time.perf_counter() - start) * 1e3
    if answer is None:
        return ("unknown", None, None, 0.0), latency, None
    try:
        target, distance, bearing, confidence = answer
    except (TypeError, ValueError):
        return ("unknown", None, None, 0.0), latency, \
            f"student.{name} returned {answer!r}, expected 4 values"
    return (str(target), distance, bearing, float(confidence or 0.0)), latency, None


# ------------------------------------------------------------------- scoring
def load_labels() -> list[dict]:
    path = DATA / "labels.csv"
    if not path.exists():
        raise SystemExit("data/labels.csv is missing — "
                         "run python tools/render_scenes.py")
    return list(csv.DictReader(path.open(encoding="utf-8")))


def score(rows: list[dict], show: bool) -> tuple[list[dict], list[dict], list[str]]:
    results, confusions, complaints = [], {}, []
    by_condition: dict[str, list[dict]] = {}
    for row in rows:
        by_condition.setdefault(row["condition"], []).append(row)

    for condition, group in by_condition.items():
        readings = {key: {"latency": [], "hit": 0, "family_hit": 0, "present": 0,
                          "other": 0, "fp": 0, "dist_err": [], "bear_err": []}
                    for key in DETECTORS}
        for row in group:
            image = cv2.cvtColor(cv2.imread(str(DATA / row["file"])), cv2.COLOR_BGR2RGB)
            harness.frame(pathlib.Path(row["file"]).stem)
            for key, (family, function) in DETECTORS.items():
                answer, latency, complaint = call(function, image)
                if complaint and complaint not in complaints:
                    complaints.append(complaint)
                target, distance, bearing, _confidence = answer
                bucket = readings[key]
                bucket["latency"].append(latency)
                mine = row["family"] == family
                if mine:
                    bucket["present"] += 1
                    if key == "sign" and SIGN_FAMILY.get(target) == \
                            SIGN_FAMILY.get(row["target"]):
                        bucket["family_hit"] += 1
                    if target == row["target"]:
                        bucket["hit"] += 1
                        if distance is not None:
                            bucket["dist_err"].append(
                                abs(float(distance) - float(row["distance_m"])))
                        if bearing is not None:
                            bucket["bear_err"].append(
                                abs(float(bearing) - float(row["bearing_deg"])))
                    else:
                        pair = (condition, key, row["target"], target)
                        confusions[pair] = confusions.get(pair, 0) + 1
                else:
                    bucket["other"] += 1
                    if target != "unknown":
                        bucket["fp"] += 1
                        pair = (condition, key, row["target"], target)
                        confusions[pair] = confusions.get(pair, 0) + 1

        for key, bucket in readings.items():
            rate = bucket["fp"] / bucket["other"] if bucket["other"] else 0.0
            results.append({
                "condition": condition, "detector": key,
                "n_present": bucket["present"],
                "accuracy": round(bucket["hit"] / bucket["present"], 3)
                            if bucket["present"] else "",
                "family_accuracy": round(bucket["family_hit"] / bucket["present"], 3)
                                   if bucket["present"] and key == "sign" else "",
                "distance_mae_m": round(statistics.fmean(bucket["dist_err"]), 3)
                                  if bucket["dist_err"] else "",
                "bearing_mae_deg": round(statistics.fmean(bucket["bear_err"]), 2)
                                   if bucket["bear_err"] else "",
                "n_other": bucket["other"],
                "false_positive_rate": round(rate, 3),
                "fp_per_minute": round(rate * FPS * 60, 1),
                "latency_ms_median": round(statistics.median(bucket["latency"]), 2),
                "latency_ms_p90": round(sorted(bucket["latency"])[
                    max(0, int(0.9 * len(bucket["latency"])) - 1)], 2),
            })

    rows_out = [{"condition": c, "detector": d, "truth": t, "answer": a, "count": n}
                for (c, d, t, a), n in sorted(confusions.items(), key=lambda kv: -kv[1])]
    return results, rows_out, complaints


def write(results: list[dict], confusions: list[dict]) -> None:
    order = {name: i for i, name in enumerate(CONDITIONS)}
    results.sort(key=lambda r: (order.get(r["condition"], 9), r["detector"]))
    with (HERE / "results.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    if confusions:
        with (HERE / "confusions.csv").open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(confusions[0].keys()))
            writer.writeheader()
            writer.writerows(confusions)


def report(results: list[dict], confusions: list[dict], complaints: list[str]) -> None:
    print(f"{'condition':<9} {'det':<7} {'n':>4} {'acc':>6} {'family':>7} "
          f"{'dist MAE':>9} {'bear MAE':>9} {'FP rate':>8} {'FP/min':>7} "
          f"{'med ms':>7} {'p90 ms':>7}")
    for r in results:
        print(f"{r['condition']:<9} {r['detector']:<7} {r['n_present']:>4} "
              f"{str(r['accuracy']):>6} {str(r['family_accuracy']):>7} "
              f"{str(r['distance_mae_m']):>9} "
              f"{str(r['bearing_mae_deg']):>9} {r['false_positive_rate']:>8} "
              f"{r['fp_per_minute']:>7} {r['latency_ms_median']:>7} "
              f"{r['latency_ms_p90']:>7}")
    print("\nresults.csv written, one row per condition and per detector.")
    if confusions:
        print("\nwhat was taken for what, most frequent first:")
        for row in confusions[:6]:
            print(f"  {row['condition']:<9} {row['detector']:<7} "
                  f"{row['truth']:<10} read as {row['answer']:<10} x{row['count']}")
    if complaints:
        print("\nthe engine had to correct you:")
        for line in complaints[:5]:
            print(f"  {line}")


# ------------------------------------------------------------------- policy
# The approach, replayed. Detections are given to you, so this part measures the
# policy alone: what the robot does, not what it saw. `history` carries the
# decisions you already took, which is where a memory of the stop sign lives.
APPROACH = [
    ({"target": "sign-stop", "distance_m": 2.40, "bearing_deg": 3.0, "confidence": 0.91},
     {"speed": 0.22, "person_ahead": False}),
    ({"target": "sign-stop", "distance_m": 1.20, "bearing_deg": 2.0, "confidence": 0.93},
     {"speed": 0.22, "person_ahead": False}),
    ({"target": "sign-stop", "distance_m": 0.60, "bearing_deg": 1.0, "confidence": 0.90},
     {"speed": 0.15, "person_ahead": False}),
    ({"target": "sign-stop", "distance_m": 0.30, "bearing_deg": 0.0, "confidence": 0.88},
     {"speed": 0.08, "person_ahead": False}),
    ({"target": "unknown", "distance_m": None, "bearing_deg": None, "confidence": 0.0},
     {"speed": 0.00, "person_ahead": False}),
    ({"target": "unknown", "distance_m": None, "bearing_deg": None, "confidence": 0.0},
     {"speed": 0.00, "person_ahead": True, "person_distance_m": 0.8}),
    ({"target": "sign-free", "distance_m": 1.00, "bearing_deg": -4.0, "confidence": 0.84},
     {"speed": 0.00, "person_ahead": False}),
    ({"target": "sign-left", "distance_m": 0.90, "bearing_deg": 6.0, "confidence": 0.79},
     {"speed": 0.20, "person_ahead": False}),
    ({"target": "unknown", "distance_m": None, "bearing_deg": None, "confidence": 0.0},
     {"speed": 0.20, "person_ahead": False}),
    ({"target": "unknown", "distance_m": None, "bearing_deg": None, "confidence": 0.0},
     {"speed": 0.20, "person_ahead": False}),
    ({"target": "sign-up", "distance_m": 1.10, "bearing_deg": 0.0, "confidence": 0.19},
     {"speed": 0.20, "person_ahead": False}),
    ({"target": "sign-stop", "distance_m": 1.10, "bearing_deg": 0.0, "confidence": 0.19},
     {"speed": 0.20, "person_ahead": False}),
]
MOVING = {"forward", "slow", "turn_left", "turn_right"}


def replay() -> int:
    history, faults = [], []
    stop_seen, stopped = False, False
    for step, (detection, state) in enumerate(APPROACH):
        state = dict(state, history=list(history))
        try:
            action = student.policy(dict(detection), state)
        except Exception:                              # noqa: BLE001
            print(f"  step {step}: policy raised "
                  f"{traceback.format_exc(limit=2).strip().splitlines()[-1]}")
            action = None
        action = str(action) if action else "none"
        if action not in ACTIONS and action != "none":
            faults.append(f"step {step}: {action!r} is not one of {sorted(ACTIONS)}")
        history.append({"detection": dict(detection), "state": dict(state),
                        "action": action})
        distance = detection["distance_m"]
        shown = "-" if distance is None else f"{distance:.2f} m"
        print(f"  step {step:>2}  {detection['target']:<10} {shown:<8}"
              f" conf {detection['confidence']:.2f}  ->  {action}")

        if detection["target"] == "sign-stop" and distance and distance <= 1.2 \
                and detection["confidence"] >= 0.5:
            stop_seen = True
        if stop_seen and action == "stop":
            stopped = True
        if stop_seen and not stopped and detection["target"] == "sign-stop" \
                and distance and distance <= 0.3:
            faults.append(f"step {step}: ran a stop sign at 0.30 m")
        if state.get("person_ahead") and action in MOVING:
            faults.append(f"step {step}: moved with a person "
                          f"{state.get('person_distance_m', '?')} m ahead")

    unknowns = [h for h in history if h["detection"]["target"] == "unknown"]
    if unknowns and not any(h["action"] == "report" for h in unknowns):
        faults.append("the robot never said anything while it did not know: "
                      "stopped in the lane with nothing said is the third "
                      "forbidden outcome")
    last = history[-1]["action"]
    if last in MOVING:
        faults.append("step 11: acted on a stop sign seen at confidence 0.19 by "
                      "moving. A refusal and a wrong action do not cost the same "
                      "when the wrong action is not stopping")

    print()
    if faults:
        print(f"{len(faults)} forbidden outcome(s):")
        for fault in faults:
            print(f"  - {fault}")
        return 1
    print("no forbidden outcome. Say in one line what your policy does when it "
          "is unsure, and why that is the safe side here.")
    return 0


# ------------------------------------------------------------------- device
def device() -> int:
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("no camera — use --input files")
        return 1
    print("q to quit. No score here: there is no ground truth on your desk.")
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                break
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            marker, _, _ = call("detect_marker", image)
            sign, _, _ = call("detect_sign", image)
            best = marker if marker[3] >= sign[3] else sign
            print(f"\r{best[0]:<12} {best[1] if best[1] else '-':<8} "
                  f"conf {best[3]:.2f}   ", end="", flush=True)
            cv2.imshow("lab 3", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()
    return 0


# --------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", choices=["files", "device"], default="files")
    ap.add_argument("--set", dest="sets", action="append",
                    choices=CONDITIONS + ["all"], help="repeatable; default all")
    ap.add_argument("--task", choices=["detect", "policy"], default="detect")
    ap.add_argument("--show", action="store_true",
                    help="write your intermediate images to out/")
    ap.add_argument("--limit", type=int, help="stop after N images per condition")
    args = ap.parse_args()

    if args.task == "policy":
        return replay()
    if args.input == "device":
        return device()

    wanted = CONDITIONS if not args.sets or "all" in args.sets else args.sets
    rows = [r for r in load_labels() if r["condition"] in wanted]
    if args.limit:
        keep, seen = [], {}
        for row in rows:
            seen[row["condition"]] = seen.get(row["condition"], 0) + 1
            if seen[row["condition"]] <= args.limit:
                keep.append(row)
        rows = keep
    if args.show:
        harness.enable(str(HERE / "out"))

    camera = json.loads((DATA / "camera.json").read_text(encoding="utf-8"))
    print(f"{len(rows)} images, {', '.join(wanted)} — camera "
          f"{camera['width']}x{camera['height']}, focal {camera['focal_px']:.0f} px\n")
    results, confusions, complaints = score(rows, args.show)
    write(results, confusions)
    report(results, confusions, complaints)
    return 0


if __name__ == "__main__":
    sys.exit(main())
