#!/usr/bin/env python3
"""EC7 — check that the lab environment runs on this machine, and print a report.

Run it once at the very beginning of the course, on the machine you will use:

    python3 -m venv .venv
    . .venv/bin/activate            # Windows: .venv\\Scripts\\activate
    pip install -r requirements.txt
    python check_env.py > env.txt

The report names your platform, your Python, the version of every package that
imported, and whether a microphone and a camera were reachable. Nothing is sent
anywhere; the file is yours to commit.

A missing microphone or camera is not a failure: every EC7 lab also runs from
provided files. What must not fail is the import of the packages.
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.metadata as md
import platform
import sys
import time

PACKAGES = [
    "numpy", "scipy", "soundfile", "sounddevice",
    "cv2", "skimage", "sklearn", "jiwer",
    "faster_whisper", "piper", "torch", "ultralytics",
]
DISTRIBUTION = {"cv2": "opencv-python", "skimage": "scikit-image",
                "sklearn": "scikit-learn", "piper": "piper-tts"}


def version_of(module: str) -> str:
    try:
        return md.version(DISTRIBUTION.get(module, module))
    except md.PackageNotFoundError:
        return getattr(importlib.import_module(module), "__version__", "unknown")


def check_packages() -> list[str]:
    lines, present = [], []
    for name in PACKAGES:
        try:
            importlib.import_module(name)
        except Exception as exc:                      # noqa: BLE001 - we report anything
            lines.append(f"  {name:<16} not installed ({type(exc).__name__})")
            continue
        version = version_of(name)
        lines.append(f"  {name:<16} {version}")
        present.append(f"{name}=={version}")
    return lines, present


def check_microphone() -> str:
    try:
        import sounddevice as sd
    except Exception as exc:                          # noqa: BLE001
        return f"sounddevice unavailable ({type(exc).__name__})"
    try:
        inputs = [d for d in sd.query_devices() if d["max_input_channels"] > 0]
    except Exception as exc:                          # noqa: BLE001
        return f"no audio backend ({type(exc).__name__})"
    if not inputs:
        return "no input device — use --input files"
    return f"{len(inputs)} input device(s), default: {inputs[0]['name']}"


def check_camera() -> str:
    try:
        import cv2
    except Exception as exc:                          # noqa: BLE001
        return f"opencv unavailable ({type(exc).__name__})"
    capture = cv2.VideoCapture(0)
    try:
        if not capture.isOpened():
            return "no camera — use --input files"
        ok, frame = capture.read()
        return f"camera ok, frame {frame.shape}" if ok else "camera opened but returned no frame"
    finally:
        capture.release()


def main() -> int:
    package_lines, present = check_packages()
    print("EC7 environment report")
    print(f"  date             {time.strftime('%Y-%m-%dT%H:%M:%S%z')}")
    print(f"  platform         {platform.platform()}")
    print(f"  machine          {platform.machine()}")
    print(f"  python           {sys.version.split()[0]} ({sys.executable})")
    print("packages")
    print("\n".join(package_lines))
    print("devices")
    print(f"  microphone       {check_microphone()}")
    print(f"  camera           {check_camera()}")
    digest = hashlib.sha256("\n".join(present).encode()).hexdigest()[:16]
    print(f"environment digest {digest}")
    required = {"numpy", "scipy", "soundfile", "cv2", "skimage", "sklearn", "jiwer"}
    missing = required - {p.split("==")[0] for p in present}
    if missing:
        print(f"MISSING REQUIRED PACKAGES: {', '.join(sorted(missing))}")
        return 1
    print("required packages: all present")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
