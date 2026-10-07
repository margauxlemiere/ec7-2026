#!/usr/bin/env python3
"""Fetch the workshop ambiences from Freesound into `data/noise/`.

    FREESOUND_TOKEN=... python tools/fetch_ambiences.py

Five field recordings, all CC0, chosen by ear on 27/09/2026. Each one is cut to
a minute, a few seconds in to skip the fade-in, and written at 16 kHz mono: a
minute is twenty times the longest order, and the whole bed does not need to
travel with every fork.

What comes down is the high quality preview (128 kb/s mp3), which the API gives
with a token alone; the original needs an OAuth login. Mixed at 16 kHz under a
voice, the difference does not survive.

`data/noise/sources.json` records, for every bed, the recording, its author, its
licence and the excerpt kept. CC0 asks for no credit; we give it anyway.
"""
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import tempfile
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent.parent
NOISE = HERE / "data" / "noise"
API = "https://freesound.org/apiv2/sounds/{}/?fields=id,name,username,license,url,duration,previews"

# name in the corpus: (Freesound id, start of the excerpt in seconds)
BEDS = {
    "workshop-hall": (450126, 5.0),
    "sorting-centre": (519026, 5.0),
    "conveyor-belt": (519027, 5.0),
    "factory-impacts": (439401, 5.0),
    # Not an ambience: a layer. build_corpus.py lays it over half the mixes, as
    # the robot moving while the human speaks to it (scenario.md §3.5).
    "robot-motors": (614362, 5.0),
}
SECONDS = 60.0


def get(url: str, token: str) -> bytes:
    request = urllib.request.Request(url, headers={"Authorization": f"Token {token}"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def main() -> int:
    token = os.environ.get("FREESOUND_TOKEN")
    if not token:
        print("FREESOUND_TOKEN is not set: https://freesound.org/apiv2/apply")
        return 1
    NOISE.mkdir(parents=True, exist_ok=True)
    sources = {}
    for name, (sound_id, start) in BEDS.items():
        meta = json.loads(get(API.format(sound_id), token))
        with tempfile.NamedTemporaryFile(suffix=".mp3") as mp3:
            mp3.write(get(meta["previews"]["preview-hq-mp3"], token))
            mp3.flush()
            subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-ss", str(start),
                            "-t", str(SECONDS), "-i", mp3.name, "-ac", "1",
                            "-ar", "16000", "-sample_fmt", "s16",
                            str(NOISE / f"{name}.wav")], check=True)
        sources[name] = {"freesound_id": sound_id, "title": meta["name"],
                         "author": meta["username"], "licence": meta["license"],
                         "url": meta["url"], "excerpt_s": [start, start + SECONDS],
                         "from": "preview-hq-mp3, resampled to 16 kHz mono"}
        print(f"  {name:<16} {meta['name'][:48]:<48} {meta['username']}")
    (NOISE / "sources.json").write_text(json.dumps(sources, indent=2) + "\n",
                                        encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
