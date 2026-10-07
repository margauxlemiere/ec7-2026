"""The robot's voice. Provided, three options behind one flag. Do not modify it.

    recorded   a fixed set of phrases, played back. Nothing else can be said
    onboard    synthesised here and now, on this CPU
    hosted     a remote service, simulated with a stated network delay

Each one answers `(audio, latency_s, note)`, where `latency_s` is the time to
**first sound**, which is what the human waits through. It is measured for
`recorded` and `onboard`; for `hosted` it is a stated figure, because nobody has
an account today, and the engine labels that column as such.
"""
from __future__ import annotations

import pathlib
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "tools"))
import voices                                                    # noqa: E402

RATE = voices.RATE
HOSTED_DELAY_S = 0.62        # round trip to a hosted service, stated, not measured
OPTIONS = ("recorded", "onboard", "hosted")


def recorded(sentence: str):
    """Play a phrase from the fixed set. Refuse anything that is not in it."""
    start = time.perf_counter()
    name = sentence.strip().lower().replace(" ", "-")
    path = HERE / "data" / "phrases" / f"{name}.wav"
    if not path.exists():
        return None, time.perf_counter() - start, f"no recording for {sentence!r}"
    return voices.read_wav(path), time.perf_counter() - start, None


def onboard(sentence: str):
    """Synthesise it here. Costs CPU, says anything."""
    start = time.perf_counter()
    backend = voices.available()
    audio = voices.speak(sentence, "a", backend, 17)
    return audio, time.perf_counter() - start, f"backend {backend}"


def hosted(sentence: str):
    """A remote service. The delay is stated, and nothing leaves this machine."""
    start = time.perf_counter()
    audio = voices.speak(sentence, "a", voices.available(), 17)
    return audio, (time.perf_counter() - start) + HOSTED_DELAY_S, \
        "network delay is a stated figure"


def say(sentence: str, option: str):
    if option not in OPTIONS:
        raise ValueError(f"{option!r} is not one of {OPTIONS}")
    return {"recorded": recorded, "onboard": onboard, "hosted": hosted}[option](sentence)


if __name__ == "__main__":
    for name in OPTIONS:
        audio, latency, note = say(" ".join(sys.argv[1:]) or "ok", name)
        length = 0 if audio is None else len(audio) / RATE
        print(f"{name:<9} first sound {latency * 1e3:7.1f} ms   "
              f"{length:.2f} s   {note or ''}")
