"""Speech for the lab corpus, from whatever this machine can actually say.

Five backends, tried in this order, and the one that answered is written into
`data/manifest.csv` so that no number is ever read without knowing what made it:

    kokoro     Kokoro-82M, Apache-2.0, local, on the CPU, about real time.
               `pip install kokoro`. Two of its voices are the two speakers
    piper      piper-tts, a local neural voice
    say        the macOS system voice
    espeak     espeak-ng, a formant synthesiser
    tone       the fallback below: no speech at all, a signal shaped like it

The fallback is here so that the engine runs on a machine with none of the
above. It carries the timing of speech and enough spectral difference between
words for template matching to work, and it is not speech: a pre-trained
recogniser transcribes nothing from it. `engine.py` refuses to report the
pre-trained row on a `tone` corpus, and says why.
"""
from __future__ import annotations

import ctypes.util
import math
import pathlib
import shutil
import subprocess
import sys
import tempfile
import wave

import numpy as np

RATE = 16000

# A vowel is its two first formants; a consonant is a burst or a hiss. Enough
# to tell "left" from "right" on a template, and not one word more.
VOWELS = {"a": (700, 1220), "e": (530, 1840), "i": (270, 2290),
          "o": (570, 840), "u": (300, 870), "y": (300, 1700)}
WORDS = {
    "robot": [("v", "o", .10), ("p", "b", .03), ("v", "o", .09), ("p", "t", .03)],
    "three": [("f", "s", .09), ("v", "i", .13)],
    "forward": [("f", "f", .07), ("v", "o", .10), ("v", "u", .08), ("p", "d", .03)],
    "stop": [("f", "s", .09), ("p", "t", .03), ("v", "o", .10), ("p", "p", .03)],
    "turn": [("p", "t", .03), ("v", "u", .13), ("v", "a", .05)],
    "left": [("v", "e", .12), ("f", "f", .07), ("p", "t", .03)],
    "right": [("v", "a", .09), ("v", "i", .08), ("p", "t", .03)],
    "carry": [("p", "k", .03), ("v", "a", .10), ("v", "i", .09)],
    "blue": [("p", "b", .03), ("v", "u", .15)],
    "pallet": [("p", "p", .03), ("v", "a", .09), ("v", "e", .09)],
    "red": [("v", "e", .13), ("p", "d", .03)],
    "make": [("p", "m", .04), ("v", "e", .12), ("p", "k", .03)],
    "green": [("p", "g", .03), ("v", "i", .13), ("v", "a", .05)],
    "crate": [("p", "k", .03), ("v", "e", .13), ("p", "t", .03)],
    "grey": [("p", "g", .03), ("v", "e", .14)],
    "box": [("p", "b", .03), ("v", "o", .11), ("f", "s", .08)],
    "now": [("p", "n", .04), ("v", "a", .13)],
    "go": [("p", "g", .03), ("v", "o", .13)],
    "why": [("f", "f", .05), ("v", "a", .09), ("v", "i", .08)],
    "the": [("p", "d", .03), ("v", "u", .06)],
}
# Four speakers, because one template set that only ever met one voice is not a
# recogniser, it is a memory. Kokoro publishes 54 voices; these four differ in
# pitch and in accent, which is what a closed-vocabulary matcher has to survive.
SPEAKERS = {
    "a": {"f0": 112, "kokoro": "am_michael", "piper": "en_US-lessac-medium",
          "say": "Alex", "espeak": "en-us+m3", "rate": 1.00},
    "b": {"f0": 196, "kokoro": "af_heart", "piper": "en_US-amy-medium",
          "say": "Samantha", "espeak": "en-us+f3", "rate": 1.08},
    "c": {"f0": 128, "kokoro": "bm_george", "piper": "en_GB-alan-medium",
          "say": "Daniel", "espeak": "en-gb+m1", "rate": 0.95},
    "d": {"f0": 210, "kokoro": "bf_emma", "piper": "en_GB-jenny_dioco-medium",
          "say": "Karen", "espeak": "en-gb+f2", "rate": 1.04},
}
KOKORO_RATE = 24000


# ------------------------------------------------------------------- the fallback
def vowel(symbol: str, seconds: float, f0: float, rng) -> np.ndarray:
    """A voiced sound: harmonics of f0, shaped by two formant peaks."""
    n = int(seconds * RATE)
    t = np.arange(n) / RATE
    f1, f2 = VOWELS[symbol]
    out = np.zeros(n)
    for k in range(1, int(RATE / 2 / f0)):
        frequency = k * f0
        gain = (math.exp(-((frequency - f1) / 180) ** 2)
                + 0.7 * math.exp(-((frequency - f2) / 260) ** 2)
                + 0.05 / k)
        out += gain * np.sin(2 * math.pi * frequency * t + rng.uniform(0, 6.28))
    envelope = np.minimum(1, np.minimum(t, seconds - t) / 0.02)
    return out / (np.max(np.abs(out)) + 1e-9) * envelope


def burst(symbol: str, seconds: float, rng) -> np.ndarray:
    """A plosive or a fricative: a silence and a shaped noise."""
    n = int(seconds * RATE)
    noise = rng.normal(0, 1, n)
    if symbol in "sfh":                                # hiss, high
        noise *= np.linspace(0.6, 1.0, n)
        kernel = np.array([1.0, -0.85])
    else:                                              # plosive, short and low
        noise[: n // 3] *= 0.05
        kernel = np.array([1.0, 0.75, 0.35])
    out = np.convolve(noise, kernel, mode="same")
    return out / (np.max(np.abs(out)) + 1e-9) * 0.45


LETTER_VOWEL = {"a": "a", "e": "e", "i": "i", "o": "o", "u": "u", "y": "y"}


def phones_for(word: str) -> list[tuple[str, str, float]]:
    """The phones of a word, from the table, or spelled out from its letters.

    The table covers the order set. Everything else is spelled: the `onboard`
    voice has to be able to say a sentence nobody prepared, which is the whole
    difference between it and the `recorded` one.
    """
    if word in WORDS:
        return WORDS[word]
    out = []
    for letter in word[:6]:
        if letter in LETTER_VOWEL:
            out.append(("v", LETTER_VOWEL[letter], 0.10))
        elif letter.isalpha():
            out.append(("p", "t" if letter in "tdkgpb" else "s", 0.035))
    return out or [("v", "u", 0.10)]


def synthesise(words: list[str], speaker: str, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    voice = SPEAKERS[speaker]
    pieces = []
    for word in words:
        for kind, symbol, seconds in phones_for(word):
            seconds *= voice["rate"] * rng.uniform(0.92, 1.08)
            if kind == "v":
                pieces.append(vowel(symbol, seconds, voice["f0"], rng))
            else:
                pieces.append(burst(symbol, seconds, rng))
        pieces.append(np.zeros(int(0.06 * RATE)))      # the gap between words
    return np.concatenate(pieces) * 0.6


# ------------------------------------------------------------------- the real ones
def read_wav(path: pathlib.Path) -> np.ndarray:
    with wave.open(str(path), "rb") as fh:
        raw = fh.readframes(fh.getnframes())
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float64) / 32768.0
        if fh.getnchannels() == 2:
            data = data.reshape(-1, 2).mean(axis=1)
        data = resample(data, fh.getframerate(), RATE)
    return data


def write_wav(path: pathlib.Path, signal: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(signal, -1, 1)
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(RATE)
        fh.writeframes((clipped * 32767).astype("<i2").tobytes())


_kokoro = {}


def use_system_espeak() -> None:
    """Point the phonemiser at the system espeak-ng, when there is one.

    Kokoro phonemises through `misaki`, which pins the espeak-ng that
    `espeakng-loader` bundles. On some machines that build looks for its phoneme
    tables under the path of the machine that compiled it, and the C library
    stops there. A system espeak-ng knows where its own data is.
    """
    binary = shutil.which("espeak-ng")
    if not binary:
        return
    try:
        import misaki.espeak                              # noqa: F401 — it pins
        from phonemizer.backend.espeak.wrapper import EspeakWrapper
    except Exception:                                     # noqa: BLE001
        return
    version = subprocess.run([binary, "--version"], capture_output=True, text=True)
    data = version.stdout.partition("Data at:")[2].strip()
    library = ctypes.util.find_library("espeak-ng")
    if library:
        EspeakWrapper.set_library(library)
    if data and pathlib.Path(data).is_dir():
        EspeakWrapper.set_data_path(data)


def kokoro_pipeline():
    """Kokoro-82M, loaded once. Apache-2.0, on the CPU, no account, and no
    network after the first run."""
    if "pipeline" not in _kokoro:
        try:
            use_system_espeak()
            from kokoro import KPipeline
            _kokoro["pipeline"] = KPipeline(lang_code="a")      # American English
        except Exception:                                       # noqa: BLE001
            _kokoro["pipeline"] = None
    return _kokoro["pipeline"]


def kokoro_speak(sentence: str, voice: str) -> np.ndarray:
    pipeline = kokoro_pipeline()
    chunks = [np.asarray(audio, dtype=np.float64)
              for _gs, _ps, audio in pipeline(sentence, voice=voice)]
    signal = np.concatenate(chunks) if chunks else np.zeros(1)
    return resample(signal, KOKORO_RATE, RATE)


def resample(signal: np.ndarray, source: int, target: int) -> np.ndarray:
    """Linear, with a low pass first when going down.

    Interpolating a 32 kHz signal down to 16 kHz without cutting what lives
    above 8 kHz folds it back into the band the speech is in, and the fold is
    audible. A short moving average is a poor filter and a sufficient one here.
    """
    if source == target:
        return signal
    if target < source:
        width = max(2, int(round(source / target)))
        kernel = np.ones(width) / width
        signal = np.convolve(signal, kernel, mode="same")
    n = int(len(signal) * target / source)
    return np.interp(np.linspace(0, len(signal) - 1, n),
                     np.arange(len(signal)), signal)


def trim(signal: np.ndarray, floor: float = 2e-3) -> np.ndarray:
    """Drop the silence a synthesiser puts at both ends of what it says."""
    live = np.where(np.abs(signal) > floor)[0]
    return signal[live[0]:live[-1] + 1] if len(live) else signal


def speak_words(words: list[str], speaker: str, backend: str, seed: int,
                gap_s: float = 0.22) -> tuple[np.ndarray, list[tuple[float, float]]]:
    """One word at a time, with a gap between them: (signal, word boundaries).

    The baseline of `scenario.md` §3.6 is isolated word recognition over a
    closed vocabulary, and this is what it hears. The human pays for it by
    speaking like that, which is a line of the grid and not an accident of the
    corpus.
    """
    rng = np.random.default_rng(seed)
    pieces, bounds, cursor = [], [], 0.0
    for i, word in enumerate(words):
        audio = trim(speak(word, speaker, backend, seed + i))
        bounds.append((cursor, cursor + len(audio) / RATE))
        pieces.append(audio)
        cursor += len(audio) / RATE
        if i < len(words) - 1:
            gap = gap_s * rng.uniform(0.85, 1.2)
            pieces.append(np.zeros(int(gap * RATE)))
            cursor += gap
    return np.concatenate(pieces), bounds


def available() -> str:
    if kokoro_pipeline() is not None:
        return "kokoro"
    if shutil.which("piper"):
        return "piper"
    if sys.platform == "darwin" and shutil.which("say") and shutil.which("afconvert"):
        return "say"
    if shutil.which("espeak-ng") or shutil.which("espeak"):
        return "espeak"
    return "tone"


def speak(sentence: str, speaker: str, backend: str, seed: int) -> np.ndarray:
    if backend == "tone":
        return synthesise(sentence.split(), speaker, seed)
    voice = SPEAKERS[speaker]
    if backend == "kokoro":
        return kokoro_speak(sentence, voice["kokoro"])
    with tempfile.TemporaryDirectory() as tmp:
        out = pathlib.Path(tmp) / "utterance.wav"
        if backend == "piper":
            subprocess.run(["piper", "--model", voice["piper"],
                            "--output_file", str(out)],
                           input=sentence.encode(), check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        elif backend == "say":
            aiff = pathlib.Path(tmp) / "utterance.aiff"
            subprocess.run(["say", "-v", voice["say"], "-o", str(aiff), sentence],
                           check=True)
            subprocess.run(["afconvert", "-f", "WAVE", "-d", f"LEI16@{RATE}",
                            str(aiff), str(out)], check=True)
        else:
            binary = shutil.which("espeak-ng") or shutil.which("espeak")
            subprocess.run([binary, "-v", voice["espeak"], "-w", str(out), sentence],
                           check=True)
        return read_wav(out)
