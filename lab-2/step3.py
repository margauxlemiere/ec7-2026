"""Step 3 — The baseline, with no model at all.

You write two functions here:

* `recognise(x, rate)`: sound -> words, with templates;
* `interpret(text)`: words -> (intent, arguments, confidence).

Run them with:

    python engine.py --input files --set all
    python engine.py --input files --set all --speech words

Read this file from top to bottom. Do what each TODO asks.
"""

# The type hints below (`x: np.ndarray`, `-> str`…) say what each function
# takes and returns. Python does not check them, but your editor uses them to
# help you. This line lets us write them in the modern way.
from __future__ import annotations

# This line imports everything from step2.py: your speech_mask, your two
# renamed functions, RATE and FRAME. You do not copy them again.
from step2 import *

# numpy is already imported by step2.py. We import it again so that your editor
# knows what `np` is in this file.
import numpy as np
# json reads JSON files. JSON is a text format for lists and dictionaries.
import json
# pathlib builds paths to files, the same way on Linux, macOS and Windows.
import pathlib
# difflib compares two texts. We use it in `similarity`, below.
import difflib

# The folder of this file. `__file__` is the path of this file. `.resolve()`
# makes the path complete. `.parent` is the folder that holds the file.
HERE = pathlib.Path(__file__).resolve().parent
# The folder of the corpus. With pathlib, "/" joins two parts of a path.
DATA = HERE / "data"


# -----------------------------------------------------------------------------
# The corpus
# -----------------------------------------------------------------------------

# data/corpus.json describes the corpus: the wake word, the vocabulary, the
# orders. Open it in your editor and look at it.
#
# TODO: load it into CORPUS, as a Python dictionary. Two steps:
#   1. read the text of the file
#      (https://docs.python.org/3/library/pathlib.html#pathlib.Path.read_text)
#   2. turn the text into a dictionary with the json module.
#      (https://docs.python.org/3/library/json.html#json.loads)
# At the end, CORPUS["wake_word"] is "robot three".
CORPUS = json.loads((DATA / "corpus.json").read_text(encoding="utf-8"))

# TODO: the list of the accepted orders, without the wake word.
# CORPUS["orders"] is a list of dictionaries. Each one looks like:
#   {"text": "carry blue pallet", "intent": "carry", "arguments": "blue pallet"}
# Keep only the "text" of each order. A for loop and .append() are fine.
# At the end, ACCEPTED looks like: ["forward", "stop", "turn left", ...]
ACCEPTED = []
for order in CORPUS["orders"]:
    ACCEPTED.append(order["text"])

# -----------------------------------------------------------------------------
# How to describe a piece of sound with a few numbers
# -----------------------------------------------------------------------------
# numpy has many functions for signal processing: windows, Fourier transforms,
# means, norms… We use a few of them below. For each one, the comment says what
# it does and links to its documentation. Read the documentation when a TODO
# asks you to choose an argument.

BANDS = 22                                 # frequency bands in a description
BAND_LOW_HZ, BAND_HIGH_HZ = 250.0, 5000.0  # below 250 Hz is the rumble: ignored


def band_power(segment: np.ndarray) -> np.ndarray:
    """The mean power of `segment` in 22 frequency bands, from 250 to 5000 Hz.

    The bands are narrow at low frequencies and wide at high frequencies, like
    the ear.
    """
    # Too short to hold two windows: no description.
    if len(segment) < 288:
        return np.zeros(BANDS)

    # We cut the segment into short windows of 256 samples (16 ms). A window
    # cut out of a signal has sharp edges, and the sharp edges add false
    # frequencies to its spectrum: "spectral leakage". The Hann window is a
    # bell shape: multiplied with the samples, it brings the edges smoothly to
    # zero. https://numpy.org/doc/stable/reference/generated/numpy.hanning.html
    window = np.hanning(256)

    # The power spectrum of each window.
    #   - `range(0, len(segment) - 256, 128)`: the start of each window. A new
    #     window every 128 samples, so two neighbours overlap by half.
    #   - `segment[start:start + 256] * window`: the 256 samples, times the
    #     bell shape.
    #   - `np.fft.rfft(...)`: their Fourier transform. For 256 real samples, it
    #     gives 129 complex numbers, from 0 Hz to RATE / 2 = 8000 Hz.
    #     https://numpy.org/doc/stable/reference/generated/numpy.fft.rfft.html
    #   - `np.abs(...) ** 2`: the power at each frequency.
    # The form `[f(start) for start in ...]` is a "list comprehension": it
    # builds a list with one value per turn of the loop. It is the same as:
    #   frames = []
    #   for start in range(0, len(segment) - 256, 128):
    #       frames.append(np.abs(np.fft.rfft(segment[start:start + 256] * window)) ** 2)
    frames = [np.abs(np.fft.rfft(segment[start:start + 256] * window)) ** 2
              for start in range(0, len(segment) - 256, 128)]
    if not frames:
        return np.zeros(BANDS)

    # TODO: `frames` is a list of spectra, all of the same length (129 values).
    # We want one spectrum: at each frequency, the mean over all the windows.
    # Find the numpy function that computes a mean. It needs the argument
    # `axis`: along which direction to average. Choose its value: think of
    # `frames` as a table, one row per window, one column per frequency.
    # https://numpy.org/doc/stable/reference/generated/numpy.mean.html
    spectrogram = np.mean(frames, axis=0)

    # TODO: the index, in `spectrogram`, of BAND_LOW_HZ and of BAND_HIGH_HZ.
    # `spectrogram` has len(spectrogram) values, spread evenly from 0 Hz (index
    # 0) to RATE / 2 Hz (the last index, len(spectrogram) - 1). A frequency f
    # is at the index f / (RATE / 2) * (len(spectrogram) - 1), rounded down
    # with int(...). Keep `low` at 1 or more, and `high` at the last index or
    # less: max(...) and min(...) do that.
    low = max(1, int(BAND_LOW_HZ / (RATE / 2.0) * (len(spectrogram) - 1)))
    high = min(len(spectrogram) - 1, int(BAND_HIGH_HZ / (RATE / 2.0) * (len(spectrogram) - 1)))

    # TODO: the edges of the 22 bands, from `low` to `high`. The bands must be
    # narrow at low frequencies and wide at high frequencies: each edge is the
    # previous one times the same factor. numpy calls this a geometric
    # sequence: np.geomspace. Read its documentation, and replace each None:
    # where the sequence starts, where it stops, and how many edges 22 bands
    # need. https://numpy.org/doc/stable/reference/generated/numpy.geomspace.html
    # `.astype(int)` turns the edges into whole numbers: they are indices.
    edges = np.geomspace(low, high, BANDS +1).astype(int)

    # The mean power in each band: the mean of `spectrogram` between two
    # neighbouring edges. `zip(edges[:-1], edges[1:])` walks two lists side by
    # side: (edge 0, edge 1), (edge 1, edge 2)… so `a` and `b` are the two
    # edges of one band. `max(a + 1, b)` keeps at least one value in a narrow
    # band.
    return np.array([spectrogram[a:max(a + 1, b)].mean()
                     for a, b in zip(edges[:-1], edges[1:])])


def describe(segment: np.ndarray, noise: np.ndarray | None = None) -> np.ndarray:
    """A word, as a vector of 66 numbers: 3 slices x 22 bands.

    Three slices, because the order of the sounds matters: averaged over the
    whole word, "robot" looks like "go".
    """
    parts = []
    # np.array_split cuts an array into 3 parts of (almost) equal length, even
    # when the length is not a multiple of 3.
    # https://numpy.org/doc/stable/reference/generated/numpy.array_split.html
    for third in np.array_split(segment, 3):
        power = band_power(third)
        # If we know the noise of the room in each band, we remove it. We keep
        # at least 5 % of the power, so that nothing becomes 0 or negative.
        if noise is not None:
            power = np.maximum(power - noise, power * 0.05)
        # Log scale, like the ear, then centred: the loudness of the speaker
        # no longer matters, only the shape of the spectrum.
        logged = np.log(power + 1e-9)
        logged = logged - logged.mean()
        # np.linalg.norm is the length of a vector: the square root of the sum
        # of its squares. Dividing by it gives a vector of length 1.
        # https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html
        parts.append(logged / (np.linalg.norm(logged) + 1e-9))
    # np.concatenate puts the three vectors end to end: 3 x 22 = 66 numbers.
    return np.concatenate(parts)


def read_wav(path: pathlib.Path) -> np.ndarray:
    """The samples of a .wav file of the corpus, as numbers between -1 and 1."""
    # A .wav file stores a sound as a list of whole numbers, one per sample.
    # Here each number is a 16-bit integer ("int16"). Python reads .wav files
    # with its `wave` module: https://docs.python.org/3/library/wave.html
    import wave
    with wave.open(str(path), "rb") as fh:
        raw = np.frombuffer(fh.readframes(fh.getnframes()), dtype=np.int16)
    # TODO: we want numbers between -1 and 1. A 16-bit integer goes from
    # -32 768 to 32 767. By which number must we divide? Replace the None.
    return raw.astype(np.float64) / 32768


# One entry per word: the word -> the list of its templates (one per speaker).
_templates: dict[str, list[np.ndarray]] = {}


def templates() -> dict[str, list[np.ndarray]]:
    """A dictionary: word -> list of vectors, one per speaker.

    data/templates/ has one file per word and per speaker, for example
    "carry-a.wav" (the word "carry", said by speaker a). The folder is read
    once, the first time you call this function.
    """
    if not _templates:
        # Each .wav file of data/templates/, in alphabetical order.
        for path in sorted((DATA / "templates").glob("*.wav")):
            # `path.stem` is the file name without ".wav": "carry-a".
            # `.rsplit("-", 1)[0]` cuts at the last "-" and keeps the left
            # part: "carry".
            word = path.stem.rsplit("-", 1)[0]
            # TODO: add the description of this file to the list of `word`.
            # In words: read the file (read_wav), describe it (describe), and
            # append it to _templates[word]. The first time a word comes, it
            # has no list yet: `_templates.setdefault(word, [])` returns its
            # list, and creates an empty one if needed.
            samples=read_wav(path)
            desc=describe(samples)
            _templates.setdefault(word, []).append(desc)
    return _templates


# -----------------------------------------------------------------------------
# recognise: sound -> words
# -----------------------------------------------------------------------------

def segments_of(mask: np.ndarray) -> list[tuple[int, int]]:
    """The pieces of voice in `mask`, as a list of (first frame, end frame).

    Example: mask = [F, T, T, F, F, T, T, T, F] gives [(1, 3), (5, 8)].
    """
    out = []
    start = None                       # where the current piece started
    # We add one False at the end, so that a piece at the very end also closes.
    # `enumerate` gives, at each turn, the position i and the value `live`.
    for i, live in enumerate(list(mask) + [False]):
        # TODO: two cases, in words:
        #   - frame i is voice, and no piece is open: a piece starts at i.
        #   - frame i is not voice, and a piece is open: the piece ends at i.
        #     Keep it only if it lasts 4 frames (40 ms) or more. Then no piece
        #     is open any more.
        if live and start is None:
            start = i
        elif not live and start is not None:
            if (i - start) >= 4:
                out.append((start, i))
            start = None
    return out


# A noise in the workshop can pass your mask and become one more word. A piece
# much weaker than the loudest one is not a word. How much weaker, in dB?
SEGMENT_FLOOR_DB = 14  # TODO: a number. Try 14, then look at trace.jsonl.

# To measure the noise of the room, we need enough silence: at least two of
# the windows of `band_power`, which are 256 samples long.
MIN_SILENCE_SAMPLES = 512  # TODO: how many samples is that?


def recognise(x: np.ndarray, rate: int) -> str:
    # 1. Where is the voice? Your function of step 2.
    mask = speech_mask(x, rate)

    # 2. The noise of the room, in each band: the power of the samples that
    #    are not voice.
    #    - `~mask` is the opposite of mask: True where there is no voice.
    #    - `np.repeat(~mask, FRAME)` repeats each value FRAME times: one value
    #      per frame becomes one value per sample.
    #      https://numpy.org/doc/stable/reference/generated/numpy.repeat.html
    #    - `[:len(x)]` keeps no more values than there are samples.
    quiet = np.repeat(~mask, FRAME)[:len(x)]
    #    `x[:len(quiet)][quiet]` keeps only the samples where `quiet` is True.
    #    `quiet.sum()` counts them: True counts as 1.
    noise = band_power(x[:len(quiet)][quiet]) if quiet.sum() > MIN_SILENCE_SAMPLES else None

    # 3. The pieces of voice, and the power of each one.
    pieces = segments_of(mask)
    if not pieces:
        return ""
    # TODO: replace the None. For each piece (a, b), we want the mean power of
    # its samples:
    #   1. the samples from frame a to frame b of x,
    #   2. converted to sample positions with FRAME,
    #   3. squared, before np.mean takes their mean.
    power = [np.mean(x[a * FRAME:b * FRAME] ** 2) for a, b in pieces]
    loudest = max(power)

    words = []
    # `zip(pieces, power)` walks the two lists side by side.
    for (a, b), p in zip(pieces, power):
        # TODO: skip this piece (`continue`) if its power `p` is more than
        # SEGMENT_FLOOR_DB below `loudest`. In power, -14 dB is the factor
        # 10 ** (-14 / 10).
        if p < loudest * 10 ** (-SEGMENT_FLOOR_DB / 10):
            continue

        vector = describe(x[a * FRAME:b * FRAME], noise)
        # 4. The word whose template is the closest to `vector`. Both vectors
        #    have length 1, so their dot product `vector @ template` is the
        #    cosine of their angle: 1 when they are the same, lower otherwise.
        best_word, best_score = None, -2.0
        # `.items()` gives each word with its list of templates.
        for word, examples in templates().items():
            for template in examples:
                # TODO: if this template is closer than the best one so far,
                # remember its word and its score.
                score = float(vector @ template)
                if score > best_score:
                    best_word, best_score = word, score
        if best_word is not None:
            words.append(best_word)
    # " ".join(words) puts the words in one text, separated by spaces.
    return " ".join(words)


# -----------------------------------------------------------------------------
# interpret: words -> order
# -----------------------------------------------------------------------------

def similarity(a: str, b: str) -> float:
    """How close two texts are: 1.0 when they are the same, 0.0 when nothing
    matches. difflib counts the letters the two texts share, in order.
    https://docs.python.org/3/library/difflib.html#difflib.SequenceMatcher.ratio
    """
    return difflib.SequenceMatcher(None, a, b).ratio()


# Below this similarity, an order is "unparsed": the robot does not guess.
UNPARSED_BELOW = 0.45  # TODO: a number between 0 and 1. Try 0.45.


def interpret(text: str) -> tuple[str, str | None, float]:
    # `text` looks like "robot three carry blue pallet".
    # Return a tuple of three values: (intent, arguments, confidence), for
    # example ("carry", "blue pallet", 0.93). An order you do not understand
    # returns ("unparsed", None, confidence): no crash, no guess.

    # TODO 1. If `text` does not start with the wake word, CORPUS["wake_word"],
    # return ("unparsed", None, 0.0).
    # https://docs.python.org/3/library/stdtypes.html#str.startswith
    wake = CORPUS["wake_word"]
    if not text.startswith(wake):
        return ("unparsed", None, 0.0)

    # TODO 2. `rest` is the text after the wake word. `text[n:]` is the text
    # without its first n characters; `.strip()` removes the spaces around it.
    rest = text[len(wake):].strip()

    # TODO 3. The accepted order (in ACCEPTED) with the highest similarity to
    # `rest`, and that similarity: it is your confidence. In words: start with
    # no best order and a confidence of 0; for each order of ACCEPTED, if its
    # similarity to `rest` is higher than the confidence, it becomes the best
    # order, and its similarity the confidence.
    best, confidence = None, 0.0
    for order in ACCEPTED:
        s = similarity(rest, order)
        if s > confidence:
            best, confidence = order, s

    # TODO 4. If the confidence is below UNPARSED_BELOW, return
    # ("unparsed", None, confidence).
    if confidence < UNPARSED_BELOW:
        return ("unparsed", None, confidence)

    # TODO 5. Find the order whose "text" is `best` in CORPUS["orders"], and
    # return its "intent", its "arguments", and the confidence.
    # The engine also looks at how you treat "stop": read step 5 first.
    for order in CORPUS["orders"]:
        if order["text"] == best:
            return (order["intent"], order["arguments"], confidence)
    
    return "unparsed", None, 0.0
