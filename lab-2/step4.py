"""Step 4 — The pre-trained option.

The baseline of step 3 knows twenty words and nothing else. Here you use a
pre-trained recogniser: Whisper, a neural network trained by OpenAI on 680 000
hours of speech. You write three functions:

* `load_model()`: load Whisper once;
* `transcribe(x, rate)`: sound -> text, with Whisper;
* `normalise(text)`: Whisper writes "Robot 3, carry blue pallet." where your
  `interpret` expects "robot three carry blue pallet".

The engine times `transcribe`, calls `normalise` on the text of both
recognisers, then your `interpret` of step 3.

    python engine.py --input files --set all --asr pretrained

Read this file from top to bottom. Do what each TODO asks.
"""

# Type hints, as in step 3.
from __future__ import annotations

# Everything from step 3, and so from step 2.
from step3 import *

# numpy is already imported by step 2. We import it again so that your editor
# knows what `np` is in this file.
import numpy as np

# faster-whisper runs Whisper on a CPU, faster than the original code of
# OpenAI. You installed it in step 1 (requirements.txt). Its documentation,
# with examples: https://github.com/SYSTRAN/faster-whisper
from faster_whisper import WhisperModel


# -----------------------------------------------------------------------------
# Whisper
# -----------------------------------------------------------------------------

# Whisper comes in several sizes: "tiny", "base", "small", "medium",
# "large-v3". A bigger model makes fewer errors, and is slower and heavier.
# The names ending in ".en" ("tiny.en", "base.en"…) know English only, and do
# better on English at the same size. The first time, the library downloads
# the model (tiny: 75 MB; large: 3 GB).
# Which one? This model has to run on the robot's CPU. The open ASR
# leaderboard, in the reading list of this session, compares their word error
# rate and their speed.
MODEL_SIZE = "tiny.en"  # TODO: a text, for example "base.en". Start small.

# The loaded model. None until load_model() runs.
_model = None


def load_model() -> WhisperModel:
    """Load Whisper once, and keep it: loading takes seconds, transcribing
    takes milliseconds. The engine calls this before it starts its clock."""
    global _model                  # we change the `_model` of the whole file
    if _model is None:
        # TODO: replace the two None. WhisperModel(size, device=..., compute_type=...)
        #   - device: where the computation runs, "cpu" or "cuda" (a GPU). The
        #     robot has no GPU.
        #   - compute_type: how the numbers of the network are stored. "int8"
        #     stores each weight on 8 bits instead of 32: four times smaller,
        #     faster on a CPU, a little less precise. This is "quantisation".
        #     "float32" keeps the full precision.
        # https://github.com/SYSTRAN/faster-whisper#usage
        _model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
    return _model


def transcribe(x: np.ndarray, rate: int) -> str:
    """The text Whisper hears in `x`."""
    model = load_model()

    # Whisper expects 16 000 samples per second (our files have exactly that)
    # and numbers of type float32 (ours are float64). `.astype` converts.
    audio = x.astype(np.float32)

    # TODO: replace the two None. model.transcribe(audio, language=..., beam_size=...)
    #   - language: the language code. Without it, Whisper first guesses the
    #     language, which costs time and sometimes fails on 2 seconds of sound.
    #     The code of English is "en".
    #   - beam_size: how many candidate sentences Whisper keeps while it
    #     decodes, word after word. 1 keeps only the most likely one at each
    #     word: the fastest. 5 is more accurate, and slower.
    # It returns two things: the segments of text, and information about the
    # audio (`_info`, not used here).
    segments, _info = model.transcribe(audio, language="en", beam_size=1)

    # TODO: `segments` gives the pieces of text one by one: each `segment` has
    # a `.text`. Join all the `.text` into one text, separated by spaces, and
    # return it. A for loop, or " ".join(...) with a comprehension, as in
    # step 3.
    return " ".join([segment.text for segment in segments])


# -----------------------------------------------------------------------------
# normalise: the text of Whisper -> the text your interpret expects
# -----------------------------------------------------------------------------

# Each digit, written as a word. A dictionary: DIGITS["3"] is "three".
DIGITS = {"0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
          "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine"}


def normalise(text: str) -> str:
    # What we want:
    #   "Robot 3, carry blue pallet."  ->  "robot three carry blue pallet"
    # Lower case, digits as words, no punctuation, one space between words.

    # TODO 1. The text in lower case.
    # https://docs.python.org/3/library/stdtypes.html#str.lower
    lowered = text.lower()

    # 2. We go through the characters of `lowered` one by one, and build a new
    #    text `out`. A `for` loop over a text gives one character at a time.
    out = ""
    for c in lowered:
        # TODO: three cases, in words:
        #   - c is a digit: add its word from DIGITS, with a space before and
        #     after it (so that "3," does not stick to the next word);
        #   - c is a letter, or a space: add c as it is;
        #   - anything else (a comma, a full stop…): add a space.
        # Useful: c.isdigit() and c.isalpha() are True or False.
        # https://docs.python.org/3/library/stdtypes.html#str.isdigit
        # `out = out + something` adds text at the end of `out`.
        if c.isdigit():
            # Si c'est un chiffre, on le remplace par son mot entouré d'espaces
            out += f" {DIGITS[c]} "
        elif c.isalpha() or c.isspace():
            # Si c'est une lettre ou un espace, on conserve le caractère
            out += c
        else:
            # Pour la ponctuation ou caractères spéciaux, on ajoute un espace
            out += " "

    # 3. `out.split()` cuts the text at every run of spaces: a list of words.
    #    " ".join(...) glues them back with exactly one space.
    return " ".join(out.split())
