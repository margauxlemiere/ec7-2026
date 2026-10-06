"""Step 2 — Voice activity detection.

You write one function here: `speech_mask`. For each 10 ms of sound, it says
if somebody is speaking. Run it with:

    python engine.py --input files --set all --task vad

Read this file from top to bottom. Do what each TODO asks.
"""

# The type hints below (`x: np.ndarray`, `-> np.ndarray`…) say what each
# function takes and returns. Python does not check them, but your editor uses
# them to help you. This line lets us write them in the modern way.
from __future__ import annotations

# numpy is the Python library for arrays of numbers. A sound is an array: one
# number per sample. Everybody imports it under the short name `np`.
import numpy as np

# Every file of the corpus has 16 000 samples per second.
RATE = 16000

# We cut the sound into frames of 10 ms: 16 000 / 100 = 160 samples per frame.
# The engine reads your answer at this rate: one True or False per frame.
FRAME = RATE // 100


# -----------------------------------------------------------------------------
# Two functions are given. Read them line by line. Then give each one a name
# that says what it does.
# -----------------------------------------------------------------------------

# TODO: this function is a filter. Which kind? Use the name from your physics
# course, in two words joined by "_". Rename it here, and in `speech_mask`
# below.
def highpass_filter(x: np.ndarray, cutoff_hz: float = 300.0) -> np.ndarray:
    # Why we need it: the workshop rumbles below 200 Hz, and a voice lives
    # between 300 and 3400 Hz. We want to count the energy of the voice, not
    # the energy of the rumble.

    # How many samples in one period of a 300 Hz wave? 16 000 / 300 = 53.
    width = int(RATE / cutoff_hz)
    # An array of `width` values, all equal to 1 / width. Their sum is 1.
    kernel = np.ones(width) / width
    # np.convolve slides `kernel` along `x`. At each sample, the result is the
    # mean of the `width` samples around it: a moving average. A moving
    # average keeps only the slow changes of the signal. `mode="same"` keeps
    # the result as long as `x`.
    # https://numpy.org/doc/stable/reference/generated/numpy.convolve.html
    slow = np.convolve(x, kernel, mode="same")
    # The signal minus its slow part: only the fast changes are left.
    return x - slow


# TODO: this function computes one number per frame. Which physical quantity?
# Give it a name in two words joined by "_". Rename it here, and in
# `speech_mask` below.
def frame_energy(x: np.ndarray) -> np.ndarray:
    # How many whole frames in x? `//` is the integer division.
    n = len(x) // FRAME
    # Keep the first n * FRAME samples. Arrange them as a table with n rows
    # (one per frame) and FRAME columns (one per sample).
    # https://numpy.org/doc/stable/reference/generated/numpy.reshape.html
    table = x[:n * FRAME].reshape(n, FRAME)
    # Square every sample. Then take the mean of each row: `axis=1` means
    # "along the row". The result has one number per frame.
    # We add a very small number, so that we never take the log of 0 later.
    return (table ** 2).mean(axis=1) + 1e-12


# -----------------------------------------------------------------------------
# Your function.
# -----------------------------------------------------------------------------

# A frame is voice when its energy is well above the noise of the room.
# "Well above" is this margin, in decibels. It is your only setting in this
# step. Slide 15 of the lecture, and seance_2/slides/code/vad.py, give a first
# value. Start there. Then run the engine, and change the margin until no clean
# order is clipped.
MARGIN_DB = 9.0
 # TODO: a number, in dB


def speech_mask(x: np.ndarray, rate: int) -> np.ndarray:
    # What we want: an array of True and False, one per frame. True where
    # somebody speaks.

    # 1. The energy of each frame, in decibels, after the filter.
    #    (When you rename the two functions above, rename them here too.)
    energy_db = 10 * np.log10(frame_energy(highpass_filter(x)))

    # 2. The noise of the room. Most of the time, nobody speaks: the 10 % most
    #    quiet frames are the room. np.percentile(a, 10) is the value below
    #    which 10 % of the values of `a` are.
    #    https://numpy.org/doc/stable/reference/generated/numpy.percentile.html
    floor = np.percentile(energy_db, 10)

    # 3. A frame is voice if its energy is MARGIN_DB above the floor. This line
    #    compares all the frames at once: `mask` is an array of True and False.
    mask = energy_db > floor + MARGIN_DB

    # 4. Inside a word, the energy can drop for one frame, for example just
    #    before a "t" or a "p". That frame is not a silence between two words.
    #    Fill these holes.
    #    For each frame i, except the first one and the last one:
    for i in range(1, len(mask) - 1):
        # TODO: write this condition in Python. In words:
        #   frame i is not voice, and frame i - 1 is voice, and frame i + 1 is
        #   voice.
        # You need `not`, `and`, mask[i], mask[i - 1] and mask[i + 1].
        if not mask[i] and mask[i - 1] and mask[i + 1]:
            mask[i] = True

    return mask
