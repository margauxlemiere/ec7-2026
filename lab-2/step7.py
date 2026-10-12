"""Step 7 — The robot's answer, and the robot that hears itself.

You write two functions here: `answer` and `speech_mask_while_speaking`.

    python engine.py --input files --set clean --voice recorded
    python engine.py --input files --set clean --voice onboard
    python engine.py --input files --set clean --voice hosted
    python engine.py --input files --set clean --voice onboard --loopback

(Step 6 has no code: you run the engine and write your report.)

Read this file from top to bottom. Do what each TODO asks.
"""

# Type hints, as in step 3.
from __future__ import annotations

# Everything from step 5, and so from steps 4, 3 and 2.
from step5 import *

# numpy is already imported by step 2. We import it again so that your editor
# knows what `np` is in this file.
import numpy as np


# -----------------------------------------------------------------------------
# answer: what the robot says
# -----------------------------------------------------------------------------

# The sentence the robot says when it acts, one per intent. The intents are
# the "intent" values of CORPUS["orders"]: "forward", "stop", "turn_left",
# "turn_right", "carry", "make", "why".
# The `recorded` voice can only play the phrases of data/phrases/: "ok", "no",
# "not understood", "possible", "impossible", "stop", "go forward",
# "turn left", "turn right". Keep to these if you want it to speak.
SENTENCES: dict[str, str] = {
    "forward": "go forward",
    "stop": "stop",
    # TODO: add "turn_left" and "turn_right".
    "turn_left": "turn left",
    "turn_right": "turn right",
}


def answer(action: str, reason: str) -> str:
    # `action` is an intent ("carry", "stop"…), or "refuse" when the robot
    # does not act. `reason` explains a refusal, for example
    # "conveyor two busy with robot one waiting forty seconds". It can be "".

    # TODO 1. If `action` is "refuse": return `reason` if it is not empty,
    # else "not understood".
    if action == "refuse":
        return reason if reason else "not understood"

    # TODO 2. Otherwise: the sentence of `action` in SENTENCES, or "ok" when
    # SENTENCES has none. A dictionary does that in one call:
    # https://docs.python.org/3/library/stdtypes.html#dict.get
    return SENTENCES.get(action, "ok")


# -----------------------------------------------------------------------------
# speech_mask_while_speaking: the robot hears its own voice
# -----------------------------------------------------------------------------

# With --loopback, the robot's own voice comes back into its microphone, and
# your speech_mask of step 2 fires on it. If you mute the microphone while the
# robot speaks, nobody can interrupt it. We do better: while the robot speaks,
# a frame counts as a human only if it is clearly louder than the robot.
# How many dB louder?
BARGE_IN_DB = 3.0  # TODO: a number, in dB. Try 3.

# The level of the robot's voice: a high percentile of the energy of the
# frames where it speaks, so that its loudest syllables do not count as a
# human. Which percentile?
ROBOT_PERCENTILE = 90  # TODO: a number between 50 and 100. Try 90.


def speech_mask_while_speaking(x: np.ndarray, rate: int,
                               speaking: np.ndarray) -> np.ndarray:
    # `speaking` has one True or False per frame: True while the robot speaks.

    # 1. Your mask of step 2, as usual.
    mask = speech_mask(x, rate)

    # TODO 2. The energy of each frame, in dB: the same line as the first line
    # of speech_mask. Replace None with your two functions of step 2.
    energy_db = 10 * np.log10(frame_energy(highpass_filter(x)) + 1e-9)

    # 3. `speaking` may not have the same length as `mask`: cut it.
    #    np.asarray(..., dtype=bool) makes sure it is an array of True/False.
    speaking = np.asarray(speaking, dtype=bool)[:len(mask)]

    # 4. The energy of the frames where the robot speaks. Indexing an array
    #    with an array of True/False keeps only the values where it is True.
    own = energy_db[:len(speaking)][speaking]

    # TODO 5. The level of the robot's voice: the ROBOT_PERCENTILE percentile
    # of `own`, with np.percentile, as in step 2.
    robot = np.percentile(own, ROBOT_PERCENTILE)

    # TODO 6. `louder`: True for each frame whose energy is more than
    # BARGE_IN_DB above `robot`. One comparison of the whole array, like line
    # 3 of speech_mask. Use energy_db[:len(speaking)].
    louder = energy_db[:len(speaking)] > (robot + BARGE_IN_DB)

    # 7. While the robot speaks, keep `louder`; the rest of the time, keep
    #    `mask`. np.where(condition, a, b) takes a where the condition is True,
    #    and b elsewhere.
    #    https://numpy.org/doc/stable/reference/generated/numpy.where.html
    mask[:len(speaking)] = np.where(speaking, louder, mask[:len(speaking)])
    return mask
