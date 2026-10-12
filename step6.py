"""Step 6 — The action policy.

You write one function here: `policy`. It takes what the robot sees, and
returns what the robot does. Run it with:

    python engine.py --task policy

The engine replays an approach of 12 steps: a stop sign from 2.4 m to 0.3 m,
the moment it leaves the field of view, a person in the lane, a few signs, and
two detections of low confidence. The detections are given: this step
measures the policy alone. The engine names the forbidden outcomes you
produce, and writes results/policy.csv.

(Steps 4 and 5 have no new file: you change step2.py and step3.py, and
measure.)

Read this file from top to bottom. Do what each TODO asks.
"""

# Type hints, as in step 2.
from __future__ import annotations

# Everything from step 3, and so from step 2. The policy does not need the
# detectors: they are here because each step builds on the one before.
from step3 import *


# -----------------------------------------------------------------------------
# The distances and the confidences of the policy
# -----------------------------------------------------------------------------
# scenario.md §4.2 and slide 39 of the lecture ("The action policy") give the
# situations: a stop sign at 1.2 m, the same one at 0.3 m, the moment it
# leaves the field of view, and `unknown`.

# From this distance, a stop sign makes the robot slow down.
SLOW_FROM_M = None  # TODO: the first distance of scenario.md §4.2, in metres

# From this distance, the robot stops. At 0.3 m it must already be stopped:
# the engine forbids a robot still moving at 0.3 m. The robot needs room to
# brake: choose a distance between 0.3 and SLOW_FROM_M.
STOP_FROM_M = None  # TODO: your choice. Try 0.7.

# A stop sign seen with a confidence below this value: the robot is not sure.
# A robot that stops for nothing loses a few seconds; a robot that runs a stop
# sign breaks the rule. So, not sure of a close stop sign: "stop".
STOP_TRUST = None  # TODO: a number between 0 and 1. Try 0.5.

# Any other sign seen with a confidence below this value: the robot does not
# act on it, and says so.
ACT_TRUST = None  # TODO: a number between 0 and 1. Try 0.4.


def policy(detection: dict, state: dict) -> str:
    # Return one action, as a text: "forward", "slow", "stop", "turn_left",
    # "turn_right" or "report" (the robot stops and says what it does not
    # know).
    #
    # `detection` is a dictionary, like what your detectors return:
    #   {"target": "sign-stop", "distance_m": 1.2, "bearing_deg": 2.0,
    #    "confidence": 0.93}
    # `state` is a dictionary given by the engine:
    #   state["speed"]            the speed of the robot, in m/s
    #   state["person_ahead"]     True if a person is in the lane
    #   state["history"]          the steps already played, oldest first. Each
    #                             one is a dictionary: {"detection": …,
    #                             "state": …, "action": …}
    # The history is the memory of the robot: a stop sign that has left the
    # field of view is still there.

    # The four values we use. d.get(key, default) gives d[key], or `default`
    # when the dictionary has no such key.
    # https://docs.python.org/3/library/stdtypes.html#dict.get
    target = detection.get("target", "unknown")
    distance = detection.get("distance_m")
    confidence = detection.get("confidence", 0.0)
    history = state.get("history", [])

    # TODO 1. A person in the lane (state.get("person_ahead") is True): return
    # "stop". Nothing else matters.

    # Did the robot stop in one of the last three steps? `history[-3:]` is
    # the last three elements of the list (fewer if the list is shorter).
    # any(…) is True if at least one of the values is True.
    # https://docs.python.org/3/library/functions.html#any
    stopped_here = any(h["action"] == "stop" for h in history[-3:])

    # TODO 2. A stop sign (target is "sign-stop"). In words:
    #   - if its distance is known (not None) and at most SLOW_FROM_M:
    #       return "stop" if the distance is at most STOP_FROM_M, or if the
    #       confidence is below STOP_TRUST; return "slow" otherwise;
    #   - else (farther, or no distance): return "slow".

    # The targets of the last three steps: a list comprehension, as in step 2.
    recent = [h["detection"]["target"] for h in history[-3:]]

    # TODO 3. Nothing seen (target is "unknown"). In words:
    #   - if "sign-stop" is in `recent`, and the robot has not stopped here
    #     yet (not stopped_here): the sign has just left the field of view,
    #     it did not disappear. Return "stop".
    #   - otherwise, return "report": stopped in the lane with nothing said is
    #     forbidden.

    # TODO 4. Any other sign, seen with a confidence below ACT_TRUST: return
    # "report".

    # TODO 5. The directions: "sign-left" gives "turn_left", "sign-right"
    # gives "turn_right". "sign-down" asks to go back, which the robot cannot
    # do in a lane: "stop".

    # TODO 6. Anything else ("sign-up", "sign-free"): the way is open. Replace
    # "stop" with "forward" in the line below, once the TODOs above are done.
    return "stop"
