"""Step 5 — The guardrail.

You write `should_act`: does the robot act on the order, or refuse it and say
"not understood"? The sweep tries many thresholds and counts, for each one, the
correct actions, the wrong actions and the refusals:

    python engine.py --input files --set snr5 --asr pretrained --sweep 0.0:1.0:0.05

Read this file from top to bottom. Do what each TODO asks.
"""

# Type hints, as in step 3.
from __future__ import annotations

# Everything from step 4, and so from steps 3 and 2.
from step4 import *


def should_act(intent: str, confidence: float, state: dict) -> bool:
    # Return True to act, False to refuse.
    #
    # `intent` and `confidence` come from your `interpret` of step 3.
    # `state` is a dictionary given by the engine. state["threshold"] is the
    # confidence below which the robot refuses. The sweep changes it at each
    # run: read it from `state`, do not write a number here.

    # TODO 1. An "unparsed" order is never acted on: return False.
    if intent == "unparsed":
        return False

    # TODO 2. "stop" is special. A robot that refuses a stop keeps moving
    # towards a person. A robot that stops by mistake loses a few seconds.
    # Decide what your robot does with a stop of low confidence, write it
    # here, and say why in your report (scenario.md §3.5).
    if intent == "stop":
        return True     # On fait le choix suivant -> Règle de sécurité : quelque soit le score de confiance, le robot doit s'arrêter.

    # TODO 3. Any other order: act (True) if the confidence is at least
    # state["threshold"], refuse (False) otherwise.
    if confidence >= state["threshold"]:
        return True 
    else :
        return False
