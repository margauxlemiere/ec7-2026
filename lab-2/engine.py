#!/usr/bin/env python3
"""Lab 2 — the engine. It plays, it times, it scores. **Do not modify it.**

    python engine.py --input files --set clean
    python engine.py --input files --set clean --task vad
    python engine.py --input files --set all
    python engine.py --input files --set all --asr pretrained
    python engine.py --input files --set all --speech words
    python engine.py --input files --set clean --set snr5 --intonation all
    python engine.py --input files --set snr5 --sweep 0.0:1.0:0.05
    python engine.py --input files --set clean --voice onboard --loopback
    python engine.py --input device

The degraded sets are not optional: they carry half the marks. The corpus holds
one per signal to noise ratio, from `snr30` down to `snrm5`, which is minus five
decibels, and the point of the ladder is to find the rung your option falls off.

Every order is there twice. By default you hear it as a person says it, in one
breath; `--speech words` gives the same order spoken one word at a time, which
is what an isolated word recogniser asks of the human. Said in one breath, it
also comes in three intonations: an order, cheerful, alarmed. `--intonation all`
scores them side by side, and the `stop` column says whether a stop got through.
Every number in `results.csv` was measured here; there is no way to type one in.
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import statistics
import sys
import time
import traceback

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "tools"))
import voices                                                    # noqa: E402

import say                                                       # noqa: E402
import importlib                                                 # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
DATA = HERE / "data"
if not (DATA / "corpus.json").exists():
    sys.exit("No corpus in data/. Download lab-2-data.zip from the course site "
             "and unzip it here,\nin the folder that holds engine.py: it "
             "creates data/.")
CORPUS = json.loads((DATA / "corpus.json").read_text(encoding="utf-8"))
RATE = CORPUS["rate"]
FRAME = RATE // 100                       # 10 ms, the frame of speech_mask


RESULTS = HERE / "results"

# What the pre-trained model of step 4 costs on the class of CPU the robot
# carries, measured with "tiny.en". A stated figure: you cannot measure it
# today, and the engine labels it.
ROBOT_RTF = 1.9


def keep(name: str, rows: list[dict]) -> pathlib.Path:
    """Every run leaves its own file in `results/`, named after what it ran."""
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / name
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


# ------------------------------------------------------------------- calling
# You write one file per step. Each function lives in the file of its step, and
# a file that does not load yet does not stop the steps before it.
STEP_OF = {"speech_mask": "step2", "recognise": "step3", "interpret": "step3",
           "load_model": "step4", "transcribe": "step4", "normalise": "step4", "should_act": "step5", "answer": "step7",
           "speech_mask_while_speaking": "step7"}
_modules: dict = {}
# The values you set yourself, step by step. None means "not set yet".
TODO_CONSTANTS = {"step2": ["MARGIN_DB"],
                  "step3": ["CORPUS", "SEGMENT_FLOOR_DB", "MIN_SILENCE_SAMPLES",
                            "UNPARSED_BELOW"],
                  "step4": ["MODEL_SIZE"], "step5": [],
                  "step7": ["BARGE_IN_DB", "ROBOT_PERCENTILE"]}


def module(name: str):
    if name not in _modules:
        try:
            _modules[name] = importlib.import_module(name)
        except Exception:                              # noqa: BLE001 — we report it
            _modules[name] = None
            print(f"{name}.py does not load: "
                  f"{traceback.format_exc(limit=2).strip().splitlines()[-1]}")
    return _modules[name]


def call(name: str, *args):
    step = STEP_OF[name]
    function = getattr(module(step), name, None)
    if function is None:
        return None, 0.0, f"{step}.{name} is missing"
    start = time.perf_counter()
    try:
        answer = function(*args)
    except Exception:                                  # noqa: BLE001 — we report it
        return None, (time.perf_counter() - start) * 1e3, \
            traceback.format_exc(limit=2).strip().splitlines()[-1]
    return answer, (time.perf_counter() - start) * 1e3, None


def word_error_rate(reference: str, hypothesis: str) -> float:
    a, b = reference.split(), hypothesis.split()
    previous = list(range(len(b) + 1))
    for i, word in enumerate(a, start=1):
        current = [i]
        for j, other in enumerate(b, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1,
                               previous[j - 1] + (word != other)))
        previous = current
    return previous[-1] / max(1, len(a))


# ----------------------------------------------------------------------- data
def all_rows() -> list[dict]:
    return list(csv.DictReader((DATA / "manifest.csv").open(encoding="utf-8")))


def conditions_available() -> list[str]:
    """The sets the corpus actually holds, cleanest first, then by noise."""
    found = {r["condition"] for r in all_rows()}
    order = ["clean"] + sorted(
        (c for c in found if c != "clean"),
        key=lambda c: -float(c[3:].replace("m", "-")))
    return [c for c in order if c in found]


def manifest(conditions: list[str], limit: int | None,
             speech: str = "continuous", intonation: str = "order") -> list[dict]:
    rows = [r for r in all_rows() if r["condition"] in conditions
            and r.get("speech", "continuous") == speech
            and intonation in ("all", r.get("intonation", "order"))]
    if intonation == "all":
        # One row of the table per rung and per intonation.
        for r in rows:
            r["condition"] = f"{r['condition']}/{r.get('intonation', 'order')}"
    if limit:
        keep, seen = [], {}
        for row in rows:
            seen[row["condition"]] = seen.get(row["condition"], 0) + 1
            if seen[row["condition"]] <= limit:
                keep.append(row)
        rows = keep
    return rows


# ------------------------------------------------------------------ the tasks
def run_vad(rows: list[dict], order: list[str]) -> list[dict]:
    out = []
    for condition in [c for c in order if c in {r["condition"] for r in rows}]:
        starts, ends, clipped, missing = [], [], 0, 0
        for row in (r for r in rows if r["condition"] == condition):
            signal = voices.read_wav(DATA / row["file"])
            mask, _latency, complaint = call("speech_mask", signal, RATE)
            if complaint or mask is None:
                missing += 1
                continue
            mask = np.asarray(mask, dtype=bool)
            live = np.where(mask)[0]
            if not len(live):
                missing += 1
                continue
            start_s, end_s = live[0] * FRAME / RATE, (live[-1] + 1) * FRAME / RATE
            true_start = float(row["speech_start_s"])
            true_end = float(row["speech_end_s"])
            starts.append(abs(start_s - true_start) * 1e3)
            ends.append(abs(end_s - true_end) * 1e3)
            if start_s > true_start + 0.05 or end_s < true_end - 0.05:
                clipped += 1
        out.append({"condition": condition, "n": len(starts),
                    "start_error_ms": round(statistics.fmean(starts), 1) if starts else "",
                    "end_error_ms": round(statistics.fmean(ends), 1) if ends else "",
                    "clipped": clipped, "no_answer": missing})
    return out


def run_orders(rows: list[dict], path: str, threshold: float | None):
    """One pass over the set: recognise, interpret, decide. Returns the trace."""
    trace, complaints = [], []
    if path == "pretrained":
        # Loading the model takes seconds, once: it is not what the human waits
        # for, so it happens before the clock starts.
        _model, seconds, complaint = call("load_model")
        if complaint:
            complaints.append(complaint)
        else:
            print(f"model loaded in {seconds / 1e3:.1f} s (not counted)\n")
    for row in rows:
        signal = voices.read_wav(DATA / row["file"])
        spent, note = 0.0, None
        if path == "pretrained":
            text, latency, note = call("transcribe", signal, RATE)
            spent += latency
            text = "" if text is None else str(text)
        else:
            text, latency, complaint = call("recognise", signal, RATE)
            spent += latency
            note = complaint
            text = "" if text is None else str(text)
        # From step 4 on, the text is normalised before it is interpreted.
        clean, latency, _ = call("normalise", text)
        spent += latency
        answer, latency, complaint = call("interpret", text if clean is None else str(clean))
        spent += latency
        if complaint and complaint not in complaints:
            complaints.append(complaint)
        try:
            intent, arguments, confidence = answer
        except (TypeError, ValueError):
            intent, arguments, confidence = "unparsed", None, 0.0
        confidence = float(confidence or 0.0)
        state = {"moving": False, "threshold": 0.5 if threshold is None else threshold,
                 "wake_word": CORPUS["wake_word"]}
        act, latency, _ = call("should_act", str(intent), confidence, state)
        spent += latency
        trace.append({"id": row["id"], "condition": row["condition"],
                      "file": row["file"], "reference": row["transcript"],
                      "heard": text, "intent": str(intent),
                      "expected_intent": row["intent"],
                      "confidence": round(confidence, 3), "acted": bool(act),
                      "latency_ms": round(spent, 2),
                      "wer": round(word_error_rate(row["transcript"], text), 3),
                      "note": note})
        if note and note not in complaints:
            complaints.append(note)
    return trace, complaints


def summarise(trace: list[dict], order: list[str] | None = None) -> list[dict]:
    out = []
    order = order or sorted({e["condition"] for e in trace})
    for condition in [c for c in order if c in {e["condition"] for e in trace}]:
        group = [e for e in trace if e["condition"] == condition]
        latencies = sorted(e["latency_ms"] for e in group)
        stops = [e for e in group if e["expected_intent"] == "stop"]
        out.append({
            "condition": condition, "n": len(group),
            "intent_accuracy": round(
                sum(e["intent"] == e["expected_intent"] for e in group) / len(group), 3),
            "wer": round(statistics.fmean(e["wer"] for e in group), 3),
            "latency_ms_median": round(statistics.median(latencies), 2),
            "latency_ms_p90": round(latencies[max(0, int(0.9 * len(latencies)) - 1)], 2),
            # A stop counts when it is heard as a stop and acted on.
            "stops_obeyed": f"{sum(e['intent'] == 'stop' and e['acted'] for e in stops)}"
                            f"/{len(stops)}",
        })
    return out


def run_sweep(rows: list[dict], spec: str, path: str) -> None:
    """Recognise once, then move only the threshold of `should_act`."""
    low, high, step = (float(x) for x in spec.split(":"))
    heard, _ = run_orders(rows, path, None)
    out, value = [], low
    while value <= high + 1e-9:
        trace = []
        for e in heard:
            state = {"moving": False, "threshold": value,
                     "wake_word": CORPUS["wake_word"]}
            act, _latency, _ = call("should_act", e["intent"], e["confidence"], state)
            trace.append({**e, "acted": bool(act)})
        right = sum(e["acted"] and e["intent"] == e["expected_intent"] for e in trace)
        wrong = sum(e["acted"] and e["intent"] != e["expected_intent"] for e in trace)
        refused = sum(not e["acted"] for e in trace)
        stops = [e for e in trace if e["expected_intent"] == "stop"]
        out.append({"threshold": round(value, 3), "correct_actions": right,
                    "wrong_actions": wrong, "refusals": refused,
                    "stops_refused": sum(not e["acted"] for e in stops)})
        value += step
    with (HERE / "sweep.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
        writer.writeheader()
        writer.writerows(out)
    print(f"\n{'threshold':>9} {'correct':>8} {'wrong':>6} {'refused':>8} "
          f"{'stops refused':>14}")
    for row in out:
        print(f"{row['threshold']:>9} {row['correct_actions']:>8} "
              f"{row['wrong_actions']:>6} {row['refusals']:>8} "
              f"{row['stops_refused']:>14}")
    lost = [r for r in out if r["stops_refused"]]
    print("\nsweep.csv written." + (
        f"  From {lost[0]['threshold']} up, your threshold refuses a stop. "
        f"A wrong action and a refusal do not cost the same, and a refused "
        f"stop is in neither column." if lost else ""))


# ------------------------------------------------------------------- the voice
def run_voice(option: str, loopback: bool, rows: list[dict]) -> None:
    sentences = [("ok", "an acknowledgement"),
                 ("conveyor two busy with robot one waiting forty seconds",
                  "an explanation nobody prepared in advance")]
    print(f"\nvoice: {option}")
    say.say("ok", option)          # loads the voice: not what the human waits for
    spoken = None
    for sentence, what in sentences:
        answer, _latency, _ = call("answer", "refuse", sentence)
        text = sentence if not answer else str(answer)
        audio, latency, note = say.say(text, option)
        stated = " (stated figure)" if option == "hosted" else ""
        print(f"  first sound {latency * 1e3:7.1f} ms{stated}   {what}")
        if note:
            print(f"    {note}")
        if audio is not None and spoken is None:
            spoken = audio

    if not loopback or spoken is None:
        return
    # The robot hears itself: its own output arrives in the microphone path,
    # a beat after it starts speaking.
    row = rows[0]
    signal = voices.read_wav(DATA / row["file"])
    gap = int(0.25 * RATE)
    mixed = np.zeros(max(len(signal), gap + len(spoken)))
    mixed[:len(signal)] += signal
    mixed[gap:gap + len(spoken)] += 0.8 * spoken
    speaking = np.zeros(len(mixed) // FRAME, dtype=bool)
    speaking[gap // FRAME: (gap + len(spoken)) // FRAME] = True

    mask, _latency, complaint = call("speech_mask_while_speaking", mixed, RATE, speaking)
    if complaint or mask is None:
        print(f"\n  loop-back: speech_mask did not answer ({complaint})")
        return
    mask = np.asarray(mask, dtype=bool)[:len(speaking)]
    own = int((mask & speaking).sum())
    human = int((mask & ~speaking).sum())
    print(f"\n  loop-back: your detector fired on {own} frame(s) of the robot's "
          f"own voice,\n  and on {human} frame(s) of everything else.")
    if own == 0 and human == 0:
        print("  It fired on nothing at all. Gating the microphone while the "
              "robot speaks\n  costs the human the right to interrupt: say "
              "whether you accept that.")


# ------------------------------------------------------------------- device
def run_device() -> int:
    try:
        import sounddevice as sd
    except Exception as exc:                          # noqa: BLE001
        print(f"sounddevice unavailable ({type(exc).__name__}) — use --input files")
        return 1
    seconds = 3.0
    print(f"speak for {seconds:.0f} s after the beep. No score: there is no "
          f"ground truth on your desk.")
    recording = sd.rec(int(seconds * RATE), samplerate=RATE, channels=1, dtype="float64")
    sd.wait()
    signal = recording[:, 0]
    mask, _, _ = call("speech_mask", signal, RATE)
    text, _, _ = call("recognise", signal, RATE)
    clean, _, _ = call("normalise", "" if text is None else str(text))
    answer, _, _ = call("interpret", "" if clean is None else str(clean))
    frames = 0 if mask is None else int(np.asarray(mask, dtype=bool).sum())
    print(f"  {frames} frame(s) of speech, heard {text!r}, understood {answer!r}")
    return 0


# --------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", choices=["files", "device"], default="files")
    ap.add_argument("--set", dest="sets", action="append",
                    help="repeatable; clean, snr30, snr20, snr10, snr5, snr0, snrm5, or all")
    ap.add_argument("--speech", choices=["continuous", "words"], default="continuous",
                    help="how the orders are spoken: in one breath, or one word at a time")
    ap.add_argument("--intonation", choices=["order", "cheerful", "alarmed", "all"],
                    default="order",
                    help="orders said in one breath come in three intonations")
    ap.add_argument("--task", choices=["orders", "vad"], default="orders")
    ap.add_argument("--asr", choices=["baseline", "pretrained"], default="baseline")
    ap.add_argument("--sweep", help="low:high:step, writes sweep.csv")
    ap.add_argument("--voice", choices=list(say.OPTIONS))
    ap.add_argument("--loopback", action="store_true")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    if args.input == "device":
        return run_device()

    available = conditions_available()
    conditions = available if args.sets and "all" in args.sets else \
        (args.sets or ["clean"])
    unknown = [c for c in conditions if c not in available]
    if unknown:
        print(f"no such set: {', '.join(unknown)}. This corpus has "
              f"{', '.join(available)}.")
        return 1
    # What is still to do, in words rather than in a Python error.
    step2 = module("step2")
    for placeholder in ("useful_function_1_to_rename", "useful_function_2_to_rename"):
        if step2 is not None and hasattr(step2, placeholder):
            print(f"step2.py: {placeholder} still has its placeholder name.")
    for step, names in TODO_CONSTANTS.items():
        loaded = module(step)
        for name in names:
            if loaded is not None and getattr(loaded, name, 0) is None:
                print(f"{step}.py: {name} is still None. Give it a value.")
    rows = manifest(conditions, args.limit, args.speech, args.intonation)
    if args.intonation == "all":
        available = [f"{c}/{i}" for c in available
                     for i in CORPUS.get("intonations", ["order"])]
    print(f"{len(rows)} file(s), {', '.join(conditions)}, corpus spoken by "
          f"{CORPUS['backend']}, {args.speech} speech, intonation "
          f"{args.intonation}\n")

    if args.sweep:
        run_sweep(rows, args.sweep, args.asr)
        return 0

    if args.task == "vad":
        results = run_vad(rows, available)
        kept = keep(f"vad-{args.speech}-{args.intonation}.csv",
                    [{"speech": args.speech, **r} for r in results])
        print(f"{'condition':<16} {'n':>4} {'start err ms':>13} {'end err ms':>11} "
              f"{'clipped':>8} {'no answer':>10}")
        for row in results:
            print(f"{row['condition']:<16} {row['n']:>4} "
                  f"{str(row['start_error_ms']):>13} {str(row['end_error_ms']):>11} "
                  f"{row['clipped']:>8} {row['no_answer']:>10}")
        print(f"\n{kept.relative_to(HERE)} written. Tune your margin until nothing "
              "is clipped, and write down the value you kept.")
        return 0

    if args.asr == "pretrained" and CORPUS["backend"] == "tone":
        print("This corpus was not spoken: it was synthesised as a stand-in, and "
              "there is\nnothing in it for a pre-trained recogniser to "
              "transcribe. Rebuild it with a\nvoice (`python "
              "tools/build_corpus.py --backend piper`) before you measure this "
              "row.\n")

    trace, complaints = run_orders(rows, args.asr, None)
    results = summarise(trace, available)
    print(f"{'condition':<16} {'n':>4} {'intent acc':>11} {'WER':>7} "
          f"{'med ms':>8} {'p90 ms':>8} {'stops':>6}")
    for row in results:
        print(f"{row['condition']:<16} {row['n']:>4} {row['intent_accuracy']:>11} "
              f"{row['wer']:>7} {row['latency_ms_median']:>8} "
              f"{row['latency_ms_p90']:>8} {row['stops_obeyed']:>6}")
    if args.asr == "pretrained":
        audio_seconds = sum(voices.read_wav(DATA / e["file"]).size for e in trace) / RATE
        spent = sum(e["latency_ms"] for e in trace) / 1e3
        here = spent / max(audio_seconds, 1e-9)
        print(f"\nreal time factor  {here:.2f} here, {ROBOT_RTF:.2f} on the "
              f"robot's class of CPU (stated figure)")

    with (HERE / "results.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["path", "speech"] + list(results[0].keys()))
        writer.writeheader()
        for row in results:
            writer.writerow({"path": args.asr, "speech": args.speech, **row})
    kept = keep(f"{args.asr}-{args.speech}-{args.intonation}.csv",
                [{"path": args.asr, "speech": args.speech, **row} for row in results])
    with (HERE / "trace.jsonl").open("w", encoding="utf-8") as fh:
        for entry in trace:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")

    if args.voice:
        run_voice(args.voice, args.loopback, rows)
    if complaints:
        print("\nthe engine had to correct you:")
        for line in complaints[:5]:
            print(f"  {line}")
    print(f"\n{kept.relative_to(HERE)}, results.csv and trace.jsonl written.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
