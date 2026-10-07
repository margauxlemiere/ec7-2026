#!/usr/bin/env python3
"""Build `data/`: the orders, the workshop noise, and the degraded copies.

    python tools/build_corpus.py                 # uses the best voice available
    python tools/build_corpus.py --backend tone  # force the fallback

Fifteen orders, four speakers, sixty orders. `data/clean/` has them as a person
says them, in one breath, and in three intonations when Chatterbox spoke them
(`tools/speak_chatterbox.py`): an order, cheerful, alarmed. `data/words/clean/`
has the same sixty **one word at a time**, with a pause between words, which is what the
isolated word baseline of `scenario.md` §3.6 asks of the human. Each set is
mixed with the workshop at every rung of the ladder, from 30 dB down to -5 dB
signal to noise, measured on the speech frames and inside the speech band: a
silence at the head would otherwise buy signal to noise for free, and a rumble
under 300 Hz would buy it for nothing.

The ambiences are `data/noise/*.wav`, recordings fetched by
`tools/fetch_ambiences.py`. One is drawn per file. `robot-motors` is not drawn:
it is laid over half the mixes, as the robot moving while it is spoken to.
With `data/noise/` empty, a synthetic bed stands in.

`data/manifest.csv` records, for every file, which backend spoke it, how
(`continuous` or `words`), what the measured signal to noise ratio came out at,
and which ambience it was mixed into.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import sys
import zlib

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import voices                                                    # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent.parent
DATA = HERE / "data"
RATE = voices.RATE
WAKE = "robot three"

ORDERS = [
    ("forward", None, "forward"),
    ("stop", None, "stop"),
    ("turn left", None, "turn_left"),
    ("turn right", None, "turn_right"),
    ("carry blue pallet", "blue pallet", "carry"),
    ("carry red pallet", "red pallet", "carry"),
    ("make green crate", "green crate", "make"),
    ("make grey box", "grey box", "make"),
    ("stop now", None, "stop"),
    ("go forward", None, "forward"),
    ("left", None, "turn_left"),
    ("right", None, "turn_right"),
    ("carry green crate", "green crate", "carry"),
    ("make blue pallet", "blue pallet", "make"),
    ("why", None, "why"),
]
VOCABULARY = sorted({w for order, _, _ in ORDERS for w in order.split()}
                    | set(WAKE.split()))
PHRASES = ["ok", "no", "not understood", "possible", "impossible",
           "stop", "go forward", "turn left", "turn right"]
# Du meilleur au pire. Sert à ne pas remplacer de la parole par un repli le jour
# où la machine qui reconstruit n'a pas de moteur de synthèse.
RANK = ["chatterbox", "kokoro", "piper", "say", "espeak", "tone"]

# The corpus spoken by Chatterbox, when `tools/speak_chatterbox.py` has run: the
# orders in three intonations, and every word alone for the word-by-word set.
CHATTERBOX = HERE / ".voices" / "chatterbox"
INTONATIONS = ["order", "cheerful", "alarmed"]
# The speech level of each intonation is set here, not left to the voice: raw
# Chatterbox levels do not always rise with the intonation, and some alarmed
# takes clip.
LEVEL_DBFS = {"order": -26.0, "cheerful": -23.0, "alarmed": -20.0}


def stable_seed(word: str) -> int:
    """A seed that is the same on every machine and every run.

    `hash()` is salted per process in Python, so a corpus built with it is not
    the same corpus twice, and no measurement taken on it compares with
    anybody else's.
    """
    return zlib.crc32(word.encode()) % 9973


# ---------------------------------------------------------------- the workshop
def workshop_bed(seconds: float, seed: int = 4) -> np.ndarray:
    """Conveyors, ventilation, and something dropped now and then.

    The last resort, when `data/noise/` is empty: neither a recording of the
    workshop nor a generated ambience is at hand. It is broadband, it hums, and
    it has transients, which is what a voice activity detector has to survive.
    It has no voices in it, so the detector is never asked here to tell a
    machine from a person.
    """
    rng = np.random.default_rng(seed)
    n = int(seconds * RATE)
    t = np.arange(n) / RATE
    bed = 0.35 * rng.normal(0, 1, n)                     # ventilation
    kernel = np.ones(24) / 24                            # rolled off, not white
    bed = np.convolve(bed, kernel, mode="same")
    for frequency, gain in ((49.7, 0.30), (99.4, 0.18), (148.9, 0.09), (312.0, 0.05)):
        bed += gain * np.sin(2 * math.pi * frequency * t + rng.uniform(0, 6.3))
    bed *= 1 + 0.12 * np.sin(2 * math.pi * 0.23 * t)     # the conveyor cycle
    for _ in range(int(seconds / 7)):                    # a crate set down
        start = int(rng.uniform(0, n - RATE))
        length = int(rng.uniform(0.05, 0.22) * RATE)
        clank = rng.normal(0, 1, length) * np.exp(-np.linspace(0, 7, length))
        bed[start:start + length] += 1.2 * clank         # loud, not four times
    return bed / (np.max(np.abs(bed)) + 1e-9) * 0.5


SPEECH_BAND_HZ = (300.0, 3400.0)
# A bed with almost nothing in the speech band is not a masker. Scaling it up to
# reach a stated ratio there turns the file into rumble and nothing else, so it
# is kept as a file and left out of the mix. Measured, not guessed.
MIN_BAND_SHARE = 0.02
# A bed laid over the ambience rather than drawn instead of it.
LAYERS = {"robot-motors"}


def band_power(signal: np.ndarray) -> float:
    """Power inside the band a voice lives in.

    A bed that is all rumble under 200 Hz can sit at 0 dB of broadband signal to
    noise and mask nothing at all: the number would say the order is buried
    while the words come through untouched. What has to be equal to a stated
    ratio is the power where the words are.
    """
    if len(signal) < 64:
        return 1e-12
    spectrum = np.abs(np.fft.rfft(signal)) ** 2
    freqs = np.fft.rfftfreq(len(signal), 1 / RATE)
    inside = (freqs >= SPEECH_BAND_HZ[0]) & (freqs <= SPEECH_BAND_HZ[1])
    return float(spectrum[inside].mean()) + 1e-12


def band_share(signal: np.ndarray, seconds: float = 8.0) -> float:
    """How dense a bed is where the voice is, against its average density.

    1 is flat; 0.02 is a bed with almost nothing between 300 and 3400 Hz.

    The median over pieces of `seconds`, not the first piece alone: a bed can
    open on something bright and rumble for the next two minutes.
    """
    n = int(seconds * RATE)
    pieces = [signal[i:i + n] for i in range(0, max(1, len(signal) - n + 1), n)]
    shares = []
    for piece in pieces:
        if len(piece) < 64:
            continue
        spectrum = np.abs(np.fft.rfft(piece)) ** 2
        freqs = np.fft.rfftfreq(len(piece), 1 / RATE)
        inside = (freqs >= SPEECH_BAND_HZ[0]) & (freqs <= SPEECH_BAND_HZ[1])
        shares.append(float(spectrum[inside].mean() / (spectrum.mean() + 1e-12)))
    return float(np.median(shares)) if shares else 0.0


def at_level(signal: np.ndarray, dbfs: float) -> np.ndarray:
    """Scale so that the speech frames sit at `dbfs`, and never clip."""
    frame = 160
    mask = speech_frames(signal, frame)
    voiced = signal[:len(mask) * frame].reshape(-1, frame)[mask].reshape(-1)
    rms = math.sqrt(float((voiced ** 2).mean())) if len(voiced) else 1e-9
    out = signal * (10 ** (dbfs / 20) / max(rms, 1e-9))
    peak = float(np.max(np.abs(out)))
    return out * (0.95 / peak) if peak > 0.95 else out


# Files Whisper never heard right from Chatterbox, even after many seeds: a short
# word alone is where it adds sounds of its own. Those come from Kokoro instead,
# and `corpus.json` lists them.
REPLACED: list[str] = []


def cached(kind: str, name: str) -> np.ndarray:
    rows = {r["file"]: r for r in
            csv.DictReader((CHATTERBOX / "manifest.csv").open(encoding="utf-8"))}
    if kind == "words" and rows.get(f"words/{name}.wav", {}).get("ok") != "1":
        speaker, word, _take = name.split("-")
        REPLACED.append(f"{kind}/{name}")
        return voices.trim(voices.speak(word, speaker, "kokoro", stable_seed(word)))
    return voices.trim(voices.read_wav(CHATTERBOX / kind / f"{name}.wav"))


def words_from_cache(words: list[str], speaker: str, seed: int,
                     gap_s: float = 0.22) -> np.ndarray:
    """The word-by-word order, from the takes spoken alone: as `speak_words`."""
    rng = np.random.default_rng(seed)
    pieces = []
    for i, word in enumerate(words):
        pieces.append(cached("words", f"{speaker}-{word}-spoken"))
        if i < len(words) - 1:
            pieces.append(np.zeros(int(gap_s * rng.uniform(0.85, 1.2) * RATE)))
    return np.concatenate(pieces)


def speech_frames(signal: np.ndarray, frame: int = 160) -> np.ndarray:
    """Which frames carry the voice, from the ground truth side: the top half."""
    n = len(signal) // frame
    energy = (signal[:n * frame].reshape(n, frame) ** 2).mean(axis=1)
    return energy > max(np.percentile(energy, 60), 1e-6)


def excerpt(bed: np.ndarray, length: int, rng) -> np.ndarray:
    start = int(rng.uniform(0, max(1, len(bed) - length - 1)))
    piece = bed[start:start + length].copy()
    return np.pad(piece, (0, length - len(piece))) if len(piece) < length else piece


def mix_at(signal: np.ndarray, beds: dict[str, np.ndarray],
           layers: dict[str, np.ndarray], snr_db: float,
           rng) -> tuple[np.ndarray, float, float, str]:
    """Mix at a stated signal to noise ratio, in the band a voice lives in.

    One ambience per utterance, drawn at random and written into the manifest.
    The same ratio does not cost the same in a room that hums and in a room
    where something is being dropped, and that difference is a finding. Half the
    time a layer goes on top, as loud as the ambience in the speech band, and
    the manifest names both.
    """
    name = str(rng.choice(sorted(beds)))
    noise = excerpt(beds[name], len(signal), rng)
    for layer, bed in sorted(layers.items()):
        if rng.uniform() < 0.5:
            over = excerpt(bed, len(signal), rng)
            noise = (noise / math.sqrt(band_power(noise))
                     + over / math.sqrt(band_power(over)))
            name = f"{name}+{layer}"
    frame = 160
    mask = speech_frames(signal, frame)
    voiced = signal[:len(mask) * frame].reshape(-1, frame)[mask].reshape(-1)
    speech = band_power(voiced if len(voiced) else signal)
    noise_band = band_power(noise)
    gain = math.sqrt(speech / noise_band / (10 ** (snr_db / 10)))
    mixed = signal + gain * noise
    measured = 10 * math.log10(speech / ((gain ** 2) * noise_band))
    broadband = 10 * math.log10(
        (float((voiced ** 2).mean()) if len(voiced) else 1e-9)
        / ((gain ** 2) * float((noise ** 2).mean()) + 1e-12))
    peak = np.max(np.abs(mixed))
    return ((mixed / peak * 0.95 if peak > 0.95 else mixed), measured,
            broadband, name)


# ----------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=["auto", "chatterbox", "kokoro", "piper",
                                          "say", "espeak", "tone"],
                    default="auto")
    ap.add_argument("--snr", type=float, nargs="+", default=[30.0, 20.0, 10.0, 5.0, 0.0, -5.0],
                    help="dB, one degraded set per value")
    ap.add_argument("--noise-seconds", type=float, default=120.0)
    ap.add_argument("--seed", type=int, default=20260928)
    ap.add_argument("--force", action="store_true",
                    help="rebuild even with a worse voice than the one on disk")
    args = ap.parse_args()

    backend = voices.available() if args.backend == "auto" else args.backend
    if args.backend == "auto" and (CHATTERBOX / "manifest.csv").exists():
        backend = "chatterbox"
    chatterbox = backend == "chatterbox"
    # The robot's own recorded phrases stay with the best local voice.
    robot_backend = voices.available() if chatterbox else backend
    intonations = INTONATIONS if chatterbox else ["order"]
    if chatterbox:
        spoken_by = list(csv.DictReader((CHATTERBOX / "manifest.csv").open(encoding="utf-8")))
        wrong = [r["file"] for r in spoken_by if r["ok"] != "1"]
        if wrong:
            print(f"warning: Whisper did not hear {len(wrong)} Chatterbox file(s) "
                  f"right: {', '.join(wrong)}")
    existing = DATA / "corpus.json"
    if existing.exists() and not args.force:
        was = json.loads(existing.read_text(encoding="utf-8")).get("backend")
        if was in RANK and RANK.index(was) < RANK.index(backend):
            print(f"the corpus on disk was spoken by {was}, and this machine "
                  f"only offers {backend}.\nRebuilding would replace speech "
                  f"with something worse. Use --force if you mean it.")
            return 1
    rng = np.random.default_rng(args.seed)
    print(f"voice backend: {backend}")

    paths = sorted((DATA / "noise").glob("*.wav"))
    layers = {p.stem: voices.read_wav(p) for p in paths if p.stem in LAYERS}
    found = {p.stem: voices.read_wav(p) for p in paths if p.stem not in LAYERS}
    beds: dict[str, np.ndarray] = {}
    if found:
        aside = {}
        for name, bed in found.items():
            share = band_share(bed)
            (beds if share >= MIN_BAND_SHARE else aside)[name] = bed
            print(f"  bed {name:<16} {len(bed) / RATE:5.0f} s   "
                  f"speech band {share:5.2f} x the average density"
                  f"{'' if share >= MIN_BAND_SHARE else '   set aside'}")
        for name, bed in layers.items():
            print(f"  layer {name:<14} {len(bed) / RATE:5.0f} s   over half the mixes")
        if not beds:
            print("no bed has anything where the voice is; mixing into all of them.")
            beds = found
    if not beds:
        beds = {"workshop": workshop_bed(args.noise_seconds)}
        voices.write_wav(DATA / "noise" / "workshop.wav", beds["workshop"])
        print(f"workshop bed: {args.noise_seconds:.0f} s, synthetic. "
              f"tools/fetch_ambiences.py fetches recordings.")

    # The same sixty orders, said two ways. `continuous` is how a person speaks
    # to a machine that is supposed to understand; `words` is how the isolated
    # word baseline needs to be spoken to. The distance between the two rows is
    # what that option costs the human, and it is measured, not asserted. Spoken
    # in one breath, each order also comes in three intonations when the voice
    # can do it: an order, cheerful, alarmed.
    rows = []
    for speech, prefix in (("continuous", ""), ("words", "words/")):
        for intonation in (intonations if speech == "continuous" else ["order"]):
            for speaker in voices.SPEAKERS:
                for index, (order, arguments, intent) in enumerate(ORDERS, start=1):
                    sentence = f"{WAKE} {order}"
                    if chatterbox and speech == "words":
                        spoken = words_from_cache(sentence.split(), speaker,
                                                  args.seed + index)
                    elif chatterbox:
                        spoken = cached("orders", f"{speaker}-{intonation}-{index:02d}")
                    elif speech == "words":
                        spoken, _bounds = voices.speak_words(
                            sentence.split(), speaker, backend, args.seed + index)
                    else:
                        spoken = voices.trim(voices.speak(sentence, speaker, backend,
                                                          args.seed + index))
                    if chatterbox:
                        spoken = at_level(spoken, LEVEL_DBFS[intonation])
                    lead = np.zeros(int(rng.uniform(0.15, 0.55) * RATE))
                    tail = np.zeros(int(rng.uniform(0.20, 0.60) * RATE))
                    signal = np.concatenate([lead, spoken, tail])
                    name = f"{speaker}{index:02d}-{intonation}"

                    common = {"id": f"{speaker}{index:02d}", "speaker": speaker,
                              "speech": speech, "intonation": intonation,
                              "transcript": sentence, "intent": intent,
                              "arguments": arguments or "", "backend": backend,
                              "speech_start_s": round(len(lead) / RATE, 3),
                              "speech_end_s": round((len(signal) - len(tail)) / RATE, 3)}

                    voices.write_wav(DATA / f"{prefix}clean" / f"{name}.wav", signal)
                    rows.append({**common, "file": f"{prefix}clean/{name}.wav",
                                 "condition": "clean", "snr_db": "",
                                 "snr_broadband_db": "", "ambience": ""})
                    for snr in args.snr:
                        folder = f"snr{snr:g}".replace("-", "m")
                        noisy, measured, broadband, ambience = mix_at(
                            signal, beds, layers, snr, rng)
                        voices.write_wav(DATA / f"{prefix}{folder}" / f"{name}.wav", noisy)
                        rows.append({**common, "file": f"{prefix}{folder}/{name}.wav",
                                     "condition": folder, "snr_db": round(measured, 2),
                                     "snr_broadband_db": round(broadband, 2),
                                     "ambience": ambience})

    # The fixed phrase set of the `recorded` voice: scenario.md §3.4, and
    # nothing else. What it cannot say is the finding of step 6.
    for phrase in PHRASES:
        voices.write_wav(DATA / "phrases" / f"{phrase.replace(' ', '-')}.wav",
                         voices.trim(voices.speak(phrase, "a", robot_backend,
                                                  stable_seed(phrase))))

    # One word per file, for the closed-vocabulary baseline to match against.
    for speaker in voices.SPEAKERS:
        for word in VOCABULARY:
            template = (at_level(cached("words", f"{speaker}-{word}-template"),
                                 LEVEL_DBFS["order"]) if chatterbox else
                        voices.trim(voices.speak(word, speaker, backend,
                                                 stable_seed(word))))
            voices.write_wav(DATA / "templates" / f"{word}-{speaker}.wav", template)

    with (DATA / "manifest.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (DATA / "corpus.json").write_text(json.dumps({
        "rate": RATE, "wake_word": WAKE, "vocabulary": VOCABULARY,
        "backend": backend, "snr_db_targets": args.snr,
        "snr_band_hz": list(SPEECH_BAND_HZ),
        "beds_in_the_mix": sorted(beds), "layers": sorted(layers),
        "speech": ["continuous", "words"], "intonations": intonations,
        "robot_voice": robot_backend, "spoken_by_kokoro_instead": sorted(set(REPLACED)),
        "speakers": {k: v.get("kokoro") for k, v in voices.SPEAKERS.items()},
        "orders": [{"text": o, "intent": i, "arguments": a} for o, a, i in ORDERS],
        "phrases": PHRASES, "ambiences": sorted(beds),
    }, indent=2) + "\n", encoding="utf-8")

    print(f"{len(rows)} files in {DATA}, manifest.csv written")
    if backend == "tone":
        print("\nThis corpus is not speech. Steps 1, 2, 3, 5 and 6 measure what "
              "they measure;\nthe pre-trained recogniser of step 4 has nothing "
              "to transcribe, and the engine says so.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
