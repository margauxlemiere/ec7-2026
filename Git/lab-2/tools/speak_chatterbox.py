#!/usr/bin/env python3
"""Speak the corpus with Chatterbox, in three intonations. Teacher side, on a GPU.

    ~/.venvs/kokoro/bin/python tools/speak_chatterbox.py --refs       # once
    ~/.venvs/chatterbox/bin/python tools/speak_chatterbox.py          # ~20 min

Kokoro speaks every order in the same tone: "stop", "stop." and "STOP!" come out
at the same pitch and level. Chatterbox (Resemble AI, MIT) has one knob,
`exaggeration`, that raises pitch, pitch range and loudness together. It clones
a voice from a few seconds of audio, so the four speakers stay the four Kokoro
voices of `voices.SPEAKERS`, cloned from `.voices/refs/`.

It writes into `.voices/chatterbox/`, which is not part of the lab project:

* `orders/<speaker>-<intonation>-<NN>.wav`, the fifteen orders in one breath,
  in each of the three intonations;
* `words/<speaker>-<word>-<take>.wav`, every word of the vocabulary alone,
  twice: one take becomes the template of the baseline, the other is spoken in
  the word-by-word orders. With one take for both, the baseline would match a
  file against itself.

Every file is transcribed by Whisper small and kept only if the words are right;
otherwise the next seed is tried, up to eight. `manifest.csv` records the seed,
the attempts and what Whisper heard. `build_corpus.py` then reads all of it.
The same seed gives the same audio on the same GPU.
"""
from __future__ import annotations

import argparse
import csv
import pathlib
import re
import sys
import zlib

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
LAB = HERE.parent
CACHE = LAB / ".voices"
OUT = CACHE / "chatterbox"
sys.path.insert(0, str(HERE))

# intonation -> (exaggeration, cfg_weight, final punctuation). Measured on
# 27/09/2026: alarmed is 7 semitones above order and 5 dB louder, on average.
# Above 1.3 the words break up ("Robot Bree").
INTONATIONS = {"order": (0.4, 0.5, "."), "cheerful": (0.8, 0.5, "!"),
               "alarmed": (1.2, 0.3, "!")}
TAKES = ("template", "spoken")
REF_TEXT = ("Every morning the robots in the workshop carry pallets from the "
            "conveyor to the shelves, and they slow down whenever somebody "
            "walks across their path.")

# The recogniser's spelling choices, not errors of the voice.
SAME = {"to": "two", "too": "two", "for": "four", "3": "three", "2": "two",
        "4": "four", "1": "one", "carrie": "carry", "kerry": "carry",
        "cary": "carry", "palette": "pallet", "gray": "grey", "conveyer": "conveyor"}


def words_of(text: str) -> list[str]:
    text = re.sub(r"(\d)", r" \1 ", text.lower().replace("-", " "))
    return [SAME.get(w, w) for w in re.sub(r"[^a-z0-9 ]", " ", text).split()]


def make_refs() -> None:
    """A few seconds of each Kokoro voice, continuous, at 24 kHz."""
    import wave
    import voices
    pipeline = voices.kokoro_pipeline()
    (CACHE / "refs").mkdir(parents=True, exist_ok=True)
    for speaker in voices.SPEAKERS.values():
        name = speaker["kokoro"]
        audio = np.concatenate([np.asarray(a, dtype=np.float64)
                                for _, _, a in pipeline(REF_TEXT, voice=name)])
        audio = voices.trim(audio)
        with wave.open(str(CACHE / "refs" / f"{name}.wav"), "wb") as fh:
            fh.setnchannels(1)
            fh.setsampwidth(2)
            fh.setframerate(24000)
            fh.writeframes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())
        print(f"  {name}: {len(audio) / 24000:.1f} s")


class Voice:
    def __init__(self, device: str):
        import torch
        from chatterbox.tts import ChatterboxTTS
        from faster_whisper import WhisperModel
        self.torch, self.device = torch, device
        self.model = ChatterboxTTS.from_pretrained(device=device)
        self.asr = WhisperModel("small", device="cpu", compute_type="int8")
        self.conditioned = None

    def say(self, written: str, speaker: str, intonation: str, seed: int) -> np.ndarray:
        from scipy.signal import resample_poly
        exaggeration, cfg_weight, _ = INTONATIONS[intonation]
        if self.conditioned != (speaker, intonation):
            self.model.prepare_conditionals(str(CACHE / "refs" / f"{speaker}.wav"),
                                            exaggeration=exaggeration)
            self.conditioned = (speaker, intonation)
        self.torch.manual_seed(seed)
        if self.device == "cuda":
            self.torch.cuda.manual_seed_all(seed)
        wav = self.model.generate(written, exaggeration=exaggeration,
                                  cfg_weight=cfg_weight, temperature=0.8)
        x = wav.squeeze(0).cpu().numpy().astype(np.float64)
        return resample_poly(x, 2, 3) if self.model.sr == 24000 else x

    def heard(self, x: np.ndarray) -> str:
        segments, _ = self.asr.transcribe(x.astype(np.float32), language="en",
                                          beam_size=5)
        return " ".join(s.text for s in segments).strip()


def write(path: pathlib.Path, x: np.ndarray) -> None:
    import voices
    voices.write_wav(path, x)


def one(voice: Voice, path: pathlib.Path, text: str, written: str | list[str],
        speaker: str, intonation: str, key: str, tries: int) -> dict:
    """The first take Whisper hears right, trying each written form in turn."""
    base = zlib.crc32(key.encode()) % 9973
    heard, ok, attempt = [], False, 0
    for form in ([written] if isinstance(written, str) else written):
        for _ in range(tries):
            x = voice.say(form, speaker, intonation, base + attempt)
            heard.append(voice.heard(x))
            attempt += 1
            ok = words_of(heard[-1]) == words_of(text)
            if ok:
                break
        if ok:
            break
    attempt -= 1
    write(path, x)
    row = {"file": str(path.relative_to(OUT)), "text": text, "speaker": speaker,
           "intonation": intonation, "seed": base + attempt,
           "attempts": attempt + 1, "ok": int(ok), "heard": " | ".join(heard)}
    print(f"  {'ok ' if ok else 'BAD'} {row['file']:<40} {attempt + 1}  {heard[-1]}",
          flush=True)
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--refs", action="store_true", help="write the Kokoro references and stop")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tries", type=int, default=8)
    args = ap.parse_args()
    if args.refs:
        make_refs()
        return 0

    import build_corpus
    import voices
    voice = Voice(args.device)
    manifest = OUT / "manifest.csv"
    done = {}
    if manifest.exists():
        done = {r["file"]: r for r in csv.DictReader(manifest.open(encoding="utf-8"))}
    rows = []

    for key, speaker in voices.SPEAKERS.items():
        name = speaker["kokoro"]
        for intonation, (_, _, stop) in INTONATIONS.items():
            for index, (order, _, _) in enumerate(build_corpus.ORDERS, start=1):
                text = f"{build_corpus.WAKE} {order}"
                path = OUT / "orders" / f"{key}-{intonation}-{index:02d}.wav"
                rel = str(path.relative_to(OUT))
                if rel in done and path.exists() and done[rel]["ok"] == "1":
                    rows.append(done[rel])
                    continue
                wake, rest = text.split()[:2], text.split()[2:]
                written = f"{' '.join(wake).capitalize()}, {' '.join(rest)}{stop}"
                rows.append(one(voice, path, text, written, name, intonation,
                                f"{name}-{intonation}-{text}", args.tries))
        for word in build_corpus.VOCABULARY:
            for take in TAKES:
                path = OUT / "words" / f"{key}-{word}-{take}.wav"
                rel = str(path.relative_to(OUT))
                if rel in done and path.exists() and done[rel]["ok"] == "1":
                    rows.append(done[rel])
                    continue
                # A short word alone is where Chatterbox adds sounds of its own:
                # other spellings of the same word are tried before giving up.
                forms = [f"{word.capitalize()}.", f"{word.capitalize()}!", word]
                rows.append(one(voice, path, word, forms, name,
                                "order", f"{name}-{word}-{take}", args.tries))
        with manifest.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    bad = [r["file"] for r in rows if str(r["ok"]) != "1"]
    print(f"{len(rows)} files, {len(bad)} still wrong after {args.tries} tries"
          + (": " + ", ".join(bad) if bad else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
