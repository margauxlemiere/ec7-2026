# Lab 2 — the spoken order, measured

EC7, session 2. The lab statement is assignment 2, on the course site. Fork
this project and work in your fork.

New to Python, Git or VS Code? Follow *Getting started* on the course site
first: https://ci.mines-stetienne.fr/courses/interacting-humans-real-world/getting-started.html

The corpus is not in the project (about 130 MB). Download `lab-2-data.zip` from
the course home page and unzip it here, next to `engine.py`. It creates `data/`,
which git ignores.

```sh
git clone https://gitlab.emse.fr/<you>/lab-2.git
cd lab-2
curl -O https://cps2:cps2@ci.mines-stetienne.fr/courses/interacting-humans-real-world/lab-2-data.zip
unzip lab-2-data.zip                # Windows: Expand-Archive lab-2-data.zip .
. <your ec7>/.venv/bin/activate     # the course environment, from assignment 1
                                    # Windows: <your ec7>\.venv\Scripts\activate
pip install -r requirements.txt     # what this lab adds: faster-whisper
python engine.py --input files --set clean
```

## What is here

| | |
|---|---|
| `engine.py` | scores your code, writes one file per run in `results/`. Do not modify it |
| `step2.py` … `step7.py` | one file per step, which guides you. This is all you write |
| `say.py` | the robot's three voices (`--voice`) |
| `data/` | the corpus, from `lab-2-data.zip` |
| `tools/` | how `data/` was built |

`data/manifest.csv` describes every file: transcript, speaker, form of speech,
intonation, intent, measured signal to noise ratio, ambience.
`data/templates/` has one file per word and per speaker, for the baseline.

## Where the corpus comes from

**Voices.** Four Kokoro-82M voices (two American, two British), cloned by
[Chatterbox](https://github.com/resemble-ai/chatterbox) (MIT), which can change
the intonation. The engine prints which voice spoke, on every run.

**Forms of speech.** Each order is in the corpus four times: three times in one
breath, once per intonation (order, cheerful, alarmed), and once one word at a
time (`data/words/`).

**Noise.** Four recordings from [Freesound](https://freesound.org), CC0: a
workshop hall, a sorting centre, a conveyor belt, a factory with impacts. On half
of the files, a fifth one plays on top: the motors of a CNC machine, as the
robot moving. Authors and links are in `data/noise/sources.json`.
