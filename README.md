# Lab 3 — the markers and the signs, in the light you have

EC7, session 3. The lab statement is assignment 3, on the course site. Fork
this project and work in your fork.

New to Python, Git or VS Code? Follow *Getting started* on the course site
first: https://ci.mines-stetienne.fr/courses/interacting-humans-real-world/getting-started.html

```sh
git clone https://gitlab.emse.fr/<you>/lab-3.git
cd lab-3
. <your ec7>/.venv/bin/activate     # the course environment, from assignment 1
                                    # Windows: <your ec7>\.venv\Scripts\activate
pip install -r requirements.txt     # this lab adds nothing, and says so
python engine.py --input files --set even --show
```

## What is here

| | |
|---|---|
| `engine.py` | runs your code on the images, scores it, writes one file per run in `results/`. Do not modify it |
| `step2.py` | step 2, the markers: `detect_marker` |
| `step3.py` | step 3, the signs: `detect_sign` |
| `step6.py` | step 6, the action policy: `policy` |
| `harness.py` | `show()`, which writes your intermediate images to `out/` |
| `data/` | 250 images in five conditions, `labels.csv`, and `camera.json` |
| `assets/` | the flat targets, from `markers.pdf` and `road-signs.pdf` |
| `tools/` | how `data/` was built |

The three step files are all you write. Each one guides you, and imports the
one before it. Steps 4 and 5 have no file: you change `step2.py` and
`step3.py`, and measure.

## The two families

| Family | What it is | Real size | Your function |
|---|---|---|---|
| Cylinders | the nine sheets of `markers.pdf`, rolled | the printed band, 201.5 mm tall | `detect_marker` |
| Road signs | the six model signs of the platform, a disc on a post | a 70 mm disc; the stop octagon is 57 mm | `detect_sign` |

Each detector answers for its own family, and `unknown` for everything else,
**the other family included**. A red band on a cylinder and a red octagon are
the same few thousand red pixels. The engine counts each time one is taken for
the other.

`assets/geometry.json` gives the real sizes. Your mask is smaller than the
target, and the gap goes into the distance.

## The five conditions

| Set | What is in it |
|---|---|
| `even` | flat light, nothing in the way |
| `low-sun` | warm light from one side, a threshold tuned at the centre |
| `shadow` | a hard-edged band across the target, as thick as one band of the pattern |
| `mixed` | half the target in direct sun, where hue means nothing |
| `absent` | no target at all, and four things that look like one |

The engine scores each condition apart and never averages them.

## Running it

```sh
python engine.py --input files --set all            # results/detect-all.csv
python engine.py --input files --set shadow --show  # and look in out/
python engine.py --input files --set all --tag change   # results/detect-all-change.csv
python engine.py --input files --set absent         # the false positive rate
python engine.py --task policy                      # results/policy.csv
```

Each run writes `results/detect-<set>.csv` and `results/confusions-<set>.csv`.
`--tag` adds a word to the names, to keep a run next to an earlier one.

The images are rendered, not photographed: everyone measures on the same input,
and both failures are there for sure. You need no camera and no printer.
