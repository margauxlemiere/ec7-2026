# Lab 3 — the markers and the signs, in the light you actually have

EC7, session 3. The lab statement is on the course site; this is the project it
runs in. Fork it, work in your fork, and keep the commits: assignment 3 asks for
this code.

```sh
git clone https://gitlab.emse.fr/<you>/lab-3.git
cd lab-3
python3 -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python engine.py --input files --set even --show
```

## What is here

| | |
|---|---|
| `engine.py` | feeds, times, scores, writes `results.csv`. **Do not modify it** |
| `student.py` | three functions, empty. This is all you write |
| `harness.py` | `show()`, which writes your intermediate images to `out/` |
| `data/` | 250 images in five conditions, and `labels.csv` |
| `assets/` | the flat targets, from `markers.pdf` and `road-signs.pdf` |
| `tools/` | how `data/` was built, if you want to build more of it |

## The two families, and why they are scored apart

| Family | What it is | Real size | Your function |
|---|---|---|---|
| Cylinders | the nine sheets of `markers.pdf`, rolled | the printed band, on an A4 sheet | `detect_marker` |
| Road signs | the six sheets of `road-signs.pdf`, flat | the sheet states its diameter | `detect_sign` |

Each detector answers for its own family and `unknown` for everything else,
**including the other family**. A red band on a cylinder and a red octagon on a
post are the same few thousand red pixels, and the engine counts every time one
is taken for the other.

Two sizes are asked of you and given nowhere in the code: what a detector
measures is a mask, and a mask is smaller than the sheet it came from. Whether
that gap is worth correcting is a line of your report.

## The five conditions

| Set | What is in it |
|---|---|
| `even` | flat light, nothing in the way |
| `low-sun` | warm light from one side, a threshold tuned at the centre |
| `shadow` | a hard-edged band across the target, one band of the pattern thick |
| `mixed` | half the target in direct sun, where hue stops meaning anything |
| `absent` | no target at all, and four things that look like one |

Accuracy is reported per condition and never averaged. One number over five
conditions hides the one that breaks you.

## Running it

```sh
python engine.py --input files --set all          # the whole set, results.csv
python engine.py --input files --set shadow --show   # and look in out/
python engine.py --input files --set absent       # the false positive rate
python engine.py --task policy                    # the approach, replayed
python engine.py --input device                   # your camera, no score
```

The image set is rendered, not photographed. It is here so that everyone
measures on the same input and so that both failures are certainly present.
Real light enters through `--input device`, and through the photographs your
assignment asks you to take after this session.
