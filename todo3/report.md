# Assignment 3 — report

<!-- Fill every ___ . Copy the numbers from the files named in each section:
     do not round them, do not retype a table by hand from memory.
     Write one or two short sentences where a line asks for one. -->

## 1. The markers (step 2)

Copy the `marker` rows of `results/detect-all.csv`:

| condition | accuracy | distance_mae_m |
|---|---|---|
| even | ___ | ___ |
| low-sun | ___ | ___ |
| shadow | ___ | ___ |
| mixed | ___ | ___ |

- `CLOSE_HEIGHT_PX` I kept: ___ px.
- Sheets 3 and 4: both read? ___ (yes / no)
- Success image: `todo3/___.png`

## 2. The signs (step 3)

Copy the `sign` rows of `results/detect-all.csv`:

| condition | family_accuracy | accuracy | distance_mae_m |
|---|---|---|---|
| even | ___ | ___ | ___ |
| low-sun | ___ | ___ | ___ |
| shadow | ___ | ___ | ___ |
| mixed | ___ | ___ | ___ |

- The arrows: which ones does it confuse? ___ (see `results/confusions-all.csv`)

## 3. The distance

| | real size (m) | mask ratio | `distance_mae_m` (even) |
|---|---|---|---|
| marker | ___ | 1 | ___ |
| stop sign (red octagon) | ___ | ___ | ___ |
| direction sign (blue disc) | ___ | ___ | ___ |

- One sentence: what happens to the distance if the mask ratio is wrong? ___

## 4. Where it breaks (step 4)

The change I made: ___ (for example: `s > 100` to `s > 60` in `blue = …`).

| condition | detector | accuracy before | accuracy after |
|---|---|---|---|
| shadow | marker | ___ | ___ |
| mixed | marker | ___ | ___ |
| shadow | sign | ___ | ___ |
| mixed | sign | ___ | ___ |

"Before" is `results/detect-all.csv`, "after" is `results/detect-all-change.csv`.

- Shadow image: `todo3/___.png`. Which band does the shadow add? ___
- Was the change worth it? ___ (one line, with one number)

## 5. The false positive rate (step 5)

| detector | `fp_per_minute` before (`detect-absent.csv`) | after the stricter test |
|---|---|---|
| marker | ___ | ___ |
| sign | ___ | ___ |

- Taken for what (`confusions-absent.csv`): ___
- What the stricter test costs on the other sets (`detect-all-tuned.csv` against `detect-all.csv`): ___
- False positive image: `todo3/___.png`

## 6. The action policy (step 6)

- `results/policy.csv`: forbidden outcomes left: ___ (0 is the goal)
- `STOP_FROM_M` = ___ m, because ___
- Not sure of a close stop sign, the robot does: ___
- Confidence threshold for a person: ___. For a stop sign: ___. Why they differ: ___
- Privacy, one line: where the frames are stored, for how long, who sees them: ___
