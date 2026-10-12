# Assignment 3 — one row of the grid

<!-- What you score: the pipeline you built in steps 2 and 3, with no learning.
     Against what: the need of scenario.md §4, the robot that reads markers
     and signs while it moves in a lane where people walk.
     Scale: grille.md, from −3 (strong inadequacy) to +3 (strong adequacy).
     For cost and energy, +3 is cheap and frugal. Latency is a band: +3 is
     the time the robot needs, not the fastest time.
     Each sentence gives a number from results/ or a fact from scenario.md.
     The third column says where the number comes from: keep it. -->

| Criterion | Score | One sentence, with a number or a fact | Where it comes from |
|---|---|---|---|
| Latency | ___ | ___ | `latency_ms_median` in `results/detect-all.csv` |
| Total cost | ___ | ___ | what it needs: a CPU, no GPU, no training data |
| Energy | ___ | ___ | same as cost: what runs on the robot |
| Processing mode | ___ | ___ | one image at a time, or several? |
| Deployment and sovereignty | ___ | ___ | does it need a network or a service? |
| Robustness | ___ | ___ | `accuracy` per condition: `even` against `shadow` and `mixed` |
| Explainability | ___ | ___ | the images in `out/`: can you say why it failed? |
| Confidentiality | ___ | ___ | the camera films the lane, and people walk in it |
| Maintainability | ___ | ___ | steps 4 and 5: how hard was one change? |
| Human control | ___ | ___ | step 6: what the robot does when it is not sure |
