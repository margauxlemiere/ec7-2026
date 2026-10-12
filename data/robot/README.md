# What the robot's camera actually gives

<!-- Student document, in English like all EC7 material. -->

Frames from the Pi camera of the TurtleBot3 Burger, 20 September 2026, on the
three model signs of the platform: a blue disc with an arrow, a red STOP
octagon, and a white disc crossed by a bar. The rest of `data/` is rendered so
that everyone measures on the same input. These are not.

Read them next to your step files: every number below moves when the light moves.

| | |
|---|---|
| Stream | 320x240, about 30 frames per second |
| Sensor | 32x32 to 3280x2464, so 320x240 is a setting, not a limit |
| Calibration | none. `/camera_info` publishes a zero matrix |
| Exposure, white balance, ISO | all automatic, metered on the average of the frame |
| Signs | outer disc 70 mm, red octagon 57 mm across the flats |

## The three figures

`figure-brightness.png` is the one to look at first. The same scene at five
values of the driver's `brightness` control, with the hue and the clipping of
the STOP sign measured under each frame.

| `brightness` | hue of the red | saturation | pixels clipped |
|---|---|---|---|
| 20 | 0 | 255 | 0 % |
| 35 | 175 | 255 | 0 % |
| 50 | 173 | 179 | 91 % |
| 60, what the robot runs | 166 | 154 | 98 % |
| 80 | sign not found | | |

The rendered set gives that red at hue 0, saturation 255. The robot, as
configured, gives hue 166: a threshold written `H < 10 or H > 170` lets the sign
through the net, and the detector finds a blob of 2 by 3 pixels.

`figure-resolution.png` is the same scene at 320x240, 640x480 and 1280x960. The
framing does not change, so the camera scales and does not crop, and the
apparent diameter of a sign follows the width: 43, 88 and 172 pixels. A focal
length in pixels therefore belongs to a width, not to a camera.

`figure-glare.png` is the third sign lost to a reflection, next to a frame where
it reads. Same camera, same settings.

## The files

`lit-*` is the first scene. `lit-brightness-N.png` are the five frames of the
sweep, `lit-320x240-b60.png` and its two companions the resolution test, and
`lit-glare-b60.png` the reflection.

`dim-*` is the same platform an hour later, in much weaker light and with the
camera pointing elsewhere: the camera is not bolted to the robot and it moved.
`dim-bB-sS.png` crosses `brightness` B with `saturation` S. The value that gave
a clean red in the first scene gives a nearly black image in this one, which is
the point: there is no setting that is right twice.
