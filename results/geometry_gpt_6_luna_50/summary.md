# Luna geometry test — simple report

**Luna read the player locations accurately when explicitly asked. Occasional
angle calculations were a clearer weakness than locating the players.**

Tested 50 randomly selected images from the 300-shot benchmark, across 37 matches.
The original images were unchanged. The prompt supplied the pitch scale and
explained how to measure distance and angle; it supplied no shot-specific answers.

| Measurement | Average error | Result within the preset tolerance |
|---|---:|---:|
| Distance to goal center | 0.19 pitch units | 50/50 within 2 units |
| Shooter location | 0.23 pitch units | 50/50 within 2 units |
| Goalkeeper location | 0.15 pitch units | 50/50 within 2 units |
| Goal-opening angle | 1.91° | 48/50 within 5° |

Pitch units use the benchmark's normalized 120×80 coordinate system; they are
not measured stadium meters or verified physical yards.

Calculating angles in ordinary code from Luna's reported shooter locations
reduced average angle error from **1.91° to 0.64°**; 49/50 then fell within 5°.
For one shot, Luna gave the correct location but estimated an 85° opening angle
instead of 38°. Code using that same location calculated 38°.

**What I would test next:** a hybrid system in which Luna extracts positions,
code calculates geometry, and a trained statistical model converts those features
into xG. This is more directly supported by these results than starting with a
custom VLM or vision fine-tuning.

This does **not** demonstrate better xG predictions. The geometry prompt is more
explicit than the earlier xG prompt, so this test cannot establish why the earlier
xG estimates were inaccurate. Any hybrid needs its own evaluation on held-out
matches, using comparable inputs for the baselines.

All 22 offline software checks passed. All 50 model responses were accounted for,
with no missing answers. Original prediction files were verified unchanged.

[Detailed results and methods](report.md) · [Error plots](geometry_errors.png) ·
[Illustrative examples](examples.png)
