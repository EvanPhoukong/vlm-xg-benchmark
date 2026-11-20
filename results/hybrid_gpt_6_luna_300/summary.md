# Hybrid xG experiment — simple report

**This version did not improve the scores.** It scored slightly worse than
direct Luna and remained behind the original logistic baseline and StatsBomb.

Tested the same **300 shots, 42 goals, across 63 matches**. The original 50
geometry responses were reused; 250 additional images were processed. The
features and model settings were fixed before scoring.

| Method | Brier — lower is better | AUC — higher is better |
|---|---:|---:|
| Direct Luna | 0.1124 | 0.728 |
| Hybrid: Luna positions → code → logistic xG | **0.1150** | **0.723** |
| Same hybrid model using known positions | 0.1107 | 0.730 |
| Original logistic baseline | **0.1070** | **0.762** |
| StatsBomb | **0.0989** | **0.793** |

The hybrid learned xG from known coordinates and outcomes in other matches.
At test time, it received only geometry calculated from Luna's reported shooter
and goalkeeper positions. Luna itself was unchanged. Whole test matches,
including their other shots, were excluded from statistical training.

What we learned:

- **Most positions were close, but a few were badly wrong.** Shooter locations
  were within two pitch units on 297/300 images. Goalkeeper locations were within
  two units on 296/300; one keeper estimate was missing. Averages hid rare errors.
- In the worst case, the shooter was on the goal line, 14.6 pitch units to the
  side of goal center. Luna reported the shooter at the goal center. The identical
  statistical model assigned **91.9%** from the extracted position, versus **6.6%**
  from known coordinates. This illustrates the effect of an extraction error;
  neither probability is being treated as ground-truth xG.
- **Better extraction alone would not make this model beat the baseline.** The
  known-position version scored 0.1107. This five-feature model omits header/body
  part and other defender positions, which may contain useful information.

The hybrid-minus-direct-Luna Brier difference was **+0.0026**, with an approximate
95% whole-match bootstrap interval of **−0.0134 to +0.0196**. The sample does not
establish that the hybrid is generally worse; it provides no convincing evidence
of improvement. These intervals condition on fitted models and do not include
refitting uncertainty or multiple-comparison adjustment.

What I would try next: extract opponent locations as well, add checks that flag
implausible measurements for review or a predefined fallback, and test the frozen
method on new matches. A separate condition could supply body-part context to both
the hybrid and its comparator. That would need to be reported separately from the
image-only experiment. Do not delete difficult shots to improve the score.

The hybrid changes both the task prompt and the probability-mapping step, so this
comparison cannot attribute differences to one component alone. Training used
known coordinates while evaluation used VLM measurements; training on noisy
extractions is a possible later experiment, not something tested here.

All **29 offline checks passed**. All 300 shots and responses were accounted for,
training/test matches were disjoint, the existing baseline was reproduced, and
original results and image files were verified unchanged. No settings were tuned
after seeing these scores. This remains exploratory work on one competition.

[Full results and methods](report.md) · [Comparison figure](comparison.png) ·
[Worst extraction example](worst_case_1.png)
