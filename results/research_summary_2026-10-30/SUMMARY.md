# Can visual models estimate expected goals?
Research summary through 30 October 2026

We tested whether general-purpose vision-language models can estimate the probability of a goal from a shot image. Most inputs were standardized pitch diagrams showing the shooter, teammates, opponents and goalkeeper. Later experiments tested different prompts, reasoning settings, models, datasets, visual examples and past tracking information. Separate historical studies used the visual model to measure positions for a trained statistical model. Recalibration experiments are excluded.

**Reading the tables:** Higher AUC means better discrimination between shots that scored and those that did not. Lower Brier and log loss mean better probability predictions; Brier was our main score. Mean xG is the average predicted probability, not an accuracy score. Compare methods on the same shots: goal rates and difficulty vary between datasets.

## 1. Initial benchmark

The table uses the **298 shots shared by all five saved model files**, including 42 goals (14.1%). We aligned shot identifiers and checked outcomes and coordinates; predictions were unchanged. Earlier 300-shot summaries therefore differ slightly.

| Method | AUC ↑ | Brier ↓ | Log loss ↓ | Mean xG |
|---|---:|---:|---:|---:|
| Gemma 4 (26B) | 0.597 | 0.1901 | 0.5717 | 37.9% |
| Claude Sonnet 5.5 | 0.683 | 0.1210 | 0.4084 | 8.5% |
| Claude Opus 5.5 | 0.721 | 0.1110 | 0.3777 | 10.9% |
| Gemini 3.8 Flash | 0.721 | 0.1157 | 0.3797 | 16.7% |
| GPT-6 Luna | 0.726 | 0.1131 | 0.3706 | 18.2% |
| Original logistic | 0.768 | 0.1057 | 0.3513 | 10.8% |
| StatsBomb | 0.792 | 0.0995 | 0.3399 | 10.0% |

Opus was closest to the logistic baseline on probability error, but the initial study did not establish a VLM advantage. Similar ranking did not imply similar probability estimates: Gemini and Opus had the same AUC but different mean xG.

## 2. Changes to direct visual predictions

Each cell below shows **original → changed condition** on the same shots. These are small development or exploratory comparisons, not independent confirmation. The two 50-shot AFCON studies used separate model calls.

| Change tested (number of shots) | AUC ↑ | Brier ↓ | Log loss ↓ | Mean xG |
|---|---:|---:|---:|---:|
| Add body-part information (50 World Cup shots) | 0.610 → 0.658 | 0.1276 → 0.1204 | 0.4069 → 0.3915 | 19.5% → 17.5% |
| Low → medium reasoning: early pilot (12) | 0.225 → 0.550 | 0.1888 → 0.1470 | 0.6063 → 0.4603 | 20.1% → 16.8% |
| Low → medium reasoning: larger AFCON test (50) | 0.557 → 0.479 | 0.1227 → 0.1317 | 0.4162 → 0.4690 | 15.4% → 17.3% |
| Low → high reasoning: larger AFCON test (50) | 0.557 → 0.424 | 0.1227 → 0.1340 | 0.4162 → 0.4762 | 15.4% → 15.4% |
| Filled → hollow shooter marker (40) | 0.854 → 0.831 | 0.0880 → 0.0923 | 0.2990 → 0.3058 | 14.8% → 15.2% |
| Position-first → repeated/zoom views (200 Euro shots) | 0.635 → 0.658 | 0.0797 → 0.0792 | 0.2907 → 0.2839 | 11.5% → 11.7% |
| Position-first → four geometry examples (same 200) | 0.635 → 0.618 | 0.0797 → 0.0868 | 0.2907 → 0.3081 | 11.5% → 10.7% |
| Luna → Terra, original prompt (50 AFCON shots) | 0.557 → 0.428 | 0.1240 → 0.1227 | 0.4117 → 0.4454 | 17.7% → 12.3% |
| Luna → Sol, original prompt (same 50) | 0.557 → 0.475 | 0.1240 → 0.1333 | 0.4117 → 0.4569 | 17.7% → 16.8% |
| Original → missing-information prompt, Luna (same 50) | 0.557 → 0.498 | 0.1240 → 0.1179 | 0.4117 → 0.4190 | 17.7% → 13.9% |
| Original → missing-information prompt, Terra (same 50) | 0.428 → 0.487 | 0.1227 → 0.1222 | 0.4454 → 0.4318 | 12.3% → 12.7% |

The early reasoning benefit did not repeat in the larger test. Neither the hollow marker nor four geometry examples established an improvement. The missing-information prompt lowered Luna's development Brier but weakened ranking; it was then tested on new matches below. Luna and Sol used low reasoning; Terra used medium, so these are comparisons of complete model configurations.

A separate **50-image geometry diagnostic** found small average shooter and goalkeeper location errors (0.23 and 0.15 normalized pitch units). Angle error averaged 1.91°, falling to 0.64° when code calculated angles from the extracted positions. These are measurement results, not xG accuracy scores. Larger studies revealed occasional costly errors and defenders hidden by overlapping markers.

## 3. Checks on new matches and different datasets

The main VLM and logistic results are shown together. Brackets give **shots / goals**. The Metrica comparison is a two-game pilot, rather than strong independent confirmation.

| Test set [shots/goals] and method | AUC ↑ | Brier ↓ | Log loss ↓ | Mean xG |
|---|---:|---:|---:|---:|
| Bundesliga 2023/24 [200/19] — Direct Luna | 0.612 | 0.0914 | 0.3293 | 12.6% |
| Bundesliga 2023/24 [200/19] — Logistic | 0.641 | 0.0825 | 0.3053 | 9.2% |
| AFCON 2023 [150/14] — Luna | 0.676 | 0.0980 | 0.3339 | 18.1% |
| AFCON 2023 [150/14] — Logistic | 0.690 | 0.0827 | 0.2976 | 10.0% |
| La Liga 2015/16 [200/16] — Original Luna | 0.808 | 0.0787 | 0.2695 | 16.3% |
| La Liga 2015/16 [200/16] — Luna, missing-information prompt | 0.764 | 0.0753 | 0.2693 | 15.1% |
| La Liga 2015/16 [200/16] — Logistic | 0.805 | 0.0662 | 0.2352 | 9.4% |
| Premier League 2015/16 [500/50] — Luna | 0.711 | 0.0926 | 0.3199 | 16.4% |
| Premier League 2015/16 [500/50] — Logistic | 0.737 | 0.0807 | 0.2912 | 9.7% |
| Women's World Cup 2019 [500/43] — Luna | 0.745 | 0.0843 | 0.3030 | 17.3% |
| Women's World Cup 2019 [500/43] — Logistic | 0.788 | 0.0666 | 0.2464 | 10.3% |
| Metrica pilot [46/7] — Luna | 0.804 | 0.1054 | 0.3436 | 18.2% |
| Metrica pilot [46/7] — Logistic | 0.810 | 0.1013 | 0.3398 | 10.5% |

Changing datasets did not solve Luna's probability errors. Logistic regression had reliably lower Brier on the Premier League and Women's World Cup cohorts under the planned match-based intervals. On La Liga 2015/16, the missing-information prompt's small Brier improvement was inconclusive, and ranking fell.

## 4. Stronger model and outcome examples

The latest fixed test used **150 new La Liga 2016/17 shots, 25 goals and 34 matches**. Astra received the same original prompt and image as Luna. Separate Luna conditions received 48 historical images with goal/no-goal outcomes, or the same examples with shuffled labels as a negative control.

| Method | AUC ↑ | Brier ↓ | Log loss ↓ | Mean xG |
|---|---:|---:|---:|---:|
| Original Luna | 0.640 | 0.1382 | 0.4479 | 16.9% |
| Astra | 0.721 | 0.1231 | 0.4084 | 14.9% |
| Luna + correct outcome examples | 0.655 | 0.1319 | 0.4285 | 16.5% |
| Luna + shuffled-label control | 0.746 | 0.1215 | 0.3968 | 16.2% |
| Logistic | 0.744 | 0.1264 | 0.4119 | 10.0% |
| StatsBomb | 0.781 | 0.1194 | 0.3958 | 11.0% |

Astra reduced Brier by **11% versus Luna**. The 99% match-based interval for the difference was **−0.0275 to −0.0022**, adjusted for the five planned comparisons in this round. Its small point advantage over logistic was inconclusive: **−0.0191 to +0.0097**. Correct examples did not establish an advantage over shuffled labels; the control's favorable point score is not evidence of useful learning.

## 5. Earlier hybrid, training and specialist experiments

These methods used extracted positions and a **trained statistical probability model**, rather than asking the VLM alone for final xG. Noise training changed that statistical model, not the VLM. The table keeps representative conditions; the complete tables retain auxiliary controls and variants.

| Method | AUC ↑ | Brier ↓ | Log loss ↓ | Mean xG |
|---|---:|---:|---:|---:|
| World Cup, 300 — first shooter/keeper hybrid | 0.723 | 0.1150 | 0.3816 | 11.1% |
| Euro 2024, 200 — richer hybrid | 0.675 | 0.0776 | 0.2789 | 9.0% |
| Same Euro sample — noisy-position training | 0.677 | 0.0750 | 0.2734 | 8.5% |
| Same Euro sample — multiple views + noise training | 0.687 | 0.0727 | 0.2670 | 8.1% |
| Same Euro sample — four geometry examples | 0.676 | 0.0779 | 0.2799 | 9.2% |
| Same Euro sample — pixel reader + noise training | 0.705 | 0.0726 | 0.2642 | 7.8% |
| Same Euro sample — noise training + event context | 0.695 | 0.0763 | 0.2721 | 8.6% |
| Bundesliga, 200 — original richer hybrid | 0.697 | 0.0801 | 0.2915 | 9.4% |
| Same Bundesliga sample — noisy-position training | 0.696 | 0.0800 | 0.2919 | 8.9% |
| Same Bundesliga sample — new single reader | 0.676 | 0.0827 | 0.2996 | 9.2% |
| Same Bundesliga sample — two specialist readers | 0.688 | 0.0807 | 0.2938 | 8.9% |
| Same Bundesliga sample — specialists + event context | 0.696 | 0.0816 | 0.2934 | 9.4% |

Noise training and multiple views improved exploratory Euro scores, but the fresh Bundesliga noise-training comparison did not confirm a benefit. Four geometry examples did not help. Event context sometimes improved ranking while worsening Brier. Two specialist readers did not establish an improvement over one reader and did not beat the earlier noise-trained pipeline by point estimate.

The pixel reader is a deterministic computer-vision control that exploits the diagrams' fixed colors; it is not a VLM result. Separate linear, spline and boosted-tree statistical candidates were developed on 3,068 World Cup shots. The selected richer model, using **known dataset coordinates**, improved on a simpler matched baseline in the 1,234-shot Euro 2020 test (Brier **0.0734 versus 0.0797**, AUC **0.794 versus 0.752**). The corresponding Euro 2024 gain was inconclusive. These results support richer spatial features, not direct VLM superiority.

Two studies remained incomplete: the clearer, role-separated Euro 2020 images stopped at **62/200** responses; confirmation of the multiple-view/noise-training candidate stopped at **46/200**. Neither has a final accuracy result.

## 6. Past tracking and consistency

On **43 reused Metrica shots, six goals and two games**, we compared snapshots with two actual earlier tracking frames. These were diagrams, not broadcast footage.

| Method | AUC ↑ | Brier ↓ | Log loss ↓ | Mean xG |
|---|---:|---:|---:|---:|
| Original snapshot | 0.687 | 0.1138 | 0.3720 | 17.3% |
| Current frame + ball marker | 0.775 | 0.1037 | 0.3421 | 17.8% |
| Past frames + identical current frame | 0.658 | 0.1363 | 0.4324 | 20.2% |

Adding history worsened this pilot. The current-frame condition also changed the original prompt, so its advantage cannot be attributed solely to the ball marker.

Repeated calls and controlled image changes exposed instability. In earlier tests, exact repeats changed xG by roughly 4–7 percentage points on average, and the missing-information prompt reached 12.8 points. In the latest 12-scene check, Astra's repeat change was **0.88 points**, versus **6.75 for Luna**. Astra followed the expected direction in 10/12 synthetic comparisons, versus Luna's 9/12. These small, correlated diagnostics do not establish population-wide consistency or true shot probabilities.

## Key findings across the experiments

- **Model choice mattered more than the tested prompt and reasoning adjustments.** Astra is the clearest supported improvement over Luna on a new fixed cohort, with better Brier, ranking and observed repeat stability.
- **Better ranking is not the same as accurate xG.** Several models ordered shots reasonably while predicting probabilities that were too high or too low. Luna's short-range overprediction persisted across datasets.
- **The Astra gain extends beyond matching the overall average.** Luna already averaged 16.9% against 16.7% actual scoring on the latest cohort; Astra averaged 14.9% but made better individual predictions.
- **Most proposed fixes remain unconfirmed.** More reasoning, clearer markers, examples, event context, task division and past tracking did not establish consistent improvements on new matches. Several hybrid gains weakened outside the development sample.
- **We have evidence of improvement, not yet proof that direct visual xG matches established statistical models.** Astra needs independent replication and a stronger baseline comparison. Richer known-coordinate models demonstrate useful spatial information, while current VLM extraction and probability judgments do not consistently retain its value.

**Interpretation limits:** Actual goal/no-goal outcomes were used for evaluation; an individual shot's true probability is unobserved. StatsBomb is a reference model, not ground truth. The four-feature logistic baseline receives a header flag absent from the diagrams, as well as recorded coordinates; later studies also checked no-header baselines separately. Many older samples were reused. Most inputs omit body pose, ball height and execution, so this work does not establish accuracy on broadcast video. No VLM fine-tuning was performed. An inconclusive difference does not prove equivalence.
