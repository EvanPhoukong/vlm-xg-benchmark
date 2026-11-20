# Calibration test report

Input: `/home/evan/xG_Research/vlm-xg-benchmark/results/gpt_6_luna/shots.csv`

**300 shots; 42 goals; 63 matches.** Observed scoring rate: 14.0%.

## What happened

- Original VLM Brier: **0.1124**.
- Mean correction Brier: **0.1098** (better than original).
- Sigmoid correction Brier: **0.1106** (better than original).
- Logistic baseline Brier: **0.1070**. The sigmoid-corrected VLM was **worse by point estimate**.
- Sigmoid minus original Brier difference: **-0.0018**; approximate 95% match-bootstrap interval **[-0.0072, +0.0038]**. Negative favors calibration.

## Scores

| Model | Brier (lower better) | AUC | Mean xG | Log loss |
|---|---:|---:|---:|---:|
| VLM, original | 0.1124 | 0.728 | 18.1% | 0.3687 |
| VLM, mean correction | 0.1098 | 0.723 | 14.0% | 0.3629 |
| VLM, sigmoid correction | 0.1106 | 0.719 | 14.0% | 0.3665 |
| Logistic baseline | 0.1070 | 0.762 | 10.7% | 0.3563 |
| StatsBomb | 0.0989 | 0.793 | 10.0% | 0.3378 |
| Training goal-rate reference | 0.1205 | 0.464 | 14.0% | 0.4056 |

## How it was tested

- 5-fold cross-validation, grouping whole matches and using fixed seed 7 by default.
- Each correction was fitted only on the other folds, with no goals from the tested matches.
- Mean correction fits one intercept on log odds. Sigmoid correction fits an intercept and a nonnegative slope, with fixed weak shrinkage toward slope 1.
- Both methods were fixed before evaluating results; there was no search for the best settings.
- Baselines are the existing saved predictions, rather than newly fitted models.
- The training-rate reference predicts the goal rate of the other folds; it does not use the tested matches' outcomes.
- Bootstrap resamples whole matches. Its intervals condition on the fitted cross-validation predictions and do not include refitting uncertainty.

## Limits

- This is an exploratory cross-validation result, not validation on a new competition.
- With few goals, fitted corrections and score differences can be unstable. Independent matches are needed to confirm any improvement.
- Calibration is monotonic within each fold. Pooled AUC may change because different folds use different corrections; this does not demonstrate improved within-model ranking.
- Calibrated predictions include a supervised training step and should be reported separately from zero-shot results.

![Calibration comparison](calibration_comparison.png)
