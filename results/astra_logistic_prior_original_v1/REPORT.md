# Astra hybrid on the original World Cup benchmark

The evaluation uses the original common set of 298 shots, with 42 goals from 63 matches. Astra saw the archived diagram and the original match-grouped out-of-fold logistic estimate. The fixed hybrid instructions carry over from La Liga.

| Method | AUC ↑ | Brier ↓ | Mean xG | Brier vs logistic [95% CI] |
|---|---:|---:|---:|---:|
| Astra + logistic xG | 0.769 | 0.10530 | 9.5% | -0.00038 [-0.00278, +0.00206] |
| Gemma 4 (26B) | 0.597 | 0.19008 | 37.9% | +0.08440 [+0.05880, +0.11039] |
| Claude Sonnet 5.5 | 0.683 | 0.12102 | 8.5% | +0.01534 [+0.00413, +0.02708] |
| Claude Opus 5.5 | 0.721 | 0.11102 | 10.9% | +0.00534 [-0.00259, +0.01320] |
| Gemini 3.8 Flash | 0.721 | 0.11569 | 16.7% | +0.01001 [+0.00043, +0.01910] |
| Original logistic baseline | 0.768 | 0.10568 | 10.8% | — |
| StatsBomb xG | 0.792 | 0.09951 | 10.0% | -0.00617 [-0.01259, +0.00013] |

The primary whole-match bootstrap gives a Brier difference of -0.00038 [-0.00292, +0.00218] for hybrid minus logistic. The intervals in the table resample shots to match the original analysis.

The original table clips probabilities to [0.001, 0.999] when computing point scores. The raw hybrid answers are preserved in the prediction file and the primary comparison uses raw Brier errors. The logistic estimate indirectly supplies the header flag. The comparison reuses the published test cohort and archived model predictions, with a fresh hybrid response for each diagram.

All 298 hybrid calls completed, with estimated spend $3.21 and conservative accounting of $3.63.
