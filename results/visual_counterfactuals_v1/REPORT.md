# Does direct xG respond to controlled visual changes?

Twelve fixed synthetic pairs cover distance, equal-distance shooting angle, keeper displacement and defender blocking. Every base image has an exact-repeat control. Both original Luna and the selected missing-information prompt see images plus generic instructions only. No coordinates, target probabilities, outcomes or corrections are supplied.

The protocol was frozen before the independent real-shot accuracy results were scored. This is a separate visual-behavior diagnostic, not extra goal-labelled confirmation or a method for changing the predictions.

| Method | Expected direction | Mean favorable change | Mean absolute repeat change |
|---|---:|---:|---:|
| luna_original | 10/12 | 10.6% | 6.3% |
| luna_uncertainty | 7/12 | 9.9% | 12.8% |

All probabilities and individual ties/reversals are in predictions.csv. The expected direction is qualitative, not an exact xG target. Template cases are correlated, simplify football scenes and cannot establish absolute probability accuracy. One repeat per case is a variability control, not a complete noise estimate.

All 72 requests, raw answers, source/image/request hashes and usage were audited. New diagnostic estimate $0.014583. Its separate $0.25 cap and the combined $3 round cap both include reservations. No retries or output averaging.

luna_original: 1 ties; reversals: angle_0.
luna_uncertainty: 0 ties; reversals: distance_0, angle_2, blocking_0, blocking_1, blocking_2.
