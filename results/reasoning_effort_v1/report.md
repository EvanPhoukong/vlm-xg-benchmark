# Direct Luna xG: reasoning-effort pilot

Completed 12 shots (2 goals; 11 matches), 24 successful requests. Only gpt-6-luna was requested. Original image bytes, prompt, schema, high image detail and output allowance are identical; only reasoning effort differs.

This reuses the previously studied World Cup development cohort, with four shots per prespecified distance stratum. It is not independent confirmation or a naturally weighted tournament sample. No probability adjustment, hybrid head, new dataset, renderer change or fine-tuning.

| Method | Brier (lower) | Log loss | AUC | Mean xG |
|---|---:|---:|---:|---:|
| low | 0.18877 | 0.60625 | 0.225 | 20.1% |
| medium | 0.14697 | 0.46025 | 0.550 | 16.8% |
| baseline_xg | 0.15696 | 0.52302 | 0.650 | 8.2% |
| sb_xg | 0.14834 | 0.47728 | 0.700 | 8.5% |

Medium minus low Brier: -0.04181; exploratory whole-match 95% interval [-0.10505, +0.00395]. Twelve shots cannot establish superiority or equivalence. Intervals condition on this selected small cohort.

| Effort | Input | Cached input | Output including reasoning | Reasoning subset | Estimated USD |
|---|---:|---:|---:|---:|---:|
| low | 5172 | 0 | 4265 | 3721 | $0.002650 |
| medium | 5172 | 0 | 6285 | 5747 | $0.003660 |

Total usage-derived estimated inference cost: $0.006309. Published standard rates: $0.10/M input, $0.01/M cached input, $0.50/M output. This is not an account billing statement or balance check. Token-count preflights generate no model responses.

The original logistic comparator includes a header flag absent from the image, and uses saved match-grouped out-of-fold predictions. It is a practical reference, not an equal-input ablation. StatsBomb xG is a comparator, not ground truth. Outcomes, source coordinates and comparator predictions were never included in API inputs.

No hidden retries or automatic expansion. Raw responses, usage, randomized schedule, source/image hashes and predictions are preserved. Rationales are available for factual review; fluent explanations do not prove the internal reasoning process. A larger untouched-match confirmation requires a separate budget and sample-size proposal.

Sources: [Luna rates/settings](https://developers.openai.com/api/docs/models/gpt-6-luna), [token accounting](https://developers.openai.com/api/docs/guides/token-counting).

## Illustrative factual review

A post-hoc review screened all 24 short rationales with effort and outcomes hidden, then checked two clear distance misdescriptions against the source images. These examples suggest a possible perceptual benefit; they do not establish the cause of the score difference.

- Shot d2f58c81-92d6-4a50-811a-2413929f5c1d: distance 32.2 normalized pitch units, visibly outside the penalty area. Low xG 12%: The shot is from close range but at a left-sided angle, with several defenders crowding the shooting lane and the goalkeeper covering the goal. Medium xG 2%: The shot is from roughly 30 metres at a narrow angle, with several defenders crowding the goal area.

- Shot 9000f0f2-73ed-4a71-917d-2d525af7ba80: distance 26.0 normalized pitch units, visibly outside the penalty area. Low xG 12%: The shot is from close range but slightly left of center, with several defenders crowding the shooting lane and the goalkeeper positioned in goal. Medium xG 3%: The shot is from outside the box at moderate range, with several defenders crowding the shooting lane.

All response/model/effort checks, paired-input checks, protected input hashes and budget checks passed. No API calls were made for this review.

## Pilot fragility

The largest paired loss reduction came from one scored shot (a84dc163-7627-4553-900a-7720ce6b6099): low predicted 8% and medium 27%. That case accounts for approximately 62% of the net Brier reduction. The illustrative distance-description corrections are not established as the cause of the aggregate gain. With two goals and deliberately balanced distance strata, this pilot is especially sensitive to which shots happened to be selected.
