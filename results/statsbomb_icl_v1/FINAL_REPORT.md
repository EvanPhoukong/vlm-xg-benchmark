# StatsBomb in-context learning investigation

In-context learning supplies labelled examples inside the inference prompt, without changing model weights. This study tested GPT-6.2 Sol at low reasoning effort on positional shot diagrams.

Outcome-labelled examples did not establish improved Brier over zero-shot predictions. A lower point estimate alone is insufficient evidence of a reliable benefit.

Teacher-labelled examples improved Brier versus zero-shot predictions, but did not establish an advantage over shuffled teacher labels. The result does not isolate learning from the correct image–probability pairing.

## Outcome-labelled examples

A fixed development comparison used 40 shots from 28 separate matches. It tested 0, 8, 32 and 64 naturally sampled training examples, plus shuffled outcomes for 64 examples. There were only two goals in development, so count selection is noisy. The lowest development Brier among correctly labelled conditions selected 64 examples. Those examples contain four goals; the shuffled control preserves that exact count.

The frozen confirmation contains **200 shots, 28 goals, 48 matches**. Lower Brier is better; higher AUC is better.

| Method | Brier | AUC | Mean predicted xG |
|---|---:|---:|---:|
| 64 examples + actual outcomes | 0.103870 | 0.7732 | 0.1042 |
| 64 examples + shuffled outcomes | 0.106253 | 0.7501 | 0.1021 |
| No examples | 0.105254 | 0.7497 | 0.1409 |
| Logistic: image features | 0.106537 | 0.7236 | 0.1059 |
| Logistic: plus header information | 0.104936 | 0.7253 | 0.1104 |
| Training goal-rate constant | 0.122850 | 0.5000 | 0.0905 |
| StatsBomb reference | 0.096492 | 0.8021 | 0.1125 |

## Teacher-probability examples

This is a separate learning target: the same 64 training diagrams are paired with StatsBomb xG predictions, rounded to four decimals. They are teacher-model labels, not observed outcomes or ground-truth probabilities. The control permutes these values across the same images and preserves their complete distribution. It tests use of the correct pairing beyond simply copying a probability scale.

This secondary comparison was declared before inspecting any confirmation accuracy. It uses a fresh zero-shot control. Its cohort is a usage-selected prefix of the original confirmation pool, so it is not an independent replication. Compare methods within each table.

The secondary comparison contains **100 shots, 10 goals, 44 matches**.

| Method | Brier | AUC | Mean predicted xG |
|---|---:|---:|---:|
| No examples | 0.093860 | 0.7167 | 0.1491 |
| 64 examples + StatsBomb xG | 0.081542 | 0.7478 | 0.1194 |
| 64 examples + shuffled StatsBomb xG | 0.082607 | 0.7306 | 0.1180 |
| Logistic: image features | 0.084708 | 0.7189 | 0.1116 |
| Logistic: plus header information | 0.084194 | 0.7233 | 0.1183 |
| StatsBomb reference | 0.072882 | 0.7811 | 0.1164 |

## Uncertainty

For the combined investigation, four comparisons use 98.75% whole-match bootstrap intervals (20,000 resamples), a Bonferroni adjustment across all four contrasts. Negative Brier differences favor correctly labelled ICL. The original per-study 97.5% intervals remain in their separate summaries. Intervals crossing zero do not show equivalence.

| Brier difference | Estimate | 98.75% match interval |
|---|---:|---:|
| outcome64_minus_zero | -0.001384 | [-0.010023, +0.007430] |
| outcome64_minus_shuffled64 | -0.002383 | [-0.007080, +0.001209] |
| teacher64_minus_zero | -0.012319 | [-0.024417, -0.003203] |
| teacher64_minus_shuffled_teacher64 | -0.001065 | [-0.005887, +0.003343] |

## Budget and audit

Completed **1,100 new generation calls**, with no failed or uncertain requests. Estimated new cost: **$6.1932**. Conservative accounting: **$7.6221**, below the **$10 combined cap**. This is experiment accounting, not a live account balance.

The primary audit passed 30 checks; the teacher audit passed 19. Every request was rebuilt from the frozen images and prompts and matched its saved request hash. Raw responses, model settings, probabilities, token usage and costs reconcile. No paid runner is active.

## Limits and reproducibility

Training examples, development matches and query matches are disjoint. All source matches have prior research exposure; this is exploratory evaluation, not untouched external confirmation. The two ICL studies share their query source pool. No query outcomes, reference probabilities, IDs, body parts or coordinates enter an inference request.

Examples are packed into numbered sheets with 192-pixel diagrams; queries retain their 512-pixel diagrams. Small markers and occlusion remain limitations. Only player positions are visible, without shot body part, ball height or previous motion. One fixed example sample does not settle the value of all ICL approaches.

Logistic controls train on the separate 2,000-shot training split. The image-feature model uses distance, angle and opponents in the shooting cone. The header model has extra information. StatsBomb is a richer-input reference; the teacher condition explicitly imitates its training labels. All evaluated VLM probabilities are raw: no calibration, averaging, probability head or weight updates.

Preparation and scoring: `experiments/statsbomb_icl/experiment.py`, `teacher.py`, `audit.py` and `finalize.py`. Native Node inference: `run.mjs` and `teacher_run.mjs`. The teacher runner retains the same reservation engine and adds immutable parent-ledger accounting. Run `test_budget.mjs` for offline cap checks. Protocols, labels, request records and per-shot CSVs are in the two result directories.

Data attribution: StatsBomb Open Data, https://github.com/hudl/open-data. Retain StatsBomb attribution and follow the source repository license.

API rate source: https://developers.openai.com/api/docs/models/gpt-6.2-sol (standard input $2, cached input $0.10 and output $10 per million tokens). Reservations cover maximum output, including reasoning, concurrent calls and uncertain charges. No automatic retries.
