# SoccerTrack v2 in-context learning

In-context examples did not establish improved xG predictions on this SoccerTrack v2 sample. Both 16-example Brier comparisons with zero-shot predictions have intervals crossing zero. Logistic regression had the strongest point scores.

In-context learning supplies labelled examples in the inference prompt without updating model weights. This study used GPT-6.2 Sol at low reasoning effort on the existing SoccerTrack v2 panoramic frames. This is the dataset previously referred to as SoccerNet v2 in the project.

All methods below use the same **80 shots, 10 goals, 10 matches**. There are 480 complete Sol predictions, one per shot and condition. Lower Brier is better; higher AUC is better.

| Method | Brier | AUC | Mean predicted xG |
|---|---:|---:|---:|
| Sol 6.2: no examples | 0.108515 | 0.6329 | 0.0920 |
| 8 examples + observed outcomes | 0.112684 | 0.4300 | 0.1156 |
| 16 examples + observed outcomes | 0.112716 | 0.4443 | 0.1224 |
| 16 examples + shuffled outcomes | 0.111565 | 0.4743 | 0.1187 |
| 16 examples + logistic xG labels | 0.109574 | 0.5579 | 0.1316 |
| 16 examples + shuffled logistic xG labels | 0.113333 | 0.4486 | 0.1313 |
| Logistic: distance + angle | 0.087125 | 0.7471 | 0.1396 |
| Logistic: also header information | 0.081533 | 0.7700 | 0.1470 |
| Other-match goal-rate constant | 0.109554 | 0.5650 | 0.1440 |

## Interpretation

The 16 outcome examples moved mean predicted xG from 9.2% to 12.24%, close to the observed 12.5% goal rate. However, AUC fell from 0.633 to 0.444 on this sample, and Brier rose from 0.1085 to 0.1127. Matching the overall goal rate did not improve shot-specific predictions.

The logistic-labelled examples are a separate teacher-model condition. These labels come from a distance/angle logistic model fitted on the other nine matches, not from StatsBomb. Their Brier 0.1096 and AUC 0.558 did not improve on no-example predictions. The correctly paired teacher examples had a lower point Brier than shuffled teacher labels, but their adjusted interval also includes zero. Neither label type established an improvement beyond zero-shot or its shuffled control.

## Primary uncertainty intervals

Four prespecified Brier contrasts use 98.75% whole-match percentile bootstrap intervals, 20,000 resamples, a Bonferroni adjustment across four comparisons. Negative differences favor correctly labelled ICL. Ten match groups and ten goals limit precision. Intervals condition on the fitted other-match models and example sets, which overlap across folds.

| Brier difference | Estimate | 98.75% match interval |
|---|---:|---:|
| outcome16_minus_zero | +0.004201 | [-0.004646, +0.011329] |
| outcome16_minus_shuffled_outcome16 | +0.001151 | [-0.000893, +0.002956] |
| teacher16_minus_zero | +0.001059 | [-0.007060, +0.008723] |
| teacher16_minus_shuffled_teacher16 | -0.003760 | [-0.007447, +0.000211] |

## Design and limits

For every query, all 16 examples and all logistic fitting data exclude its entire match. The 8-example condition is a nested prefix. Binary shuffling preserves the number of goals; teacher shuffling preserves every probability value. Image pixels remain identical within each correct/shuffled pair. All conditions receive the unchanged query frame. No query labels, numerical positions, reference probabilities or explicit player/match IDs enter requests.

The natural ordered 306-shot pool and all six conditions were fixed before inference. A token-count-only rule selected 80 queries spanning all ten matches under the $10 cap. There was no accuracy-driven sample extension or example-count selection. All source matches have prior research exposure: this is exploratory match-held-out evaluation, not untouched external confirmation.

Examples are panoramic images letterboxed to at most 768×384 in numbered JPEG quality 92 sheets, with eight images per page. Query PNGs remain unchanged. Shooters are unmarked, players and the ball can be tiny, and images precede the event by approximately 0.32 seconds. Missing or ambiguous shot localization is a plausible limitation; this experiment does not isolate its causal effect.

The distance/angle baseline uses at-event source coordinates. Its header variant has extra body-part information. BAS SHOT/GOAL can omit blocked or headed attempts classified under other actions, and unidentified penalties may remain. The professional-finisher prompt is retained on university matches. All VLM outputs are raw probabilities: no calibration, averaging, probability head or fine-tuning.

## Completion, recovery and spending

OpenAI credits ran out after 96 successful Sol calls. Six rejected requests and one bounded diagnostic rejection were preserved. A resource-only alternative-provider attempt completed four Sonnet predictions; after the user added $5 credits, the original Sol plan resumed. Sonnet predictions are excluded from all score tables. The same 80-shot Sol schedule completed without replacing any query or replaying a successful prediction.

A transient filesystem rename error occurred before a later batch was dispatched. All 372 completed Sol predictions were intact. Six uncommitted reservations were not sent to the API. The recovered runner retries filesystem rename operations only; API requests are not retried automatically. A saved global preflight input bound reduced repeated token-count calls without changing inference payloads.

Estimated Sol successful-call spending: **$5.4419**. Four Sonnet startup calls: **$0.0461**. Combined successful-call estimate: **$5.4880**. Conservative combined accounting: **$7.0654**, including the full **$0.1879** maximum reservation for all seven rejected attempts. The single combined cap remains **$10**. This is experiment accounting, not a live balance or reconciled invoice.

Audit passed 29 checks, rebuilding all 480 Sol and four Sonnet request hashes from frozen input bytes. Source annotation hashes, match exclusions, label permutations, raw answers, model settings, usage and spending reconcile. No paid runner remains active.

Raw response journals remain in `../soccertrack_icl_v1/` and `../soccertrack_icl_resume_v1/`. The unused alternative-provider startup remains in `../soccertrack_icl_sonnet_v1/`. This directory contains consolidated per-shot probabilities, summary and audit. Source and protocols are in `experiments/soccertrack_icl/` and the original result folders.

Data attribution: SoccerTrack v2 by Atom Scott, Ikuma Uchida, Kento Kuroda, Yufi Kim and Keisuke Fujii. [Dataset](https://huggingface.co/datasets/atomscott/soccertrack-v2), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
API rates: [OpenAI](https://developers.openai.com/api/docs/models/gpt-6.2-sol), [Anthropic](https://platform.claude.com/docs/en/about-claude/pricing).
