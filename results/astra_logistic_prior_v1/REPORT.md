# Astra with a logistic xG estimate

The fixed test covered 100 non-penalty shots with freeze frames from 40 previously unused La Liga 2015/16 matches. 13 shots scored (13.0%). The same diagrams went to Astra in two independent calls per shot. The hybrid prompt supplied the frozen logistic estimate as a starting point and asked Astra to adjust it when the image gave a reason. Astra's raw answer was scored.

| Method | Brier ↓ | AUC ↑ | Log loss ↓ | Mean xG |
|---|---:|---:|---:|---:|
| Astra + logistic estimate | 0.09949 | 0.729 | 0.34635 | 9.3% |
| Astra, diagram only | 0.10883 | 0.686 | 0.36800 | 15.7% |
| Logistic regression (supplied estimate) | 0.10270 | 0.723 | 0.35348 | 9.9% |
| Logistic regression (without header) | 0.10439 | 0.692 | 0.36297 | 9.9% |
| StatsBomb xG | 0.09210 | 0.742 | 0.33774 | 11.0% |

## Paired comparisons

Each 97.5% interval resamples whole matches. Negative Brier differences favor the hybrid.

- Hybrid minus Astra, diagram only was -0.00934 [-0.02469, +0.00669]. The interval includes zero, leaving the difference uncertain.
- Hybrid minus Logistic regression (supplied estimate) was -0.00321 [-0.00623, -0.00077]. The interval supports lower Brier error for the hybrid.

## How Astra used the starting estimate

The mean absolute adjustment was 1.6 percentage points and the median was 0.9. Astra raised the estimate by at least one point on 10 shots and lowered it by at least one point on 38.

The regression was fitted on 3,068 World Cup shots. It uses recorded distance, goal angle, a header flag and the opponent count in the shooting cone. The header flag is absent from the diagram and reaches the hybrid indirectly through the estimate. This tests the complete hybrid prompt, including the instruction to check for visual evidence beyond those features. Shot outcomes and StatsBomb xG were used for scoring and reference only. The results cover one league and season, with one model response per arm per shot.

All 200 calls completed. Estimated API usage cost $2.07 using standard rates. Conservative accounting was $2.32 against the $2.90 cap. [OpenAI pricing](https://developers.openai.com/api/docs/models/gpt-6-astra).

Shot and freeze-frame data from [StatsBomb Open Data](https://github.com/hudl/open-data).
