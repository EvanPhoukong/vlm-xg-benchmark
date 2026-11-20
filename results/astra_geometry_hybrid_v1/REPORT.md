# Astra and logistic regression for xG

This matched pilot reused 150 La Liga 2016/17 shot diagrams across 34 matches. 25 shots scored. Astra read the shooter and goalkeeper positions. Logistic regression fitted on 3,068 World Cup shots estimated xG from distance, goal angle and goalkeeper geometry. The same fitted model also ran on recorded positions to measure the effect of reading coordinates from the diagrams.

| Method | Brier ↓ | AUC ↑ | Log loss ↓ | Mean xG |
|---|---:|---:|---:|---:|
| Astra positions + logistic regression | 0.1311 | 0.699 | 0.4389 | 9.9% |
| Astra, direct xG | 0.1231 | 0.721 | 0.4084 | 14.9% |
| Luna, direct xG | 0.1382 | 0.640 | 0.4479 | 16.9% |
| Same hybrid model, recorded shooter and keeper positions | 0.1306 | 0.700 | 0.4376 | 9.9% |
| Logistic baseline, recorded positions and defenders | 0.1322 | 0.724 | 0.4303 | 9.2% |
| Original logistic baseline, also receives header flag | 0.1264 | 0.744 | 0.4119 | 10.0% |
| StatsBomb | 0.1194 | 0.781 | 0.3958 | 11.0% |

### Paired Brier differences

Differences are hybrid minus comparator. The 95% intervals resample whole matches with predictions fixed. Only the direct Astra comparison was primary.

- Astra, direct xG — +0.0080 [-0.0015, +0.0180].
- Same hybrid model, recorded shooter and keeper positions — +0.0004 [-0.0002, +0.0012].
- Original logistic baseline, also receives header flag — +0.0047 [-0.0058, +0.0138].

### Position measurements

- Shooter — mean error 0.13 normalized pitch units and 90th percentile 0.22 across 150 paired positions.
- Keeper — mean error 0.10 normalized pitch units and 90th percentile 0.20 across 150 paired positions.

Astra received the original diagram and geometry instructions. Only reported positions entered the hybrid. The training fit supplied imputation values for missing positions. The recorded-position control is separate from the image-only hybrid. The existing statistical baselines use recorded defender positions and the original baseline also receives the header flag. The cohort informed the prior selection of Astra, so this is an exploratory comparison on diagrams. Confirmation requires unused matches.

Estimated Astra cost $2.38 for 150 geometry calls.
