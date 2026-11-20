# Rendering
 
40 shots, 5 goals, 27 matches.

| Method | Brier (lower) | AUC (higher) | Log loss (lower) | Mean xG |
|---|---:|---:|---:|---:|
| standard | 0.08800 | 0.854 | 0.29902 | 14.8% |
| outline | 0.09232 | 0.831 | 0.30584 | 15.2% |
| baseline_xg | 0.07820 | 0.857 | 0.26886 | 10.6% |
| sb_xg | 0.07172 | 0.874 | 0.25667 | 11.0% |

| Primary comparison | Brier difference | Match interval |
|---|---:|---|
| outline minus standard | +0.00433 | 95.0%: [-0.01095, +0.01825] |

Estimated stage inference cost: $0.015211. Reasoning tokens are included in output-token charges.

This is direct VLM probability prediction. Raw values are unadjusted. The four-feature baseline knows the header flag; the body-part-free comparator is listed separately where available. Both numerical baselines use recorded coordinates. No numerical baseline values or body-part flags enter the image-only model inputs.

Development evidence only. A selected winner is not confirmed by the scores used to select it. Secondary comparisons are exploratory.
