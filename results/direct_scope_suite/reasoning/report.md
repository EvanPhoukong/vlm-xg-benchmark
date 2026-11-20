# Reasoning
 
50 shots, 6 goals, 15 matches.

| Method | Brier (lower) | AUC (higher) | Log loss (lower) | Mean xG |
|---|---:|---:|---:|---:|
| medium | 0.13170 | 0.479 | 0.46900 | 17.3% |
| low | 0.12275 | 0.557 | 0.41619 | 15.4% |
| high | 0.13404 | 0.424 | 0.47624 | 15.4% |
| baseline_xg | 0.11569 | 0.553 | 0.42628 | 10.1% |
| sb_xg | 0.11128 | 0.568 | 0.42029 | 10.4% |
| baseline_image_xg | 0.11803 | 0.508 | 0.42943 | 10.0% |

| Primary comparison | Brier difference | Match interval |
|---|---:|---|
| medium minus low | +0.00896 | 95.0%: [-0.01202, +0.02888] |

Estimated stage inference cost: $0.040666. Reasoning tokens are included in output-token charges.

This is direct VLM probability prediction. Raw values are unadjusted. The four-feature baseline knows the header flag; the body-part-free comparator is listed separately where available. Both numerical baselines use recorded coordinates. No numerical baseline values or body-part flags enter the image-only model inputs.

Development evidence only. A selected winner is not confirmed by the scores used to select it. Secondary comparisons are exploratory.
