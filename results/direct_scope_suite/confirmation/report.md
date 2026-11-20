# Confirmation
 
150 shots, 14 goals, 36 matches.

| Method | Brier (lower) | AUC (higher) | Log loss (lower) | Mean xG |
|---|---:|---:|---:|---:|
| original_low | 0.09798 | 0.676 | 0.33387 | 18.1% |
| candidate | 0.09798 | 0.676 | 0.33387 | 18.1% |
| baseline_xg | 0.08267 | 0.690 | 0.29756 | 10.0% |
| sb_xg | 0.07754 | 0.755 | 0.27644 | 9.7% |
| baseline_image_xg | 0.08479 | 0.685 | 0.30143 | 10.2% |

| Primary comparison | Brier difference | Match interval |
|---|---:|---|
| candidate minus baseline_xg | +0.01531 | 97.5%: [-0.00185, +0.03097] |
| candidate minus original_low | +0.00000 | 97.5%: [+0.00000, +0.00000] |

Estimated stage inference cost: $0.029524. Reasoning tokens are included in output-token charges.

This is direct VLM probability prediction. Raw values are unadjusted. The four-feature baseline knows the header flag; the body-part-free comparator is listed separately where available. Both numerical baselines use recorded coordinates. No numerical baseline values or body-part flags enter the image-only model inputs.

Two primary comparisons use 97.5% intervals to account for multiplicity. Development choices were frozen before these match-disjoint confirmation outcomes were scored. These intervals condition on the fixed fitted baselines; one tournament does not establish universal performance.
