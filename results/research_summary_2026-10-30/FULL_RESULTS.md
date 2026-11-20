# Complete non-recalibration result tables

Compiled offline on 30 October 2026. No API calls, retraining or probability correction. Compare methods within the same cohort. Cohorts overlap and must not be pooled as independent replications. Direct VLM outputs, supervised hybrids, known-coordinate statistical models and pixel controls answer different questions. Incomplete trials have no final score. The common-shot initial table is an aligned view of saved outputs; original-size scores are also retained.

## Original saved cohort: Gemma 4 (26B)

| Method | AUC ↑ | Brier ↓ | Log loss ↓ | Mean xG |
|---|---:|---:|---:|---:|
| Direct VLM | 0.597 | 0.1896 | 0.5707 | 37.9% |
| Paired saved logistic | 0.768 | 0.1050 | 0.3494 | 10.8% |
| StatsBomb | 0.793 | 0.0989 | 0.3378 | 10.0% |

## Original saved cohort: Claude Sonnet 5.5

| Method | AUC ↑ | Brier ↓ | Log loss ↓ | Mean xG |
|---|---:|---:|---:|---:|
| Direct VLM | 0.683 | 0.1202 | 0.4061 | 8.5% |
| Paired saved logistic | 0.768 | 0.1050 | 0.3494 | 10.8% |
| StatsBomb | 0.793 | 0.0989 | 0.3378 | 10.0% |

## Original saved cohort: Claude Opus 5.5

| Method | AUC ↑ | Brier ↓ | Log loss ↓ | Mean xG |
|---|---:|---:|---:|---:|
| Direct VLM | 0.721 | 0.1103 | 0.3758 | 10.9% |
| Paired saved logistic | 0.768 | 0.1050 | 0.3494 | 10.8% |
| StatsBomb | 0.793 | 0.0989 | 0.3378 | 10.0% |

## Original saved cohort: Gemini 3.8 Flash

| Method | AUC ↑ | Brier ↓ | Log loss ↓ | Mean xG |
|---|---:|---:|---:|---:|
| Direct VLM | 0.721 | 0.1157 | 0.3797 | 16.7% |
| Paired saved logistic | 0.768 | 0.1057 | 0.3513 | 10.8% |
| StatsBomb | 0.792 | 0.0995 | 0.3399 | 10.0% |

## Original saved cohort: GPT-6 Luna

| Method | AUC ↑ | Brier ↓ | Log loss ↓ | Mean xG |
|---|---:|---:|---:|---:|
| Direct VLM | 0.728 | 0.1124 | 0.3687 | 18.1% |
| Paired saved logistic | 0.762 | 0.1070 | 0.3563 | 10.7% |
| StatsBomb | 0.793 | 0.0989 | 0.3378 | 10.0% |

## Initial benchmark: common 298 shots

| Method | AUC ↑ | Brier ↓ | Log loss ↓ | Mean xG |
|---|---:|---:|---:|---:|
| Gemma 4 (26B) | 0.597 | 0.1901 | 0.5717 | 37.9% |
| Claude Sonnet 5.5 | 0.683 | 0.1210 | 0.4084 | 8.5% |
| Claude Opus 5.5 | 0.721 | 0.1110 | 0.3777 | 10.9% |
| Gemini 3.8 Flash | 0.721 | 0.1157 | 0.3797 | 16.7% |
| GPT-6 Luna | 0.726 | 0.1131 | 0.3706 | 18.2% |
| Original logistic | 0.768 | 0.1057 | 0.3513 | 10.8% |
| StatsBomb | 0.792 | 0.0995 | 0.3399 | 10.0% |

## Reasoning pilot: 12 reused World Cup shots

| Method | AUC ↑ | Brier ↓ | Log loss ↓ | Mean xG |
|---|---:|---:|---:|---:|
| low | 0.225 | 0.1888 | 0.6063 | 20.1% |
| medium | 0.550 | 0.1470 | 0.4603 | 16.8% |
| baseline_xg | 0.650 | 0.1570 | 0.5230 | 8.2% |
| sb_xg | 0.700 | 0.1483 | 0.4773 | 8.5% |

## Reasoning: 50 AFCON development shots

| Method | AUC ↑ | Brier ↓ | Log loss ↓ | Mean xG |
|---|---:|---:|---:|---:|
| medium | 0.479 | 0.1317 | 0.4690 | 17.3% |
| low | 0.557 | 0.1227 | 0.4162 | 15.4% |
| high | 0.424 | 0.1340 | 0.4762 | 15.4% |
| baseline_xg | 0.553 | 0.1157 | 0.4263 | 10.1% |
| sb_xg | 0.568 | 0.1113 | 0.4203 | 10.4% |
| baseline_image_xg | 0.508 | 0.1180 | 0.4294 | 10.0% |

## Marker: 40 reused World Cup shots

| Method | AUC ↑ | Brier ↓ | Log loss ↓ | Mean xG |
|---|---:|---:|---:|---:|
| standard | 0.854 | 0.0880 | 0.2990 | 14.8% |
| outline | 0.831 | 0.0923 | 0.3058 | 15.2% |
| baseline_xg | 0.857 | 0.0782 | 0.2689 | 10.6% |
| sb_xg | 0.874 | 0.0717 | 0.2567 | 11.0% |

## Independent AFCON: 150 shots

| Method | AUC ↑ | Brier ↓ | Log loss ↓ | Mean xG |
|---|---:|---:|---:|---:|
| original_low | 0.676 | 0.0980 | 0.3339 | 18.1% |
| candidate | 0.676 | 0.0980 | 0.3339 | 18.1% |
| baseline_xg | 0.690 | 0.0827 | 0.2976 | 10.0% |
| sb_xg | 0.755 | 0.0775 | 0.2764 | 9.7% |
| baseline_image_xg | 0.685 | 0.0848 | 0.3014 | 10.2% |

## Model and prompt screen: 50 reused AFCON shots

| Method | AUC ↑ | Brier ↓ | Log loss ↓ | Mean xG |
|---|---:|---:|---:|---:|
| luna_uncertainty | 0.498 | 0.1179 | 0.4190 | 13.9% |
| terra_original | 0.428 | 0.1227 | 0.4454 | 12.3% |
| terra_uncertainty | 0.487 | 0.1222 | 0.4318 | 12.7% |
| luna_original | 0.557 | 0.1240 | 0.4117 | 17.7% |
| sol_original | 0.475 | 0.1333 | 0.4569 | 16.8% |
| baseline_xg | 0.553 | 0.1157 | 0.4263 | 10.1% |
| baseline_image_xg | 0.508 | 0.1180 | 0.4294 | 10.0% |
| sb_xg | 0.568 | 0.1113 | 0.4203 | 10.4% |

## Independent La Liga 2015/16: 200 shots

| Method | AUC ↑ | Brier ↓ | Log loss ↓ | Mean xG |
|---|---:|---:|---:|---:|
| luna_uncertainty | 0.764 | 0.0753 | 0.2693 | 15.1% |
| luna_original | 0.808 | 0.0787 | 0.2695 | 16.3% |
| baseline_xg | 0.805 | 0.0662 | 0.2352 | 9.4% |
| baseline_image_xg | 0.805 | 0.0674 | 0.2375 | 9.6% |
| sb_xg | 0.867 | 0.0580 | 0.2070 | 8.9% |

## Independent Premier League: 500 shots

| Method | AUC ↑ | Brier ↓ | Log loss ↓ | Mean xG |
|---|---:|---:|---:|---:|
| vlm_xg | 0.711 | 0.0926 | 0.3199 | 16.4% |
| original_logistic | 0.737 | 0.0807 | 0.2912 | 9.7% |
| original_no_header | 0.711 | 0.0826 | 0.2998 | 9.4% |
| local_logistic | 0.726 | 0.0815 | 0.2966 | 10.2% |
| local_no_header | 0.711 | 0.0832 | 0.3035 | 10.0% |
| sb_xg | 0.790 | 0.0737 | 0.2646 | 10.0% |
| training_prior | 0.500 | 0.0900 | 0.3251 | 9.7% |

## Independent Women's World Cup: 500 shots

| Method | AUC ↑ | Brier ↓ | Log loss ↓ | Mean xG |
|---|---:|---:|---:|---:|
| vlm_xg | 0.745 | 0.0843 | 0.3030 | 17.3% |
| original_logistic | 0.788 | 0.0666 | 0.2464 | 10.3% |
| original_no_header | 0.769 | 0.0679 | 0.2541 | 10.0% |
| local_logistic | 0.780 | 0.0658 | 0.2564 | 10.4% |
| local_no_header | 0.767 | 0.0669 | 0.2626 | 10.1% |
| sb_xg | 0.845 | 0.0623 | 0.2301 | 9.7% |
| training_prior | 0.500 | 0.0786 | 0.2933 | 8.9% |

## Metrica pilot: 46 shots

| Method | AUC ↑ | Brier ↓ | Log loss ↓ | Mean xG |
|---|---:|---:|---:|---:|
| vlm_xg | 0.804 | 0.1054 | 0.3436 | 18.2% |
| original_logistic | 0.810 | 0.1013 | 0.3398 | 10.5% |
| original_no_header | 0.777 | 0.1060 | 0.3537 | 11.6% |
| training_prior | 0.500 | 0.1325 | 0.4438 | 9.4% |

## Independent La Liga 2016/17: 150 shots

| Method | AUC ↑ | Brier ↓ | Log loss ↓ | Mean xG |
|---|---:|---:|---:|---:|
| luna | 0.640 | 0.1382 | 0.4479 | 16.9% |
| astra | 0.721 | 0.1231 | 0.4084 | 14.9% |
| examples | 0.655 | 0.1319 | 0.4285 | 16.5% |
| shuffled | 0.746 | 0.1215 | 0.3968 | 16.2% |
| original_logistic | 0.744 | 0.1264 | 0.4119 | 10.0% |
| original_no_header | 0.724 | 0.1322 | 0.4303 | 9.2% |
| sb_xg | 0.781 | 0.1194 | 0.3958 | 11.0% |
| training_prior | 0.500 | 0.1442 | 0.4767 | 9.4% |
| example_rate | 0.500 | 0.1458 | 0.4867 | 8.3% |

## Past tracking: 43 reused Metrica shots

| Method | AUC ↑ | Brier ↓ | Log loss ↓ | Mean xG |
|---|---:|---:|---:|---:|
| original | 0.687 | 0.1138 | 0.3720 | 17.3% |
| single_ball | 0.775 | 0.1037 | 0.3421 | 17.8% |
| history | 0.658 | 0.1363 | 0.4324 | 20.2% |
| original_logistic | 0.770 | 0.1065 | 0.3547 | 9.5% |
| original_no_header | 0.730 | 0.1095 | 0.3650 | 10.9% |

## First hybrid: 300 World Cup shots

| Method | AUC ↑ | Brier ↓ | Log loss ↓ | Mean xG |
|---|---:|---:|---:|---:|
| vlm_xg | 0.728 | 0.1124 | 0.3687 | 18.1% |
| hybrid_xg | 0.723 | 0.1150 | 0.3816 | 11.1% |
| reference_geometry_xg | 0.730 | 0.1107 | 0.3688 | 10.5% |
| distance_angle_xg | 0.703 | 0.1136 | 0.3775 | 10.9% |
| baseline_xg | 0.762 | 0.1070 | 0.3563 | 10.7% |
| sb_xg | 0.793 | 0.0989 | 0.3378 | 10.0% |

## Spatial development: 3068 World Cup shots

| Method | AUC ↑ | Brier ↓ | Log loss ↓ | Mean xG |
|---|---:|---:|---:|---:|
| spline_context | 0.791 | 0.0710 | 0.2541 | 9.3% |
| linear_context | 0.792 | 0.0714 | 0.2549 | 9.3% |
| spline_image | 0.784 | 0.0719 | 0.2571 | 9.3% |
| linear_image | 0.786 | 0.0720 | 0.2573 | 9.4% |
| hgb7_context | 0.776 | 0.0729 | 0.2607 | 9.3% |
| hgb15_context | 0.776 | 0.0733 | 0.2628 | 9.3% |
| hgb7_image | 0.769 | 0.0735 | 0.2634 | 9.3% |
| baseline_context | 0.773 | 0.0740 | 0.2648 | 9.3% |
| hgb15_image | 0.768 | 0.0741 | 0.2663 | 9.3% |
| baseline_image | 0.758 | 0.0757 | 0.2711 | 9.4% |

## Known coordinates: 1304 Euro 2024 shots

| Method | AUC ↑ | Brier ↓ | Log loss ↓ | Mean xG |
|---|---:|---:|---:|---:|
| StatsBomb | 0.787 | 0.0615 | 0.2251 | 8.5% |
| baseline_image | 0.718 | 0.0658 | 0.2446 | 8.8% |
| baseline_context | 0.746 | 0.0649 | 0.2388 | 8.8% |
| spatial_image | 0.764 | 0.0642 | 0.2343 | 8.6% |
| spatial_context | 0.774 | 0.0640 | 0.2322 | 8.7% |

## Known coordinates: 1234 Euro 2020 shots

| Method | AUC ↑ | Brier ↓ | Log loss ↓ | Mean xG |
|---|---:|---:|---:|---:|
| StatsBomb | 0.812 | 0.0703 | 0.2508 | 9.7% |
| baseline_image | 0.752 | 0.0797 | 0.2830 | 9.7% |
| baseline_context | 0.767 | 0.0777 | 0.2768 | 9.8% |
| spatial_image | 0.794 | 0.0734 | 0.2627 | 9.8% |
| spatial_context | 0.799 | 0.0736 | 0.2616 | 9.9% |

## Initial richer hybrid: 200 Euro 2024 shots

| Method | AUC ↑ | Brier ↓ | Log loss ↓ | Mean xG |
|---|---:|---:|---:|---:|
| vlm_direct | 0.635 | 0.0797 | 0.2907 | 11.5% |
| hybrid_image | 0.675 | 0.0776 | 0.2789 | 9.0% |
| hybrid_context | 0.675 | 0.0787 | 0.2802 | 9.4% |
| spatial_image | 0.697 | 0.0730 | 0.2657 | 8.1% |
| spatial_context | 0.697 | 0.0743 | 0.2678 | 8.5% |
| baseline_image | 0.669 | 0.0753 | 0.2746 | 8.6% |
| baseline_context | 0.705 | 0.0757 | 0.2704 | 8.9% |
| sb_xg | 0.690 | 0.0745 | 0.2700 | 8.7% |

## Improvement screen: 200 reused Euro 2024 shots

| Method | AUC ↑ | Brier ↓ | Log loss ↓ | Mean xG |
|---|---:|---:|---:|---:|
| original_hybrid | 0.675 | 0.0776 | 0.2789 | 9.0% |
| noise_trained | 0.677 | 0.0750 | 0.2734 | 8.5% |
| multiview | 0.683 | 0.0745 | 0.2711 | 8.6% |
| multiview_noise_head | 0.687 | 0.0727 | 0.2670 | 8.1% |
| fewshot | 0.676 | 0.0779 | 0.2799 | 9.2% |
| fewshot_noise_head | 0.676 | 0.0752 | 0.2741 | 8.7% |
| pixels | 0.702 | 0.0743 | 0.2680 | 8.2% |
| pixels_noise_head | 0.705 | 0.0726 | 0.2642 | 7.8% |
| baseline_image | 0.669 | 0.0753 | 0.2746 | 8.6% |
| sb_xg | 0.690 | 0.0745 | 0.2700 | 8.7% |
| hybrid_plus_context | 0.692 | 0.0790 | 0.2787 | 9.1% |
| noise_training_plus_context | 0.695 | 0.0763 | 0.2721 | 8.6% |
| training_rate_constant | 0.500 | 0.0738 | 0.2799 | 9.4% |

## Direct position-first outputs: 200 reused Euro shots

| Method | AUC ↑ | Brier ↓ | Log loss ↓ | Mean xG |
|---|---:|---:|---:|---:|
| original_direct | 0.635 | 0.0797 | 0.2907 | 11.5% |
| multiview_direct | 0.658 | 0.0792 | 0.2839 | 11.7% |
| fewshot_direct | 0.618 | 0.0868 | 0.3081 | 10.7% |

## Pre-shot event context: 200 reused Euro shots

| Method | AUC ↑ | Brier ↓ | Log loss ↓ | Mean xG |
|---|---:|---:|---:|---:|
| original_hybrid | 0.675 | 0.0776 | 0.2789 | 9.0% |
| hybrid_plus_context | 0.692 | 0.0790 | 0.2787 | 9.1% |
| noise_training_plus_context | 0.695 | 0.0763 | 0.2721 | 8.6% |
| noise_training_image_only | 0.677 | 0.0750 | 0.2734 | 8.5% |
| simple_known_positions | 0.669 | 0.0753 | 0.2746 | 8.6% |
| simple_plus_context_known_positions | 0.722 | 0.0746 | 0.2664 | 8.6% |
| rich_known_positions | 0.697 | 0.0730 | 0.2657 | 8.1% |
| rich_plus_context_known_positions | 0.709 | 0.0754 | 0.2676 | 8.3% |
| statsbomb | 0.690 | 0.0745 | 0.2700 | 8.7% |

## Independent Bundesliga: 200 shots

| Method | AUC ↑ | Brier ↓ | Log loss ↓ | Mean xG |
|---|---:|---:|---:|---:|
| sb_xg | 0.667 | 0.0825 | 0.3002 | 9.5% |
| original_hybrid | 0.697 | 0.0801 | 0.2915 | 9.4% |
| noise_trained | 0.696 | 0.0800 | 0.2919 | 8.9% |
| pixels | 0.696 | 0.0806 | 0.2920 | 9.2% |
| pixels_noise_head | 0.694 | 0.0806 | 0.2922 | 8.7% |
| simple_known_positions | 0.641 | 0.0825 | 0.3053 | 9.2% |
| rich_known_positions | 0.694 | 0.0810 | 0.2931 | 9.0% |
| direct_luna | 0.612 | 0.0914 | 0.3293 | 12.6% |
| noise_plus_context | 0.700 | 0.0816 | 0.2929 | 9.4% |
| simple_plus_context_known_positions | 0.682 | 0.0823 | 0.2986 | 9.5% |

## Secondary full Bundesliga pool: 908 shots, includes 200-shot subset

| Method | AUC ↑ | Brier ↓ | Log loss ↓ | Mean xG |
|---|---:|---:|---:|---:|
| sb_xg | 0.787 | 0.0826 | 0.2891 | 10.4% |
| rich_known_positions | 0.749 | 0.0853 | 0.3096 | 9.9% |
| noise_head_known_positions | 0.753 | 0.0854 | 0.3052 | 9.3% |
| simple_known_positions | 0.718 | 0.0891 | 0.3176 | 9.5% |
| training_rate_constant | 0.500 | 0.1001 | 0.3533 | 9.4% |
| rich_plus_context | 0.764 | 0.0841 | 0.3025 | 10.3% |
| simple_plus_context | 0.756 | 0.0860 | 0.3042 | 9.9% |

## Known positions and context: full Euro 2024

| Method | AUC ↑ | Brier ↓ | Log loss ↓ | Mean xG |
|---|---:|---:|---:|---:|
| sb_xg | 0.787 | 0.0615 | 0.2251 | 8.5% |
| original_rich | 0.764 | 0.0642 | 0.2343 | 8.6% |
| rich_plus_context | 0.776 | 0.0636 | 0.2307 | 8.7% |
| simple_plus_context | 0.758 | 0.0644 | 0.2361 | 8.7% |

## Known positions and context: full Euro 2020

| Method | AUC ↑ | Brier ↓ | Log loss ↓ | Mean xG |
|---|---:|---:|---:|---:|
| sb_xg | 0.812 | 0.0703 | 0.2508 | 9.7% |
| original_rich | 0.794 | 0.0734 | 0.2627 | 9.8% |
| rich_plus_context | 0.799 | 0.0726 | 0.2588 | 10.1% |
| simple_plus_context | 0.778 | 0.0760 | 0.2709 | 10.0% |

## Specialists: 200 reused Bundesliga shots

| Method | AUC ↑ | Brier ↓ | Log loss ↓ | Mean xG |
|---|---:|---:|---:|---:|
| monolithic_image | 0.676 | 0.0827 | 0.2996 | 9.2% |
| specialists_image | 0.688 | 0.0807 | 0.2938 | 8.9% |
| monolithic_context | 0.685 | 0.0842 | 0.3007 | 9.7% |
| specialists_context | 0.696 | 0.0816 | 0.2934 | 9.4% |
| metadata_only | 0.652 | 0.0869 | 0.3086 | 10.3% |
| training_rate | 0.500 | 0.0860 | 0.3140 | 9.4% |
| previous_noise_hybrid | 0.696 | 0.0800 | 0.2919 | 8.9% |
| pixels_image | 0.694 | 0.0806 | 0.2922 | 8.7% |
| pixels_context | 0.698 | 0.0820 | 0.2936 | 9.2% |
| known_image | 0.690 | 0.0810 | 0.2937 | 8.5% |
| known_context | 0.699 | 0.0822 | 0.2942 | 9.1% |
| simple_known_positions | 0.641 | 0.0825 | 0.3053 | 9.2% |
| sb_xg | 0.667 | 0.0825 | 0.3002 | 9.5% |

## Body part: image-only control, 50 World Cup shots

| Method | AUC ↑ | Brier ↓ | Log loss ↓ | Mean xG |
|---|---:|---:|---:|---:|
| vlm_xg | 0.610 | 0.1276 | 0.4069 | 19.5% |
| baseline_xg | 0.674 | 0.1155 | 0.3910 | 13.0% |
| sb_xg | 0.601 | 0.1111 | 0.4220 | 11.6% |

## Body part supplied: same 50 World Cup shots

| Method | AUC ↑ | Brier ↓ | Log loss ↓ | Mean xG |
|---|---:|---:|---:|---:|
| vlm_xg | 0.658 | 0.1204 | 0.3915 | 17.5% |
| baseline_xg | 0.674 | 0.1155 | 0.3910 | 13.0% |
| sb_xg | 0.601 | 0.1111 | 0.4220 | 11.6% |
