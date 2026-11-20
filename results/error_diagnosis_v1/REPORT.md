# Why the tested changes have not improved direct xG reliably

Post-hoc offline diagnosis of existing predictions. No paid calls, probability corrections or source-output changes. This localizes observed errors; it does not establish one causal failure inside the VLM.

## The main observed problem

Raw Luna repeatedly overpredicts ordinary short-range chances:

| Cohort, 6–12 normalized pitch units | Shots | Goals | Mean raw Luna xG | Observed scoring |
|---|---:|---:|---:|---:|
| Premier League | 106 | 13 | 31.9% | 12.3% |
| Women's World Cup | 125 | 18 | 32.3% | 14.4% |

Distances are normalized coordinate units, not verified physical yards/metres. These bins were inspected after the full study and are descriptive, not new prespecified subgroup tests. The repeated pattern agrees with the original study's short-range concern.

The 6–12 bin contributes 0.007960 of the Premier League's total 0.011925 Brier gap versus the original four-feature logistic, and 0.008381 of the Women's World Cup's 0.017715 gap. These are exact additive contributions on the recorded samples, not confidence intervals or causal percentages.

## More than an overall mean problem

An exact algebraic decomposition separates squared overall mean bias from error after centering both predictions and outcomes. Against the no-header logistic comparator, the difference in squared overall bias accounts for approximately 40% of Luna's Premier League Brier deficit and 45% of its Women's World Cup deficit. The rest remains in centered error, which mixes probability spread and association with outcomes. Centered error cannot be equated with visual measurement error.

No probabilities were centered, shifted or rescaled for a reported performance result: this is an accounting identity applied to the existing errors. It supports the user's objection to treating a mean correction as a complete fix.

On Premier League shots, Luna and the no-header logistic have almost identical AUC (0.71087 and 0.71080), while their Brier scores differ materially (0.09265 and 0.08258). The VLM has some ability to order chances, but its numerical probability assignment is weaker. Women's World Cup also shows weaker ranking: 0.745 versus 0.769 without the header flag.

## What the current evidence can explain

1. The tested prompt/reasoning changes have not supplied task-specific supervised learning. The logistic baseline was fitted on 3,068 labelled shots; our direct VLM method remains zero-shot. More inference reasoning does not itself add new labelled examples.
2. All recent datasets were converted to the same coloured-marker diagrams. They expose positions but omit body pose, ball height/motion, balance and preceding play. More source datasets have not restored those visual cues. The no-header baseline still beats Luna in both larger cohorts, so missing body-part information alone does not explain the deficit.
3. Probability output is unstable. Original Luna's controlled exact repeats differed by 6.3 percentage points on average; the missing-information prompt differed by 12.8 points. The latter showed fewer expected-direction changes. These are small correlated synthetic probes, not universal variance estimates.
4. Model coverage is limited. The large new transfer cohorts used Luna. Terra and Sol were screened on only 50 development shots with six goals; new Claude/Gemini runs were unavailable. This does not establish that every visual model has the same limitation.

An earlier 50-image geometry diagnostic found small average position/distance errors under an explicit measurement prompt, alongside occasional angle mistakes. Because that prompt differs from direct xG elicitation and a larger study revealed rare location errors, those measurements do not prove the original xG errors are unrelated to perception.

## Implication for the next experiment

Further population swaps can measure how widely the problem persists, but do not teach the fixed model a better probability mapping. The more informative input experiment is a paired, outcome-blind comparison of real pre-shot imagery and the diagram for the same shots, with an unchanged model and raw outputs. It would test whether omitted context is a material bottleneck; improvement is not guaranteed. Authorized xGHub footage could support this direction, but its released clips require approved institutional access.

A stronger-model comparison on a fixed independent cohort is another unresolved test. The initial small development screens cannot settle the capabilities of all model families. End-to-end task-specific visual learning would be a separate research scope from the current zero-shot benchmark, rather than a post-hoc output correction; it has not been authorized or performed as part of this diagnosis.

Evidence: [new dataset study](../dataset_transfer_v1/REPORT.md), [controlled visual changes](../visual_counterfactuals_v1/REPORT.md), [earlier geometry diagnostic](../geometry_gpt_6_luna_50/summary.md), [original study context](../../ORIGINAL_STUDY_CONTEXT.md). Computations: diagnosis.json; code: experiments/error_diagnosis/diagnose.py.

Related background: [Groot and Valdenegro-Toro, 2024](https://arxiv.org/abs/2405.02917) found unreliable prompted uncertainty in older LLMs/VLMs. It supplies general motivation, not a model-specific explanation of Luna's football errors. [OpenAI's vision limitations](https://developers.openai.com/api/docs/guides/images-vision) also flag spatial/counting inaccuracies; those general limitations do not identify the cause of a specific xG miss.
