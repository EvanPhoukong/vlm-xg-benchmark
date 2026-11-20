"""Build the professor-facing summary exclusively from compiled saved metrics."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parent
M=json.loads((ROOT/'metrics.json').read_text())
def row(group,key,label):
    return label,M[group][key]
def table(rows,first='Method'):
    lines=[f'| {first} | AUC ↑ | Brier ↓ | Log loss ↓ | Mean xG |','|---|---:|---:|---:|---:|']
    for label,v in rows:
        lines.append(f"| {label} | {v['auc']:.3f} | {v['brier']:.4f} | {v['log_loss']:.4f} | {v['mean_xg']:.1%} |")
    return '\n'.join(lines)
parts=["""# Can visual models estimate expected goals?
Research summary through 30 October 2026

We tested whether general-purpose vision-language models can estimate the probability of a goal from a shot image. Most inputs were standardized pitch diagrams showing the shooter, teammates, opponents and goalkeeper. Later experiments tested different prompts, reasoning settings, models, datasets, visual examples and past tracking information. Separate historical studies used the visual model to measure positions for a trained statistical model. Recalibration experiments are excluded.

**Reading the tables:** Higher AUC means better discrimination between shots that scored and those that did not. Lower Brier and log loss mean better probability predictions; Brier was our main score. Mean xG is the average predicted probability, not an accuracy score. Compare methods on the same shots: goal rates and difficulty vary between datasets.

## 1. Initial benchmark

The table uses the **298 shots shared by all five saved model files**, including 42 goals (14.1%). We aligned shot identifiers and checked outcomes and coordinates; predictions were unchanged. Earlier 300-shot summaries therefore differ slightly.""",
table(M['Initial benchmark: common 298 shots'].items()),
"""Opus was closest to the logistic baseline on probability error, but the initial study did not establish a VLM advantage. Similar ranking did not imply similar probability estimates: Gemini and Opus had the same AUC but different mean xG.

## 2. Changes to direct visual predictions

Each cell below shows **original → changed condition** on the same shots. These are small development or exploratory comparisons, not independent confirmation. The two 50-shot AFCON studies used separate model calls."""]

comparisons=[
('Add body-part information (50 World Cup shots)','Body part: image-only control, 50 World Cup shots','vlm_xg','Body part supplied: same 50 World Cup shots','vlm_xg'),
('Low → medium reasoning: early pilot (12)','Reasoning pilot: 12 reused World Cup shots','low','Reasoning pilot: 12 reused World Cup shots','medium'),
('Low → medium reasoning: larger AFCON test (50)','Reasoning: 50 AFCON development shots','low','Reasoning: 50 AFCON development shots','medium'),
('Low → high reasoning: larger AFCON test (50)','Reasoning: 50 AFCON development shots','low','Reasoning: 50 AFCON development shots','high'),
('Filled → hollow shooter marker (40)','Marker: 40 reused World Cup shots','standard','Marker: 40 reused World Cup shots','outline'),
('Position-first → repeated/zoom views (200 Euro shots)','Direct position-first outputs: 200 reused Euro shots','original_direct','Direct position-first outputs: 200 reused Euro shots','multiview_direct'),
('Position-first → four geometry examples (same 200)','Direct position-first outputs: 200 reused Euro shots','original_direct','Direct position-first outputs: 200 reused Euro shots','fewshot_direct'),
('Luna → Terra, original prompt (50 AFCON shots)','Model and prompt screen: 50 reused AFCON shots','luna_original','Model and prompt screen: 50 reused AFCON shots','terra_original'),
('Luna → Sol, original prompt (same 50)','Model and prompt screen: 50 reused AFCON shots','luna_original','Model and prompt screen: 50 reused AFCON shots','sol_original'),
('Original → missing-information prompt, Luna (same 50)','Model and prompt screen: 50 reused AFCON shots','luna_original','Model and prompt screen: 50 reused AFCON shots','luna_uncertainty'),
('Original → missing-information prompt, Terra (same 50)','Model and prompt screen: 50 reused AFCON shots','terra_original','Model and prompt screen: 50 reused AFCON shots','terra_uncertainty'),
]
lines=['| Change tested (number of shots) | AUC ↑ | Brier ↓ | Log loss ↓ | Mean xG |','|---|---:|---:|---:|---:|']
for label,ga,ka,gb,kb in comparisons:
    a,b=M[ga][ka],M[gb][kb]
    lines.append(f"| {label} | {a['auc']:.3f} → {b['auc']:.3f} | {a['brier']:.4f} → {b['brier']:.4f} | {a['log_loss']:.4f} → {b['log_loss']:.4f} | {a['mean_xg']:.1%} → {b['mean_xg']:.1%} |")
parts+=['\n'.join(lines),
"""The early reasoning benefit did not repeat in the larger test. Neither the hollow marker nor four geometry examples established an improvement. The missing-information prompt lowered Luna's development Brier but weakened ranking; it was then tested on new matches below. Luna and Sol used low reasoning; Terra used medium, so these are comparisons of complete model configurations.

A separate **50-image geometry diagnostic** found small average shooter and goalkeeper location errors (0.23 and 0.15 normalized pitch units). Angle error averaged 1.91°, falling to 0.64° when code calculated angles from the extracted positions. These are measurement results, not xG accuracy scores. Larger studies revealed occasional costly errors and defenders hidden by overlapping markers.

## 3. Checks on new matches and different datasets

The main VLM and logistic results are shown together. Brackets give **shots / goals**. The Metrica comparison is a two-game pilot, rather than strong independent confirmation."""]

rows=[]
for group,short,n,g,keys in [
('Independent Bundesliga: 200 shots','Bundesliga 2023/24',200,19,[('direct_luna','Direct Luna'),('simple_known_positions','Logistic')]),
('Independent AFCON: 150 shots','AFCON 2023',150,14,[('original_low','Luna'),('baseline_xg','Logistic')]),
('Independent La Liga 2015/16: 200 shots','La Liga 2015/16',200,16,[('luna_original','Original Luna'),('luna_uncertainty','Luna, missing-information prompt'),('baseline_xg','Logistic')]),
('Independent Premier League: 500 shots','Premier League 2015/16',500,50,[('vlm_xg','Luna'),('original_logistic','Logistic')]),
("Independent Women's World Cup: 500 shots","Women's World Cup 2019",500,43,[('vlm_xg','Luna'),('original_logistic','Logistic')]),
('Metrica pilot: 46 shots','Metrica pilot',46,7,[('vlm_xg','Luna'),('original_logistic','Logistic')])]:
    for key,label in keys: rows.append(row(group,key,f'{short} [{n}/{g}] — {label}'))
parts += [table(rows,'Test set [shots/goals] and method'),
"""Changing datasets did not solve Luna's probability errors. Logistic regression had reliably lower Brier on the Premier League and Women's World Cup cohorts under the planned match-based intervals. On La Liga 2015/16, the missing-information prompt's small Brier improvement was inconclusive, and ranking fell.

## 4. Stronger model and outcome examples

The latest fixed test used **150 new La Liga 2016/17 shots, 25 goals and 34 matches**. Astra received the same original prompt and image as Luna. Separate Luna conditions received 48 historical images with goal/no-goal outcomes, or the same examples with shuffled labels as a negative control.""",
table([row('Independent La Liga 2016/17: 150 shots',k,l) for k,l in [
('luna','Original Luna'),('astra','Astra'),('examples','Luna + correct outcome examples'),
('shuffled','Luna + shuffled-label control'),('original_logistic','Logistic'),('sb_xg','StatsBomb')]]),
"""Astra reduced Brier by **11% versus Luna**. The 99% match-based interval for the difference was **−0.0275 to −0.0022**, adjusted for the five planned comparisons in this round. Its small point advantage over logistic was inconclusive: **−0.0191 to +0.0097**. Correct examples did not establish an advantage over shuffled labels; the control's favorable point score is not evidence of useful learning.

## 5. Earlier hybrid, training and specialist experiments

These methods used extracted positions and a **trained statistical probability model**, rather than asking the VLM alone for final xG. Noise training changed that statistical model, not the VLM. The table keeps representative conditions; the complete tables retain auxiliary controls and variants.""",
table([row(g,k,l) for g,k,l in [
('First hybrid: 300 World Cup shots','hybrid_xg','World Cup, 300 — first shooter/keeper hybrid'),
('Improvement screen: 200 reused Euro 2024 shots','original_hybrid','Euro 2024, 200 — richer hybrid'),
('Improvement screen: 200 reused Euro 2024 shots','noise_trained','Same Euro sample — noisy-position training'),
('Improvement screen: 200 reused Euro 2024 shots','multiview_noise_head','Same Euro sample — multiple views + noise training'),
('Improvement screen: 200 reused Euro 2024 shots','fewshot','Same Euro sample — four geometry examples'),
('Improvement screen: 200 reused Euro 2024 shots','pixels_noise_head','Same Euro sample — pixel reader + noise training'),
('Improvement screen: 200 reused Euro 2024 shots','noise_training_plus_context','Same Euro sample — noise training + event context'),
('Independent Bundesliga: 200 shots','original_hybrid','Bundesliga, 200 — original richer hybrid'),
('Independent Bundesliga: 200 shots','noise_trained','Same Bundesliga sample — noisy-position training'),
('Specialists: 200 reused Bundesliga shots','monolithic_image','Same Bundesliga sample — new single reader'),
('Specialists: 200 reused Bundesliga shots','specialists_image','Same Bundesliga sample — two specialist readers'),
('Specialists: 200 reused Bundesliga shots','specialists_context','Same Bundesliga sample — specialists + event context'),
]]),
"""Noise training and multiple views improved exploratory Euro scores, but the fresh Bundesliga noise-training comparison did not confirm a benefit. Four geometry examples did not help. Event context sometimes improved ranking while worsening Brier. Two specialist readers did not establish an improvement over one reader and did not beat the earlier noise-trained pipeline by point estimate.

The pixel reader is a deterministic computer-vision control that exploits the diagrams' fixed colors; it is not a VLM result. Separate linear, spline and boosted-tree statistical candidates were developed on 3,068 World Cup shots. The selected richer model, using **known dataset coordinates**, improved on a simpler matched baseline in the 1,234-shot Euro 2020 test (Brier **0.0734 versus 0.0797**, AUC **0.794 versus 0.752**). The corresponding Euro 2024 gain was inconclusive. These results support richer spatial features, not direct VLM superiority.

Two studies remained incomplete: the clearer, role-separated Euro 2020 images stopped at **62/200** responses; confirmation of the multiple-view/noise-training candidate stopped at **46/200**. Neither has a final accuracy result.

## 6. Past tracking and consistency

On **43 reused Metrica shots, six goals and two games**, we compared snapshots with two actual earlier tracking frames. These were diagrams, not broadcast footage.""",
table([row('Past tracking: 43 reused Metrica shots',k,l) for k,l in [
('original','Original snapshot'),('single_ball','Current frame + ball marker'),('history','Past frames + identical current frame')]]),
"""Adding history worsened this pilot. The current-frame condition also changed the original prompt, so its advantage cannot be attributed solely to the ball marker.

Repeated calls and controlled image changes exposed instability. In earlier tests, exact repeats changed xG by roughly 4–7 percentage points on average, and the missing-information prompt reached 12.8 points. In the latest 12-scene check, Astra's repeat change was **0.88 points**, versus **6.75 for Luna**. Astra followed the expected direction in 10/12 synthetic comparisons, versus Luna's 9/12. These small, correlated diagnostics do not establish population-wide consistency or true shot probabilities.

## Key findings across the experiments

- **Model choice mattered more than the tested prompt and reasoning adjustments.** Astra is the clearest supported improvement over Luna on a new fixed cohort, with better Brier, ranking and observed repeat stability.
- **Better ranking is not the same as accurate xG.** Several models ordered shots reasonably while predicting probabilities that were too high or too low. Luna's short-range overprediction persisted across datasets.
- **The Astra gain extends beyond matching the overall average.** Luna already averaged 16.9% against 16.7% actual scoring on the latest cohort; Astra averaged 14.9% but made better individual predictions.
- **Most proposed fixes remain unconfirmed.** More reasoning, clearer markers, examples, event context, task division and past tracking did not establish consistent improvements on new matches. Several hybrid gains weakened outside the development sample.
- **We have evidence of improvement, not yet proof that direct visual xG matches established statistical models.** Astra needs independent replication and a stronger baseline comparison. Richer known-coordinate models demonstrate useful spatial information, while current VLM extraction and probability judgments do not consistently retain its value.

**Interpretation limits:** Actual goal/no-goal outcomes were used for evaluation; an individual shot's true probability is unobserved. StatsBomb is a reference model, not ground truth. The four-feature logistic baseline receives a header flag absent from the diagrams, as well as recorded coordinates; later studies also checked no-header baselines separately. Many older samples were reused. Most inputs omit body pose, ball height and execution, so this work does not establish accuracy on broadcast video. No VLM fine-tuning was performed. An inconclusive difference does not prove equivalence.
"""]
text='\n\n'.join(parts).strip()+'\n'
(ROOT/'SUMMARY.md').write_text(text,encoding='utf-8')
print(json.dumps({'words_whitespace':len(text.split()),'lines':len(text.splitlines()),
    'summary':str(ROOT/'SUMMARY.md'),'recalibration_rows_included':False,'paid_calls':0},indent=2))

