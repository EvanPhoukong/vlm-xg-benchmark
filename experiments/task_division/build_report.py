"""Present completed fixed-protocol results; never makes API calls or fits models."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from run_experiment import ROOT, LABELS

def build(folder):
    protocol=json.loads((folder/'protocol.json').read_text())
    integrity=json.loads((folder/'integrity.json').read_text())
    status=json.loads((folder/'status.json').read_text())
    if status['status']!='complete' or integrity['complete_requests']!=600:
        raise ValueError('Only completed cohorts may be reported.')
    scores=pd.read_csv(folder/'scores.csv')
    comparisons=pd.read_csv(folder/'comparisons.csv')
    geometry=pd.read_csv(folder/'geometry_summary.csv')
    usage=pd.read_csv(folder/'usage.csv')
    predictions=pd.read_csv(folder/'predictions.csv')
    by_method=scores.set_index('method')
    display=['monolithic_image','specialists_image','monolithic_context','specialists_context',
             'previous_noise_hybrid','metadata_only','pixels_context','simple_known_positions','sb_xg']
    figure,axis=plt.subplots(figsize=(10.5,6.3))
    values=by_method.loc[display,'brier']
    bars=axis.barh([LABELS[m] for m in display],values,
    color=['#526d82','#168a7b','#526d82','#168a7b','#61764b','#a4aab3','#d0a44c','#a4aab3','#a4aab3'])
    axis.invert_yaxis()
    for bar,value in zip(bars,values):
        axis.text(value+.0008,bar.get_y()+bar.get_height()/2,f'{value:.5f}',va='center',fontsize=10)
    axis.set_xlim(0,max(values)*1.17)
    axis.set_xlabel('Brier score - lower is better')
    axis.set_title('Task division and event-context fusion\n200 existing Bundesliga shots - exploratory reuse')
    axis.spines[['top','right']].set_visible(False)
    figure.tight_layout(); figure.savefig(folder/'comparison.png',dpi=170); plt.close(figure)
    findings=[]
    for row in comparisons[comparisons.primary].itertuples():
        verdict='favors task division' if row.ci_high<0 else ('favors one reader' if row.ci_low>0 else 'does not establish a difference')
        findings.append(f'- {LABELS[row.first]} versus {LABELS[row.second]}: Brier difference **{row.brier_difference:+.5f}**, adjusted 97.5% interval **[{row.ci_low:+.5f}, {row.ci_high:+.5f}]**; {verdict}.')
    table=['| Method | Brier - lower | AUC - higher | Log loss - lower | Mean xG |','|---|---:|---:|---:|---:|']
    for row in scores.itertuples():
        table.append(f'| {row.label} | {row.brier:.5f} | {row.auc:.3f} | {row.log_loss:.5f} | {row.mean_xg:.1%} |')
    intervals=['| First minus second | Brier difference | Interval |','|---|---:|---:|']
    for row in comparisons.itertuples():
        intervals.append(f'| {LABELS[row.first]} minus {LABELS[row.second]} | {row.brier_difference:+.5f} | {row.confidence_level:.1%}: [{row.ci_low:+.5f}, {row.ci_high:+.5f}] |')
    measurement=['| Reader | Shooter mean error | Keeper mean error | Defender count MAE | Defender recall within 2 units |',
                 '|---|---:|---:|---:|---:|']
    for row in geometry.itertuples():
        measurement.append(f'| {row.method} | {row.shooter_mean_error_returned:.3f} | {row.keeper_mean_error_returned:.3f} | {row.defender_count_mae:.3f} | {row.defender_recall_within_2:.1%} |')
    context_row=comparisons[(comparisons['first']=='specialists_context')&(comparisons.second=='specialists_image')].iloc[0]
    context_verdict='lower' if context_row.brier_difference<0 else 'higher'
    text='\n'.join([
    '# Luna task division and multimodal fusion','',
    'Research date: 23 October 2026 (America/Los_Angeles).','',
    f'Completed **200 shots, {int(predictions.goal.sum())} goals, {predictions.match_id.nunique()} matches**, using **600 successful Luna requests**. Only gpt-6-luna was requested. No new dataset was downloaded.','',
    '## Main findings','',*findings,'',
    f'The previous noise-trained Luna pipeline scored Brier **{by_method.loc["previous_noise_hybrid","brier"]:.5f}**, compared with **{by_method.loc["specialists_image","brier"]:.5f}** for the specialists. Task division therefore does not improve the previous pipeline by point estimate; the new monolithic control is a different geometry-only prompt.','',
    f'Adding saved event context to the specialist pipeline produced {context_verdict} Brier by {abs(context_row.brier_difference):.5f}; secondary 95% interval [{context_row.ci_low:+.5f}, {context_row.ci_high:+.5f}]. This comparison changes from the frozen image head to the frozen context head and is not a primary task-division test.','',
    'Negative paired differences favor the first method. The two primary intervals adjust for two comparisons. All results reuse a previously studied sample and remain exploratory.','',
    '![Probability error comparison](comparison.png)','',
    '## Probability results','',*table,'',
    'Every method retains the same 200 shots. Brier and log loss measure probability error; lower is better. AUC measures ranking; higher is better. Mean xG is not accuracy.','',
    '## Division of work','',
    '1. A shooter/goalkeeper Luna specialist reads those two markers.',
    '2. A separate defender Luna specialist reads the visible red defender markers.',
    '3. Code concatenates these measurements and computes geometry.',
    '4. Frozen supervised probability heads score geometry alone or geometry plus saved event context.','',
    'A new single-reader geometry-only Luna control reads all markers in one call. All roles receive the identical original diagram, shared geometry instructions and model settings. None estimates xG. Role order is randomized within each shot, and readers never see one another\'s answers.','',
    'The multimodal condition is late fusion of image-derived measurements and structured event metadata. It does not add video, inferred body pose, a VLM judge, VLM fine-tuning or different underlying VLM models.','',
    '## Context and fitting','',
    'The nine context variables are shot type, play pattern, under pressure, existence of a recent pass, cross/cutback/through-ball flags, pass height, and seconds since that pass. The pass must precede the shot, come from the same team in the same possession and period, and be within 15 seconds. No body part, shot outcome, placement or commercial xG enters a predictor.','',
    f'The image and context probability heads were already trained and frozen on {protocol["training_shots"]:,} World Cup shots ({protocol["training_goals"]} goals). The metadata-only logistic baseline uses those same training matches and context transform, C=0.3, without search. The metadata-only baseline fits original training rows, while the frozen fusion head used weighted augmented rows; their rare-category grouping can differ. It is a practical reference, not an identical-preprocessing ablation. The constant reference is the fixed training scoring rate. No evaluation labels fit a model, calibrator or selection rule.','',
    'Pixel and known-position controls receive the same context and fitted heads. Known positions are diagnostic controls and never replace a VLM measurement. Missing visual measurements use frozen training imputation; no shots are dropped. Saved prior Luna/context/pixel control predictions reproduce to numerical precision.','',
    '## Paired uncertainty','',*intervals,'',
    'Whole matches are resampled 4,000 times with a fixed seed. The two primary comparisons use 97.5% Bonferroni intervals. Other 95% comparisons are secondary and unadjusted for multiplicity. Intervals condition on fitted models and omit refitting uncertainty.','',
    '## Measurement audit','',*measurement,'',
    'Errors use normalized pitch units, not meters. The audit follows inference and cannot alter prompts or predictions. Per-shot errors, coverage and defender precision are saved separately. Fully hidden markers cannot reliably be recovered.','',
    '## Cost, provenance and checks','',
    f'Total successful-response usage: {int(usage.input_tokens.sum()):,} input tokens and {int(usage.output_tokens.sum()):,} output tokens. Per-role counts and elapsed times are in usage.csv; these are not exact billing or failed-attempt counts.','',
    'Specialists use two calls per shot versus one for the new monolithic control. This is a practical task-division comparison, not equal inference cost. Time order is randomized within shots; service variability and older saved comparison outputs remain limitations.','',
    f'Returned models: {", ".join(integrity["returned_models"])}. Existing protected inputs were unchanged; training/test matches are disjoint. The protocol fixes prompts, schemas, code/input/image hashes and request schedule before new inference. Responses are individually saved and validated for exact resumption. No partial cohort is scored.','',
    'Eight targeted tests cover the Luna restriction, response validation, outcome-blind request payloads, missingness, deterministic fusion, cache provenance, protected inputs and uncertainty. An integration replay with saved readings separately exercises the full 200-shot evaluator without paid requests.','',
    '## Interpretation limits','',
    'This is a supervised hybrid study rather than the original zero-shot xG benchmark. The sample contains only 19 goals and comes from 34 Leverkusen fixtures, not a full-league random sample. It has already been studied; new API readings do not make it an untouched confirmation set. Positive results require later reserved matches. An interval crossing zero does not prove equivalence.','',
    'Sources: [StatsBomb open data](https://github.com/hudl/open-data), [OpenAI image inputs](https://developers.openai.com/api/docs/guides/images-vision?api-mode=responses), [OpenAI structured outputs](https://developers.openai.com/api/docs/guides/structured-outputs?api-mode=responses).',''])
    (folder/'report.md').write_text(text,encoding='utf-8')
    print(folder/'report.md')

if __name__=='__main__':
    build(ROOT/'task-division-v1')
