"""Readable final dataset-transfer report and static scientific figure."""
import json
import run as r
NAMES={'premier_league':'Premier League 2015/16','womens_world_cup':"Women's World Cup 2019",'metrica_v2':'Metrica Sports (two-game pilot)'}
LABELS={'vlm_xg':'Raw original Luna','original_logistic':'Original WC-trained logistic (4 features)',
    'original_no_header':'Original WC logistic without header','local_logistic':'Same-dataset logistic (4 features)',
    'local_no_header':'Same-dataset logistic without header','sb_xg':'StatsBomb reference','training_prior':'Training-only constant'}


def table(scores):
    lines=['| Method | Brier ↓ | AUC ↑ | Log loss ↓ | Mean xG |','|---|---:|---:|---:|---:|']
    for key,value in scores.items():
        auc=f"{value['auc']:.3f}" if value['auc'] is not None else 'undefined'
        lines.append(f"| {LABELS[key]} | {value['brier']:.5f} | {auc} | {value['log_loss']:.5f} | {value['mean_xg']:.1%} |")
    return lines


def conclusion(summary):
    supported=[name for name,s in summary.items() if name!='metrica_v2' and s['primary']['interval'][1]<0]
    worse=[name for name,s in summary.items() if name!='metrica_v2' and s['primary']['interval'][0]>0]
    if supported:
        return 'Raw Luna has lower Brier error than the original frozen logistic reference on '+', '.join(NAMES[x] for x in supported)+', under the prespecified intervals. This is cohort-specific evidence, not a general proof of visual-model xG accuracy; inspect the local and uninformative references too.'
    if len(worse)==2:
        return 'Changing datasets did not solve raw probability accuracy: original Luna has higher Brier error than the original frozen logistic reference in both larger cohorts under the prespecified intervals.'
    return 'These new datasets do not establish a reliable raw-Luna improvement over the original logistic reference. Inconclusive comparisons do not prove equivalence; the two-game Metrica pilot cannot establish population-level superiority.'


def figure(cohorts):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(16,5.8))
    short={'vlm_xg':'Raw Luna','original_logistic':'Original logistic','local_logistic':'Local logistic',
           'sb_xg':'StatsBomb','training_prior':'Training constant'}
    for axis,(name,data) in zip(axes,cohorts.items()):
        keys=[k for k in short if k in data['scores']]
        values=[data['scores'][k]['brier'] for k in keys]
        bars=axis.barh([short[k] for k in keys],values,color=['#176b83' if k=='vlm_xg' else '#9ba8ad' for k in keys])
        axis.invert_yaxis(); axis.bar_label(bars,fmt='%.4f',padding=4,fontsize=9)
        axis.set_xlim(0,max(values)*1.27)
        axis.set_xlabel('Brier probability error (lower is better)')
        axis.set_title(NAMES[name]+f"\n{data['shots']} shots; {data['goals']} goals; {data['matches']} matches",fontsize=11,pad=16)
        axis.spines[['top','right']].set_visible(False)
    fig.suptitle('Same direct VLM method on new datasets — raw answers, no correction',fontsize=15,y=.99)
    fig.tight_layout(rect=(0,0,1,.94)); fig.savefig(r.OUT/'comparison.png',dpi=160); plt.close(fig)


def main():
    if not r.base.read(r.OUT/'complete.json'): raise ValueError('Study incomplete')
    combined=r.base.read(r.OUT/'summary.json'); cohorts=combined['cohorts']; audit=combined['audit']
    if not audit['passed']: raise ValueError('Audit did not pass')
    verdict=conclusion(cohorts); plan=r.base.read(r.OUT/'data_plan.json')
    lines=['# Direct visual xG on different datasets','', '29 October 2026 — America/Los_Angeles','',
        '**'+verdict+'**','',
        f"Completed **{audit['calls']} new generation calls**, estimated **${audit['estimated_usd']:.6f}** (conservative **${audit['conservative_usd']:.6f}**), below the **$1** incremental cap. All probabilities are raw original GPT-6 Luna answers. No paid runner remains active.",'',
        '## What changed','',
        'Only the evaluation data changed. The VLM kept the original zero-shot prompt, original half-pitch renderer, low effort, high image detail and 4,096-token maximum. The later missing-information prompt was not adopted. No numerical coordinates, goal rates, outcomes, body-part flags, cohort identity or baseline predictions entered VLM requests. No recalibration, probability head, baseline blend, repeat average or fine-tuning.','',
        'Metrica is a separate provider with synchronized tracking/event data. Premier League and Women’s World Cup are new populations within StatsBomb, not independent providers. They are reported separately, with no pooling to hide a failed cohort.','',
        'Two prespecified primary comparisons test original Luna minus the original frozen World Cup four-feature logistic Brier in the larger cohorts, using 97.5% whole-match bootstrap intervals (10,000 resamples) for two comparisons. Negative favors Luna. Metrica is descriptive only because it has two games. This continuation belongs to a broader exploratory program; the intervals do not adjust for every earlier study family.','',
        'Lower Brier/log loss means better probability estimates. Higher AUC means better shot ranking. A lower raw Brier on a different dataset is not automatically a model improvement, since goal rates and chance distributions differ.','']
    for name,data in cohorts.items():
        lines+=['## '+NAMES[name],'',f"{data['shots']} eligible shots, {data['goals']} goals ({data['goal_rate']:.1%}), {data['matches']} evaluated matches.",'']
        lines+=table(data['scores'])
        lines+=['',f"Mean Luna xG minus observed goal rate: **{data['mean_overprediction']:+.1%}**. Mean alone does not establish individual prediction accuracy."]
        if name!='metrica_v2':
            item=data['primary']; low,high=item['interval']
            lines+=['',f"**Primary:** Luna minus original logistic Brier **{item['difference']:+.5f}**, 97.5% whole-match interval **[{low:+.5f}, {high:+.5f}]**.",'',
                ('This interval supports a lower Brier for Luna in this cohort.' if high<0 else
                 'This interval supports a higher Brier for Luna in this cohort.' if low>0 else
                 'The interval crosses zero: superiority and equivalence are not established.'),'',
                'Secondary comparisons (95% intervals, not substitutes for the primary comparison):','']
            for field in ('secondary_local','secondary_no_header'):
                item=data[field]; low,high=item['interval']
                lines.append(f"- Luna minus {LABELS[item['second']]}: {item['difference']:+.5f}, [{low:+.5f}, {high:+.5f}].")
            baseline=r.base.read(r.OUT/name/'baseline.json')
            lines+=['',f"The extra local statistical models were fitted on {baseline['training_shots']} shots ({baseline['training_goals']} goals) from {len(baseline['training_matches'])} disjoint matches, with the same fixed logistic fitting recipe and no outcome-driven tuning. The training-only constant uses only those training outcomes. These models are comparators; nothing supplies or corrects a VLM answer."]
        else:
            lines+=['',f"Descriptive Luna-minus-original-logistic Brier: **{data['descriptive_brier_difference']:+.5f}**. Two anonymized games cannot support a trustworthy population-level interval or superiority/equivalence claim. The constant uses the original World Cup training goal rate (287/3068).",'',
                'The two source files contain 48 shot events. One annotated penalty and one shot without synchronized shooter tracking were excluded. All 46 remaining shots were retained. Goalkeeper roles were inferred from opening tracking before any shot, then held consistent across halves. An offline test caught a single-frame half-time mislabel before inference; corrected images are in metrica_v2. Initial failed-preflight images were retained and never submitted.']
        lines+=['']
    lines+=['## Source integrity and fair comparisons','',
        'For StatsBomb, whole match groups were split before event downloads. Premier League uses 40 test and 40 training matches; Women’s World Cup uses 26 test and 26 training matches. Exactly 500 eligible test shots per cohort were randomly sampled without goal balancing or prior-error selection. All 356 previously recorded match groups were excluded. No partial-cohort scores were inspected and no result-driven samples were added. All 132 newly downloaded match groups are now used and must be excluded from future untouched-data claims.','',
        'The original logistic model is unchanged, fitted on 3,068 World Cup 2018/2022 shots. Four-feature models know distance, angle, header and opponent count in the shooting cone. The header flag supplies information absent from the image. Three-feature no-header references are therefore shown separately and never silently replace the original comparator. All numerical references also receive recorded coordinates. StatsBomb xG is another estimate, not ground-truth shot probability.','',
        'The unchanged generic VLM is not told which images come from women’s football; this is an explicit transfer limitation. Its source population may differ from the implicit professional-football prior. The extra local statistical references are fitted to each population, and therefore have population-specific training information.','',
        'Metrica uses the event Start Frame only. End coordinates, end frames, outcome tags, names and clocks are not rendered. Tracking positions on the provider’s 105-by-68-metre field are linearly normalized to the existing 120-by-80 diagram conventions and rotated for attacking direction. This shared coordinate representation is approximate rather than a new metric-geometry benchmark. Goalkeeper roles are inferred, not certified by a provider role label. Anonymized matches prevent ruling out cross-provider identity overlap. Pretraining exposure is unknown for every source.','',
        'Binary goal outcomes are noisy. Match-bootstrap intervals condition on these fixed predictions and fitted references; they do not capture all model-call variability, provider measurement error or training uncertainty. No equivalence margin was prespecified. A favorable single dataset would still require independent replication.','',
        'The requested model is an API alias; returned model identifiers and UTC request times are preserved. Dataset coverage and source bytes are frozen in this study, while remote repositories may change later.','',
        '## Other datasets assessed','',
        '**xGHub:** the author release has shot clips, pose/orientation context and goal labels, making it relevant to information missing from position-only diagrams. The actual Hugging Face release requires institutional access and an NDA for restricted clips. The public GitHub repository contains the README and a sample animation, not the full footage. No access request, NDA acceptance, restricted download or outside message was made. It was not evaluated here. [Official release/access requirements](https://huggingface.co/datasets/mguti97/xGHub), [author repository](https://github.com/mguti97/xGHub).','',
        'SkillCorner and Wyscout were also reviewed as alternatives. They were not silently treated as the same input/target format or added after seeing scores.','',
        '## Cost and verification','',
        f"New OpenAI usage estimate **${audit['estimated_usd']:.6f}**; conservative **${audit['conservative_usd']:.6f}**. Gemini and Claude **$0 new spending**. The prior round cost $0.3985504 estimated; older tracked records remain separately preserved. Complete account bills and live balances are unavailable, so these are usage estimates, not an account-wide remaining balance.",'',
        f"Input tokens: {audit['input_tokens']:,}; output including reasoning: {audit['output_tokens']:,}; reasoning tokens: {audit['reasoning_tokens']:,}. Six concurrent workers maximum. Every input preflight plus a safety allowance and every full next output allowance was reserved before dispatch, including concurrent requests. No hidden retries, difficult-shot replacement, discarded cohort or probability conversion.",'',
        f"All **{len(audit['checks'])}** final audit checks passed: immutable source/code/cohort/reference hashes, train/test separation, prior-match exclusion, exact request reconstruction, unchanged raw predictions, identity/settings/usage, reproduction of local reference predictions, budget and private exact-credential exposure. Four offline transfer checks passed after the documented Metrica preflight correction.",'',
        'Artifacts: protocol.json and data_plan.json; data_amendment.json; per-cohort selection/baseline/provenance, predictions.csv and summary.json; raw response files; token counts and attempts ledger; audit.json; source files and original-renderer images. Initial preflight revisions are preserved separately.','',
        '![Dataset-specific raw probability errors](comparison.png)','',
        'Sources: [Metrica Sports sample data](https://github.com/metrica-sports/sample-data), [Hudl StatsBomb Open Data](https://github.com/hudl/open-data), [official Luna settings/rates](https://developers.openai.com/api/docs/models/gpt-6-luna).']
    (r.OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    figure(cohorts)
    r.base.save(r.OUT/'final_status.json',{'status':'complete','calls':audit['calls'],'new_estimated_usd':audit['estimated_usd'],
        'conservative_usd':audit['conservative_usd'],'audit_passed':True,'active_runner':False,'conclusion':verdict})
    print(json.dumps(r.base.read(r.OUT/'final_status.json')),flush=True)


if __name__=='__main__': main()
